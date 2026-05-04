import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline, 
    StableDiffusionInpaintPipeline,
    ControlNetModel, 
    UniPCMultistepScheduler,
    LCMScheduler
)
from transformers import CLIPVisionModelWithProjection
import moviepy.editor as mp
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import mediapipe as mp_lib
from deepface import DeepFace
from logger_utils import logger, CONFIG

class VisualPipeline:
    def __init__(self, sd_model_path=None, controlnet_path=None, image_encoder_path=None, ip_adapter_path=None):
        cfg = CONFIG.get('models', {}).get('visual', {})
        sd_model_path = sd_model_path or cfg.get('stable_diffusion')
        controlnet_path = controlnet_path or cfg.get('controlnet_canny')
        image_encoder_path = image_encoder_path or cfg.get('image_encoder')
        ip_adapter_path = ip_adapter_path or cfg.get('ip_adapter')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"Initializing Visual Pipeline on {self.device}")
        
        # 1. Load Image Encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=self.dtype).to(self.device)
        
        # 2. Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        
        # 3. Load SD Pipeline (Main Replacement)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path, controlnet=self.controlnet, image_encoder=self.image_encoder,
            torch_dtype=self.dtype, safety_checker=None, feature_extractor=None
        )
        
        # 4. Load Inpainting Pipeline (Background Cleaning)
        # We reuse the same base model if possible to save VRAM, but for now we'll assume a dedicated path
        # or load the inpaint weights.
        logger.info("Loading Inpainting Pipeline...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            torch_dtype=self.dtype, safety_checker=None
        )
        
        # 5. Optimization: LCM
        self.use_lcm = CONFIG.get('defaults', {}).get('use_lcm', False)
        if self.use_lcm:
            self.pipe.load_lora_weights(CONFIG.get('optimizations', {}).get('lcm_lora'))
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
        # 6. Load IP-Adapter
        self.pipe.load_ip_adapter(os.path.join(ip_adapter_path, "models"), subfolder="", weight_name="ip-adapter-plus_sd15.bin")
        self.pipe.set_ip_adapter_scale(CONFIG.get('defaults', {}).get('ip_adapter_scale', 0.7))
        
        # 7. Face & HD
        self.face_restorer = GFPGANer(model_path=CONFIG.get('models', {}).get('restoration', {}).get('gfpgan'), upscale=1, arch='clean', channel_multiplier=2, device=self.device)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upscaler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth', model=model, tile=400, tile_pad=10, pre_pad=0, half=True if self.device == "cuda" else False, device=self.device)
        
        self.segmenter = mp_lib.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        if self.device == "cuda": 
            self.pipe.enable_model_cpu_offload()
            self.inpaint_pipe.enable_model_cpu_offload()

    def get_canny_image(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def restore_faces(self, frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, _, restored_img = self.face_restorer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    def upscale_frame(self, frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output, _ = self.upscaler.enhance(frame_bgr, outscale=2)
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    def get_person_mask(self, frame, dilate=0):
        """Returns a binary mask of the person in the frame."""
        results = self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        if dilate > 0:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def inpaint_background(self, frame, mask):
        """Cleans the background by inpainting the person area."""
        pil_frame = Image.fromarray(frame).resize((512, 512))
        pil_mask = Image.fromarray(mask).resize((512, 512))
        
        # Inpaint to fill the hole with background-like textures
        clean_bg = self.inpaint_pipe(
            prompt="background, seamless, high quality",
            image=pil_frame,
            mask_image=pil_mask,
            num_inference_steps=20
        ).images[0]
        
        return np.array(clean_bg.resize((frame.shape[1], frame.shape[0])))

    def process_frame(self, frame, ref_images, ref_embeddings, prompt, restore_face=True, use_mask=True, erase_original=True):
        # 1. Masking: Get the original person area
        mask_raw = self.get_person_mask(frame)
        
        # 2. Erase: If requested, clean the background first
        bg_frame = frame
        if erase_original and use_mask:
            # Dilate mask to ensure we cover the original person fully
            dilate_mask = self.get_person_mask(frame, dilate=20)
            bg_frame = self.inpaint_background(frame, dilate_mask)
        
        # 3. Generate replacement character
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
        
        output = self.pipe(
            prompt=prompt, image=pil_frame, ip_adapter_image=ref_images, control_image=canny_image,
            strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), num_inference_steps=num_steps, guidance_scale=1.0 if self.use_lcm else 7.5
        ).images[0]
        
        transformed_person = np.array(output.resize((frame.shape[1], frame.shape[0])))
        
        # 4. Composite: Put new person onto the (potentially cleaned) background
        if use_mask:
            # We use the raw mask for the AI character to keep it sharp
            mask_norm = cv2.GaussianBlur(mask_raw, (15, 15), 0).astype(float) / 255.0
            if mask_norm.ndim == 2: mask_norm = np.stack([mask_norm]*3, axis=-1)
            
            final_frame = (transformed_person.astype(float) * mask_norm + bg_frame.astype(float) * (1.0 - mask_norm)).astype(np.uint8)
        else:
            final_frame = transformed_person
            
        if restore_face: final_frame = self.restore_faces(final_frame)
        return final_frame

    def is_scene_cut(self, curr_frame, prev_frame, threshold=0.8):
        """Detects if there is a scene cut between two frames using histogram correlation."""
        if prev_frame is None: return False
        
        h1 = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        
        correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return correlation < threshold

    def process_video(self, video_path, ref_image_paths, output_video_path, prompt="a person", restore_face=True, smooth=True, use_mask=True, erase_original=True, upscale=False):
        """Processes video with scene-aware temporal smoothing."""
        if isinstance(ref_image_paths, str): ref_image_paths = [ref_image_paths]
        logger.info(f"Processing video {video_path} with Scene-Aware Smoothing...")
        clip = mp.VideoFileClip(video_path)
        ref_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in ref_image_paths]
        ref_embeddings = ref_image_paths

        fps = clip.fps; width, height = clip.size
        out_w, out_h = (width * 2, height * 2) if upscale else (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_out_path = "temp_streaming_visual.mp4"
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (out_w, out_h))
        
        test_duration = min(clip.duration, 0.5)
        prev_original = None; prev_transformed = None; count = 0
        
        try:
            for frame in clip.iter_frames():
                if count / fps > test_duration: break
                
                # Check for scene cut to reset smoothing
                if smooth and prev_original is not None:
                    if self.is_scene_cut(frame, prev_original):
                        logger.info(f"Scene cut detected at frame {count}. Resetting smoothing buffer.")
                        prev_transformed = None
                
                curr_transformed = self.process_frame(frame, ref_images, ref_embeddings, prompt, restore_face=restore_face, use_mask=use_mask, erase_original=erase_original)
                
                if smooth and prev_transformed is not None:
                    warped_prev = self.warp_frame(prev_transformed, prev_original, frame)
                    curr_transformed = cv2.addWeighted(curr_transformed, 0.6, warped_prev, 0.4, 0)
                
                if upscale: curr_transformed = self.upscale_frame(curr_transformed)
                out.write(cv2.cvtColor(curr_transformed, cv2.COLOR_RGB2BGR))
                
                prev_original = frame.copy()
                prev_transformed = curr_transformed.copy()
                count += 1
        finally:
            out.release(); clip.close()
        
        if os.path.exists(output_video_path): os.remove(output_video_path)
        os.rename(temp_out_path, output_video_path)
        return output_video_path

if __name__ == "__main__":
    pass
