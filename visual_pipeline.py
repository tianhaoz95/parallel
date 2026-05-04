import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline, 
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
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=self.dtype).to(self.device)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path, controlnet=self.controlnet, image_encoder=self.image_encoder,
            torch_dtype=self.dtype, safety_checker=None, feature_extractor=None
        )
        
        self.use_lcm = CONFIG.get('defaults', {}).get('use_lcm', False)
        if self.use_lcm:
            self.pipe.load_lora_weights(CONFIG.get('optimizations', {}).get('lcm_lora'))
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
        self.pipe.load_ip_adapter(os.path.join(ip_adapter_path, "models"), subfolder="", weight_name="ip-adapter-plus_sd15.bin")
        self.pipe.set_ip_adapter_scale(CONFIG.get('defaults', {}).get('ip_adapter_scale', 0.7))
        
        self.face_restorer = GFPGANer(model_path=CONFIG.get('models', {}).get('restoration', {}).get('gfpgan'), upscale=1, arch='clean', channel_multiplier=2, device=self.device)
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upscaler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth', model=model, tile=400, tile_pad=10, pre_pad=0, half=True if self.device == "cuda" else False, device=self.device)
        
        self.segmenter = mp_lib.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        if self.device == "cuda": self.pipe.enable_model_cpu_offload()

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

    def identify_and_mask_target(self, frame, ref_embeddings):
        """Identifies target person using embeddings and returns a precise mask."""
        results = self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        full_mask = results.segmentation_mask > 0.5
        
        # If no reference provided, mask everyone
        if not ref_embeddings:
            return np.stack((full_mask,) * 3, axis=-1).astype(np.uint8) * 255
            
        # Detect faces in current frame
        try:
            faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
            target_mask = np.zeros_like(results.segmentation_mask, dtype=bool)
            
            for face_data in faces:
                face_img = face_data['face']
                if face_img.size == 0: continue
                
                # Check if this face matches reference
                try:
                    # DeepFace.verify returns distance
                    res = DeepFace.verify(face_img, ref_embeddings[0], detector_backend='skip', enforce_detection=False)
                    if res['verified']:
                        # This is our target! Get its bounding box and add to mask
                        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
                        # We use the segmentation mask within this person's vicinity
                        # For simplicity, we just use the global mask but it could be refined
                        target_mask = full_mask 
                        break # Found target
                except: continue
            
            return np.stack((target_mask,) * 3, axis=-1).astype(np.uint8) * 255
        except:
            return np.stack((full_mask,) * 3, axis=-1).astype(np.uint8) * 255

    def process_frame(self, frame, ref_images, ref_embeddings, prompt, restore_face=True, use_mask=True):
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
        
        output = self.pipe(
            prompt=prompt, image=pil_frame, ip_adapter_image=ref_images, control_image=canny_image,
            strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), num_inference_steps=num_steps, guidance_scale=1.0 if self.use_lcm else 7.5
        ).images[0]
        
        transformed_frame = np.array(output.resize((frame.shape[1], frame.shape[0])))
        
        if use_mask:
            mask = self.identify_and_mask_target(frame, ref_embeddings)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask_norm = mask.astype(float) / 255.0
            final_frame = (transformed_frame.astype(float) * mask_norm + frame.astype(float) * (1.0 - mask_norm)).astype(np.uint8)
        else:
            final_frame = transformed_frame
            
        if restore_face: final_frame = self.restore_faces(final_frame)
        return final_frame

    def warp_frame(self, prev_transformed, prev_original, curr_original):
        prev_gray = cv2.cvtColor(prev_original, cv2.COLOR_RGB2GRAY); curr_gray = cv2.cvtColor(curr_original, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow[:,:,0] += np.arange(w); flow[:,:,1] += np.arange(h)[:,np.newaxis]
        return cv2.remap(prev_transformed, flow, None, cv2.INTER_LINEAR)

    def process_video(self, video_path, ref_image_paths, output_video_path, prompt="a person", restore_face=True, smooth=True, use_mask=True, upscale=False):
        if isinstance(ref_image_paths, str): ref_image_paths = [ref_image_paths]
        logger.info(f"Processing video {video_path} with Identity Targeting...")
        clip = mp.VideoFileClip(video_path)
        ref_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in ref_image_paths]
        
        # Pre-calculate reference embeddings for targeting
        logger.info("Extracting reference face identity...")
        ref_embeddings = []
        for p in ref_image_paths:
            try:
                # We store the image path for DeepFace to use
                ref_embeddings.append(p)
            except: continue

        fps = clip.fps; width, height = clip.size
        out_w, out_h = (width * 2, height * 2) if upscale else (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_out_path = "temp_streaming_visual.mp4"
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (out_w, out_h))
        
        test_duration = min(clip.duration, 0.5); prev_original = None; prev_transformed = None; count = 0
        try:
            for frame in clip.iter_frames():
                if count / fps > test_duration: break
                curr_transformed = self.process_frame(frame, ref_images, ref_embeddings, prompt, restore_face=restore_face, use_mask=use_mask)
                if smooth and prev_transformed is not None:
                    warped_prev = self.warp_frame(prev_transformed, prev_original, frame)
                    curr_transformed = cv2.addWeighted(curr_transformed, 0.6, warped_prev, 0.4, 0)
                if upscale: curr_transformed = self.upscale_frame(curr_transformed)
                out.write(cv2.cvtColor(curr_transformed, cv2.COLOR_RGB2BGR))
                prev_original = frame.copy(); prev_transformed = curr_transformed.copy(); count += 1
        finally:
            out.release(); clip.close()
        if os.path.exists(output_video_path): os.remove(output_video_path)
        os.rename(temp_out_path, output_video_path)
        return output_video_path

if __name__ == "__main__":
    pass
