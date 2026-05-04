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
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, torch_dtype=self.dtype).to(self.device)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path, controlnet=self.controlnet, image_encoder=self.image_encoder,
            torch_dtype=self.dtype, safety_checker=None, feature_extractor=None
        )
        
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            torch_dtype=self.dtype, safety_checker=None
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
        
        if self.device == "cuda": 
            self.pipe.enable_model_cpu_offload()
            self.inpaint_pipe.enable_model_cpu_offload()

    def match_skin_tone(self, target_img, source_img):
        """Matches the color palette of target_img to source_img."""
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
        
        t_mean, t_std = cv2.meanStdDev(target_img)
        s_mean, s_std = cv2.meanStdDev(source_img)
        
        target_img = (target_img - t_mean.flatten()) * (s_std.flatten() / (t_std.flatten() + 1e-5)) + s_mean.flatten()
        target_img = np.clip(target_img, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(target_img, cv2.COLOR_LAB2RGB)

    def identify_and_mask_target(self, frame, ref_embeddings, tracker_pos=None):
        """Identifies target person and returns a mask. Uses tracker_pos if available."""
        results = self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        full_mask = results.segmentation_mask > 0.5
        
        if not ref_embeddings:
            return np.stack((full_mask,) * 3, axis=-1).astype(np.uint8) * 255, None
            
        # If we have a tracked position, we skip full recognition for speed
        if tracker_pos is not None:
            # We assume the target is still the person at tracker_pos
            # Simple implementation: just return full mask for now but could be refined
            return np.stack((full_mask,) * 3, axis=-1).astype(np.uint8) * 255, tracker_pos

        try:
            faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
            target_mask = np.zeros_like(results.segmentation_mask, dtype=bool)
            new_pos = None
            
            for face_data in faces:
                face_img = face_data['face']
                if face_img.size == 0: continue
                try:
                    res = DeepFace.verify(face_img, ref_embeddings[0], detector_backend='skip', enforce_detection=False)
                    if res['verified']:
                        target_mask = full_mask 
                        new_pos = face_data['facial_area']
                        break 
                except: continue
            
            return np.stack((target_mask,) * 3, axis=-1).astype(np.uint8) * 255, new_pos
        except:
            return np.stack((full_mask,) * 3, axis=-1).astype(np.uint8) * 255, None

    def process_frame(self, frame, ref_images, ref_embeddings, prompt, tracker_pos=None, restore_face=True, use_mask=True):
        # 1. Identity Masking with Tracking
        mask_raw, next_pos = self.identify_and_mask_target(frame, ref_embeddings, tracker_pos)
        
        # 2. Character generation
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
        
        output = self.pipe(
            prompt=prompt, image=pil_frame, ip_adapter_image=ref_images, control_image=canny_image,
            strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), num_inference_steps=num_steps, guidance_scale=1.0 if self.use_lcm else 7.5
        ).images[0]
        
        transformed_person = np.array(output.resize((frame.shape[1], frame.shape[0])))
        
        # 3. Skin Tone Matching
        if use_mask:
            transformed_person = self.match_skin_tone(transformed_person, frame)
            
            mask_norm = cv2.GaussianBlur(mask_raw, (15, 15), 0).astype(float) / 255.0
            if mask_norm.ndim == 2: mask_norm = np.stack([mask_norm]*3, axis=-1)
            final_frame = (transformed_person.astype(float) * mask_norm + frame.astype(float) * (1.0 - mask_norm)).astype(np.uint8)
        else:
            final_frame = transformed_person
            
        if restore_face: final_frame = self.restore_faces(final_frame)
        return final_frame, next_pos

    def process_video(self, video_path, ref_image_paths, output_video_path, prompt="a person", restore_face=True, smooth=True, use_mask=True, upscale=False):
        if isinstance(ref_image_paths, str): ref_image_paths = [ref_image_paths]
        clip = mp.VideoFileClip(video_path)
        ref_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in ref_image_paths]
        ref_embeddings = ref_image_paths

        fps = clip.fps; width, height = clip.size
        out_w, out_h = (width * 2, height * 2) if upscale else (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_out_path = "temp_streaming_visual.mp4"
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (out_w, out_h))
        
        test_duration = min(clip.duration, 1.0); tracker_pos = None; count = 0
        try:
            for frame in clip.iter_frames():
                if count / fps > test_duration: break
                
                # Every 30 frames or if lost, re-verify identity
                if count % 30 == 0: tracker_pos = None 
                
                curr_transformed, tracker_pos = self.process_frame(frame, ref_images, ref_embeddings, prompt, tracker_pos=tracker_pos, restore_face=restore_face, use_mask=use_mask)
                
                if upscale: curr_transformed = self.upscale_frame(curr_transformed)
                out.write(cv2.cvtColor(curr_transformed, cv2.COLOR_RGB2BGR))
                count += 1
        finally:
            out.release(); clip.close()
        os.rename(temp_out_path, output_video_path)
        return output_video_path

if __name__ == "__main__":
    pass
