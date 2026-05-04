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
import mediapipe as mp_lib
from logger_utils import logger, CONFIG

class VisualPipeline:
    def __init__(self, 
                 sd_model_path=None, 
                 controlnet_path=None,
                 image_encoder_path=None,
                 ip_adapter_path=None):
        
        cfg = CONFIG.get('models', {}).get('visual', {})
        sd_model_path = sd_model_path or cfg.get('stable_diffusion')
        controlnet_path = controlnet_path or cfg.get('controlnet_canny')
        image_encoder_path = image_encoder_path or cfg.get('image_encoder')
        ip_adapter_path = ip_adapter_path or cfg.get('ip_adapter')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"Initializing Visual Pipeline on {self.device}")
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, torch_dtype=self.dtype
        ).to(self.device)
        
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path, 
            controlnet=self.controlnet, 
            image_encoder=self.image_encoder,
            torch_dtype=self.dtype,
            safety_checker=None, 
            feature_extractor=None
        )
        
        self.use_lcm = CONFIG.get('defaults', {}).get('use_lcm', False)
        if self.use_lcm:
            logger.info("Enabling LCM speedup...")
            lcm_lora_id = CONFIG.get('optimizations', {}).get('lcm_lora', "models/lcm-lora-sdv1-5")
            self.pipe.load_lora_weights(lcm_lora_id)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        logger.info("Loading IP-Adapter Plus weights...")
        self.pipe.load_ip_adapter(
            os.path.join(ip_adapter_path, "models"), 
            subfolder="", 
            weight_name="ip-adapter-plus_sd15.bin"
        )
        self.pipe.set_ip_adapter_scale(CONFIG.get('defaults', {}).get('ip_adapter_scale', 0.7))
        
        logger.info("Loading GFPGAN...")
        self.face_restorer = GFPGANer(
            model_path=CONFIG.get('models', {}).get('restoration', {}).get('gfpgan'),
            upscale=1, arch='clean', channel_multiplier=2, device=self.device
        )
        
        # Initialize MediaPipe Segmentation
        self.mp_selfie = mp_lib.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1) # 1 for landscape
        
        if self.device == "cuda":
            if CONFIG.get('optimizations', {}).get('low_vram', True):
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)

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

    def get_person_mask(self, frame):
        """Returns a binary mask of the person in the frame."""
        results = self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        return condition.astype(np.uint8) * 255

    def process_frame(self, frame, ref_images, prompt, restore_face=True):
        # 1. Generate replacement frame
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        
        num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
        guidance_scale = 1.0 if self.use_lcm else 7.5
        
        output = self.pipe(
            prompt=prompt,
            image=pil_frame,
            ip_adapter_image=ref_images,
            control_image=canny_image,
            strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), 
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        transformed_frame = np.array(output.resize((frame.shape[1], frame.shape[0])))
        
        # 2. Masking pass: Blend transformed character back into original high-res background
        mask = self.get_person_mask(frame)
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_norm = mask.astype(float) / 255.0
        
        # Composite: (transformed * mask) + (original * (1-mask))
        final_frame = (transformed_frame.astype(float) * mask_norm + 
                       frame.astype(float) * (1.0 - mask_norm))
        final_frame = final_frame.astype(np.uint8)
        
        if restore_face:
            final_frame = self.restore_faces(final_frame)
            
        return final_frame

    def warp_frame(self, prev_transformed, prev_original, curr_original):
        prev_gray = cv2.cvtColor(prev_original, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_original, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(prev_transformed, flow, None, cv2.INTER_LINEAR)
        return warped

    def process_video(self, video_path, ref_image_paths, output_video_path, prompt="a person", restore_face=True, smooth=True):
        if isinstance(ref_image_paths, str):
            ref_image_paths = [ref_image_paths]
            
        logger.info(f"Processing video {video_path} with Masking & Temporal Smoothing...")
        clip = mp.VideoFileClip(video_path)
        ref_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in ref_image_paths]
        
        fps = clip.fps
        test_duration = min(clip.duration, 0.5)
        
        processed_frames = []
        prev_original = None
        prev_transformed = None
        
        count = 0
        for frame in clip.iter_frames():
            if count / fps > test_duration:
                break
            
            curr_transformed = self.process_frame(frame, ref_images, prompt, restore_face=restore_face)
            
            if smooth and prev_transformed is not None:
                warped_prev = self.warp_frame(prev_transformed, prev_original, frame)
                curr_transformed = cv2.addWeighted(curr_transformed, 0.6, warped_prev, 0.4, 0)
            
            processed_frames.append(curr_transformed)
            prev_original = frame.copy()
            prev_transformed = curr_transformed.copy()
            count += 1
            
        new_clip = mp.ImageSequenceClip(processed_frames, fps=fps)
        new_clip.write_videofile(output_video_path, codec="libx264", audio=False, logger=None)
        
        clip.close()
        logger.info(f"Video saved: {output_video_path}")
        return output_video_path

if __name__ == "__main__":
    pass
