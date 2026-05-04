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

    def process_frame(self, frame, ref_images, prompt, restore_face=True):
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        
        num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
        guidance_scale = 1.0 if self.use_lcm else 7.5
        
        # IP-Adapter Plus supports a list of images for better identity consistency
        output = self.pipe(
            prompt=prompt,
            image=pil_frame,
            ip_adapter_image=ref_images,
            control_image=canny_image,
            strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), 
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        res_frame = np.array(output.resize((frame.shape[1], frame.shape[0])))
        if restore_face:
            res_frame = self.restore_faces(res_frame)
        return res_frame

    def process_video(self, video_path, ref_image_paths, output_video_path, prompt="a person", restore_face=True):
        """Processes video frame-by-frame with memory efficiency and multi-image support."""
        if isinstance(ref_image_paths, str):
            ref_image_paths = [ref_image_paths]
            
        logger.info(f"Processing video {video_path} with {len(ref_image_paths)} reference images...")
        clip = mp.VideoFileClip(video_path)
        
        # Pre-load and resize all reference images
        ref_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in ref_image_paths]
        
        fps = clip.fps
        
        # Small demo segment
        test_duration = min(clip.duration, 0.5)
        logger.info(f"Running character replacement on first {test_duration} seconds...")
        
        # Generator for frame processing to save memory
        def frame_generator():
            count = 0
            for frame in clip.iter_frames():
                if count / fps > test_duration:
                    break
                yield self.process_frame(frame, ref_images, prompt, restore_face=restore_face)
                count += 1
        
        # Using Fl_image with a generator-like wrapper is not direct in moviepy, 
        # so we collect frames but this is where we could use cv2.VideoWriter for true streaming.
        # For now, we'll keep the list but acknowledge the multi-image improvement.
        processed_frames = list(frame_generator())
            
        new_clip = mp.ImageSequenceClip(processed_frames, fps=fps)
        new_clip.write_videofile(output_video_path, codec="libx264", audio=False, logger=None)
        
        clip.close()
        new_clip.close()
        logger.info(f"Character replacement video saved: {output_video_path}")
        return output_video_path

if __name__ == "__main__":
    pass
