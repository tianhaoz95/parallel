import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection
import moviepy.editor as mp
from gfpgan import GFPGANer
from logger_utils import logger

class VisualPipeline:
    def __init__(self, 
                 sd_model_path="models/stable-diffusion-v1-5-pretrained", 
                 controlnet_path="models/sd-controlnet-canny",
                 image_encoder_path="models/image_encoder_H14",
                 ip_adapter_path="models/IP-Adapter_plus"):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"Initializing Visual Pipeline on {self.device}")
        
        # 1. Load Image Encoder
        logger.info(f"Loading Image Encoder from {image_encoder_path}...")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, torch_dtype=self.dtype
        ).to(self.device)
        
        # 2. Load ControlNet
        logger.info(f"Loading ControlNet from {controlnet_path}...")
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        
        # 3. Load SD Pipeline
        logger.info(f"Loading Stable Diffusion from {sd_model_path}...")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model_path, 
            controlnet=self.controlnet, 
            image_encoder=self.image_encoder,
            torch_dtype=self.dtype,
            safety_checker=None, 
            feature_extractor=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # 4. Load IP-Adapter Plus
        logger.info("Loading IP-Adapter Plus weights...")
        self.pipe.load_ip_adapter(
            os.path.join(ip_adapter_path, "models"), 
            subfolder="", 
            weight_name="ip-adapter-plus_sd15.bin"
        )
        self.pipe.set_ip_adapter_scale(0.7)
        
        # 5. Load GFPGAN
        logger.info("Loading GFPGAN for face restoration...")
        self.face_restorer = GFPGANer(
            model_path='models/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            device=self.device
        )
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def get_canny_image(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def restore_faces(self, frame):
        """Public method to restore faces in a single frame."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, _, restored_img = self.face_restorer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    def process_frame(self, frame, ref_image, prompt, restore_face=True):
        pil_frame = Image.fromarray(frame).resize((512, 512))
        canny_image = self.get_canny_image(pil_frame)
        
        output = self.pipe(
            prompt=prompt,
            image=pil_frame,
            ip_adapter_image=ref_image,
            control_image=canny_image,
            strength=0.6, 
            num_inference_steps=15
        ).images[0]
        
        res_frame = np.array(output.resize((frame.shape[1], frame.shape[0])))
        if restore_face:
            res_frame = self.restore_faces(res_frame)
        return res_frame

    def process_video(self, video_path, ref_image_path, output_video_path, prompt="a person", restore_face=True):
        logger.info(f"Processing video {video_path} for character replacement...")
        clip = mp.VideoFileClip(video_path)
        ref_image = Image.open(ref_image_path).convert("RGB").resize((224, 224))
        
        # Demo duration
        test_duration = min(clip.duration, 0.5)
        logger.info(f"Running character replacement on first {test_duration} seconds...")
        
        frames = []
        count = 0
        fps = clip.fps
        for frame in clip.iter_frames():
            if count / fps > test_duration:
                break
            new_frame = self.process_frame(frame, ref_image, prompt, restore_face=restore_face)
            frames.append(new_frame)
            count += 1
            
        new_clip = mp.ImageSequenceClip(frames, fps=fps)
        new_clip.write_videofile(output_video_path, codec="libx264", audio=False)
        clip.close()
        logger.info(f"Character replacement video saved: {output_video_path}")
        return output_video_path

if __name__ == "__main__":
    pass
