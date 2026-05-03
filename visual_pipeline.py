import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection
import moviepy as mp

class VisualPipeline:
    def __init__(self, 
                 sd_model_path="models/stable-diffusion-v1-5-pretrained", 
                 controlnet_path="models/sd-controlnet-canny",
                 image_encoder_path="models/image_encoder_H14",
                 ip_adapter_path="models/IP-Adapter_plus"):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading Visual Pipeline on {self.device}...")
        
        # 1. Load Image Encoder for IP-Adapter Plus (CLIP-ViT-H-14)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, torch_dtype=self.dtype
        ).to(self.device)
        
        # 2. Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        
        # 3. Load SD Pipeline
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
        print("Loading IP-Adapter Plus weights...")
        self.pipe.load_ip_adapter(
            os.path.join(ip_adapter_path, "models"), 
            subfolder="", 
            weight_name="ip-adapter-plus_sd15.bin"
        )
        self.pipe.set_ip_adapter_scale(0.7)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def get_canny_image(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def process_frame(self, frame, ref_image, prompt):
        # Resize to 512x512 for SD 1.5
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
        
        return np.array(output.resize((frame.shape[1], frame.shape[0])))

    def process_video(self, video_path, ref_image_path, output_video_path, prompt="a person"):
        clip = mp.VideoFileClip(video_path)
        ref_image = Image.open(ref_image_path).convert("RGB").resize((224, 224))
        
        test_duration = min(clip.duration, 0.5)
        print(f"Processing {test_duration} seconds of video with IP-Adapter Plus...")
        
        frames = []
        count = 0
        for frame in clip.iter_frames():
            if count / clip.fps > test_duration:
                break
            new_frame = self.process_frame(frame, ref_image, prompt)
            frames.append(new_frame)
            count += 1
            
        new_clip = mp.ImageSequenceClip(frames, fps=clip.fps)
        new_clip.write_videofile(output_video_path, codec="libx264", audio=False)
        return output_video_path

if __name__ == "__main__":
    pass
