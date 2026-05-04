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
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
import moviepy.editor as mp
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import mediapipe as mp_lib
from deepface import DeepFace
from logger_utils import logger, CONFIG
from vlm_prompter import VLMPrompter

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
        self.vlm = VLMPrompter() if CONFIG.get('optimizations', {}).get('use_vlm', True) else None
        
        if self.device == "cuda": self.pipe.enable_model_cpu_offload()

    def consolidate_identity(self, image_paths):
        embeddings = []; valid_images = []
        for path in image_paths:
            try:
                res = DeepFace.represent(path, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                if res: 
                    embeddings.append(np.array(res[0]['embedding']))
                    valid_images.append(Image.open(path).convert("RGB").resize((224, 224)))
            except: pass
        if not embeddings: return None, []
        return np.mean(embeddings, axis=0), valid_images

    def get_canny_image(self, image):
        image = np.array(image); image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]; image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def restore_faces(self, frame):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, _, restored_img = self.face_restorer.enhance(frame_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    def match_skin_tone(self, target_img, source_img):
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        t_mean, t_std = cv2.meanStdDev(target_lab); s_mean, s_std = cv2.meanStdDev(source_lab)
        for i in range(3):
            target_lab[:,:,i] = (target_lab[:,:,i] - t_mean[i]) * (s_std[i] / (t_std[i] + 1e-6)) + s_mean[i]
        target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)

    def process_frame_multi(self, frame, identity_map, prompt_base, restore_face=True):
        final_frame = frame.copy()
        try:
            faces_data = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        except: return frame

        for face_data in faces_data:
            face_img = (face_data['face'] * 255).astype(np.uint8)
            if face_img.size == 0: continue
            try:
                res = DeepFace.represent(face_img, model_name='VGG-Face', detector_backend='skip', enforce_detection=False)
                if not res: continue
                curr_emb = np.array(res[0]['embedding'])
            except: continue
            
            matched_id = None; best_dist = 0.4
            for name, data in identity_map.items():
                dist = 1 - (np.dot(curr_emb, data['consolidated_embedding']) / (np.linalg.norm(curr_emb) * np.linalg.norm(data['consolidated_embedding'])))
                if dist < best_dist: best_dist = dist; matched_id = name
            
            if matched_id:
                # Use Identity-specific prompt if available, else fallback to base
                char_prompt = identity_map[matched_id].get('prompt', prompt_base)
                
                pil_frame = Image.fromarray(frame).resize((512, 512))
                canny_image = self.get_canny_image(pil_frame)
                num_steps = 4 if self.use_lcm else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
                
                full_prompt = f"{char_prompt}, {matched_id}"
                if hasattr(self, 'current_context') and self.current_context:
                    full_prompt += f", {self.current_context}"

                output = self.pipe(
                    prompt=full_prompt, image=pil_frame, 
                    ip_adapter_image=identity_map[matched_id]['images'], 
                    control_image=canny_image,
                    strength=CONFIG.get('defaults', {}).get('sd_strength', 0.6), 
                    num_inference_steps=num_steps
                ).images[0]
                
                transformed_full = np.array(output.resize((frame.shape[1], frame.shape[0])))
                transformed_matched = self.match_skin_tone(transformed_full, frame)
                mask = (self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask > 0.5).astype(np.uint8) * 255
                mask_norm = cv2.GaussianBlur(mask, (15, 15), 0).astype(float) / 255.0
                if mask_norm.ndim == 2: mask_norm = np.stack([mask_norm]*3, axis=-1)
                final_frame = (transformed_matched.astype(float) * mask_norm + final_frame.astype(float) * (1.0 - mask_norm)).astype(np.uint8)

        if restore_face: final_frame = self.restore_faces(final_frame)
        return final_frame

    def process_video_multi(self, video_path, identity_map_paths, output_video_path, prompt="a person", restore_face=True, upscale=False):
        logger.info(f"Processing multi-character video with Identity Customization...")
        
        identity_map = {}
        for name, data in identity_map_paths.items():
            # data can be list of paths or dict {'images': [], 'prompt': ''}
            paths = data['images'] if isinstance(data, dict) else data
            emb, imgs = self.consolidate_identity(paths)
            if emb is not None:
                identity_map[name] = {
                    'images': imgs, 
                    'consolidated_embedding': emb,
                    'prompt': data.get('prompt') if isinstance(data, dict) else None
                }

        clip = mp.VideoFileClip(video_path)
        fps = clip.fps; out_w, out_h = (clip.w * 2, clip.h * 2) if upscale else (clip.w, clip.h)
        out = cv2.VideoWriter("temp_streaming.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
        
        test_duration = min(clip.duration, 1.0); count = 0; prev_frame = None
        self.current_context = ""
        try:
            for frame in clip.iter_frames():
                if count / fps > test_duration: break
                if (count % 30 == 0) and self.vlm:
                    self.current_context = self.vlm.describe_frame(frame)
                
                transformed = self.process_frame_multi(frame, identity_map, prompt, restore_face=restore_face)
                if upscale: 
                    frame_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                    transformed, _ = self.upscaler.enhance(frame_bgr, outscale=2)
                    out.write(transformed)
                else: out.write(cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
                prev_frame = frame.copy(); count += 1
        finally:
            out.release(); clip.close()
        os.rename("temp_streaming.mp4", output_video_path)
        return output_video_path

if __name__ == "__main__":
    pass
