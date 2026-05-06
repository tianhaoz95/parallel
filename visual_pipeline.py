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
        
        self.is_mock = os.environ.get("USE_CPU") == "1"
        logger.info(f"Initializing Visual Pipeline on {self.device} (Mock Mode: {self.is_mock})")
        
        if not self.is_mock:
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
            if self.device == "cuda": self.pipe.enable_model_cpu_offload()
            
            self.face_restorer = GFPGANer(model_path=CONFIG.get('models', {}).get('restoration', {}).get('gfpgan'), upscale=1, arch='clean', channel_multiplier=2, device=self.device)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth', model=model, tile=400, tile_pad=10, pre_pad=0, half=True if self.device == "cuda" else False, device=self.device)
        
        import urllib.request
        model_path = 'models/selfie_segmenter.tflite'
        if not os.path.exists(model_path):
            os.makedirs('models', exist_ok=True)
            logger.info("Downloading Mediapipe Selfie Segmenter model...")
            urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite', model_path)
            
        options = mp_lib.tasks.vision.ImageSegmenterOptions(
            base_options=mp_lib.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_lib.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True)
        self.segmenter = mp_lib.tasks.vision.ImageSegmenter.create_from_options(options)
        
        # Face Landmarker for Tiny Face Swap
        landmarker_model_path = 'models/face_landmarker.task'
        if not os.path.exists(landmarker_model_path):
            logger.info("Downloading Mediapipe Face Landmarker model...")
            urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', landmarker_model_path)
        
        landmarker_options = mp_lib.tasks.vision.FaceLandmarkerOptions(
            base_options=mp_lib.tasks.BaseOptions(model_asset_path=landmarker_model_path),
            running_mode=mp_lib.tasks.vision.RunningMode.IMAGE,
            num_faces=5)
        self.landmarker = mp_lib.tasks.vision.FaceLandmarker.create_from_options(landmarker_options)
        
        self.vlm = VLMPrompter() if CONFIG.get('optimizations', {}).get('use_vlm', True) else None

    def consolidate_identity(self, image_paths):
        embeddings = []; valid_images = []
        for path in image_paths:
            try:
                res = DeepFace.represent(path, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                if res: 
                    embeddings.append(np.array(res[0]['embedding']))
                    valid_images.append(Image.open(path).convert("RGB").resize((224, 224)))
            except Exception as e:
                logger.error(f"Error processing reference image {path}: {str(e)}")
                pass
        if not embeddings: 
            logger.error(f"consolidate_identity: No valid faces found in {image_paths}")
            return None, []
        logger.info(f"consolidate_identity: Found {len(embeddings)} valid faces.")
        return np.mean(embeddings, axis=0), valid_images

    def get_canny_image(self, image):
        image = np.array(image); image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]; image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def restore_faces(self, frame):
        if getattr(self, 'is_mock', False): return frame
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

    def tiny_face_swap(self, target_frame, face_bbox, ref_image):
        """Ultra-lightweight face overlay using Mediapipe landmarks and affine warp."""
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']
        face_roi = target_frame[y:y+h, x:x+w]
        if face_roi.size == 0: return target_frame

        # 1. Get landmarks for reference image (could be cached)
        ref_np = np.array(ref_image)
        ref_mp = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=ref_np)
        ref_res = self.landmarker.detect(ref_mp)
        
        # 2. Get landmarks for target ROI
        roi_mp = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=face_roi)
        target_res = self.landmarker.detect(roi_mp)
        
        if not ref_res.face_landmarks or not target_res.face_landmarks:
            # Fallback to simple resize and overlay if landmarks fail
            ref_resized = cv2.resize(ref_np, (w, h))
            target_frame[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 0.3, ref_resized, 0.7, 0)
            return target_frame

        # Use 3 stable points for affine transform (e.g., eyes and nose)
        # Mediapipe indices: Left eye outer 33, Right eye outer 263, Nose tip 1
        ref_pts = np.array([
            [ref_res.face_landmarks[0][33].x * ref_np.shape[1], ref_res.face_landmarks[0][33].y * ref_np.shape[0]],
            [ref_res.face_landmarks[0][263].x * ref_np.shape[1], ref_res.face_landmarks[0][263].y * ref_np.shape[0]],
            [ref_res.face_landmarks[0][1].x * ref_np.shape[1], ref_res.face_landmarks[0][1].y * ref_np.shape[0]]
        ], dtype=np.float32)

        target_pts = np.array([
            [target_res.face_landmarks[0][33].x * w, target_res.face_landmarks[0][33].y * h],
            [target_res.face_landmarks[0][263].x * w, target_res.face_landmarks[0][263].y * h],
            [target_res.face_landmarks[0][1].x * w, target_res.face_landmarks[0][1].y * h]
        ], dtype=np.float32)

        matrix = cv2.getAffineTransform(ref_pts, target_pts)
        warped_face = cv2.warpAffine(ref_np, matrix, (w, h))
        
        try:
            # 100% replacement for clear visual proof in PoC
            ref_np = np.array(ref_image)
            ref_resized = cv2.resize(ref_np, (w, h))
            target_frame[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 0.0, warped_face, 1.0, 0)
            # Visual watermark for debug
            cv2.circle(target_frame, (x+10, y+10), 10, (0, 255, 0), -1) 
            return target_frame
        except Exception as e:
            logger.error(f"Tiny Face Swap failed during blending: {str(e)}")
            # Fail-safe: just draw a blue box so we know it tried
            cv2.rectangle(target_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            return target_frame

    def process_frame_multi(self, frame, identity_map, prompt_base, restore_face=True):
        final_frame = frame.copy()
        try:
            faces_data = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        except Exception as e:
            logger.error(f"DeepFace face extraction failed: {str(e)}")
            return frame

        for face_data in faces_data:
            face_img = (face_data['face'] * 255).astype(np.uint8)
            if face_img.size == 0: continue
            try:
                res = DeepFace.represent(face_img, model_name='VGG-Face', detector_backend='skip', enforce_detection=False)
                if not res: continue
                curr_emb = np.array(res[0]['embedding'])
            except Exception as e:
                logger.warning(f"Failed to represent face: {str(e)}")
                continue
            
            matched_id = None; 
            # If only one identity is provided, be extremely lenient for PoC
            best_dist = 0.7 if len(identity_map) == 1 else 0.4
            
            for name, data in identity_map.items():
                dist = 1 - (np.dot(curr_emb, data['consolidated_embedding']) / (np.linalg.norm(curr_emb) * np.linalg.norm(data['consolidated_embedding'])))
                if dist < best_dist: best_dist = dist; matched_id = name
            
            if matched_id:
                logger.info(f"Matched face to {matched_id} with distance {best_dist:.3f}")
                if getattr(self, 'is_mock', False):
                    # Improved CPU Mocking: Apply tiny face swap
                    face_bbox = face_data['facial_area']
                    # Add debug label
                    cv2.putText(final_frame, f"Match: {best_dist:.2f}", (face_bbox['x'], face_bbox['y']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    ref_img = identity_map[matched_id]['images'][0] if identity_map[matched_id]['images'] else None
                    if ref_img:
                        try:
                            final_frame = self.tiny_face_swap(final_frame, face_bbox, ref_img)
                        except Exception as e:
                            logger.error(f"tiny_face_swap call failed: {str(e)}")
                    else:
                        # Fallback to green tint if no reference images
                        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']
                        face_roi = final_frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 100, 200)
                            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                            edges_rgb[:, :, 0] = 0; edges_rgb[:, :, 2] = 0
                            final_frame[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 0.3, edges_rgb, 0.7, 0)
                    continue

                # Use Identity-specific prompt if available, else fallback to base
                char_prompt = identity_map[matched_id].get('prompt', prompt_base)
                
                pil_frame = Image.fromarray(frame).resize((512, 512))
                canny_image = self.get_canny_image(pil_frame)
                num_steps = 4 if getattr(self, 'use_lcm', False) else CONFIG.get('defaults', {}).get('num_inference_steps', 15)
                
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
                mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                seg_result = self.segmenter.segment(mp_image)
                mask = (seg_result.category_mask.numpy_view() > 0.5).astype(np.uint8) * 255
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
        import imageio
        writer = imageio.get_writer("temp_streaming.mp4", fps=fps, codec='libx264', quality=8)
        
        test_duration = min(clip.duration, 5.0) if os.environ.get("USE_CPU") == "1" else clip.duration; count = 0; prev_frame = None
        self.current_context = ""
        try:
            for frame in clip.iter_frames():
                if count / fps > test_duration: break
                if (count % 30 == 0) and self.vlm:
                    self.current_context = self.vlm.describe_frame(frame)
                
                transformed = self.process_frame_multi(frame, identity_map, prompt, restore_face=restore_face)
                
                # DIFF CHECK
                diff = np.sum(np.abs(transformed.astype(float) - frame.astype(float)))
                if diff > 0:
                    logger.info(f"Frame {count} modified! Diff: {diff:.0f}")
                
                if upscale: 
                    frame_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                    transformed_up, _ = self.upscaler.enhance(frame_bgr, outscale=2)
                    writer.append_data(cv2.cvtColor(transformed_up, cv2.COLOR_BGR2RGB))
                else: 
                    writer.append_data(transformed)
                prev_frame = frame.copy(); count += 1
        finally:
            writer.close()
            clip.close()
            import time
            time.sleep(1.0)
        os.rename("temp_streaming.mp4", output_video_path)
        return output_video_path

if __name__ == "__main__":
    pass
