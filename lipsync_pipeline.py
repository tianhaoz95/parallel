import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import librosa
import os
from tqdm import tqdm
import static_ffmpeg

# Ensure ffmpeg/ffprobe are in PATH
static_ffmpeg.add_paths()

class LipsyncPipeline:
    def __init__(self, model_path="models/wav2lip_256.onnx", input_size=256):
        self.input_size = input_size
        
        # Initialize ONNX Runtime session
        print(f"Loading Lipsync model from {model_path}...")
        # Use CUDA if available, otherwise CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def get_face_crop(self, frame):
        """Detects face and returns coordinates for cropping."""
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        
        # Get the first detected face
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
        
        # Wav2Lip expects a square crop including the chin
        # Expand the box slightly
        padding_h = int((y2 - y1) * 0.2)
        padding_w = int((x2 - x1) * 0.1)
        
        y1 = max(0, y1 - padding_h)
        y2 = min(h, y2 + padding_h)
        x1 = max(0, x1 - padding_w)
        x2 = min(w, x2 + padding_w)
        
        return (x1, y1, x2, y2)

    def preprocess_audio(self, audio_path, fps=25):
        """Converts audio to mel-spectrogram chunks."""
        wav, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, 
                                            n_fft=800, hop_length=200, win_length=800)
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize mel
        mel = (mel + 40) / 40
        mel = np.clip(mel, 0, 1)
        
        # Chunking (16 mel frames per video frame at fps)
        mel_chunks = []
        mel_idx_multiplier = 16000 / (fps * 200)
        
        num_frames = int(len(wav) / (16000 / fps))
        for i in range(num_frames):
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
            
        return mel_chunks

    def process_video(self, video_path, audio_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0: fps = 25
            
        mel_chunks = self.preprocess_audio(audio_path, fps)
        
        temp_video = "temp_lipsync_no_audio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        print(f"Processing {len(mel_chunks)} frames for lipsync...")
        for i in tqdm(range(len(mel_chunks))):
            ret, frame = cap.read()
            if not ret:
                break
                
            coords = self.get_face_crop(frame)
            if coords is None:
                out.write(frame)
                continue
                
            x1, y1, x2, y2 = coords
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                out.write(frame)
                continue
                
            # Resize for model
            face_resized = cv2.resize(face, (self.input_size, self.input_size))
            
            # Prepare inputs
            face_masked = face_resized.copy()
            face_masked[self.input_size//2:, :] = 0
            
            img_input = np.concatenate([face_masked, face_resized], axis=2) # (H, W, 6)
            img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            mel_input = np.expand_dims(mel_chunks[i], axis=(0, 1)).astype(np.float32)
            
            # Inference
            # The model has two inputs: 'audio' and 'video' (check metadata if needed)
            # Standard Wav2Lip ONNX naming: 'input.1' (mel), 'input.2' (img) or similar
            input_names = [inp.name for inp in self.session.get_inputs()]
            
            pred = self.session.run(None, {
                input_names[0]: mel_input,
                input_names[1]: img_input
            })[0][0]
            
            # Post-process
            pred = pred.transpose(1, 2, 0) * 255.0
            pred = cv2.resize(pred.astype(np.uint8), (x2-x1, y2-y1))
            
            # Paste back
            frame[y1:y2, x1:x2] = pred
            out.write(frame)
            
        cap.release()
        out.release()
        
        # Merge audio
        print("Merging audio and video...")
        os.system(f"ffmpeg -y -i {temp_video} -i {audio_path} -c:v libx264 -c:a aac -shortest {output_path}")
        
        if os.path.exists(temp_video):
            os.remove(temp_video)
            
        print(f"Lipsync result saved to {output_path}")
        return output_path

if __name__ == "__main__":
    pass
