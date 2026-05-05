import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import librosa
import os
from tqdm import tqdm
import static_ffmpeg
from deepface import DeepFace
from logger_utils import logger

# Ensure ffmpeg/ffprobe are in PATH
static_ffmpeg.add_paths()

class LipsyncPipeline:
    def __init__(self, model_path="models/wav2lip_256.onnx", input_size=256):
        import os
        if os.environ.get("USE_CPU") == "1":
            model_path = "models/wav2lip.onnx"
            input_size = 96
            logger.info("CPU Testing Mode active: falling back to 96x96 wav2lip model.")
            
        self.input_size = input_size
        
        logger.info(f"Loading segment-aware Lipsync model from {model_path}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Optimize ORT for CPU if in test mode
        sess_options = ort.SessionOptions()
        if os.environ.get("USE_CPU") == "1":
            sess_options.intra_op_num_threads = os.cpu_count() or 4
            
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def get_all_face_crops(self, frame):
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_data = []
        if not results.detections: return face_data
        
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
            padding_h = int((y2 - y1) * 0.2); padding_w = int((x2 - x1) * 0.1)
            x1_p, y1_p = max(0, x1 - padding_w), max(0, y1 - padding_h)
            x2_p, y2_p = min(w, x2 + padding_w), min(h, y2 + padding_h)
            
            face_data.append({
                'coords': (x1_p, y1_p, x2_p, y2_p),
                'crop': frame[y1_p:y2_p, x1_p:x2_p]
            })
        return face_data

    def preprocess_audio(self, audio_path, fps=25):
        wav, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, n_fft=800, hop_length=200, win_length=800)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel + 40) / 40
        mel = np.clip(mel, 0, 1)
        mel_chunks = []
        mel_idx_multiplier = 16000 / (fps * 200)
        num_frames = int(len(wav) / (16000 / fps))
        for i in range(num_frames):
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]: break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
        return mel_chunks

    def process_video_segmented(self, video_path, audio_path, output_path, speaker_segments):
        """
        speaker_segments: list of {'start', 'end', 'target_embedding'}
        """
        logger.info(f"Segment-Aware Lipsync: {video_path} -> {output_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps == 0: fps = 25
            
        mel_chunks = self.preprocess_audio(audio_path, fps)
        temp_video = "temp_segmented_sync.mp4"
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(mel_chunks): break
            
            curr_time = frame_idx / fps
            
            # Find which character is speaking at this time
            target_emb = None
            for s in speaker_segments:
                if s['start'] <= curr_time <= s['end']:
                    target_emb = s.get('target_embedding')
                    break
            
            if target_emb is None:
                # No one speaking, or no target mapped
                out.write(frame); frame_idx += 1; continue
                
            # 1. Find all faces
            faces = self.get_all_face_crops(frame)
            if not faces:
                out.write(frame); frame_idx += 1; continue
            
            # 2. Identify target face
            target_face = None
            for f in faces:
                try:
                    res = DeepFace.verify(f['crop'], target_emb, detector_backend='skip', enforce_detection=False)
                    if res['verified']:
                        target_face = f; break
                except: continue
            
            if target_face is None:
                out.write(frame); frame_idx += 1; continue
                
            # 3. Perform Sync on target
            x1, y1, x2, y2 = target_face['coords']
            face = cv2.resize(target_face['crop'], (self.input_size, self.input_size))
            face_masked = face.copy(); face_masked[self.input_size//2:, :] = 0
            img_input = np.expand_dims(np.concatenate([face_masked, face], axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
            mel_input = np.expand_dims(mel_chunks[frame_idx], axis=(0, 1)).astype(np.float32)
            
            pred = self.session.run(None, {input_names[0]: mel_input, input_names[1]: img_input})[0][0]
            pred = pred.transpose(1, 2, 0) * 255.0
            pred = cv2.resize(pred.astype(np.uint8), (x2-x1, y2-y1))
            
            frame[y1:y2, x1:x2] = pred
            out.write(frame)
            frame_idx += 1
            
        cap.release(); out.release()
        os.system(f"ffmpeg -y -i {temp_video} -i {audio_path} -c:v libx264 -c:a aac -shortest {output_path} > /dev/null 2>&1")
        if os.path.exists(temp_video): os.remove(temp_video)
        return output_path

if __name__ == "__main__":
    pass
