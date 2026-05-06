import cv2
import numpy as np
import mediapipe as mp
import os
from logger_utils import logger

class SpeakerIdentification:
    def __init__(self):
        import urllib.request
        model_path = 'models/face_landmarker.task'
        if not os.path.exists(model_path):
            os.makedirs('models', exist_ok=True)
            logger.info("Downloading Mediapipe Face Landmarker model...")
            urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=5)
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        # Landmark indices for upper and lower lips
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.MOUTH_CENTER = 13 # Approx center

    def get_mouth_openness(self, face_landmarks):
        """Calculates a simple distance metric for mouth openness."""
        # Top of upper lip (landmark 0) to bottom of lower lip (landmark 17)
        top = face_landmarks[0]
        bottom = face_landmarks[17]
        distance = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        return distance

    def analyze_frame(self, frame):
        """Returns a list of (face_index, openness) for all detected faces."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.face_mesh.detect(mp_image)
        speaker_data = []
        
        if results.face_landmarks:
            for i, face_landmarks in enumerate(results.face_landmarks):
                openness = self.get_mouth_openness(face_landmarks)
                # Get face center/id proxy
                bbox = face_landmarks[1] # Tip of nose
                speaker_data.append({
                    'id': i,
                    'pos': (bbox.x, bbox.y),
                    'openness': openness
                })
        return speaker_data

    def detect_speakers_in_video(self, video_path, duration=None):
        """Analyzes a video to track mouth activity for each detected character."""
        logger.info(f"Analyzing visual speaker activity in {video_path}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Track openness scores over time for each face
        # { face_id: [openness_at_frame_n, ...] }
        activity_log = {}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if duration and (frame_count / fps) > duration: break
            
            # Analyze every 2nd frame for speed
            if frame_count % 2 == 0:
                speakers = self.analyze_frame(frame)
                for s in speakers:
                    face_id = s['id'] # In a real implementation, we'd use a persistent tracker ID
                    if face_id not in activity_log: activity_log[face_id] = []
                    activity_log[face_id].append((frame_count / fps, s['openness']))
            
            frame_count += 1
        
        cap.release()
        return activity_log

if __name__ == "__main__":
    pass
