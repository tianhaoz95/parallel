import cv2
import numpy as np
from deepface import DeepFace
from logger_utils import logger
import os

class IdentityDiscoverer:
    def __init__(self, threshold=0.4, model_name='VGG-Face'):
        self.threshold = threshold
        self.model_name = model_name

    def find_unique_faces(self, video_path, sample_rate=1.0):
        """Scans video and extracts unique face identities."""
        logger.info(f"Scanning {video_path} for unique identities using {self.model_name}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        identities = []
        
        thumb_dir = "temp_thumbnails"
        if os.path.exists(thumb_dir):
            import shutil
            shutil.rmtree(thumb_dir)
        os.makedirs(thumb_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Sample every second
            if int(fps) > 0 and frame_count % int(fps) == 0:
                try:
                    # Detect faces
                    faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
                    
                    for face_data in faces:
                        if face_data['confidence'] < 0.9: continue
                        
                        face_img = face_data['face']
                        # DeepFace returns face normalized [0,1], convert to uint8
                        thumb = (face_img * 255).astype(np.uint8)
                        
                        # Get embedding
                        objs = DeepFace.represent(face_img, model_name=self.model_name, enforce_detection=False)
                        if not objs: continue
                        emb = objs[0]['embedding']
                        
                        # Check for existing identity
                        is_new = True
                        for existing in identities:
                            dist = self.calculate_distance(emb, existing['embedding'])
                            if dist < self.threshold:
                                is_new = False
                                break
                        
                        if is_new:
                            id_num = len(identities)
                            thumb_path = os.path.join(thumb_dir, f"id_{id_num}.jpg")
                            cv2.imwrite(thumb_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
                            identities.append({
                                'id': id_num,
                                'embedding': emb,
                                'thumbnail_path': thumb_path
                            })
                            logger.info(f"New identity found: ID {id_num}")
                except Exception as e:
                    pass
            
            frame_count += 1
            if len(identities) >= 12: break # Limit
            
        cap.release()
        return identities

    def calculate_distance(self, emb1, emb2):
        a = np.array(emb1)
        b = np.array(emb2)
        return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

if __name__ == "__main__":
    pass
