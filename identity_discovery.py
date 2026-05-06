import cv2
import numpy as np
from deepface import DeepFace
from logger_utils import logger
from identity_library import IdentityLibrary
import os

class IdentityDiscoverer:
    def __init__(self, threshold=0.4, model_name='VGG-Face'):
        self.threshold = threshold
        self.model_name = model_name
        self.library = IdentityLibrary()

    def find_unique_faces(self, video_path, sample_rate=1.0):
        """Scans video and extracts unique face identities with library matching."""
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
            
            if int(fps) > 0 and frame_count % int(fps) == 0:
                try:
                    faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
                    
                    for face_data in faces:
                        if face_data['confidence'] < 0.9: continue
                        
                        face_img = face_data['face']
                        thumb = (face_img * 255).astype(np.uint8)
                        
                        objs = DeepFace.represent(face_img, model_name=self.model_name, enforce_detection=False)
                        if not objs: continue
                        emb = objs[0]['embedding']
                        
                        # 1. Check for existing identity in current scan
                        is_new = True
                        for existing in identities:
                            dist = self.calculate_distance(emb, existing['embedding'])
                            if dist < self.threshold:
                                is_new = False
                                break
                        
                        if is_new:
                            # 2. Check for match in Library
                            library_match = self.library.find_match(emb, threshold=self.threshold)
                            
                            id_num = len(identities)
                            thumb_path = os.path.join(thumb_dir, f"id_{id_num}.jpg")
                            cv2.imwrite(thumb_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
                            
                            identities.append({
                                'id': id_num,
                                'embedding': emb,
                                'thumbnail_path': thumb_path,
                                'library_match': library_match # Name of character if matched
                            })
                            
                            if library_match:
                                logger.info(f"Recognized character from library: {library_match}")
                            else:
                                logger.info(f"New identity found: ID {id_num}")
                except Exception as e:
                    logger.error(f"Error in identity discovery: {str(e)}")
                    pass
            
            frame_count += 1
            if len(identities) >= 12: break
            
        cap.release()
        return identities

    def calculate_distance(self, emb1, emb2):
        a = np.array(emb1)
        b = np.array(emb2)
        return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

if __name__ == "__main__":
    pass
