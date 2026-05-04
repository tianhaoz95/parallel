import os
import json
import shutil
import numpy as np
from logger_utils import logger

class IdentityLibrary:
    def __init__(self, library_dir="identity_library"):
        self.library_dir = library_dir
        self.metadata_file = os.path.join(library_dir, "identities.json")
        os.makedirs(library_dir, exist_ok=True)
        self.identities = self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load identity metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.identities, f, indent=4)

    def save_character(self, name, face_embedding, ref_image_paths, ref_audio_path=None):
        """Saves a character profile to the persistent library."""
        char_dir = os.path.join(self.library_dir, name)
        os.makedirs(char_dir, exist_ok=True)
        
        # Copy images to library
        saved_img_paths = []
        for i, img_path in enumerate(ref_image_paths):
            ext = os.path.splitext(img_path)[1]
            dest = os.path.join(char_dir, f"ref_{i}{ext}")
            shutil.copy(img_path, dest)
            saved_img_paths.append(dest)
            
        # Copy audio if provided
        saved_audio_path = None
        if ref_audio_path and os.path.exists(ref_audio_path):
            ext = os.path.splitext(ref_audio_path)[1]
            dest = os.path.join(char_dir, f"voice{ext}")
            shutil.copy(ref_audio_path, dest)
            saved_audio_path = dest
            
        # Update metadata
        self.identities[name] = {
            "name": name,
            "embedding": face_embedding.tolist() if isinstance(face_embedding, np.ndarray) else face_embedding,
            "images": saved_img_paths,
            "audio": saved_audio_path,
            "created_at": str(os.path.getctime(char_dir))
        }
        self._save_metadata()
        logger.info(f"Character '{name}' saved to library.")
        return True

    def get_all_characters(self):
        return self.identities

    def find_match(self, current_embedding, threshold=0.4):
        """Tries to match a face embedding against the library."""
        best_match = None
        min_dist = threshold
        
        curr_emb = np.array(current_embedding)
        
        for name, data in self.identities.items():
            lib_emb = np.array(data['embedding'])
            # Cosine distance
            dist = 1 - (np.dot(curr_emb, lib_emb) / (np.linalg.norm(curr_emb) * np.linalg.norm(lib_emb)))
            
            if dist < min_dist:
                min_dist = dist
                best_match = name
                
        return best_match

    def remove_character(self, name):
        if name in self.identities:
            del self.identities[name]
            shutil.rmtree(os.path.join(self.library_dir, name), ignore_errors=True)
            self._save_metadata()
            return True
        return False

if __name__ == "__main__":
    pass
