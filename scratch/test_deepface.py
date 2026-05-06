import os
import numpy as np
from deepface import DeepFace
import cv2

def test_deepface():
    # Try to find an image in samples
    sample_dir = "samples"
    if not os.path.exists(sample_dir):
        print(f"Sample directory {sample_dir} not found.")
        return

    images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"No images found in {sample_dir}.")
        return

    img_path = os.path.join(sample_dir, images[0])
    print(f"Testing DeepFace with {img_path}...")

    try:
        res = DeepFace.represent(img_path, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
        print(f"Success! Found {len(res)} faces.")
        print(f"Embedding size: {len(res[0]['embedding'])}")
    except Exception as e:
        print(f"Failed: {str(e)}")

if __name__ == "__main__":
    test_deepface()
