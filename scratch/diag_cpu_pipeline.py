import os
import cv2
import numpy as np
from deepface import DeepFace
import moviepy.editor as mp
import sys
import subprocess
import json

def create_test_assets():
    print("Creating test assets with DIFFERENT identities...")
    # 1. Base Image (Person A)
    if os.path.exists("temp_thumbnails/id_0.jpg"):
        img_a = cv2.imread("temp_thumbnails/id_0.jpg")
    else:
        print("Error: temp_thumbnails/id_0.jpg not found.")
        return None, None

    # 2. Reference Image (Person B - just a color-shifted version to be "different")
    # Actually, let's just use a totally different image if possible, 
    # but for a self-contained test, we'll invert colors to ensure a huge embedding distance.
    img_b = cv2.bitwise_not(img_a) 
    
    ref_path = "scratch/test_ref_person_b.jpg"
    cv2.imwrite(ref_path, img_b)

    # 3. Video (Person A)
    video_path = "scratch/test_input_person_a.mp4"
    height, width, _ = img_a.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    for _ in range(30):
        video_writer.write(img_a)
    video_writer.release()

    return video_path, ref_path

def run_pipeline(video_path, ref_path):
    print("Running pipeline in CPU mode...")
    output_path = "scratch/test_output_swapped.mp4"
    if os.path.exists(output_path): os.remove(output_path)
    
    identity_map = {"Character_0": {"images": [os.path.abspath(ref_path)]}}
    
    cmd = [
        ".venv/bin/python3", "main.py",
        "--video", video_path,
        "--identity_map", json.dumps(identity_map),
        "--output", output_path,
        "--skip_lipsync"
    ]
    env = os.environ.copy()
    env["USE_CPU"] = "1"
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"Pipeline failed: {res.stderr}")
        return None
    return output_path

def check_similarity(original_video, transformed_video, reference_image):
    print("Checking similarity...")
    def get_emb(img):
        res = DeepFace.represent(img, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
        return np.array(res[0]['embedding']) if res else None

    ref_img = cv2.imread(reference_image)
    ref_emb = get_emb(ref_img)

    orig_clip = mp.VideoFileClip(original_video)
    orig_frame = orig_clip.get_frame(0.5)
    orig_emb = get_emb(orig_frame)

    trans_clip = mp.VideoFileClip(transformed_video)
    trans_frame = trans_clip.get_frame(0.5)
    trans_emb = get_emb(trans_frame)

    if ref_emb is None or orig_emb is None or trans_emb is None:
        return False

    def sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_to_orig = sim(trans_emb, orig_emb)
    sim_to_ref = sim(trans_emb, ref_emb)

    print(f"Similarity to Original (Person A): {sim_to_orig:.4f}")
    print(f"Similarity to Reference (Person B): {sim_to_ref:.4f}")

    # Success if Transformed is closer to B than A
    return sim_to_ref > sim_to_orig

if __name__ == "__main__":
    video_path, ref_path = create_test_assets()
    if not video_path: sys.exit(1)
    
    output_path = run_pipeline(video_path, ref_path)
    if not output_path: sys.exit(1)
    
    if check_similarity(video_path, output_path, ref_path):
        print("✅ SUCCESS: The face is now mathematically more similar to the reference image!")
        sys.exit(0)
    else:
        print("❌ FAILURE: The face is still mathematically similar to the original.")
        sys.exit(1)
