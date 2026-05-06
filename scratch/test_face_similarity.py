import os
import cv2
import numpy as np
from deepface import DeepFace
import moviepy.editor as mp
import sys

def get_face_embedding(img):
    try:
        res = DeepFace.represent(img, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
        if res:
            return np.array(res[0]['embedding'])
    except:
        pass
    return None

def calculate_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    # Cosine similarity
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def run_face_comparison_test(original_video, transformed_video, reference_image):
    print(f"--- Running Face Comparison Test ---")
    print(f"Original: {original_video}")
    print(f"Transformed: {transformed_video}")
    print(f"Reference: {reference_image}")

    # 1. Load Reference Face
    ref_img = cv2.imread(reference_image)
    if ref_img is None:
        print(f"Error: Could not load reference image {reference_image}")
        return False
    ref_emb = get_face_embedding(ref_img)
    if ref_emb is None:
        print("Error: Could not find face in reference image")
        return False
    print("✅ Reference face embedding captured.")

    # 2. Extract frame from Original Video
    orig_clip = mp.VideoFileClip(original_video)
    orig_frame = orig_clip.get_frame(1.0) # Get frame at 1 second
    orig_emb = get_face_embedding(orig_frame)
    if orig_emb is None:
        print("Error: Could not find face in original video frame at 1s")
        return False
    print("✅ Original face embedding captured.")

    # 3. Extract frame from Transformed Video
    trans_clip = mp.VideoFileClip(transformed_video)
    trans_frame = trans_clip.get_frame(1.0)
    trans_emb = get_face_embedding(trans_frame)
    if trans_emb is None:
        print("Error: Could not find face in transformed video frame at 1s")
        return False
    print("✅ Transformed face embedding captured.")

    # 4. Compare
    sim_to_orig = calculate_similarity(trans_emb, orig_emb)
    sim_to_ref = calculate_similarity(trans_emb, ref_emb)

    print(f"\nResults:")
    print(f"Similarity to Original Face:  {sim_to_orig:.4f}")
    print(f"Similarity to Reference Face: {sim_to_ref:.4f}")

    if sim_to_ref > sim_to_orig:
        print("\n✅ TEST PASSED: Transformed face is more similar to the reference image!")
        return True
    else:
        print("\n❌ TEST FAILED: Transformed face is still more similar to the original video.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_face_similarity.py <orig_video> <trans_video> <ref_image>")
        sys.exit(1)
    
    success = run_face_comparison_test(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)
