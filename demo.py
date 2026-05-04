import os
import sys
from main import main as run_pipeline
from unittest.mock import patch

def run_test():
    print("=== Running End-to-End Functional Test ===")
    
    # 1. Check for sample assets
    video_path = "samples/input_video.mp4"
    ref_image = "samples/ref_image.jpg"
    ref_audio = "samples/ref_audio.flac"
    
    if not all(os.path.exists(p) for p in [video_path, ref_image, ref_audio]):
        print("Error: Missing sample assets in samples/ directory.")
        return

    # 2. Configure test arguments
    output_path = "test_output.mp4"
    args = [
        "--video", video_path,
        "--ref_image", ref_image,
        "--ref_audio", ref_audio,
        "--target_lang", "es",
        "--prompt", "a portrait of a character",
        "--output", output_path
    ]
    
    print(f"Executing: python main.py {' '.join(args)}")
    
    # 3. Run the pipeline
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            run_pipeline()
            if os.path.exists(output_path):
                print(f"\nSUCCESS: Test passed. Generated {output_path}")
            else:
                print("\nFAILURE: Pipeline completed but output file is missing.")
        except Exception as e:
            print(f"\nFAILURE: Pipeline crashed with error: {str(e)}")

if __name__ == "__main__":
    run_test()
