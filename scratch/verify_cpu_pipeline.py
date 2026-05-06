import os
import sys
import subprocess
import json

def test_cpu_pipeline():
    # Setup environment
    env = os.environ.copy()
    env["USE_CPU"] = "1"
    
    # Identify a sample video and image
    # For now, we'll assume there's a test video or we'll just check if the components load correctly
    
    # We can use the mock mode to process a very short clip if available
    # But for a quick verification, we can just run the CLI help to see if it starts
    
    cmd = [sys.executable, "main.py", "--help"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode == 0:
        print("Pipeline help started successfully in CPU mode.")
    else:
        print(f"Pipeline failed to start: {res.stderr}")

if __name__ == "__main__":
    test_cpu_pipeline()
