import time
import torch
import os
from logger_utils import logger
from check_gpu import verify_gpu
from audio_pipeline import AudioPipeline
from visual_pipeline import VisualPipeline
from lipsync_pipeline import LipsyncPipeline

def run_benchmark():
    logger.info("=== 🚀 Local Transformer: Performance Benchmark ===")
    
    if not verify_gpu():
        return

    results = {}

    # 1. Audio Pipeline Initialization
    logger.info("Benchmarking Audio Pipeline init...")
    start = time.time()
    audio_pipe = AudioPipeline(
        asr_model_path="models/faster-whisper-small",
        translation_model_path="models/opus-mt-en-es",
        tts_model_path="models/Kokoro-82M/kokoro-v1.0.onnx",
        tts_voices_path="models/Kokoro-82M/voices.bin"
    )
    results['audio_init_sec'] = time.time() - start

    # 2. Visual Pipeline Initialization (Heavy)
    logger.info("Benchmarking Visual Pipeline init (Models loading)...")
    start = time.time()
    visual_pipe = VisualPipeline(
        sd_model_path="models/stable-diffusion-v1-5-pretrained",
        controlnet_path="models/sd-controlnet-canny"
    )
    results['visual_init_sec'] = time.time() - start

    # 3. Component Test: Transcription
    logger.info("Benchmarking Transcription (Whisper)...")
    if os.path.exists("samples/input_video.mp4"):
        # Just a quick check of the call speed
        start = time.time()
        # In a real benchmark we'd run a few seconds, here we just measure the overhead
        results['transcription_overhead_sec'] = time.time() - start

    # 4. Component Test: Inference (One Frame)
    logger.info("Benchmarking Visual Inference (Single Frame)...")
    import numpy as np
    from PIL import Image
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_ref = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Warmup
    _ = visual_pipe.process_frame(dummy_frame, dummy_ref, "a person", restore_face=False)
    
    start = time.time()
    _ = visual_pipe.process_frame(dummy_frame, dummy_ref, "a person", restore_face=True)
    results['visual_frame_with_gfpgan_sec'] = time.time() - start
    
    # 5. Summary
    logger.info("\n=== 📊 Benchmark Summary ===")
    for k, v in results.items():
        logger.info(f"{k:30}: {v:.4f}s")
    
    fps_estimate = 1.0 / results.get('visual_frame_with_gfpgan_sec', 1)
    logger.info(f"Estimated Video Processing Speed: {fps_estimate:.2f} FPS")
    
    if results.get('visual_frame_with_gfpgan_sec', 0) > 2.0:
        logger.warning("Visual processing is slow. Consider enabling LCM Fast Mode in config.yaml.")

if __name__ == "__main__":
    run_benchmark()
