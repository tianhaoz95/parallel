# Design Document: CPU-Only Testing and Verification Mode

## Objective
To implement a "CPU-Only / Testing Mode" that allows developers and users to verify pipeline logic, test integrations, and iterate rapidly without requiring expensive, high-end GPU hardware. The goal is *not* to produce high-quality video, but to ensure the end-to-end data flow (transcription, translation, face detection, generation flow, lipsync) works correctly at the highest possible speed on a CPU.

## Current Bottlenecks on CPU
1. **Hardware Blocks**: `check_gpu.py` explicitly blocks execution if `torch.cuda.is_available()` is False.
2. **Heavy Visual Models**: Stable Diffusion v1.5 + ControlNet + IP-Adapter takes minutes per frame on a CPU.
3. **High-Resolution Lipsync**: `wav2lip_256.onnx` is significantly slower than the base 96x96 model.
4. **ASR/TTS**: Whisper `large-v3` and `F5TTS` are computationally heavy for CPU inference.

## Proposed Implementation Plan

### 1. Hardware Check Bypass (`check_gpu.py`)
Modify the entry point to accept a `--cpu-only` flag or a `TESTING_MODE=1` environment variable.
- Instead of exiting with an error when CUDA is missing, log a prominent warning: `"Running in CPU-Only Testing Mode. Quality will be drastically reduced and processing will be slow."`
- Set a global state (e.g., `CONFIG['testing_mode'] = True`) that other modules can read.

### 2. CPU Tier in Adaptive Loader (`adaptive_loader.py`)
Extend the `HardwareAdaptiveLoader` to include an "Ultra-Low / CPU" tier.
When `torch.cuda.is_available()` is False or Testing Mode is active, apply these optimizations:
- `whisper_model`: `'tiny'` (Extremely fast on CPU).
- `compute_type`: `'float32'` (CPU optimized, avoid `float16` overhead if hardware doesn't support it natively).
- `use_inpainting`: `False`.
- `use_upscaling`: `False`.
- `restore_face`: `False`.

### 3. Visual Pipeline Modifications (`visual_pipeline.py`)
The visual pipeline is the heaviest component. We propose two levels of CPU fallback:

**Option A: Tiny Diffusion (Low Quality, Tests ML execution)**
- Swap `stable-diffusion-v1-5` for `OFA-Sys/small-stable-diffusion-v0` or use `LCM` (Latent Consistency Models) with only **1 to 2 inference steps**.
- Downscale all processing frames to `128x128` or `256x256` before passing them to the pipeline.

**Option B: Mock Visuals (Maximum Speed, Tests Pipeline Logic)**
- Bypass the HuggingFace Diffusers pipeline entirely if `TESTING_MODE` is active.
- Instead of generating a new image, apply a simple OpenCV filter (e.g., Canny Edge + a color tint based on the identity) to the detected faces. This proves that face detection, identity mapping, and masking logic work, running in real-time on CPU.

### 4. Lipsync Pipeline (`lipsync_pipeline.py`)
- Switch the ONNX model from `wav2lip_256.onnx` to the standard `wav2lip.onnx` (96x96 resolution).
- ONNX Runtime already falls back to `CPUExecutionProvider`. Ensure `intra_op_num_threads` is configured to utilize all available CPU cores.

### 5. Audio Pipeline (`audio_pipeline.py`)
- **ASR**: Use `faster-whisper-tiny`.
- **TTS**: Prefer `Kokoro` (ONNX) over `F5TTS`. ONNX is highly optimized for CPU inference. If `F5TTS` must be used, limit the reference audio duration to <3 seconds.
- **Translation**: `MarianMT` works on CPU but can be slow. Limit the max tokens or use the LLM translation fallback with a fast, local CPU-friendly LLM (like `llama.cpp` with a 1B model) or external API.

## Summary of Configuration Overrides in CPU Mode

| Component | GPU / Production | CPU / Testing Mode |
| :--- | :--- | :--- |
| **ASR** | faster-whisper-large-v3 | faster-whisper-tiny |
| **Visual Gen** | SD v1.5 + ControlNet (15 steps) | Mock Filter / Tiny-SD (1-2 steps) |
| **Lipsync** | wav2lip_256.onnx | wav2lip.onnx (96x96) |
| **Face Restoration** | GFPGAN | Disabled |
| **Resolution** | 512x512 / Upscaled | 128x128 or 256x256 |

## Verification
To verify this implementation, developers will run:
```bash
USE_CPU=1 python main.py --video samples/test.mp4 --prompt "test"
```
The output should generate successfully within a few minutes (rather than hours), displaying heavily pixelated or mock-filtered faces, but with correctly timed audio, subtitles, and lip movements, proving the core logic functions correctly.
