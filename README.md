# Local Video Character Replacement & Audio Translation

A local, end-to-end pipeline for transforming video content using state-of-the-art machine learning models. This project allows you to replace a character in a video with a person from a reference image and translate the audio into another language while maintaining vocal characteristics.

## Features
- **Visual Character Replacement**: Uses **Stable Diffusion 1.5** with **ControlNet Canny** and **IP-Adapter Plus** to replace people in videos based on a reference image.
- **High-Fidelity Visual Refinement**: Integrated **GFPGAN v1.4** to restore and sharpen faces in every frame.
- **Audio Pipeline**: 
    - **Transcription**: Powered by `faster-whisper`.
    - **Translation**: Uses `MarianMT` (Opus-MT) models for high-quality local translation.
    - **Zero-Shot Voice Cloning**: Powered by **F5-TTS** to clone voices from a 5-15s reference recording.
    - **Atmosphere Preservation**: Uses **Demucs** audio source separation to preserve original background music and sound effects.
- **Lip-Syncing**: Uses **Wav2Lip 256 ONNX** with **MediaPipe** face detection to synchronize the new character's lips.
- **Automatic Orchestration**: Unified CLI tool and Gradio Web UI that handle the entire process.

## Requirements
- **Hardware**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or higher). 24GB VRAM (RTX 3090/4090) recommended for best performance.
- **Operating System**: Linux (Ubuntu 22.04+ recommended).
- **Drivers**: NVIDIA Driver 550+ and CUDA 12.x+.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd parallel
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Note: Base dependencies are already installed if you are using the provided environment)*.

3. **Download Models**:
   The models are automatically managed, but you can trigger the download by running:
   ```bash
   # See demo.py or main.py for details on model paths
   ```
## Usage

### Option 1: Web Interface (Recommended)
A user-friendly Gradio web interface is provided for easier interaction.
```bash
source .venv/bin/activate
python3 app.py
```
Then open `http://localhost:7860` in your browser.

### Option 2: Docker (Easiest for setup)
Run the entire system in a container with all dependencies pre-configured.
```bash
docker-compose up --build
```
*(Requires NVIDIA Container Toolkit installed)*

### Option 3: CLI Tool
For batch processing or automation:


### Options
- `--video`: Path to the source MP4 video.
- `--ref_image`: Path to the reference image of the character to replace with.
- `--target_lang`: Target language code (e.g., `es` for Spanish, `fr` for French, `de` for German).
- `--prompt`: Text description to guide the diffusion model (important for lighting/style).
- `--output`: Path to save the final video.

## Project Structure
- `main.py`: The entry point for the full pipeline.
- `audio_pipeline.py`: Handles ASR, Translation, and TTS.
- `visual_pipeline.py`: Handles character replacement using IP-Adapter.
- `models/`: Local storage for all ML weights.
- `samples/`: Test assets.

## Implementation Details
- **Temporal Consistency**: Handled via ControlNet structure guidance at every frame.
- **VRAM Management**: Uses `enable_model_cpu_offload()` and FP16/CPU-mixed precision to fit into consumer GPUs (12GB+).
- **Zero-Shot Transfer**: Visuals use IP-Adapter Plus; Audio uses F5-TTS for high-fidelity zero-shot voice cloning.

## 💡 Tips for High-Quality Results

### 🖼 Visuals
- **Prompt Engineering**: The `--prompt` is critical. Use descriptions like "cinematic lighting", "high resolution", and "matching environment" to help the diffusion model blend the character into the scene.
- **Reference Image**: Use a clear, front-facing portrait with neutral lighting for the best IP-Adapter results.
- **Motion**: ControlNet Canny works best with clear, high-contrast video. If the video is too dark, try brightening it before processing.

### 🎙 Audio
- **Reference Audio**: Use 5-15 seconds of clean speech (no background music/noise).
- **Zero-Shot Accuracy**: If the cloned voice sounds "electronic", try a longer reference recording or ensure the transcription of the reference (handled automatically) is accurate.
- **Language**: MarianMT supports many language pairs. The default is `es` (Spanish), but you can change it to `fr`, `de`, `it`, etc., provided the model is downloaded.

## 🔧 Troubleshooting
- **Out of Memory (OOM)**: If you run out of VRAM, try closing other GPU-heavy apps or reducing the resolution in `visual_pipeline.py`.
- **ffmpeg errors**: Ensure `static-ffmpeg` is correctly installed. The script automatically handles paths, but sometimes a manual `source .venv/bin/activate` is required.
- **Indentation/Python errors**: Ensure you are using Python 3.12+ as specified in the requirements.
