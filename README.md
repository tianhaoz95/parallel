# Local Video Character Replacement & Audio Translation

A local, end-to-end pipeline for transforming video content using state-of-the-art machine learning models. This project allows you to replace a character in a video with a person from a reference image and translate the audio into another language while maintaining vocal characteristics.

## Features
- **Visual Character Replacement**: Uses **Stable Diffusion 1.5** with **ControlNet Canny** (for structure) and **IP-Adapter Plus** (for character features) to replace people in videos based on a reference image.
- **Audio Pipeline**: 
    - **Transcription**: Powered by `faster-whisper`.
    - **Translation**: Uses `MarianMT` (Opus-MT) models for high-quality local translation.
    - **Zero-Shot Voice Cloning**: Powered by **F5-TTS (Flow-Matching Diffusion)** to clone voices from a 5-15s reference recording.
- **Lip-Syncing**: Uses **Wav2Lip 256 ONNX** with **MediaPipe** face detection to synchronize the new character's lips with the translated audio.
- **Automatic Orchestration**: A unified CLI tool that handles the entire multi-step process and merges the results.

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

### Option 2: CLI Tool
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
- **Temporal Consistency**: Handled via ControlNet structure guidance.
- **VRAM Management**: Uses `enable_model_cpu_offload()` to run on consumer hardware.
- **Zero-Shot Transfer**: Visuals use IP-Adapter Plus; Audio uses F5-TTS for high-fidelity zero-shot voice cloning.
