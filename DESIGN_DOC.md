# Design Document: Local Video Character Replacement & Audio Translation System

## 1. Introduction
This project delivers a local, end-to-end pipeline for transforming video content. The two primary objectives are:
1.  **Visual Character Replacement**: Replacing characters in a video with a person from a reference image while maintaining motion and environmental consistency.
2.  **Audio Translation & Style Transfer**: Translating the soundtrack into a target language while adopting the vocal characteristics (timbre, style) of a reference audio recording.

## 2. System Architecture
The system consists of three modular pipelines orchestrated by a central Python script or Gradio Web UI.

### 2.1 Visual Pipeline (Character Replacement)
The visual pipeline replaces characters frame-by-frame while preserving structure and motion.

#### Implemented Models:
*   **Stable Diffusion 1.5**: The core generative engine.
*   **ControlNet Canny**: Extracts the structural edges from the original video to ensure the replacement character follows the exact same motion and pose.
*   **IP-Adapter Plus (ViT-H/14)**: Extracts semantic facial and clothing features from the reference image and injects them into the diffusion process for accurate character replacement.

#### Workflow:
1.  **Extraction**: Video frames are extracted and processed individually.
2.  **Structural Mapping**: ControlNet generates edge maps from the original frame.
3.  **Semantic Injection**: IP-Adapter extracts character identity from the reference image.
4.  **Denoising**: Stable Diffusion generates the new frame guided by both the edge map (motion) and the character features (identity).

### 2.2 Audio Pipeline (ASR, Translation & Zero-Shot Cloning)
The audio pipeline handles the semantic and vocal transformation of the soundtrack.

#### Implemented Models:
*   **Faster-Whisper (small)**: High-speed local Automatic Speech Recognition.
*   **MarianMT (Opus-MT)**: Transformer-based local translation (English to Spanish/French/German/etc.).
*   **F5-TTS**: A state-of-the-art Flow-Matching Diffusion model for **true zero-shot voice cloning**. It clones the reference voice from a short 5-15s recording.

#### Workflow:
1.  **Transcription**: Whisper converts source audio to text.
2.  **Translation**: MarianMT translates the text into the target language.
3.  **Style Extraction**: F5-TTS analyzes the reference recording to capture vocal timbre and cadence.
4.  **Synthesis**: F5-TTS generates the translated text in the cloned voice.

### 2.3 Synchronization Pipeline (Lip-Sync)
Ensures the visuals and transformed audio are perfectly aligned.

#### Implemented Models:
*   **Wav2Lip 256 (ONNX)**: High-resolution lip-syncing model.
*   **MediaPipe**: Lightweight, fast face detection used to track the character's mouth area for lip-syncing.

## 3. Tech Stack & Integration
*   **Orchestration**: Python 3.12 with custom modules.
*   **Video Processing**: MoviePy and OpenCV.
*   **Audio Processing**: SoundFile and Pydub.
*   **Portability**: Bundled `static-ffmpeg` to provide `ffmpeg`/`ffprobe` binaries without system-wide dependencies.
*   **Interface**: Gradio for the Web UI and `argparse` for the CLI.

## 4. Hardware Requirements
Verified on **NVIDIA GB10 (Blackwell)** hardware:
*   **GPU**: NVIDIA RTX 3060 (12GB VRAM) minimum. 24GB recommended for high-resolution processing.
*   **CPU**: Modern multi-core processor for ASR and orchestration.
*   **RAM**: 16GB+ (32GB preferred).

## 5. Implementation Summary
The project was implemented in 5 phases, moving from basic audio/visual prototypes to a unified, synchronized system. The final deliverable includes both a CLI tool and a Gradio Web UI for maximum accessibility.

## 6. Challenges & Mitigations
*   **Environmental Conflicts**: Resolved issues with `torchcodec` and `moviepy` versions by implementing custom loaders and pinning dependencies.
*   **Temporal Consistency**: Mitigated flickering by using ControlNet structure guidance at every frame.
*   **Hallucinations**: Improved translation robustness with deduplication logic for the MarianMT output.
