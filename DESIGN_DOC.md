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
*   **ControlNet Canny**: Extracts structural edges to maintain motion and pose.
*   **IP-Adapter Plus (ViT-H/14)**: Extracts semantic character features from the reference image.
*   **GFPGAN v1.4**: Restores facial details in a post-sync pass for maximum clarity.
*   **LCM LoRA**: Provides an optional 8x speedup (Fast Mode) for video generation.

### 2.2 Audio Pipeline (ASR, Translation & Zero-Shot Cloning)
The audio pipeline handles the semantic and vocal transformation of the soundtrack.

#### Implemented Models:
*   **Faster-Whisper**: High-speed local ASR with **Automatic Language Detection**.
*   **MarianMT (Opus-MT)**: **Dynamic translation** engine that loads required models on the fly.
*   **F5-TTS**: State-of-the-art **Zero-Shot Voice Cloning** from a 10s reference recording.
*   **Demucs**: Audio source separation for **Background Music/SFX Preservation**.
*   **pysubs2**: Automatic generation of synchronized **SRT Subtitles**.

### 2.3 Synchronization Pipeline (Lip-Sync)
Ensures the visuals and transformed audio are perfectly aligned.

#### Implemented Models:
*   **Wav2Lip 256 (ONNX)**: High-resolution lip-syncing model.
*   **MediaPipe**: Fast face detection for mouth tracking.

## 3. Tech Stack & Integration
*   **Orchestration**: Python 3.12 with clean, decoupled modules.
*   **UI**: **Gradio Web UI** with **Real-time Progress Tracking**, **Creative Style Presets**, and a **Transformation History Gallery**.
*   **CLI**: Robust script for automated and batch processing.
*   **Automation**: setup scripts (Linux/Windows), **Makefile**, and **Docker**.

## 4. Hardware Requirements
Verified on **NVIDIA GB10 (Blackwell)** hardware:
*   **GPU**: NVIDIA RTX 3060 (12GB VRAM) minimum. 24GB recommended.
*   **VRAM Optimization**: Uses aggressive CPU offloading and FP16 precision.

## 5. Implementation Summary
The project was implemented in 5 phases, evolving from prototypes into a comprehensive, production-ready suite. All engineering standards, including unit testing (`pytest`) and CI/CD (GitHub Actions), are met.

## 6. Challenges & Mitigations
*   **Multilingual Support**: Mitigated via dynamic model loading based on Whisper's language detection.
*   **Visual Fidelity**: Optimized by running face restoration *after* the lip-syncing pass.
*   **User Experience**: Solved via a persistent history system and automated dependency management.
