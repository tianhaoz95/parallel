# Project Roadmap: Remaining Features & Technical Improvements

This report identifies the missing components required to elevate the Local Video & Audio Transformation suite to a commercial-grade production tool.

## 🚀 High Priority (Visual Realism)

### 1. Actual LivePortrait Integration
- **Status**: Placeholder Wrapper (`expression_pipeline.py`)
- **Task**: Replace the placeholder with the full LivePortrait inference engine. 
- **Requirement**: Implement the appearance and motion feature extraction to transfer blinking, gaze, and head pose from the source video to the replacement character.
- **Benefit**: Replaced characters will feel "alive" by mimicking the subtle micro-expressions of the original actor.

### 2. Character-Specific Visual Prompting
- **Status**: Global Prompt implemented.
- **Task**: Allow unique visual description overrides for each character in the `identity_map`.
- **Benefit**: Users can specify that "Character A" should be in a suit while "Character B" should be in armor, even in the same scene.

## 🎙 High Priority (Audio Quality)

### 3. Consolidated Reference Audio
- **Status**: Single Reference only.
- **Task**: Implement "Multi-Audio Embedding Averaging" (similar to visual identity consolidation).
- **Benefit**: Users can provide several clips of a voice to create a more robust "Master Voice Profile" that captures the full emotional and tonal range of the target speaker.

### 4. Granular Phoneme-to-Lip Sync (DTW)
- **Status**: Segment-based timing.
- **Task**: Integrate **Dynamic Time Warping (DTW)** or phoneme alignment.
- **Benefit**: Perfectly aligns specific sounds to specific mouth shapes on a frame-by-frame basis, eliminating "dead air" at the ends of translated segments.

## 🛠 Workflow & UX

### 5. Functional Project "Load" Interface
- **Status**: Backend serialization ready; UI is list-only.
- **Task**: Add a "Load" button to the Projects tab that restores the full state (mappings, settings, files) into the Transformation tab.
- **Benefit**: Enables fast switching between different transformation setups without manual re-entry.

### 6. Streaming UI Progress
- **Status**: Broad steps implemented.
- **Task**: Refactor `main.py` to accept a callback for granular progress updates (e.g., "Processing Frame 140/500").
- **Benefit**: Provides better user feedback for very long-running video tasks.

## 📦 Optimization

### 7. TensorRT / OpenVINO Acceleration
- **Status**: PyTorch/CUDA standard.
- **Task**: Implement optimized inference engines for the diffusion and voice models.
- **Benefit**: Potential 2x-4x speedup on NVIDIA and Intel hardware beyond current LCM levels.
