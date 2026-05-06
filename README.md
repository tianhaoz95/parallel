# 🎬 Local Video Character Studio

A professional-grade, local, end-to-end pipeline for transforming video content using state-of-the-art AI. Replace characters in videos with reference identities and translate audio with zero-shot voice cloning.

## 🌟 Key Features
- **Identity-Aware Replacement**: Replace specific individuals in multi-person scenes using **IP-Adapter Plus** and **DeepFace**.
- **Skeleton-Guided Motion**: Uses **DWPose** to perfectly transfer actor motion to the replacement character while isolating the background.
- **Zero-Shot Voice Cloning**: Clone any voice from a 10s sample using **F5-TTS**, with **automatic emotion transfer**.
- **Multilingual Support**: **Automatic language detection** and dynamic translation via **Llama 3** or **MarianMT**.
- **Studio Quality**: Automatic **GFPGAN** face restoration, **Real-ESRGAN** HD upscaling, and **Demucs** background music preservation.
- **Precision Sync**: High-resolution lip-synchronization via **Wav2Lip 256 ONNX**.
- **Professional Workflow**: Interactive script editor, persistent identity library, and project serialization.

---

## 🛠 Prerequisites
- **Hardware**: NVIDIA GPU (RTX 30xx/40xx/Blackwell) with **12GB+ VRAM** (24GB recommended).
- **OS**: Linux (Ubuntu 22.04+) or Windows 10/11.
- **Software**: Python 3.12, CUDA 12.1+.

---

## ⚙️ Environment Setup

### 1. System Dependencies
Before installing Python packages, ensure your system has the necessary drivers and utilities:

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y git ffmpeg wget python3.12-venv
```

**Windows:**
- Install [Git for Windows](https://git-scm.com/download/win).
- Install [Python 3.12](https://www.python.org/downloads/).
- Install [FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH.
- Install [NVIDIA CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive).

### 2. Manual Installation
We recommend using a virtual environment to manage dependencies.

**Step-by-Step:**
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd parallel
   ```
2. **Initialize Environment:**
   - **Linux:** Run `./setup.sh`
   - **Windows:** Run `setup.bat`
   
   *This will automatically create a `.venv`, install all dependencies from `requirements.txt`, and download the required ML models (~15GB).*

3. **Manual Dependency Install (Alternative):**
   If you prefer to run steps manually:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python download_models.py
   ```

### 3. Docker Installation (Containerized)
If you prefer to avoid local dependency hell, use Docker:
```bash
# Ensure NVIDIA Container Toolkit is installed
docker-compose up --build
```

### 4. Verification
Check if your GPU is correctly detected by the studio:
```bash
python check_gpu.py
```

---

## 🚀 Quick Start (Linux/WSL)

1. **Clone & Setup**:
   ```bash
   git clone <repo-url>
   cd parallel
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Download Sample Assets**:
   ```bash
   ./samples/download_samples.sh
   ```

3. **Launch the Studio (Web UI)**:
   ```bash
   source .venv/bin/activate
   python3 app.py
   ```
   Open `http://localhost:7860` in your browser.

### 🌐 Remote Access
To access the UI from another machine on your local network:
1. Find the server's IP address (run `hostname -I` on Linux or `ipconfig` on Windows).
2. On your other machine, navigate to `http://<server-ip>:7860`.
3. The server is pre-configured to listen on `0.0.0.0`.

---

## 🚀 Quick Start (Windows)

1. **Setup**: Double-click `setup.bat` to install dependencies and models.
2. **Launch**:
   ```cmd
   call .venv\Scripts\activate.bat
   python app.py
   ```

---

## 📖 How to Use the Studio

### 1. Identity Discovery (Multi-Character Workflow)
- Upload your video in the **Identity Discovery** tab.
- Click **"Discover Characters"**. The system will scan the video and show thumbnails of Every unique person found.
- (Optional) Name the character and click **"Save to Library"** to reuse this identity in future videos.

### 2. Character Mapping
- For Every discovered person, upload the **Replacement Images** and a **Replacement Voice** sample.
- If you recognize a character from your library, they will be auto-mapped!

### 3. Production & Rendering
- Go to the **Production** tab.
- Choose your **Target Language** and **Visual Style Preset** (e.g., Cinematic, Animated).
- Toggle **Fast Mode (LCM)** if you want an 8x speed boost.
- Click **"Start Production"**. Monitor real-time progress via the status bar.

---

## 💻 CLI Usage (Automation)
For batch processing, use `main.py`:
```bash
python3 main.py --video input.mp4 \
                --identity_map project_config.json \
                --target_lang es \
                --output final_dub.mp4 \
                --upscale --subtitles
```

---

## 📂 Project Structure
- `app.py`: Browser-based studio interface.
- `main.py`: Core pipeline orchestration.
- `identity_library/`: Persistent character profiles.
- `projects/`: Saved production setups (.avt files).
- `models/`: Local storage for 10+ ML model weights.

---

## 🧪 Testing & Benchmarking

### 1. Hardware Verification
Run the verification script to ensure your GPU and VRAM are sufficient:
```bash
python check_gpu.py
```

### 2. Performance Benchmark
Measure the processing speed of your system (Audio, Visual, and Lipsync components):
```bash
make benchmark
```

### 3. CPU-Only Testing Mode
If you don't have an NVIDIA GPU, you can run in CPU mode for logic testing:
```bash
export USE_CPU=1  # Windows: set USE_CPU=1
python app.py
```

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA Errors** | Ensure NVIDIA drivers are installed and `nvcc --version` shows 12.1+. |
| **Out of Memory (OOM)** | Close other GPU apps. Set `fast_mode: true` in `config.yaml`. |
| **Model Download Fail** | Ensure `huggingface-cli` is installed: `pip install huggingface_hub[cli]`. |
| **FFmpeg Not Found** | Ensure FFmpeg is in your PATH. Run `ffmpeg -version` to verify. |

---

## 📄 License
MIT License. Individual models (Stable Diffusion, F5-TTS, etc.) are subject to their respective licenses.
