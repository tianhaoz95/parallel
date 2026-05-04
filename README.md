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

## 🧪 Testing & Engineering
- **Unit Tests**: `make test`
- **Benchmarking**: `make benchmark`
- **Containerization**: `docker-compose up --build`

---

## 📄 License
MIT License. Individual models (Stable Diffusion, F5-TTS, etc.) are subject to their respective licenses.
