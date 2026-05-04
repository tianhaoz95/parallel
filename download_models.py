import os
import subprocess

def run_command(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_models():
    print("--- 📥 Local Video & Audio Transformer: Model Downloader ---")
    
    # 1. Base directory
    os.makedirs("models", exist_ok=True)

    # 2. Audio Models
    print("\n[1/5] Downloading Audio Models (ASR & Translation)...")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download Systran/faster-whisper-small --local-dir models/faster-whisper-small")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download Helsinki-NLP/opus-mt-en-es --local-dir models/opus-mt-en-es")
    
    # 3. TTS & Cloning Models
    print("\n[2/5] Downloading TTS & Voice Cloning Models (Kokoro & F5-TTS)...")
    os.makedirs("models/Kokoro-82M", exist_ok=True)
    if not os.path.exists("models/Kokoro-82M/kokoro-v1.0.onnx"):
        run_command("wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -O models/Kokoro-82M/kokoro-v1.0.onnx")
    if not os.path.exists("models/Kokoro-82M/voices.bin"):
        run_command("wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -O models/Kokoro-82M/voices.bin")
    # F5-TTS is handled via its internal downloader on first run, but we can pre-cache if needed.

    # 4. Visual Models (SD 1.5 & ControlNet)
    print("\n[3/5] Downloading Visual Models (Stable Diffusion & ControlNet)...")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download stable-diffusion-v1-5/stable-diffusion-v1-5 unet/diffusion_pytorch_model.bin vae/diffusion_pytorch_model.bin vae/config.json --local-dir models/stable-diffusion-v1-5-pretrained")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download lllyasviel/sd-controlnet-canny --local-dir models/sd-controlnet-canny")

    # 5. IP-Adapter & CLIP Encoder
    print("\n[4/5] Downloading Character Replacement Models (IP-Adapter Plus & CLIP H14)...")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download h94/IP-Adapter models/ip-adapter-plus_sd15.bin --local-dir models/IP-Adapter_plus")
    run_command("HF_HUB_ENABLE_HF_TRANSFER=1 hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K pytorch_model.bin config.json --local-dir models/image_encoder_H14")

    # 6. Face Restoration Model (GFPGAN)
    print("\n[5/6] Downloading Face Restoration Models (GFPGAN v1.4)...")
    if not os.path.exists("models/GFPGANv1.4.pth"):
        run_command("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O models/GFPGANv1.4.pth")

    # 7. Lipsync Model
    print("\n[6/6] Downloading Lip-Sync Models (Wav2Lip 256 ONNX)...")
    if not os.path.exists("models/wav2lip_256.onnx"):
        run_command("wget https://github.com/instant-high/wav2lip-onnx-256/releases/download/v1.0.0/wav2lip_256.onnx -O models/wav2lip_256.onnx")

    print("\n✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
    print("You can now run the transformation using 'python3 app.py' or 'python3 main.py'")

if __name__ == "__main__":
    download_models()
