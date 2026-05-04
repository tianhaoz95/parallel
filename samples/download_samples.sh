#!/bin/bash

# Local Video & Audio Transformer - Sample Asset Downloader
echo "--- 📥 Downloading Sample Assets ---"

mkdir -p samples

# 1. Sample Input Video (Walking person)
echo "[1/3] Downloading sample input video..."
curl -L https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4 -o samples/input_video.mp4

# 2. Reference Character Image (Portrait)
echo "[2/3] Downloading reference character image..."
curl -L https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg -o samples/ref_image.jpg

# 3. Reference Voice Audio (Speech sample)
echo "[3/3] Downloading reference voice audio..."
# Using a clean speech sample from a reliable source
curl -L https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac -o samples/ref_audio.flac

echo ""
echo "--- ✅ Samples Ready! ---"
echo "You can now run: make run-ui"
