#!/bin/bash

# Local Video & Audio Transformer - Setup Script
echo "--- 🛠 Setting up Local Video & Audio Transformer ---"

# 1. Check Python version
python3 --version | grep -E "3.12|3.13" > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3.12+ is required."
    exit 1
fi

# 2. Create Virtual Environment
echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install requirements
echo "[2/4] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download Models
echo "[3/4] Downloading ML models..."
python3 download_models.py

# 5. Verify Installation
echo "[4/4] Verifying installation..."
python3 -c "import torch; import diffusers; import gradio; print('✅ Core libraries verified')"

echo ""
echo "--- 🎉 Setup Complete! ---"
echo "To start the Web UI, run:"
echo "source .venv/bin/activate && python3 app.py"
echo ""
echo "To run a test transformation, run:"
echo "source .venv/bin/activate && python3 demo.py"
