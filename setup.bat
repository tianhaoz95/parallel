@echo off
setlocal

echo --- 🛠 Setting up Local Video ^& Audio Transformer (Windows) ---

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

:: 3. Upgrade pip and install requirements
echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Download Models
echo [3/4] Downloading ML models...
python download_models.py

:: 5. Verify
echo [4/4] Verifying installation...
python -c "import torch; print('✅ PyTorch version: ' + torch.__version__)"

echo.
echo --- 🎉 Setup Complete! ---
echo To start the Web UI, run:
echo call .venv\Scripts\activate.bat ^&^& python app.py
echo.
pause
