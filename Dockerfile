# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install core ML dependencies
RUN python3.12 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3.12 -m pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# Default command: launch the Web UI
CMD ["python3.12", "app.py"]
