import os
import torch
import sys
from logger_utils import logger

def verify_gpu():
    """Verifies that a compatible NVIDIA GPU is available with sufficient VRAM."""
    if os.environ.get("USE_CPU") == "1":
        logger.warning("Running in CPU-Only Testing Mode. Quality will be drastically reduced and processing will be slow.")
        return True
        
    logger.info("Checking hardware compatibility...")
    
    if not torch.cuda.is_available():
        logger.error("No CUDA-compatible GPU detected. This system requires an NVIDIA GPU.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
    
    logger.info(f"Detected GPU: {device_name}")
    logger.info(f"Total VRAM: {total_vram:.2f} GB")
    
    if total_vram < 10:
        logger.warning("VRAM is below 12GB. The pipeline may fail during Character Replacement or Lipsync.")
        logger.warning("Consider using a lower resolution or enabling more aggressive CPU offloading in config.yaml.")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA Version: {cuda_version}")
    
    return True

if __name__ == "__main__":
    if verify_gpu():
        logger.info("✅ Hardware check passed.")
        sys.exit(0)
    else:
        logger.error("❌ Hardware check failed.")
        sys.exit(1)
