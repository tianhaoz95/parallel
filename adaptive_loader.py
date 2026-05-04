import torch
from logger_utils import logger, CONFIG

def get_gpu_memory():
    """Returns the total and available VRAM in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # This is a bit complex for a simple call, but we'll use total as the proxy for class
    return total, total # simplified for now

def get_optimal_config():
    """Returns an optimized configuration based on detected hardware."""
    total_vram, _ = get_gpu_memory()
    logger.info(f"Detected {total_vram:.2f} GB of VRAM. Optimizing configuration...")
    
    # Defaults
    opt_cfg = {
        'whisper_model': 'small',
        'sd_model': 'models/stable-diffusion-v1-5-pretrained',
        'use_inpainting': False,
        'use_upscaling': False,
        'low_vram': True,
        'compute_type': 'float32'
    }
    
    if total_vram >= 23: # High-end (3090/4090/GB10)
        logger.info("Hardware Class: HIGH-END. Enabling all features.")
        opt_cfg.update({
            'whisper_model': 'large-v3',
            'use_inpainting': True,
            'use_upscaling': True,
            'low_vram': False,
            'compute_type': 'float16'
        })
    elif total_vram >= 11: # Mid-range (3060 12GB / 4070)
        logger.info("Hardware Class: MID-RANGE. Balanced settings.")
        opt_cfg.update({
            'whisper_model': 'medium',
            'use_inpainting': True,
            'use_upscaling': False,
            'low_vram': True,
            'compute_type': 'float16'
        })
    else: # Low-end
        logger.info("Hardware Class: ENTRY-LEVEL. Minimal settings.")
        opt_cfg.update({
            'whisper_model': 'tiny',
            'sd_model': 'models/tiny-sd', # Should be downloaded
            'low_vram': True,
            'compute_type': 'int8'
        })
        
    return opt_cfg

class HardwareAdaptiveLoader:
    @staticmethod
    def apply_optimizations():
        """Updates the global CONFIG with hardware-optimized values."""
        opt = get_optimal_config()
        
        # Update CONFIG paths and defaults
        CONFIG['models']['asr'] = f"Systran/faster-whisper-{opt['whisper_model']}"
        CONFIG['defaults']['preserve_bg'] = opt['use_inpainting']
        CONFIG['optimizations']['low_vram'] = opt['low_vram']
        
        # Update internal defaults for main.py
        return opt

if __name__ == "__main__":
    print(get_optimal_config())
