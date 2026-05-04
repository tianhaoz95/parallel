import yaml
import logging
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config):
    logging_cfg = config.get('logging', {})
    level_str = logging_cfg.get('level', 'INFO')
    level = getattr(logging, level_str.upper())
    log_format = logging_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(level=level, format=log_format)
    return logging.getLogger("transformer")

# Shared logger instance
CONFIG = load_config()
logger = setup_logging(CONFIG)
