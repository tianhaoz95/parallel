import pytest
import numpy as np
from visual_pipeline import VisualPipeline

def test_visual_pipeline_init(mocker):
    # Mock all heavy dependencies
    mocker.patch('transformers.CLIPVisionModelWithProjection.from_pretrained')
    mocker.patch('diffusers.ControlNetModel.from_pretrained')
    mock_pipe_class = mocker.patch('diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained')
    
    # Mock the pipeline instance and its scheduler
    mock_pipe = mock_pipe_class.return_value
    mock_pipe.scheduler.config = {"beta_start": 0.00085} # Minimal config
    
    mocker.patch('diffusers.UniPCMultistepScheduler.from_config')
    mocker.patch('visual_pipeline.GFPGANer')
    
    pipeline = VisualPipeline(
        sd_model_path="dummy",
        controlnet_path="dummy",
        image_encoder_path="dummy",
        ip_adapter_path="dummy"
    )
    
    assert pipeline is not None
    assert hasattr(pipeline, 'restore_faces')

def test_get_canny_image():
    # Instantiate without calling __init__
    pipeline = VisualPipeline.__new__(VisualPipeline)
    
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    canny = pipeline.get_canny_image(dummy_img)
    
    # PIL Image check
    assert canny is not None
    assert canny.size == (100, 100)
