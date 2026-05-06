import pytest
import numpy as np
import os
from lipsync_pipeline import LipsyncPipeline

def test_lipsync_pipeline_init(mocker):
    mocker.patch('onnxruntime.InferenceSession')
    mocker.patch('deepface.DeepFace.extract_faces')
    
    pipeline = LipsyncPipeline(model_path="dummy.onnx")
    assert pipeline is not None
    assert pipeline.input_size == 256

def test_get_face_crop(mocker):
    # Skip init
    pipeline = LipsyncPipeline.__new__(LipsyncPipeline)
    
    mocker.patch('deepface.DeepFace.extract_faces', return_value=[{
        'confidence': 0.9,
        'facial_area': {'x': 10, 'y': 10, 'w': 50, 'h': 50}
    }])
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    coords = pipeline.get_all_face_crops(dummy_frame)
    
    assert coords is not None
    assert len(coords) == 1
    assert len(coords[0]['coords']) == 4
    # Check that coords are within frame boundaries
    assert all(0 <= c <= 100 for c in coords[0]['coords'])
