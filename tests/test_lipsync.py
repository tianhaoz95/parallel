import pytest
import numpy as np
import os
from lipsync_pipeline import LipsyncPipeline

def test_lipsync_pipeline_init(mocker):
    mocker.patch('onnxruntime.InferenceSession')
    mocker.patch('mediapipe.solutions.face_detection.FaceDetection')
    
    pipeline = LipsyncPipeline(model_path="dummy.onnx")
    assert pipeline is not None
    assert pipeline.input_size == 256

def test_get_face_crop(mocker):
    # Skip init
    pipeline = LipsyncPipeline.__new__(LipsyncPipeline)
    
    # Mock face detector
    mock_detector = mocker.Mock()
    pipeline.face_detector = mock_detector
    
    # Mock detection result
    mock_results = mocker.Mock()
    mock_results.detections = [mocker.Mock()]
    mock_results.detections[0].location_data.relative_bounding_box.xmin = 0.1
    mock_results.detections[0].location_data.relative_bounding_box.ymin = 0.1
    mock_results.detections[0].location_data.relative_bounding_box.width = 0.5
    mock_results.detections[0].location_data.relative_bounding_box.height = 0.5
    
    mock_detector.process.return_value = mock_results
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    coords = pipeline.get_face_crop(dummy_frame)
    
    assert coords is not None
    assert len(coords) == 4
    # Check that coords are within frame boundaries
    assert all(0 <= c <= 100 for c in coords)
