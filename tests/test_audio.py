import pytest
import os
from audio_pipeline import AudioPipeline

def test_audio_pipeline_init(mocker):
    # Mock models to avoid heavy loading during unit tests
    mocker.patch('audio_pipeline.WhisperModel')
    mocker.patch('transformers.MarianTokenizer.from_pretrained')
    mocker.patch('transformers.MarianMTModel.from_pretrained')
    mocker.patch('audio_pipeline.F5TTS')
    mocker.patch('audio_pipeline.Kokoro')
    
    # Test initialization
    pipeline = AudioPipeline(
        asr_model_path="dummy_asr",
        translation_model_path="dummy_trans",
        tts_model_path="dummy_tts",
        tts_voices_path="dummy_voices"
    )
    
    assert pipeline is not None
    assert pipeline.device in ["cuda", "cpu"]

def test_transcribe_audio_call(mocker):
    # Mock Whisper
    mock_whisper = mocker.patch('audio_pipeline.WhisperModel')
    mock_instance = mock_whisper.return_value
    
    # Mock segments return
    class MockSegment:
        def __init__(self, text):
            self.text = text
            
    mock_instance.transcribe.return_value = ([MockSegment("Hello"), MockSegment("World")], None)
    
    mocker.patch('transformers.MarianTokenizer.from_pretrained')
    mocker.patch('transformers.MarianMTModel.from_pretrained')
    mocker.patch('audio_pipeline.F5TTS')
    
    pipeline = AudioPipeline("dummy", "dummy")
    result = pipeline.transcribe_audio("dummy.wav")
    
    assert result == "Hello World"
    mock_instance.transcribe.assert_called_once()
