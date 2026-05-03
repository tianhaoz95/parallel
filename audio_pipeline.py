import os
import torch
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from kokoro_onnx import Kokoro
import moviepy as mp

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path, tts_model_path, tts_voices_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize ASR
        print(f"Loading ASR model from {asr_model_path}...")
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        
        # Initialize Translation
        print(f"Loading translation model from {translation_model_path}...")
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model_path)
        self.translation_model = MarianMTModel.from_pretrained(translation_model_path).to(self.device)
        
        # Initialize TTS (Kokoro)
        print(f"Loading TTS model from {tts_model_path}...")
        self.kokoro = Kokoro(tts_model_path, tts_voices_path)

    def process_video(self, video_path, output_audio_path, target_lang="es"):
        # 1. Extract audio from video
        video = mp.VideoFileClip(video_path)
        temp_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_audio, logger=None)
        
        # 2. Transcribe
        print("Transcribing...")
        segments, _ = self.asr_model.transcribe(temp_audio, beam_size=5)
        full_text = " ".join([s.text for s in segments])
        print(f"Source Text: {full_text}")
        
        # 3. Translate
        print(f"Translating to {target_lang}...")
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        translated_tokens = self.translation_model.generate(**inputs)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        print(f"Translated Text: {translated_text}")
        
        # 4. TTS (Voice Style Transfer)
        print("Synthesizing speech...")
        # Use a high-quality preset voice
        samples, sample_rate = self.kokoro.create(translated_text, voice="af_sarah", speed=1.0, lang="en-us") 
        
        sf.write(output_audio_path, samples, sample_rate)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        print(f"Generated translated audio: {output_audio_path}")
        return output_audio_path

if __name__ == "__main__":
    # Example usage
    # pipeline = AudioPipeline(...)
    pass
