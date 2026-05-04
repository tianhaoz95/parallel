import os
import torch
import soundfile as sf
import numpy as np
import torchaudio
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from kokoro_onnx import Kokoro
from f5_tts.api import F5TTS
import moviepy.editor as mp
import static_ffmpeg

# Ensure ffmpeg/ffprobe are in PATH for pydub and others
static_ffmpeg.add_paths()

# Monkeypatch torchaudio.load to avoid torchcodec issues
def soundfile_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
    # soundfile uses 'frames' and 'start'
    # torchaudio uses 'frame_offset' and 'num_frames'
    actual_frames = num_frames if num_frames > 0 else -1
    audio, sr = sf.read(filepath, start=frame_offset, frames=actual_frames, dtype='float32')
    
    # sf.read returns (frames, channels)
    audio = torch.from_numpy(audio)
    if channels_first:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
    return audio, sr

torchaudio.load = soundfile_load

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 1. Initialize ASR (Whisper)
        print(f"Loading ASR model from {asr_model_path}...")
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        
        # 2. Initialize Translation (MarianMT)
        print(f"Loading translation model from {translation_model_path}...")
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model_path)
        self.translation_model = MarianMTModel.from_pretrained(translation_model_path).to(self.device)
        
        # 3. Initialize Zero-Shot TTS (F5-TTS)
        print("Loading F5-TTS for zero-shot voice cloning...")
        self.f5tts = F5TTS(device=self.device)
        
        # 4. Fallback TTS (Kokoro) if needed
        if tts_model_path and tts_voices_path:
            print(f"Loading Kokoro fallback from {tts_model_path}...")
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else:
            self.kokoro = None

    def transcribe_audio(self, audio_path):
        segments, _ = self.asr_model.transcribe(audio_path, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        return text

    def process_video(self, video_path, output_audio_path, ref_audio_path=None, target_lang="es"):
        # 1. Extract audio from video
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        # 2. Transcribe source
        print("Transcribing source video...")
        source_text = self.transcribe_audio(temp_source_audio)
        if not source_text or len(source_text.strip()) < 2:
            print("No significant speech detected in source video.")
            source_text = "Hello." # Default to avoid empty translation errors
        print(f"Source Text: {source_text}")
        
        # 3. Translate
        print(f"Translating to {target_lang}...")
        inputs = self.tokenizer(source_text, return_tensors="pt").to(self.device)
        translated_tokens = self.translation_model.generate(**inputs, max_length=128)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        # Simple deduplication for hallucinations
        words = translated_text.split()
        if len(words) > 5 and len(set(words)) == 1:
            translated_text = words[0] # Fix "Tú, tú, tú" issue
            
        print(f"Translated Text: {translated_text}")
        
        # 4. Zero-Shot TTS with F5-TTS
        if ref_audio_path and os.path.exists(ref_audio_path):
            print(f"Using reference audio for zero-shot cloning: {ref_audio_path}")
            print("Transcribing reference audio...")
            ref_text = self.transcribe_audio(ref_audio_path)
            print(f"Reference Text: {ref_text}")
            
            print("Synthesizing cloned speech with F5-TTS...")
            wav, sr, _ = self.f5tts.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=translated_text
            )
            
            sf.write(output_audio_path, wav, sr)
        elif self.kokoro:
            print("No reference audio provided, falling back to Kokoro presets...")
            samples, sample_rate = self.kokoro.create(translated_text, voice="af_sarah", speed=1.0, lang="en-us") 
            sf.write(output_audio_path, samples, sample_rate)
        else:
            raise ValueError("No reference audio provided and Kokoro fallback is not configured.")

        # Cleanup
        if os.path.exists(temp_source_audio):
            os.remove(temp_source_audio)
            
        print(f"Generated audio: {output_audio_path}")
        return output_audio_path

if __name__ == "__main__":
    pass
