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
import subprocess
from logger_utils import logger

# Ensure ffmpeg/ffprobe are in PATH
static_ffmpeg.add_paths()

# Monkeypatch torchaudio.load to avoid torchcodec issues
def soundfile_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
    actual_frames = num_frames if num_frames > 0 else -1
    audio, sr = sf.read(filepath, start=frame_offset, frames=actual_frames, dtype='float32')
    audio = torch.from_numpy(audio)
    if channels_first:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
    return audio, sr

torchaudio.load = soundfile_load

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path_prefix, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Audio Pipeline on {self.device}")
        
        # 1. Initialize ASR
        logger.info(f"Loading ASR model from {asr_model_path}...")
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        
        # 2. Translation settings (Lazy loading)
        self.model_prefix = translation_model_path_prefix
        self.tokenizer = None
        self.translation_model = None
        self.current_pair = None
        
        # ... existing F5-TTS and Kokoro init ...

    def _load_translation_model(self, source_lang, target_lang):
        pair = f"{source_lang}-{target_lang}"
        if self.current_pair == pair:
            return
            
        model_id = f"Helsinki-NLP/opus-mt-{pair}"
        local_path = f"{self.model_prefix}{pair}"
        
        # Download if doesn't exist locally
        if not os.path.exists(local_path):
            logger.info(f"Local translation model for {pair} not found. Downloading...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=local_path)
            
        logger.info(f"Loading translation model for {pair}...")
        self.tokenizer = MarianTokenizer.from_pretrained(local_path)
        self.translation_model = MarianMTModel.from_pretrained(local_path).to(self.device)
        self.current_pair = pair

    def transcribe_audio(self, audio_path, detect_language=False):
        segments, info = self.asr_model.transcribe(audio_path, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        if detect_language:
            return text, info.language
        return text

    def process_video(self, video_path, output_audio_path, ref_audio_path=None, target_lang="es", preserve_bg=True):
        # 1. Extract audio
        # ... same as before ...
        
        # 3. Transcribe with Language Detection
        logger.info("Transcribing vocal track and detecting language...")
        source_text, detected_lang = self.transcribe_audio(vocal_track, detect_language=True)
        logger.info(f"Detected Source Language: {detected_lang}")
        
        if not source_text or len(source_text.strip()) < 2:
            source_text = "Hello."
        logger.info(f"Source Text: {source_text}")
        
        # 4. Dynamic Translation Model Loading
        if detected_lang != target_lang:
            try:
                self._load_translation_model(detected_lang, target_lang)
                logger.info(f"Translating from {detected_lang} to {target_lang}...")
                inputs = self.tokenizer(source_text, return_tensors="pt").to(self.device)
                translated_tokens = self.translation_model.generate(**inputs, max_length=128)
                translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Translation model loading failed for {detected_lang}-{target_lang}: {e}. Falling back to default or skipping translation.")
                translated_text = source_text
        else:
            logger.info("Source and target languages match. Skipping translation.")
            translated_text = source_text
        
        # ... deduplication and synthesis ...
        
        # 5. Synthesis
        temp_translated_vocals = "temp_translated_vocals.wav"
        if ref_audio_path and os.path.exists(ref_audio_path):
            logger.info("Synthesizing zero-shot cloning with F5-TTS...")
            ref_text = self.transcribe_audio(ref_audio_path)
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=translated_text)
            sf.write(temp_translated_vocals, wav, sr)
        elif self.kokoro:
            logger.info("Synthesizing with Kokoro presets...")
            samples, sample_rate = self.kokoro.create(translated_text, voice="af_sarah", speed=1.0, lang="en-us") 
            sf.write(temp_translated_vocals, samples, sample_rate)
        
        # 6. Final Remix
        if background_track and os.path.exists(background_track):
            logger.info("Remixing translated vocals with original background...")
            vocal_clip = mp.AudioFileClip(temp_translated_vocals)
            bg_clip = mp.AudioFileClip(background_track)
            final_audio = mp.CompositeAudioClip([bg_clip, vocal_clip])
            final_audio.write_audiofile(output_audio_path, fps=24000, logger=None)
            vocal_clip.close()
            bg_clip.close()
        else:
            if os.path.exists(temp_translated_vocals):
                os.rename(temp_translated_vocals, output_audio_path)

        # Cleanup
        for f in [temp_source_audio, temp_translated_vocals]:
            if os.path.exists(f): os.remove(f)
            
        logger.info(f"Successfully generated audio: {output_audio_path}")
        return output_audio_path

if __name__ == "__main__":
    pass
