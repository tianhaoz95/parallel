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
    def __init__(self, asr_model_path, translation_model_path, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Audio Pipeline on {self.device}")
        
        # 1. Initialize ASR
        logger.info(f"Loading ASR model from {asr_model_path}...")
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        
        # 2. Initialize Translation
        logger.info(f"Loading translation model from {translation_model_path}...")
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model_path)
        self.translation_model = MarianMTModel.from_pretrained(translation_model_path).to(self.device)
        
        # 3. Initialize Zero-Shot TTS
        logger.info("Loading F5-TTS for zero-shot voice cloning...")
        self.f5tts = F5TTS(device=self.device)
        
        # 4. Fallback TTS
        if tts_model_path and tts_voices_path:
            logger.info(f"Loading Kokoro fallback from {tts_model_path}...")
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else:
            self.kokoro = None

    def transcribe_audio(self, audio_path):
        segments, _ = self.asr_model.transcribe(audio_path, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        return text

    def separate_audio(self, audio_path, output_dir="separated"):
        logger.info("Separating audio sources with Demucs...")
        cmd = f"python3 -m demucs.separate --two-stems=vocals -n htdemucs --out {output_dir} {audio_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
        background_path = os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")
        return vocals_path, background_path

    def process_video(self, video_path, output_audio_path, ref_audio_path=None, target_lang="es", preserve_bg=True):
        # 1. Extract audio
        logger.info(f"Extracting audio from {video_path}")
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        vocal_track = temp_source_audio
        background_track = None
        
        # 2. Source Separation
        if preserve_bg:
            try:
                vocal_track, background_track = self.separate_audio(temp_source_audio)
            except Exception as e:
                logger.warning(f"Demucs separation failed: {e}. Proceeding without background preservation.")
        
        # 3. Transcribe
        logger.info("Transcribing vocal track...")
        source_text = self.transcribe_audio(vocal_track)
        if not source_text or len(source_text.strip()) < 2:
            source_text = "Hello."
        logger.info(f"Source Text: {source_text}")
        
        # 4. Translate
        logger.info(f"Translating to {target_lang}...")
        inputs = self.tokenizer(source_text, return_tensors="pt").to(self.device)
        translated_tokens = self.translation_model.generate(**inputs, max_length=128)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        # Deduplication fix
        words = translated_text.split()
        if len(words) > 5 and len(set(words)) == 1:
            translated_text = words[0]
        logger.info(f"Translated Text: {translated_text}")
        
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
