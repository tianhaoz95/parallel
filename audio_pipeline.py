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
from logger_utils import logger, CONFIG
import pysubs2
import librosa

# Ensure ffmpeg/ffprobe are in PATH
static_ffmpeg.add_paths()

# Monkeypatch torchaudio.load
def soundfile_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
    actual_frames = num_frames if num_frames > 0 else -1
    audio, sr = sf.read(filepath, start=frame_offset, frames=actual_frames, dtype='float32')
    audio = torch.from_numpy(audio)
    if channels_first:
        if audio.ndim == 1: audio = audio.unsqueeze(0)
        else: audio = audio.T
    return audio, sr

torchaudio.load = soundfile_load

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path_prefix, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Audio Pipeline on {self.device}")
        
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        self.model_prefix = translation_model_path_prefix
        self.tokenizer = None
        self.translation_model = None
        self.current_pair = None
        
        self.f5tts = F5TTS(device=self.device)
        
        if tts_model_path and tts_voices_path:
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else:
            self.kokoro = None

    def _load_translation_model(self, source_lang, target_lang):
        pair = f"{source_lang}-{target_lang}"
        if self.current_pair == pair: return
        local_path = f"{self.model_prefix}{pair}"
        if not os.path.exists(local_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"Helsinki-NLP/opus-mt-{pair}", local_dir=local_path)
        self.tokenizer = MarianTokenizer.from_pretrained(local_path)
        self.translation_model = MarianMTModel.from_pretrained(local_path).to(self.device)
        self.current_pair = pair

    def transcribe_audio(self, audio_path, detect_language=False):
        segments, info = self.asr_model.transcribe(audio_path, beam_size=5)
        text_segments = []
        for s in segments:
            text_segments.append({'start': s.start, 'end': s.end, 'text': s.text.strip()})
        full_text = " ".join([s['text'] for s in text_segments])
        if detect_language: return full_text, info.language, text_segments
        return full_text, text_segments

    def separate_audio(self, audio_path, output_dir="separated"):
        logger.info("Separating sources with Demucs...")
        cmd = f"python3 -m demucs.separate --two-stems=vocals -n htdemucs --out {output_dir} {audio_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(output_dir, "htdemucs", base_name, "vocals.wav"), os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

    def stretch_audio(self, input_path, target_duration, output_path):
        """Stretches or compresses audio to match target duration without changing pitch."""
        y, sr = librosa.load(input_path, sr=None)
        curr_duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate rate (ratio of original to target)
        # Avoid extreme stretching (max 1.5x)
        rate = curr_duration / target_duration
        rate = np.clip(rate, 0.7, 1.4)
        
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        sf.write(output_path, y_stretched, sr)
        return output_path

    def apply_ducking(self, bg_audio_path, vocal_clips, output_path):
        """Applies dynamic volume ducking to background music."""
        logger.info("Applying intelligent audio ducking...")
        bg_clip = mp.AudioFileClip(bg_audio_path)
        
        # Create a volume envelope for the background
        # We start with 1.0 (full volume)
        duration = bg_clip.duration
        fps = 10 # 10 samples per second for the envelope
        times = np.linspace(0, duration, int(duration * fps))
        envelope = np.ones_like(times)
        
        # Lower volume during vocal segments
        for v in vocal_clips:
            start, end = v.start, v.end
            # Ducking range with slight padding for smooth transition
            mask = (times >= start - 0.2) & (times <= end + 0.2)
            envelope[mask] = 0.25 # Lower to 25% volume
            
        # Apply smoothing to the envelope to avoid clicks
        envelope = np.convolve(envelope, np.ones(5)/5, mode='same')
        
        # Custom volume function for moviepy
        def volume_filter(t):
            idx = int(t * fps)
            if idx >= len(envelope): return envelope[-1]
            return envelope[idx]
            
        bg_ducked = bg_clip.fl(lambda gf, t: volume_filter(t) * gf(t))
        
        vocal_composite = mp.CompositeAudioClip(vocal_clips)
        final = mp.CompositeAudioClip([bg_ducked, vocal_composite])
        final.write_audiofile(output_path, fps=24000, logger=None)
        
        bg_clip.close()
        bg_ducked.close()
        return output_path

    def synthesize_segment(self, text, ref_audio_path, ref_text, output_path):
        if ref_audio_path:
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=text)
            sf.write(output_path, wav, sr)
        elif self.kokoro:
            samples, sr = self.kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
            sf.write(output_path, samples, sr)
        return output_path

    def process_video(self, video_path, output_audio_path, ref_audio_path=None, target_lang="es", preserve_bg=True, output_srt=None):
        logger.info(f"Processing audio for {video_path}")
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        vocal_track = temp_source_audio
        background_track = None
        if preserve_bg:
            try: vocal_track, background_track = self.separate_audio(temp_source_audio)
            except: pass
        
        _, detected_lang, segments = self.transcribe_audio(vocal_track, detect_language=True)
        
        ref_text = ""
        if ref_audio_path:
            ref_text, _ = self.transcribe_audio(ref_audio_path)

        if detected_lang != target_lang:
            self._load_translation_model(detected_lang, target_lang)
            for s in segments:
                inputs = self.tokenizer(s['text'], return_tensors="pt").to(self.device)
                tokens = self.translation_model.generate(**inputs)
                s['translated_text'] = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        else:
            for s in segments: s['translated_text'] = s['text']

        if output_srt:
            subs = pysubs2.SSAFile()
            for s in segments:
                subs.append(pysubs2.SSAEvent(start=int(s['start']*1000), end=int(s['end']*1000), text=s['translated_text']))
            subs.save(output_srt)

        logger.info("Synthesizing and time-stretching segments...")
        audio_clips = []
        os.makedirs("temp_segments", exist_ok=True)
        
        for i, s in enumerate(segments):
            orig_duration = s['end'] - s['start']
            raw_path = f"temp_segments/raw_{i}.wav"
            stretched_path = f"temp_segments/stretch_{i}.wav"
            
            self.synthesize_segment(s['translated_text'], ref_audio_path, ref_text, raw_path)
            # Stretch to match original timing
            self.stretch_audio(raw_path, orig_duration, stretched_path)
            
            clip = mp.AudioFileClip(stretched_path).set_start(s['start'])
            audio_clips.append(clip)

        if background_track:
            self.apply_ducking(background_track, audio_clips, output_audio_path)
        else:
            mp.CompositeAudioClip(audio_clips).write_audiofile(output_audio_path, fps=24000, logger=None)
        
        # Cleanup
        video.close()
        for c in audio_clips: c.close()
        import shutil
        if os.path.exists("temp_segments"): shutil.rmtree("temp_segments")
        if os.path.exists(temp_source_audio): os.remove(temp_source_audio)
        
        return output_audio_path

if __name__ == "__main__":
    pass
