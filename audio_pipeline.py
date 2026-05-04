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
import pysubs2

# Ensure ffmpeg/ffprobe are in PATH
static_ffmpeg.add_paths()

# Monkeypatch torchaudio.load
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
        
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        self.model_prefix = translation_model_path_prefix
        self.tokenizer = None
        self.translation_model = None
        self.current_pair = None
        
        logger.info("Loading F5-TTS...")
        self.f5tts = F5TTS(device=self.device)
        
        if tts_model_path and tts_voices_path:
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else:
            self.kokoro = None

    def _load_translation_model(self, source_lang, target_lang):
        pair = f"{source_lang}-{target_lang}"
        if self.current_pair == pair:
            return
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
        if detect_language:
            return full_text, info.language, text_segments
        return full_text, text_segments

    def separate_audio(self, audio_path, output_dir="separated"):
        logger.info("Separating sources with Demucs...")
        cmd = f"python3 -m demucs.separate --two-stems=vocals -n htdemucs --out {output_dir} {audio_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(output_dir, "htdemucs", base_name, "vocals.wav"), os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

    def generate_srt(self, segments, output_srt):
        subs = pysubs2.SSAFile()
        for i, s in enumerate(segments):
            event = pysubs2.SSAEvent(start=int(s['start'] * 1000), end=int(s['end'] * 1000), text=s['text'])
            subs.append(event)
        subs.save(output_srt)
        return output_srt

    def process_video(self, video_path, output_audio_path, ref_audio_path=None, target_lang="es", preserve_bg=True, output_srt=None):
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        vocal_track = temp_source_audio
        background_track = None
        if preserve_bg:
            try: vocal_track, background_track = self.separate_audio(temp_source_audio)
            except: pass
        
        logger.info("Transcribing...")
        full_text, detected_lang, segments = self.transcribe_audio(vocal_track, detect_language=True)
        
        # Translate segments for SRT
        translated_segments = []
        if detected_lang != target_lang:
            self._load_translation_model(detected_lang, target_lang)
            logger.info("Translating...")
            for s in segments:
                inputs = self.tokenizer(s['text'], return_tensors="pt").to(self.device)
                tokens = self.translation_model.generate(**inputs)
                trans_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                translated_segments.append({'start': s['start'], 'end': s['end'], 'text': trans_text})
            translated_text = " ".join([s['text'] for s in translated_segments])
        else:
            translated_text = full_text
            translated_segments = segments

        if output_srt:
            self.generate_srt(translated_segments, output_srt)

        # Synthesis
        temp_vocals = "temp_translated_vocals.wav"
        if ref_audio_path:
            ref_text, _ = self.transcribe_audio(ref_audio_path)
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=translated_text)
            sf.write(temp_vocals, wav, sr)
        elif self.kokoro:
            samples, sr = self.kokoro.create(translated_text, voice="af_sarah", speed=1.0, lang="en-us")
            sf.write(temp_vocals, samples, sr)
        
        if background_track:
            v_clip, b_clip = mp.AudioFileClip(temp_vocals), mp.AudioFileClip(background_track)
            mp.CompositeAudioClip([b_clip, v_clip]).write_audiofile(output_audio_path, fps=24000, logger=None)
            v_clip.close(); b_clip.close()
        else:
            if os.path.exists(temp_vocals): os.rename(temp_vocals, output_audio_path)

        for f in [temp_source_audio, temp_vocals]:
            if os.path.exists(f): os.remove(f)
        return output_audio_path

if __name__ == "__main__":
    pass
