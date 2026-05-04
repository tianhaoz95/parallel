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
        
        logger.info("Loading F5-TTS...")
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

    def synthesize_segment(self, text, ref_audio_path, ref_text, output_path):
        """Synthesizes a single segment of speech."""
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
            except Exception as e: logger.warning(f"Source separation failed: {e}")
        
        logger.info("Transcribing and detecting language...")
        _, detected_lang, segments = self.transcribe_audio(vocal_track, detect_language=True)
        
        # Get reference text once if needed
        ref_text = ""
        if ref_audio_path:
            logger.info("Transcribing reference audio...")
            ref_text, _ = self.transcribe_audio(ref_audio_path)

        if detected_lang != target_lang:
            self._load_translation_model(detected_lang, target_lang)
            logger.info(f"Translating {len(segments)} segments...")
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

        # Precise Segment-based Synthesis
        logger.info("Synthesizing audio segments precisely...")
        audio_clips = []
        os.makedirs("temp_segments", exist_ok=True)
        
        for i, s in enumerate(segments):
            seg_path = f"temp_segments/seg_{i}.wav"
            self.synthesize_segment(s['translated_text'], ref_audio_path, ref_text, seg_path)
            
            clip = mp.AudioFileClip(seg_path).set_start(s['start'])
            # Adjust duration to fit the original segment or slightly expand/compress
            # For now, we keep the synthesized length but capped at the original end if needed
            # or just let it play. Professional tools often stretch/compress.
            audio_clips.append(clip)

        # Assemble new vocals
        new_vocal_audio = mp.CompositeAudioClip(audio_clips)
        
        # Final Mix
        if background_track:
            logger.info("Remixing with background and ducking...")
            bg_audio = mp.AudioFileClip(background_track)
            
            # Simple Ducking: Lower BG volume to 20% when vocals play
            # We can use fl_filter or just volume effects. 
            # For simplicity, we lower BG volume globally or use the composite.
            # Real ducking requires volume envelopes, but 30% BG is usually safe.
            bg_audio = bg_audio.volumex(0.3) 
            final_audio = mp.CompositeAudioClip([bg_audio, new_vocal_audio.volumex(1.2)])
        else:
            final_audio = new_vocal_audio

        final_audio.write_audiofile(output_audio_path, fps=24000, logger=None)
        
        # Cleanup
        video.close()
        for c in audio_clips: c.close()
        if background_track: bg_audio.close()
        
        import shutil
        if os.path.exists("temp_segments"): shutil.rmtree("temp_segments")
        if os.path.exists(temp_source_audio): os.remove(temp_source_audio)
        
        logger.info(f"Audio processing complete: {output_audio_path}")
        return output_audio_path

if __name__ == "__main__":
    pass
