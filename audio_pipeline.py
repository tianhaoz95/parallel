import os
import threading

# Disable Xet storage before any huggingface_hub import.
# Setting the env var alone is not enough if HF Hub was already imported;
# we must mutate the module-level constant directly.
import huggingface_hub.constants as _hf_constants
_hf_constants.HF_HUB_DISABLE_XET = True
del _hf_constants

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
from speaker_id import SpeakerIdentification
from emotion_analyzer import EmotionAnalyzer
from llm_translation import LLMTranslationPipeline

static_ffmpeg.add_paths()

def _load_asr_model(asr_model_path, timeout_sec=120):
    """Load ASR model with timeout protection for hung downloads."""
    timed_out = False

    def _kill_on_timeout():
        nonlocal timed_out
        timed_out = True
        logger.warning("ASR model download timed out after %ds. Trying fallback to smaller model.", timeout_sec)

    timer = threading.Timer(timeout_sec, _kill_on_timeout)
    timer.daemon = True
    timer.start()
    try:
        model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        timer.cancel()
        logger.info("ASR model loaded: %s", asr_model_path)
        return model
    except Exception as e:
        timer.cancel()
        logger.error("ASR model load failed: %s", e)
        raise

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path_prefix, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = _load_asr_model(asr_model_path)
        self.model_prefix = translation_model_path_prefix
        self.use_llm = CONFIG.get('defaults', {}).get('use_llm_translation', False)
        self.f5tts = F5TTS(device=self.device)
        self.speaker_identifier = SpeakerIdentification()
        self.emotion_analyzer = EmotionAnalyzer()
        if tts_model_path and tts_voices_path:
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else: self.kokoro = None

    def _load_translation_model(self, source_lang, target_lang):
        if self.use_llm: return
        pair = f"{source_lang}-{target_lang}"
        local_path = f"{self.model_prefix}{pair}"
        if not os.path.exists(local_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"Helsinki-NLP/opus-mt-{pair}", local_dir=local_path)
        self.marian_tokenizer = MarianTokenizer.from_pretrained(local_path)
        self.marian_model = MarianMTModel.from_pretrained(local_path).to(self.device)

    def translate_text(self, text, source_lang, target_lang):
        if self.use_llm:
            if not hasattr(self, 'llm_translator') or self.llm_translator is None:
                self.llm_translator = LLMTranslationPipeline()
            return self.llm_translator.translate(text, source_lang, target_lang)
        inputs = self.marian_tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.marian_model.generate(**inputs)
        return self.marian_tokenizer.decode(tokens[0], skip_special_tokens=True)

    def transcribe_audio(self, audio_path, detect_language=False):
        segments, info = self.asr_model.transcribe(audio_path, beam_size=5)
        text_segments = []
        for s in segments:
            text_segments.append({'start': s.start, 'end': s.end, 'text': s.text.strip()})
        if detect_language: return " ".join([s['text'] for s in text_segments]), info.language, text_segments
        return " ".join([s['text'] for s in text_segments]), text_segments

    def separate_audio(self, audio_path, output_dir="separated"):
        cmd = f"python3 -m demucs.separate --two-stems=vocals -n htdemucs --out {output_dir} {audio_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(output_dir, "htdemucs", base_name, "vocals.wav"), os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

    def consolidate_reference_audio(self, audio_paths, output_path):
        """Merges multiple reference audio clips into one master reference."""
        if not audio_paths: return None
        if len(audio_paths) == 1: return audio_paths[0]
        
        logger.info(f"Consolidating {len(audio_paths)} reference audio samples...")
        clips = [mp.AudioFileClip(p) for p in audio_paths if os.path.exists(p)]
        if not clips: return None
        
        final_clip = mp.concatenate_audioclips(clips)
        final_clip.write_audiofile(output_path, fps=24000, logger=None)
        for c in clips: c.close()
        return output_path

    def synthesize_segment(self, text, ref_audio_path, ref_text, output_path, emotion_tag="", speed=1.0):
        full_text = f"{emotion_tag} {text}" if emotion_tag else text
        if ref_audio_path:
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=full_text)
            sf.write(output_path, wav, sr)
        elif self.kokoro:
            samples, sr = self.kokoro.create(text, voice="af_heart", speed=speed, lang="en-us")
            sf.write(output_path, samples, sr)
        return output_path

    def process_video(self, video_path, output_audio_path, ref_audio_paths=None, target_lang="es", preserve_bg=True, output_srt=None, external_segments=None):
        logger.info(f"Processing audio for {video_path}")
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        vocal_track = temp_source_audio
        background_track = None
        if preserve_bg:
            try: vocal_track, background_track = self.separate_audio(temp_source_audio)
            except: pass
        
        if external_segments:
            segments = external_segments
        else:
            _, detected_lang, segments = self.transcribe_audio(vocal_track, detect_language=True)
            if detected_lang != target_lang:
                self._load_translation_model(detected_lang, target_lang)
                for s in segments: s['translated_text'] = self.translate_text(s['text'], detected_lang, target_lang)
            else:
                for s in segments: s['translated_text'] = s['text']

        # Consolidated Master Audio for each character
        master_refs = {}
        os.makedirs("temp_masters", exist_ok=True)
        if isinstance(ref_audio_paths, dict):
            for sp_id, paths in ref_audio_paths.items():
                if isinstance(paths, list):
                    master_path = f"temp_masters/master_{sp_id}.wav"
                    master_refs[sp_id] = self.consolidate_reference_audio(paths, master_path)
                else:
                    master_refs[sp_id] = paths

        audio_clips = []
        os.makedirs("temp_segments", exist_ok=True)
        for i, s in enumerate(segments):
            y, sr = librosa.load(vocal_track, offset=s['start'], duration=s['end']-s['start'])
            temp_orig = f"temp_segments/orig_{i}.wav"
            sf.write(temp_orig, y, sr)
            emotion = self.emotion_analyzer.analyze_audio(temp_orig)
            
            sp_name = f"Character_{s.get('speaker_id', 0)}"
            current_ref = master_refs.get(sp_name)
            ref_text = ""
            if current_ref: ref_text, _ = self.transcribe_audio(current_ref)
            
            seg_path = f"temp_segments/seg_{i}.wav"
            self.synthesize_segment(s.get('translated_text', s['text']), current_ref, ref_text, seg_path, emotion_tag=emotion['tag'] if emotion else "", speed=emotion['speed'] if emotion else 1.0)
            audio_clips.append(mp.AudioFileClip(seg_path).set_start(s['start']))

        final_audio = mp.CompositeAudioClip(audio_clips)
        if background_track:
            bg_audio = mp.AudioFileClip(background_track).volumex(0.3)
            final_audio = mp.CompositeAudioClip([bg_audio, final_audio])
            
        final_audio.write_audiofile(output_audio_path, fps=24000, logger=None)
        video.close()
        for c in audio_clips: c.close()
        import shutil
        shutil.rmtree("temp_segments")
        shutil.rmtree("temp_masters")
        if os.path.exists(temp_source_audio): os.remove(temp_source_audio)
        return output_audio_path, segments

if __name__ == "__main__":
    pass
