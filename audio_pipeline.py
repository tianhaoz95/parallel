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
from speaker_id import SpeakerIdentification
from emotion_analyzer import EmotionAnalyzer
from llm_translation import LLMTranslationPipeline

static_ffmpeg.add_paths()

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path_prefix, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        self.model_prefix = translation_model_path_prefix
        self.use_llm = CONFIG.get('defaults', {}).get('use_llm_translation', False)
        self.f5tts = F5TTS(device=self.device)
        self.speaker_identifier = SpeakerIdentification()
        self.emotion_analyzer = EmotionAnalyzer()
        if tts_model_path and tts_voices_path:
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else: self.kokoro = None

    def _load_translation_model(self, source_lang, target_lang):
        if self.use_llm: return # Handled by LLM class
        pair = f"{source_lang}-{target_lang}"
        local_path = f"{self.model_prefix}{pair}"
        if not os.path.exists(local_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"Helsinki-NLP/opus-mt-{pair}", local_dir=local_path)
        self.marian_tokenizer = MarianTokenizer.from_pretrained(local_path)
        self.marian_model = MarianMTModel.from_pretrained(local_path).to(self.device)

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

    def synthesize_segment(self, text, ref_audio_path, ref_text, output_path, emotion_tag="", speed=1.0):
        # F5-TTS supports emotion tags in the text
        full_text = f"{emotion_tag} {text}" if emotion_tag else text
        if ref_audio_path:
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=full_text)
            sf.write(output_path, wav, sr)
        elif self.kokoro:
            samples, sr = self.kokoro.create(text, voice="af_heart", speed=speed, lang="en-us")
            sf.write(output_path, samples, sr)
        return output_path

    def process_video(self, video_path, output_audio_path, ref_audio_paths=None, target_lang="es", preserve_bg=True, output_srt=None):
        logger.info(f"Processing audio with Emotion-Aware synthesis...")
        visual_activity = self.speaker_identifier.detect_speakers_in_video(video_path, duration=10.0)
        video = mp.VideoFileClip(video_path)
        temp_source_audio = "temp_source.wav"
        video.audio.write_audiofile(temp_source_audio, logger=None)
        
        vocal_track = temp_source_audio
        background_track = None
        if preserve_bg:
            try: vocal_track, background_track = self.separate_audio(temp_source_audio)
            except: pass
        
        _, detected_lang, segments = self.transcribe_audio(vocal_track, detect_language=True)
        
        audio_clips = []
        os.makedirs("temp_segments", exist_ok=True)
        
        for i, s in enumerate(segments):
            # 1. Map to speaker
            start, end = s['start'], s['end']
            max_act = -1; best_sp = 0
            for sp_id, act in visual_activity.items():
                seg_act = [v for t, v in act if start <= t <= end]
                if seg_act and sum(seg_act)/len(seg_act) > max_act:
                    max_act = sum(seg_act)/len(seg_act); best_sp = sp_id
            
            # 2. Extract emotion of original segment
            y, sr = librosa.load(vocal_track, offset=s['start'], duration=s['end']-s['start'])
            temp_seg_orig = f"temp_segments/orig_{i}.wav"
            sf.write(temp_seg_orig, y, sr)
            emotion = self.emotion_analyzer.analyze_audio(temp_seg_orig)
            
            # 3. Synthesis with emotion tag
            current_ref = ref_audio_paths.get(f"Character_{best_sp}") if isinstance(ref_audio_paths, dict) else None
            ref_text = ""
            if current_ref: ref_text, _ = self.transcribe_audio(current_ref)
            
            seg_path = f"temp_segments/seg_{i}.wav"
            self.synthesize_segment(s['text'], current_ref, ref_text, seg_path, emotion_tag=emotion['tag'] if emotion else "", speed=emotion['speed'] if emotion else 1.0)
            
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
        if os.path.exists(temp_source_audio): os.remove(temp_source_audio)
        return output_audio_path, segments

if __name__ == "__main__":
    pass
