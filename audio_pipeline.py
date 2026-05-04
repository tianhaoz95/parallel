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

class AudioPipeline:
    def __init__(self, asr_model_path, translation_model_path_prefix, tts_model_path=None, tts_voices_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="float32")
        self.model_prefix = translation_model_path_prefix
        self.tokenizer = None
        self.translation_model = None
        self.current_pair = None
        self.f5tts = F5TTS(device=self.device)
        self.speaker_identifier = SpeakerIdentification()
        if tts_model_path and tts_voices_path:
            self.kokoro = Kokoro(tts_model_path, tts_voices_path)
        else: self.kokoro = None

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
        cmd = f"python3 -m demucs.separate --two-stems=vocals -n htdemucs --out {output_dir} {audio_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(output_dir, "htdemucs", base_name, "vocals.wav"), os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

    def synthesize_segment(self, text, ref_audio_path, ref_text, output_path):
        if ref_audio_path:
            wav, sr, _ = self.f5tts.infer(ref_file=ref_audio_path, ref_text=ref_text, gen_text=text)
            sf.write(output_path, wav, sr)
        elif self.kokoro:
            samples, sr = self.kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
            sf.write(output_path, samples, sr)
        return output_path

    def map_audio_to_speakers(self, audio_segments, visual_activity):
        mapped_segments = []
        for s in audio_segments:
            start, end = s['start'], s['end']
            max_activity = -1; best_speaker = 0
            for speaker_id, activity in visual_activity.items():
                segment_activity = [val for time, val in activity if start <= time <= end]
                if segment_activity:
                    avg = sum(segment_activity) / len(segment_activity)
                    if avg > max_activity:
                        max_activity = avg; best_speaker = speaker_id
            s['speaker_id'] = best_speaker
            mapped_segments.append(s)
        return mapped_segments

    def process_video(self, video_path, output_audio_path, ref_audio_paths=None, target_lang="es", preserve_bg=True, output_srt=None):
        logger.info(f"Processing multi-speaker audio for {video_path}")
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
        segments = self.map_audio_to_speakers(segments, visual_activity)
        
        logger.info("Synthesizing segments with speaker mapping...")
        audio_clips = []
        os.makedirs("temp_segments", exist_ok=True)
        for i, s in enumerate(segments):
            current_ref = ref_audio_paths.get(f"Character_{s['speaker_id']}") if isinstance(ref_audio_paths, dict) else None
            ref_text = ""
            if current_ref: ref_text, _ = self.transcribe_audio(current_ref)
            
            seg_path = f"temp_segments/seg_{i}.wav"
            self.synthesize_segment(s['text'], current_ref, ref_text, seg_path)
            audio_clips.append(mp.AudioFileClip(seg_path).set_start(s['start']))

        final_audio = mp.CompositeAudioClip(audio_clips)
        if background_track:
            bg_audio = mp.AudioFileClip(background_track).volumex(0.3)
            final_audio = mp.CompositeAudioClip([bg_audio, final_audio])
            
        final_audio.write_audiofile(output_audio_path, fps=24000, logger=None)
        video.close()
        for c in audio_clips: c.close()
        # Return segments for orchestration
        return output_audio_path, segments

if __name__ == "__main__":
    pass
