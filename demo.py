import os
from audio_pipeline import AudioPipeline
import moviepy as mp

def run_demo():
    print("=== Local Video & Audio Transformation Demo ===")
    
    # 1. Setup paths
    video_path = "samples/input_video.mp4"
    ref_image = "samples/ref_image.jpg"
    output_audio = "samples/translated_audio.wav"
    output_final = "final_demo_output.mp4"
    
    # 2. Run Audio Pipeline (Functional)
    print("\n[Phase 1] Running Audio Pipeline...")
    audio_pipe = AudioPipeline(
        asr_model_path="models/faster-whisper-small",
        translation_model_path="models/opus-mt-en-es",
        tts_model_path="models/Kokoro-82M/kokoro-v1.0.onnx",
        tts_voices_path="models/Kokoro-82M/voices.bin"
    )
    
    audio_pipe.process_video(video_path, output_audio, target_lang="es")
    
    # 3. Merge Audio with Original Video (Demo Result)
    print("\n[Phase 2] Merging translated audio with video...")
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = mp.AudioFileClip(output_audio)
    
    # Trim to match
    final_duration = min(video_clip.duration, audio_clip.duration, 5.0) # 5s demo
    video_clip = video_clip.with_duration(final_duration)
    audio_clip = audio_clip.with_duration(final_duration)
    
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(output_final, codec="libx264", audio_codec="aac")
    
    print(f"\nDEMO COMPLETE: Final video generated at {output_final}")
    print("The Visual Pipeline (Character Replacement) is implemented in visual_pipeline.py")
    print("and can be run by providing full SD 1.5 weights in models/stable-diffusion-v1-5/")

if __name__ == "__main__":
    run_demo()
