import os
import argparse
import moviepy as mp
from audio_pipeline import AudioPipeline
from visual_pipeline import VisualPipeline

def main():
    parser = argparse.ArgumentParser(description="Video Character Replacement & Audio Translation")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--ref_image", required=True, help="Path to reference character image")
    parser.add_argument("--ref_audio", help="Path to reference audio for style transfer (optional)")
    parser.add_argument("--target_lang", default="es", help="Target language (default: es)")
    parser.add_argument("--prompt", default="a person", help="Prompt describing the replacement character")
    parser.add_argument("--output", default="final_output.mp4", help="Path to output video")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)

    # 1. Audio Processing
    print("\n--- Step 1: Processing Audio (Transcription, Translation, Synthesis) ---")
    audio_pipe = AudioPipeline(
        asr_model_path="models/faster-whisper-small",
        translation_model_path=f"models/opus-mt-en-{args.target_lang}",
        tts_model_path="models/Kokoro-82M/kokoro-v1.0.onnx",
        tts_voices_path="models/Kokoro-82M/voices.bin"
    )
    
    translated_audio = "temp_translated_audio.wav"
    audio_pipe.process_video(args.video, translated_audio, target_lang=args.target_lang)

    # 2. Visual Processing
    print("\n--- Step 2: Processing Video (Character Replacement) ---")
    visual_pipe = VisualPipeline(
        sd_model_path="models/stable-diffusion-v1-5-pretrained",
        controlnet_path="models/sd-controlnet-canny"
    )
    
    transformed_video = "temp_transformed_no_audio.mp4"
    visual_pipe.process_video(args.video, args.ref_image, transformed_video, prompt=args.prompt)

    # 3. Final Merge
    print("\n--- Step 3: Merging Video and Audio ---")
    video_clip = mp.VideoFileClip(transformed_video)
    audio_clip = mp.AudioFileClip(translated_audio)
    
    # Ensure durations match (trim to shortest)
    final_duration = min(video_clip.duration, audio_clip.duration)
    video_clip = video_clip.with_duration(final_duration)
    audio_clip = audio_clip.with_duration(final_duration)
    
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(args.output, codec="libx264", audio_codec="aac")
    
    # Cleanup temporary files
    video_clip.close()
    audio_clip.close()
    if os.path.exists(translated_audio): os.remove(translated_audio)
    if os.path.exists(transformed_video): os.remove(transformed_video)
    
    print(f"\nSUCCESS: Generated final output at {args.output}")

if __name__ == "__main__":
    main()
