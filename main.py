import os
import argparse
import moviepy.editor as mp
from audio_pipeline import AudioPipeline
from visual_pipeline import VisualPipeline
from lipsync_pipeline import LipsyncPipeline
from logger_utils import logger, CONFIG
from check_gpu import verify_gpu

def generate_comparison(input_video, transformed_video, output_path):
    logger.info("Generating side-by-side comparison video...")
    clip1 = mp.VideoFileClip(input_video).margin(10)
    clip2 = mp.VideoFileClip(transformed_video).margin(10)
    
    # Text labels
    txt1 = mp.TextClip("Original", fontsize=50, color='white', font='Arial-Bold').set_duration(clip1.duration).set_position(('center', 'top'))
    txt2 = mp.TextClip("Transformed", fontsize=50, color='white', font='Arial-Bold').set_duration(clip2.duration).set_position(('center', 'top'))
    
    comp1 = mp.CompositeVideoClip([clip1, txt1])
    comp2 = mp.CompositeVideoClip([clip2, txt2])
    
    final = mp.clips_array([[comp1, comp2]])
    final.write_videofile(output_path, codec="libx264")
    clip1.close(); clip2.close()

def main():
    if not verify_gpu():
        logger.error("Incompatible hardware detected. Aborting.")
        return

    parser = argparse.ArgumentParser(description="Video Character Replacement & Audio Translation")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--ref_image", required=True, nargs="+", help="Path to reference character image(s)")
    parser.add_argument("--ref_audio", help="Path to reference audio for style transfer (optional)")
    parser.add_argument("--target_lang", default="es", help="Target language (default: es)")
    parser.add_argument("--prompt", default="a person", help="Prompt describing the replacement character")
    parser.add_argument("--output", default="final_output.mp4", help="Path to output video")
    parser.add_argument("--skip_lipsync", action="store_true", help="Skip the final lipsync step")
    parser.add_argument("--preserve_bg", action="store_true", default=True, help="Preserve background music/SFX using Demucs")
    parser.add_argument("--no_preserve_bg", action="store_false", dest="preserve_bg", help="Do not preserve background music")
    parser.add_argument("--subtitles", action="store_true", help="Generate and hardcode subtitles")
    parser.add_argument("--comparison", action="store_true", help="Generate a side-by-side comparison video")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)

    # 1. Audio Processing
    logger.info("Step 1: Processing Audio")
    audio_pipe = AudioPipeline(
        asr_model_path=CONFIG.get('models', {}).get('asr'),
        translation_model_path_prefix=CONFIG.get('models', {}).get('translation_prefix'),
        tts_model_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_onnx'),
        tts_voices_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_voices')
    )
    
    translated_audio = "temp_translated_audio.wav"
    output_srt = "subtitles.srt" if args.subtitles else None
    audio_pipe.process_video(args.video, translated_audio, ref_audio_path=args.ref_audio, target_lang=args.target_lang, preserve_bg=args.preserve_bg, output_srt=output_srt)

    # 2. Visual Processing
    logger.info("Step 2: Character Replacement")
    visual_pipe = VisualPipeline()
    transformed_video = "temp_transformed_no_audio.mp4"
    visual_pipe.process_video(args.video, args.ref_image, transformed_video, prompt=args.prompt, restore_face=False)

    # 3. Final Merge & Lipsync
    final_raw_video = args.output
    if not args.skip_lipsync:
        logger.info("Step 3: Processing Lipsync")
        temp_merged = "temp_merged_for_lipsync.mp4"
        v_clip, a_clip = mp.VideoFileClip(transformed_video), mp.AudioFileClip(translated_audio)
        dur = min(v_clip.duration, a_clip.duration)
        v_clip.set_duration(dur).set_audio(a_clip).write_videofile(temp_merged, codec="libx264", audio_codec="aac", logger=None)
        v_clip.close(); a_clip.close()
        
        ls_pipe = LipsyncPipeline()
        temp_synced = "temp_synced_no_restoration.mp4"
        ls_pipe.process_video(temp_merged, translated_audio, temp_synced)
        final_raw_video = temp_synced
        if os.path.exists(temp_merged): os.remove(temp_merged)
    else:
        v_clip, a_clip = mp.VideoFileClip(transformed_video), mp.AudioFileClip(translated_audio)
        dur = min(v_clip.duration, a_clip.duration)
        v_clip.set_duration(dur).set_audio(a_clip).write_videofile(args.output, codec="libx264", audio_codec="aac")
        v_clip.close(); a_clip.close()

    # 4. Final Face Restoration & Subtitles
    logger.info("Step 4: Final Polishing")
    clip = mp.VideoFileClip(final_raw_video)
    final_clip = clip.fl_image(lambda frame: visual_pipe.restore_faces(frame))
    
    if args.subtitles and os.path.exists("subtitles.srt"):
        # Placeholder for hardcoding subtitles if needed, for now we just provide the file
        logger.info("Subtitles generated at subtitles.srt")

    final_clip.write_videofile(args.output, codec="libx264", audio=True)
    clip.close()
    if os.path.exists(final_raw_video) and final_raw_video != args.output: os.remove(final_raw_video)

    # 5. Optional Comparison
    if args.comparison:
        generate_comparison(args.video, args.output, "comparison_view.mp4")
    
    # Cleanup
    for f in [translated_audio, transformed_video]:
        if os.path.exists(f): os.remove(f)
    logger.info(f"SUCCESS: Result saved at {args.output}")

if __name__ == "__main__":
    main()
