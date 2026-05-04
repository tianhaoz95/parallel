import os
import argparse
import moviepy.editor as mp
from audio_pipeline import AudioPipeline
from visual_pipeline import VisualPipeline
from lipsync_pipeline import LipsyncPipeline
from expression_pipeline import FacialExpressionPipeline
from logger_utils import logger, CONFIG
from check_gpu import verify_gpu

def generate_comparison(input_video, transformed_video, output_path):
    logger.info("Generating side-by-side comparison video...")
    clip1 = mp.VideoFileClip(input_video).margin(10)
    clip2 = mp.VideoFileClip(transformed_video).margin(10)
    final = mp.clips_array([[clip1, clip2]])
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
    parser.add_argument("--expression", action="store_true", help="Use LivePortrait for expression retargeting")
    parser.add_argument("--preserve_bg", action="store_true", default=True, help="Preserve background music")
    parser.add_argument("--no_preserve_bg", action="store_false", dest="preserve_bg")
    parser.add_argument("--no_smooth", action="store_false", dest="smooth")
    parser.add_argument("--no_mask", action="store_false", dest="use_mask")
    parser.add_argument("--upscale", action="store_true", help="Upscale final video to HD")
    parser.add_argument("--subtitles", action="store_true", help="Generate subtitles")
    parser.add_argument("--comparison", action="store_true", help="Generate comparison video")
    
    args = parser.parse_args()
    if args.smooth is None: args.smooth = True
    if args.use_mask is None: args.use_mask = True
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)

    # 1. Audio Processing
    logger.info("--- Step 1: Processing Audio ---")
    audio_pipe = AudioPipeline(
        asr_model_path=CONFIG.get('models', {}).get('asr'),
        translation_model_path_prefix=CONFIG.get('models', {}).get('translation_prefix'),
        tts_model_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_onnx'),
        tts_voices_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_voices')
    )
    translated_audio = "temp_translated_audio.wav"
    output_srt = "subtitles.srt" if args.subtitles else None
    audio_pipe.process_video(args.video, translated_audio, ref_audio_path=args.ref_audio, target_lang=args.target_lang, preserve_bg=args.preserve_bg, output_srt=output_srt)

    # 2. Visual Processing (Character Replacement)
    logger.info("--- Step 2: Character Replacement ---")
    visual_pipe = VisualPipeline()
    transformed_video = "temp_transformed_no_audio.mp4"
    visual_pipe.process_video(args.video, args.ref_image, transformed_video, prompt=args.prompt, restore_face=False, smooth=args.smooth, use_mask=args.use_mask, upscale=args.upscale)

    # 3. Refinement Pass (Expressions)
    refined_video = transformed_video
    if args.expression:
        logger.info("--- Step 3: Expression Retargeting (LivePortrait) ---")
        temp_expression = "temp_expression_retargeted.mp4"
        exp_pipe = FacialExpressionPipeline()
        exp_pipe.retarget_expressions(transformed_video, args.ref_image[0], temp_expression)
        refined_video = temp_expression

    # 4. Final Sync
    final_raw_video = args.output
    if not args.skip_lipsync:
        logger.info("--- Step 4: Processing Lipsync ---")
        temp_merged = "temp_merged_for_lipsync.mp4"
        v_clip, a_clip = mp.VideoFileClip(refined_video), mp.AudioFileClip(translated_audio)
        dur = min(v_clip.duration, a_clip.duration)
        v_clip.set_duration(dur).set_audio(a_clip).write_videofile(temp_merged, codec="libx264", audio_codec="aac", logger=None)
        v_clip.close(); a_clip.close()
        
        ls_pipe = LipsyncPipeline()
        temp_synced = "temp_synced_no_restoration.mp4"
        ls_pipe.process_video(temp_merged, translated_audio, temp_synced)
        final_raw_video = temp_synced
        if os.path.exists(temp_merged): os.remove(temp_merged)
    else:
        v_clip, a_clip = mp.VideoFileClip(refined_video), mp.AudioFileClip(translated_audio)
        dur = min(v_clip.duration, a_clip.duration)
        v_clip.set_duration(dur).set_audio(a_clip).write_videofile(args.output, codec="libx264", audio_codec="aac")
        v_clip.close(); a_clip.close()

    # 5. Final Restoration
    logger.info("--- Step 5: Final Face Restoration ---")
    clip = mp.VideoFileClip(final_raw_video)
    final_clip = clip.fl_image(lambda frame: visual_pipe.restore_faces(frame))
    final_clip.write_videofile(args.output, codec="libx264", audio=True)
    clip.close()
    
    if os.path.exists(final_raw_video) and final_raw_video != args.output: os.remove(final_raw_video)
    if args.comparison: generate_comparison(args.video, args.output, "comparison_view.mp4")
    for f in [translated_audio, transformed_video, "temp_expression_retargeted.mp4"]:
        if os.path.exists(f): os.remove(f)
    logger.info(f"SUCCESS: Result saved at {args.output}")

if __name__ == "__main__":
    main()
