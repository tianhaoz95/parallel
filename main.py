import os
import argparse
import json
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
    
    # Advanced Identity Mapping (JSON string or path)
    # Format: {"Character1": {"images": ["img1.jpg", "img2.jpg"], "audio": "voice1.wav"}, ...}
    parser.add_argument("--identity_map", help="JSON string or path specifying multi-character mapping")
    
    # Fallback/Legacy args
    parser.add_argument("--ref_image", nargs="+", help="Reference character image(s) - Fallback for single char")
    parser.add_argument("--ref_audio", help="Reference audio for style transfer - Fallback for single char")
    
    parser.add_argument("--target_lang", default="es", help="Target language (default: es)")
    parser.add_argument("--prompt", default="a portrait of a beautiful character", help="Base prompt")
    parser.add_argument("--output", default="final_output.mp4", help="Path to output video")
    
    # Toggles
    parser.add_argument("--skip_lipsync", action="store_true", help="Skip lipsync step")
    parser.add_argument("--expression", action="store_true", help="Use LivePortrait")
    parser.add_argument("--preserve_bg", action="store_true", default=True)
    parser.add_argument("--no_smooth", action="store_false", dest="smooth")
    parser.add_argument("--no_mask", action="store_false", dest="use_mask")
    parser.add_argument("--no_erase", action="store_false", dest="erase_original")
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    
    args = parser.parse_args()
    
    # Parse Identity Map
    identity_data = {}
    if args.identity_map:
        if os.path.exists(args.identity_map):
            with open(args.identity_map, 'r') as f: identity_data = json.load(f)
        else:
            identity_data = json.loads(args.identity_map)
    elif args.ref_image:
        # Construct single character map from legacy args
        identity_data = {"DefaultCharacter": {"images": args.ref_image, "audio": args.ref_audio}}

    if not identity_data:
        logger.error("No identity mapping provided. Please use --identity_map or --ref_image.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else ".", exist_ok=True)

    # 1. Audio Pipeline (Multi-Speaker)
    logger.info("--- Phase 1: Audio Transformation (Multi-Speaker) ---")
    audio_pipe = AudioPipeline(
        asr_model_path=CONFIG.get('models', {}).get('asr'),
        translation_model_path_prefix=CONFIG.get('models', {}).get('translation_prefix'),
        tts_model_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_onnx'),
        tts_voices_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_voices')
    )
    
    # Map audio for all characters
    ref_audios = {name: data.get('audio') for name, data in identity_data.items()}
    translated_audio = "temp_translated_audio.wav"
    audio_pipe.process_video(args.video, translated_audio, ref_audio_paths=ref_audios, target_lang=args.target_lang, preserve_bg=args.preserve_bg)

    # 2. Visual Pipeline (Multi-Character)
    logger.info("--- Phase 2: Visual Transformation (Multi-Character) ---")
    visual_pipe = VisualPipeline()
    transformed_video = "temp_transformed_no_audio.mp4"
    
    # Extract only image paths for visual pipe
    visual_map = {name: data.get('images', []) for name, data in identity_data.items()}
    visual_pipe.process_video_multi(
        args.video, 
        visual_map, 
        transformed_video, 
        prompt=args.prompt, 
        restore_face=False, 
        upscale=args.upscale
    )

    # 3. Refinement Pass (Expressions & Sync)
    current_video = transformed_video
    
    if args.expression:
        logger.info("--- Phase 3: Expression Retargeting ---")
        temp_exp = "temp_expression.mp4"
        exp_pipe = FacialExpressionPipeline()
        # We retarget using the first identity's reference
        exp_pipe.retarget_expressions(current_video, list(visual_map.values())[0][0], temp_exp)
        current_video = temp_exp

    if not args.skip_lipsync:
        logger.info("--- Phase 4: High-Res Lipsync ---")
        temp_merged = "temp_merged.mp4"
        v_clip = mp.VideoFileClip(current_video); a_clip = mp.AudioFileClip(translated_audio)
        v_clip.set_duration(min(v_clip.duration, a_clip.duration)).set_audio(a_clip).write_videofile(temp_merged, codec="libx264", logger=None)
        v_clip.close(); a_clip.close()
        
        ls_pipe = LipsyncPipeline()
        temp_synced = "temp_synced.mp4"
        ls_pipe.process_video(temp_merged, translated_audio, temp_synced)
        current_video = temp_synced
        if os.path.exists(temp_merged): os.remove(temp_merged)

    # 4. Final Final Restoration
    logger.info("--- Phase 5: Final HD Restoration ---")
    clip = mp.VideoFileClip(current_video)
    final_clip = clip.fl_image(lambda frame: visual_pipe.restore_faces(frame))
    final_clip.write_videofile(args.output, codec="libx264", audio=True)
    clip.close()

    if args.comparison: generate_comparison(args.video, args.output, "comparison_view.mp4")
    
    # Cleanup
    for f in [translated_audio, transformed_video, "temp_expression.mp4", "temp_synced.mp4"]:
        if os.path.exists(f): os.remove(f)
    logger.info(f"SUCCESS: Result saved at {args.output}")

if __name__ == "__main__":
    main()
