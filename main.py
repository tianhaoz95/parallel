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
from adaptive_loader import HardwareAdaptiveLoader
import subprocess

def generate_comparison(input_video, transformed_video, output_path):
    logger.info("Generating side-by-side comparison video...")
    clip1 = mp.VideoFileClip(input_video).margin(10)
    clip2 = mp.VideoFileClip(transformed_video).margin(10)
    final = mp.clips_array([[clip1, clip2]])
    final.write_videofile(output_path, codec="libx264")
    clip1.close(); clip2.close()

def normalize_audio(video_path, output_path):
    """Normalizes audio to EBU R128 standard."""
    logger.info(f"Normalizing audio loudness for {video_path}...")
    cmd = f"ffmpeg-normalize {video_path} -o {output_path} -c:a aac -b:a 192k --force"
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def burn_subtitles(video_path, srt_path, output_path):
    """Hardcodes subtitles into the video."""
    logger.info(f"Burning subtitles from {srt_path} into {video_path}...")
    # Escape path for ffmpeg filter
    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = f'ffmpeg -y -i "{video_path}" -vf "subtitles={escaped_srt}" -c:a copy "{output_path}"'
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    # 0. Hardware-Adaptive Optimization
    _ = HardwareAdaptiveLoader.apply_optimizations()
    if not verify_gpu(): return

    parser = argparse.ArgumentParser(description="Video Character Replacement & Audio Translation")
    parser.add_argument("--video", required=True)
    parser.add_argument("--identity_map")
    parser.add_argument("--ref_image", nargs="+")
    parser.add_argument("--ref_audio")
    parser.add_argument("--target_lang", default="es")
    parser.add_argument("--prompt", default="a portrait of a character")
    parser.add_argument("--output", default="final_output.mp4")
    parser.add_argument("--skip_lipsync", action="store_true")
    parser.add_argument("--expression", action="store_true")
    parser.add_argument("--preserve_bg", action="store_true", default=True)
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--subtitles", action="store_true", help="Generate and burn subtitles")
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize audio loudness")
    parser.add_argument("--comparison", action="store_true")
    
    args = parser.parse_args()
    
    identity_data = {}
    if args.identity_map:
        if os.path.exists(args.identity_map):
            with open(args.identity_map, 'r') as f: identity_data = json.load(f)
        else: identity_data = json.loads(args.identity_map)
    elif args.ref_image:
        identity_data = {"Character_0": {"images": args.ref_image, "audio": args.ref_audio}}

    # 1. Audio
    logger.info("--- Phase 1: Audio ---")
    audio_pipe = AudioPipeline(
        asr_model_path=CONFIG.get('models', {}).get('asr'),
        translation_model_path_prefix=CONFIG.get('models', {}).get('translation_prefix'),
        tts_model_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_onnx'),
        tts_voices_path=CONFIG.get('models', {}).get('tts', {}).get('kokoro_voices')
    )
    ref_audios = {name: data.get('audio') for name, data in identity_data.items()}
    translated_audio = "temp_translated_audio.wav"
    output_srt = "temp_subtitles.srt" if args.subtitles else None
    _, segments = audio_pipe.process_video(args.video, translated_audio, ref_audio_paths=ref_audios, target_lang=args.target_lang, preserve_bg=args.preserve_bg, output_srt=output_srt)

    # 2. Visual
    logger.info("--- Phase 2: Visual ---")
    visual_pipe = VisualPipeline()
    transformed_video = "temp_transformed_no_audio.mp4"
    visual_map = {name: data.get('images', []) for name, data in identity_data.items()}
    visual_pipe.process_video_multi(args.video, visual_map, transformed_video, prompt=args.prompt, restore_face=False, upscale=args.upscale)

    # 3. Refinement
    current_video = transformed_video
    if args.expression:
        temp_exp = "temp_expression.mp4"
        FacialExpressionPipeline().retarget_expressions(current_video, list(visual_map.values())[0][0], temp_exp)
        current_video = temp_exp

    if not args.skip_lipsync:
        logger.info("--- Phase 4: Lipsync ---")
        for s in segments:
            char_name = f"Character_{s['speaker_id']}"
            if char_name in visual_map and visual_map[char_name]: s['target_embedding'] = visual_map[char_name][0]
        temp_synced = "temp_synced.mp4"
        LipsyncPipeline().process_video_segmented(current_video, translated_audio, temp_synced, segments)
        current_video = temp_synced

    # 4. HD Restoration
    logger.info("--- Phase 5: Final HD Restoration ---")
    clip = mp.VideoFileClip(current_video)
    final_raw = "temp_restored.mp4"
    clip.fl_image(lambda frame: visual_pipe.restore_faces(frame)).write_videofile(final_raw, codec="libx264", audio=True)
    clip.close()

    # 5. Professional Mastering (Subtitles & Loudness)
    logger.info("--- Phase 6: Professional Mastering ---")
    mastered_video = final_raw
    
    if args.subtitles and os.path.exists("temp_subtitles.srt"):
        burn_result = "temp_burned.mp4"
        burn_subtitles(mastered_video, "temp_subtitles.srt", burn_result)
        mastered_video = burn_result

    if args.normalize:
        norm_result = "temp_normalized.mp4"
        normalize_audio(mastered_video, norm_result)
        mastered_video = norm_result

    if os.path.exists(args.output): os.remove(args.output)
    os.rename(mastered_video, args.output)

    if args.comparison: generate_comparison(args.video, args.output, "comparison_view.mp4")
    
    # Cleanup
    for f in [translated_audio, transformed_video, "temp_expression.mp4", "temp_synced.mp4", "temp_restored.mp4", "temp_burned.mp4", "temp_normalized.mp4", "temp_subtitles.srt"]:
        if os.path.exists(f): os.remove(f)
    logger.info(f"SUCCESS: Mastered result saved at {args.output}")

if __name__ == "__main__":
    main()
