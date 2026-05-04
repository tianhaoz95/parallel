import os
import gradio as gr
from main import main as run_pipeline
import sys
from unittest.mock import patch
from logger_utils import logger, CONFIG

def transform_video(video, ref_image, ref_audio, target_lang, prompt, skip_lipsync, preserve_bg, use_lcm, show_comparison, progress=gr.Progress()):
    if video is None or ref_image is None:
        return None, None, "Error: Video and Reference Image are required."
    
    output_path = "transformed_output.mp4"
    comparison_path = "comparison_view.mp4"
    
    # Prepare arguments
    args = [
        "--video", video,
        "--ref_image", ref_image,
        "--target_lang", target_lang,
        "--prompt", prompt,
        "--output", output_path
    ]
    
    if ref_audio:
        args.extend(["--ref_audio", ref_audio])
    if skip_lipsync:
        args.append("--skip_lipsync")
    if not preserve_bg:
        args.append("--no_preserve_bg")
    if show_comparison:
        args.append("--comparison")
        
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    logger.info(f"UI transformation. LCM: {use_lcm}, Comparison: {show_comparison}")
    progress(0, desc="🚀 Starting Pipeline...")
    
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🎙 Processing Audio...")
            run_pipeline()
            
            res_video = output_path if os.path.exists(output_path) else None
            comp_video = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            
            if res_video:
                progress(1.0, desc="✅ Success!")
                return res_video, comp_video, "SUCCESS: Transformation complete!"
            else:
                return None, None, "Error: Transformation failed to generate output."
        except Exception as e:
            logger.error(f"UI Error: {str(e)}")
            return None, None, f"Error: {str(e)}"

# Define the UI
with gr.Blocks(title="Video & Audio Transformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video & Audio Transformer")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="1. Input Video")
            input_ref_image = gr.Image(label="2. Reference Character Image", type="filepath")
            input_ref_audio = gr.Audio(label="3. Reference Voice Audio (Optional)", type="filepath")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                with gr.Row():
                    target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Target Language")
                    prompt = gr.Textbox(value="a portrait of a beautiful character", label="Visual Prompt")
            
            with gr.Row():
                use_lcm = gr.Checkbox(label="🚀 Fast Mode (LCM)", value=False)
                skip_lipsync = gr.Checkbox(label="Skip Lip-Sync", value=False)
                preserve_bg = gr.Checkbox(label="Preserve BGM", value=True)
                show_comparison = gr.Checkbox(label="🎥 Generate Comparison", value=True)
                
            run_btn = gr.Button("🚀 Start Transformation", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_video = gr.Video(label="✨ Transformed Result")
            comparison_video = gr.Video(label="🎞 Side-by-Side Comparison")
            status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("""
            ### 💡 Tips
            - **Fast Mode (LCM)**: Generates 8x faster but may have slightly less detail.
            - **Visuals**: Use a high-quality portrait for the best results.
            - **Audio**: Use clean speech for cloning.
            """)

    run_btn.click(
        fn=transform_video,
        inputs=[input_video, input_ref_image, input_ref_audio, target_lang, prompt, skip_lipsync, preserve_bg, use_lcm, show_comparison],
        outputs=[output_video, comparison_video, status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
