import os
import gradio as gr
from main import main as run_pipeline
import sys
from unittest.mock import patch
from logger_utils import logger

def transform_video(video, ref_image, ref_audio, target_lang, prompt, skip_lipsync, preserve_bg, progress=gr.Progress()):
    if video is None or ref_image is None:
        return None, "Error: Video and Reference Image are required."
    
    output_path = "transformed_output.mp4"
    
    # Prepare arguments for main.py's main function
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
        
    logger.info(f"UI initiating transformation with args: {args}")
    
    # Progress tracking logic (we'll use a mocked sys.stdout or just broad steps)
    progress(0, desc="🚀 Starting Pipeline...")
    
    # Map steps for the progress bar
    progress(0.1, desc="🎙 Processing Audio (ASR & Translation)...")
    
    # We use a trick to call main() with our args
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            # Note: For granular progress, we would need to refactor main.py 
            # to accept a callback. For now, we simulate the major phases.
            progress(0.3, desc="🖼 Character Replacement (Visual Pipeline)...")
            
            run_pipeline()
            
            if not skip_lipsync:
                progress(0.7, desc="👄 Synchronizing Lips (Lipsync Pass)...")
                progress(0.9, desc="✨ Finalizing & Restoring Faces...")
            
            if os.path.exists(output_path):
                progress(1.0, desc="✅ Success!")
                return output_path, "SUCCESS: Transformation complete!"
            else:
                return None, "Error: Transformation failed to generate output."
        except Exception as e:
            logger.error(f"UI Transformation Error: {str(e)}")
            return None, f"Error during transformation: {str(e)}"

# Define the UI
with gr.Blocks(title="Video & Audio Transformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video & Audio Transformer")
    gr.Markdown("Replace characters in video and translate audio with zero-shot voice cloning.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="1. Input Video")
            input_ref_image = gr.Image(label="2. Reference Character Image", type="filepath")
            input_ref_audio = gr.Audio(label="3. Reference Voice Audio (Optional)", type="filepath")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")
                with gr.Row():
                    target_lang = gr.Dropdown(
                        choices=["es", "fr", "de", "it", "zh"], 
                        value="es", 
                        label="Target Language"
                    )
                    prompt = gr.Textbox(
                        value="a portrait of a beautiful character", 
                        label="Visual Prompt"
                    )
            
            with gr.Row():
                skip_lipsync = gr.Checkbox(label="Skip Lip-Sync (Faster)", value=False)
                preserve_bg = gr.Checkbox(label="Preserve BGM (Demucs)", value=True)
                
            run_btn = gr.Button("🚀 Start Transformation", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_video = gr.Video(label="✨ Transformed Output")
            status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("""
            ### 💡 Tips
            - **Visuals**: Use a clear, front-facing reference image.
            - **Audio**: Use 5-15s of clean speech for the best cloning result.
            - **Hardware**: This process is heavy and requires an NVIDIA GPU with 12GB+ VRAM.
            """)

    run_btn.click(
        fn=transform_video,
        inputs=[input_video, input_ref_image, input_ref_audio, target_lang, prompt, skip_lipsync, preserve_bg],
        outputs=[output_video, status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
