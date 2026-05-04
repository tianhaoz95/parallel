import os
import gradio as gr
from main import main as run_pipeline
import sys
from unittest.mock import patch

def transform_video(video, ref_image, ref_audio, target_lang, prompt, skip_lipsync):
    if video is None or ref_image is None:
        return "Error: Video and Reference Image are required."
    
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
        
    print(f"Running pipeline with args: {args}")
    
    # We use a trick to call main() with our args
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            run_pipeline()
            if os.path.exists(output_path):
                return output_path
            else:
                return "Error: Transformation failed to generate output."
        except Exception as e:
            return f"Error during transformation: {str(e)}"

# Define the UI
with gr.Blocks(title="Video & Audio Transformer") as demo:
    gr.Markdown("# 🎬 Local Video & Audio Transformer")
    gr.Markdown("Replace characters in video and translate audio with zero-shot voice cloning.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            input_ref_image = gr.Image(label="Reference Character Image", type="filepath")
            input_ref_audio = gr.Audio(label="Reference Voice Audio (Optional)", type="filepath")
            
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
            
            skip_lipsync = gr.Checkbox(label="Skip Lip-Sync (Faster)", value=False)
            run_btn = gr.Button("🚀 Start Transformation", variant="primary")
            
        with gr.Column():
            output_video = gr.Video(label="Transformed Output")
            status = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=transform_video,
        inputs=[input_video, input_ref_image, input_ref_audio, target_lang, prompt, skip_lipsync],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
