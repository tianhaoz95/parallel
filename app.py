import os
import gradio as gr
from main import main as run_pipeline
import sys
import json
from datetime import datetime
from unittest.mock import patch
from logger_utils import logger, CONFIG

HISTORY_FILE = "transformation_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry) # Newest first
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[:50], f, indent=2)

def transform_video(video, ref_images, ref_audio, target_lang, prompt, style, skip_lipsync, preserve_bg, use_lcm, show_comparison, progress=gr.Progress()):
    if video is None or not ref_images:
        return None, None, "Error: Video and at least one Reference Image are required.", gr.update()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.mp4"
    comparison_path = f"comparison_{timestamp}.mp4"
    
    style_suffix = CONFIG.get('styles', {}).get(style, "")
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt
    
    # args expects a list of images for --ref_image
    args = [
        "--video", video,
        "--ref_image"
    ]
    args.extend(ref_images)
    
    args.extend([
        "--target_lang", target_lang,
        "--prompt", full_prompt,
        "--output", output_path
    ])
    
    if ref_audio: args.extend(["--ref_audio", ref_audio])
    if skip_lipsync: args.append("--skip_lipsync")
    if not preserve_bg: args.append("--no_preserve_bg")
    if show_comparison: args.append("--comparison")
        
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    logger.info(f"UI transformation. Images: {len(ref_images)}, Style: {style}")
    progress(0, desc="🚀 Starting Pipeline...")
    
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🎙 Processing Audio...")
            run_pipeline()
            
            res_video = output_path if os.path.exists(output_path) else None
            comp_video = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            
            if res_video:
                entry = {
                    "timestamp": timestamp,
                    "video": res_video,
                    "comparison": comp_video,
                    "prompt": prompt,
                    "style": style,
                    "lang": target_lang
                }
                save_history(entry)
                
                progress(1.0, desc="✅ Success!")
                return res_video, comp_video, f"SUCCESS: Generated {output_path}", get_history_html()
            else:
                return None, None, "Error: Transformation failed.", gr.update()
        except Exception as e:
            logger.error(f"UI Error: {str(e)}")
            return None, None, f"Error: {str(e)}", gr.update()

def get_history_html():
    history = load_history()
    if not history:
        return "<p style='text-align: center;'>No history yet.</p>"
    
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;'>"
    for entry in history:
        html += f"""
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'>
            <p><strong>{entry['timestamp']}</strong></p>
            <p style='font-size: 0.8em;'>Style: {entry['style']} | Lang: {entry['lang']}</p>
            <a href='file/{os.path.abspath(entry['video'])}' target='_blank' style='color: #2196F3;'>View Video</a>
        </div>
        """
    html += "</div>"
    return html

# Define the UI
with gr.Blocks(title="Video & Audio Transformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video & Audio Transformer")
    
    with gr.Tabs():
        with gr.Tab("Transform"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="1. Input Video")
                    # Changed to File with file_count="multiple" to allow plural images
                    input_ref_images = gr.File(label="2. Reference Character Images (Multiple allowed)", file_count="multiple", file_types=["image"])
                    input_ref_audio = gr.Audio(label="3. Reference Voice Audio (Optional)", type="filepath")
                    
                    with gr.Group():
                        gr.Markdown("### 🎨 Creative Settings")
                        with gr.Row():
                            target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Target Language")
                            style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Visual Style Preset")
                        prompt = gr.Textbox(value="a portrait of a beautiful character", label="Character Description")
                    
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
                    
        with gr.Tab("History"):
            history_display = gr.HTML(value=get_history_html())
            refresh_btn = gr.Button("🔄 Refresh History")

    run_btn.click(
        fn=transform_video,
        inputs=[input_video, input_ref_images, input_ref_audio, target_lang, prompt, style, skip_lipsync, preserve_bg, use_lcm, show_comparison],
        outputs=[output_video, comparison_video, status, history_display]
    )
    
    refresh_btn.click(fn=get_history_html, outputs=[history_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
