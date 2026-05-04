import os
import gradio as gr
from main import main as run_pipeline
from identity_discovery import IdentityDiscoverer
from identity_library import IdentityLibrary
from project_manager import ProjectManager
import sys
import json
from datetime import datetime
from unittest.mock import patch
from logger_utils import logger, CONFIG
from dotenv import load_dotenv

load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

HISTORY_FILE = "transformation_history.json"
discoverer = IdentityDiscoverer()
lib = IdentityLibrary()
proj = ProjectManager()

# Global state to store detected identities
CURRENT_IDENTITIES = []

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f: return json.load(f)
        except: return []
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, 'w') as f: json.dump(history[:50], f, indent=2)

def scan_video_identities(video):
    global CURRENT_IDENTITIES
    if not video: return "Please upload a video first.", []
    CURRENT_IDENTITIES = discoverer.find_unique_faces(video)
    gallery_items = []
    for id_data in CURRENT_IDENTITIES:
        label = id_data['library_match'] if id_data['library_match'] else f"ID {id_data['id']}"
        gallery_items.append((id_data['thumbnail_path'], label))
    return f"✅ Found {len(CURRENT_IDENTITIES)} unique characters.", gallery_items

def start_transformation(video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, upscale, use_lcm, show_comparison, use_cpu, *char_inputs, progress=gr.Progress()):
    if video is None: return None, None, "Error: Video required.", gr.update()
    
    if use_cpu:
        os.environ["USE_CPU"] = "1"
    else:
        os.environ.pop("USE_CPU", None)
    
    identity_map = {}
    for i, identity in enumerate(CURRENT_IDENTITIES):
        # We now have 4 inputs per character: images, audio, save_name, custom_prompt
        ref_files = char_inputs[i*4]
        ref_audio = char_inputs[i*4 + 1]
        char_prompt = char_inputs[i*4 + 3]
        
        if ref_files is not None:
            # Handle list or single file, and handle if they are objects with .name or just strings
            if isinstance(ref_files, list):
                img_paths = [f.name if hasattr(f, 'name') else f for f in ref_files]
            else:
                img_paths = [ref_files.name if hasattr(ref_files, 'name') else ref_files]
            
            identity_map[f"Character_{i}"] = {
                "images": img_paths, 
                "audio": ref_audio if ref_audio else None,
                "prompt": char_prompt if char_prompt else None
            }
        elif identity['library_match']:
            lib_data = lib.get_all_characters()[identity['library_match']]
            identity_map[f"Character_{i}"] = {
                "images": lib_data['images'], 
                "audio": lib_data['audio'],
                "prompt": char_prompt if char_prompt else None
            }

    if not identity_map: return None, None, "Error: No replacements.", gr.update()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.mp4"
    comparison_path = f"comparison_{timestamp}.mp4"
    
    args = ["--video", video, "--identity_map", json.dumps(identity_map), "--target_lang", target_lang, "--prompt", f"{prompt}, {CONFIG.get('styles', {}).get(style, '')}", "--output", output_path]
    if skip_lipsync: args.append("--skip_lipsync")
    if not preserve_bg: args.append("--no_preserve_bg")
    if not smooth: args.append("--no_smooth")
    if not use_mask: args.append("--no_mask")
    if upscale: args.append("--upscale")
    if show_comparison: args.append("--comparison")
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🎙 Processing...")
            run_pipeline()
            res_v = output_path if os.path.exists(output_path) else None
            comp_v = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            if res_v:
                save_history({"timestamp": timestamp, "video": res_v, "comparison": comp_v, "prompt": prompt, "style": style, "lang": target_lang})
                return res_v, comp_v, "SUCCESS!", get_history_html()
            return None, None, "Error: Generation failed.", gr.update()
        except Exception as e: return None, None, f"Error: {str(e)}", gr.update()

def get_history_html():
    history = load_history()
    if not history: return "<p style='text-align: center;'>No history yet.</p>"
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;'>"
    for entry in history:
        html += f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'><p><strong>{entry['timestamp']}</strong></p><p>{entry['prompt']}</p><a href='file/{os.path.abspath(entry['video'])}' target='_blank'>View</a></div>"
    return html + "</div>"

# UI Definition
with gr.Blocks(title="AI Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video Character Studio")
    
    with gr.Tabs():
        with gr.Tab("1. Identity Discovery"):
            with gr.Row():
                with gr.Column():
                    scan_video = gr.Video(label="Input Video"); scan_btn = gr.Button("🔍 Discover Characters", variant="primary")
                with gr.Column():
                    scan_status = gr.Textbox(label="Status", interactive=False); id_gallery = gr.Gallery(label="Found Characters", columns=4)
            gr.Markdown("---"); gr.Markdown("### 🎭 Step 2: Assign & Customize Identities")
            mapping_rows = []; mapping_components = []
            for i in range(8):
                with gr.Row(visible=False) as row:
                    with gr.Column(scale=1):
                        gr.Markdown(f"**Identity {i}**"); save_name = gr.Textbox(label="Save Name"); save_btn = gr.Button("💾 Save", size="sm")
                    with gr.Column(scale=2):
                        with gr.Row():
                            ref_imgs = gr.File(label="Images", file_count="multiple")
                            ref_voice = gr.Audio(label="Voice", type="filepath")
                        char_prompt = gr.Textbox(label="Visual Description Override", placeholder="e.g., wearing a red tuxedo, cyberpunk style")
                    mapping_rows.append(row); mapping_components.extend([ref_imgs, ref_voice, save_name, char_prompt])
            def update_rows(msg, thumbs):
                num = len(thumbs); return [msg, thumbs] + [gr.update(visible=(i < num)) for i in range(8)]
            scan_btn.click(fn=scan_video_identities, inputs=[scan_video], outputs=[scan_status, id_gallery]).then(fn=update_rows, inputs=[scan_status, id_gallery], outputs=[scan_status, id_gallery] + mapping_rows)

        with gr.Tab("2. Production & Rendering"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Translation")
                        style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Global Visual Style")
                        prompt = gr.Textbox(value="a beautiful character", label="Global Base Prompt")
                    with gr.Row():
                        use_lcm = gr.Checkbox(label="🚀 Fast Mode", value=False); smooth = gr.Checkbox(label="✨ Smooth", value=True)
                        preserve_bg = gr.Checkbox(label="Preserve BGM", value=True); show_comp = gr.Checkbox(label="🎥 Comparison", value=True)
                        use_cpu = gr.Checkbox(label="🖥️ CPU Testing Mode", value=False)
                    run_btn = gr.Button("🚀 Start Production", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_v = gr.Video(label="Result"); comp_v = gr.Video(label="Side-by-Side"); stat = gr.Textbox(label="Status")

        with gr.Tab("History"):
            hist_display = gr.HTML(value=get_history_html())

    run_btn.click(fn=start_transformation, inputs=[scan_video, target_lang, prompt, style, gr.State(False), preserve_bg, smooth, gr.State(True), gr.State(False), use_lcm, show_comp, use_cpu] + mapping_components, outputs=[out_v, comp_v, stat, hist_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
