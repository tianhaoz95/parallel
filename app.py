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

def save_to_library(char_index, name, ref_files, ref_audio):
    if char_index >= len(CURRENT_IDENTITIES): return "Error: Invalid index."
    if not name or ref_files is None: return "Error: Name and images required."
    id_data = CURRENT_IDENTITIES[char_index]
    img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
    if lib.save_character(name, id_data['embedding'], img_paths, ref_audio):
        return f"Successfully saved '{name}' to library!"
    return "Failed to save."

def get_library_html():
    chars = lib.get_all_characters()
    if not chars: return "<p style='text-align: center;'>Library is empty.</p>"
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;'>"
    for name, data in chars.items():
        thumb = data['images'][0] if data['images'] else ""
        html += f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center;'><img src='file/{os.path.abspath(thumb)}' style='width: 100px; height: 100px; border-radius: 50%; object-fit: cover;'><p><strong>{name}</strong></p></div>"
    return html + "</div>"

def get_projects_html():
    projects = proj.list_projects()
    if not projects: return "<p style='text-align: center;'>No projects saved.</p>"
    html = "<div style='display: grid; grid-template-columns: 1fr; gap: 10px;'>"
    for p in projects:
        html += f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'><p><strong>{p['name']}</strong> ({p['id']})</p><p style='font-size: 0.8em;'>Video: {p['video_path']}</p></div>"
    return html + "</div>"

def save_current_project(name, video, target_lang, prompt, style, *char_inputs):
    identity_map = {}
    for i, identity in enumerate(CURRENT_IDENTITIES):
        ref_files = char_inputs[i*3]
        ref_audio = char_inputs[i*3+1]
        if ref_files:
            img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
            identity_map[f"Character_{i}"] = {"images": img_paths, "audio": ref_audio}
            
    settings = {"target_lang": target_lang, "prompt": prompt, "style": style}
    proj.create_project(name, video, identity_map, settings)
    return f"Project '{name}' saved successfully!"

def start_transformation(video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, upscale, use_lcm, show_comparison, *char_inputs, progress=gr.Progress()):
    if video is None: return None, None, "Error: Video required.", gr.update()
    identity_map = {}
    for i, identity in enumerate(CURRENT_IDENTITIES):
        ref_files = char_inputs[i*3]; ref_audio = char_inputs[i*3 + 1]
        if ref_files:
            img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
            identity_map[f"Character_{i}"] = {"images": img_paths, "audio": ref_audio if ref_audio else None}
        elif identity['library_match']:
            lib_data = lib.get_all_characters()[identity['library_match']]
            identity_map[f"Character_{i}"] = {"images": lib_data['images'], "audio": lib_data['audio']}
    
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
            return None, None, "Error: Failed.", gr.update()
        except Exception as e: return None, None, f"Error: {str(e)}", gr.update()

# UI
with gr.Blocks(title="AI Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video Character Studio")
    with gr.Tabs():
        with gr.Tab("1. Identity Discovery"):
            with gr.Row():
                with gr.Column():
                    scan_video = gr.Video(label="Input Video"); scan_btn = gr.Button("🔍 Discover Characters", variant="primary")
                with gr.Column():
                    scan_status = gr.Textbox(label="Status", interactive=False); id_gallery = gr.Gallery(label="Found Characters", columns=4)
            gr.Markdown("---"); gr.Markdown("### 🎭 Step 2: Assign & Save Identities")
            mapping_rows = []; mapping_components = []
            for i in range(8):
                with gr.Row(visible=False) as row:
                    with gr.Column(scale=1):
                        gr.Markdown(f"**Identity {i}**"); save_name = gr.Textbox(label="Save Name"); save_btn = gr.Button("💾 Save", size="sm")
                    with gr.Column(scale=2):
                        ref_imgs = gr.File(label="Images", file_count="multiple"); ref_voice = gr.Audio(label="Voice", type="filepath")
                    mapping_rows.append(row); mapping_components.extend([ref_imgs, ref_voice, save_name])
                    save_btn.click(fn=save_to_library, inputs=[gr.State(i), save_name, ref_imgs, ref_voice], outputs=[scan_status])
            def update_rows(msg, thumbs):
                num = len(thumbs); return [msg, thumbs] + [gr.update(visible=(i < num)) for i in range(8)]
            scan_btn.click(fn=scan_video_identities, inputs=[scan_video], outputs=[scan_status, id_gallery] + mapping_rows).then(fn=update_rows, inputs=[scan_status, id_gallery], outputs=[scan_status, id_gallery] + mapping_rows)

        with gr.Tab("2. Production & Rendering"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Translation")
                        style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Style")
                        prompt = gr.Textbox(value="a portrait of a beautiful character", label="Prompt")
                    with gr.Row():
                        use_lcm = gr.Checkbox(label="🚀 Fast Mode", value=False); smooth = gr.Checkbox(label="✨ Smooth", value=True)
                        preserve_bg = gr.Checkbox(label="Preserve BGM", value=True); show_comp = gr.Checkbox(label="🎥 Comparison", value=True)
                    run_btn = gr.Button("🚀 Start Production", variant="primary", size="lg")
                    gr.Markdown("---"); proj_name = gr.Textbox(label="Project Name", placeholder="My New Film")
                    save_proj_btn = gr.Button("📁 Save Project Setup")
                with gr.Column(scale=1):
                    out_v = gr.Video(label="Result"); comp_v = gr.Video(label="Side-by-Side"); stat = gr.Textbox(label="Status")

        with gr.Tab("Identity Library"):
            lib_display = gr.HTML(value=get_library_html()); refresh_lib = gr.Button("🔄 Refresh"); refresh_lib.click(fn=get_library_html, outputs=[lib_display])
        with gr.Tab("Projects"):
            proj_display = gr.HTML(value=get_projects_html()); refresh_proj = gr.Button("🔄 Refresh"); refresh_proj.click(fn=get_projects_html, outputs=[proj_display])
        with gr.Tab("History"):
            hist_display = gr.HTML(value=get_history_html()); refresh_hist = gr.Button("🔄 Refresh"); refresh_hist.click(fn=get_history_html, outputs=[hist_display])

    run_btn.click(fn=start_transformation, inputs=[scan_video, target_lang, prompt, style, gr.State(False), preserve_bg, smooth, gr.State(True), use_lcm, show_comp] + mapping_components, outputs=[out_v, comp_v, stat, hist_display])
    save_proj_btn.click(fn=save_current_project, inputs=[proj_name, scan_video, target_lang, prompt, style] + mapping_components, outputs=[stat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
