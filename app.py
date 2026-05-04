import os
import gradio as gr
from main import main as run_pipeline
from identity_discovery import IdentityDiscoverer
from identity_library import IdentityLibrary
import sys
import json
from datetime import datetime
from unittest.mock import patch
from logger_utils import logger, CONFIG

HISTORY_FILE = "transformation_history.json"
discoverer = IdentityDiscoverer()
lib = IdentityLibrary()

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
    
    msg = f"✅ Found {len(CURRENT_IDENTITIES)} unique characters."
    return msg, gallery_items

def save_to_library(char_index, name, ref_files, ref_audio):
    if char_index >= len(CURRENT_IDENTITIES):
        return "Error: Invalid character index."
    if not name:
        return "Error: Name is required."
    if ref_files is None:
        return "Error: At least one reference image is required."
        
    id_data = CURRENT_IDENTITIES[char_index]
    img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
    
    success = lib.save_character(name, id_data['embedding'], img_paths, ref_audio)
    if success:
        return f"Successfully saved '{name}' to library!"
    return "Failed to save character."

def get_library_html():
    chars = lib.get_all_characters()
    if not chars:
        return "<p style='text-align: center;'>Library is empty.</p>"
    
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;'>"
    for name, data in chars.items():
        thumb = data['images'][0] if data['images'] else ""
        html += f"""
        <div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center;'>
            <img src='file/{os.path.abspath(thumb)}' style='width: 100px; height: 100px; object-fit: cover; border-radius: 50%; margin-bottom: 10px;'>
            <p><strong>{name}</strong></p>
            <p style='font-size: 0.8em; color: #666;'>Saved Identity</p>
        </div>
        """
    return html + "</div>"

def start_transformation(video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, use_lcm, show_comparison, *char_inputs, progress=gr.Progress()):
    if video is None: return None, None, "Error: Input video required.", gr.update()
    
    identity_map = {}
    for i, identity in enumerate(CURRENT_IDENTITIES):
        ref_files = char_inputs[i*3] # images
        ref_audio = char_inputs[i*3 + 1] # audio
        # Optional: check if user selected a library character instead
        
        if ref_files is not None:
            img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
            identity_map[f"Character_{i}"] = {"images": img_paths, "audio": ref_audio if ref_audio else None}
        elif identity['library_match']:
            # Auto-map from library
            lib_data = lib.get_all_characters()[identity['library_match']]
            identity_map[f"Character_{i}"] = {"images": lib_data['images'], "audio": lib_data['audio']}

    if not identity_map: return None, None, "Error: No replacements assigned.", gr.update()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.mp4"
    comparison_path = f"comparison_{timestamp}.mp4"
    
    args = ["--video", video, "--identity_map", json.dumps(identity_map), "--target_lang", target_lang, "--prompt", f"{prompt}, {CONFIG.get('styles', {}).get(style, '')}", "--output", output_path]
    if skip_lipsync: args.append("--skip_lipsync")
    if not preserve_bg: args.append("--no_preserve_bg")
    if not smooth: args.append("--no_smooth")
    if not use_mask: args.append("--no_mask")
    if show_comparison: args.append("--comparison")
        
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🎙 Processing...")
            run_pipeline()
            res_video = output_path if os.path.exists(output_path) else None
            comp_video = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            if res_video:
                save_history({"timestamp": timestamp, "video": res_video, "comparison": comp_video, "prompt": prompt, "style": style, "lang": target_lang})
                return res_video, comp_video, "SUCCESS!", get_history_html()
            return None, None, "Error: Generation failed.", gr.update()
        except Exception as e:
            return None, None, f"Error: {str(e)}", gr.update()

# UI Definition
with gr.Blocks(title="AI Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video Character Studio")
    
    with gr.Tabs():
        with gr.Tab("1. Identity Discovery"):
            with gr.Row():
                with gr.Column():
                    scan_video = gr.Video(label="Input Video")
                    scan_btn = gr.Button("🔍 Discover Characters", variant="primary")
                with gr.Column():
                    scan_status = gr.Textbox(label="Status", interactive=False)
                    id_gallery = gr.Gallery(label="Found Characters", columns=4)
            
            gr.Markdown("---")
            gr.Markdown("### 🎭 Step 2: Assign & Save Identities")
            
            mapping_rows = []
            mapping_components = []
            for i in range(8):
                with gr.Row(visible=False) as row:
                    with gr.Column(scale=1):
                        gr.Markdown(f"**Identity {i}**")
                        save_name = gr.Textbox(label="Save Name (Library)", placeholder="Hero, Actor Name, etc.")
                        save_btn = gr.Button("💾 Save to Library", size="sm")
                    with gr.Column(scale=2):
                        ref_imgs = gr.File(label="Replacement Images", file_count="multiple")
                        ref_voice = gr.Audio(label="Replacement Voice", type="filepath")
                    mapping_rows.append(row)
                    mapping_components.extend([ref_imgs, ref_voice, save_name])
                    
                    # Save handler for this row
                    # Note: We need a way to pass 'i' correctly. In Gradio we use gr.State or closure.
                    save_btn.click(fn=save_to_library, inputs=[gr.State(i), save_name, ref_imgs, ref_voice], outputs=[scan_status])

            def update_rows(msg, thumbs):
                num = len(thumbs)
                return [msg, thumbs] + [gr.update(visible=(i < num)) for i in range(8)]

            scan_btn.click(fn=scan_video_identities, inputs=[scan_video], outputs=[scan_status, id_gallery] + mapping_rows).then(fn=update_rows, inputs=[scan_status, id_gallery], outputs=[scan_status, id_gallery] + mapping_rows)

        with gr.Tab("2. Production & Rendering"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Translation")
                        style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Visual Style")
                        prompt = gr.Textbox(value="a beautiful character", label="Prompt")
                    with gr.Row():
                        use_lcm = gr.Checkbox(label="🚀 Fast Mode", value=False)
                        smooth = gr.Checkbox(label="✨ Smooth", value=True)
                    with gr.Row():
                        preserve_bg = gr.Checkbox(label="Preserve BGM", value=True)
                        show_comp = gr.Checkbox(label="🎥 Comparison", value=True)
                    run_btn = gr.Button("🚀 Start Production", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_v = gr.Video(label="Result"); comp_v = gr.Video(label="Side-by-Side"); stat = gr.Textbox(label="Status")

        with gr.Tab("Identity Library"):
            lib_display = gr.HTML(value=get_library_html())
            refresh_lib = gr.Button("🔄 Refresh Library")
            refresh_lib.click(fn=get_library_html, outputs=[lib_display])

        with gr.Tab("History"):
            hist_display = gr.HTML(value=get_history_html())
            refresh_hist = gr.Button("🔄 Refresh History")
            refresh_hist.click(fn=get_history_html, outputs=[hist_display])

    run_btn.click(fn=start_transformation, inputs=[scan_video, target_lang, prompt, style, gr.State(False), preserve_bg, smooth, gr.State(True), use_lcm, show_comp] + mapping_components, outputs=[out_v, comp_v, stat, hist_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
