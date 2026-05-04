import os
import gradio as gr
from main import main as run_pipeline
from identity_discovery import IdentityDiscoverer
import sys
import json
from datetime import datetime
from unittest.mock import patch
from logger_utils import logger, CONFIG

HISTORY_FILE = "transformation_history.json"
discoverer = IdentityDiscoverer()

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
    thumbs = [id_data['thumbnail_path'] for id_data in CURRENT_IDENTITIES]
    
    # We create a help message
    msg = f"✅ Found {len(CURRENT_IDENTITIES)} unique characters. See thumbnails below."
    return msg, thumbs

def start_transformation(video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, use_lcm, show_comparison, *char_inputs, progress=gr.Progress()):
    """
    char_inputs is a flat list of [img1, img2, audio, img1, img2, audio, ...] 
    for each detected character.
    """
    if video is None:
        return None, None, "Error: Input video is required.", gr.update()

    # 1. Build the Identity Map from dynamic inputs
    identity_map = {}
    num_fields = 3 # (images, audio) - using 2 for simplicity: images (File), audio (Audio)
    # Actually let's use a simpler mapping: 
    # For each identity, the user provides:
    # - images (gr.File, multiple)
    # - audio (gr.Audio)
    
    for i, identity in enumerate(CURRENT_IDENTITIES):
        # Index into char_inputs
        ref_files = char_inputs[i*2]
        ref_audio = char_inputs[i*2 + 1]
        
        if ref_files is not None:
            # ref_files is a list of paths
            img_paths = [f.name for f in ref_files] if isinstance(ref_files, list) else [ref_files.name]
            identity_map[f"Character_{i}"] = {
                "images": img_paths,
                "audio": ref_audio if ref_audio else None
            }

    if not identity_map:
        return None, None, "Error: No replacements assigned. Please map at least one character.", gr.update()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.mp4"
    comparison_path = f"comparison_{timestamp}.mp4"
    
    # 2. Prepare CLI Arguments
    args = [
        "--video", video,
        "--identity_map", json.dumps(identity_map),
        "--target_lang", target_lang,
        "--prompt", f"{prompt}, {CONFIG.get('styles', {}).get(style, '')}",
        "--output", output_path
    ]
    
    if skip_lipsync: args.append("--skip_lipsync")
    if not preserve_bg: args.append("--no_preserve_bg")
    if not smooth: args.append("--no_smooth")
    if not use_mask: args.append("--no_mask")
    if show_comparison: args.append("--comparison")
        
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    logger.info(f"UI transformation started for {len(identity_map)} characters.")
    progress(0, desc="🚀 Initializing...")
    
    # 3. Execute
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🎙 Processing Pipeline...")
            run_pipeline()
            
            res_video = output_path if os.path.exists(output_path) else None
            comp_video = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            
            if res_video:
                save_history({"timestamp": timestamp, "video": res_video, "comparison": comp_video, "prompt": prompt, "style": style, "lang": target_lang})
                progress(1.0, desc="✅ Success!")
                return res_video, comp_video, "SUCCESS!", get_history_html()
            else:
                return None, None, "Error: Generation failed.", gr.update()
        except Exception as e:
            logger.error(f"UI Error: {str(e)}")
            return None, None, f"Error: {str(e)}", gr.update()

def get_history_html():
    history = load_history()
    if not history: return "<p style='text-align: center;'>No history yet.</p>"
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;'>"
    for entry in history:
        html += f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'><p><strong>{entry['timestamp']}</strong></p><p>{entry['prompt']}</p><a href='file/{os.path.abspath(entry['video'])}' target='_blank'>View</a></div>"
    return html + "</div>"

# Define the UI
with gr.Blocks(title="Video & Audio Transformer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Video & Audio Transformer")
    
    with gr.Tabs() as tabs:
        with gr.Tab("1. Identity Discovery"):
            with gr.Row():
                with gr.Column():
                    scan_input_video = gr.Video(label="Upload Video to Scan")
                    scan_btn = gr.Button("🔍 Scan for Characters", variant="primary")
                with gr.Column():
                    scan_status = gr.Textbox(label="Scan Status", interactive=False)
                    id_gallery = gr.Gallery(label="Detected Characters", columns=3, height="auto")
            
            gr.Markdown("---")
            gr.Markdown("### 🎭 Step 2: Assign Replacements")
            gr.Markdown("For each detected character above, upload their new identity.")
            
            # Dynamic mapping area
            mapping_rows = []
            for i in range(10): # Support up to 10 characters
                with gr.Row(visible=False) as row:
                    gr.Markdown(f"**Character {i}**")
                    ref_imgs = gr.File(label="Replacement Images", file_count="multiple", file_types=["image"])
                    ref_voice = gr.Audio(label="Replacement Voice", type="filepath")
                    mapping_rows.append(row)
                    # We store the inputs to pass to the function
                    # These will be at indices 0, 1, 2, ... in the variadic *args
            
            mapping_inputs = []
            for row in mapping_rows:
                # Extract the components from the row (children)
                mapping_inputs.extend(row.children[1:]) # Skip the Markdown label

            def show_mapping_rows(msg, thumbs):
                updates = []
                num_found = len(thumbs)
                for i in range(10):
                    updates.append(gr.update(visible=(i < num_found)))
                return [msg, thumbs] + updates

        with gr.Tab("2. Global Settings & Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### ⚙️ Transformation Settings")
                        with gr.Row():
                            target_lang = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Target Language")
                            style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Visual Style")
                        prompt = gr.Textbox(value="a portrait of a beautiful character", label="Base Prompt")
                    
                    with gr.Row():
                        use_lcm = gr.Checkbox(label="🚀 Fast Mode", value=False)
                        smooth = gr.Checkbox(label="✨ Smooth", value=True)
                        use_mask = gr.Checkbox(label="🖼 Masking", value=True)
                    with gr.Row():
                        skip_lipsync = gr.Checkbox(label="Skip Sync", value=False)
                        preserve_bg = gr.Checkbox(label="Preserve BGM", value=True)
                        show_comparison = gr.Checkbox(label="🎥 Comparison", value=True)
                        
                    run_btn = gr.Button("🚀 Start Transformation", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    output_video = gr.Video(label="✨ Transformed Result")
                    comparison_video = gr.Video(label="🎞 Side-by-Side")
                    status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("History"):
            history_display = gr.HTML(value=get_history_html())
            refresh_btn = gr.Button("🔄 Refresh History")

    # Event Handlers
    scan_btn.click(
        fn=scan_video_identities, 
        inputs=[scan_input_video], 
        outputs=[scan_status, id_gallery] + mapping_rows
    ).then(
        fn=show_mapping_rows,
        inputs=[scan_status, id_gallery],
        outputs=[scan_status, id_gallery] + mapping_rows
    )
    
    run_btn.click(
        fn=transform_video,
        inputs=[scan_input_video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, use_lcm, show_comparison] + mapping_inputs,
        outputs=[output_video, comparison_video, status, history_display]
    )
    
    refresh_btn.click(fn=get_history_html, outputs=[history_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
