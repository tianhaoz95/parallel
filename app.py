import os
import gradio as gr
from main import main as run_pipeline
from audio_pipeline import AudioPipeline
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

# Global state
CURRENT_IDENTITIES = []
CURRENT_SEGMENTS = []

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

def prepare_script(video, target_lang):
    global CURRENT_SEGMENTS
    if not video: return "Please upload a video.", []
    
    # 1. Initialize pipe
    audio_pipe = AudioPipeline(
        asr_model_path=CONFIG.get('models', {}).get('asr'),
        translation_model_path_prefix=CONFIG.get('models', {}).get('translation_prefix')
    )
    
    # 2. Extract and Transcribe
    logger.info("Generating initial script...")
    # Use moviepy to get audio path
    import moviepy.editor as mp
    v = mp.VideoFileClip(video)
    audio_path = "temp_script_gen.wav"
    v.audio.write_audiofile(audio_path, logger=None)
    
    full_text, detected_lang, segments = audio_pipe.transcribe_audio(audio_path, detect_language=True)
    
    # 3. Initial Translation
    if detected_lang != target_lang:
        audio_pipe._load_translation_model(detected_lang, target_lang)
        for s in segments:
            inputs = audio_pipe.tokenizer(s['text'], return_tensors="pt").to(audio_pipe.device)
            tokens = audio_pipe.translation_model.generate(**inputs)
            s['translated_text'] = audio_pipe.tokenizer.decode(tokens[0], skip_special_tokens=True)
    else:
        for s in segments: s['translated_text'] = s['text']
        
    CURRENT_SEGMENTS = segments
    
    # Prepare dataframe-ready format
    script_data = [[s['start'], s['end'], s['text'], s['translated_text']] for s in segments]
    
    v.close()
    if os.path.exists(audio_path): os.remove(audio_path)
    
    return f"Script generated! Detected language: {detected_lang}", script_data

def scan_video_identities(video):
    global CURRENT_IDENTITIES
    if not video: return "Please upload a video first.", []
    CURRENT_IDENTITIES = discoverer.find_unique_faces(video)
    gallery_items = []
    for id_data in CURRENT_IDENTITIES:
        label = id_data['library_match'] if id_data['library_match'] else f"ID {id_data['id']}"
        gallery_items.append((id_data['thumbnail_path'], label))
    return f"✅ Found {len(CURRENT_IDENTITIES)} unique characters.", gallery_items

def start_transformation(video, target_lang, prompt, style, skip_lipsync, preserve_bg, smooth, use_mask, upscale, use_lcm, show_comparison, script_df, *char_inputs, progress=gr.Progress()):
    if video is None: return None, None, "Error: Video required.", gr.update()
    
    # 1. Update segments from editor
    # script_df is [ [start, end, src, trans], ... ]
    edited_segments = []
    for row in script_df:
        edited_segments.append({
            'start': float(row[0]),
            'end': float(row[1]),
            'text': row[2],
            'translated_text': row[3]
        })
    
    # 2. Build Identity Map
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
    
    # 3. Save edited script to a temp JSON for the backend to use
    with open("temp_edited_script.json", "w") as f:
        json.dump(edited_segments, f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.mp4"
    comparison_path = f"comparison_{timestamp}.mp4"
    
    args = [
        "--video", video, 
        "--identity_map", json.dumps(identity_map), 
        "--target_lang", target_lang, 
        "--prompt", f"{prompt}, {CONFIG.get('styles', {}).get(style, '')}", 
        "--output", output_path,
        "--external_script", "temp_edited_script.json" # New flag needed in main.py
    ]
    if skip_lipsync: args.append("--skip_lipsync")
    if not preserve_bg: args.append("--no_preserve_bg")
    if not smooth: args.append("--no_smooth")
    if not use_mask: args.append("--no_mask")
    if upscale: args.append("--upscale")
    if show_comparison: args.append("--comparison")
    
    CONFIG['defaults']['use_lcm'] = use_lcm
    
    with patch.object(sys, 'argv', ["main.py"] + args):
        try:
            progress(0.1, desc="🚀 Transformation in progress...")
            run_pipeline()
            res_v = output_path if os.path.exists(output_path) else None
            comp_v = comparison_path if show_comparison and os.path.exists(comparison_path) else None
            if res_v:
                save_history({"timestamp": timestamp, "video": res_v, "comparison": comp_v, "prompt": prompt, "style": style, "lang": target_lang})
                return res_v, comp_v, "SUCCESS!", get_history_html()
            return None, None, "Error: Failed.", gr.update()
        except Exception as e: return None, None, f"Error: {str(e)}", gr.update()

def get_history_html():
    history = load_history()
    if not history: return "<p style='text-align: center;'>No history yet.</p>"
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px;'>"
    for entry in history:
        html += f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px;'><p><strong>{entry['timestamp']}</strong></p><p>{entry['prompt']}</p><a href='file/{os.path.abspath(entry['video'])}' target='_blank'>View</a></div>"
    return html + "</div>"

# UI
with gr.Blocks(title="Professional AI Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Local Professional AI Video Studio")
    
    with gr.Tabs():
        with gr.Tab("1. Project Discovery"):
            with gr.Row():
                with gr.Column():
                    scan_video = gr.Video(label="Input Video")
                    target_lang_ui = gr.Dropdown(choices=["es", "fr", "de", "it", "zh"], value="es", label="Target Language")
                    scan_btn = gr.Button("🔍 Discover Characters & Generate Script", variant="primary")
                with gr.Column():
                    scan_status = gr.Textbox(label="Status", interactive=False)
                    id_gallery = gr.Gallery(label="Found Characters", columns=4)
            
            gr.Markdown("---")
            gr.Markdown("### 📝 Step 2: Edit Script & Assign Identities")
            
            with gr.Row():
                with gr.Column(scale=2):
                    script_editor = gr.Dataframe(
                        headers=["Start", "End", "Original Text", "Translated Text"],
                        datatype=["number", "number", "str", "str"],
                        label="Interactive Script Editor",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    mapping_rows = []; mapping_components = []
                    for i in range(8):
                        with gr.Row(visible=False) as row:
                            gr.Markdown(f"**ID {i}**")
                            ref_imgs = gr.File(label="Images", file_count="multiple")
                            ref_voice = gr.Audio(label="Voice", type="filepath")
                            mapping_rows.append(row); mapping_components.extend([ref_imgs, ref_voice, gr.State(f"ID_{i}")])
            
            def update_ui(msg, thumbs):
                num = len(thumbs)
                return [msg, thumbs] + [gr.update(visible=(i < num)) for i in range(8)]

            scan_btn.click(prepare_script, inputs=[scan_video, target_lang_ui], outputs=[scan_status, script_editor])
            scan_btn.click(scan_video_identities, inputs=[scan_video], outputs=[scan_status, id_gallery] + mapping_rows).then(fn=update_ui, inputs=[scan_status, id_gallery], outputs=[scan_status, id_gallery] + mapping_rows)

        with gr.Tab("2. Production & Rendering"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        style = gr.Dropdown(choices=list(CONFIG.get('styles', {}).keys()), value="Cinematic", label="Visual Style")
                        prompt = gr.Textbox(value="a beautiful character", label="Character Prompt Override")
                    with gr.Row():
                        use_lcm = gr.Checkbox(label="🚀 Fast Mode", value=False); smooth = gr.Checkbox(label="✨ Smooth", value=True)
                        preserve_bg = gr.Checkbox(label="Preserve BGM", value=True); show_comp = gr.Checkbox(label="🎥 Comparison", value=True); use_mask = gr.Checkbox(label="🖼 Masking", value=True); upscale = gr.Checkbox(label="💎 HD", value=True)
                    run_btn = gr.Button("🚀 Finalize & Render", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_v = gr.Video(label="Result"); comp_v = gr.Video(label="Side-by-Side"); stat = gr.Textbox(label="Status")

        with gr.Tab("Identity Library"):
            gr.HTML("<p>Library View Coming Soon (Persistence Layer Active)</p>")
        with gr.Tab("History"):
            hist_display = gr.HTML(value=get_history_html())

    run_btn.click(fn=start_transformation, inputs=[scan_video, target_lang_ui, prompt, style, gr.State(False), preserve_bg, smooth, use_mask, upscale, use_lcm, show_comp, script_editor] + mapping_components, outputs=[out_v, comp_v, stat, hist_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
