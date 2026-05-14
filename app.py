"""
Text-to-Shot Demo: Flask GUI for generating camera trajectories from text.

Usage:
    PYTHONPATH=. python app.py --checkpoint /transfer/fm-v10-checkpoints/fm_best.pth --port 7861
"""

import argparse
import os
import json
import time
import base64
import torch
import numpy as np
from flask import Flask, request, render_template_string

from experiments.flow_matching.generate import (
    load_model, smooth_trajectory, regularize_person_trajectory,
    freeze_static_dims, SHOT_TYPE_MAP, MOTION_TYPE_MAP,
)
from generate import visualize_joint

# Global state
FLOW_MODEL = None
CONFIG = None
TEXT_ENCODER = None
NORM_MEAN = None
NORM_STD = None
DEVICE = None

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Text-to-Shot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #fff; font-family: -apple-system, 'Segoe UI', sans-serif; }
  .header { padding: 24px 40px; border-bottom: 1px solid #21262d; }
  .header h1 { font-size: 28px; color: #58a6ff; }
  .header p { color: #8b949e; margin-top: 4px; font-size: 14px; }
  .container { display: flex; gap: 24px; padding: 24px 40px; min-height: calc(100vh - 100px); }
  .panel-left { width: 320px; flex-shrink: 0; }
  .panel-right { flex: 1; display: flex; align-items: flex-start; justify-content: center; }
  label { display: block; color: #8b949e; font-size: 13px; margin-bottom: 6px; margin-top: 16px; }
  textarea { width: 100%; background: #161b22; border: 1px solid #30363d; color: #fff;
             border-radius: 6px; padding: 10px; font-size: 14px; resize: vertical; min-height: 70px; }
  textarea:focus { border-color: #58a6ff; outline: none; }
  select, input[type=range] { width: 100%; background: #161b22; border: 1px solid #30363d;
           color: #fff; border-radius: 6px; padding: 8px; font-size: 14px; }
  select:focus { border-color: #58a6ff; outline: none; }
  input[type=range] { -webkit-appearance: none; height: 6px; border-radius: 3px; margin-top: 8px;
                      background: #30363d; border: none; padding: 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: #58a6ff; cursor: pointer; }
  .range-row { display: flex; justify-content: space-between; align-items: center; }
  .range-val { color: #58a6ff; font-size: 14px; font-weight: bold; min-width: 30px; text-align: right; }
  button { width: 100%; margin-top: 24px; padding: 12px; background: #238636; color: #fff;
           border: none; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; }
  button:hover { background: #2ea043; }
  button:disabled { background: #21262d; color: #484f58; cursor: wait; }
  .result-img { max-width: 100%; border-radius: 8px; border: 1px solid #21262d; }
  .placeholder { color: #484f58; font-size: 15px; text-align: center; padding: 80px 20px;
                 border: 2px dashed #21262d; border-radius: 8px; width: 100%; }
  .examples { margin-top: 20px; }
  .examples h3 { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .ex-btn { display: block; width: 100%; text-align: left; padding: 8px 10px; margin-bottom: 4px;
            background: #161b22; border: 1px solid #21262d; color: #c9d1d9; border-radius: 4px;
            font-size: 12px; cursor: pointer; }
  .ex-btn:hover { border-color: #58a6ff; color: #58a6ff; }
  .spinner { display: none; text-align: center; padding: 60px; color: #58a6ff; font-size: 16px; }
  .spinner.active { display: block; }
</style>
</head>
<body>
<div class="header">
  <h1>Text-to-Shot</h1>
  <p>Generate cinematographic camera trajectories from natural language</p>
</div>
<div class="container">
  <div class="panel-left">
    <form id="genForm">
      <label>Text Prompt</label>
      <textarea id="text" name="text" placeholder="e.g. The camera orbits around the character as they stand still"></textarea>

      <label>Motion Type</label>
      <select id="motion" name="motion">
        {% for m in motions %}<option value="{{m}}">{{m}}</option>{% endfor %}
      </select>

      <label>Shot Type</label>
      <select id="shot" name="shot">
        {% for s in shots %}<option value="{{s}}" {% if s=='medium-shot' %}selected{% endif %}>{{s}}</option>{% endfor %}
      </select>

      <label>
        <div class="range-row">
          <span>Guidance Scale</span>
          <span class="range-val" id="gsVal">3.0</span>
        </div>
      </label>
      <input type="range" id="gs" name="guidance_scale" min="1" max="10" step="0.5" value="3.0"
             oninput="document.getElementById('gsVal').textContent=this.value">

      <button type="submit" id="genBtn">Generate</button>
    </form>

    <div class="examples">
      <h3>Examples</h3>
      <button class="ex-btn" onclick="fillExample('The camera remains static while the character walks forward','static')">Static — character walks forward</button>
      <button class="ex-btn" onclick="fillExample('As the character moves forward, the camera pushes in','dolly-in')">Dolly-in — camera approaches</button>
      <button class="ex-btn" onclick="fillExample('The camera pulls out as the character stands still','dolly-out')">Dolly-out — camera retreats</button>
      <button class="ex-btn" onclick="fillExample('The camera pans left as the character walks forward','pan-left')">Pan left</button>
      <button class="ex-btn" onclick="fillExample('The camera orbits around the character as they stand still','orbit')">Orbit — camera circles person</button>
      <button class="ex-btn" onclick="fillExample('The camera cranes up as the character moves forward','crane-up')">Crane up</button>
      <button class="ex-btn" onclick="fillExample('The camera tracks the character as they walk to the right','track')">Track — follow character</button>
    </div>
  </div>

  <div class="panel-right">
    <div id="spinner" class="spinner">Generating trajectory...</div>
    <div id="placeholder" class="placeholder">Generated trajectory will appear here</div>
    <img id="resultImg" class="result-img" style="display:none" />
  </div>
</div>

<script>
function fillExample(text, motion) {
  document.getElementById('text').value = text;
  document.getElementById('motion').value = motion;
}
document.getElementById('genForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('genBtn');
  const spinner = document.getElementById('spinner');
  const placeholder = document.getElementById('placeholder');
  const img = document.getElementById('resultImg');

  btn.disabled = true;
  btn.textContent = 'Generating...';
  spinner.classList.add('active');
  placeholder.style.display = 'none';
  img.style.display = 'none';

  const params = new URLSearchParams({
    text: document.getElementById('text').value,
    motion: document.getElementById('motion').value,
    shot: document.getElementById('shot').value,
    guidance_scale: document.getElementById('gs').value,
  });

  try {
    const resp = await fetch('/generate?' + params.toString());
    const data = await resp.json();
    if (data.image) {
      img.src = 'data:image/png;base64,' + data.image;
      img.style.display = 'block';
    } else {
      placeholder.textContent = 'Error: ' + (data.error || 'unknown');
      placeholder.style.display = 'block';
    }
  } catch(err) {
    placeholder.textContent = 'Error: ' + err.message;
    placeholder.style.display = 'block';
  }
  spinner.classList.remove('active');
  btn.disabled = false;
  btn.textContent = 'Generate';
});
</script>
</body>
</html>
"""


def init_model(checkpoint_path, device):
    global FLOW_MODEL, CONFIG, TEXT_ENCODER, NORM_MEAN, NORM_STD, DEVICE
    DEVICE = device
    FLOW_MODEL, CONFIG = load_model(checkpoint_path, device)

    try:
        from src.models.text_encoder import CLIPTextEncoder
        TEXT_ENCODER = CLIPTextEncoder(
            model_name=CONFIG['text_encoder']['model_name'], device=device
        ).to(device)
    except Exception:
        print("CLIP unavailable, using random embeddings.")

    norm_stats_path = CONFIG['data'].get('norm_stats_path', None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        NORM_MEAN = torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        NORM_STD = torch.tensor(stats['std'], dtype=torch.float32, device=device)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
        motions=list(MOTION_TYPE_MAP.keys()),
        shots=list(SHOT_TYPE_MAP.keys()),
    )


@app.route('/generate')
@torch.no_grad()
def generate():
    text = request.args.get('text', '').strip()
    motion_type = request.args.get('motion', 'static')
    shot_type = request.args.get('shot', 'medium-shot')
    guidance_scale = float(request.args.get('guidance_scale', 3.0))

    if not text:
        return json.dumps({"error": "empty prompt"})

    model_cfg = CONFIG['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = CONFIG['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    num_steps = CONFIG['flow_matching']['num_steps']

    if TEXT_ENCODER:
        text_embed = TEXT_ENCODER([text])
    else:
        text_embed = torch.randn(1, 512, device=DEVICE)

    shot_idx = SHOT_TYPE_MAP.get(shot_type, 1)
    motion_idx = MOTION_TYPE_MAP.get(motion_type, 0)
    shot_t = torch.tensor([shot_idx], device=DEVICE)
    motion_t = torch.tensor([motion_idx], device=DEVICE)

    y = FLOW_MODEL.sample(
        text_embed, shot_type=shot_t, motion_type=motion_t,
        device=DEVICE, guidance_scale=guidance_scale,
        num_steps=num_steps,
    )

    if NORM_MEAN is not None:
        y = y * NORM_STD + NORM_MEAN

    y_np = y[0].cpu().numpy()
    person_traj = y_np[:person_total].reshape(num_frames, person_dim)
    camera_traj = y_np[person_total:].reshape(num_frames, camera_dim)

    # Post-process
    pw = min(31, person_traj.shape[0] if person_traj.shape[0] % 2 == 1 else person_traj.shape[0] - 1)
    person_traj = smooth_trajectory(person_traj, window=pw)
    person_traj = regularize_person_trajectory(person_traj, static_threshold=0.08, segment_cost=0.3)
    person_traj = freeze_static_dims(person_traj, threshold=0.05)
    camera_traj = smooth_trajectory(camera_traj, window=21, angle_dims=[3, 4, 5], angle_window=31)
    camera_traj = freeze_static_dims(camera_traj, threshold=0.05)

    # Save to temp file, read as base64
    save_path = f"/tmp/tts_{int(time.time())}.png"
    visualize_joint(person_traj, camera_traj, text, motion_type, save_path=save_path)

    with open(save_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    os.remove(save_path)

    return json.dumps({"image": img_b64})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()

    print("Loading model...")
    init_model(args.checkpoint, args.device if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded. Starting server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(host='0.0.0.0', port=args.port, debug=False)
