"""
Text-to-Shot Demo: Gradio GUI for generating camera trajectories from text.

Usage:
    PYTHONPATH=. python app.py --checkpoint /transfer/fm-v10-checkpoints/fm_best.pth
    # Then open http://localhost:7860 in browser
"""

import argparse
import os
import json
import tempfile
import torch
import numpy as np
import gradio as gr

from experiments.flow_matching.generate import (
    load_model, smooth_trajectory, regularize_person_trajectory,
    freeze_static_dims, SHOT_TYPE_MAP, MOTION_TYPE_MAP,
)
from generate import visualize_joint


# Global state (loaded once)
FLOW_MODEL = None
CONFIG = None
TEXT_ENCODER = None
NORM_MEAN = None
NORM_STD = None
DEVICE = None


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


@torch.no_grad()
def generate_trajectory(text, motion_type, shot_type, guidance_scale):
    if not text.strip():
        return None

    model_cfg = CONFIG['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = CONFIG['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    num_steps = CONFIG['flow_matching']['num_steps']

    # Encode text
    if TEXT_ENCODER:
        text_embed = TEXT_ENCODER([text])
    else:
        text_embed = torch.randn(1, 512, device=DEVICE)

    shot_idx = SHOT_TYPE_MAP.get(shot_type, 1)
    motion_idx = MOTION_TYPE_MAP.get(motion_type, 0)
    shot_t = torch.tensor([shot_idx], device=DEVICE)
    motion_t = torch.tensor([motion_idx], device=DEVICE)

    # Sample
    y = FLOW_MODEL.sample(
        text_embed, shot_type=shot_t, motion_type=motion_t,
        device=DEVICE, guidance_scale=guidance_scale,
        num_steps=num_steps,
    )

    # Denormalize
    if NORM_MEAN is not None:
        y = y * NORM_STD + NORM_MEAN

    y_np = y[0].cpu().numpy()
    person_traj = y_np[:person_total].reshape(num_frames, person_dim)
    camera_traj = y_np[person_total:].reshape(num_frames, camera_dim)

    # Post-process
    person_smooth_window = min(31, person_traj.shape[0] if person_traj.shape[0] % 2 == 1 else person_traj.shape[0] - 1)
    person_traj = smooth_trajectory(person_traj, window=person_smooth_window)
    person_traj = regularize_person_trajectory(person_traj, static_threshold=0.08, segment_cost=0.3)
    person_traj = freeze_static_dims(person_traj, threshold=0.05)
    camera_traj = smooth_trajectory(camera_traj, window=21, angle_dims=[3, 4, 5], angle_window=31)
    camera_traj = freeze_static_dims(camera_traj, threshold=0.05)

    # Visualize to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    visualize_joint(person_traj, camera_traj, text, motion_type, save_path=tmp.name)
    return tmp.name


def build_ui():
    with gr.Blocks(
        title="Text-to-Shot",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        gr.Markdown("# Text-to-Shot\nGenerate cinematographic camera trajectories from natural language.")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="e.g. The camera orbits around the character as they stand still",
                    lines=2,
                )
                motion_dropdown = gr.Dropdown(
                    choices=list(MOTION_TYPE_MAP.keys()),
                    value="static",
                    label="Motion Type",
                )
                shot_dropdown = gr.Dropdown(
                    choices=list(SHOT_TYPE_MAP.keys()),
                    value="medium-shot",
                    label="Shot Type",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                    label="Guidance Scale",
                )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Generated Trajectory", type="filepath")

        # Examples
        gr.Examples(
            examples=[
                ["The camera remains static while the character walks forward", "static", "medium-shot", 3.0],
                ["As the character moves forward, the camera pushes in", "dolly-in", "medium-shot", 3.0],
                ["The camera pulls out as the character stands still", "dolly-out", "medium-shot", 3.0],
                ["The camera pans left as the character walks forward", "pan-left", "medium-shot", 3.0],
                ["The camera orbits around the character as they stand still", "orbit", "medium-shot", 3.0],
                ["The camera cranes up as the character moves forward", "crane-up", "medium-shot", 3.0],
                ["The camera tracks the character as they walk to the right", "track", "medium-shot", 3.0],
            ],
            inputs=[text_input, motion_dropdown, shot_dropdown, guidance_slider],
        )

        generate_btn.click(
            fn=generate_trajectory,
            inputs=[text_input, motion_dropdown, shot_dropdown, guidance_slider],
            outputs=output_image,
        )

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    args = parser.parse_args()

    print("Loading model...")
    init_model(args.checkpoint, args.device if torch.cuda.is_available() else 'cpu')
    print("Model loaded. Starting UI...")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
