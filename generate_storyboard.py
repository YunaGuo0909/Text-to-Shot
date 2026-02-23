"""
Camera Trajectory Generation & Inference Script.

Modes:
  --demo          Rule-based demo (no model needed)
  --scene + --checkpoint   Generate from text using trained model

Usage:
    python generate_storyboard.py --demo
    python generate_storyboard.py --scene "A tense confrontation in an alley" --checkpoint checkpoints/checkpoint_final.pth
    python generate_storyboard.py --scene "Camera slowly tracks a person walking" --checkpoint checkpoints/checkpoint_final.pth --motion dolly-in
"""

import argparse
import os
import torch
import yaml
import numpy as np

from src.pipeline.shot_decomposer import ShotConfig, StoryboardPlan, CAMERA_MOTION_MAP, SHOT_TYPE_MAP
from src.pipeline.storyboard_generator import GeneratedShot, GeneratedStoryboard, TrajectoryPipeline
from src.pipeline.storyboard_renderer import TrajectoryRenderer
from src.pipeline.camera_trajectory import CameraTrajectory, CameraTrajectoryGenerator


def load_model(checkpoint_path, device='cuda'):
    """Load trained diffusion model from checkpoint."""
    from src.models.denoiser import CameraTrajectoryDenoiser
    from src.models.diffusion import GaussianDiffusion

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model_cfg = config['model']
    traj_cfg = config['trajectory']

    denoiser = CameraTrajectoryDenoiser(
        toric_dim=model_cfg['toric_dim'],
        num_frames=traj_cfg['default_num_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512,
        timestep_dim=128,
        num_shot_types=len(config['shot_types']['categories']),
        shot_type_dim=config['shot_types']['embedding_dim'],
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=0.0,
    ).to(device)

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    diffusion.load_state_dict(ckpt['model_state_dict'])
    diffusion.eval()

    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f})")
    return diffusion, config


def load_text_encoder(config, device='cuda'):
    """Load CLIP text encoder."""
    from src.models.text_encoder import CLIPTextEncoder
    model_name = config['text_encoder']['model_name']
    print(f"Loading CLIP text encoder: {model_name}")
    encoder = CLIPTextEncoder(model_name=model_name, device=device)
    encoder = encoder.to(device)
    return encoder


@torch.no_grad()
def generate_from_text(
    text: str,
    diffusion,
    text_encoder,
    device='cuda',
    shot_type='medium-shot',
    camera_motion='static',
    num_samples=1,
):
    """
    Generate camera trajectory from a text description.

    Args:
        text: Scene/camera description
        diffusion: Trained diffusion model
        text_encoder: CLIP text encoder
        device: Computation device
        shot_type: Shot type string
        camera_motion: Camera motion type string
        num_samples: Number of trajectories to generate

    Returns:
        List of CameraTrajectory objects
    """
    toric_dim = diffusion.denoiser.toric_dim
    num_frames = diffusion.denoiser.num_frames

    # Encode text
    text_embed = text_encoder([text] * num_samples)

    # Encode conditions
    shot_idx = SHOT_TYPE_MAP.get(shot_type, 1)
    motion_idx = CAMERA_MOTION_MAP.get(camera_motion, 0)
    shot_type_t = torch.tensor([shot_idx] * num_samples, device=device)
    motion_type_t = torch.tensor([motion_idx] * num_samples, device=device)

    # Generate
    y_0 = diffusion.sample(
        text_embed=text_embed,
        shot_type=shot_type_t,
        motion_type=motion_type_t,
        device=device,
    )

    # Convert to CameraTrajectory objects
    trajectories = []
    for i in range(num_samples):
        traj_data = y_0[i].cpu().numpy().reshape(num_frames, toric_dim)
        traj = CameraTrajectory(
            motion_type=camera_motion,
            num_frames=num_frames,
            keyframes=traj_data[np.linspace(0, num_frames - 1, 4, dtype=int)],
            trajectory=traj_data,
            timestamps=np.linspace(0, 1, num_frames),
        )
        trajectories.append(traj)

    return trajectories


def inference_single(args):
    """Generate trajectory for a single text description."""
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load model
    diffusion, config = load_model(args.checkpoint, device)
    text_encoder = load_text_encoder(config, device)

    # Generate
    print(f"\nGenerating trajectory for: \"{args.scene}\"")
    print(f"  Shot type: {args.shot_type}")
    print(f"  Camera motion: {args.motion}")

    trajectories = generate_from_text(
        text=args.scene,
        diffusion=diffusion,
        text_encoder=text_encoder,
        device=device,
        shot_type=args.shot_type,
        camera_motion=args.motion,
        num_samples=args.num_samples,
    )

    # Visualize
    renderer = TrajectoryRenderer()
    os.makedirs(os.path.dirname(args.output) or 'outputs', exist_ok=True)

    # Build GeneratedShot for rendering
    shots = []
    for i, traj in enumerate(trajectories):
        shot = GeneratedShot(
            shot_config=ShotConfig(
                shot_index=i + 1,
                description=args.scene,
                shot_type=args.shot_type,
                camera_motion=args.motion,
            ),
            camera_trajectory=traj,
            toric_start=traj.trajectory[0],
            toric_end=traj.trajectory[-1],
        )
        shots.append(shot)

    plan = StoryboardPlan(scene_description=args.scene, shots=[s.shot_config for s in shots], total_shots=len(shots))
    storyboard = GeneratedStoryboard(plan=plan, shots=shots)

    # Save visualizations
    base_path = args.output.rsplit('.', 1)[0]

    renderer.render_storyboard(storyboard, cols=min(len(shots), 3),
                               save_path=f"{base_path}_grid.png")
    renderer.render_trajectory_detail(shots[0],
                                      save_path=f"{base_path}_detail.png")
    if len(shots) > 1:
        renderer.render_camera_path_topdown(storyboard,
                                            save_path=f"{base_path}_path.png")

    # Print metrics
    print(f"\nGenerated {len(trajectories)} trajectory(ies):")
    for i, traj in enumerate(trajectories):
        sm = CameraTrajectoryGenerator.compute_trajectory_smoothness(traj.trajectory)
        print(f"  [{i+1}] frames={traj.num_frames}  jerk={sm['mean_jerk']:.4f}  "
              f"path_len={sm['total_path_length']:.3f}")

    print(f"\nOutputs saved to: {base_path}_*.png")

    # 3D animation
    from visualize_3d import create_trajectory_animation, create_static_3d_view
    traj_data = trajectories[0].trajectory
    anim_title = f"{args.motion} | \"{args.scene[:50]}\""
    create_trajectory_animation(traj_data, title=anim_title,
                                save_path=f"{base_path}_3d.gif")
    create_static_3d_view(traj_data, title=anim_title,
                          save_path=f"{base_path}_3d_static.png")


def demo_with_mock_data():
    """Demo trajectory generation with rule-based motion profiles."""
    print("=" * 60)
    print("Script-to-Camera: Trajectory Generation Demo")
    print("Generating Cinematic Camera Trajectories from Text")
    print("=" * 60)

    plan = StoryboardPlan(
        scene_description="Two people meet at a cafe. They greet each other, shake hands, and sit down for a conversation.",
        shots=[
            ShotConfig(shot_index=1, description="Establishing wide shot of the cafe exterior, camera descends to entrance level.",
                       shot_type="wide-shot", camera_motion="crane-down", duration_hint=4.0, emotional_tone="calm"),
            ShotConfig(shot_index=2, description="Medium shot panning to follow a person approaching the cafe.",
                       shot_type="medium-shot", camera_motion="pan-right", duration_hint=3.0, emotional_tone="anticipation"),
            ShotConfig(shot_index=3, description="Tracking shot following the character as they walk inside.",
                       shot_type="medium-shot", camera_motion="track", duration_hint=3.5, emotional_tone="movement"),
            ShotConfig(shot_index=4, description="Dolly in to a close-up of the handshake between the two people.",
                       shot_type="close-up", camera_motion="dolly-in", duration_hint=2.5, emotional_tone="intimate"),
            ShotConfig(shot_index=5, description="Two-shot of both people sitting down at a table.",
                       shot_type="two-shot", camera_motion="static", duration_hint=3.0, emotional_tone="settled"),
            ShotConfig(shot_index=6, description="Slow orbit around the table as conversation begins.",
                       shot_type="medium-shot", camera_motion="orbit", duration_hint=5.0, emotional_tone="engaging"),
        ],
        total_shots=6,
    )

    pipeline = TrajectoryPipeline(diffusion_model=None, text_encoder=None, device='cpu')
    storyboard = pipeline.generate(plan, mode="rule_based", smooth_transitions=True)

    renderer = TrajectoryRenderer()
    os.makedirs('outputs', exist_ok=True)

    print("\nRendering trajectory storyboard grid...")
    renderer.render_storyboard(storyboard, cols=3, save_path='outputs/demo_trajectory_storyboard.png')

    print("Rendering detailed trajectory curves for Shot 4 (dolly-in)...")
    renderer.render_trajectory_detail(storyboard.shots[3], save_path='outputs/demo_trajectory_detail.png')

    print("Rendering top-down camera path...")
    renderer.render_camera_path_topdown(storyboard, save_path='outputs/demo_camera_path.png')

    print("\n" + "=" * 60)
    print(f"Generated trajectories for {len(storyboard.shots)} shots")
    for shot in storyboard.shots:
        sm = CameraTrajectoryGenerator.compute_trajectory_smoothness(shot.camera_trajectory.trajectory)
        print(f"  Shot {shot.shot_config.shot_index}: [{shot.shot_config.shot_type:18s}] "
              f"[CAM: {shot.shot_config.camera_motion:10s}] "
              f"frames={shot.camera_trajectory.num_frames:3d}  jerk={sm['mean_jerk']:.4f}")

    print(f"\nOutputs saved to: outputs/demo_*.png")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Script-to-Camera: Generate Camera Trajectories')
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene/camera description text')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/generated.png',
                        help='Output image base path')
    parser.add_argument('--shot-type', type=str, default='medium-shot',
                        choices=list(SHOT_TYPE_MAP.keys()),
                        help='Shot type')
    parser.add_argument('--motion', type=str, default='static',
                        choices=list(CAMERA_MOTION_MAP.keys()),
                        help='Camera motion type')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of trajectories to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with rule-based trajectories')
    args = parser.parse_args()

    if args.demo:
        demo_with_mock_data()
    elif args.scene and args.checkpoint:
        inference_single(args)
    elif args.scene:
        print("Error: --checkpoint is required for generation.")
        print("Use --demo for rule-based demo without a trained model.")
    else:
        print("Script-to-Camera: Cinematic Camera Trajectory Generation")
        print("\nUsage:")
        print("  Demo mode:       python generate_storyboard.py --demo")
        print("  Generate:        python generate_storyboard.py --scene 'Camera dollies in slowly' --checkpoint checkpoints/checkpoint_final.pth")
        print("  With options:    python generate_storyboard.py --scene '...' --checkpoint ... --motion dolly-in --shot-type close-up")


if __name__ == '__main__':
    main()
