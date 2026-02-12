"""
Storyboard Generation Demo Script.

Takes a scene description and generates a multi-shot storyboard
with camera motion trajectories using the trained diffusion model.

Usage:
    python generate_storyboard.py --demo
    python generate_storyboard.py --scene "Two people meet at a park..." --output storyboard.png
"""

import argparse
import os
import torch
import numpy as np

from src.pipeline.shot_decomposer import ShotDecomposer, ShotConfig
from src.pipeline.storyboard_generator import GeneratedShot, GeneratedStoryboard
from src.pipeline.storyboard_renderer import StoryboardRenderer
from src.pipeline.shot_decomposer import StoryboardPlan
from src.pipeline.camera_trajectory import CameraTrajectoryGenerator


def demo_with_mock_data():
    """
    Demo storyboard generation with mock data and camera trajectories.
    Use this to test the visualization pipeline without a trained model.
    """
    print("=" * 60)
    print("AI-Driven Storyboard Generation Demo")
    print("with Camera Motion Trajectories")
    print("=" * 60)

    # Create manual shot plan with camera motion types
    plan = StoryboardPlan(
        scene_description="Two people meet at a cafe. Person A waves and walks toward Person B. They shake hands and sit down together.",
        shots=[
            ShotConfig(
                shot_index=1,
                description="Wide shot establishing the cafe scene. Person A enters from the left.",
                shot_type="wide-shot",
                camera_motion="crane-down",
                character_a_action="Walking into frame",
                character_b_action="Sitting at table",
            ),
            ShotConfig(
                shot_index=2,
                description="Medium shot of Person A noticing Person B and waving.",
                shot_type="medium-shot",
                camera_motion="pan-right",
                character_a_action="Waving hand",
                character_b_action="Looking up",
            ),
            ShotConfig(
                shot_index=3,
                description="Tracking shot following A as they walk toward B.",
                shot_type="medium-shot",
                camera_motion="track",
                character_a_action="Walking forward",
                character_b_action="Turning to face A",
            ),
            ShotConfig(
                shot_index=4,
                description="Dolly in to close-up of their handshake.",
                shot_type="close-up",
                camera_motion="dolly-in",
                character_a_action="Extending hand",
                character_b_action="Shaking hand",
            ),
            ShotConfig(
                shot_index=5,
                description="Two-shot of both characters sitting down together.",
                shot_type="two-shot",
                camera_motion="static",
                character_a_action="Sitting down",
                character_b_action="Gesturing to seat",
            ),
            ShotConfig(
                shot_index=6,
                description="Orbit shot as they start a conversation.",
                shot_type="medium-shot",
                camera_motion="orbit",
                character_a_action="Talking, leaning forward",
                character_b_action="Listening, nodding",
            ),
        ],
        total_shots=6,
    )

    # Initialize camera trajectory generator
    traj_generator = CameraTrajectoryGenerator(default_num_frames=48)

    # Generate mock 3D data + camera trajectories for each shot
    mock_shots = []
    for shot_config in plan.shots:
        # Random pose data (in actual use, this comes from the diffusion model)
        char_a_pose = torch.randn(150) * 0.1
        char_b_pose = torch.randn(150) * 0.1
        camera_state = torch.randn(6) * 0.1

        # Generate camera trajectory from motion type
        trajectory = traj_generator.generate_from_torch(
            camera_state,
            motion_type=shot_config.camera_motion,
            num_frames=int(shot_config.duration_hint * 24),  # 24 fps
            intensity=1.0,
        )

        mock_shots.append(GeneratedShot(
            shot_config=shot_config,
            char_a_pose=char_a_pose,
            char_b_pose=char_b_pose,
            camera_state=camera_state,
            camera_trajectory=trajectory,
        ))

    storyboard = GeneratedStoryboard(plan=plan, shots=mock_shots)

    # Render storyboard
    renderer = StoryboardRenderer()
    
    os.makedirs('outputs', exist_ok=True)
    
    image = renderer.render_storyboard(
        storyboard,
        cols=3,
        save_path='outputs/demo_storyboard.png'
    )

    # Render trajectory details for a specific shot (e.g., dolly-in)
    print("\nRendering trajectory visualization for Shot 4 (dolly-in)...")
    renderer.render_trajectory_visualization(
        mock_shots[3],  # dolly-in shot
        save_path='outputs/demo_trajectory_dolly_in.png'
    )

    print("\n" + "=" * 60)
    print(f"Generated storyboard with {len(mock_shots)} shots")
    print(f"Scene: {plan.scene_description}")
    print(f"\nShots:")
    for shot in plan.shots:
        traj = [s for s in mock_shots if s.shot_config.shot_index == shot.shot_index][0]
        smoothness = CameraTrajectoryGenerator.compute_trajectory_smoothness(
            traj.camera_trajectory.trajectory
        )
        print(f"  {shot.shot_index}. [{shot.shot_type:18s}] "
              f"[CAM: {shot.camera_motion:10s}] "
              f"jerk={smoothness['mean_jerk']:.4f}  "
              f"{shot.description[:40]}...")
    
    print(f"\nOutputs saved to:")
    print(f"  - outputs/demo_storyboard.png")
    print(f"  - outputs/demo_trajectory_dolly_in.png")
    print("=" * 60)

    return image


def main():
    parser = argparse.ArgumentParser(description='Generate AI Storyboard')
    parser.add_argument('--scene', type=str, default=None,
                       help='Scene description text')
    parser.add_argument('--output', type=str, default='outputs/storyboard.png',
                       help='Output image path')
    parser.add_argument('--max-shots', type=int, default=6,
                       help='Maximum number of shots')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with mock data')
    parser.add_argument('--cols', type=int, default=3,
                       help='Number of columns in storyboard grid')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'outputs', exist_ok=True)

    if args.demo:
        demo_with_mock_data()
    elif args.scene:
        print("Full pipeline requires a trained model checkpoint.")
        print("Use --demo flag for a demonstration with mock data.")
        print("Run: python generate_storyboard.py --demo")
    else:
        print("Please provide --scene or use --demo flag.")
        print("Example: python generate_storyboard.py --demo")
        print("Example: python generate_storyboard.py --scene 'Two people argue in an office'")


if __name__ == '__main__':
    main()
