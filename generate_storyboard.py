"""
Camera Trajectory Generation Demo Script.

Generates multi-shot camera motion trajectories from scene descriptions
and visualizes them as trajectory plots and camera path diagrams.

Usage:
    python generate_storyboard.py --demo
    python generate_storyboard.py --scene "A tense confrontation in an alley" --output outputs/trajectory.png
"""

import argparse
import os
import numpy as np

from src.pipeline.shot_decomposer import ShotDecomposer, ShotConfig, StoryboardPlan
from src.pipeline.storyboard_generator import GeneratedShot, GeneratedStoryboard, TrajectoryPipeline
from src.pipeline.storyboard_renderer import TrajectoryRenderer
from src.pipeline.camera_trajectory import CameraTrajectoryGenerator


def demo_with_mock_data():
    """
    Demo trajectory generation with rule-based motion profiles.
    Tests the full visualization pipeline without a trained model.
    """
    print("=" * 60)
    print("Script-to-Camera: Trajectory Generation Demo")
    print("Generating Cinematic Camera Trajectories from Text")
    print("=" * 60)

    # Create manual shot plan
    plan = StoryboardPlan(
        scene_description="Two people meet at a cafe. They greet each other, shake hands, and sit down for a conversation.",
        shots=[
            ShotConfig(
                shot_index=1,
                description="Establishing wide shot of the cafe exterior, camera descends to entrance level.",
                shot_type="wide-shot",
                camera_motion="crane-down",
                duration_hint=4.0,
                emotional_tone="calm",
            ),
            ShotConfig(
                shot_index=2,
                description="Medium shot panning to follow a person approaching the cafe.",
                shot_type="medium-shot",
                camera_motion="pan-right",
                duration_hint=3.0,
                emotional_tone="anticipation",
            ),
            ShotConfig(
                shot_index=3,
                description="Tracking shot following the character as they walk inside.",
                shot_type="medium-shot",
                camera_motion="track",
                duration_hint=3.5,
                emotional_tone="movement",
            ),
            ShotConfig(
                shot_index=4,
                description="Dolly in to a close-up of the handshake between the two people.",
                shot_type="close-up",
                camera_motion="dolly-in",
                duration_hint=2.5,
                emotional_tone="intimate",
            ),
            ShotConfig(
                shot_index=5,
                description="Two-shot of both people sitting down at a table.",
                shot_type="two-shot",
                camera_motion="static",
                duration_hint=3.0,
                emotional_tone="settled",
            ),
            ShotConfig(
                shot_index=6,
                description="Slow orbit around the table as conversation begins.",
                shot_type="medium-shot",
                camera_motion="orbit",
                duration_hint=5.0,
                emotional_tone="engaging",
            ),
        ],
        total_shots=6,
    )

    # Generate trajectories using rule-based pipeline
    pipeline = TrajectoryPipeline(diffusion_model=None, text_encoder=None, device='cpu')
    storyboard = pipeline.generate(plan, mode="rule_based", smooth_transitions=True)

    # Render visualizations
    renderer = TrajectoryRenderer()

    os.makedirs('outputs', exist_ok=True)

    # 1. Multi-shot storyboard grid with trajectory curves
    print("\nRendering trajectory storyboard grid...")
    renderer.render_storyboard(
        storyboard, cols=3,
        save_path='outputs/demo_trajectory_storyboard.png',
    )

    # 2. Detailed parameter curves for a specific shot (dolly-in)
    print("Rendering detailed trajectory curves for Shot 4 (dolly-in)...")
    renderer.render_trajectory_detail(
        storyboard.shots[3],
        save_path='outputs/demo_trajectory_detail.png',
    )

    # 3. Top-down camera path view
    print("Rendering top-down camera path...")
    renderer.render_camera_path_topdown(
        storyboard,
        save_path='outputs/demo_camera_path.png',
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Generated trajectories for {len(storyboard.shots)} shots")
    print(f"Scene: {plan.scene_description}")
    print(f"\nShot Summary:")

    for shot in storyboard.shots:
        smoothness = CameraTrajectoryGenerator.compute_trajectory_smoothness(
            shot.camera_trajectory.trajectory
        )
        print(f"  Shot {shot.shot_config.shot_index}: "
              f"[{shot.shot_config.shot_type:18s}] "
              f"[CAM: {shot.shot_config.camera_motion:10s}] "
              f"frames={shot.camera_trajectory.num_frames:3d}  "
              f"jerk={smoothness['mean_jerk']:.4f}  "
              f"path_len={smoothness['total_path_length']:.3f}")

    print(f"\nOutputs saved to:")
    print(f"  - outputs/demo_trajectory_storyboard.png  (multi-shot grid)")
    print(f"  - outputs/demo_trajectory_detail.png      (parameter curves)")
    print(f"  - outputs/demo_camera_path.png            (top-down path)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Script-to-Camera: Generate Camera Trajectories')
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene description text')
    parser.add_argument('--output', type=str, default='outputs/trajectory.png',
                        help='Output image path')
    parser.add_argument('--max-shots', type=int, default=6,
                        help='Maximum number of shots')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with rule-based trajectories')
    parser.add_argument('--cols', type=int, default=3,
                        help='Number of columns in storyboard grid')
    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else 'outputs',
        exist_ok=True,
    )

    if args.demo:
        demo_with_mock_data()
    elif args.scene:
        print("Full pipeline requires a trained model checkpoint.")
        print("Use --demo flag for a demonstration with rule-based trajectories.")
        print("Run: python generate_storyboard.py --demo")
    else:
        print("Please provide --scene or use --demo flag.")
        print("Example: python generate_storyboard.py --demo")
        print("Example: python generate_storyboard.py --scene 'A tense confrontation in an alley'")


if __name__ == '__main__':
    main()
