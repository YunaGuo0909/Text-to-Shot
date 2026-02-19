"""
Camera Trajectory Visualization Module.

Renders generated camera trajectories as visual panels, including:
- Toric parameter evolution curves
- Top-down camera path visualization
- Multi-shot trajectory storyboard grid
- Trajectory smoothness metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple
from PIL import Image
import io

from .storyboard_generator import GeneratedShot, GeneratedStoryboard
from .camera_trajectory import CameraTrajectory, CameraTrajectoryGenerator


# Camera motion type to icon mapping
MOTION_ICONS = {
    "static":     "●",
    "dolly-in":   "⟶",
    "dolly-out":  "⟵",
    "pan-left":   "↶",
    "pan-right":  "↷",
    "crane-up":   "↑",
    "crane-down": "↓",
    "track":      "⇢",
    "orbit":      "↻",
}

# Toric parameter names
TORIC_PARAM_NAMES = ['pA_x', 'pA_y', 'pB_x', 'pB_y', 'θ (theta)', 'φ (phi)']
TORIC_PARAM_COLORS = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#FFE66D', '#FFE66D']


class TrajectoryRenderer:
    """
    Renders camera trajectory visualizations.

    Provides multiple views:
    1. Storyboard grid: multi-shot overview with trajectory mini-plots
    2. Parameter curves: per-shot Toric parameter evolution
    3. Camera path: top-down 2D visualization of camera motion
    """

    def __init__(
        self,
        trajectory_color: str = '#FFE66D',
        bg_color: str = '#2C3E50',
        font_size: int = 10,
    ):
        self.trajectory_color = trajectory_color
        self.bg_color = bg_color
        self.font_size = font_size

    def render_storyboard(
        self,
        storyboard: GeneratedStoryboard,
        cols: int = 3,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Render multi-shot trajectory storyboard as a grid.

        Each panel shows:
        - Camera motion type badge
        - Mini trajectory plot (theta and phi over time)
        - Shot type and description
        - Smoothness metrics
        """
        num_shots = len(storyboard.shots)
        rows = (num_shots + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 6, rows * 4),
            facecolor='#1a1a2e',
        )

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx, shot in enumerate(storyboard.shots):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            self._render_trajectory_panel(ax, shot)

        for idx in range(num_shots, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        plt.suptitle(
            'Script-to-Camera: Generated Trajectories',
            fontsize=18, fontweight='bold', color='white', y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"Storyboard saved to: {save_path}")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image

    def _render_trajectory_panel(self, ax: plt.Axes, shot: GeneratedShot):
        """Render a single shot's trajectory as a panel."""
        ax.set_facecolor(self.bg_color)

        traj = shot.camera_trajectory
        t = traj.timestamps

        # Plot theta (azimuth) and phi (elevation) curves
        ax.plot(t, traj.trajectory[:, 4], color='#FFE66D', linewidth=2,
                label='θ (azimuth)', alpha=0.9)
        ax.plot(t, traj.trajectory[:, 5], color='#FF6B6B', linewidth=2,
                label='φ (elevation)', alpha=0.9)

        # Plot screen positions (pA, pB)
        ax.plot(t, traj.trajectory[:, 0], color='#4ECDC4', linewidth=1,
                label='pA_x', alpha=0.5, linestyle='--')
        ax.plot(t, traj.trajectory[:, 2], color='#C44ECD', linewidth=1,
                label='pB_x', alpha=0.5, linestyle='--')

        # Mark keyframes
        t_key = np.linspace(0, 1, len(traj.keyframes))
        ax.scatter(t_key, traj.keyframes[:, 4], color='white', s=20,
                   zorder=5, edgecolors='#FFE66D', linewidths=1)
        ax.scatter(t_key, traj.keyframes[:, 5], color='white', s=20,
                   zorder=5, edgecolors='#FF6B6B', linewidths=1)

        # Styling
        ax.set_xlabel('time', color='gray', fontsize=8)
        ax.tick_params(colors='gray', labelsize=7)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.3,
                  labelcolor='white')

        # Title with motion badge
        shot_info = shot.shot_config
        motion_icon = MOTION_ICONS.get(shot_info.camera_motion, "")
        title = f"Shot {shot_info.shot_index}: {shot_info.shot_type} | {motion_icon} {shot_info.camera_motion}"
        ax.set_title(title, fontsize=self.font_size, fontweight='bold',
                     color='white', pad=8)

        # Description at bottom (inside axes)
        desc = shot_info.description[:60] + ('...' if len(shot_info.description) > 60 else '')
        ax.text(0.5, -0.12, desc, ha='center', va='top',
                fontsize=self.font_size - 2, color='lightgray', style='italic',
                transform=ax.transAxes)

        # Smoothness metric badge
        smoothness = CameraTrajectoryGenerator.compute_trajectory_smoothness(traj.trajectory)
        ax.text(0.02, 0.95, f"jerk={smoothness['mean_jerk']:.4f}",
                ha='left', va='top', fontsize=7, color='#95E66D',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    def render_trajectory_detail(
        self,
        shot: GeneratedShot,
        save_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Render detailed Toric parameter curves for a single shot.

        Shows all 6 Toric parameters evolving over time with keyframe markers.
        """
        if shot.camera_trajectory is None:
            return None

        traj = shot.camera_trajectory
        t = traj.timestamps

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#1a1a2e')
        fig.suptitle(
            f'Camera Trajectory: {traj.motion_type} ({traj.num_frames} frames)',
            fontsize=14, fontweight='bold', color='white',
        )

        for i, (ax, name, color) in enumerate(
            zip(axes.flat, TORIC_PARAM_NAMES, TORIC_PARAM_COLORS)
        ):
            ax.set_facecolor('#2C3E50')
            ax.plot(t, traj.trajectory[:, i], color=color, linewidth=2)

            t_key = np.linspace(0, 1, len(traj.keyframes))
            ax.scatter(t_key, traj.keyframes[:, i], color='white',
                       s=30, zorder=5, edgecolors=color)

            ax.set_title(name, color='white', fontsize=11)
            ax.set_xlabel('time', color='gray', fontsize=9)
            ax.tick_params(colors='gray')
            ax.grid(alpha=0.15)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image

    def render_camera_path_topdown(
        self,
        storyboard: GeneratedStoryboard,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Render top-down 2D camera path for all shots.

        Maps theta (azimuth) to X and phi (elevation) to Y to create
        a bird's-eye view of camera movement across the shot sequence.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#2C3E50')

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(storyboard.shots)))

        for idx, shot in enumerate(storyboard.shots):
            traj = shot.camera_trajectory.trajectory
            theta = traj[:, 4]
            phi = traj[:, 5]

            # Draw path
            ax.plot(theta, phi, color=colors[idx], linewidth=2, alpha=0.8,
                    label=f"Shot {shot.shot_config.shot_index}: {shot.shot_config.camera_motion}")

            # Start marker
            ax.scatter(theta[0], phi[0], color=colors[idx], s=80,
                       marker='o', zorder=5, edgecolors='white', linewidths=1.5)
            # End marker (arrow)
            if len(theta) >= 2:
                ax.annotate('', xy=(theta[-1], phi[-1]),
                            xytext=(theta[-3], phi[-3]),
                            arrowprops=dict(arrowstyle='->', color=colors[idx], lw=2))

            # Shot number label
            ax.text(theta[0], phi[0] + 0.02, f"S{shot.shot_config.shot_index}",
                    color='white', fontsize=9, fontweight='bold', ha='center')

        ax.set_xlabel('θ (Azimuth)', color='white', fontsize=12)
        ax.set_ylabel('φ (Elevation)', color='white', fontsize=12)
        ax.set_title('Camera Path — Top-down View (Toric Space)',
                     color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='gray')
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.3, labelcolor='white')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return image
