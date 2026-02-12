"""
Storyboard Renderer Module.

Converts generated 3D character-camera configurations into 2D storyboard
panels with stick figure visualizations, camera framing, camera motion
trajectory overlays, and annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional, Tuple
from PIL import Image
import io

from .storyboard_generator import GeneratedShot, GeneratedStoryboard
from .camera_trajectory import CameraTrajectory


# SMPL skeleton connectivity (joint index pairs)
# Based on 22-joint SMPL model (excluding hand joints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),     # pelvis → left hip, right hip, spine1
    (1, 4), (2, 5), (3, 6),     # → left knee, right knee, spine2
    (4, 7), (5, 8), (6, 9),     # → left ankle, right ankle, spine3
    (9, 12), (9, 13), (9, 14),  # spine3 → neck, left collar, right collar
    (12, 15),                     # neck → head
    (13, 16), (14, 17),         # → left shoulder, right shoulder
    (16, 18), (17, 19),         # → left elbow, right elbow
    (18, 20), (19, 21),         # → left wrist, right wrist
]

# Camera motion type to arrow/icon mapping for visual annotation
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


class StoryboardRenderer:
    """
    Renders generated shots as 2D storyboard panels.
    
    Each panel shows:
    - Stick figure representations of two characters
    - Camera framing (based on Toric parameters)
    - Camera motion trajectory overlay (arrows and path)
    - Shot annotations (type, description, motion type, index)
    """

    def __init__(
        self,
        panel_width: int = 640,
        panel_height: int = 360,
        char_a_color: str = '#FF6B6B',
        char_b_color: str = '#4ECDC4',
        trajectory_color: str = '#FFE66D',
        bg_color: str = '#2C3E50',
        font_size: int = 10,
    ):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.char_a_color = char_a_color
        self.char_b_color = char_b_color
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
        Render the full storyboard as a grid of panels.
        
        Args:
            storyboard: Generated storyboard with all shots
            cols: Number of columns in the grid
            save_path: Optional path to save the image
            
        Returns:
            PIL Image of the complete storyboard
        """
        num_shots = len(storyboard.shots)
        rows = (num_shots + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 6, rows * 4.5),
            facecolor='#1a1a2e'
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
            self._render_single_panel(ax, shot)

        # Hide unused panels
        for idx in range(num_shots, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        plt.suptitle(
            'AI-Generated Storyboard',
            fontsize=18,
            fontweight='bold',
            color='white',
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            print(f"Storyboard saved to: {save_path}")

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        
        return image

    def _render_single_panel(self, ax: plt.Axes, shot: GeneratedShot):
        """
        Render a single storyboard panel.
        
        Args:
            ax: Matplotlib axes to draw on
            shot: Generated shot data
        """
        ax.set_facecolor(self.bg_color)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')

        # Extract joint positions from pose vectors
        joints_a = self._pose_to_joints_2d(shot.char_a_pose, offset_x=-0.5)
        joints_b = self._pose_to_joints_2d(shot.char_b_pose, offset_x=0.5)

        # Draw stick figures
        self._draw_stick_figure(ax, joints_a, self.char_a_color, label='A')
        self._draw_stick_figure(ax, joints_b, self.char_b_color, label='B')

        # Draw camera motion trajectory overlay
        self._draw_camera_motion_overlay(ax, shot)

        # Draw ground plane
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        # Add panel border
        border = patches.FancyBboxPatch(
            (-2, -0.5), 4, 3,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor='white',
            facecolor='none',
            alpha=0.5
        )
        ax.add_patch(border)

        # Add annotations
        shot_info = shot.shot_config
        motion_icon = MOTION_ICONS.get(shot_info.camera_motion, "")
        title = f"Shot {shot_info.shot_index}: {shot_info.shot_type} {motion_icon}"
        ax.set_title(title, fontsize=self.font_size, fontweight='bold',
                    color='white', pad=8)

        # Add description at bottom
        desc = shot_info.description[:55] + ('...' if len(shot_info.description) > 55 else '')
        ax.text(
            0, -0.35, desc,
            ha='center', va='top',
            fontsize=self.font_size - 2,
            color='lightgray',
            style='italic',
            wrap=True
        )

        # Add camera motion label (bottom-right corner)
        if shot_info.camera_motion != "static":
            ax.text(
                1.85, 2.35,
                f"CAM: {shot_info.camera_motion}",
                ha='right', va='top',
                fontsize=self.font_size - 2,
                fontweight='bold',
                color=self.trajectory_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )

        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_camera_motion_overlay(self, ax: plt.Axes, shot: GeneratedShot):
        """
        Draw camera motion trajectory as visual overlay on the panel.
        
        Uses arrows and path lines to indicate camera movement direction
        and type. If a CameraTrajectory is available, draws the actual
        trajectory path; otherwise uses stylized motion arrows.
        
        Args:
            ax: Matplotlib axes
            shot: Generated shot with trajectory data
        """
        motion = shot.shot_config.camera_motion
        if motion == "static":
            return

        color = self.trajectory_color

        if shot.camera_trajectory is not None:
            # Draw actual trajectory path from trajectory data
            traj = shot.camera_trajectory.trajectory
            # Map Toric theta/phi to 2D overlay coordinates
            # theta (col 4) → x direction, phi (col 5) → y direction
            path_x = np.linspace(-1.7, 1.7, len(traj))
            path_y = 2.2 + traj[:, 5] * 0.5  # phi modulates vertical position
            
            ax.plot(path_x, path_y, color=color, linewidth=1.5, alpha=0.7,
                   linestyle='-', solid_capstyle='round')
            # Arrow at the end
            if len(path_x) >= 2:
                ax.annotate('', xy=(path_x[-1], path_y[-1]),
                           xytext=(path_x[-3], path_y[-3]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            # Stylized motion arrows based on type
            self._draw_motion_arrow(ax, motion, color)

    def _draw_motion_arrow(self, ax: plt.Axes, motion: str, color: str):
        """Draw a stylized arrow indicating camera motion type."""
        center_x, center_y = 0.0, 2.25
        arrow_len = 0.6

        arrow_configs = {
            "dolly-in":   {"dx": 0, "dy": -0.3, "x": 0, "y": 2.35},
            "dolly-out":  {"dx": 0, "dy": 0.2, "x": 0, "y": 2.15},
            "pan-left":   {"dx": -arrow_len, "dy": 0, "x": 0.3, "y": center_y},
            "pan-right":  {"dx": arrow_len, "dy": 0, "x": -0.3, "y": center_y},
            "crane-up":   {"dx": 0, "dy": 0.25, "x": -1.6, "y": 1.0},
            "crane-down": {"dx": 0, "dy": -0.25, "x": -1.6, "y": 1.5},
            "track":      {"dx": arrow_len, "dy": 0, "x": -0.5, "y": center_y},
            "orbit":      None,  # Special handling
        }

        config = arrow_configs.get(motion)

        if motion == "orbit":
            # Draw a curved arrow for orbit
            arc = patches.Arc(
                (center_x, 1.2), 2.8, 1.2,
                angle=0, theta1=20, theta2=160,
                color=color, linewidth=1.5, alpha=0.7,
            )
            ax.add_patch(arc)
            # Arrow tip
            ax.annotate('', xy=(1.2, 1.65), xytext=(1.05, 1.72),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        elif config:
            ax.annotate(
                '', xy=(config["x"] + config["dx"], config["y"] + config["dy"]),
                xytext=(config["x"], config["y"]),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=2,
                    connectionstyle='arc3,rad=0.05'
                )
            )

    def _pose_to_joints_2d(
        self,
        pose_vector,
        offset_x: float = 0.0,
    ) -> np.ndarray:
        """
        Convert SMPL pose vector to simplified 2D joint positions.
        
        This is a simplified visualization - actual implementation would
        use forward kinematics with SMPL model to get 3D joint positions,
        then project to 2D using camera parameters.
        
        Args:
            pose_vector: (150,) pose vector
            offset_x: Horizontal offset for character placement
            
        Returns:
            joints_2d: (22, 2) array of 2D joint positions
        """
        pose_np = pose_vector.cpu().numpy() if hasattr(pose_vector, 'cpu') else np.array(pose_vector)
        
        # Use pose data to add variation to default positions
        variation = pose_np[:22] * 0.1 if len(pose_np) >= 22 else np.zeros(22)
        
        # Default standing pose (simplified 22 joints)
        default_joints = np.array([
            [0.0, 0.9],   # 0: pelvis
            [-0.15, 0.85], # 1: left hip
            [0.15, 0.85],  # 2: right hip
            [0.0, 1.1],    # 3: spine1
            [-0.15, 0.5],  # 4: left knee
            [0.15, 0.5],   # 5: right knee
            [0.0, 1.3],    # 6: spine2
            [-0.15, 0.05], # 7: left ankle
            [0.15, 0.05],  # 8: right ankle
            [0.0, 1.5],    # 9: spine3
            [0.0, 0.0],    # 10: (placeholder)
            [0.0, 0.0],    # 11: (placeholder)
            [0.0, 1.7],    # 12: neck
            [-0.1, 1.5],   # 13: left collar
            [0.1, 1.5],    # 14: right collar
            [0.0, 1.85],   # 15: head
            [-0.25, 1.45], # 16: left shoulder
            [0.25, 1.45],  # 17: right shoulder
            [-0.4, 1.15],  # 18: left elbow
            [0.4, 1.15],   # 19: right elbow
            [-0.45, 0.85], # 20: left wrist
            [0.45, 0.85],  # 21: right wrist
        ])

        # Add variation and offset
        joints = default_joints.copy()
        for i in range(min(22, len(variation))):
            joints[i, 0] += variation[i] * 0.3
            joints[i, 1] += abs(variation[i]) * 0.1
        joints[:, 0] += offset_x

        return joints

    def _draw_stick_figure(
        self,
        ax: plt.Axes,
        joints: np.ndarray,
        color: str,
        label: str = '',
    ):
        """
        Draw a stick figure on the axes.
        
        Args:
            ax: Matplotlib axes
            joints: (22, 2) joint positions
            color: Color for the figure
            label: Character label
        """
        # Draw bones
        for start, end in SKELETON_CONNECTIONS:
            if start < len(joints) and end < len(joints):
                ax.plot(
                    [joints[start, 0], joints[end, 0]],
                    [joints[start, 1], joints[end, 1]],
                    color=color, linewidth=2, alpha=0.9,
                    solid_capstyle='round'
                )

        # Draw joints
        for i, (x, y) in enumerate(joints):
            if i in [10, 11]:  # Skip placeholder joints
                continue
            size = 6 if i == 15 else 3  # Larger for head
            ax.plot(x, y, 'o', color=color, markersize=size, alpha=0.9)

        # Draw head circle
        head = plt.Circle(
            (joints[15, 0], joints[15, 1]), 0.08,
            color=color, fill=True, alpha=0.7
        )
        ax.add_patch(head)

        # Add label
        if label:
            ax.text(
                joints[15, 0], joints[15, 1] + 0.15, label,
                ha='center', va='bottom',
                fontsize=self.font_size - 1,
                fontweight='bold',
                color=color,
            )

    def render_single_shot(
        self,
        shot: GeneratedShot,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """Render a single shot as a standalone panel."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='#1a1a2e')
        self._render_single_panel(ax, shot)
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

    def render_trajectory_visualization(
        self,
        shot: GeneratedShot,
        save_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Render a detailed 3D visualization of the camera trajectory.
        
        Shows the Toric parameter evolution over time for a single shot.
        
        Args:
            shot: Generated shot with camera trajectory
            save_path: Optional save path
            
        Returns:
            PIL Image of trajectory visualization, or None if no trajectory
        """
        if shot.camera_trajectory is None:
            print("No camera trajectory available for this shot.")
            return None

        traj = shot.camera_trajectory
        t = traj.timestamps

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#1a1a2e')
        fig.suptitle(
            f'Camera Trajectory: {traj.motion_type} ({traj.num_frames} frames)',
            fontsize=14, fontweight='bold', color='white'
        )

        param_names = ['pA_x', 'pA_y', 'pB_x', 'pB_y', 'θ (theta)', 'φ (phi)']
        colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#FFE66D', '#FFE66D']

        for i, (ax, name, color) in enumerate(zip(axes.flat, param_names, colors)):
            ax.set_facecolor('#2C3E50')
            ax.plot(t, traj.trajectory[:, i], color=color, linewidth=2)
            
            # Mark keyframes
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
