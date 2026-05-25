"""
Script 05: Speckle Tracking & Motion Analysis
===============================================
Demonstrates speckle tracking and motion analysis in ultrasound:
- Block matching between consecutive frames
- Displacement vector field visualization
- Strain estimation from displacement fields
- Tissue motion quantification

Equivalent to FAST's BlockMatching process object with various
matching metrics (SAD, SSD, NCC).

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization

Usage:
    python 05_speckle_tracking.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import generate_cardiac_phantom
from utils.image_utils import nlm_fast, create_ultrasound_colormap, block_matching

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_strain(displacement_field):
    """
    Compute strain from a displacement field using finite differences.

    Parameters
    ----------
    displacement_field : np.ndarray
        Shape (H, W, 2), where channel 0 is dy and channel 1 is dx.

    Returns
    -------
    dict
        Dictionary with strain components:
        - 'radial': strain in the vertical (depth) direction
        - 'lateral': strain in the horizontal direction
        - 'shear': shear strain
        - 'principal': principal strain magnitude
    """
    dy = displacement_field[..., 0]
    dx = displacement_field[..., 1]

    # Strain = gradient of displacement
    e_yy = np.gradient(dy, axis=0)  # Radial strain
    e_xx = np.gradient(dx, axis=1)  # Lateral strain
    e_xy = 0.5 * (np.gradient(dy, axis=1) + np.gradient(dx, axis=0))  # Shear

    # Principal strain
    principal = np.sqrt(((e_yy - e_xx) / 2) ** 2 + e_xy ** 2)

    return {
        'radial': e_yy,
        'lateral': e_xx,
        'shear': e_xy,
        'principal': principal
    }


def demo_block_matching():
    """
    Demo: Block matching speckle tracking between cardiac frames.

    Equivalent to FAST's BlockMatching with SAD metric.
    """
    print("\n" + "=" * 60)
    print("  5.1 — Block Matching Speckle Tracking")
    print("=" * 60)

    # Generate cardiac sequence
    cardiac = generate_cardiac_phantom(256, 256, n_frames=20, seed=42)

    # Select two consecutive frames (mid-systole transition)
    frame_idx1 = 3
    frame_idx2 = 5
    frame1 = cardiac[frame_idx1]
    frame2 = cardiac[frame_idx2]

    print(f"  Tracking motion between frames {frame_idx1 + 1} and {frame_idx2 + 1}...")
    print("  Computing block matching (this may take a moment)...")

    # Perform block matching
    displacement = block_matching(frame1, frame2, block_size=13, search_size=11, metric='sad')

    # Compute magnitude of displacement
    disp_magnitude = np.sqrt(displacement[..., 0] ** 2 + displacement[..., 1] ** 2)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    us_cmap = create_ultrasound_colormap()

    # Frame 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame1, cmap=us_cmap, vmin=0, vmax=1)
    ax1.set_title(f'Frame {frame_idx1 + 1} (Reference)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Frame 2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(frame2, cmap=us_cmap, vmin=0, vmax=1)
    ax2.set_title(f'Frame {frame_idx2 + 1} (Target)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Difference image
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(frame1 - frame2)
    ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax3.set_title('Frame Difference', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Displacement vector field (quiver plot)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(frame1, cmap='gray', vmin=0, vmax=1, alpha=0.7)
    step = 8
    h, w = frame1.shape
    Y, X = np.mgrid[0:h:step, 0:w:step]

    # Subsample displacement field
    dy = displacement[::step, ::step, 0]
    dx = displacement[::step, ::step, 1]

    # Ensure shapes match
    min_h = min(Y.shape[0], dy.shape[0])
    min_w = min(Y.shape[1], dy.shape[1])
    ax4.quiver(X[:min_h, :min_w], Y[:min_h, :min_w],
               dx[:min_h, :min_w], dy[:min_h, :min_w],
               color='#00E5FF', scale=step * 12, width=0.003, headwidth=4)
    ax4.set_title('Displacement Vectors', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Displacement magnitude
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(disp_magnitude, cmap='magma', vmin=0)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label='Pixels')
    ax5.set_title('Displacement Magnitude', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # Displacement components
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(frame1, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    # Color-code vectors by direction
    magnitude = np.sqrt(dx[:min_h, :min_w] ** 2 + dy[:min_h, :min_w] ** 2)
    ax6.quiver(X[:min_h, :min_w], Y[:min_h, :min_w],
               dx[:min_h, :min_w], dy[:min_h, :min_w],
               magnitude.ravel(), cmap='coolwarm', scale=step * 12,
               width=0.003, headwidth=4)
    ax6.set_title('Direction-Coded Vectors', fontsize=12, fontweight='bold')
    ax6.axis('off')

    plt.suptitle('Block Matching Speckle Tracking — Cardiac Motion Analysis',
                 fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '05_block_matching.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/05_block_matching.png")

    return displacement


def demo_strain_estimation():
    """
    Demo: Strain estimation from displacement fields.

    Computes radial, lateral, shear, and principal strain from
    the tracked displacement field.
    """
    print("\n" + "=" * 60)
    print("  5.2 — Strain Estimation from Displacement")
    print("=" * 60)

    cardiac = generate_cardiac_phantom(256, 256, n_frames=20, seed=42)

    frame1 = cardiac[3]
    frame2 = cardiac[5]

    displacement = block_matching(frame1, frame2, block_size=13, search_size=11, metric='sad')
    strain = compute_strain(displacement)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    us_cmap = create_ultrasound_colormap()

    # Original frame with overlay
    axes[0, 0].imshow(frame1, cmap=us_cmap, vmin=0, vmax=1)
    axes[0, 0].set_title('Reference Frame', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Displacement magnitude
    disp_mag = np.sqrt(displacement[..., 0] ** 2 + displacement[..., 1] ** 2)
    axes[0, 1].imshow(disp_mag, cmap='magma')
    axes[0, 1].set_title('Displacement Magnitude', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Radial strain
    vmax = np.percentile(np.abs(strain['radial']), 95)
    im_r = axes[0, 2].imshow(strain['radial'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('Radial Strain (εyy)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_r, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Lateral strain
    vmax_l = np.percentile(np.abs(strain['lateral']), 95)
    im_l = axes[1, 0].imshow(strain['lateral'], cmap='RdBu_r', vmin=-vmax_l, vmax=vmax_l)
    axes[1, 0].set_title('Lateral Strain (εxx)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_l, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Shear strain
    vmax_s = np.percentile(np.abs(strain['shear']), 95)
    im_s = axes[1, 1].imshow(strain['shear'], cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s)
    axes[1, 1].set_title('Shear Strain (εxy)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_s, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Principal strain
    im_p = axes[1, 2].imshow(strain['principal'], cmap='hot')
    axes[1, 2].set_title('Principal Strain', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im_p, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle('Tissue Strain Estimation — From Speckle Tracking Displacement',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_strain_estimation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/05_strain_estimation.png")


def demo_temporal_motion():
    """
    Demo: Temporal motion analysis over the cardiac cycle.

    Tracks a region of interest across multiple frames and plots
    the displacement over time.
    """
    print("\n" + "=" * 60)
    print("  5.3 — Temporal Motion Analysis (Cardiac Cycle)")
    print("=" * 60)

    n_frames = 20
    cardiac = generate_cardiac_phantom(128, 128, n_frames=n_frames, seed=42)

    # Track displacement between consecutive frame pairs
    mean_displacements = []
    max_displacements = []

    reference_frame = cardiac[0]
    for t in range(1, n_frames):
        target_frame = cardiac[t]
        # Use simplified block matching for speed
        disp = block_matching(reference_frame, target_frame,
                              block_size=9, search_size=7, metric='sad')
        mag = np.sqrt(disp[..., 0] ** 2 + disp[..., 1] ** 2)
        mean_displacements.append(mag.mean())
        max_displacements.append(mag.max())
        reference_frame = target_frame

    # Plot temporal motion curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean displacement over time
    time = np.arange(1, n_frames)
    axes[0, 0].plot(time, mean_displacements, 'o-', color='#2196F3',
                    linewidth=2, markersize=6, label='Mean Displacement')
    axes[0, 0].plot(time, max_displacements, 's--', color='#F44336',
                    linewidth=2, markersize=5, alpha=0.7, label='Max Displacement')
    axes[0, 0].set_xlabel('Frame', fontsize=11)
    axes[0, 0].set_ylabel('Displacement (pixels)', fontsize=11)
    axes[0, 0].set_title('Displacement Over Cardiac Cycle', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase diagram (velocity vs displacement)
    velocities = np.diff(mean_displacements)
    axes[0, 1].plot(mean_displacements[1:], velocities, 'o-', color='#4CAF50',
                    linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Displacement', fontsize=11)
    axes[0, 1].set_ylabel('Velocity (Δ displacement)', fontsize=11)
    axes[0, 1].set_title('Phase Diagram', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    # Mark start/end
    axes[0, 1].annotate('Start', xy=(mean_displacements[1], velocities[0]),
                        fontsize=10, fontweight='bold', color='red')

    # Frame montage showing motion
    us_cmap = create_ultrasound_colormap()
    montage_frames = [0, 4, 9, 14, 19]
    montage = np.hstack([cardiac[f] for f in montage_frames])
    axes[1, 0].imshow(montage, cmap=us_cmap, vmin=0, vmax=1, aspect='auto')
    axes[1, 0].set_title('Selected Frames (1, 5, 10, 15, 20)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    for i, f in enumerate(montage_frames):
        axes[1, 0].text(i * 128 + 64, 5, f'F{f + 1}', ha='center', va='top',
                        color='lime', fontsize=10, fontweight='bold')

    # M-mode (single scanline over time)
    mid_col = 64
    m_mode = cardiac[:, :, mid_col]  # shape: (n_frames, height)
    axes[1, 1].imshow(m_mode.T, cmap=us_cmap, aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Frame (Time)', fontsize=11)
    axes[1, 1].set_ylabel('Depth', fontsize=11)
    axes[1, 1].set_title('M-Mode (Single Scanline Over Time)', fontsize=12, fontweight='bold')

    plt.suptitle('Temporal Motion Analysis — Cardiac Cycle Dynamics',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_temporal_motion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/05_temporal_motion.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FAST Ultrasound Tutorial — Speckle Tracking           ║")
    print("║   Python Implementation                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_block_matching()
    demo_strain_estimation()
    demo_temporal_motion()

    print("\n" + "=" * 60)
    print("  All outputs saved to: output/")
    print("  Files generated:")
    print("    • 05_block_matching.png")
    print("    • 05_strain_estimation.png")
    print("    • 05_temporal_motion.png")
    print("=" * 60)
