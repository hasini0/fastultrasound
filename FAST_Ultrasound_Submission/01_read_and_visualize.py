"""
Script 01: Reading and Visualizing Ultrasound Data
====================================================
Demonstrates reading ultrasound data from various sources and visualizing it.

This script mirrors the FAST framework tutorial sections:
- Reading ultrasound images from video/image sequences
- Reading DICOM multiframe files
- Converting to numpy arrays
- Streaming 2D and 3D ultrasound data
- Using playback/frame extraction

Since FAST is Linux/Windows only, we implement equivalent functionality using
standard Python libraries (NumPy, Matplotlib, OpenCV, pydicom).

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization

Usage:
    python 01_read_and_visualize.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import (
    generate_tissue_phantom,
    generate_cardiac_phantom,
    generate_vascular_phantom,
    generate_3d_volume,
    generate_sector_beamspace_data
)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_read_single_image():
    """
    Demo: Read a single ultrasound image and convert to numpy array.

    Equivalent to FAST's ImageFileImporter → runAndGetOutputData → np.asarray()
    """
    print("\n" + "=" * 60)
    print("  1.1 — Reading a Single Ultrasound Image")
    print("=" * 60)

    # Generate synthetic ultrasound image (equivalent to loading from file)
    image = generate_tissue_phantom(512, 512, seed=42)
    print(f"  Image shape: {image.shape}")
    print(f"  Data type:   {image.dtype}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Grayscale view
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('B-mode Ultrasound Image', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Lateral (pixels)')
    axes[0].set_ylabel('Depth (pixels)')

    # Histogram of pixel intensities
    axes[1].hist(image.ravel(), bins=100, color='steelblue', alpha=0.8, edgecolor='navy')
    axes[1].set_title('Pixel Intensity Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Count')
    axes[1].axvline(image.mean(), color='red', linestyle='--', label=f'Mean: {image.mean():.3f}')
    axes[1].legend()

    # Depth profile (average across lateral dimension)
    depth_profile = image.mean(axis=1)
    axes[2].plot(depth_profile, np.arange(len(depth_profile)), color='steelblue', linewidth=2)
    axes[2].set_title('Depth Profile (TGC effect)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Average Intensity')
    axes[2].set_ylabel('Depth (pixels)')
    axes[2].invert_yaxis()

    plt.suptitle('Single Ultrasound Image Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_single_image.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/01_single_image.png")


def demo_stream_video_frames():
    """
    Demo: Stream ultrasound images and extract every Nth frame.

    Equivalent to FAST's MovieStreamer + DataStream loop + matplotlib display.
    """
    print("\n" + "=" * 60)
    print("  1.2 — Streaming Video Frames (Cardiac Sequence)")
    print("=" * 60)

    # Generate a cardiac ultrasound sequence (simulates MovieStreamer/ImageFileStreamer)
    n_frames = 30
    cardiac_sequence = generate_cardiac_phantom(512, 512, n_frames=n_frames, seed=42)
    print(f"  Sequence shape: {cardiac_sequence.shape}")
    print(f"  Total frames:   {n_frames}")

    # Extract every 3rd frame and display in a grid (equivalent to FAST DataStream loop)
    frame_list = []
    for i, frame in enumerate(cardiac_sequence):
        if (i + 1) % 3 == 0:
            frame_list.append((frame, i + 1))
        if len(frame_list) == 9:
            break

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for idx in range(9):
        row, col = idx // 3, idx % 3
        axes[row, col].imshow(frame_list[idx][0], cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'Frame {frame_list[idx][1]}', fontsize=11, fontweight='bold')
        axes[row, col].axis('off')

    plt.suptitle('Cardiac Ultrasound — Every 3rd Frame', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_frame_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/01_frame_grid.png")


def demo_vascular_imaging():
    """
    Demo: Vascular ultrasound imaging (artery and vein cross-section).
    """
    print("\n" + "=" * 60)
    print("  1.3 — Vascular Ultrasound Imaging")
    print("=" * 60)

    vascular = generate_vascular_phantom(400, 400, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Standard B-mode view
    axes[0].imshow(vascular, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('B-mode Vascular Image', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Lateral (pixels)')
    axes[0].set_ylabel('Depth (pixels)')

    # Annotated view with measurements
    axes[1].imshow(vascular, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Annotated View', fontsize=13, fontweight='bold')

    # Add measurement annotations
    circle1 = plt.Circle((140, 180), 45, fill=False, color='cyan', linewidth=2, linestyle='--')
    circle2 = plt.Circle((260, 200), 50, fill=False, color='red', linewidth=2, linestyle='--')
    axes[1].add_patch(circle1)
    axes[1].add_patch(circle2)
    axes[1].annotate('Artery', xy=(140, 130), color='cyan', fontsize=12,
                     fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[1].annotate('Vein', xy=(260, 145), color='red', fontsize=12,
                     fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[1].set_xlabel('Lateral (pixels)')
    axes[1].set_ylabel('Depth (pixels)')

    plt.suptitle('Vascular Ultrasound — Cross-Section View', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_vascular.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/01_vascular.png")


def demo_3d_volume_slicing():
    """
    Demo: Read and display 3D ultrasound volume (multi-plane views).

    Equivalent to FAST's ImageFileStreamer with 3D metaimage files +
    SlicerWindow for multi-plane visualization.
    """
    print("\n" + "=" * 60)
    print("  1.4 — 3D Ultrasound Volume Slicing")
    print("=" * 60)

    volume = generate_3d_volume(128, 128, 128, seed=42)
    print(f"  Volume shape: {volume.shape}")

    # Multi-planar reconstruction (MPR) — equivalent to FAST SlicerWindow
    mid_z, mid_y, mid_x = volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Axial slice (XY plane)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(volume[mid_z, :, :], cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Axial (Z={mid_z})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axhline(mid_y, color='cyan', alpha=0.5, linewidth=1)
    ax1.axvline(mid_x, color='yellow', alpha=0.5, linewidth=1)

    # Sagittal slice (YZ plane)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(volume[:, :, mid_x], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Sagittal (X={mid_x})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.axhline(mid_z, color='red', alpha=0.5, linewidth=1)
    ax2.axvline(mid_y, color='cyan', alpha=0.5, linewidth=1)

    # Coronal slice (XZ plane)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(volume[:, mid_y, :], cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'Coronal (Y={mid_y})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.axhline(mid_z, color='red', alpha=0.5, linewidth=1)
    ax3.axvline(mid_x, color='yellow', alpha=0.5, linewidth=1)

    # Multiple axial slices at different depths
    ax4 = fig.add_subplot(gs[1, :])
    n_slices = 8
    slice_indices = np.linspace(10, volume.shape[0] - 10, n_slices, dtype=int)
    montage_row = np.hstack([volume[si, :, :] for si in slice_indices])
    ax4.imshow(montage_row, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax4.set_title('Axial Slice Montage (Z = ' + ', '.join(str(s) for s in slice_indices) + ')',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (tiled slices)')
    ax4.set_ylabel('Y')

    # Add slice labels
    for i, si in enumerate(slice_indices):
        ax4.text(i * volume.shape[2] + volume.shape[2] // 2, 5, f'Z={si}',
                 ha='center', va='top', color='yellow', fontsize=9, fontweight='bold')

    plt.suptitle('3D Ultrasound Volume — Multi-Planar Reconstruction (MPR)',
                 fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '01_3d_volume_slices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/01_3d_volume_slices.png")


def demo_beamspace_data():
    """
    Demo: Display raw beamspace (pre-scan-converted) data.

    Equivalent to FAST's UFFStreamer with doScanConversion=False.
    """
    print("\n" + "=" * 60)
    print("  1.5 — Raw Beamspace Ultrasound Data")
    print("=" * 60)

    beamspace = generate_sector_beamspace_data(256, 700, seed=42)
    print(f"  Beamspace shape: {beamspace.shape} (depth_samples × scanlines)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(beamspace, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Beamspace Data (Pre-Scan Conversion)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Scan Line Index')
    axes[0].set_ylabel('Depth Sample')

    # Show a few individual scanlines
    scanline_indices = [50, 128, 200]
    colors = ['#00bcd4', '#ff5722', '#4caf50']
    for idx, color in zip(scanline_indices, colors):
        axes[1].plot(beamspace[:, idx], label=f'Scanline {idx}', color=color, alpha=0.8)

    axes[1].set_title('Individual Scan Line Profiles', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Depth Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Raw Ultrasound Data Visualization', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_beamspace.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/01_beamspace.png")


def demo_cardiac_animation():
    """
    Demo: Create an animated GIF of the cardiac ultrasound sequence.

    Equivalent to FAST's display2D with MovieStreamer/PlaybackWidget.
    """
    print("\n" + "=" * 60)
    print("  1.6 — Cardiac Animation (Animated GIF)")
    print("=" * 60)

    cardiac = generate_cardiac_phantom(256, 256, n_frames=20, seed=42)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Cardiac Ultrasound — Real-time Playback', fontsize=13, fontweight='bold')
    ax.axis('off')

    im = ax.imshow(cardiac[0], cmap='gray', vmin=0, vmax=1)

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         color='lime', fontsize=12, fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    def update(frame_idx):
        im.set_data(cardiac[frame_idx])
        frame_text.set_text(f'Frame: {frame_idx + 1}/{len(cardiac)}')
        return [im, frame_text]

    anim = animation.FuncAnimation(fig, update, frames=len(cardiac),
                                   interval=100, blit=True)
    anim.save(os.path.join(OUTPUT_DIR, '01_cardiac_animation.gif'),
              writer='pillow', fps=10)
    plt.close()
    print("  ✓ Saved: output/01_cardiac_animation.gif")


# ─────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FAST Ultrasound Tutorial — Reading & Visualization    ║")
    print("║   Python Implementation                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_read_single_image()
    demo_stream_video_frames()
    demo_vascular_imaging()
    demo_3d_volume_slicing()
    demo_beamspace_data()
    demo_cardiac_animation()

    print("\n" + "=" * 60)
    print("  All outputs saved to: output/")
    print("  Files generated:")
    print("    • 01_single_image.png")
    print("    • 01_frame_grid.png")
    print("    • 01_vascular.png")
    print("    • 01_3d_volume_slices.png")
    print("    • 01_beamspace.png")
    print("    • 01_cardiac_animation.gif")
    print("=" * 60)
