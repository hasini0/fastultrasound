"""
Script 03: 3D Ultrasound Visualization ⭐ (Creative Element)
=============================================================
Interactive 3D volume rendering and visualization of ultrasound data.

This is the creative component demonstrating:
- 3D volume rendering with PyVista/VTK
- Interactive multi-plane slicer
- Isosurface extraction from volumetric data
- Animated 4D (3D + time) visualization
- Color-mapped 3D rendering
- Surface extraction and mesh generation

Equivalent to FAST's:
- AlphaBlendingVolumeRenderer
- SlicerWindow
- SimpleWindow3D

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization

Usage:
    python 03_3d_visualization.py
    python 03_3d_visualization.py --interactive   # for live interactive viewer
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import generate_3d_volume, generate_4d_volume

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if PyVista is available (for interactive 3D)
PYVISTA_AVAILABLE = False
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("  Note: PyVista not installed. Interactive 3D views will use matplotlib.")
    print("  Install with: pip install pyvista")


def demo_multiplanar_3d():
    """
    Multi-planar reconstruction with interactive slice selection.

    Renders axial, sagittal, and coronal views of the 3D volume
    with crosshair overlays showing the slice positions.
    """
    print("\n" + "=" * 60)
    print("  3.1 — Multi-Planar 3D Reconstruction (MPR)")
    print("=" * 60)

    volume = generate_3d_volume(128, 128, 128, seed=42)
    nz, ny, nx = volume.shape

    fig = plt.figure(figsize=(18, 14))

    # Create a 3x3 grid layout
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
                  hspace=0.35, wspace=0.3)

    slices_z = [nz // 4, nz // 2, 3 * nz // 4]
    slices_y = [ny // 4, ny // 2, 3 * ny // 4]
    slices_x = [nx // 4, nx // 2, 3 * nx // 4]

    from utils.image_utils import create_ultrasound_colormap
    us_cmap = create_ultrasound_colormap()

    # Row 1: Axial slices at different depths
    for i, sz in enumerate(slices_z):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(volume[sz, :, :], cmap=us_cmap, vmin=0, vmax=1)
        ax.set_title(f'Axial (Z = {sz})', fontsize=11, fontweight='bold')
        ax.axis('off')
        # Add crosshair
        ax.axhline(ny // 2, color='cyan', alpha=0.4, linewidth=0.8)
        ax.axvline(nx // 2, color='yellow', alpha=0.4, linewidth=0.8)

    # Row 2: Sagittal slices
    for i, sx in enumerate(slices_x):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(volume[:, :, sx], cmap=us_cmap, vmin=0, vmax=1)
        ax.set_title(f'Sagittal (X = {sx})', fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.axhline(nz // 2, color='red', alpha=0.4, linewidth=0.8)
        ax.axvline(ny // 2, color='cyan', alpha=0.4, linewidth=0.8)

    # Row 3: Coronal slices
    for i, sy in enumerate(slices_y):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(volume[:, sy, :], cmap=us_cmap, vmin=0, vmax=1)
        ax.set_title(f'Coronal (Y = {sy})', fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.axhline(nz // 2, color='red', alpha=0.4, linewidth=0.8)
        ax.axvline(nx // 2, color='yellow', alpha=0.4, linewidth=0.8)

    # Colorbar
    cax = fig.add_subplot(gs[:, 3])
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=us_cmap, norm=norm)
    fig.colorbar(sm, cax=cax, label='Echo Intensity')

    plt.suptitle('3D Ultrasound — Multi-Planar Reconstruction (MPR)\n'
                 'Axial · Sagittal · Coronal Views at Multiple Depths',
                 fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_multiplanar_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_multiplanar_3d.png")


def demo_volume_rendering_matplotlib():
    """
    Volume rendering using matplotlib's 3D projection.

    Creates a pseudo volume rendering by stacking semi-transparent slices.
    """
    print("\n" + "=" * 60)
    print("  3.2 — 3D Volume Rendering (Matplotlib)")
    print("=" * 60)

    volume = generate_3d_volume(64, 64, 64, seed=42)
    nz, ny, nx = volume.shape

    fig = plt.figure(figsize=(16, 7))

    # Method 1: Stacked slice rendering
    ax1 = fig.add_subplot(121, projection='3d')

    # Show selected slices with transparency
    from matplotlib.colors import Normalize
    from matplotlib import cm

    n_display_slices = 12
    slice_indices = np.linspace(5, nz - 5, n_display_slices, dtype=int)

    for idx, sz in enumerate(slice_indices):
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        Z = np.full_like(X, sz, dtype=float)

        # Create RGBA colors from the slice data
        slice_data = volume[sz, :, :]
        colors = cm.gray(slice_data)
        colors[..., 3] = np.clip(slice_data * 1.5, 0.05, 0.6)  # Alpha from intensity

        ax1.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2,
                         shade=False, antialiased=False)

    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z (Depth)', fontsize=10)
    ax1.set_title('Stacked Slice Rendering', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.set_box_aspect([1, 1, 1])

    # Method 2: Maximum Intensity Projection (MIP)
    ax2 = fig.add_subplot(122)

    # MIP from 3 different angles
    mip_axial = np.max(volume, axis=0)
    mip_sagittal = np.max(volume, axis=2)
    mip_coronal = np.max(volume, axis=1)

    # Combine into a triptych
    gap = 4
    combined = np.zeros((ny, nx * 3 + gap * 2), dtype=np.float32)
    combined[:, :nx] = mip_axial
    combined[:, nx + gap:2 * nx + gap] = mip_sagittal
    combined[:, 2 * nx + 2 * gap:] = mip_coronal

    from utils.image_utils import create_ultrasound_colormap
    us_cmap = create_ultrasound_colormap()

    ax2.imshow(combined, cmap=us_cmap, vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Maximum Intensity Projection (MIP)', fontsize=12, fontweight='bold')

    # Add labels
    ax2.text(nx // 2, ny + 5, 'Axial MIP', ha='center', fontsize=10,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.8))
    ax2.text(nx + gap + nx // 2, ny + 5, 'Sagittal MIP', ha='center', fontsize=10,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.8))
    ax2.text(2 * nx + 2 * gap + nx // 2, ny + 5, 'Coronal MIP', ha='center', fontsize=10,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.8))
    ax2.axis('off')

    plt.suptitle('3D Ultrasound Volume Rendering',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_volume_rendering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_volume_rendering.png")


def demo_isosurface_extraction():
    """
    Isosurface extraction from 3D ultrasound volume.

    Uses marching cubes to extract surfaces at different intensity thresholds.
    """
    print("\n" + "=" * 60)
    print("  3.3 — Isosurface Extraction (Marching Cubes)")
    print("=" * 60)

    volume = generate_3d_volume(96, 96, 96, seed=42)
    # Smooth the volume for better surfaces
    volume_smooth = gaussian_filter(volume, sigma=2.0)

    from skimage.measure import marching_cubes

    fig = plt.figure(figsize=(16, 6))

    thresholds = [0.3, 0.5, 0.7]
    colors_list = ['#2196F3', '#FF9800', '#F44336']
    titles = ['Low Threshold (0.3)\nSoft Tissue Boundary',
              'Mid Threshold (0.5)\nStructure Boundaries',
              'High Threshold (0.7)\nBright Structures']

    for idx, (thresh, color, title) in enumerate(zip(thresholds, colors_list, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        try:
            verts, faces, normals, values = marching_cubes(volume_smooth, level=thresh)

            ax.plot_trisurf(verts[:, 2], verts[:, 1], faces, verts[:, 0],
                            color=color, alpha=0.5, edgecolor='none')
        except Exception:
            ax.text(0.5, 0.5, 0.5, 'No surface\nat this level',
                    ha='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=135)

    plt.suptitle('Isosurface Extraction — 3D Structure Identification',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_isosurface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_isosurface.png")


def demo_4d_animation():
    """
    4D (3D + time) ultrasound visualization.

    Creates animated views of a pulsating 3D volume over time.
    """
    print("\n" + "=" * 60)
    print("  3.4 — 4D Ultrasound Animation (3D + Time)")
    print("=" * 60)

    volumes = generate_4d_volume(64, 64, 64, n_frames=16, seed=42)
    n_frames = volumes.shape[0]
    nz = volumes.shape[1]

    from utils.image_utils import create_ultrasound_colormap
    us_cmap = create_ultrasound_colormap()

    # Create an animated GIF showing 3 orthogonal slices over time
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mid = nz // 2

    ims = []
    for t in range(n_frames):
        im_list = []
        for ax_idx, (sl, title) in enumerate([
            (volumes[t, mid, :, :], 'Axial'),
            (volumes[t, :, :, mid], 'Sagittal'),
            (volumes[t, :, mid, :], 'Coronal')
        ]):
            im = axes[ax_idx].imshow(sl, cmap=us_cmap, vmin=0, vmax=1, animated=True)
            im_list.append(im)

        ims.append(im_list)

    # Set titles (static)
    for ax_idx, title in enumerate(['Axial (Z=mid)', 'Sagittal (X=mid)', 'Coronal (Y=mid)']):
        axes[ax_idx].set_title(title, fontsize=12, fontweight='bold')
        axes[ax_idx].axis('off')

    plt.suptitle('4D Ultrasound — Pulsating Volume Over Time',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    import matplotlib.animation as animation

    def update(frame_idx):
        vol = volumes[frame_idx]
        ims_frame = []
        for ax_idx, sl in enumerate([vol[mid, :, :], vol[:, :, mid], vol[:, mid, :]]):
            axes[ax_idx].clear()
            im = axes[ax_idx].imshow(sl, cmap=us_cmap, vmin=0, vmax=1)
            ims_frame.append(im)
            titles = ['Axial (Z=mid)', 'Sagittal (X=mid)', 'Coronal (Y=mid)']
            axes[ax_idx].set_title(titles[ax_idx], fontsize=12, fontweight='bold')
            axes[ax_idx].axis('off')
        fig.suptitle(f'4D Ultrasound — Frame {frame_idx + 1}/{n_frames}',
                     fontsize=15, fontweight='bold')
        return ims_frame

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=200)
    anim.save(os.path.join(OUTPUT_DIR, '03_4d_animation.gif'), writer='pillow', fps=5)
    plt.close()
    print("  ✓ Saved: output/03_4d_animation.gif")

    # Also save a static montage
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    frame_indices = np.linspace(0, n_frames - 1, 8, dtype=int)
    for idx, fi in enumerate(frame_indices):
        row, col = idx // 4, idx % 4
        axes[row, col].imshow(volumes[fi, mid, :, :], cmap=us_cmap, vmin=0, vmax=1)
        axes[row, col].set_title(f'Frame {fi + 1}', fontsize=11, fontweight='bold')
        axes[row, col].axis('off')

    plt.suptitle('4D Ultrasound — Axial Slices Over Cardiac Cycle',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_4d_montage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_4d_montage.png")


def demo_pyvista_interactive(volume_data=None):
    """
    Interactive 3D visualization using PyVista/VTK.

    This creates a rich interactive viewer with:
    - Orthogonal slice planes
    - Volume rendering with opacity transfer function
    - Isosurface overlays
    """
    if not PYVISTA_AVAILABLE:
        print("\n  ⚠ PyVista not installed. Skipping interactive 3D viewer.")
        print("    Install with: pip install pyvista")
        return

    print("\n" + "=" * 60)
    print("  3.5 — Interactive 3D Viewer (PyVista)")
    print("=" * 60)

    if volume_data is None:
        volume_data = generate_3d_volume(96, 96, 96, seed=42)

    # Create PyVista uniform grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume_data.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = volume_data.ravel(order="F")

    # Create plotter
    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 700))

    # Left: Volume rendering with orthogonal slices
    plotter.subplot(0, 0)
    plotter.add_text("Orthogonal Slice Viewer", font_size=14)

    slices = grid.slice_orthogonal()
    plotter.add_mesh(slices, cmap='gray', show_scalar_bar=True,
                     scalar_bar_args={'title': 'Echo Intensity'})
    plotter.add_mesh(grid.outline(), color='white', line_width=2)

    # Right: Volume rendering
    plotter.subplot(0, 1)
    plotter.add_text("Volume Rendering", font_size=14)
    plotter.add_volume(grid, cmap='bone', opacity='sigmoid_5',
                       scalar_bar_args={'title': 'Echo Intensity'})

    plotter.link_views()
    plotter.show()
    print("  ✓ Interactive viewer closed")


def demo_pyvista_screenshot():
    """
    Create a static screenshot from PyVista 3D visualization.
    """
    if not PYVISTA_AVAILABLE:
        print("\n  ⚠ PyVista not installed. Generating matplotlib 3D view instead.")
        _fallback_3d_rendering()
        return

    print("\n" + "=" * 60)
    print("  3.5 — 3D Visualization Screenshot (PyVista)")
    print("=" * 60)

    volume_data = generate_3d_volume(96, 96, 96, seed=42)

    grid = pv.ImageData()
    grid.dimensions = np.array(volume_data.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = volume_data.ravel(order="F")

    # Off-screen rendering for screenshot
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 800), shape=(1, 2))

    # Left panel: Orthogonal slices
    plotter.subplot(0, 0)
    plotter.add_text("Orthogonal Slice Viewer", font_size=14, color='white')
    slices = grid.slice_orthogonal()
    plotter.add_mesh(slices, cmap='gray', show_scalar_bar=False)
    plotter.add_mesh(grid.outline(), color='white', line_width=2)
    plotter.set_background('black')

    # Right panel: Volume rendering
    plotter.subplot(0, 1)
    plotter.add_text("Volume Rendering", font_size=14, color='white')
    plotter.add_volume(grid, cmap='bone', opacity='sigmoid_5', show_scalar_bar=False)
    plotter.set_background('black')

    plotter.link_views()
    screenshot_path = os.path.join(OUTPUT_DIR, '03_pyvista_3d.png')
    plotter.screenshot(screenshot_path)
    plotter.close()
    print(f"  ✓ Saved: output/03_pyvista_3d.png")


def _fallback_3d_rendering():
    """Fallback 3D rendering using matplotlib when PyVista is not available."""
    volume = generate_3d_volume(64, 64, 64, seed=42)

    fig = plt.figure(figsize=(16, 12))

    # Create comprehensive 3D overview
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    from utils.image_utils import create_ultrasound_colormap
    us_cmap = create_ultrasound_colormap()

    nz, ny, nx = volume.shape
    mid = nz // 2

    # Orthogonal views
    for idx, (sl, title) in enumerate([
        (volume[mid, :, :], 'Axial (Z=mid)'),
        (volume[:, :, mid], 'Sagittal (X=mid)'),
        (volume[:, mid, :], 'Coronal (Y=mid)')
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(sl, cmap=us_cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # 3D scatter plot of high-intensity voxels
    ax3d = fig.add_subplot(gs[1, 0], projection='3d')
    # Sample high-intensity voxels
    threshold = 0.6
    z_idx, y_idx, x_idx = np.where(volume > threshold)
    if len(z_idx) > 2000:
        sample = np.random.choice(len(z_idx), 2000, replace=False)
        z_idx, y_idx, x_idx = z_idx[sample], y_idx[sample], x_idx[sample]

    colors = volume[z_idx, y_idx, x_idx]
    scatter = ax3d.scatter(x_idx, y_idx, z_idx, c=colors, cmap=us_cmap,
                           alpha=0.3, s=3, vmin=0, vmax=1)
    ax3d.set_title('3D Point Cloud\n(High Echo)', fontsize=11, fontweight='bold')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.view_init(elev=25, azim=135)

    # MIP renderings
    ax_mip1 = fig.add_subplot(gs[1, 1])
    mip = np.max(volume, axis=0)
    ax_mip1.imshow(mip, cmap=us_cmap, vmin=0, vmax=1)
    ax_mip1.set_title('MIP (Top-Down)', fontsize=12, fontweight='bold')
    ax_mip1.axis('off')

    ax_mip2 = fig.add_subplot(gs[1, 2])
    mip_side = np.max(volume, axis=2)
    ax_mip2.imshow(mip_side, cmap=us_cmap, vmin=0, vmax=1)
    ax_mip2.set_title('MIP (Side View)', fontsize=12, fontweight='bold')
    ax_mip2.axis('off')

    plt.suptitle('3D Ultrasound Volume Visualization',
                 fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_pyvista_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/03_pyvista_3d.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Ultrasound Visualization')
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive 3D viewer (requires PyVista)')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FAST Ultrasound Tutorial — 3D Visualization ⭐        ║")
    print("║   Python Implementation                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_multiplanar_3d()
    demo_volume_rendering_matplotlib()
    demo_isosurface_extraction()
    demo_4d_animation()
    demo_pyvista_screenshot()

    if args.interactive:
        demo_pyvista_interactive()

    print("\n" + "=" * 60)
    print("  All outputs saved to: output/")
    print("  Files generated:")
    print("    • 03_multiplanar_3d.png")
    print("    • 03_volume_rendering.png")
    print("    • 03_isosurface.png")
    print("    • 03_4d_animation.gif")
    print("    • 03_4d_montage.png")
    print("    • 03_pyvista_3d.png")
    print("=" * 60)
