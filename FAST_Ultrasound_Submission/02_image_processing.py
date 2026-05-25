"""
Script 02: Ultrasound Image Processing
========================================
Demonstrates ultrasound image processing techniques from the FAST tutorial:
- Noise removal / De-speckling (Non-Local Means filter)
- Scan conversion (beamspace → sector/linear image)
- Envelope detection and log compression
- Applying colormaps (standard, ultrasound S-curve, thermal)
- Automatic ultrasound sector cropping
- Custom processing pipeline

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization

Usage:
    python 02_image_processing.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import (
    generate_tissue_phantom,
    generate_sector_beamspace_data,
    generate_iq_data,
    generate_cardiac_phantom,
    generate_vascular_phantom
)
from utils.image_utils import (
    nlm_fast,
    scan_convert_sector,
    scan_convert_linear,
    envelope_detection,
    log_compress,
    envelope_and_log_compress,
    create_ultrasound_colormap,
    create_thermal_colormap,
    create_doppler_colormap,
    ultrasound_image_crop
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_nlm_denoising():
    """
    Demo: Non-Local Means denoising for de-speckling.

    Equivalent to FAST's NonLocalMeans filter with DualViewWindow2D.
    Shows original vs denoised with different smoothing parameters.
    """
    print("\n" + "=" * 60)
    print("  2.1 — Non-Local Means De-Speckling")
    print("=" * 60)

    # Generate noisy ultrasound image
    image = generate_tissue_phantom(256, 256, seed=42)

    # Apply NLM with different smoothing amounts
    smoothing_levels = [0.05, 0.10, 0.20, 0.35]
    results = []
    for h in smoothing_levels:
        print(f"    Processing h={h:.2f}...")
        denoised = nlm_fast(image, filter_size=5, search_size=11, h=h)
        results.append(denoised)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # Original
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original (Speckled)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Denoised versions
    for idx, (result, h) in enumerate(zip(results, smoothing_levels)):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'NLM (h={h:.2f})', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

    # Difference image (noise that was removed)
    diff = np.abs(image - results[2])
    axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Removed Noise (h=0.20)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Non-Local Means Denoising — De-speckling Comparison',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_nlm_denoising.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_nlm_denoising.png")


def demo_scan_conversion():
    """
    Demo: Scan conversion of beamspace data.

    Equivalent to FAST's ScanConverter for sector and linear scans.
    """
    print("\n" + "=" * 60)
    print("  2.2 — Scan Conversion (Beamspace → Image)")
    print("=" * 60)

    beamspace = generate_sector_beamspace_data(256, 700, seed=42)

    # Sector scan conversion (phased array)
    sector_image = scan_convert_sector(
        beamspace, width=512, height=512,
        start_depth=0, end_depth=120,
        start_angle=-0.785398, end_angle=0.785398  # ±45°
    )

    # Linear scan conversion
    linear_image = scan_convert_linear(
        beamspace, width=512, height=512,
        start_depth=0, end_depth=120,
        left=-50, right=50
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(beamspace, cmap='gray', aspect='auto')
    axes[0].set_title('Raw Beamspace Data', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Scanline Index')
    axes[0].set_ylabel('Depth Sample')

    axes[1].imshow(sector_image, cmap='gray')
    axes[1].set_title('Sector Scan Conversion (±45°)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Lateral (mm)')
    axes[1].set_ylabel('Depth (mm)')

    axes[2].imshow(linear_image, cmap='gray')
    axes[2].set_title('Linear Scan Conversion', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Lateral (mm)')
    axes[2].set_ylabel('Depth (mm)')

    plt.suptitle('Scan Conversion — Beamspace to Image Coordinates',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_scan_conversion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_scan_conversion.png")


def demo_envelope_log_compression():
    """
    Demo: Envelope detection and log compression of IQ data.

    Equivalent to FAST's EnvelopeAndLogCompressor pipeline.
    """
    print("\n" + "=" * 60)
    print("  2.3 — Envelope Detection & Log Compression")
    print("=" * 60)

    iq_data = generate_iq_data(512, 256, seed=42)
    print(f"  IQ data shape: {iq_data.shape}")

    # Step 1: Envelope detection
    envelope = envelope_detection(iq_data)

    # Step 2: Log compression with different dynamic ranges
    dr_values = [30, 50, 60, 80]
    compressed = [log_compress(envelope, dr, gain_db=0) for dr in dr_values]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig)

    # Raw IQ components
    ax_i = fig.add_subplot(gs[0, 0])
    ax_i.imshow(iq_data[..., 0], cmap='RdBu', aspect='auto')
    ax_i.set_title('In-phase (I)', fontsize=11, fontweight='bold')
    ax_i.axis('off')

    ax_q = fig.add_subplot(gs[0, 1])
    ax_q.imshow(iq_data[..., 1], cmap='RdBu', aspect='auto')
    ax_q.set_title('Quadrature (Q)', fontsize=11, fontweight='bold')
    ax_q.axis('off')

    # Envelope
    ax_env = fig.add_subplot(gs[0, 2])
    ax_env.imshow(envelope, cmap='gray', aspect='auto')
    ax_env.set_title('Envelope √(I²+Q²)', fontsize=11, fontweight='bold')
    ax_env.axis('off')

    # Scan converted envelope
    sector = scan_convert_sector(log_compress(envelope, 60), 512, 512)
    ax_sc = fig.add_subplot(gs[0, 3])
    ax_sc.imshow(sector, cmap='gray')
    ax_sc.set_title('Scan Converted', fontsize=11, fontweight='bold')
    ax_sc.axis('off')

    # Different dynamic ranges
    for idx, (comp, dr) in enumerate(zip(compressed, dr_values)):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(comp, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'DR = {dr} dB', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('IQ Data Processing Pipeline — Envelope Detection & Log Compression',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_envelope_logcompress.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_envelope_logcompress.png")


def demo_colormaps():
    """
    Demo: Applying different colormaps to ultrasound images.

    Equivalent to FAST's Colormap.Ultrasound() and ApplyColormap.
    """
    print("\n" + "=" * 60)
    print("  2.4 — Ultrasound Colormaps")
    print("=" * 60)

    image = generate_tissue_phantom(512, 512, seed=42)

    # Get custom colormaps
    us_cmap = create_ultrasound_colormap()
    thermal_cmap = create_thermal_colormap()
    doppler_cmap = create_doppler_colormap()

    cmaps = [
        ('gray', 'Standard Grayscale'),
        (us_cmap, 'Ultrasound S-Curve'),
        (thermal_cmap, 'Thermal'),
        ('bone', 'Bone'),
        ('copper', 'Copper'),
        (doppler_cmap, 'Doppler (Blue-Red)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    for idx, (cmap, name) in enumerate(cmaps):
        row, col = idx // 3, idx % 3
        im = axes[row, col].imshow(image, cmap=cmap, vmin=0, vmax=1)
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.suptitle('Ultrasound Colormaps — Visual Comparison',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_colormaps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_colormaps.png")


def demo_auto_crop():
    """
    Demo: Automatic ultrasound sector cropping.

    Equivalent to FAST's UltrasoundImageCropper.
    """
    print("\n" + "=" * 60)
    print("  2.5 — Automatic Sector Cropping")
    print("=" * 60)

    # Create an image with padding (simulating scanner GUI around the image)
    core_image = generate_tissue_phantom(300, 300, seed=42)
    padded_image = np.zeros((512, 512), dtype=np.float32)
    # Place the ultrasound in the center with black borders
    padded_image[80:380, 106:406] = core_image
    # Add some simulated scanner GUI text area
    padded_image[0:60, :] = 0.15  # Top bar
    padded_image[420:512, :] = 0.1  # Bottom bar

    # Crop
    cropped, (top, bottom, left, right) = ultrasound_image_crop(
        padded_image, threshold_vertical=30, threshold_horizontal=10
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(padded_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (with GUI/Padding)', fontsize=12, fontweight='bold')

    # Show crop region
    import matplotlib.patches as patches
    rect = patches.Rectangle((left, top), right - left, bottom - top,
                              linewidth=2, edgecolor='lime', facecolor='none',
                              linestyle='--')
    axes[1].imshow(padded_image, cmap='gray', vmin=0, vmax=1)
    axes[1].add_patch(rect)
    axes[1].set_title('Detected ROI', fontsize=12, fontweight='bold')

    axes[2].imshow(cropped, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Cropped ({cropped.shape[1]}×{cropped.shape[0]})',
                      fontsize=12, fontweight='bold')

    for ax in axes:
        ax.axis('off')

    plt.suptitle('Automatic Ultrasound Image Cropping',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_autocrop.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_autocrop.png")


def demo_full_pipeline():
    """
    Demo: Complete processing pipeline (IQ → Envelope → Log Compress →
    Scan Convert → NLM Filter → Colormap).

    Equivalent to the full FAST pipeline with VBeamStreamer + processing chain.
    """
    print("\n" + "=" * 60)
    print("  2.6 — Full Processing Pipeline")
    print("=" * 60)

    # Step 1: Generate raw IQ data
    iq_data = generate_iq_data(512, 256, seed=42)
    print("  Step 1: Raw IQ data generated")

    # Step 2: Envelope detection + log compression
    compressed = envelope_and_log_compress(iq_data, dynamic_range_db=60, gain_db=0)
    print("  Step 2: Envelope detection + log compression done")

    # Step 3: Scan conversion
    scan_converted = scan_convert_sector(compressed, 512, 512)
    print("  Step 3: Scan conversion done")

    # Step 4: NLM filtering
    filtered = nlm_fast(scan_converted, filter_size=5, search_size=11, h=0.15)
    print("  Step 4: NLM filtering done")

    # Visualize the pipeline
    us_cmap = create_ultrasound_colormap()

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # Row 1: Pipeline steps
    axes[0, 0].imshow(np.sqrt(iq_data[..., 0] ** 2 + iq_data[..., 1] ** 2),
                      cmap='gray', aspect='auto')
    axes[0, 0].set_title('① Raw IQ Amplitude', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(compressed, cmap='gray', aspect='auto')
    axes[0, 1].set_title('② Envelope + Log Compress', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(scan_converted, cmap='gray')
    axes[0, 2].set_title('③ Scan Converted', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Final results
    axes[1, 0].imshow(filtered, cmap='gray')
    axes[1, 0].set_title('④ NLM Filtered', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(filtered, cmap=us_cmap)
    axes[1, 1].set_title('⑤ Ultrasound Colormap', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    thermal = create_thermal_colormap()
    axes[1, 2].imshow(filtered, cmap=thermal)
    axes[1, 2].set_title('⑥ Thermal Colormap', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # Add pipeline flow arrows
    fig.text(0.22, 0.52, '→', fontsize=30, ha='center', fontweight='bold', color='steelblue')
    fig.text(0.52, 0.52, '→', fontsize=30, ha='center', fontweight='bold', color='steelblue')
    fig.text(0.12, 0.47, '↓', fontsize=30, ha='center', fontweight='bold', color='steelblue')

    plt.suptitle('Complete Ultrasound Processing Pipeline\n'
                 'IQ Data → Envelope → Log Compress → Scan Convert → NLM Filter → Colormap',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, '02_full_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/02_full_pipeline.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FAST Ultrasound Tutorial — Image Processing           ║")
    print("║   Python Implementation                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_nlm_denoising()
    demo_scan_conversion()
    demo_envelope_log_compression()
    demo_colormaps()
    demo_auto_crop()
    demo_full_pipeline()

    print("\n" + "=" * 60)
    print("  All outputs saved to: output/")
    print("  Files generated:")
    print("    • 02_nlm_denoising.png")
    print("    • 02_scan_conversion.png")
    print("    • 02_envelope_logcompress.png")
    print("    • 02_colormaps.png")
    print("    • 02_autocrop.png")
    print("    • 02_full_pipeline.png")
    print("=" * 60)
