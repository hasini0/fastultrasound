"""
Script 04: Ultrasound Image Segmentation
==========================================
Demonstrates segmentation techniques applied to ultrasound images:
- Thresholding-based segmentation
- Region growing
- Watershed segmentation for tissue boundaries
- Contour detection and overlay
- Morphological operations for cleanup

Equivalent to FAST's SegmentationNetwork and SegmentationRenderer
(using classical methods instead of neural networks).

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization

Usage:
    python 04_segmentation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_generator import generate_tissue_phantom, generate_vascular_phantom, generate_cardiac_phantom
from utils.image_utils import nlm_fast, create_ultrasound_colormap

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def otsu_threshold(image):
    """
    Compute Otsu's optimal threshold for a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image with values in [0, 1].

    Returns
    -------
    float
        Optimal threshold.
    """
    # Histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()

    # Otsu's method
    best_thresh = 0
    best_var = 0

    w0 = 0
    sum0 = 0
    total_sum = np.sum(bin_centers * hist)

    for i in range(256):
        w0 += hist[i]
        if w0 == 0:
            continue

        w1 = total - w0
        if w1 == 0:
            break

        sum0 += bin_centers[i] * hist[i]
        mean0 = sum0 / w0
        mean1 = (total_sum - sum0) / w1

        var_between = w0 * w1 * (mean0 - mean1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bin_centers[i]

    return best_thresh


def region_growing(image, seed_point, threshold=0.1):
    """
    Region growing segmentation from a seed point.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    seed_point : tuple
        (row, col) seed coordinate.
    threshold : float
        Maximum intensity difference from seed for region inclusion.

    Returns
    -------
    np.ndarray
        Binary mask of the segmented region.
    """
    h, w = image.shape
    mask = np.zeros((h, w), dtype=bool)
    seed_value = image[seed_point[0], seed_point[1]]

    # BFS queue
    queue = [seed_point]
    mask[seed_point[0], seed_point[1]] = True

    while queue:
        r, c = queue.pop(0)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                if abs(float(image[nr, nc]) - float(seed_value)) < threshold:
                    mask[nr, nc] = True
                    queue.append((nr, nc))

    return mask


def watershed_segmentation(image, n_markers=5):
    """
    Simplified watershed-like segmentation using distance transform approach.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    n_markers : int
        Number of seed regions.

    Returns
    -------
    np.ndarray
        Labeled segmentation map.
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt

    # Smooth the image
    smooth = gaussian_filter(image, sigma=3)

    # Create binary mask using Otsu
    thresh = otsu_threshold(smooth)
    binary = smooth > thresh

    # Distance transform
    distance = distance_transform_edt(binary)

    # Find markers (local maxima of distance transform)
    coords = peak_local_max(distance, min_distance=20, num_peaks=n_markers)
    markers = np.zeros_like(image, dtype=int)
    for idx, (r, c) in enumerate(coords):
        markers[r, c] = idx + 1

    # Watershed
    labels = watershed(-distance, markers, mask=binary)
    return labels


def demo_threshold_segmentation():
    """
    Demo: Thresholding-based segmentation with Otsu's method.
    """
    print("\n" + "=" * 60)
    print("  4.1 — Threshold-based Segmentation")
    print("=" * 60)

    image = generate_tissue_phantom(512, 512, seed=42)

    # Denoise first for better segmentation
    denoised = nlm_fast(image, filter_size=5, search_size=11, h=0.15)

    # Otsu threshold
    thresh = otsu_threshold(denoised)
    binary_otsu = denoised > thresh

    # Multi-level thresholding
    levels = [0.2, 0.35, 0.5, 0.65]
    multi_label = np.zeros_like(denoised, dtype=int)
    for i, lvl in enumerate(levels):
        multi_label[denoised > lvl] = i + 1

    # Create segmentation overlay
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('NLM Denoised', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(binary_otsu, cmap='gray')
    axes[0, 2].set_title(f'Otsu Threshold ({thresh:.3f})', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Multi-level segmentation
    colors = ['#000000', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
    seg_cmap = ListedColormap(colors)

    axes[1, 0].imshow(multi_label, cmap=seg_cmap, vmin=0, vmax=4)
    axes[1, 0].set_title('Multi-Level Segmentation', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay on original
    axes[1, 1].imshow(image, cmap='gray')
    overlay = np.ma.masked_where(multi_label == 0, multi_label)
    axes[1, 1].imshow(overlay, cmap='Set1', alpha=0.3, vmin=1, vmax=4)
    axes[1, 1].set_title('Segmentation Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Histogram with thresholds
    axes[1, 2].hist(denoised.ravel(), bins=100, color='steelblue', alpha=0.7, edgecolor='navy')
    for lvl in levels:
        axes[1, 2].axvline(lvl, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1, 2].axvline(thresh, color='lime', linestyle='-', linewidth=2,
                       label=f'Otsu: {thresh:.3f}')
    axes[1, 2].set_title('Intensity Histogram', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()

    plt.suptitle('Ultrasound Image Segmentation — Threshold Methods',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_threshold_segmentation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/04_threshold_segmentation.png")


def demo_vascular_segmentation():
    """
    Demo: Segmentation of vascular structures (artery and vein).
    """
    print("\n" + "=" * 60)
    print("  4.2 — Vascular Structure Segmentation")
    print("=" * 60)

    image = generate_vascular_phantom(400, 400, seed=42)
    denoised = nlm_fast(image, filter_size=5, search_size=11, h=0.12)

    # Detect dark regions (blood pools / lumens)
    lumen_mask = denoised < 0.12  # Anechoic regions

    # Clean up with morphological operations
    struct = np.ones((5, 5))
    lumen_clean = binary_opening(lumen_mask, structure=struct, iterations=2)
    lumen_clean = binary_closing(lumen_clean, structure=struct, iterations=2)
    lumen_clean = binary_fill_holes(lumen_clean)

    # Label connected components
    labeled, n_features = label(lumen_clean)
    print(f"  Detected {n_features} vessel lumens")

    # Find contours using gradient
    from skimage.measure import find_contours

    contours = find_contours(lumen_clean.astype(float), 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Vascular Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('NLM Denoised', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(lumen_mask, cmap='gray')
    axes[0, 2].set_title('Initial Lumen Detection', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(lumen_clean, cmap='gray')
    axes[1, 0].set_title('After Morphological Cleanup', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Labeled view
    label_cmap = ListedColormap(['black', '#F44336', '#2196F3', '#4CAF50', '#FF9800'])
    axes[1, 1].imshow(labeled, cmap=label_cmap, vmin=0, vmax=max(4, n_features))
    axes[1, 1].set_title(f'Labeled Structures ({n_features})', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Contour overlay
    axes[1, 2].imshow(image, cmap='gray')
    colors_contour = ['#FF1744', '#00E5FF', '#76FF03', '#FFEA00']
    for idx, contour in enumerate(contours):
        color = colors_contour[idx % len(colors_contour)]
        axes[1, 2].plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)
    axes[1, 2].set_title('Contour Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Vascular Segmentation — Lumen Detection & Contouring',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_vascular_segmentation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/04_vascular_segmentation.png")


def demo_cardiac_segmentation():
    """
    Demo: Cardiac chamber segmentation from ultrasound.
    """
    print("\n" + "=" * 60)
    print("  4.3 — Cardiac Chamber Segmentation")
    print("=" * 60)

    cardiac = generate_cardiac_phantom(512, 512, n_frames=1, seed=42)
    image = cardiac[0]

    # Denoise
    denoised = nlm_fast(image, filter_size=5, search_size=11, h=0.12)

    # Segment chambers (dark regions)
    chamber_mask = denoised < 0.1

    # Morphological cleanup
    struct = np.ones((7, 7))
    chamber_clean = binary_opening(chamber_mask, structure=struct, iterations=2)
    chamber_clean = binary_closing(chamber_clean, structure=struct, iterations=3)
    chamber_clean = binary_fill_holes(chamber_clean)

    # Label chambers
    labeled, n_chambers = label(chamber_clean)
    print(f"  Detected {n_chambers} cardiac chambers")

    # Find contours
    from skimage.measure import find_contours
    contours = find_contours(chamber_clean.astype(float), 0.5)

    # Detect wall (bright tissue around chambers)
    wall_mask = (denoised > 0.3) & binary_dilation(chamber_clean, iterations=15) & ~chamber_clean

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    us_cmap = create_ultrasound_colormap()

    axes[0, 0].imshow(image, cmap=us_cmap)
    axes[0, 0].set_title('Cardiac Ultrasound', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised, cmap=us_cmap)
    axes[0, 1].set_title('Denoised', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(chamber_mask, cmap='gray')
    axes[0, 2].set_title('Chamber Detection', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Labeled chambers
    axes[1, 0].imshow(labeled, cmap='Set1', vmin=0, vmax=max(4, n_chambers))
    axes[1, 0].set_title(f'Labeled Chambers ({n_chambers})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Wall + Chamber overlay
    overlay = np.zeros((*image.shape, 4))
    overlay[..., 3] = 0  # Transparent background
    overlay[chamber_clean, 0] = 0.2  # Blue chambers
    overlay[chamber_clean, 2] = 0.8
    overlay[chamber_clean, 3] = 0.4
    overlay[wall_mask, 0] = 0.9  # Red wall
    overlay[wall_mask, 1] = 0.2
    overlay[wall_mask, 3] = 0.3

    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Wall + Chamber Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Contour overlay (publication-quality)
    axes[1, 2].imshow(image, cmap=us_cmap)
    colors_c = ['#FF1744', '#00E5FF', '#76FF03', '#FFEA00']
    labels_c = ['Chamber 1', 'Chamber 2', 'Chamber 3', 'Chamber 4']
    for idx, contour in enumerate(contours[:4]):
        color = colors_c[idx % len(colors_c)]
        label_text = labels_c[idx] if idx < 4 else f'Region {idx + 1}'
        axes[1, 2].plot(contour[:, 1], contour[:, 0], color=color, linewidth=2,
                        label=label_text)
    axes[1, 2].legend(loc='upper right', fontsize=9,
                      facecolor='black', edgecolor='white', labelcolor='white')
    axes[1, 2].set_title('Contour Detection', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Cardiac Chamber Segmentation',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_cardiac_segmentation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/04_cardiac_segmentation.png")


def demo_watershed():
    """
    Demo: Watershed segmentation for tissue boundary detection.
    """
    print("\n" + "=" * 60)
    print("  4.4 — Watershed Segmentation")
    print("=" * 60)

    image = generate_tissue_phantom(256, 256, seed=42)
    denoised = nlm_fast(image, filter_size=5, search_size=11, h=0.15)

    try:
        labels = watershed_segmentation(denoised, n_markers=8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(labels, cmap='tab10')
        axes[1].set_title('Watershed Regions', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(labels, cmap='tab10', alpha=0.3)
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('Watershed Segmentation — Tissue Boundary Detection',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '04_watershed.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output/04_watershed.png")
    except Exception as e:
        print(f"  ⚠ Watershed failed: {e}. Skipping this demo.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FAST Ultrasound Tutorial — Segmentation               ║")
    print("║   Python Implementation                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_threshold_segmentation()
    demo_vascular_segmentation()
    demo_cardiac_segmentation()
    demo_watershed()

    print("\n" + "=" * 60)
    print("  All outputs saved to: output/")
    print("  Files generated:")
    print("    • 04_threshold_segmentation.png")
    print("    • 04_vascular_segmentation.png")
    print("    • 04_cardiac_segmentation.png")
    print("    • 04_watershed.png")
    print("=" * 60)
