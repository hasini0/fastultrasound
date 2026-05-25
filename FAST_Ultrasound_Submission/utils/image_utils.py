"""
Ultrasound Image Processing Utilities
=======================================
Core image processing functions for ultrasound data, implementing the same
algorithms described in the FAST framework tutorial:
- Non-Local Means denoising / de-speckling
- Scan conversion (beamspace → sector/linear image)
- Envelope detection and log compression
- Colormap utilities

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.interpolate import RegularGridInterpolator


def non_local_means(image, filter_size=3, search_size=11, smoothing_amount=0.2,
                    input_weight=0.5):
    """
    Non-Local Means (NLM) denoising filter for ultrasound de-speckling.

    This is a Python implementation equivalent to FAST's NonLocalMeans filter.
    NLM works by averaging pixels with similar local neighborhoods, preserving
    edges while removing speckle noise.

    Parameters
    ----------
    image : np.ndarray
        Input 2D grayscale image, float32 in [0, 1].
    filter_size : int
        Size of the local patch for comparison (must be odd).
    search_size : int
        Size of the search window (must be odd).
    smoothing_amount : float
        Controls the degree of smoothing (h parameter). Higher = more smoothing.
    input_weight : float
        Weight given to the original input pixel (0 to 1).

    Returns
    -------
    np.ndarray
        Denoised image, same shape as input.
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image")

    h, w = image.shape
    half_filter = filter_size // 2
    half_search = search_size // 2
    h_sq = smoothing_amount ** 2

    # Pad image
    padded = np.pad(image, half_search + half_filter, mode='reflect')
    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            pi = i + half_search + half_filter
            pj = j + half_search + half_filter

            # Extract reference patch
            ref_patch = padded[pi - half_filter:pi + half_filter + 1,
                        pj - half_filter:pj + half_filter + 1]

            weight_sum = 0.0
            pixel_sum = 0.0

            # Search over neighborhood
            for si in range(-half_search, half_search + 1):
                for sj in range(-half_search, half_search + 1):
                    ni = pi + si
                    nj = pj + sj

                    # Extract comparison patch
                    cmp_patch = padded[ni - half_filter:ni + half_filter + 1,
                                nj - half_filter:nj + half_filter + 1]

                    # Compute squared difference
                    diff = np.sum((ref_patch - cmp_patch) ** 2)
                    diff /= (filter_size ** 2)

                    # Compute weight
                    weight = np.exp(-diff / h_sq)
                    weight_sum += weight
                    pixel_sum += weight * padded[ni, nj]

            if weight_sum > 0:
                result[i, j] = pixel_sum / weight_sum

    # Blend with original
    result = input_weight * image + (1 - input_weight) * result
    return np.clip(result, 0, 1).astype(np.float32)


def nlm_fast(image, filter_size=5, search_size=11, h=0.1):
    """
    Fast approximate Non-Local Means using vectorized operations.

    Much faster than pixel-by-pixel NLM for real-time use.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    filter_size : int
        Patch size.
    search_size : int
        Search window size.
    h : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    pad = search_size // 2 + filter_size // 2
    padded = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image)
    weight_sum = np.zeros_like(image)
    rows, cols = image.shape

    half_search = search_size // 2
    half_filter = filter_size // 2

    for di in range(-half_search, half_search + 1):
        for dj in range(-half_search, half_search + 1):
            # Shifted image
            shifted = padded[pad + di:pad + di + rows, pad + dj:pad + dj + cols]

            # Compute patch distance using uniform filter (fast)
            diff_sq = (image - shifted) ** 2
            patch_dist = uniform_filter(diff_sq, size=filter_size)

            # Weight
            w = np.exp(-patch_dist / (h ** 2 + 1e-10))
            result += w * shifted
            weight_sum += w

    result /= (weight_sum + 1e-10)
    return np.clip(result, 0, 1).astype(np.float32)


def scan_convert_sector(beamspace_data, width=1024, height=1024,
                        start_depth=0.0, end_depth=120.0,
                        start_angle=-0.785398, end_angle=0.785398):
    """
    Scan conversion for sector/phased array ultrasound data.

    Converts beamspace data (scanline × depth) to a sector-shaped image in
    Cartesian coordinates. Equivalent to FAST's ScanConverter.

    Parameters
    ----------
    beamspace_data : np.ndarray
        Input beamspace data, shape (n_depth_samples, n_scanlines).
    width : int
        Output image width.
    height : int
        Output image height.
    start_depth : float
        Start depth (physical units).
    end_depth : float
        End depth (physical units).
    start_angle : float
        Start angle in radians.
    end_angle : float
        End angle in radians.

    Returns
    -------
    np.ndarray
        Scan-converted sector image, shape (height, width), float32.
    """
    n_samples, n_scanlines = beamspace_data.shape

    # Create interpolator for beamspace data
    depth_coords = np.linspace(start_depth, end_depth, n_samples)
    angle_coords = np.linspace(start_angle, end_angle, n_scanlines)
    interpolator = RegularGridInterpolator(
        (depth_coords, angle_coords),
        beamspace_data,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Create output Cartesian grid
    output = np.zeros((height, width), dtype=np.float32)

    # Compute the extent of the sector in Cartesian coordinates
    x_min = end_depth * np.sin(start_angle)
    x_max = end_depth * np.sin(end_angle)
    y_min = start_depth
    y_max = end_depth

    x_coords = np.linspace(x_min, x_max, width)
    y_coords = np.linspace(y_max, y_min, height)  # Flip for image coordinates

    xx, yy = np.meshgrid(x_coords, y_coords)

    # Convert Cartesian to polar
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)

    # Interpolate
    points = np.stack([r.ravel(), theta.ravel()], axis=-1)
    output = interpolator(points).reshape(height, width)

    return np.clip(output, 0, 1).astype(np.float32)


def scan_convert_linear(beamspace_data, width=1024, height=1024,
                        start_depth=0.0, end_depth=120.0,
                        left=-50.0, right=50.0):
    """
    Scan conversion for linear array ultrasound data.

    Parameters
    ----------
    beamspace_data : np.ndarray
        Input beamspace data, shape (n_depth_samples, n_scanlines).
    width, height : int
        Output image dimensions.
    start_depth, end_depth : float
        Depth range.
    left, right : float
        Lateral extent.

    Returns
    -------
    np.ndarray
        Scan-converted linear image, shape (height, width), float32.
    """
    n_samples, n_scanlines = beamspace_data.shape

    depth_coords = np.linspace(start_depth, end_depth, n_samples)
    lateral_coords = np.linspace(left, right, n_scanlines)
    interpolator = RegularGridInterpolator(
        (depth_coords, lateral_coords),
        beamspace_data,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    x_coords = np.linspace(left, right, width)
    y_coords = np.linspace(start_depth, end_depth, height)
    xx, yy = np.meshgrid(x_coords, y_coords)

    points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    output = interpolator(points).reshape(height, width)

    return np.clip(output, 0, 1).astype(np.float32)


def envelope_detection(iq_data):
    """
    Compute the envelope (amplitude) of IQ ultrasound data.

    The envelope is the magnitude of the complex analytic signal:
        envelope = sqrt(I^2 + Q^2)

    Equivalent to FAST's EnvelopeAndLogCompressor (envelope part).

    Parameters
    ----------
    iq_data : np.ndarray
        IQ data, shape (..., 2) where channel 0=I, channel 1=Q.

    Returns
    -------
    np.ndarray
        Envelope (amplitude) data.
    """
    I = iq_data[..., 0]
    Q = iq_data[..., 1]
    envelope = np.sqrt(I ** 2 + Q ** 2)
    return envelope.astype(np.float32)


def log_compress(data, dynamic_range_db=60, gain_db=0):
    """
    Log compression of ultrasound data for display.

    Converts linear amplitude data to logarithmic scale (dB) and maps
    to display range [0, 1].

    Equivalent to FAST's EnvelopeAndLogCompressor (log compression part).

    Parameters
    ----------
    data : np.ndarray
        Input amplitude data (positive values).
    dynamic_range_db : float
        Dynamic range in dB. Data below this range from the maximum is clipped.
    gain_db : float
        Gain adjustment in dB.

    Returns
    -------
    np.ndarray
        Log-compressed data in [0, 1].
    """
    # Avoid log(0)
    data = np.maximum(data, 1e-10)

    # Convert to dB
    data_db = 20 * np.log10(data / data.max()) + gain_db

    # Apply dynamic range
    data_db = np.clip(data_db, -dynamic_range_db, 0)

    # Normalize to [0, 1]
    result = (data_db + dynamic_range_db) / dynamic_range_db

    return result.astype(np.float32)


def envelope_and_log_compress(iq_data, dynamic_range_db=60, gain_db=0):
    """
    Combined envelope detection and log compression pipeline.

    Equivalent to FAST's EnvelopeAndLogCompressor process object.

    Parameters
    ----------
    iq_data : np.ndarray
        IQ data with shape (..., 2).
    dynamic_range_db : float
        Dynamic range in dB.
    gain_db : float
        Gain in dB.

    Returns
    -------
    np.ndarray
        Processed data in [0, 1].
    """
    env = envelope_detection(iq_data)
    compressed = log_compress(env, dynamic_range_db, gain_db)
    return compressed


def create_ultrasound_colormap():
    """
    Create an ultrasound-specific colormap (S-curve with blue tint).

    Equivalent to FAST's Colormap.Ultrasound() which applies an S-curve
    colormap with a hint of blue.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom ultrasound colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # S-curve colormap with slight blue tint (matches FAST's Ultrasound colormap)
    colors = [
        (0.0, 0.0, 0.02),     # Very dark blue-black
        (0.05, 0.05, 0.10),   # Dark blue
        (0.15, 0.13, 0.20),   # Dark blue-grey
        (0.35, 0.30, 0.38),   # Blue-grey
        (0.60, 0.55, 0.58),   # Light grey
        (0.80, 0.78, 0.75),   # Warm grey
        (0.95, 0.93, 0.88),   # Near white
        (1.0, 1.0, 0.98),     # White with warm tint
    ]

    cmap = LinearSegmentedColormap.from_list('ultrasound', colors, N=256)
    return cmap


def create_thermal_colormap():
    """
    Create a thermal/hot colormap for ultrasound visualization.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.0, 0.0, 0.0),     # Black
        (0.3, 0.0, 0.0),     # Dark red
        (0.7, 0.2, 0.0),     # Orange-red
        (1.0, 0.6, 0.0),     # Orange
        (1.0, 0.9, 0.4),     # Yellow
        (1.0, 1.0, 1.0),     # White
    ]

    cmap = LinearSegmentedColormap.from_list('thermal_us', colors, N=256)
    return cmap


def create_doppler_colormap():
    """
    Create a color Doppler colormap (blue-black-red).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.0, 0.0, 0.8),     # Blue (flow toward)
        (0.0, 0.0, 0.4),
        (0.0, 0.0, 0.0),     # Black (no flow)
        (0.4, 0.0, 0.0),
        (0.8, 0.0, 0.0),     # Red (flow away)
    ]

    cmap = LinearSegmentedColormap.from_list('doppler', colors, N=256)
    return cmap


def ultrasound_image_crop(image, threshold_vertical=30, threshold_horizontal=10):
    """
    Automatic ultrasound sector cropping.

    Removes padding and scanner GUI from ultrasound images by analyzing
    the distribution of non-zero pixels in rows and columns.

    Equivalent to FAST's UltrasoundImageCropper.

    Parameters
    ----------
    image : np.ndarray
        Input ultrasound image (2D).
    threshold_vertical : int
        Minimum number of non-zero pixels per row to keep.
    threshold_horizontal : int
        Minimum number of non-zero pixels per column to keep.

    Returns
    -------
    np.ndarray
        Cropped image.
    tuple
        Crop coordinates (top, bottom, left, right).
    """
    # Threshold the image
    binary = (image > 0.05).astype(np.uint8)

    # Count non-zero pixels in rows and columns
    row_counts = np.sum(binary, axis=1)
    col_counts = np.sum(binary, axis=0)

    # Find valid rows and columns
    valid_rows = np.where(row_counts > threshold_vertical)[0]
    valid_cols = np.where(col_counts > threshold_horizontal)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return image, (0, image.shape[0], 0, image.shape[1])

    top, bottom = valid_rows[0], valid_rows[-1] + 1
    left, right = valid_cols[0], valid_cols[-1] + 1

    cropped = image[top:bottom, left:right]
    return cropped, (top, bottom, left, right)


def block_matching(frame1, frame2, block_size=13, search_size=11, metric='sad'):
    """
    Block matching for speckle tracking between two ultrasound frames.

    Finds the displacement of each block in frame1 by searching for the
    best match in frame2.

    Equivalent to FAST's BlockMatching process object.

    Parameters
    ----------
    frame1 : np.ndarray
        Reference frame (2D).
    frame2 : np.ndarray
        Target frame (2D).
    block_size : int
        Size of the matching block.
    search_size : int
        Size of the search region.
    metric : str
        Matching metric: 'sad' (sum of absolute differences),
        'ssd' (sum of squared differences), or 'ncc' (normalized cross-correlation).

    Returns
    -------
    np.ndarray
        Displacement field, shape (H, W, 2) — [dy, dx] per pixel.
    """
    h, w = frame1.shape
    half_block = block_size // 2
    half_search = search_size // 2

    # Downsample for speed
    step = max(1, block_size // 2)
    n_blocks_y = (h - 2 * (half_block + half_search)) // step
    n_blocks_x = (w - 2 * (half_block + half_search)) // step

    disp_y = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)
    disp_x = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

    pad = half_block + half_search
    padded2 = np.pad(frame2, pad, mode='reflect')

    for bi in range(n_blocks_y):
        for bj in range(n_blocks_x):
            cy = pad + bi * step
            cx = pad + bj * step

            # Reference block from frame1
            ry = bi * step + half_search
            rx = bj * step + half_search
            ref_block = frame1[ry:ry + block_size, rx:rx + block_size]

            if ref_block.shape != (block_size, block_size):
                continue

            best_score = np.inf if metric != 'ncc' else -np.inf
            best_dy, best_dx = 0, 0

            for dy in range(-half_search, half_search + 1):
                for dx in range(-half_search, half_search + 1):
                    sy = cy + dy
                    sx = cx + dx
                    search_block = padded2[sy:sy + block_size, sx:sx + block_size]

                    if search_block.shape != (block_size, block_size):
                        continue

                    if metric == 'sad':
                        score = np.sum(np.abs(ref_block - search_block))
                        if score < best_score:
                            best_score = score
                            best_dy, best_dx = dy, dx
                    elif metric == 'ssd':
                        score = np.sum((ref_block - search_block) ** 2)
                        if score < best_score:
                            best_score = score
                            best_dy, best_dx = dy, dx
                    elif metric == 'ncc':
                        ref_norm = ref_block - ref_block.mean()
                        srch_norm = search_block - search_block.mean()
                        denom = np.sqrt(np.sum(ref_norm ** 2) * np.sum(srch_norm ** 2))
                        if denom > 1e-10:
                            score = np.sum(ref_norm * srch_norm) / denom
                        else:
                            score = 0
                        if score > best_score:
                            best_score = score
                            best_dy, best_dx = dy, dx

            disp_y[bi, bj] = best_dy
            disp_x[bi, bj] = best_dx

    # Upsample to full resolution
    from scipy.ndimage import zoom
    scale_y = h / n_blocks_y
    scale_x = w / n_blocks_x
    full_disp_y = zoom(disp_y, (scale_y, scale_x), order=1)[:h, :w]
    full_disp_x = zoom(disp_x, (scale_y, scale_x), order=1)[:h, :w]

    displacement = np.stack([full_disp_y, full_disp_x], axis=-1)
    return displacement


if __name__ == "__main__":
    """Quick test of processing utilities."""
    from data_generator import generate_tissue_phantom, generate_sector_beamspace_data, generate_iq_data
    import matplotlib.pyplot as plt

    print("Testing image processing utilities...")

    # Test NLM denoising
    phantom = generate_tissue_phantom(256, 256)
    denoised = nlm_fast(phantom, filter_size=5, search_size=11, h=0.12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('NLM Denoised')
    plt.savefig('output/nlm_test.png', dpi=150)
    plt.show()
    print("NLM test done!")
