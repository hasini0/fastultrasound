"""
Synthetic Ultrasound Data Generator
====================================
Generates realistic synthetic ultrasound data (2D frames, 3D volumes, 4D time-series)
with proper speckle noise patterns, tissue-like structures, and cardiac/vascular anatomy.

This module provides test data for demonstrating ultrasound processing pipelines
when real clinical data is not available.

Author: Sai Hasini Dandapanthula
Project: FAST Ultrasound Processing & 3D Visualization
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from scipy.interpolate import RegularGridInterpolator


def generate_speckle_noise(shape, intensity=0.3, seed=None):
    """
    Generate realistic ultrasound speckle noise using Rayleigh distribution.

    Speckle noise in ultrasound is caused by constructive and destructive interference
    of backscattered waves from sub-resolution scatterers within tissue.

    Parameters
    ----------
    shape : tuple
        Shape of the output noise array.
    intensity : float
        Controls the intensity/variance of the speckle noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Speckle noise array with values in [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)

    # Rayleigh distribution models ultrasound speckle well
    noise = np.random.rayleigh(scale=intensity, size=shape)
    noise = noise / noise.max()
    return noise.astype(np.float32)


def generate_tissue_phantom(height=512, width=512, seed=42):
    """
    Generate a synthetic 2D ultrasound tissue phantom with layered structures.

    Creates a B-mode ultrasound image with:
    - Depth-dependent attenuation (TGC-like falloff)
    - Layered tissue regions with different echogenicity
    - Circular cyst-like structures (anechoic regions)
    - Hyperechoic boundaries between tissue layers

    Parameters
    ----------
    height : int
        Image height (depth direction).
    width : int
        Image width (lateral direction).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Synthetic B-mode ultrasound image, shape (height, width), float32 in [0, 1].
    """
    np.random.seed(seed)
    image = np.zeros((height, width), dtype=np.float32)

    # Create depth-dependent attenuation (TGC simulation)
    depth_attenuation = np.linspace(1.0, 0.3, height).reshape(-1, 1)
    depth_attenuation = np.broadcast_to(depth_attenuation, (height, width)).copy()

    # Create tissue layers with different echogenicity
    layer_boundaries = [0, int(0.15 * height), int(0.35 * height),
                        int(0.55 * height), int(0.75 * height), height]
    layer_echogenicity = [0.8, 0.4, 0.7, 0.3, 0.6]

    for i in range(len(layer_echogenicity)):
        start, end = layer_boundaries[i], layer_boundaries[i + 1]
        # Add slight wave to boundary for realism
        for col in range(width):
            wave = int(5 * np.sin(2 * np.pi * col / 80))
            s = max(0, start + wave)
            e = min(height, end + wave)
            image[s:e, col] = layer_echogenicity[i]

        # Add hyperechoic boundary line between layers
        if i < len(layer_echogenicity) - 1:
            boundary = layer_boundaries[i + 1]
            for col in range(width):
                wave = int(5 * np.sin(2 * np.pi * col / 80))
                b = boundary + wave
                if 0 < b < height - 2:
                    image[b:b + 2, col] = 1.0

    # Add circular anechoic cysts
    cyst_centers = [(int(0.25 * height), int(0.3 * width), 25),
                    (int(0.45 * height), int(0.7 * width), 18),
                    (int(0.65 * height), int(0.5 * width), 30)]

    for cy, cx, radius in cyst_centers:
        y, x = np.ogrid[:height, :width]
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        image[mask] = 0.02  # Nearly anechoic

        # Posterior acoustic enhancement below cyst
        enhancement_mask = ((y - cy) ** 2 + (x - cx) ** 2 > radius ** 2) & \
                           (y > cy + radius) & (y < cy + radius + 15) & \
                           (np.abs(x - cx) < radius)
        image[enhancement_mask] *= 1.4

    # Add hyperechoic bright spots (calcifications)
    n_calcifications = 8
    for _ in range(n_calcifications):
        cy = np.random.randint(50, height - 50)
        cx = np.random.randint(50, width - 50)
        size = np.random.randint(2, 5)
        image[cy:cy + size, cx:cx + size] = 1.0
        # Acoustic shadow below calcification
        image[cy + size:cy + size + 20, cx:cx + size] *= 0.2

    # Apply speckle noise
    speckle = generate_speckle_noise((height, width), intensity=0.25, seed=seed)
    image = image * (0.7 + 0.6 * speckle)

    # Apply depth attenuation
    image *= depth_attenuation

    # Smooth slightly for realism
    image = gaussian_filter(image, sigma=1.2)

    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    return image


def generate_cardiac_phantom(height=512, width=512, n_frames=30, seed=42):
    """
    Generate a synthetic 2D+t cardiac ultrasound sequence (apical four-chamber view).

    Creates a time-series of ultrasound frames simulating cardiac motion:
    - Elliptical chamber walls that contract and expand
    - Valve motion simulation
    - Realistic speckle patterns that deform with the tissue

    Parameters
    ----------
    height : int
        Image height.
    width : int
        Image width.
    n_frames : int
        Number of temporal frames (one cardiac cycle).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_frames, height, width), float32 in [0, 1].
    """
    np.random.seed(seed)
    frames = np.zeros((n_frames, height, width), dtype=np.float32)

    center_y, center_x = height // 2, width // 2

    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames  # Cardiac cycle phase

        # Contraction factor (systole = smaller, diastole = larger)
        contraction = 1.0 - 0.2 * np.sin(phase)

        # Create background tissue
        frame = np.ones((height, width), dtype=np.float32) * 0.35

        # Outer heart wall (myocardium)
        y, x = np.ogrid[:height, :width]
        outer_a = int(180 * contraction)  # Semi-major axis
        outer_b = int(140 * contraction)  # Semi-minor axis
        outer_wall = ((y - center_y) / outer_a) ** 2 + ((x - center_x) / outer_b) ** 2 <= 1
        frame[outer_wall] = 0.6

        # Inner chamber (blood pool - anechoic)
        inner_a = int(140 * contraction)
        inner_b = int(100 * contraction)
        inner_chamber = ((y - center_y) / inner_a) ** 2 + ((x - center_x) / inner_b) ** 2 <= 1
        frame[inner_chamber] = 0.05

        # Septum (dividing wall)
        septum_width = 12
        septum = (np.abs(x - center_x) < septum_width // 2) & outer_wall & ~inner_chamber
        # Make the septum extend into the chamber
        septum_extended = (np.abs(x - center_x) < septum_width // 2) & outer_wall
        frame[septum_extended] = 0.7

        # Four chambers
        # Left chambers (x < center)
        left_upper = inner_chamber & (y < center_y) & (x < center_x - septum_width // 2)
        left_lower = inner_chamber & (y >= center_y) & (x < center_x - septum_width // 2)
        right_upper = inner_chamber & (y < center_y) & (x > center_x + septum_width // 2)
        right_lower = inner_chamber & (y >= center_y) & (x > center_x + septum_width // 2)

        frame[left_upper] = 0.04
        frame[left_lower] = 0.06
        frame[right_upper] = 0.05
        frame[right_lower] = 0.04

        # Valve leaflets (thin bright lines at chamber boundaries)
        valve_y = center_y
        valve_thickness = 2
        valve_motion = int(8 * np.sin(phase))
        v_y = valve_y + valve_motion
        if 0 < v_y < height - valve_thickness:
            valve_mask = (np.abs(y - v_y) < valve_thickness) & inner_chamber
            frame[valve_mask] = 0.9

        # Add depth-dependent attenuation
        depth_atten = np.linspace(1.0, 0.5, height).reshape(-1, 1)
        frame *= depth_atten

        # Add speckle noise
        speckle = generate_speckle_noise((height, width), intensity=0.2, seed=seed + t)
        frame = frame * (0.7 + 0.6 * speckle)

        # Smooth for realism
        frame = gaussian_filter(frame, sigma=1.0)
        frame = np.clip(frame, 0, 1)
        frames[t] = frame

    return frames


def generate_vascular_phantom(height=400, width=400, seed=42):
    """
    Generate a synthetic vascular ultrasound image (cross-section of artery and vein).

    Parameters
    ----------
    height, width : int
        Image dimensions.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (height, width), float32 in [0, 1].
    """
    np.random.seed(seed)
    image = np.ones((height, width), dtype=np.float32) * 0.4

    y, x = np.ogrid[:height, :width]

    # Artery (circular, thick wall, pulsatile)
    artery_cx, artery_cy = int(0.35 * width), int(0.45 * height)
    artery_outer_r = 45
    artery_inner_r = 35
    artery_wall = ((y - artery_cy) ** 2 + (x - artery_cx) ** 2 <= artery_outer_r ** 2) & \
                  ((y - artery_cy) ** 2 + (x - artery_cx) ** 2 > artery_inner_r ** 2)
    artery_lumen = (y - artery_cy) ** 2 + (x - artery_cx) ** 2 <= artery_inner_r ** 2

    image[artery_wall] = 0.8  # Bright vessel wall
    image[artery_lumen] = 0.03  # Dark blood pool

    # Vein (slightly oval, thinner wall, compressible)
    vein_cx, vein_cy = int(0.65 * width), int(0.5 * height)
    vein_outer_a, vein_outer_b = 50, 35
    vein_inner_a, vein_inner_b = 44, 29
    vein_wall = (((y - vein_cy) / vein_outer_b) ** 2 + ((x - vein_cx) / vein_outer_a) ** 2 <= 1) & \
                (((y - vein_cy) / vein_inner_b) ** 2 + ((x - vein_cx) / vein_inner_a) ** 2 > 1)
    vein_lumen = ((y - vein_cy) / vein_inner_b) ** 2 + ((x - vein_cx) / vein_inner_a) ** 2 <= 1

    image[vein_wall] = 0.65  # Thinner, less echogenic wall
    image[vein_lumen] = 0.05  # Blood pool

    # Add surrounding tissue texture
    tissue_texture = generate_speckle_noise((height, width), intensity=0.2, seed=seed)
    muscle_fiber = np.zeros_like(image)
    for row in range(0, height, 8):
        muscle_fiber[row:row + 1, :] = 0.15
    image += muscle_fiber * 0.3

    # Apply speckle
    speckle = generate_speckle_noise((height, width), intensity=0.25, seed=seed)
    image = image * (0.65 + 0.7 * speckle)

    # Depth attenuation
    depth = np.linspace(1.0, 0.6, height).reshape(-1, 1)
    image *= depth

    image = gaussian_filter(image, sigma=0.8)
    return np.clip(image, 0, 1).astype(np.float32)


def generate_3d_volume(nx=128, ny=128, nz=128, seed=42):
    """
    Generate a synthetic 3D ultrasound volume with embedded structures.

    Creates a 3D volume containing:
    - A spherical structure (simulating a fetal head or cyst)
    - An ellipsoidal structure (simulating an organ)
    - Layered tissue planes
    - 3D speckle noise

    Parameters
    ----------
    nx, ny, nz : int
        Volume dimensions.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (nz, ny, nx), float32 in [0, 1].
    """
    np.random.seed(seed)
    volume = np.ones((nz, ny, nx), dtype=np.float32) * 0.3

    z, y, x = np.ogrid[:nz, :ny, :nx]
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    # Spherical cyst
    sphere_r = min(nx, ny, nz) // 5
    sphere_mask = (z - cz) ** 2 + (y - cy) ** 2 + (x - int(cx * 0.6)) ** 2 <= sphere_r ** 2
    sphere_wall = ((z - cz) ** 2 + (y - cy) ** 2 + (x - int(cx * 0.6)) ** 2 <= (sphere_r + 4) ** 2) & ~sphere_mask
    volume[sphere_wall] = 0.85
    volume[sphere_mask] = 0.03

    # Ellipsoidal organ
    ea, eb, ec = nx // 4, ny // 5, nz // 6
    organ_cx = int(cx * 1.4)
    ellipsoid = ((z - cz) / ec) ** 2 + ((y - cy) / eb) ** 2 + ((x - organ_cx) / ea) ** 2 <= 1
    ellipsoid_inner = ((z - cz) / (ec - 3)) ** 2 + ((y - cy) / (eb - 3)) ** 2 + ((x - organ_cx) / (ea - 3)) ** 2 <= 1
    volume[ellipsoid & ~ellipsoid_inner] = 0.75
    volume[ellipsoid_inner] = 0.5

    # Horizontal tissue planes
    for plane_z in [nz // 4, 3 * nz // 4]:
        volume[plane_z:plane_z + 2, :, :] = 0.9

    # 3D speckle noise
    speckle_3d = generate_speckle_noise((nz, ny, nx), intensity=0.2, seed=seed)
    volume = volume * (0.6 + 0.8 * speckle_3d)

    # Depth attenuation along z-axis
    depth_atten = np.linspace(1.0, 0.4, nz).reshape(-1, 1, 1)
    volume *= depth_atten

    volume = gaussian_filter(volume, sigma=0.8)
    return np.clip(volume, 0, 1).astype(np.float32)


def generate_4d_volume(nx=64, ny=64, nz=64, n_frames=20, seed=42):
    """
    Generate a synthetic 4D (3D + time) ultrasound volume sequence.

    Simulates a pulsating spherical structure (e.g., cardiac chamber)
    that contracts and expands over time.

    Parameters
    ----------
    nx, ny, nz : int
        Spatial volume dimensions.
    n_frames : int
        Number of temporal frames.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_frames, nz, ny, nx), float32 in [0, 1].
    """
    np.random.seed(seed)
    volumes = np.zeros((n_frames, nz, ny, nx), dtype=np.float32)

    cz, cy, cx = nz // 2, ny // 2, nx // 2
    z, y, x = np.ogrid[:nz, :ny, :nx]

    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames
        contraction = 1.0 - 0.25 * np.sin(phase)

        vol = np.ones((nz, ny, nx), dtype=np.float32) * 0.35

        # Pulsating sphere
        radius = int(min(nx, ny, nz) * 0.3 * contraction)
        wall_thickness = 4

        sphere = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        sphere_inner = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= (radius - wall_thickness) ** 2

        vol[sphere & ~sphere_inner] = 0.8  # Wall
        vol[sphere_inner] = 0.04  # Cavity

        # Speckle
        speckle = generate_speckle_noise((nz, ny, nx), intensity=0.18, seed=seed + t)
        vol = vol * (0.6 + 0.8 * speckle)

        # Depth attenuation
        depth = np.linspace(1.0, 0.5, nz).reshape(-1, 1, 1)
        vol *= depth

        vol = gaussian_filter(vol, sigma=0.6)
        volumes[t] = np.clip(vol, 0, 1)

    return volumes.astype(np.float32)


def generate_sector_beamspace_data(n_scanlines=256, n_samples=700, seed=42):
    """
    Generate synthetic beamspace (pre-scan-converted) ultrasound data.

    This simulates raw data as it would come from a phased array transducer
    before scan conversion to a sector image.

    Parameters
    ----------
    n_scanlines : int
        Number of scan lines (lateral samples).
    n_samples : int
        Number of depth samples per scan line.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_scanlines), float32 in [0, 1].
    """
    np.random.seed(seed)
    data = np.zeros((n_samples, n_scanlines), dtype=np.float32)

    # Background tissue
    data += 0.3

    # Add depth-varying echogenicity layers
    for depth in [150, 300, 450, 600]:
        width = np.random.randint(20, 60)
        echo = np.random.uniform(0.4, 0.8)
        data[depth:depth + width, :] = echo

    # Add a bright reflector (arc shape in beamspace = flat surface)
    for scanline in range(n_scanlines):
        depth = 200 + int(30 * np.sin(np.pi * scanline / n_scanlines))
        if 0 <= depth < n_samples - 2:
            data[depth:depth + 2, scanline] = 1.0

    # Add anechoic region (cyst in beamspace)
    cy, cx = 400, n_scanlines // 2
    for s in range(n_scanlines):
        for d in range(n_samples):
            if (d - cy) ** 2 + (s - cx) ** 2 < 25 ** 2:
                data[d, s] = 0.02

    # Speckle
    speckle = generate_speckle_noise((n_samples, n_scanlines), intensity=0.2, seed=seed)
    data = data * (0.6 + 0.8 * speckle)

    # Depth attenuation
    depth_atten = np.linspace(1.0, 0.3, n_samples).reshape(-1, 1)
    data *= depth_atten

    return np.clip(data, 0, 1).astype(np.float32)


def generate_iq_data(n_samples=512, n_scanlines=256, frequency=5e6, seed=42):
    """
    Generate synthetic IQ (In-phase / Quadrature) ultrasound data.

    IQ data represents the complex-valued RF signal after demodulation.
    This is what would be input to envelope detection and log compression.

    Parameters
    ----------
    n_samples : int
        Number of depth samples.
    n_scanlines : int
        Number of scan lines.
    frequency : float
        Center frequency in Hz.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_scanlines, 2), float32. Channel 0 = I, Channel 1 = Q.
    """
    np.random.seed(seed)

    # Generate envelope (amplitude of scatterers)
    envelope = generate_sector_beamspace_data(n_scanlines, n_samples, seed)

    # Generate phase variation
    t = np.linspace(0, 1, n_samples).reshape(-1, 1)
    phase = 2 * np.pi * frequency * t * 1e-7 + np.random.uniform(0, 2 * np.pi, (1, n_scanlines))

    # IQ components
    I = envelope * np.cos(phase)
    Q = envelope * np.sin(phase)

    # Add noise
    I += np.random.normal(0, 0.02, I.shape)
    Q += np.random.normal(0, 0.02, Q.shape)

    iq_data = np.stack([I, Q], axis=-1).astype(np.float32)
    return iq_data


if __name__ == "__main__":
    """Quick test of all generators."""
    import matplotlib.pyplot as plt

    print("Generating synthetic ultrasound data...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 2D Tissue phantom
    tissue = generate_tissue_phantom()
    axes[0, 0].imshow(tissue, cmap='gray')
    axes[0, 0].set_title('2D Tissue Phantom')
    axes[0, 0].axis('off')

    # Cardiac frame
    cardiac = generate_cardiac_phantom(n_frames=1)
    axes[0, 1].imshow(cardiac[0], cmap='gray')
    axes[0, 1].set_title('Cardiac Phantom')
    axes[0, 1].axis('off')

    # Vascular phantom
    vascular = generate_vascular_phantom()
    axes[0, 2].imshow(vascular, cmap='gray')
    axes[0, 2].set_title('Vascular Phantom')
    axes[0, 2].axis('off')

    # 3D volume slice
    volume = generate_3d_volume()
    axes[1, 0].imshow(volume[64], cmap='gray')
    axes[1, 0].set_title('3D Volume (Axial Slice)')
    axes[1, 0].axis('off')

    # Beamspace data
    beamspace = generate_sector_beamspace_data()
    axes[1, 1].imshow(beamspace, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Beamspace Data')
    axes[1, 1].axis('off')

    # IQ data (amplitude)
    iq = generate_iq_data()
    amplitude = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
    axes[1, 2].imshow(amplitude, cmap='gray', aspect='auto')
    axes[1, 2].set_title('IQ Data (Amplitude)')
    axes[1, 2].axis('off')

    plt.suptitle('Synthetic Ultrasound Data Gallery', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/data_generator_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done! Output saved to output/data_generator_test.png")
