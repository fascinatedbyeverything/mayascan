"""Feature density heatmap generation.

Creates Gaussian kernel density maps from detected features, useful for
visualizing settlement patterns and identifying areas of high archaeological
activity. Supports per-class and combined heatmaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from mayascan.features import Feature


def feature_density_map(
    features: list[Feature],
    shape: tuple[int, int],
    sigma: float = 20.0,
    weight_by_area: bool = False,
) -> np.ndarray:
    """Generate a Gaussian kernel density estimate from feature centroids.

    Parameters
    ----------
    features : list[Feature]
        Detected features with centroid coordinates.
    shape : tuple of int
        Output map shape (H, W).
    sigma : float
        Gaussian kernel standard deviation in pixels.
    weight_by_area : bool
        If True, weight each feature by its area.

    Returns
    -------
    np.ndarray
        Float32 density map, shape (H, W). Values are normalized to [0, 1].
    """
    h, w = shape
    density = np.zeros((h, w), dtype=np.float64)

    for feat in features:
        r = int(round(feat.centroid_row))
        c = int(round(feat.centroid_col))
        if 0 <= r < h and 0 <= c < w:
            weight = feat.area_m2 if weight_by_area else 1.0
            density[r, c] += weight

    if density.max() == 0:
        return np.zeros((h, w), dtype=np.float32)

    smoothed = gaussian_filter(density, sigma=sigma)

    # Normalize to [0, 1]
    max_val = smoothed.max()
    if max_val > 0:
        smoothed /= max_val

    return smoothed.astype(np.float32)


def class_density_maps(
    features: list[Feature],
    shape: tuple[int, int],
    sigma: float = 20.0,
) -> dict[str, np.ndarray]:
    """Generate per-class density maps.

    Parameters
    ----------
    features : list[Feature]
        Detected features.
    shape : tuple of int
        Output map shape (H, W).
    sigma : float
        Gaussian kernel standard deviation.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from class name to density map (H, W), float32, [0, 1].
    """
    by_class: dict[str, list[Feature]] = {}
    for f in features:
        by_class.setdefault(f.class_name, []).append(f)

    return {
        cls_name: feature_density_map(feats, shape, sigma=sigma)
        for cls_name, feats in by_class.items()
    }


def density_to_rgba(
    density: np.ndarray,
    colormap: str = "hot",
    alpha_scale: float = 0.8,
) -> np.ndarray:
    """Convert a density map to an RGBA image for overlay.

    Parameters
    ----------
    density : np.ndarray
        Float32 density map (H, W) with values in [0, 1].
    colormap : str
        Color scheme: "hot" (red-yellow), "cool" (blue-cyan),
        or "viridis" (blue-green-yellow).
    alpha_scale : float
        Maximum alpha value (0-1).

    Returns
    -------
    np.ndarray
        RGBA uint8 image, shape (H, W, 4).
    """
    h, w = density.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    d = np.clip(density, 0, 1)

    if colormap == "hot":
        # Black → Red → Yellow → White
        rgba[:, :, 0] = np.clip(d * 3, 0, 1) * 255          # R
        rgba[:, :, 1] = np.clip(d * 3 - 1, 0, 1) * 255      # G
        rgba[:, :, 2] = np.clip(d * 3 - 2, 0, 1) * 255      # B
    elif colormap == "cool":
        # Black → Blue → Cyan → White
        rgba[:, :, 0] = np.clip(d * 3 - 2, 0, 1) * 255
        rgba[:, :, 1] = np.clip(d * 3 - 1, 0, 1) * 255
        rgba[:, :, 2] = np.clip(d * 3, 0, 1) * 255
    else:  # viridis-like
        # Dark purple → Blue → Green → Yellow
        rgba[:, :, 0] = np.clip(1.5 * d - 0.5, 0, 1) * 255
        rgba[:, :, 1] = np.clip(d, 0, 1) * 255
        rgba[:, :, 2] = np.clip(1 - 1.5 * d, 0, 1) * 255

    # Alpha proportional to density
    rgba[:, :, 3] = (d * alpha_scale * 255).astype(np.uint8)

    return rgba


def save_density_png(
    density: np.ndarray,
    output_path: str,
    colormap: str = "hot",
) -> str:
    """Save a density map as a PNG with transparency.

    Parameters
    ----------
    density : np.ndarray
        Float32 density map (H, W).
    output_path : str
        Output PNG file path.
    colormap : str
        Color scheme.

    Returns
    -------
    str
        The output file path.
    """
    from PIL import Image

    rgba = density_to_rgba(density, colormap=colormap)
    Image.fromarray(rgba, "RGBA").save(output_path)
    return output_path
