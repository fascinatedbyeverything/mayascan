"""Archaeological visualization rasters from DEMs.

Generates Sky-View Factor (SVF), positive openness, and slope from a
digital elevation model.  Uses *rvt-py* when available; falls back to
*scipy.ndimage* approximations so the module works without optional
dependencies.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Scale *arr* to [0, 1] as float32.  Constant arrays map to 0."""
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

def compute_slope(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute slope in degrees from a 2-D DEM using ``np.gradient``.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size in the same unit as the elevation values.

    Returns
    -------
    np.ndarray
        Slope in degrees, same shape as *dem*, dtype float32.
    """
    dy, dx = np.gradient(dem.astype(np.float64), resolution)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad).astype(np.float32)


# ---------------------------------------------------------------------------
# Sky-View Factor
# ---------------------------------------------------------------------------

def compute_svf(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute sky-view factor.

    Attempts to use ``rvt.vis.sky_view_factor``.  When *rvt-py* is not
    installed, falls back to a *scipy.ndimage.uniform_filter* approximation
    that highlights local depressions (lower SVF) vs. ridges (higher SVF).

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size in the same unit as the elevation values.

    Returns
    -------
    np.ndarray
        SVF array in [0, 1], shape (H, W), dtype float32.
    """
    try:
        from rvt.vis import sky_view_factor  # type: ignore[import-untyped]

        result = sky_view_factor(
            dem.astype(np.float64),
            resolution=resolution,
            compute_svf=True,
            compute_opns=False,
        )
        svf = result["svf"]
        return _normalize(svf)
    except ImportError:
        pass

    # Fallback: SVF ≈ 1 - normalised local relief
    from scipy.ndimage import uniform_filter  # noqa: E402

    dem64 = dem.astype(np.float64)
    radius = max(3, int(10 * resolution))
    size = 2 * radius + 1
    local_mean = uniform_filter(dem64, size=size, mode="reflect")
    # Points well below their neighbourhood get lower SVF.
    relief = dem64 - local_mean
    svf = 1.0 - _normalize(-relief)  # invert so depressions → low SVF
    return svf.astype(np.float32)


# ---------------------------------------------------------------------------
# Positive Openness
# ---------------------------------------------------------------------------

def compute_openness(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute positive openness.

    Attempts ``rvt.vis.sky_view_factor`` with ``compute_opns=True``.
    Falls back to a *scipy.ndimage.maximum_filter* approximation.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size in the same unit as the elevation values.

    Returns
    -------
    np.ndarray
        Openness array in [0, 1], shape (H, W), dtype float32.
    """
    try:
        from rvt.vis import sky_view_factor  # type: ignore[import-untyped]

        result = sky_view_factor(
            dem.astype(np.float64),
            resolution=resolution,
            compute_svf=False,
            compute_opns=True,
        )
        opns = result["opns"]
        return _normalize(opns)
    except ImportError:
        pass

    # Fallback: openness ≈ how much the point "opens" upward relative
    # to its local maximum.
    from scipy.ndimage import maximum_filter  # noqa: E402

    dem64 = dem.astype(np.float64)
    radius = max(3, int(10 * resolution))
    size = 2 * radius + 1
    local_max = maximum_filter(dem64, size=size, mode="reflect")
    # Openness is high where a point is close to the local max (ridges)
    # and low in deep depressions.
    diff = local_max - dem64
    openness = 1.0 - _normalize(diff)
    return openness.astype(np.float32)


# ---------------------------------------------------------------------------
# Combined visualizations
# ---------------------------------------------------------------------------

def compute_visualizations(
    dem: np.ndarray,
    resolution: float = 0.5,
) -> np.ndarray:
    """Stack SVF, openness, and slope into a (3, H, W) float32 array.

    Each channel is independently normalised to [0, 1].

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size.

    Returns
    -------
    np.ndarray
        Shape (3, H, W), dtype float32.  Channels: [SVF, openness, slope].
    """
    svf = _normalize(compute_svf(dem, resolution))
    opns = _normalize(compute_openness(dem, resolution))
    slope = _normalize(compute_slope(dem, resolution))
    return np.stack([svf, opns, slope], axis=0).astype(np.float32)
