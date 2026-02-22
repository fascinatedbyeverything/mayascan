"""MayaScan: Open-source archaeological LiDAR feature detection."""

from __future__ import annotations

import numpy as np

__version__ = "0.1.0"

from mayascan.detect import DetectionResult
from mayascan.visualize import compute_visualizations as _compute_visualizations
from mayascan.detect import run_detection as _run_detection

__all__ = [
    "__version__",
    "DetectionResult",
    "visualize",
    "detect",
    "process_dem",
]


def visualize(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute SVF, openness, and slope visualizations from a DEM.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size in the same unit as the elevation values (default 0.5).

    Returns
    -------
    np.ndarray
        Shape (3, H, W), dtype float32.  Channels: [SVF, openness, slope].
    """
    return _compute_visualizations(dem, resolution=resolution)


def detect(
    visualization: np.ndarray,
    model_path: str | None = None,
    confidence_threshold: float = 0.5,
) -> DetectionResult:
    """Run tiled U-Net inference on a visualization raster.

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)`` where *C* is typically 3.
    model_path : str or None
        Path to saved model weights.  If *None*, random weights are used.
    confidence_threshold : float
        Pixels below this confidence are reset to background.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    return _run_detection(
        visualization,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )


def process_dem(
    dem: np.ndarray,
    resolution: float = 0.5,
    model_path: str | None = None,
    confidence_threshold: float = 0.5,
) -> DetectionResult:
    """Run the full MayaScan pipeline: visualize a DEM then detect features.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (H, W).
    resolution : float
        Cell size (default 0.5).
    model_path : str or None
        Path to saved model weights.  If *None*, random weights are used.
    confidence_threshold : float
        Pixels below this confidence are reset to background.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    viz = _compute_visualizations(dem, resolution=resolution)
    return _run_detection(
        viz,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
