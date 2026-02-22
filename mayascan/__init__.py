"""MayaScan: Open-source archaeological LiDAR feature detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__version__ = "0.4.0"

from mayascan.detect import DetectionResult, GeoInfo
from mayascan.visualize import compute_visualizations as _compute_visualizations
from mayascan.detect import run_detection as _run_detection
from mayascan.detect import run_detection_v2 as _run_detection_v2
from mayascan.detect import discover_v2_models

from mayascan.report import generate_report, report_to_text, report_to_html, save_report
from mayascan.features import Feature, extract_features, filter_features, feature_summary
from mayascan.augment import augment_sample, cutmix
from mayascan.ensemble import average_probabilities, majority_vote, merge_results

__all__ = [
    "__version__",
    "DetectionResult",
    "GeoInfo",
    "visualize",
    "detect",
    "detect_v2",
    "discover_v2_models",
    "process_dem",
    "read_raster",
    "read_geo_info",
    "generate_report",
    "report_to_text",
    "report_to_html",
    "save_report",
    "Feature",
    "extract_features",
    "filter_features",
    "feature_summary",
    "augment_sample",
    "cutmix",
    "average_probabilities",
    "majority_vote",
    "merge_results",
]


def read_geo_info(path: str | Path) -> GeoInfo:
    """Read georeferencing metadata from a raster file.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF or other rasterio-supported file.

    Returns
    -------
    GeoInfo
        CRS, affine transform, bounds, and resolution.
        Fields are None if rasterio is unavailable or the file
        lacks georeferencing.
    """
    path = Path(path)
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            crs = str(src.crs) if src.crs else None
            transform = tuple(src.transform)[:6] if src.transform else None
            bounds = tuple(src.bounds) if src.bounds else None
            res = src.res[0] if src.res else 0.5
            return GeoInfo(crs=crs, transform=transform, bounds=bounds, resolution=res)
    except (ImportError, Exception):
        return GeoInfo()


def read_raster(path: str | Path) -> tuple[np.ndarray, GeoInfo]:
    """Load a raster file and its georeferencing metadata.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF (.tif), or NumPy (.npy) file.

    Returns
    -------
    data : np.ndarray
        Raster data as float32. For single-band: (H, W). For multi-band: (C, H, W).
    geo : GeoInfo
        Georeferencing metadata (CRS, transform, bounds, resolution).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    geo = GeoInfo()

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(str(path)) as src:
                if src.crs:
                    geo.crs = str(src.crs)
                if src.transform:
                    geo.transform = tuple(src.transform)[:6]
                if src.bounds:
                    geo.bounds = tuple(src.bounds)
                if src.res:
                    geo.resolution = src.res[0]
                if src.count == 1:
                    data = src.read(1).astype(np.float32)
                else:
                    data = src.read().astype(np.float32)
        except ImportError:
            from PIL import Image

            data = np.array(Image.open(str(path)), dtype=np.float32)
    elif suffix == ".npy":
        data = np.load(str(path)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .tif or .npy")

    return data, geo


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


def detect_v2(
    visualization: np.ndarray,
    model_dir: str = "models",
    confidence_threshold: float = 0.5,
    use_tta: bool = True,
) -> DetectionResult:
    """Run v2 per-class binary model inference with TTA.

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)`` where *C* is typically 3.
    model_dir : str
        Directory containing per-class model files.
    confidence_threshold : float
        Pixels below this confidence are reset to background.
    use_tta : bool
        If True, use 8-fold test-time augmentation.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    return _run_detection_v2(
        visualization,
        model_dir=model_dir,
        confidence_threshold=confidence_threshold,
        use_tta=use_tta,
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
