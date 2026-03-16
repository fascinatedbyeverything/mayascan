"""MayaScan: Open-source archaeological LiDAR feature detection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import numpy as np

__version__ = "0.7.0"

if TYPE_CHECKING:
    from mayascan.benchmark import BenchmarkResult
    from mayascan.comparison import ComparisonResult
    from mayascan.crossval import FoldSplit
    from mayascan.detect import DetectionResult, GeoInfo
    from mayascan.features import Feature
    from mayascan.heatmap import (
        class_density_maps,
        density_to_rgba,
        feature_density_map,
        save_density_png,
    )
    from mayascan.morphology import FeatureProfile, ShapeDescriptors
    from mayascan.spatial import Cluster


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DetectionResult": ("mayascan.detect", "DetectionResult"),
    "GeoInfo": ("mayascan.detect", "GeoInfo"),
    "discover_v2_models": ("mayascan.detect", "discover_v2_models"),
    "discover_fold_models": ("mayascan.crossval", "discover_fold_models"),
    "generate_report": ("mayascan.report", "generate_report"),
    "report_to_text": ("mayascan.report", "report_to_text"),
    "report_to_html": ("mayascan.report", "report_to_html"),
    "save_report": ("mayascan.report", "save_report"),
    "Feature": ("mayascan.features", "Feature"),
    "extract_features": ("mayascan.features", "extract_features"),
    "filter_features": ("mayascan.features", "filter_features"),
    "feature_summary": ("mayascan.features", "feature_summary"),
    "augment_sample": ("mayascan.augment", "augment_sample"),
    "cutmix": ("mayascan.augment", "cutmix"),
    "average_probabilities": ("mayascan.ensemble", "average_probabilities"),
    "majority_vote": ("mayascan.ensemble", "majority_vote"),
    "merge_results": ("mayascan.ensemble", "merge_results"),
    "run_multiscale_detection": ("mayascan.multiscale", "run_multiscale_detection"),
    "BenchmarkResult": ("mayascan.benchmark", "BenchmarkResult"),
    "run_benchmark": ("mayascan.benchmark", "run_benchmark"),
    "format_benchmark": ("mayascan.benchmark", "format_benchmark"),
    "FoldSplit": ("mayascan.crossval", "FoldSplit"),
    "create_folds": ("mayascan.crossval", "create_folds"),
    "cv_fold_summary": ("mayascan.crossval", "fold_summary"),
    "train_fold": ("mayascan.crossval", "train_fold"),
    "train_kfold": ("mayascan.crossval", "train_kfold"),
    "train_kfold_all": ("mayascan.crossval", "train_kfold_all"),
    "Cluster": ("mayascan.spatial", "Cluster"),
    "cluster_features": ("mayascan.spatial", "cluster_features"),
    "identify_site_core": ("mayascan.spatial", "identify_site_core"),
    "settlement_hierarchy": ("mayascan.spatial", "settlement_hierarchy"),
    "feature_density_map": ("mayascan.heatmap", "feature_density_map"),
    "class_density_maps": ("mayascan.heatmap", "class_density_maps"),
    "density_to_rgba": ("mayascan.heatmap", "density_to_rgba"),
    "save_density_png": ("mayascan.heatmap", "save_density_png"),
    "ComparisonResult": ("mayascan.comparison", "ComparisonResult"),
    "compare_detections": ("mayascan.comparison", "compare_detections"),
    "comparison_summary": ("mayascan.comparison", "comparison_summary"),
    "difference_map": ("mayascan.comparison", "difference_map"),
    "count_feature_changes": ("mayascan.comparison", "count_feature_changes"),
    "ShapeDescriptors": ("mayascan.morphology", "ShapeDescriptors"),
    "FeatureProfile": ("mayascan.morphology", "FeatureProfile"),
    "compute_shape_descriptors": ("mayascan.morphology", "compute_shape_descriptors"),
    "analyze_features": ("mayascan.morphology", "analyze_features"),
    "classify_structure": ("mayascan.morphology", "classify_structure"),
    "settlement_summary": ("mayascan.morphology", "settlement_summary"),
}

__all__ = [
    "__version__",
    "DetectionResult",
    "GeoInfo",
    "visualize",
    "detect",
    "detect_v2",
    "detect_v2_ensemble",
    "discover_v2_models",
    "discover_fold_models",
    "FoldSplit",
    "create_folds",
    "cv_fold_summary",
    "train_fold",
    "train_kfold",
    "train_kfold_all",
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
    "run_multiscale_detection",
    "BenchmarkResult",
    "run_benchmark",
    "format_benchmark",
    "ShapeDescriptors",
    "FeatureProfile",
    "compute_shape_descriptors",
    "analyze_features",
    "classify_structure",
    "settlement_summary",
    "Cluster",
    "cluster_features",
    "identify_site_core",
    "settlement_hierarchy",
    "feature_density_map",
    "class_density_maps",
    "density_to_rgba",
    "save_density_png",
    "ComparisonResult",
    "compare_detections",
    "comparison_summary",
    "difference_map",
    "count_feature_changes",
]


def __getattr__(name: str):
    """Load public exports lazily to keep package import lightweight."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports to interactive tooling."""
    return sorted(set(globals()) | set(__all__))


def read_geo_info(path: str | Path) -> "GeoInfo":
    """Read georeferencing metadata from a raster file."""
    geo_cls = __getattr__("GeoInfo")
    path = Path(path)
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            crs = str(src.crs) if src.crs else None
            transform = tuple(src.transform)[:6] if src.transform else None
            bounds = tuple(src.bounds) if src.bounds else None
            res = src.res[0] if src.res else 0.5
            return geo_cls(crs=crs, transform=transform, bounds=bounds, resolution=res)
    except (ImportError, Exception):
        return geo_cls()


def read_raster(path: str | Path) -> tuple[np.ndarray, "GeoInfo"]:
    """Load a raster file and its georeferencing metadata."""
    geo_cls = __getattr__("GeoInfo")
    path = Path(path)
    suffix = path.suffix.lower()
    geo = geo_cls()

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
    """Compute SVF, openness, and slope visualizations from a DEM."""
    compute_visualizations = cast(
        Callable[[np.ndarray, float], np.ndarray],
        import_module("mayascan.visualize").compute_visualizations,
    )
    return compute_visualizations(dem, resolution)


def detect(
    visualization: np.ndarray,
    model_path: str | None = None,
    confidence_threshold: float = 0.5,
) -> "DetectionResult":
    """Run tiled U-Net inference on a visualization raster."""
    run_detection = import_module("mayascan.detect").run_detection
    return run_detection(
        visualization,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )


def detect_v2(
    visualization: np.ndarray,
    model_dir: str = "models",
    confidence_threshold: float = 0.5,
    use_tta: bool = True,
) -> "DetectionResult":
    """Run v2 per-class binary model inference with TTA."""
    run_detection_v2 = import_module("mayascan.detect").run_detection_v2
    return run_detection_v2(
        visualization,
        model_dir=model_dir,
        confidence_threshold=confidence_threshold,
        use_tta=use_tta,
    )


def detect_v2_ensemble(
    visualization: np.ndarray,
    model_dir: str = "models",
    confidence_threshold: float = 0.5,
    use_tta: bool = True,
) -> "DetectionResult":
    """Run K-fold ensemble inference for highest accuracy."""
    run_detection_v2_ensemble = import_module("mayascan.detect").run_detection_v2_ensemble
    return run_detection_v2_ensemble(
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
) -> "DetectionResult":
    """Run the full MayaScan pipeline: visualize a DEM then detect features."""
    compute_visualizations = import_module("mayascan.visualize").compute_visualizations
    run_detection = import_module("mayascan.detect").run_detection
    viz = compute_visualizations(dem, resolution=resolution)
    return run_detection(
        viz,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
