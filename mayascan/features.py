"""Feature extraction and filtering utilities.

Provides functions to extract individual features from a DetectionResult,
filter by area/confidence/class, and compute per-feature statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label

from mayascan.detect import DetectionResult, GeoInfo


@dataclass
class Feature:
    """A single detected archaeological feature.

    Attributes
    ----------
    feature_id : int
        Unique ID within its class.
    class_id : int
        Class index (1=building, 2=platform, 3=aguada).
    class_name : str
        Human-readable class name.
    pixel_count : int
        Number of pixels in this feature.
    area_m2 : float
        Area in square metres.
    confidence : float
        Mean confidence score (0-1).
    centroid_row : float
        Centroid row in pixel coordinates.
    centroid_col : float
        Centroid column in pixel coordinates.
    centroid_geo : tuple[float, float] | None
        Centroid in map coordinates (x, y), if georeferenced.
    bbox : tuple[int, int, int, int]
        Bounding box (row_min, col_min, row_max, col_max).
    mask : np.ndarray
        Binary mask for this feature only (same shape as full image).
    """

    feature_id: int
    class_id: int
    class_name: str
    pixel_count: int
    area_m2: float
    confidence: float
    centroid_row: float
    centroid_col: float
    centroid_geo: tuple[float, float] | None
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def extract_features(
    result: DetectionResult,
    pixel_size: float = 0.5,
) -> list[Feature]:
    """Extract all individual features from a DetectionResult.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    list[Feature]
        List of Feature objects, sorted by area (largest first).
    """
    geo = result.geo
    if geo and geo.resolution:
        pixel_size = geo.resolution

    features: list[Feature] = []
    global_id = 0

    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue

        mask = result.classes == class_id
        if not mask.any():
            continue

        labeled_arr, num_features = label(mask)

        for feat_idx in range(1, num_features + 1):
            feat_mask = labeled_arr == feat_idx
            px_count = int(feat_mask.sum())
            area = px_count * pixel_size * pixel_size
            conf = float(result.confidence[feat_mask].mean())

            rows, cols = np.where(feat_mask)
            cy = float(rows.mean())
            cx = float(cols.mean())

            bbox = (int(rows.min()), int(cols.min()),
                    int(rows.max()), int(cols.max()))

            # Geo coordinates
            centroid_geo = None
            if geo and geo.transform:
                a, b, c, d, e, f = geo.transform
                gx = c + cx * a + cy * b
                gy = f + cx * d + cy * e
                centroid_geo = (gx, gy)

            global_id += 1
            features.append(Feature(
                feature_id=global_id,
                class_id=class_id,
                class_name=class_name,
                pixel_count=px_count,
                area_m2=area,
                confidence=conf,
                centroid_row=cy,
                centroid_col=cx,
                centroid_geo=centroid_geo,
                bbox=bbox,
                mask=feat_mask,
            ))

    # Sort by area descending
    features.sort(key=lambda f: f.area_m2, reverse=True)
    return features


def filter_features(
    features: list[Feature],
    min_area: float | None = None,
    max_area: float | None = None,
    min_confidence: float | None = None,
    classes: list[str] | None = None,
) -> list[Feature]:
    """Filter features by area, confidence, and/or class.

    Parameters
    ----------
    features : list[Feature]
        Input feature list.
    min_area : float or None
        Minimum area in m². Features smaller than this are excluded.
    max_area : float or None
        Maximum area in m². Features larger than this are excluded.
    min_confidence : float or None
        Minimum confidence score.
    classes : list[str] or None
        Class names to include (e.g. ["building", "aguada"]).
        If None, all classes are included.

    Returns
    -------
    list[Feature]
        Filtered features (same order as input).
    """
    result = features

    if min_area is not None:
        result = [f for f in result if f.area_m2 >= min_area]
    if max_area is not None:
        result = [f for f in result if f.area_m2 <= max_area]
    if min_confidence is not None:
        result = [f for f in result if f.confidence >= min_confidence]
    if classes is not None:
        classes_lower = [c.lower() for c in classes]
        result = [f for f in result if f.class_name.lower() in classes_lower]

    return result


def feature_summary(features: list[Feature]) -> dict[str, Any]:
    """Compute summary statistics for a list of features.

    Parameters
    ----------
    features : list[Feature]
        Feature list.

    Returns
    -------
    dict
        Summary with total count, total area, per-class counts, etc.
    """
    if not features:
        return {
            "total_count": 0,
            "total_area_m2": 0.0,
            "mean_confidence": 0.0,
            "by_class": {},
        }

    total_area = sum(f.area_m2 for f in features)
    mean_conf = sum(f.confidence for f in features) / len(features)

    by_class: dict[str, dict[str, Any]] = {}
    for f in features:
        cn = f.class_name
        if cn not in by_class:
            by_class[cn] = {
                "count": 0,
                "total_area_m2": 0.0,
                "confidences": [],
            }
        by_class[cn]["count"] += 1
        by_class[cn]["total_area_m2"] += f.area_m2
        by_class[cn]["confidences"].append(f.confidence)

    # Compute mean confidence per class
    for cn, data in by_class.items():
        confs = data.pop("confidences")
        data["mean_confidence"] = sum(confs) / len(confs)
        data["total_area_m2"] = round(data["total_area_m2"], 2)

    return {
        "total_count": len(features),
        "total_area_m2": round(total_area, 2),
        "mean_confidence": round(mean_conf, 4),
        "by_class": by_class,
    }
