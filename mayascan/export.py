"""Export module: convert DetectionResult to CSV, GeoJSON, and GeoTIFF formats.

Provides three export functions for archaeological feature maps:
- ``to_csv``: feature centroids with area and confidence
- ``to_geojson``: bounding-box polygon features
- ``to_geotiff``: class-map raster (rasterio or PIL fallback)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from mayascan.detect import DetectionResult


def _extract_components(
    result: DetectionResult,
    pixel_size: float,
) -> list[dict[str, Any]]:
    """Extract connected components for all non-background classes.

    Returns a list of dicts, each describing one connected component with
    keys: class_name, class_id, centroid_x, centroid_y, area_m2,
    confidence, pixel_count, bbox (row_min, row_max, col_min, col_max).
    """
    components: list[dict[str, Any]] = []

    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue  # skip background

        mask = result.classes == class_id
        if not mask.any():
            continue

        labeled, num_features = label(mask)

        for feat_idx in range(1, num_features + 1):
            feat_mask = labeled == feat_idx
            pixel_count = int(feat_mask.sum())
            area_m2 = pixel_count * pixel_size * pixel_size

            # centroid in pixel coordinates
            rows, cols = np.where(feat_mask)
            centroid_y = float(rows.mean())
            centroid_x = float(cols.mean())

            # mean confidence over the component
            confidence = float(result.confidence[feat_mask].mean())

            # bounding box (pixel coords)
            row_min, row_max = int(rows.min()), int(rows.max())
            col_min, col_max = int(cols.min()), int(cols.max())

            components.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "area_m2": area_m2,
                    "confidence": confidence,
                    "pixel_count": pixel_count,
                    "bbox": (row_min, row_max, col_min, col_max),
                }
            )

    return components


def to_csv(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export feature centroids to CSV.

    Parameters
    ----------
    result : DetectionResult
        Detection output with ``classes``, ``confidence``, and ``class_names``.
    output_path : str or Path
        Destination CSV file path.
    pixel_size : float
        Ground resolution in metres per pixel (default 0.5 m).

    Returns
    -------
    Path
        The written CSV file path.
    """
    output_path = Path(output_path)
    components = _extract_components(result, pixel_size)

    fieldnames = [
        "class",
        "class_id",
        "centroid_x",
        "centroid_y",
        "area_m2",
        "confidence",
        "pixel_count",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for comp in components:
            writer.writerow(comp)

    return output_path


def to_geojson(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export feature bounding boxes as a GeoJSON FeatureCollection.

    Each connected component becomes a polygon feature with its bounding
    box as geometry and class / area / confidence as properties.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    output_path : str or Path
        Destination GeoJSON file path.
    pixel_size : float
        Ground resolution in metres per pixel (default 0.5 m).

    Returns
    -------
    Path
        The written GeoJSON file path.
    """
    output_path = Path(output_path)
    components = _extract_components(result, pixel_size)

    features: list[dict[str, Any]] = []
    for comp in components:
        row_min, row_max, col_min, col_max = comp["bbox"]

        # Convert pixel bounding box to coordinate space (pixel_size metres)
        x_min = col_min * pixel_size
        x_max = (col_max + 1) * pixel_size
        y_min = row_min * pixel_size
        y_max = (row_max + 1) * pixel_size

        # GeoJSON polygon ring (closed, counter-clockwise)
        coordinates = [
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        ]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
            "properties": {
                "class": comp["class"],
                "class_id": comp["class_id"],
                "area_m2": comp["area_m2"],
                "confidence": comp["confidence"],
            },
        }
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w") as f:
        json.dump(collection, f, indent=2)

    return output_path


def to_geotiff(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export the class map as a GeoTIFF (or plain TIFF fallback).

    Attempts to use ``rasterio`` for a proper GeoTIFF with spatial
    metadata.  If rasterio is not installed (it requires GDAL), falls
    back to ``PIL.Image`` to write a uint8 TIFF of the class indices.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    output_path : str or Path
        Destination TIFF file path.
    pixel_size : float
        Ground resolution in metres per pixel (default 0.5 m).

    Returns
    -------
    Path
        The written TIFF file path.
    """
    output_path = Path(output_path)
    class_map = result.classes.astype(np.uint8)

    try:
        import rasterio
        from rasterio.transform import from_bounds

        h, w = class_map.shape
        transform = from_bounds(0, 0, w * pixel_size, h * pixel_size, w, h)

        with rasterio.open(
            str(output_path),
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            crs="EPSG:32616",
            transform=transform,
        ) as dst:
            dst.write(class_map, 1)

    except ImportError:
        from PIL import Image

        img = Image.fromarray(class_map, mode="L")
        img.save(str(output_path), format="TIFF")

    return output_path
