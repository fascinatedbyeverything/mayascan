"""Export module: convert DetectionResult to CSV, GeoJSON, and GeoTIFF formats.

Provides three export functions for archaeological feature maps:
- ``to_csv``: feature centroids with area and confidence
- ``to_geojson``: polygon features with optional real-world coordinates
- ``to_geotiff``: class-map raster with full georeferencing (rasterio or PIL fallback)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from mayascan.detect import DetectionResult, GeoInfo


def _pixel_to_map(row: float, col: float, geo: GeoInfo | None) -> tuple[float, float]:
    """Convert pixel coordinates to map coordinates using affine transform.

    Returns (x, y) in map units. If no georeferencing, returns pixel coords
    scaled by resolution.
    """
    if geo and geo.transform:
        a, b, c, d, e, f = geo.transform
        x = c + col * a + row * b
        y = f + col * d + row * e
        return x, y
    res = geo.resolution if geo else 0.5
    return col * res, row * res


def _extract_components(
    result: DetectionResult,
    pixel_size: float,
) -> list[dict[str, Any]]:
    """Extract connected components for all non-background classes.

    Returns a list of dicts, each describing one connected component with
    keys: class_name, class_id, centroid_x, centroid_y, area_m2,
    confidence, pixel_count, bbox (row_min, row_max, col_min, col_max),
    and optionally geo_centroid_x, geo_centroid_y if georeferencing is available.
    """
    geo = result.geo
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

            comp: dict[str, Any] = {
                "class": class_name,
                "class_id": class_id,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "area_m2": area_m2,
                "confidence": confidence,
                "pixel_count": pixel_count,
                "bbox": (row_min, row_max, col_min, col_max),
            }

            # Add georeferenced centroid if available
            if geo and geo.transform:
                gx, gy = _pixel_to_map(centroid_y, centroid_x, geo)
                comp["geo_x"] = gx
                comp["geo_y"] = gy

            components.append(comp)

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

    # Add geo columns if georeferencing is available
    has_geo = any("geo_x" in c for c in components)
    if has_geo:
        fieldnames.extend(["geo_x", "geo_y"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for comp in components:
            writer.writerow(comp)

    return output_path


def _extract_contour(mask: np.ndarray) -> list[list[float]] | None:
    """Extract the outer contour of a binary mask as a polygon ring.

    Uses a simple marching-squares-like boundary tracing via numpy.
    Returns a list of [col, row] coordinate pairs forming a closed ring,
    or None if the mask is too small.
    """
    rows, cols = np.where(mask)
    if len(rows) < 3:
        return None

    # Use convex hull approach for clean polygons
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack([cols, rows]).astype(np.float64)
        if len(points) < 3:
            return None
        hull = ConvexHull(points)
        ring = points[hull.vertices].tolist()
        ring.append(ring[0])  # close the ring
        return ring
    except Exception:
        # Fallback to bounding box
        return None


def to_geojson(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export detected features as a GeoJSON FeatureCollection.

    Uses actual feature contours (convex hulls) instead of bounding boxes.
    When georeferencing is available, coordinates are in the source CRS.

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
    geo = result.geo

    features: list[dict[str, Any]] = []

    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue

        mask = result.classes == class_id
        if not mask.any():
            continue

        labeled, num_features = label(mask)

        for feat_idx in range(1, num_features + 1):
            feat_mask = labeled == feat_idx
            pixel_count = int(feat_mask.sum())
            area_m2 = pixel_count * pixel_size * pixel_size
            confidence = float(result.confidence[feat_mask].mean())

            # Try contour extraction, fall back to bounding box
            contour = _extract_contour(feat_mask)

            if contour is not None:
                # Convert pixel coords to map coords
                coordinates = []
                for col, row in contour:
                    x, y = _pixel_to_map(row, col, geo)
                    coordinates.append([x, y])
            else:
                # Bounding box fallback
                rows, cols = np.where(feat_mask)
                row_min, row_max = int(rows.min()), int(rows.max())
                col_min, col_max = int(cols.min()), int(cols.max())

                corners = [
                    (row_min, col_min),
                    (row_min, col_max + 1),
                    (row_max + 1, col_max + 1),
                    (row_max + 1, col_min),
                    (row_min, col_min),
                ]
                coordinates = [
                    list(_pixel_to_map(r, c, geo)) for r, c in corners
                ]

            # Centroid
            rows_all, cols_all = np.where(feat_mask)
            cx, cy = _pixel_to_map(float(rows_all.mean()), float(cols_all.mean()), geo)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
                "properties": {
                    "class": class_name,
                    "class_id": class_id,
                    "area_m2": round(area_m2, 2),
                    "confidence": round(confidence, 4),
                    "pixel_count": pixel_count,
                    "centroid_x": round(cx, 6),
                    "centroid_y": round(cy, 6),
                },
            }
            features.append(feature)

    crs_name = geo.crs if geo and geo.crs else None
    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    if crs_name:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": crs_name},
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

    Uses georeferencing from the source raster when available, producing
    a properly georeferenced output that can be loaded in QGIS or ArcGIS.
    Falls back to ``PIL.Image`` if rasterio is not installed.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    output_path : str or Path
        Destination TIFF file path.
    pixel_size : float
        Ground resolution in metres per pixel (default 0.5 m).
        Only used when no georeferencing is available in the result.

    Returns
    -------
    Path
        The written TIFF file path.
    """
    output_path = Path(output_path)
    class_map = result.classes.astype(np.uint8)
    geo = result.geo

    try:
        import rasterio
        from rasterio.transform import Affine, from_bounds

        h, w = class_map.shape

        # Use source georeferencing if available
        if geo and geo.transform:
            transform = Affine(*geo.transform)
            crs = geo.crs or "EPSG:32616"
        else:
            transform = from_bounds(0, 0, w * pixel_size, h * pixel_size, w, h)
            crs = "EPSG:32616"

        with rasterio.open(
            str(output_path),
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(class_map, 1)
            # Write class names as band description
            dst.update_tags(
                classes=json.dumps(result.class_names),
                software="MayaScan",
            )

    except ImportError:
        from PIL import Image

        img = Image.fromarray(class_map, mode="L")
        img.save(str(output_path), format="TIFF")

    return output_path


def to_confidence_geotiff(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export the confidence map as a float32 GeoTIFF.

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
    geo = result.geo

    try:
        import rasterio
        from rasterio.transform import Affine, from_bounds

        h, w = result.confidence.shape

        if geo and geo.transform:
            transform = Affine(*geo.transform)
            crs = geo.crs or "EPSG:32616"
        else:
            transform = from_bounds(0, 0, w * pixel_size, h * pixel_size, w, h)
            crs = "EPSG:32616"

        with rasterio.open(
            str(output_path),
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(result.confidence.astype(np.float32), 1)

    except ImportError:
        # Save confidence as uint8 TIFF (scaled 0-255)
        from PIL import Image

        conf_uint8 = (result.confidence * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(conf_uint8, mode="L")
        img.save(str(output_path), format="TIFF")

    return output_path
