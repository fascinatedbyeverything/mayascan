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


# Class colours for overlay rendering (RGBA)
_OVERLAY_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),           # background — transparent
    1: (255, 60, 60, 180),     # building — red
    2: (60, 200, 60, 180),     # platform — green
    3: (50, 120, 255, 180),    # aguada — blue
}


def to_overlay_png(
    result: DetectionResult,
    viz: np.ndarray,
    output_path: str | Path,
    opacity: float = 0.6,
) -> Path:
    """Export a blended overlay of detections on the visualization.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    viz : np.ndarray
        Visualization raster (3, H, W) float32 in [0, 1].
    output_path : str or Path
        Destination PNG file path.
    opacity : float
        Overlay opacity (0 = transparent, 1 = opaque).

    Returns
    -------
    Path
        The written PNG file path.
    """
    from PIL import Image

    output_path = Path(output_path)

    # Convert viz to RGB uint8
    base = (np.transpose(viz, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

    # Create RGBA overlay
    h, w = result.classes.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in _OVERLAY_COLORS.items():
        mask = result.classes == cls_id
        overlay[mask] = color

    # Blend
    base_f = base.astype(np.float32) / 255.0
    over_f = overlay[:, :, :3].astype(np.float32) / 255.0
    alpha = (overlay[:, :, 3].astype(np.float32) / 255.0 * opacity)[:, :, np.newaxis]
    blended = base_f * (1 - alpha) + over_f * alpha
    blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(blended_uint8).save(str(output_path))
    return output_path


def to_shapefile(
    result: DetectionResult,
    output_path: str | Path,
    pixel_size: float = 0.5,
) -> Path:
    """Export detected features as an ESRI Shapefile using geopandas.

    Requires the ``geo`` optional dependency (``pip install mayascan[geo]``).

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    output_path : str or Path
        Destination .shp file path.
    pixel_size : float
        Ground resolution in metres per pixel (default 0.5 m).

    Returns
    -------
    Path
        The written Shapefile path.

    Raises
    ------
    ImportError
        If geopandas or shapely are not installed.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, Point

    output_path = Path(output_path)
    geo = result.geo

    records = []
    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue

        mask = result.classes == class_id
        if not mask.any():
            continue

        labeled_arr, num_features = label(mask)

        for feat_idx in range(1, num_features + 1):
            feat_mask = labeled_arr == feat_idx
            pixel_count = int(feat_mask.sum())
            area_m2 = pixel_count * pixel_size * pixel_size
            confidence = float(result.confidence[feat_mask].mean())

            rows, cols = np.where(feat_mask)
            centroid_row = float(rows.mean())
            centroid_col = float(cols.mean())

            # Create polygon via convex hull
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([cols, rows]).astype(np.float64)
                if len(points) >= 3:
                    hull = ConvexHull(points)
                    hull_coords = [
                        _pixel_to_map(points[v, 1], points[v, 0], geo)
                        for v in hull.vertices
                    ]
                    hull_coords.append(hull_coords[0])
                    geometry = Polygon(hull_coords)
                else:
                    cx, cy = _pixel_to_map(centroid_row, centroid_col, geo)
                    geometry = Point(cx, cy).buffer(pixel_size)
            except Exception:
                row_min, row_max = int(rows.min()), int(rows.max())
                col_min, col_max = int(cols.min()), int(cols.max())
                corners = [
                    _pixel_to_map(row_min, col_min, geo),
                    _pixel_to_map(row_min, col_max + 1, geo),
                    _pixel_to_map(row_max + 1, col_max + 1, geo),
                    _pixel_to_map(row_max + 1, col_min, geo),
                ]
                geometry = Polygon(corners)

            records.append({
                "class": class_name,
                "class_id": class_id,
                "area_m2": round(area_m2, 2),
                "confidence": round(confidence, 4),
                "pixels": pixel_count,
                "geometry": geometry,
            })

    crs = geo.crs if geo and geo.crs else None
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(str(output_path))
    return output_path
