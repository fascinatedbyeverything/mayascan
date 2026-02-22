"""Coordinate Reference System utilities.

Handles CRS transformations for converting feature coordinates between
projected (e.g. UTM) and geographic (lat/lon) systems. Useful for Google
Earth export and cross-dataset comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mayascan.detect import GeoInfo


def transform_coordinates(
    x: float | np.ndarray,
    y: float | np.ndarray,
    source_crs: str,
    target_crs: str = "EPSG:4326",
) -> tuple[Any, Any]:
    """Transform coordinates between CRS systems.

    Parameters
    ----------
    x, y : float or array
        Input coordinates.
    source_crs : str
        Source CRS (e.g. "EPSG:32616" for UTM 16N).
    target_crs : str
        Target CRS (default "EPSG:4326" for WGS84 lat/lon).

    Returns
    -------
    tuple
        (x_transformed, y_transformed) in the target CRS.

    Raises
    ------
    ImportError
        If pyproj is not installed.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transformer.transform(x, y)


def pixel_to_latlon(
    row: float,
    col: float,
    geo: GeoInfo,
) -> tuple[float, float] | None:
    """Convert pixel coordinates to WGS84 latitude/longitude.

    Parameters
    ----------
    row, col : float
        Pixel coordinates (row, column).
    geo : GeoInfo
        Georeferencing with CRS and affine transform.

    Returns
    -------
    tuple or None
        (longitude, latitude) in WGS84, or None if transformation fails.
    """
    if not geo or not geo.transform or not geo.crs:
        return None

    # Pixel to map coordinates
    a, b, c, d, e, f = geo.transform
    map_x = c + col * a + row * b
    map_y = f + col * d + row * e

    # If already geographic (lat/lon), return directly
    if geo.crs.upper() in ("EPSG:4326", "WGS84", "WGS 84"):
        return (map_x, map_y)

    try:
        lon, lat = transform_coordinates(map_x, map_y, geo.crs, "EPSG:4326")
        return (float(lon), float(lat))
    except Exception:
        return None


def get_bounds_latlon(geo: GeoInfo) -> tuple[float, float, float, float] | None:
    """Get the bounding box of a GeoInfo in WGS84 lat/lon.

    Parameters
    ----------
    geo : GeoInfo
        Georeferencing metadata.

    Returns
    -------
    tuple or None
        (min_lon, min_lat, max_lon, max_lat) in WGS84,
        or None if transformation fails.
    """
    if not geo or not geo.bounds or not geo.crs:
        return None

    left, bottom, right, top = geo.bounds

    if geo.crs.upper() in ("EPSG:4326", "WGS84", "WGS 84"):
        return (left, bottom, right, top)

    try:
        corners_x = [left, right, left, right]
        corners_y = [bottom, bottom, top, top]
        lons, lats = transform_coordinates(
            np.array(corners_x), np.array(corners_y),
            geo.crs, "EPSG:4326"
        )
        return (
            float(np.min(lons)),
            float(np.min(lats)),
            float(np.max(lons)),
            float(np.max(lats)),
        )
    except Exception:
        return None
