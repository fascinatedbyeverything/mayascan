"""Ground-point classification and DEM generation.

Provides PDAL-based ground filtering via SMRF (requires ``python-pdal``,
available through the ``mayascan[pdal]`` extra), and an always-available
:func:`dem_from_array` that builds a raster DEM from raw point arrays
using scipy interpolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class PdalNotAvailableError(ImportError):
    """Raised when a PDAL operation is requested but python-pdal is missing."""


# ---------------------------------------------------------------------------
# PDAL pipeline (optional dependency)
# ---------------------------------------------------------------------------

def classify_ground(
    input_path: str | Path,
    output_dem_path: str | Path | None = None,
    resolution: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify ground points with SMRF and return a DEM.

    Builds a PDAL pipeline:

    1. ``readers.las`` -- read the point cloud
    2. ``filters.outlier`` -- statistical outlier removal
    3. ``filters.smrf`` -- Simple Morphological Filter for ground
    4. ``filters.range`` -- keep only Classification == 2 (ground)
    5. (optionally) ``writers.gdal`` -- write a GeoTIFF DEM

    Parameters
    ----------
    input_path : str | Path
        Path to a LAS/LAZ point cloud file.
    output_dem_path : str | Path | None
        If provided, write the DEM to this GeoTIFF path.
    resolution : float
        Cell size in the same units as the point cloud (default 0.5).

    Returns
    -------
    dem : np.ndarray
        2-D float32 elevation grid.
    extent : dict
        ``{"xmin", "xmax", "ymin", "ymax"}`` in point-cloud CRS units.

    Raises
    ------
    PdalNotAvailableError
        If ``python-pdal`` is not installed.
    """
    try:
        import pdal  # noqa: F811
    except ImportError as exc:
        raise PdalNotAvailableError(
            "python-pdal is required for classify_ground(). "
            "Install it with:  pip install mayascan[pdal]  "
            "or:  conda install -c conda-forge python-pdal"
        ) from exc

    input_path = str(Path(input_path).resolve())

    # Build pipeline stages ------------------------------------------------
    stages: list[dict] = [
        {"type": "readers.las", "filename": input_path},
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 12,
            "multiplier": 2.2,
        },
        {
            "type": "filters.smrf",
            "cell": 1.0,
            "slope": 0.15,
            "window": 30,
            "threshold": 0.5,
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]",
        },
    ]

    if output_dem_path is not None:
        stages.append(
            {
                "type": "writers.gdal",
                "filename": str(Path(output_dem_path).resolve()),
                "output_type": "idw",
                "resolution": resolution,
            }
        )

    import json as _json

    pipeline = pdal.Pipeline(_json.dumps(stages))
    pipeline.execute()

    arrays = pipeline.arrays
    pts = arrays[0]
    x, y, z = pts["X"], pts["Y"], pts["Z"]

    return dem_from_array(x, y, z, resolution=resolution)


# ---------------------------------------------------------------------------
# Pure-scipy DEM builder (always available)
# ---------------------------------------------------------------------------

def dem_from_array(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a raster DEM from scattered x/y/z points.

    Uses ``scipy.interpolate.griddata`` with linear interpolation,
    then fills remaining NaN cells with nearest-neighbor values.

    Parameters
    ----------
    x, y, z : np.ndarray
        1-D coordinate arrays of equal length.
    resolution : float
        Grid cell size in the same units as *x* / *y* (default 0.5).

    Returns
    -------
    dem : np.ndarray
        2-D float32 elevation grid with shape ``(nrows, ncols)``.
    extent : dict
        ``{"xmin", "xmax", "ymin", "ymax"}`` matching the grid bounds.
    """
    from scipy.interpolate import griddata

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    ncols = max(1, int(np.ceil((xmax - xmin) / resolution)))
    nrows = max(1, int(np.ceil((ymax - ymin) / resolution)))

    grid_x = np.linspace(xmin, xmax, ncols)
    grid_y = np.linspace(ymin, ymax, nrows)
    gx, gy = np.meshgrid(grid_x, grid_y)

    points = np.column_stack((x, y))

    # Primary: linear interpolation
    dem = griddata(points, z, (gx, gy), method="linear").astype(np.float32)

    # Fill remaining NaN with nearest-neighbor
    nan_mask = np.isnan(dem)
    if nan_mask.any():
        nearest = griddata(points, z, (gx, gy), method="nearest").astype(np.float32)
        dem[nan_mask] = nearest[nan_mask]

    extent = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    return dem, extent
