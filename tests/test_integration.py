"""End-to-end integration tests for the full MayaScan pipeline.

Tests the complete workflow: synthetic DEM -> visualize -> detect -> export.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import mayascan
from mayascan.detect import DetectionResult
from mayascan.export import to_csv, to_geojson, to_geotiff


def _make_mound_dem(size: int = 480, num_mounds: int = 5, seed: int = 42) -> np.ndarray:
    """Create a synthetic DEM with circular mound features.

    Uses np.ogrid to generate Gaussian-like mounds on a flat terrain,
    simulating archaeological mound structures.
    """
    rng = np.random.default_rng(seed)
    dem = np.zeros((size, size), dtype=np.float64)

    # Add gentle background terrain
    y, x = np.ogrid[0:size, 0:size]
    dem += 50.0 + 2.0 * np.sin(2 * np.pi * x / size) + 1.5 * np.cos(2 * np.pi * y / size)

    # Add circular mound features
    for _ in range(num_mounds):
        cx = rng.integers(60, size - 60)
        cy = rng.integers(60, size - 60)
        radius = rng.integers(15, 40)
        height = rng.uniform(3.0, 10.0)

        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mound = height * np.exp(-dist_sq / (2 * radius ** 2))
        dem += mound

    return dem


class TestFullPipelineSynthetic:
    """End-to-end test: synthetic DEM -> visualize -> detect -> export."""

    def test_full_pipeline_synthetic(self, tmp_path: Path):
        """Run the complete pipeline on a synthetic DEM with mound features."""
        # --- Step 1: Create synthetic DEM ---
        dem = _make_mound_dem(size=480, num_mounds=5, seed=42)
        assert dem.shape == (480, 480)

        # --- Step 2: Visualize ---
        viz = mayascan.visualize(dem, resolution=0.5)
        assert viz.shape == (3, 480, 480)
        assert viz.dtype == np.float32
        # Each channel should be in [0, 1]
        for ch in range(3):
            assert viz[ch].min() >= 0.0
            assert viz[ch].max() <= 1.0 + 1e-6

        # --- Step 3: Detect ---
        result = mayascan.detect(viz, confidence_threshold=0.0)
        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert result.classes.dtype in (np.int64, np.int32, np.intp)
        assert result.confidence.dtype == np.float32 or np.issubdtype(
            result.confidence.dtype, np.floating
        )
        assert result.classes.min() >= 0
        assert result.classes.max() < len(result.class_names)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0 + 1e-6

        # --- Step 4: Export ---
        csv_path = tmp_path / "features.csv"
        geojson_path = tmp_path / "features.geojson"
        geotiff_path = tmp_path / "classmap.tif"

        to_csv(result, csv_path)
        to_geojson(result, geojson_path)
        to_geotiff(result, geotiff_path)

        # Verify all files exist and are non-empty
        assert csv_path.exists(), "CSV file was not created"
        assert csv_path.stat().st_size > 0, "CSV file is empty"

        assert geojson_path.exists(), "GeoJSON file was not created"
        assert geojson_path.stat().st_size > 0, "GeoJSON file is empty"

        assert geotiff_path.exists(), "GeoTIFF file was not created"
        assert geotiff_path.stat().st_size > 0, "GeoTIFF file is empty"

        # Verify GeoJSON is valid JSON with correct structure
        with open(geojson_path) as f:
            geojson_data = json.load(f)

        assert geojson_data["type"] == "FeatureCollection"
        assert "features" in geojson_data
        assert isinstance(geojson_data["features"], list)


class TestProcessDemConvenience:
    """Test the convenience function mayascan.process_dem."""

    def test_process_dem_convenience(self):
        """process_dem runs the full pipeline in one call and returns correct shapes."""
        rng = np.random.default_rng(99)
        dem = rng.random((480, 480)).astype(np.float64) * 100.0

        result = mayascan.process_dem(dem, confidence_threshold=0.0)

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert isinstance(result.class_names, dict)
        assert 0 in result.class_names
        assert result.class_names[0] == "background"
        assert result.classes.min() >= 0
        assert result.classes.max() < len(result.class_names)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0 + 1e-6
