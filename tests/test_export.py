"""Tests for mayascan.export — CSV, GeoJSON, and GeoTIFF export."""

import csv
import json

import numpy as np
import pytest

from mayascan.detect import CLASS_NAMES, DetectionResult
from mayascan.export import to_csv, to_geojson, to_geotiff


@pytest.fixture
def detection_result() -> DetectionResult:
    """Create a DetectionResult with a 20x20 building block and a 10x10 platform block.

    Layout on a 50x50 grid:
    - Background everywhere by default (class 0)
    - Building block (class 1): rows 5-24, cols 5-24 (20x20 = 400 pixels)
    - Platform block (class 2): rows 30-39, cols 30-39 (10x10 = 100 pixels)
    """
    classes = np.zeros((50, 50), dtype=np.int64)
    confidence = np.full((50, 50), 0.1, dtype=np.float32)  # low bg confidence

    # 20x20 building block
    classes[5:25, 5:25] = 1
    confidence[5:25, 5:25] = 0.95

    # 10x10 platform block
    classes[30:40, 30:40] = 2
    confidence[30:40, 30:40] = 0.85

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )


class TestToCsv:
    """Tests for to_csv export."""

    def test_to_csv(self, detection_result: DetectionResult, tmp_path):
        """CSV has feature centroids with correct header and contains 'building'."""
        csv_path = tmp_path / "features.csv"
        result_path = to_csv(detection_result, csv_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

        with open(result_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have header with "class" column
        assert "class" in reader.fieldnames

        # Should contain "building" somewhere
        class_values = [row["class"] for row in rows]
        assert "building" in class_values

        # Should have at least 2 features (building + platform)
        assert len(rows) >= 2

        # Verify centroid_x and centroid_y are present and numeric
        for row in rows:
            assert "centroid_x" in row
            assert "centroid_y" in row
            float(row["centroid_x"])  # should not raise
            float(row["centroid_y"])  # should not raise


class TestToGeojson:
    """Tests for to_geojson export."""

    def test_to_geojson(self, detection_result: DetectionResult, tmp_path):
        """Creates valid GeoJSON FeatureCollection with at least 1 feature."""
        geojson_path = tmp_path / "features.geojson"
        result_path = to_geojson(detection_result, geojson_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

        with open(result_path) as f:
            data = json.load(f)

        # Must be a FeatureCollection
        assert data["type"] == "FeatureCollection"
        assert "features" in data

        # At least 1 feature
        assert len(data["features"]) >= 1

        # Each feature has proper structure
        for feature in data["features"]:
            assert feature["type"] == "Feature"
            assert feature["geometry"]["type"] == "Polygon"
            assert "coordinates" in feature["geometry"]
            assert "class" in feature["properties"]
            assert "area_m2" in feature["properties"]
            assert "confidence" in feature["properties"]


class TestToGeotiff:
    """Tests for to_geotiff export."""

    def test_to_geotiff(self, detection_result: DetectionResult, tmp_path):
        """Creates a TIFF file with non-zero size."""
        tiff_path = tmp_path / "classmap.tif"
        result_path = to_geotiff(detection_result, tiff_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0
