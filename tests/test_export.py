"""Tests for mayascan.export — CSV, GeoJSON, and GeoTIFF export."""

import csv
import json

import numpy as np
import pytest

from mayascan.detect import CLASS_NAMES, DetectionResult, GeoInfo
from mayascan.export import to_csv, to_geojson, to_geotiff, to_confidence_geotiff


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


class TestGeoExport:
    """Tests for georeferenced export."""

    def test_csv_with_geo(self, tmp_path):
        """CSV includes geo_x, geo_y columns when GeoInfo is present."""
        classes = np.zeros((50, 50), dtype=np.int64)
        confidence = np.full((50, 50), 0.1, dtype=np.float32)
        classes[10:20, 10:20] = 1
        confidence[10:20, 10:20] = 0.9

        geo = GeoInfo(
            crs="EPSG:32616",
            transform=(0.5, 0.0, 500000.0, 0.0, -0.5, 2000000.0),
            resolution=0.5,
        )
        result = DetectionResult(
            classes=classes, confidence=confidence,
            class_names=dict(CLASS_NAMES), geo=geo,
        )

        csv_path = to_csv(result, tmp_path / "geo.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) >= 1
        assert "geo_x" in reader.fieldnames
        assert "geo_y" in reader.fieldnames
        # geo_x should be near 500000 + 14.5 * 0.5 = 500007.25
        gx = float(rows[0]["geo_x"])
        assert 500000 < gx < 500050

    def test_geojson_with_geo(self, tmp_path):
        """GeoJSON includes CRS and real-world coordinates."""
        classes = np.zeros((50, 50), dtype=np.int64)
        confidence = np.full((50, 50), 0.1, dtype=np.float32)
        classes[10:20, 10:20] = 1
        confidence[10:20, 10:20] = 0.9

        geo = GeoInfo(
            crs="EPSG:32616",
            transform=(0.5, 0.0, 500000.0, 0.0, -0.5, 2000000.0),
            resolution=0.5,
        )
        result = DetectionResult(
            classes=classes, confidence=confidence,
            class_names=dict(CLASS_NAMES), geo=geo,
        )

        geojson_path = to_geojson(result, tmp_path / "geo.geojson")
        with open(geojson_path) as f:
            data = json.load(f)

        assert "crs" in data
        assert data["crs"]["properties"]["name"] == "EPSG:32616"
        assert len(data["features"]) >= 1

        # Coordinates should be in map space (near 500000)
        coords = data["features"][0]["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        assert all(x > 500000 for x in xs)

    def test_confidence_geotiff(self, detection_result, tmp_path):
        """Confidence GeoTIFF is written."""
        conf_path = to_confidence_geotiff(
            detection_result, tmp_path / "conf.tif"
        )
        assert conf_path.exists()
        assert conf_path.stat().st_size > 0

    def test_geojson_contours(self, tmp_path):
        """GeoJSON uses convex hull contours for features with enough pixels."""
        classes = np.zeros((100, 100), dtype=np.int64)
        confidence = np.full((100, 100), 0.1, dtype=np.float32)
        # Create a circular-ish blob
        for r in range(30, 70):
            for c in range(30, 70):
                if (r - 50) ** 2 + (c - 50) ** 2 < 400:
                    classes[r, c] = 1
                    confidence[r, c] = 0.9

        result = DetectionResult(
            classes=classes, confidence=confidence,
            class_names=dict(CLASS_NAMES),
        )

        geojson_path = to_geojson(result, tmp_path / "contour.geojson")
        with open(geojson_path) as f:
            data = json.load(f)

        assert len(data["features"]) == 1
        coords = data["features"][0]["geometry"]["coordinates"][0]
        # Convex hull should have more than 4 points (not just a bounding box)
        assert len(coords) > 4
