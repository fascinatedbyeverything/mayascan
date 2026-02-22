"""Tests for mayascan.export and mayascan.report modules."""

import csv
import json

import numpy as np
import pytest

from mayascan.detect import CLASS_NAMES, DetectionResult, GeoInfo
from mayascan.export import to_csv, to_geojson, to_geotiff, to_confidence_geotiff
from mayascan.report import generate_report, report_to_text, report_to_html, save_report


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


class TestReport:
    """Tests for mayascan.report module."""

    def test_generate_report_structure(self, detection_result):
        """generate_report returns a dict with expected top-level keys."""
        report = generate_report(detection_result)

        assert "timestamp" in report
        assert report["software"] == "MayaScan"
        assert report["dimensions"] == {"height": 50, "width": 50}
        assert report["total_features"] >= 2  # building + platform
        assert report["total_feature_area_m2"] > 0
        assert "classes" in report
        assert "feature_density_per_km2" in report

    def test_report_classes(self, detection_result):
        """Report includes per-class stats with feature lists."""
        report = generate_report(detection_result)

        assert "building" in report["classes"]
        assert "platform" in report["classes"]

        building = report["classes"]["building"]
        assert building["count"] >= 1
        assert building["total_area_m2"] > 0
        assert 0 < building["mean_confidence"] <= 1
        assert len(building["features"]) >= 1

        # Features sorted by area descending
        areas = [f["area_m2"] for f in building["features"]]
        assert areas == sorted(areas, reverse=True)

    def test_report_feature_details(self, detection_result):
        """Each feature has required fields."""
        report = generate_report(detection_result, pixel_size=0.5)

        for cls_data in report["classes"].values():
            for feat in cls_data["features"]:
                assert "id" in feat
                assert "pixel_count" in feat
                assert "area_m2" in feat
                assert "confidence" in feat
                assert "centroid_px" in feat
                assert "bbox_px" in feat

    def test_report_with_geo(self):
        """Report features include geo coordinates when GeoInfo is present."""
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

        report = generate_report(result)
        assert report["crs"] == "EPSG:32616"
        assert report["resolution_m"] == 0.5

        building = report["classes"]["building"]
        assert building["count"] >= 1
        feat = building["features"][0]
        assert "centroid_geo" in feat
        # geo_x should be near 500000
        assert feat["centroid_geo"][0] > 500000

    def test_report_to_text(self, detection_result):
        """report_to_text produces readable text with key sections."""
        report = generate_report(detection_result)
        text = report_to_text(report)

        assert "MAYASCAN DETECTION REPORT" in text
        assert "SUMMARY" in text
        assert "BUILDING" in text
        assert "PLATFORM" in text
        assert "Total features:" in text

    def test_report_empty_class(self):
        """Report handles classes with zero features."""
        classes = np.zeros((20, 20), dtype=np.int64)
        confidence = np.full((20, 20), 0.1, dtype=np.float32)
        result = DetectionResult(
            classes=classes, confidence=confidence,
            class_names=dict(CLASS_NAMES),
        )
        report = generate_report(result)
        assert report["total_features"] == 0
        for cls_data in report["classes"].values():
            assert cls_data["count"] == 0

    def test_save_report_text(self, detection_result, tmp_path):
        """save_report writes a text file."""
        out = tmp_path / "report.txt"
        result_path = save_report(detection_result, out, format="text")
        assert result_path.exists()
        content = result_path.read_text()
        assert "MAYASCAN" in content

    def test_save_report_json(self, detection_result, tmp_path):
        """save_report writes valid JSON."""
        out = tmp_path / "report.json"
        result_path = save_report(detection_result, out, format="json")
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["software"] == "MayaScan"
        assert "classes" in data

    def test_report_to_html(self, detection_result):
        """report_to_html produces valid HTML with key elements."""
        report = generate_report(detection_result)
        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "MayaScan" in html
        assert "Total Features" in html
        assert "building" in html.lower() or "Building" in html

    def test_save_report_html(self, detection_result, tmp_path):
        """save_report writes an HTML file."""
        out = tmp_path / "report.html"
        result_path = save_report(detection_result, out, format="html")
        assert result_path.exists()
        content = result_path.read_text()
        assert "<html>" in content
