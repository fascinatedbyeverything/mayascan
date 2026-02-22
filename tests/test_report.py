"""Tests for mayascan.report — report generation."""

import json

import numpy as np
import pytest

from mayascan.config import CLASS_NAMES
from mayascan.detect import DetectionResult, GeoInfo
from mayascan.report import generate_report, report_to_html, report_to_text, save_report


@pytest.fixture
def detection_result():
    """Create a simple detection result with known features."""
    h, w = 100, 100
    classes = np.zeros((h, w), dtype=np.int64)
    confidence = np.zeros((h, w), dtype=np.float32)

    # One building blob
    classes[10:20, 10:20] = 1
    confidence[10:20, 10:20] = 0.9

    # One platform blob
    classes[50:65, 50:65] = 2
    confidence[50:65, 50:65] = 0.75

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )


@pytest.fixture
def geo_result(detection_result):
    """Detection result with georeferencing."""
    detection_result.geo = GeoInfo(
        crs="EPSG:32616",
        transform=(0.5, 0, 500000.0, 0, -0.5, 2000000.0),
        bounds=(500000.0, 1999950.0, 500050.0, 2000000.0),
        resolution=0.5,
    )
    return detection_result


class TestGenerateReport:
    def test_basic_structure(self, detection_result):
        report = generate_report(detection_result)
        assert "timestamp" in report
        assert "software" in report
        assert "dimensions" in report
        assert "classes" in report
        assert "total_features" in report

    def test_feature_counts(self, detection_result):
        report = generate_report(detection_result)
        assert report["total_features"] == 2
        assert report["classes"]["building"]["count"] == 1
        assert report["classes"]["platform"]["count"] == 1
        assert report["classes"]["aguada"]["count"] == 0

    def test_areas(self, detection_result):
        report = generate_report(detection_result, pixel_size=0.5)
        # Building: 10x10 = 100 pixels * 0.25 m² = 25 m²
        assert report["classes"]["building"]["total_area_m2"] == 25.0
        # Platform: 15x15 = 225 pixels * 0.25 m² = 56.25 m²
        assert report["classes"]["platform"]["total_area_m2"] == 56.25

    def test_confidence(self, detection_result):
        report = generate_report(detection_result)
        assert report["classes"]["building"]["mean_confidence"] == pytest.approx(0.9, abs=0.01)
        assert report["classes"]["platform"]["mean_confidence"] == pytest.approx(0.75, abs=0.01)

    def test_feature_details(self, detection_result):
        report = generate_report(detection_result)
        building_feats = report["classes"]["building"]["features"]
        assert len(building_feats) == 1
        assert "area_m2" in building_feats[0]
        assert "confidence" in building_feats[0]
        assert "centroid_px" in building_feats[0]
        assert "bbox_px" in building_feats[0]

    def test_geo_coordinates(self, geo_result):
        report = generate_report(geo_result)
        assert report["crs"] == "EPSG:32616"
        building_feats = report["classes"]["building"]["features"]
        assert "centroid_geo" in building_feats[0]

    def test_input_path(self, detection_result):
        report = generate_report(detection_result, input_path="test.tif")
        assert report["input"] == "test.tif"

    def test_density(self, detection_result):
        report = generate_report(detection_result, pixel_size=0.5)
        assert report["feature_density_per_km2"] > 0


class TestReportToText:
    def test_contains_header(self, detection_result):
        report = generate_report(detection_result)
        text = report_to_text(report)
        assert "MAYASCAN DETECTION REPORT" in text

    def test_contains_summary(self, detection_result):
        report = generate_report(detection_result)
        text = report_to_text(report)
        assert "Total features:" in text
        assert "2" in text

    def test_contains_classes(self, detection_result):
        report = generate_report(detection_result)
        text = report_to_text(report)
        assert "BUILDING" in text
        assert "PLATFORM" in text

    def test_with_input_path(self, detection_result):
        report = generate_report(detection_result, input_path="test.tif")
        text = report_to_text(report)
        assert "test.tif" in text


class TestReportToHtml:
    def test_is_valid_html(self, detection_result):
        report = generate_report(detection_result)
        html = report_to_html(report)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_stats(self, detection_result):
        report = generate_report(detection_result)
        html = report_to_html(report)
        assert "Total Features" in html
        assert "MayaScan" in html

    def test_contains_class_colors(self, detection_result):
        report = generate_report(detection_result)
        html = report_to_html(report)
        assert "#ff3c3c" in html  # building red
        assert "#3cc83c" in html  # platform green


class TestSaveReport:
    def test_save_text(self, detection_result, tmp_path):
        path = save_report(detection_result, tmp_path / "report.txt", format="text")
        assert path.exists()
        content = path.read_text()
        assert "MAYASCAN" in content

    def test_save_json(self, detection_result, tmp_path):
        path = save_report(detection_result, tmp_path / "report.json", format="json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_features"] == 2

    def test_save_html(self, detection_result, tmp_path):
        path = save_report(detection_result, tmp_path / "report.html", format="html")
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content
