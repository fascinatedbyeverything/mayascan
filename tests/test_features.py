"""Tests for mayascan.features — feature extraction and filtering."""

import numpy as np
import pytest

from mayascan.detect import CLASS_NAMES, DetectionResult, GeoInfo
from mayascan.features import extract_features, filter_features, feature_summary


@pytest.fixture
def detection_result():
    """DetectionResult with a building block and a platform block."""
    classes = np.zeros((50, 50), dtype=np.int64)
    confidence = np.full((50, 50), 0.1, dtype=np.float32)

    # Building block: 20x20 = 400 pixels
    classes[5:25, 5:25] = 1
    confidence[5:25, 5:25] = 0.95

    # Platform block: 10x10 = 100 pixels
    classes[30:40, 30:40] = 2
    confidence[30:40, 30:40] = 0.85

    # Small building: 3x3 = 9 pixels
    classes[45:48, 45:48] = 1
    confidence[45:48, 45:48] = 0.60

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )


class TestExtractFeatures:
    def test_extracts_all_features(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        assert len(features) == 3  # 2 buildings + 1 platform

    def test_sorted_by_area(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        areas = [f.area_m2 for f in features]
        assert areas == sorted(areas, reverse=True)

    def test_feature_fields(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        f = features[0]  # Largest feature (building, 20x20)
        assert f.class_name == "building"
        assert f.class_id == 1
        assert f.pixel_count == 400
        assert f.area_m2 == 100.0  # 400 * 0.5 * 0.5
        assert 0.9 < f.confidence <= 1.0
        assert f.mask.shape == (50, 50)
        assert f.mask.sum() == 400

    def test_with_geo(self):
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
        features = extract_features(result)
        assert features[0].centroid_geo is not None
        assert features[0].centroid_geo[0] > 500000


class TestFilterFeatures:
    def test_filter_by_min_area(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        filtered = filter_features(features, min_area=10.0)
        assert len(filtered) == 2  # excludes 3x3 small building (2.25 m2)

    def test_filter_by_max_area(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        filtered = filter_features(features, max_area=50.0)
        assert len(filtered) == 2  # excludes 20x20 building (100 m2)

    def test_filter_by_confidence(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        filtered = filter_features(features, min_confidence=0.8)
        assert len(filtered) == 2  # excludes small building (conf=0.60)

    def test_filter_by_class(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        filtered = filter_features(features, classes=["building"])
        assert len(filtered) == 2
        assert all(f.class_name == "building" for f in filtered)

    def test_filter_combined(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        filtered = filter_features(
            features, min_area=10.0, min_confidence=0.8, classes=["building"]
        )
        assert len(filtered) == 1
        assert filtered[0].pixel_count == 400


class TestFeatureSummary:
    def test_summary_counts(self, detection_result):
        features = extract_features(detection_result, pixel_size=0.5)
        summary = feature_summary(features)
        assert summary["total_count"] == 3
        assert summary["total_area_m2"] > 0
        assert "building" in summary["by_class"]
        assert "platform" in summary["by_class"]
        assert summary["by_class"]["building"]["count"] == 2
        assert summary["by_class"]["platform"]["count"] == 1

    def test_empty_summary(self):
        summary = feature_summary([])
        assert summary["total_count"] == 0
        assert summary["total_area_m2"] == 0.0
