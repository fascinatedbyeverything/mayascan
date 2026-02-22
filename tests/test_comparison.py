"""Tests for mayascan.comparison — detection result comparison."""

import numpy as np
import pytest

from mayascan.config import CLASS_NAMES
from mayascan.detect import DetectionResult
from mayascan.comparison import (
    ComparisonResult,
    compare_detections,
    comparison_summary,
    count_feature_changes,
    difference_map,
)


def _make_result(classes, confidence=None):
    """Create a DetectionResult from a class map."""
    if confidence is None:
        confidence = np.where(classes > 0, 0.9, 0.0).astype(np.float32)
    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )


@pytest.fixture
def identical_results():
    """Two identical detection results."""
    classes = np.zeros((50, 50), dtype=np.int64)
    classes[10:20, 10:20] = 1
    classes[30:40, 30:40] = 2
    return _make_result(classes.copy()), _make_result(classes.copy())


@pytest.fixture
def different_results():
    """Two completely different detection results."""
    ca = np.zeros((50, 50), dtype=np.int64)
    ca[10:20, 10:20] = 1
    cb = np.zeros((50, 50), dtype=np.int64)
    cb[30:40, 30:40] = 2
    return _make_result(ca), _make_result(cb)


@pytest.fixture
def overlapping_results():
    """Results with partial overlap."""
    ca = np.zeros((50, 50), dtype=np.int64)
    ca[10:25, 10:25] = 1
    cb = np.zeros((50, 50), dtype=np.int64)
    cb[15:30, 15:30] = 1
    return _make_result(ca), _make_result(cb)


class TestCompareDetections:
    def test_identical(self, identical_results):
        a, b = identical_results
        comp = compare_detections(a, b)
        assert comp.jaccard == 1.0
        assert comp.dice == 1.0
        assert comp.only_a.sum() == 0
        assert comp.only_b.sum() == 0

    def test_completely_different(self, different_results):
        a, b = different_results
        comp = compare_detections(a, b)
        assert comp.jaccard == 0.0
        assert comp.agreement.sum() == 0
        assert comp.only_a.sum() > 0
        assert comp.only_b.sum() > 0

    def test_partial_overlap(self, overlapping_results):
        a, b = overlapping_results
        comp = compare_detections(a, b)
        assert 0 < comp.jaccard < 1.0
        assert comp.agreement.sum() > 0
        assert comp.only_a.sum() > 0
        assert comp.only_b.sum() > 0

    def test_per_class_jaccard(self, identical_results):
        a, b = identical_results
        comp = compare_detections(a, b)
        assert "building" in comp.per_class_jaccard
        assert "platform" in comp.per_class_jaccard
        assert comp.per_class_jaccard["building"] == 1.0

    def test_class_change(self):
        ca = np.zeros((20, 20), dtype=np.int64)
        ca[5:15, 5:15] = 1
        cb = np.zeros((20, 20), dtype=np.int64)
        cb[5:15, 5:15] = 2
        comp = compare_detections(_make_result(ca), _make_result(cb))
        assert comp.changed_class.sum() == 100

    def test_empty_detections(self):
        ca = np.zeros((20, 20), dtype=np.int64)
        cb = np.zeros((20, 20), dtype=np.int64)
        comp = compare_detections(_make_result(ca), _make_result(cb))
        assert comp.jaccard == 1.0


class TestComparisonSummary:
    def test_summary_structure(self, overlapping_results):
        a, b = overlapping_results
        comp = compare_detections(a, b)
        summary = comparison_summary(comp)
        assert "agreement_pixels" in summary
        assert "only_a_pixels" in summary
        assert "only_b_pixels" in summary
        assert "jaccard" in summary
        assert "dice" in summary
        assert "agreement_pct" in summary

    def test_agreement_pct_range(self, overlapping_results):
        a, b = overlapping_results
        comp = compare_detections(a, b)
        summary = comparison_summary(comp)
        assert 0 <= summary["agreement_pct"] <= 100


class TestDifferenceMap:
    def test_output_shape(self, overlapping_results):
        a, b = overlapping_results
        rgba = difference_map(a, b)
        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8

    def test_agreement_is_green(self, identical_results):
        a, b = identical_results
        rgba = difference_map(a, b)
        feature_mask = a.classes > 0
        assert rgba[feature_mask, 1].mean() > 100

    def test_only_a_is_red(self, different_results):
        a, b = different_results
        rgba = difference_map(a, b)
        only_a_mask = (a.classes > 0) & (b.classes == 0)
        assert rgba[only_a_mask, 0].mean() > 200

    def test_only_b_is_blue(self, different_results):
        a, b = different_results
        rgba = difference_map(a, b)
        only_b_mask = (b.classes > 0) & (a.classes == 0)
        assert rgba[only_b_mask, 2].mean() > 200


class TestCountFeatureChanges:
    def test_identical(self, identical_results):
        a, b = identical_results
        changes = count_feature_changes(a, b)
        assert changes["building"]["gained"] == 0
        assert changes["building"]["lost"] == 0
        assert changes["building"]["stable"] >= 1

    def test_gained_features(self, different_results):
        a, b = different_results
        changes = count_feature_changes(a, b)
        assert changes["platform"]["gained"] == 1
        assert changes["building"]["lost"] == 1

    def test_empty_detections(self):
        ca = np.zeros((20, 20), dtype=np.int64)
        cb = np.zeros((20, 20), dtype=np.int64)
        changes = count_feature_changes(_make_result(ca), _make_result(cb))
        for cls_name in ["building", "platform", "aguada"]:
            assert changes[cls_name]["count_a"] == 0
            assert changes[cls_name]["count_b"] == 0
