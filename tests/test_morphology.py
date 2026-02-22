"""Tests for mayascan.morphology — shape analysis and settlement patterns."""

import numpy as np
import pytest

from mayascan.config import CLASS_NAMES
from mayascan.detect import DetectionResult
from mayascan.features import Feature, extract_features
from mayascan.morphology import (
    FeatureProfile,
    ShapeDescriptors,
    analyze_features,
    classify_structure,
    compute_perimeter,
    compute_shape_descriptors,
    settlement_summary,
)


def _make_feature(mask, class_id=1, class_name="building", area_m2=100.0, confidence=0.9):
    """Helper to create a Feature from a mask."""
    rows, cols = np.where(mask)
    return Feature(
        feature_id=1,
        class_id=class_id,
        class_name=class_name,
        pixel_count=int(mask.sum()),
        area_m2=area_m2,
        confidence=confidence,
        centroid_row=float(rows.mean()),
        centroid_col=float(cols.mean()),
        centroid_geo=None,
        bbox=(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
        mask=mask,
    )


@pytest.fixture
def square_mask():
    """20x20 square in a 64x64 image."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


@pytest.fixture
def circle_mask():
    """Approximate circle (radius 15) in a 64x64 image."""
    mask = np.zeros((64, 64), dtype=bool)
    cy, cx = 32, 32
    for r in range(64):
        for c in range(64):
            if (r - cy) ** 2 + (c - cx) ** 2 <= 15 ** 2:
                mask[r, c] = True
    return mask


@pytest.fixture
def elongated_mask():
    """5x40 rectangle in a 64x64 image."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:25, 10:50] = True
    return mask


class TestComputePerimeter:
    def test_square_perimeter(self, square_mask):
        perim = compute_perimeter(square_mask)
        # A 20x20 square has perimeter ≈ 76 boundary pixels
        assert 60 < perim < 100

    def test_circle_perimeter(self, circle_mask):
        perim = compute_perimeter(circle_mask)
        assert perim > 0

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        perim = compute_perimeter(mask)
        assert perim == 1.0


class TestComputeShapeDescriptors:
    def test_square_compactness(self, square_mask):
        sd = compute_shape_descriptors(square_mask)
        # Square compactness = 4*pi*400 / (76^2) ≈ 0.87
        assert 0.5 < sd.compactness <= 1.0

    def test_circle_compactness(self, circle_mask):
        sd = compute_shape_descriptors(circle_mask)
        # Circle should be more compact than square
        assert sd.compactness > 0.6

    def test_square_low_elongation(self, square_mask):
        sd = compute_shape_descriptors(square_mask)
        # Square should have elongation near 1.0
        assert 0.8 < sd.elongation < 1.5

    def test_elongated_high_elongation(self, elongated_mask):
        sd = compute_shape_descriptors(elongated_mask)
        # 5x40 should have high elongation
        assert sd.elongation > 2.0

    def test_square_high_rectangularity(self, square_mask):
        sd = compute_shape_descriptors(square_mask)
        # Square should have rectangularity near 1.0
        assert sd.rectangularity > 0.7

    def test_solidity(self, square_mask):
        sd = compute_shape_descriptors(square_mask)
        # Convex shape should have high solidity
        assert sd.solidity > 0.8

    def test_equivalent_diameter(self, square_mask):
        sd = compute_shape_descriptors(square_mask)
        # Area = 400, diameter = 2*sqrt(400/pi) ≈ 22.6
        assert 20 < sd.equivalent_diameter_px < 25

    def test_orientation_range(self, elongated_mask):
        sd = compute_shape_descriptors(elongated_mask)
        assert 0 <= sd.orientation_deg <= 180

    def test_tiny_feature(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        sd = compute_shape_descriptors(mask)
        assert sd.compactness == 0.0


class TestClassifyStructure:
    def test_large_building_is_temple(self, square_mask):
        feat = _make_feature(square_mask, area_m2=3000.0)
        sd = compute_shape_descriptors(square_mask)
        result = classify_structure(feat, sd)
        assert result == "temple_pyramid"

    def test_medium_building_is_elite(self, square_mask):
        feat = _make_feature(square_mask, area_m2=800.0)
        sd = compute_shape_descriptors(square_mask)
        result = classify_structure(feat, sd)
        assert result == "elite_residence"

    def test_small_rectangular_building(self, square_mask):
        feat = _make_feature(square_mask, area_m2=100.0)
        sd = compute_shape_descriptors(square_mask)
        result = classify_structure(feat, sd)
        assert result == "residential_mound"

    def test_large_platform_is_grand_plaza(self, square_mask):
        feat = _make_feature(square_mask, class_id=2, class_name="platform", area_m2=6000.0)
        sd = compute_shape_descriptors(square_mask)
        result = classify_structure(feat, sd)
        assert result == "grand_plaza"

    def test_elongated_platform_is_causeway(self, elongated_mask):
        feat = _make_feature(elongated_mask, class_id=2, class_name="platform", area_m2=200.0)
        sd = compute_shape_descriptors(elongated_mask)
        result = classify_structure(feat, sd)
        assert result == "causeway_segment"

    def test_large_aguada_is_large_reservoir(self, circle_mask):
        feat = _make_feature(circle_mask, class_id=3, class_name="aguada", area_m2=3000.0)
        sd = compute_shape_descriptors(circle_mask)
        result = classify_structure(feat, sd)
        assert result == "large_reservoir"

    def test_compact_aguada_is_circular(self, circle_mask):
        feat = _make_feature(circle_mask, class_id=3, class_name="aguada", area_m2=500.0)
        sd = compute_shape_descriptors(circle_mask)
        result = classify_structure(feat, sd)
        assert result == "circular_reservoir"


class TestAnalyzeFeatures:
    def test_returns_profiles(self, square_mask, circle_mask):
        f1 = _make_feature(square_mask, area_m2=100.0)
        f2 = _make_feature(circle_mask, class_id=3, class_name="aguada", area_m2=50.0)
        f2.feature_id = 2
        profiles = analyze_features([f1, f2])
        assert len(profiles) == 2
        assert all(isinstance(p, FeatureProfile) for p in profiles)

    def test_nearest_neighbor(self, square_mask, circle_mask):
        f1 = _make_feature(square_mask, area_m2=100.0)
        f1.centroid_col = 20.0
        f1.centroid_row = 20.0
        f2 = _make_feature(circle_mask, area_m2=50.0)
        f2.feature_id = 2
        f2.centroid_col = 32.0
        f2.centroid_row = 32.0
        profiles = analyze_features([f1, f2])
        # Both should have finite nn distance
        assert profiles[0].nearest_neighbor_px < float("inf")
        assert profiles[1].nearest_neighbor_px < float("inf")

    def test_same_class_nn(self, square_mask):
        # Two features of the same class
        mask1 = np.zeros((64, 64), dtype=bool)
        mask1[5:15, 5:15] = True
        mask2 = np.zeros((64, 64), dtype=bool)
        mask2[40:50, 40:50] = True
        f1 = _make_feature(mask1, area_m2=25.0)
        f2 = _make_feature(mask2, area_m2=25.0)
        f2.feature_id = 2
        f2.centroid_row = 45.0
        f2.centroid_col = 45.0
        profiles = analyze_features([f1, f2])
        assert profiles[0].nearest_same_class_px < float("inf")

    def test_empty_features(self):
        profiles = analyze_features([])
        assert profiles == []

    def test_single_feature(self, square_mask):
        f = _make_feature(square_mask, area_m2=100.0)
        profiles = analyze_features([f])
        assert len(profiles) == 1
        assert profiles[0].nearest_neighbor_px == float("inf")


class TestSettlementSummary:
    def test_basic_summary(self, square_mask, circle_mask):
        f1 = _make_feature(square_mask, area_m2=100.0)
        f2 = _make_feature(circle_mask, class_id=3, class_name="aguada", area_m2=50.0)
        f2.feature_id = 2
        profiles = analyze_features([f1, f2])
        summary = settlement_summary(profiles)
        assert summary["total_features"] == 2
        assert len(summary["structure_types"]) > 0

    def test_spatial_stats(self, square_mask, circle_mask):
        f1 = _make_feature(square_mask, area_m2=100.0)
        f2 = _make_feature(circle_mask, area_m2=50.0)
        f2.feature_id = 2
        profiles = analyze_features([f1, f2])
        summary = settlement_summary(profiles, pixel_size=0.5)
        assert "mean_nn_distance_m" in summary["spatial"]

    def test_empty_summary(self):
        summary = settlement_summary([])
        assert summary["total_features"] == 0

    def test_shape_stats(self, square_mask):
        f = _make_feature(square_mask, area_m2=100.0)
        profiles = analyze_features([f])
        summary = settlement_summary(profiles)
        assert "mean_compactness" in summary["shape_stats"]
        assert "mean_elongation" in summary["shape_stats"]
