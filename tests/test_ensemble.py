"""Tests for mayascan.ensemble — model ensemble utilities."""

import numpy as np
import pytest

from mayascan.detect import CLASS_NAMES, DetectionResult
from mayascan.ensemble import (
    average_probabilities,
    majority_vote,
    merge_results,
)


@pytest.fixture
def two_results():
    """Create two DetectionResults with partially overlapping predictions."""
    # Result 1: building block at top-left
    classes1 = np.zeros((50, 50), dtype=np.int64)
    conf1 = np.full((50, 50), 0.1, dtype=np.float32)
    classes1[5:25, 5:25] = 1
    conf1[5:25, 5:25] = 0.9

    # Result 2: building block shifted right by 5px
    classes2 = np.zeros((50, 50), dtype=np.int64)
    conf2 = np.full((50, 50), 0.1, dtype=np.float32)
    classes2[5:25, 10:30] = 1
    conf2[5:25, 10:30] = 0.85

    r1 = DetectionResult(classes=classes1, confidence=conf1, class_names=dict(CLASS_NAMES))
    r2 = DetectionResult(classes=classes2, confidence=conf2, class_names=dict(CLASS_NAMES))
    return r1, r2


class TestAverageProbabilities:
    def test_equal_weights(self):
        p1 = np.array([[[0.8, 0.2], [0.1, 0.9]]])
        p2 = np.array([[[0.6, 0.4], [0.3, 0.7]]])
        avg = average_probabilities([p1, p2])
        np.testing.assert_allclose(avg[0, 0, 0], 0.7, atol=1e-6)
        np.testing.assert_allclose(avg[0, 1, 1], 0.8, atol=1e-6)

    def test_weighted(self):
        p1 = np.array([[[1.0]]])
        p2 = np.array([[[0.0]]])
        avg = average_probabilities([p1, p2], weights=[3.0, 1.0])
        np.testing.assert_allclose(avg[0, 0, 0], 0.75, atol=1e-6)

    def test_single_input(self):
        p = np.array([[[0.5]]])
        avg = average_probabilities([p])
        np.testing.assert_array_equal(avg, p)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            average_probabilities([])


class TestMajorityVote:
    def test_unanimous(self):
        c1 = np.array([[1, 0], [0, 2]])
        c2 = np.array([[1, 0], [0, 2]])
        result = majority_vote([c1, c2])
        np.testing.assert_array_equal(result, c1)

    def test_split_vote(self):
        c1 = np.array([[1, 0]])
        c2 = np.array([[0, 0]])
        c3 = np.array([[0, 0]])
        result = majority_vote([c1, c2, c3])
        # Pixel (0,0): 1 vote for class 1, 2 votes for class 0 -> class 0
        assert result[0, 0] == 0

    def test_tie_breaks_to_lower_class(self):
        c1 = np.array([[1]])
        c2 = np.array([[2]])
        result = majority_vote([c1, c2])
        # Tie at 1 vote each — argmax returns first maximum (class 1)
        assert result[0, 0] == 1


class TestMergeResults:
    def test_vote_method(self, two_results):
        r1, r2 = two_results
        merged = merge_results([r1, r2], method="vote")
        assert isinstance(merged, DetectionResult)
        assert merged.classes.shape == (50, 50)
        # Overlapping region should be building
        assert merged.classes[10, 15] == 1

    def test_average_method(self, two_results):
        r1, r2 = two_results
        merged = merge_results([r1, r2], method="average", confidence_threshold=0.3)
        assert isinstance(merged, DetectionResult)
        assert merged.classes.shape == (50, 50)
        assert merged.confidence.shape == (50, 50)

    def test_single_result_passthrough(self, two_results):
        r1, _ = two_results
        merged = merge_results([r1])
        assert merged is r1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_results([])

    def test_unknown_method_raises(self, two_results):
        r1, r2 = two_results
        with pytest.raises(ValueError, match="Unknown method"):
            merge_results([r1, r2], method="invalid")

    def test_preserves_class_names(self, two_results):
        r1, r2 = two_results
        merged = merge_results([r1, r2], method="vote")
        assert merged.class_names == r1.class_names
