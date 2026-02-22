"""Tests for mayascan.detect — tiled inference pipeline."""

import numpy as np
import pytest
import torch

from mayascan.detect import CLASS_NAMES, DetectionResult, run_detection


class TestDetectionResult:
    """Tests for the DetectionResult dataclass."""

    def test_detection_result_fields(self):
        """DetectionResult has classes, confidence, class_names attributes."""
        classes = np.zeros((10, 10), dtype=np.int64)
        confidence = np.ones((10, 10), dtype=np.float32)
        result = DetectionResult(
            classes=classes,
            confidence=confidence,
            class_names=dict(CLASS_NAMES),
        )

        assert result.classes is classes
        assert result.confidence is confidence
        assert isinstance(result.class_names, dict)
        assert result.class_names == {
            0: "background",
            1: "building",
            2: "platform",
            3: "aguada",
        }


class TestRunDetection:
    """Tests for run_detection end-to-end pipeline."""

    def test_run_detection_output_shape(self):
        """(3, 480, 480) input produces DetectionResult with (480, 480) maps."""
        rng = np.random.default_rng(42)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection(viz, model_path=None, tile_size=480, overlap=0.0, device="cpu")

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert result.classes.min() >= 0
        assert result.classes.max() < 4
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_run_detection_large_image(self):
        """(3, 960, 960) input works -- gets tiled internally."""
        rng = np.random.default_rng(7)
        viz = rng.random((3, 960, 960)).astype(np.float32)

        result = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.5, device="cpu"
        )

        assert result.classes.shape == (960, 960)
        assert result.confidence.shape == (960, 960)
        assert result.classes.min() >= 0
        assert result.classes.max() < 4

    def test_run_detection_confidence_threshold(self):
        """Higher threshold produces more background pixels than lower threshold."""
        rng = np.random.default_rng(123)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        # Fix random seed so both calls create identical model weights
        torch.manual_seed(0)
        result_low = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.0,
            confidence_threshold=0.0, device="cpu",
        )
        torch.manual_seed(0)
        result_high = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.0,
            confidence_threshold=0.99, device="cpu",
        )

        bg_low = (result_low.classes == 0).sum()
        bg_high = (result_high.classes == 0).sum()

        # Higher threshold should produce at least as much background
        assert bg_high >= bg_low, (
            f"Higher threshold should yield more background: "
            f"thresh=0.0 bg={bg_low}, thresh=0.99 bg={bg_high}"
        )
