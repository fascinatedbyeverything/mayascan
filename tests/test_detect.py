"""Tests for mayascan.detect — tiled inference pipeline."""

import numpy as np
import pytest

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
        """threshold=0.99 forces most predictions to background (0)."""
        rng = np.random.default_rng(123)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection(
            viz,
            model_path=None,
            tile_size=480,
            overlap=0.0,
            confidence_threshold=0.99,
            device="cpu",
        )

        # With random weights the softmax confidence is rarely > 0.99,
        # so almost all non-background pixels should be reset to background.
        bg_fraction = (result.classes == 0).sum() / result.classes.size
        assert bg_fraction > 0.9, (
            f"Expected >90% background with threshold=0.99, got {bg_fraction:.2%}"
        )
