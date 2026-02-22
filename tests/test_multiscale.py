"""Tests for mayascan.multiscale — multi-scale inference."""

import numpy as np
import pytest

from mayascan.multiscale import DEFAULT_SCALES, run_multiscale_detection


class TestDefaults:
    def test_default_scales(self):
        assert len(DEFAULT_SCALES) == 3
        assert all(isinstance(s, int) for s in DEFAULT_SCALES)
        assert DEFAULT_SCALES == sorted(DEFAULT_SCALES)


class TestRunMultiscaleDetection:
    def test_single_scale_passthrough(self, tmp_path, monkeypatch):
        """Single scale should just call run_detection_v2 directly."""
        import sys
        import mayascan.multiscale as ms_mod

        calls = []

        def mock_detect(visualization, **kwargs):
            from mayascan.detect import DetectionResult
            from mayascan.config import CLASS_NAMES

            calls.append(kwargs.get("tile_size"))
            h, w = visualization.shape[1], visualization.shape[2]
            return DetectionResult(
                classes=np.zeros((h, w), dtype=np.int64),
                confidence=np.full((h, w), 0.5, dtype=np.float32),
                class_names=dict(CLASS_NAMES),
            )

        monkeypatch.setattr(ms_mod, "run_detection_v2", mock_detect)

        viz = np.random.rand(3, 64, 64).astype(np.float32)
        result = run_multiscale_detection(
            viz, model_dir=str(tmp_path), scales=[480],
        )
        assert len(calls) == 1
        assert calls[0] == 480
        assert result.classes.shape == (64, 64)

    def test_multiple_scales_merges(self, tmp_path, monkeypatch):
        """Multiple scales should produce multiple calls and merge."""
        import mayascan.multiscale as ms_mod

        calls = []

        def mock_detect(visualization, **kwargs):
            from mayascan.detect import DetectionResult
            from mayascan.config import CLASS_NAMES

            calls.append(kwargs.get("tile_size"))
            h, w = visualization.shape[1], visualization.shape[2]
            classes = np.zeros((h, w), dtype=np.int64)
            # Each scale detects a building in a slightly different spot
            offset = len(calls) * 5
            classes[10 + offset:20 + offset, 10:20] = 1
            return DetectionResult(
                classes=classes,
                confidence=np.full((h, w), 0.8, dtype=np.float32),
                class_names=dict(CLASS_NAMES),
            )

        monkeypatch.setattr(ms_mod, "run_detection_v2", mock_detect)

        viz = np.random.rand(3, 64, 64).astype(np.float32)
        result = run_multiscale_detection(
            viz, model_dir=str(tmp_path), scales=[320, 480, 640],
        )
        assert len(calls) == 3
        assert result.classes.shape == (64, 64)

    def test_uses_default_scales(self, tmp_path, monkeypatch):
        """No scales argument should use DEFAULT_SCALES."""
        import mayascan.multiscale as ms_mod

        calls = []

        def mock_detect(visualization, **kwargs):
            from mayascan.detect import DetectionResult
            from mayascan.config import CLASS_NAMES

            calls.append(kwargs.get("tile_size"))
            h, w = visualization.shape[1], visualization.shape[2]
            return DetectionResult(
                classes=np.zeros((h, w), dtype=np.int64),
                confidence=np.full((h, w), 0.5, dtype=np.float32),
                class_names=dict(CLASS_NAMES),
            )

        monkeypatch.setattr(ms_mod, "run_detection_v2", mock_detect)

        viz = np.random.rand(3, 64, 64).astype(np.float32)
        result = run_multiscale_detection(viz, model_dir=str(tmp_path))
        assert len(calls) == 3
        assert set(calls) == set(DEFAULT_SCALES)

    def test_vote_method(self, tmp_path, monkeypatch):
        """Vote merge method should work."""
        import mayascan.multiscale as ms_mod

        def mock_detect(visualization, **kwargs):
            from mayascan.detect import DetectionResult
            from mayascan.config import CLASS_NAMES

            h, w = visualization.shape[1], visualization.shape[2]
            return DetectionResult(
                classes=np.ones((h, w), dtype=np.int64),
                confidence=np.full((h, w), 0.9, dtype=np.float32),
                class_names=dict(CLASS_NAMES),
            )

        monkeypatch.setattr(ms_mod, "run_detection_v2", mock_detect)

        viz = np.random.rand(3, 32, 32).astype(np.float32)
        result = run_multiscale_detection(
            viz, model_dir=str(tmp_path),
            scales=[320, 480], merge_method="vote",
        )
        # All models predicted class 1 -> vote should agree
        assert (result.classes == 1).all()
