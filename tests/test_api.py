"""Tests for the mayascan top-level public API."""

import numpy as np
import pytest

import mayascan
from mayascan.detect import DetectionResult


class TestPublicAPI:
    """Tests for the convenience functions exposed at package level."""

    def test_version(self):
        """mayascan exposes a __version__ string."""
        assert hasattr(mayascan, "__version__")
        assert isinstance(mayascan.__version__, str)
        assert len(mayascan.__version__) > 0

    def test_visualize_api(self):
        """mayascan.visualize(dem) returns (3, H, W) float32 for a 200x200 DEM."""
        rng = np.random.default_rng(42)
        dem = rng.random((200, 200)).astype(np.float64) * 100.0

        result = mayascan.visualize(dem)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 200, 200)
        assert result.dtype == np.float32
        # Each channel should be normalized to [0, 1]
        for ch in range(3):
            assert result[ch].min() >= 0.0
            assert result[ch].max() <= 1.0 + 1e-6

    def test_detect_api(self):
        """mayascan.detect(viz) returns DetectionResult with correct shape."""
        rng = np.random.default_rng(7)
        viz = rng.random((3, 200, 200)).astype(np.float32)

        result = mayascan.detect(viz)

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (200, 200)
        assert result.confidence.shape == (200, 200)
        assert result.classes.min() >= 0
        assert result.classes.max() < 4
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_process_dem(self):
        """mayascan.process_dem(dem) runs the full pipeline and returns DetectionResult."""
        rng = np.random.default_rng(123)
        dem = rng.random((200, 200)).astype(np.float64) * 50.0

        result = mayascan.process_dem(dem)

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (200, 200)
        assert result.confidence.shape == (200, 200)
        assert isinstance(result.class_names, dict)
        assert 0 in result.class_names
