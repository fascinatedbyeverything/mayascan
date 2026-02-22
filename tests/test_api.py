"""Tests for the mayascan top-level public API."""

import numpy as np
import pytest

import mayascan
from mayascan.detect import DetectionResult, GeoInfo


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

    def test_geo_info_defaults(self):
        """GeoInfo has sensible defaults."""
        geo = GeoInfo()
        assert geo.crs is None
        assert geo.transform is None
        assert geo.bounds is None
        assert geo.resolution == 0.5

    def test_detection_result_has_geo(self):
        """DetectionResult can carry GeoInfo."""
        classes = np.zeros((10, 10), dtype=np.int64)
        confidence = np.ones((10, 10), dtype=np.float32)
        geo = GeoInfo(crs="EPSG:32616", resolution=0.5)

        result = DetectionResult(
            classes=classes,
            confidence=confidence,
            geo=geo,
        )
        assert result.geo is not None
        assert result.geo.crs == "EPSG:32616"

    def test_read_raster_npy(self, tmp_path):
        """read_raster loads .npy files."""
        arr = np.random.default_rng(0).random((100, 100)).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), arr)

        data, geo = mayascan.read_raster(npy_path)
        assert data.shape == (100, 100)
        assert geo.crs is None
