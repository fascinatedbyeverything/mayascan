"""Tests for mayascan.visualize — DEM visualization rasters."""

import numpy as np
import pytest

from mayascan.visualize import (
    compute_slope,
    compute_visualizations,
)


class TestComputeSlope:
    """Tests for compute_slope."""

    def test_compute_slope_flat(self):
        """A perfectly flat DEM should produce near-zero slope everywhere."""
        dem = np.full((64, 64), 100.0, dtype=np.float64)
        slope = compute_slope(dem, resolution=0.5)

        assert slope.shape == (64, 64)
        assert slope.dtype == np.float32
        np.testing.assert_allclose(slope, 0.0, atol=1e-6)

    def test_compute_slope_ramp(self):
        """A linear ramp (tilted plane) should give uniform non-zero slope."""
        rows = np.arange(64, dtype=np.float64)
        dem = np.tile(rows[:, np.newaxis], (1, 64))  # rises 1 unit/row

        slope = compute_slope(dem, resolution=1.0)

        assert slope.shape == (64, 64)
        assert slope.dtype == np.float32

        # Interior pixels (away from boundary effects of np.gradient)
        interior = slope[2:-2, 2:-2]
        expected_deg = np.degrees(np.arctan(1.0))  # 45 degrees
        np.testing.assert_allclose(interior, expected_deg, atol=0.1)

        # Slope should be non-zero everywhere
        assert np.all(slope > 0.0)


class TestComputeVisualizations:
    """Tests for compute_visualizations."""

    def test_compute_visualizations_shape(self):
        """Output must be (3, H, W) float32."""
        dem = np.random.default_rng(42).random((48, 48)).astype(np.float64)
        vis = compute_visualizations(dem, resolution=0.5)

        assert vis.shape == (3, 48, 48)
        assert vis.dtype == np.float32

    def test_compute_visualizations_normalized(self):
        """Every channel must lie in [0, 1]."""
        rng = np.random.default_rng(7)
        dem = rng.random((48, 48)).astype(np.float64) * 500.0  # varied terrain

        vis = compute_visualizations(dem, resolution=0.5)

        for ch in range(3):
            channel = vis[ch]
            assert channel.min() >= 0.0, f"Channel {ch} has values below 0"
            assert channel.max() <= 1.0 + 1e-6, f"Channel {ch} has values above 1"
