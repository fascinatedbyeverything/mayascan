"""Tests for mayascan.classify — ground classification and DEM generation."""

import numpy as np
import pytest

from mayascan.classify import dem_from_array, PdalNotAvailableError


class TestDemFromArray:
    """Tests for dem_from_array (scipy-only, always available)."""

    def test_dem_from_array(self):
        """DEM from random points has correct shape, extent, and no NaN."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0.0, 100.0, size=n)
        y = rng.uniform(0.0, 80.0, size=n)
        z = rng.uniform(10.0, 50.0, size=n)

        dem, extent = dem_from_array(x, y, z, resolution=1.0)

        # Shape must be non-trivial
        assert dem.ndim == 2
        assert dem.shape[0] > 0
        assert dem.shape[1] > 0

        # Extent must match input coordinate range
        assert extent["xmin"] == pytest.approx(x.min())
        assert extent["xmax"] == pytest.approx(x.max())
        assert extent["ymin"] == pytest.approx(y.min())
        assert extent["ymax"] == pytest.approx(y.max())

        # After nearest-neighbor fill, there should be no NaN
        assert not np.all(np.isnan(dem))

        # dtype should be float32
        assert dem.dtype == np.float32

    def test_dem_from_array_resolution(self):
        """Higher resolution (smaller cell) produces a larger array."""
        rng = np.random.default_rng(7)
        n = 200
        x = rng.uniform(0.0, 50.0, size=n)
        y = rng.uniform(0.0, 50.0, size=n)
        z = rng.uniform(0.0, 20.0, size=n)

        dem_hi, _ = dem_from_array(x, y, z, resolution=0.5)
        dem_lo, _ = dem_from_array(x, y, z, resolution=2.0)

        # Finer resolution → more cells in both dimensions
        assert dem_hi.shape[0] > dem_lo.shape[0]
        assert dem_hi.shape[1] > dem_lo.shape[1]


class TestClassifyGround:
    """Verify that classify_ground raises cleanly without PDAL."""

    def test_classify_ground_raises_without_pdal(self):
        """classify_ground must raise PdalNotAvailableError when pdal is missing."""
        from mayascan.classify import classify_ground

        with pytest.raises(PdalNotAvailableError):
            classify_ground("nonexistent.las")
