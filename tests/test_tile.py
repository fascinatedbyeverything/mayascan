"""Tests for mayascan.tile — slice and stitch raster tiles."""

import numpy as np
import pytest

from mayascan.tile import slice_tiles, stitch_tiles


class TestSliceTiles:
    """Tests for slice_tiles."""

    def test_slice_tiles_no_overlap(self):
        """A 960x960 image with tile_size=480 and no overlap → 4 tiles."""
        img = np.random.default_rng(42).random((3, 960, 960))
        tiles, origins = slice_tiles(img, tile_size=480, overlap=0.0)

        assert len(tiles) == 4
        assert len(origins) == 4
        for t in tiles:
            assert t.shape == (3, 480, 480)

        # Check the four expected origins.
        expected_origins = [(0, 0), (0, 480), (480, 0), (480, 480)]
        assert origins == expected_origins

        # Verify pixel content matches the source image.
        np.testing.assert_array_equal(tiles[0], img[:, 0:480, 0:480])
        np.testing.assert_array_equal(tiles[3], img[:, 480:960, 480:960])

    def test_slice_tiles_with_overlap(self):
        """50 % overlap on 960x960 → stride 240 → 3x3 = 9 tiles."""
        img = np.random.default_rng(7).random((1, 960, 960))
        tiles, origins = slice_tiles(img, tile_size=480, overlap=0.5)

        assert len(tiles) == 9
        for t in tiles:
            assert t.shape == (1, 480, 480)

        # First row starts at columns 0, 240, 480.
        row_starts = sorted({o[0] for o in origins})
        col_starts = sorted({o[1] for o in origins})
        assert row_starts == [0, 240, 480]
        assert col_starts == [0, 240, 480]

    def test_slice_tiles_pads_remainder(self):
        """A 500x700 image that doesn't divide evenly is padded."""
        img = np.ones((2, 500, 700), dtype=np.float32)
        tiles, origins = slice_tiles(img, tile_size=480, overlap=0.0)

        # All tiles must be full-sized.
        for t in tiles:
            assert t.shape == (2, 480, 480)

        # There should be enough tiles to cover the original extent.
        assert len(tiles) >= 2  # at least 2 tiles needed to cover 700 cols


class TestStitchTiles:
    """Tests for stitch_tiles."""

    def test_stitch_tiles_roundtrip(self):
        """slice → stitch recovers the original shape."""
        rng = np.random.default_rng(99)
        img = rng.random((3, 960, 960)).astype(np.float32)

        tiles, origins = slice_tiles(img, tile_size=480, overlap=0.0)
        out = stitch_tiles(tiles, origins, img.shape, overlap=0.0)

        assert out.shape == img.shape
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_stitch_tiles_overlap_blending(self):
        """Uniform tiles with overlap stitch to a uniform output."""
        C, H, W = 1, 960, 960
        value = 42.0
        img = np.full((C, H, W), value, dtype=np.float32)

        tiles, origins = slice_tiles(img, tile_size=480, overlap=0.5)
        out = stitch_tiles(tiles, origins, (C, H, W), overlap=0.5)

        assert out.shape == (C, H, W)
        # Because every tile has the same constant, the average in
        # overlapping regions should still be exactly that constant.
        np.testing.assert_allclose(out, value, atol=1e-6)
