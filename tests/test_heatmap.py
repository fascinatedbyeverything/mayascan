"""Tests for mayascan.heatmap — feature density visualization."""

import numpy as np
import pytest

from mayascan.features import Feature
from mayascan.heatmap import (
    class_density_maps,
    density_to_rgba,
    feature_density_map,
    save_density_png,
)


def _make_feature(col, row, class_name="building", area_m2=100.0):
    """Create a Feature at given coordinates."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[int(row), int(col)] = True
    return Feature(
        feature_id=1,
        class_id=1 if class_name == "building" else 2,
        class_name=class_name,
        pixel_count=1,
        area_m2=area_m2,
        confidence=0.9,
        centroid_row=float(row),
        centroid_col=float(col),
        centroid_geo=None,
        bbox=(int(row), int(col), int(row), int(col)),
        mask=mask,
    )


class TestFeatureDensityMap:
    def test_output_shape(self):
        features = [_make_feature(10, 10), _make_feature(20, 20)]
        result = feature_density_map(features, shape=(64, 64))
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_normalized_0_to_1(self):
        features = [_make_feature(32, 32)]
        result = feature_density_map(features, shape=(64, 64), sigma=5.0)
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        assert result.max() > 0  # should have some density

    def test_peak_near_centroid(self):
        features = [_make_feature(32, 32)]
        result = feature_density_map(features, shape=(64, 64), sigma=5.0)
        # Peak should be near the feature centroid
        peak_r, peak_c = np.unravel_index(result.argmax(), result.shape)
        assert abs(peak_r - 32) < 3
        assert abs(peak_c - 32) < 3

    def test_multiple_features(self):
        features = [
            _make_feature(10, 10),
            _make_feature(10, 12),
            _make_feature(12, 10),
        ]
        result = feature_density_map(features, shape=(64, 64), sigma=5.0)
        assert result.max() > 0

    def test_empty_features(self):
        result = feature_density_map([], shape=(64, 64))
        assert result.max() == 0.0

    def test_weight_by_area(self):
        features = [
            _make_feature(10, 10, area_m2=1000.0),
            _make_feature(50, 50, area_m2=10.0),
        ]
        result = feature_density_map(features, shape=(64, 64), sigma=5.0, weight_by_area=True)
        # The larger feature should create a higher peak
        val_at_big = result[10, 10]
        val_at_small = result[50, 50]
        assert val_at_big > val_at_small

    def test_sigma_effect(self):
        features = [_make_feature(32, 32)]
        narrow = feature_density_map(features, shape=(64, 64), sigma=3.0)
        wide = feature_density_map(features, shape=(64, 64), sigma=15.0)
        # Wider sigma should spread density more
        # Check that wide has more non-zero pixels
        assert (wide > 0.01).sum() > (narrow > 0.01).sum()


class TestClassDensityMaps:
    def test_returns_per_class(self):
        features = [
            _make_feature(10, 10, class_name="building"),
            _make_feature(50, 50, class_name="platform"),
        ]
        maps = class_density_maps(features, shape=(64, 64))
        assert "building" in maps
        assert "platform" in maps
        assert maps["building"].shape == (64, 64)

    def test_single_class(self):
        features = [_make_feature(10, 10, class_name="building")]
        maps = class_density_maps(features, shape=(64, 64))
        assert len(maps) == 1
        assert "building" in maps


class TestDensityToRGBA:
    def test_output_shape(self):
        density = np.random.rand(64, 64).astype(np.float32)
        rgba = density_to_rgba(density)
        assert rgba.shape == (64, 64, 4)
        assert rgba.dtype == np.uint8

    def test_zero_density_transparent(self):
        density = np.zeros((32, 32), dtype=np.float32)
        rgba = density_to_rgba(density)
        assert rgba[:, :, 3].max() == 0  # fully transparent

    def test_hot_colormap(self):
        density = np.ones((4, 4), dtype=np.float32)
        rgba = density_to_rgba(density, colormap="hot")
        assert rgba[0, 0, 0] > 0  # red channel active

    def test_cool_colormap(self):
        density = np.full((4, 4), 0.5, dtype=np.float32)
        rgba = density_to_rgba(density, colormap="cool")
        assert rgba[0, 0, 2] > 0  # blue channel active

    def test_viridis_colormap(self):
        density = np.full((4, 4), 0.5, dtype=np.float32)
        rgba = density_to_rgba(density, colormap="viridis")
        assert rgba.shape == (4, 4, 4)


class TestSaveDensityPng:
    def test_saves_file(self, tmp_path):
        density = np.random.rand(32, 32).astype(np.float32)
        out = tmp_path / "density.png"
        result = save_density_png(density, str(out))
        assert out.exists()
        assert result == str(out)

    def test_rgba_mode(self, tmp_path):
        from PIL import Image

        density = np.random.rand(32, 32).astype(np.float32)
        out = tmp_path / "density.png"
        save_density_png(density, str(out))
        img = Image.open(str(out))
        assert img.mode == "RGBA"
