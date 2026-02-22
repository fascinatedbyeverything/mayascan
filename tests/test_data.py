"""Tests for mayascan.data — dataset loading utilities."""

import os

import numpy as np
import pytest
from PIL import Image

from mayascan.data import (
    BinarySegmentationDataset,
    _extract_tile_id,
    count_tiles,
    list_available_classes,
)


@pytest.fixture
def tile_dir(tmp_path):
    """Create a minimal tile directory with lidar and mask files."""
    lidar_dir = tmp_path / "lidar"
    mask_dir = tmp_path / "masks"
    lidar_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(42)

    for i in range(10):
        tid = f"{i:04d}"
        # Create a 3-channel lidar tile
        img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(lidar_dir / f"tile_{tid}_lidar.tif")

        # Create building mask (tiles 2-4 have features)
        mask = np.full((32, 32), 255, dtype=np.uint8)  # 255 = background
        if 2 <= i <= 4:
            mask[8:24, 8:24] = 0  # 0 = feature present
        Image.fromarray(mask).save(mask_dir / f"tile_{tid}_mask_building.tif")

        # Create aguada mask (only tile 5 has a feature — rare class)
        mask_a = np.full((32, 32), 255, dtype=np.uint8)
        if i == 5:
            mask_a[12:20, 12:20] = 0
        Image.fromarray(mask_a).save(mask_dir / f"tile_{tid}_mask_aguada.tif")

    return lidar_dir, mask_dir


class TestBinarySegmentationDataset:
    def test_loads_train_split(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=False,
        )
        assert len(ds) == 8  # 80% of 10

    def test_loads_val_split(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="val", augment=False, oversample_positive=False,
        )
        assert len(ds) == 2  # 20% of 10

    def test_getitem_shapes(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=False,
        )
        img, mask = ds[0]
        assert img.shape == (3, 32, 32)
        assert mask.shape == (32, 32)

    def test_mask_is_binary(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=False,
        )
        for i in range(len(ds)):
            _, mask = ds[i]
            unique = set(mask.unique().tolist())
            assert unique <= {0.0, 1.0}

    def test_oversampling_increases_size(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds_no_os = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=False,
        )
        ds_os = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=True,
        )
        assert len(ds_os) >= len(ds_no_os)

    def test_stats(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=False, oversample_positive=False,
        )
        stats = ds.stats
        assert stats["class"] == "building"
        assert stats["split"] == "train"
        assert stats["positive_tiles"] + stats["negative_tiles"] == 8

    def test_augmented_output_valid(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "building",
            split="train", augment=True, oversample_positive=False,
        )
        img, mask = ds[0]
        assert img.shape == (3, 32, 32)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BinarySegmentationDataset(
                str(tmp_path / "nonexistent"), str(tmp_path), "building"
            )

    def test_rare_class(self, tile_dir):
        lidar_dir, mask_dir = tile_dir
        ds = BinarySegmentationDataset(
            str(lidar_dir), str(mask_dir), "aguada",
            split="train", augment=False, oversample_positive=False,
        )
        stats = ds.stats
        # Only tile 5 has aguada, and it's in the train split (80%)
        assert stats["positive_tiles"] <= 1


class TestExtractTileId:
    def test_standard_format(self):
        assert _extract_tile_id("/path/to/tile_0042_lidar.tif") == "0042"

    def test_multidigit(self):
        assert _extract_tile_id("tile_12345_lidar.tif") == "12345"


class TestListAvailableClasses:
    def test_finds_classes(self, tile_dir):
        _, mask_dir = tile_dir
        classes = list_available_classes(str(mask_dir))
        assert "building" in classes
        assert "aguada" in classes

    def test_empty_dir(self, tmp_path):
        assert list_available_classes(str(tmp_path)) == []


class TestCountTiles:
    def test_counts_correctly(self, tile_dir):
        lidar_dir, _ = tile_dir
        assert count_tiles(str(lidar_dir)) == 10

    def test_empty_dir(self, tmp_path):
        assert count_tiles(str(tmp_path)) == 0
