"""Tests for mayascan.crossval — K-fold cross-validation."""

import numpy as np
import pytest
from PIL import Image

from mayascan.crossval import FoldSplit, create_folds, fold_summary


@pytest.fixture
def tile_dir(tmp_path):
    """Create 20 dummy lidar tiles."""
    lidar_dir = tmp_path / "lidar"
    lidar_dir.mkdir()
    for i in range(20):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        Image.fromarray(img).save(lidar_dir / f"tile_{i:04d}_lidar.tif")
    return str(lidar_dir)


class TestCreateFolds:
    def test_default_5_folds(self, tile_dir):
        folds = create_folds(tile_dir, n_folds=5)
        assert len(folds) == 5

    def test_fold_indices(self, tile_dir):
        folds = create_folds(tile_dir)
        for i, f in enumerate(folds):
            assert f.fold == i

    def test_no_overlap_between_train_val(self, tile_dir):
        folds = create_folds(tile_dir)
        for f in folds:
            train_set = set(f.train_tiles)
            val_set = set(f.val_tiles)
            assert train_set & val_set == set()

    def test_all_tiles_covered(self, tile_dir):
        folds = create_folds(tile_dir)
        for f in folds:
            assert len(f.train_tiles) + len(f.val_tiles) == 20

    def test_val_sets_partition_all_tiles(self, tile_dir):
        folds = create_folds(tile_dir)
        all_val = []
        for f in folds:
            all_val.extend(f.val_tiles)
        # All 20 tiles should appear exactly once across all val sets
        assert len(all_val) == 20
        assert len(set(all_val)) == 20

    def test_reproducible(self, tile_dir):
        folds1 = create_folds(tile_dir, seed=42)
        folds2 = create_folds(tile_dir, seed=42)
        for f1, f2 in zip(folds1, folds2):
            assert f1.val_tiles == f2.val_tiles
            assert f1.train_tiles == f2.train_tiles

    def test_different_seed_different_split(self, tile_dir):
        folds1 = create_folds(tile_dir, seed=42)
        folds2 = create_folds(tile_dir, seed=99)
        # Very unlikely to be the same
        assert folds1[0].val_tiles != folds2[0].val_tiles

    def test_3_folds(self, tile_dir):
        folds = create_folds(tile_dir, n_folds=3)
        assert len(folds) == 3
        all_val = []
        for f in folds:
            all_val.extend(f.val_tiles)
        assert len(set(all_val)) == 20

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_folds(str(tmp_path))


class TestFoldSummary:
    def test_produces_string(self, tile_dir):
        folds = create_folds(tile_dir)
        summary = fold_summary(folds)
        assert "5 folds" in summary
        assert "Total tiles: 20" in summary
        assert "Fold 0:" in summary
        assert "Fold 4:" in summary

    def test_train_val_counts(self, tile_dir):
        folds = create_folds(tile_dir, n_folds=5)
        summary = fold_summary(folds)
        # Each fold should have 16 train and 4 val
        assert "train=16" in summary
        assert "val=4" in summary
