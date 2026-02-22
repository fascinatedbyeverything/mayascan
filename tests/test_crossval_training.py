"""Tests for cross-validation training and fold ensemble inference."""

import importlib
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from mayascan.crossval import (
    FoldSplit,
    create_folds,
    discover_fold_models,
    fold_summary,
)
from mayascan.data import BinarySegmentationDataset


@pytest.fixture
def tile_dir(tmp_path):
    """Create a small set of fake tiles for testing."""
    lidar_dir = tmp_path / "lidar"
    mask_dir = tmp_path / "masks"
    lidar_dir.mkdir()
    mask_dir.mkdir()

    for i in range(20):
        # Create 3-channel lidar tile
        img = np.random.randint(0, 256, (480, 480, 3), dtype=np.uint8)
        Image.fromarray(img).save(str(lidar_dir / f"tile_{i:04d}_lidar.tif"))

        # Create binary masks for each class
        for cls_name in ["building", "platform", "aguada"]:
            # Some tiles have positive pixels
            if i % 3 == 0:  # ~33% positive rate
                mask = np.ones((480, 480), dtype=np.uint8) * 255
                mask[100:200, 100:200] = 0  # Dark = positive
            else:
                mask = np.ones((480, 480), dtype=np.uint8) * 255
            Image.fromarray(mask).save(
                str(mask_dir / f"tile_{i:04d}_mask_{cls_name}.tif")
            )

    return tmp_path


class TestFoldAwareDataset:
    def test_explicit_tile_paths(self, tile_dir):
        """Dataset should use explicit tile paths when provided."""
        lidar_dir = str(tile_dir / "lidar")
        mask_dir = str(tile_dir / "masks")

        # Create folds
        folds = create_folds(lidar_dir, n_folds=5)
        fold = folds[0]

        # Create dataset with explicit paths
        ds = BinarySegmentationDataset(
            lidar_dir, mask_dir, "building",
            split="train", augment=False,
            tile_paths=fold.train_tiles,
        )

        # Should have the same tiles as the fold split
        assert len(ds) >= len(fold.train_tiles)  # May have oversampled

    def test_fold_train_val_disjoint(self, tile_dir):
        """Train and val tiles should be disjoint within a fold."""
        lidar_dir = str(tile_dir / "lidar")
        folds = create_folds(lidar_dir, n_folds=5)

        for fold in folds:
            train_set = set(fold.train_tiles)
            val_set = set(fold.val_tiles)
            assert len(train_set & val_set) == 0

    def test_fold_val_covers_all_tiles(self, tile_dir):
        """Union of all val tiles should cover all tiles."""
        lidar_dir = str(tile_dir / "lidar")
        folds = create_folds(lidar_dir, n_folds=5)

        all_val = set()
        for fold in folds:
            all_val.update(fold.val_tiles)

        all_tiles = set(folds[0].train_tiles + folds[0].val_tiles)
        assert all_val == all_tiles

    def test_dataset_with_empty_tile_paths(self, tile_dir):
        """Dataset with empty tile_paths should produce empty dataset."""
        mask_dir = str(tile_dir / "masks")
        lidar_dir = str(tile_dir / "lidar")

        ds = BinarySegmentationDataset(
            lidar_dir, mask_dir, "building",
            split="train", augment=False,
            tile_paths=[],
        )
        assert len(ds) == 0


class TestDiscoverFoldModels:
    def test_no_fold_models(self, tmp_path):
        """Should return empty list when no fold models exist."""
        paths = discover_fold_models(str(tmp_path), "building")
        assert paths == []

    def test_finds_fold_models(self, tmp_path):
        """Should find fold model files matching the pattern."""
        for i in range(5):
            path = tmp_path / f"mayascan_v2_building_deeplabv3plus_resnet101_fold{i}.pth"
            path.touch()

        paths = discover_fold_models(str(tmp_path), "building")
        assert len(paths) == 5

    def test_ignores_non_fold_models(self, tmp_path):
        """Should not include non-fold model files."""
        # Regular model (no fold suffix)
        (tmp_path / "mayascan_v2_building_deeplabv3plus_resnet101.pth").touch()
        # Fold models
        for i in range(3):
            (tmp_path / f"mayascan_v2_building_deeplabv3plus_resnet101_fold{i}.pth").touch()

        paths = discover_fold_models(str(tmp_path), "building")
        assert len(paths) == 3

    def test_class_filtering(self, tmp_path):
        """Should only find models for the specified class."""
        for cls in ["building", "platform"]:
            for i in range(3):
                (tmp_path / f"mayascan_v2_{cls}_deeplabv3plus_resnet101_fold{i}.pth").touch()

        building_paths = discover_fold_models(str(tmp_path), "building")
        platform_paths = discover_fold_models(str(tmp_path), "platform")
        assert len(building_paths) == 3
        assert len(platform_paths) == 3


class TestFoldSplitReproducibility:
    def test_same_seed_same_splits(self, tile_dir):
        """Same seed should produce identical splits."""
        lidar_dir = str(tile_dir / "lidar")
        folds1 = create_folds(lidar_dir, n_folds=5, seed=42)
        folds2 = create_folds(lidar_dir, n_folds=5, seed=42)

        for f1, f2 in zip(folds1, folds2):
            assert f1.train_tiles == f2.train_tiles
            assert f1.val_tiles == f2.val_tiles

    def test_different_seed_different_splits(self, tile_dir):
        """Different seed should produce different splits."""
        lidar_dir = str(tile_dir / "lidar")
        folds1 = create_folds(lidar_dir, n_folds=5, seed=42)
        folds2 = create_folds(lidar_dir, n_folds=5, seed=123)

        # At least one fold should differ
        any_different = False
        for f1, f2 in zip(folds1, folds2):
            if f1.val_tiles != f2.val_tiles:
                any_different = True
                break
        assert any_different


class TestFoldSummary:
    def test_summary_format(self, tile_dir):
        """Summary should contain fold counts and tile numbers."""
        lidar_dir = str(tile_dir / "lidar")
        folds = create_folds(lidar_dir, n_folds=5)
        summary = fold_summary(folds)

        assert "5 folds" in summary
        assert "Total tiles: 20" in summary
        assert "Fold 0:" in summary
        assert "Fold 4:" in summary

    def test_3_fold(self, tile_dir):
        """Should work with different fold counts."""
        lidar_dir = str(tile_dir / "lidar")
        folds = create_folds(lidar_dir, n_folds=3)
        summary = fold_summary(folds)

        assert "3 folds" in summary
        assert "Fold 2:" in summary


class TestEnsembleDetection:
    def test_run_detection_v2_ensemble_exists(self):
        """Ensemble detection function should be importable."""
        det_mod = importlib.import_module("mayascan.detect")
        assert hasattr(det_mod, "run_detection_v2_ensemble")

    def test_public_api_has_ensemble(self):
        """Public API should expose detect_v2_ensemble."""
        import mayascan
        assert hasattr(mayascan, "detect_v2_ensemble")
        assert callable(mayascan.detect_v2_ensemble)

    def test_public_api_has_fold_functions(self):
        """Public API should expose fold training functions."""
        import mayascan
        assert hasattr(mayascan, "create_folds")
        assert hasattr(mayascan, "train_fold")
        assert hasattr(mayascan, "train_kfold")
        assert hasattr(mayascan, "train_kfold_all")
        assert hasattr(mayascan, "discover_fold_models")


class TestAMPTraining:
    def test_train_class_accepts_use_amp(self):
        """train_class should accept use_amp parameter."""
        import inspect
        from mayascan.train import train_class
        sig = inspect.signature(train_class)
        assert "use_amp" in sig.parameters

    def test_build_model_accepts_various_encoders(self):
        """_build_model should support multiple architectures."""
        from mayascan.train import _build_model
        # Just check it doesn't error for valid arch names
        for arch in ["deeplabv3plus", "unetplusplus", "unet"]:
            model = _build_model(arch=arch, encoder="resnet18", classes=1)
            assert model is not None
