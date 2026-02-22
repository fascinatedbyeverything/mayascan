"""Tests for mayascan.detect — tiled inference pipeline."""

import os
import tempfile

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch

from mayascan.detect import (
    CLASS_NAMES,
    V2_CLASSES,
    DetectionResult,
    _predict_tile_with_tta,
    discover_v2_models,
    run_detection,
    run_detection_v2,
)


class TestDetectionResult:
    """Tests for the DetectionResult dataclass."""

    def test_detection_result_fields(self):
        """DetectionResult has classes, confidence, class_names attributes."""
        classes = np.zeros((10, 10), dtype=np.int64)
        confidence = np.ones((10, 10), dtype=np.float32)
        result = DetectionResult(
            classes=classes,
            confidence=confidence,
            class_names=dict(CLASS_NAMES),
        )

        assert result.classes is classes
        assert result.confidence is confidence
        assert isinstance(result.class_names, dict)
        assert result.class_names == {
            0: "background",
            1: "building",
            2: "platform",
            3: "aguada",
        }


class TestRunDetection:
    """Tests for run_detection end-to-end pipeline."""

    def test_run_detection_output_shape(self):
        """(3, 480, 480) input produces DetectionResult with (480, 480) maps."""
        rng = np.random.default_rng(42)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection(viz, model_path=None, tile_size=480, overlap=0.0, device="cpu")

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert result.classes.min() >= 0
        assert result.classes.max() < 4
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_run_detection_large_image(self):
        """(3, 960, 960) input works -- gets tiled internally."""
        rng = np.random.default_rng(7)
        viz = rng.random((3, 960, 960)).astype(np.float32)

        result = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.5, device="cpu"
        )

        assert result.classes.shape == (960, 960)
        assert result.confidence.shape == (960, 960)
        assert result.classes.min() >= 0
        assert result.classes.max() < 4

    def test_run_detection_confidence_threshold(self):
        """Higher threshold produces more background pixels than lower threshold."""
        rng = np.random.default_rng(123)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        # Fix random seed so both calls create identical model weights
        torch.manual_seed(0)
        result_low = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.0,
            confidence_threshold=0.0, device="cpu",
        )
        torch.manual_seed(0)
        result_high = run_detection(
            viz, model_path=None, tile_size=480, overlap=0.0,
            confidence_threshold=0.99, device="cpu",
        )

        bg_low = (result_low.classes == 0).sum()
        bg_high = (result_high.classes == 0).sum()

        # Higher threshold should produce at least as much background
        assert bg_high >= bg_low, (
            f"Higher threshold should yield more background: "
            f"thresh=0.0 bg={bg_low}, thresh=0.99 bg={bg_high}"
        )


def _create_v2_model_dir(
    tmpdir: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
    classes: dict[int, str] | None = None,
    use_checkpoint_dict: bool = False,
) -> str:
    """Helper: save v2 model .pth files with valid state dicts to a temp dir."""
    if classes is None:
        classes = V2_CLASSES
    arch_map = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
    }
    model_cls = arch_map[arch]
    for cls_id, cls_name in classes.items():
        model = model_cls(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        filename = f"mayascan_v2_{cls_name}_{arch}_{encoder}.pth"
        path = os.path.join(tmpdir, filename)
        if use_checkpoint_dict:
            torch.save({"state_dict": model.state_dict()}, path)
        else:
            torch.save(model.state_dict(), path)
    return tmpdir


class TestDiscoverV2Models:
    """Tests for discover_v2_models file discovery."""

    def test_discovers_all_three_classes(self, tmp_path):
        """All three class models are found when all files exist."""
        model_dir = _create_v2_model_dir(str(tmp_path))
        found = discover_v2_models(model_dir)

        assert len(found) == 3
        for cls_id in V2_CLASSES:
            assert cls_id in found
            assert os.path.isfile(found[cls_id])

    def test_discovers_partial_classes(self, tmp_path):
        """Only the classes present on disk are returned."""
        # Create models for only building and aguada (skip platform)
        subset = {1: "building", 3: "aguada"}
        model_dir = _create_v2_model_dir(str(tmp_path), classes=subset)
        found = discover_v2_models(model_dir)

        assert len(found) == 2
        assert 1 in found
        assert 3 in found
        assert 2 not in found

    def test_empty_directory_returns_empty(self, tmp_path):
        """An empty directory yields an empty dict."""
        found = discover_v2_models(str(tmp_path))
        assert found == {}

    def test_wrong_arch_not_found(self, tmp_path):
        """Models saved with one arch are not found when querying another."""
        _create_v2_model_dir(str(tmp_path), arch="deeplabv3plus")
        found = discover_v2_models(str(tmp_path), arch="unetplusplus")
        assert found == {}

    def test_wrong_encoder_not_found(self, tmp_path):
        """Models saved with one encoder are not found when querying another."""
        _create_v2_model_dir(str(tmp_path), arch="deeplabv3plus", encoder="resnet101")
        found = discover_v2_models(str(tmp_path), encoder="resnet50")
        assert found == {}

    def test_correct_filename_format(self, tmp_path):
        """Discovered paths use the expected naming convention."""
        model_dir = _create_v2_model_dir(str(tmp_path))
        found = discover_v2_models(model_dir)

        for cls_id, path in found.items():
            basename = os.path.basename(path)
            cls_name = V2_CLASSES[cls_id]
            expected = f"mayascan_v2_{cls_name}_deeplabv3plus_resnet101.pth"
            assert basename == expected


class TestPredictTileWithTTA:
    """Tests for _predict_tile_with_tta augmentation pipeline."""

    @pytest.fixture
    def binary_model(self):
        """A small binary smp model on CPU for testing."""
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.eval()
        return model

    @pytest.fixture
    def multiclass_model(self):
        """A small multi-class smp model on CPU for testing."""
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        model.eval()
        return model

    def test_no_tta_binary_output_shape(self, binary_model):
        """Without TTA, binary model produces (1, H, W) output."""
        tile = np.random.default_rng(0).random((3, 480, 480)).astype(np.float32)
        device = torch.device("cpu")

        with torch.no_grad():
            result = _predict_tile_with_tta(
                binary_model, tile, device, use_tta=False, binary=True
            )

        assert result.shape == (1, 480, 480)
        # Sigmoid output should be in [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_tta_binary_same_shape_as_no_tta(self, binary_model):
        """TTA produces the same output shape as non-TTA for binary models."""
        tile = np.random.default_rng(1).random((3, 480, 480)).astype(np.float32)
        device = torch.device("cpu")

        with torch.no_grad():
            result_no_tta = _predict_tile_with_tta(
                binary_model, tile, device, use_tta=False, binary=True
            )
            result_tta = _predict_tile_with_tta(
                binary_model, tile, device, use_tta=True, binary=True
            )

        assert result_tta.shape == result_no_tta.shape
        assert result_tta.shape == (1, 480, 480)

    def test_tta_multiclass_same_shape_as_no_tta(self, multiclass_model):
        """TTA produces the same output shape as non-TTA for multi-class models."""
        tile = np.random.default_rng(2).random((3, 480, 480)).astype(np.float32)
        device = torch.device("cpu")

        with torch.no_grad():
            result_no_tta = _predict_tile_with_tta(
                multiclass_model, tile, device, use_tta=False, binary=False
            )
            result_tta = _predict_tile_with_tta(
                multiclass_model, tile, device, use_tta=True, binary=False
            )

        assert result_tta.shape == result_no_tta.shape
        assert result_tta.shape == (4, 480, 480)

    def test_tta_output_is_valid_probability(self, binary_model):
        """TTA-averaged output stays in valid probability range [0, 1]."""
        tile = np.random.default_rng(3).random((3, 480, 480)).astype(np.float32)
        device = torch.device("cpu")

        with torch.no_grad():
            result = _predict_tile_with_tta(
                binary_model, tile, device, use_tta=True, binary=True
            )

        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestRunDetectionV2:
    """Tests for run_detection_v2 end-to-end pipeline."""

    def test_output_shape_single_tile(self, tmp_path):
        """(3, 480, 480) input produces DetectionResult with (480, 480) maps."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(42)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)

    def test_valid_class_indices(self, tmp_path):
        """All predicted class indices are in [0, 3]."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(10)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert result.classes.min() >= 0
        assert result.classes.max() <= 3

    def test_valid_confidence_range(self, tmp_path):
        """All confidence values are in [0, 1]."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(11)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_class_names_present(self, tmp_path):
        """Result contains the standard class_names mapping."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(12)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert result.class_names == CLASS_NAMES

    def test_checkpoint_dict_format(self, tmp_path):
        """Models saved as {state_dict: ...} dicts load correctly."""
        _create_v2_model_dir(str(tmp_path), use_checkpoint_dict=True)
        rng = np.random.default_rng(13)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)

    def test_missing_model_dir_raises(self, tmp_path, monkeypatch):
        """FileNotFoundError raised when no models found and auto-download fails."""
        import sys
        detect_mod = sys.modules["mayascan.detect"]

        rng = np.random.default_rng(14)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        # Disable auto-download so the error is raised
        monkeypatch.setattr(detect_mod, "_auto_download_models", lambda *a, **kw: {})

        with pytest.raises(FileNotFoundError, match="No v2 models found"):
            run_detection_v2(
                viz,
                model_dir=str(tmp_path),
                device="cpu",
            )

    def test_partial_models_still_works(self, tmp_path):
        """Pipeline works with only a subset of class models present."""
        subset = {1: "building"}
        _create_v2_model_dir(str(tmp_path), classes=subset)
        rng = np.random.default_rng(15)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=False,
            device="cpu",
        )

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        # Only class 0 (background) and 1 (building) should appear
        unique_classes = set(np.unique(result.classes))
        assert unique_classes <= {0, 1}

    def test_with_tta_produces_valid_result(self, tmp_path):
        """TTA mode produces a valid DetectionResult with correct shapes."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(16)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        result = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            use_tta=True,
            device="cpu",
        )

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert result.classes.min() >= 0
        assert result.classes.max() <= 3
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0

    def test_high_threshold_yields_more_background(self, tmp_path):
        """Higher confidence threshold produces more background pixels."""
        _create_v2_model_dir(str(tmp_path))
        rng = np.random.default_rng(17)
        viz = rng.random((3, 480, 480)).astype(np.float32)

        torch.manual_seed(0)
        result_low = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            confidence_threshold=0.0,
            use_tta=False,
            device="cpu",
        )
        torch.manual_seed(0)
        result_high = run_detection_v2(
            viz,
            model_dir=str(tmp_path),
            tile_size=480,
            overlap=0.0,
            confidence_threshold=0.99,
            use_tta=False,
            device="cpu",
        )

        bg_low = (result_low.classes == 0).sum()
        bg_high = (result_high.classes == 0).sum()

        assert bg_high >= bg_low, (
            f"Higher threshold should yield more background: "
            f"thresh=0.0 bg={bg_low}, thresh=0.99 bg={bg_high}"
        )
