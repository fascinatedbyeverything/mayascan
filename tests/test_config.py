"""Tests for mayascan.config — centralized configuration constants."""

from mayascan.config import (
    BATCH_SIZE,
    CLASS_COLORS,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEFAULT_RESOLUTION,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    HF_REPO_ID,
    LEARNING_RATE,
    MIN_BLOB_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
    V2_ARCH,
    V2_CLASSES,
    V2_ENCODER,
)


class TestModelDefaults:
    def test_tile_size(self):
        assert isinstance(TILE_SIZE, int)
        assert TILE_SIZE > 0

    def test_tile_overlap(self):
        assert 0 < TILE_OVERLAP < 1

    def test_confidence_threshold(self):
        assert 0 < CONFIDENCE_THRESHOLD <= 1.0

    def test_min_blob_size(self):
        assert isinstance(MIN_BLOB_SIZE, int)
        assert MIN_BLOB_SIZE > 0

    def test_default_resolution(self):
        assert DEFAULT_RESOLUTION > 0


class TestClassDefinitions:
    def test_class_names_has_background(self):
        assert 0 in CLASS_NAMES
        assert CLASS_NAMES[0] == "background"

    def test_class_names_has_3_feature_classes(self):
        assert len(CLASS_NAMES) == 4
        assert 1 in CLASS_NAMES
        assert 2 in CLASS_NAMES
        assert 3 in CLASS_NAMES

    def test_v2_classes_excludes_background(self):
        assert 0 not in V2_CLASSES
        assert len(V2_CLASSES) == 3

    def test_class_colors_match_class_names(self):
        for cls_id in CLASS_NAMES:
            assert cls_id in CLASS_COLORS

    def test_class_colors_are_rgba(self):
        for cls_id, color in CLASS_COLORS.items():
            assert len(color) == 4
            assert all(0 <= c <= 255 for c in color)


class TestArchitecture:
    def test_v2_arch(self):
        assert V2_ARCH == "deeplabv3plus"

    def test_v2_encoder(self):
        assert V2_ENCODER == "resnet101"


class TestTrainingDefaults:
    def test_learning_rate(self):
        assert 0 < LEARNING_RATE < 1

    def test_batch_size(self):
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0

    def test_epochs(self):
        assert isinstance(EPOCHS, int)
        assert EPOCHS > 0

    def test_focal_params(self):
        assert FOCAL_GAMMA > 0
        assert 0 < FOCAL_ALPHA < 1


class TestHuggingFace:
    def test_hf_repo_id(self):
        assert "/" in HF_REPO_ID
        assert len(HF_REPO_ID) > 3
