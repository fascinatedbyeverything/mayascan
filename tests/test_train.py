"""Tests for mayascan.train — training module."""

import numpy as np
import pytest
import torch

from mayascan.train import (
    _build_model,
    compute_binary_iou,
    postprocess_mask,
    predict_with_tta,
)


class TestBuildModel:
    def test_deeplabv3plus(self):
        model = _build_model("deeplabv3plus", "resnet18", in_channels=3, classes=1)
        model.eval()  # BatchNorm in ASPP needs eval mode for small inputs
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_unet(self):
        model = _build_model("unet", "resnet18", in_channels=3, classes=1)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_unetplusplus(self):
        model = _build_model("unetplusplus", "resnet18", in_channels=3, classes=1)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            _build_model("invalid_arch", "resnet18")


class TestComputeBinaryIou:
    def test_perfect(self):
        mask = np.array([[1, 1], [0, 0]])
        assert compute_binary_iou(mask, mask) == 1.0

    def test_no_overlap(self):
        pred = np.array([[1, 0], [0, 0]])
        gt = np.array([[0, 0], [0, 1]])
        assert compute_binary_iou(pred, gt) == 0.0

    def test_partial(self):
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[1, 0], [0, 0]])
        # intersection=1, union=2
        assert compute_binary_iou(pred, gt) == 0.5

    def test_both_empty(self):
        empty = np.zeros((4, 4))
        # Both empty = true negative, excluded from IoU averaging
        assert compute_binary_iou(empty, empty) is None

    def test_only_pred_positive(self):
        pred = np.ones((4, 4))
        gt = np.zeros((4, 4))
        # All false positives → IoU = 0
        assert compute_binary_iou(pred, gt) == 0.0


class TestPostprocessMask:
    def test_removes_small_blobs(self):
        prob = np.zeros((50, 50))
        prob[5:7, 5:7] = 0.9  # 4 pixels — below min_blob_size=50
        prob[20:40, 20:40] = 0.9  # 400 pixels — above threshold
        result = postprocess_mask(prob, threshold=0.5, min_blob_size=50)
        assert result[6, 6] == 0  # small blob removed
        assert result[30, 30] == 1  # large blob kept

    def test_threshold(self):
        prob = np.full((10, 10), 0.3)
        result = postprocess_mask(prob, threshold=0.5)
        assert result.sum() == 0

    def test_all_above_threshold(self):
        prob = np.full((10, 10), 0.9)
        result = postprocess_mask(prob, threshold=0.5, min_blob_size=1)
        assert result.sum() == 100


class TestPredictWithTTA:
    def test_output_shape(self):
        model = _build_model("unet", "resnet18", in_channels=3, classes=1)
        model.eval()
        images = torch.randn(2, 3, 64, 64)
        result = predict_with_tta(model, images, "cpu")
        assert result.shape == (2, 1, 64, 64)

    def test_output_is_probability(self):
        model = _build_model("unet", "resnet18", in_channels=3, classes=1)
        model.eval()
        images = torch.randn(1, 3, 64, 64)
        result = predict_with_tta(model, images, "cpu")
        assert result.min() >= 0.0
        assert result.max() <= 1.0
