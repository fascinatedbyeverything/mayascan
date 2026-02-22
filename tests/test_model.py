"""Tests for mayascan.models.unet — U-Net wrapper."""

import torch
import pytest

from mayascan.models.unet import MayaScanUNet


class TestMayaScanUNet:
    """Tests for the MayaScanUNet wrapper."""

    @pytest.fixture()
    def model(self):
        """Return a lightweight U-Net (no pretrained weights) for testing."""
        return MayaScanUNet(num_classes=4, encoder="resnet34", pretrained=False)

    def test_model_output_shape(self, model):
        """(2, 3, 480, 480) input produces (2, 4, 480, 480) output."""
        x = torch.randn(2, 3, 480, 480)
        out = model(x)
        assert out.shape == (2, 4, 480, 480)

    def test_model_output_logits(self, model):
        """forward() returns raw logits, not softmaxed probabilities."""
        x = torch.randn(2, 3, 480, 480)
        out = model(x)
        # Raw logits can be negative; a softmax output cannot.
        assert out.detach().min().item() < 0.0, "Expected raw logits with negative values"

    def test_model_predict(self, model):
        """predict() returns (classes, confidence) with correct shapes/types."""
        x = torch.randn(1, 3, 480, 480)
        classes, confidence = model.predict(x)

        assert classes.shape == (480, 480)
        assert classes.dtype == torch.long

        assert confidence.shape == (480, 480)
        assert confidence.dtype == torch.float32
        assert confidence.min().item() >= 0.0
        assert confidence.max().item() <= 1.0
