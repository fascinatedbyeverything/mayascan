"""Tests for mayascan.models.dinov2 — DINOv2 foundation model components.

All tests use mocked backbones so no model download is required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mayascan.models.dinov2 import (
    DINOV2_DIMS,
    DINOV2_FEATURE_LAYERS,
    DINOV2_MODELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv2Encoder,
    DINOv2Segmenter,
    UPerNetHead,
)


# ---------------------------------------------------------------------------
# UPerNet decoder tests
# ---------------------------------------------------------------------------


class TestUPerNetHead:
    """Tests for the UPerNet decoder head."""

    def test_output_shape_518(self):
        """Decoder produces correct output shape at 518x518."""
        head = UPerNetHead(in_channels=1024, hidden_dim=256, classes=1)
        # 4 feature maps at 37x37 (518 // 14)
        features = [torch.randn(2, 1024, 37, 37) for _ in range(4)]
        out = head(features, output_size=(518, 518))
        assert out.shape == (2, 1, 518, 518)

    def test_output_shape_448(self):
        """Decoder produces correct output shape at 448x448."""
        head = UPerNetHead(in_channels=768, hidden_dim=128, classes=1)
        head.eval()  # eval mode avoids BatchNorm batch_size=1 issue
        features = [torch.randn(1, 768, 32, 32) for _ in range(4)]
        out = head(features, output_size=(448, 448))
        assert out.shape == (1, 1, 448, 448)

    def test_multiclass(self):
        """Decoder works with multiple output classes."""
        head = UPerNetHead(in_channels=384, hidden_dim=64, classes=3)
        head.eval()
        features = [torch.randn(1, 384, 16, 16) for _ in range(4)]
        out = head(features, output_size=(224, 224))
        assert out.shape == (1, 3, 224, 224)

    def test_gradient_flow(self):
        """Gradients flow through the decoder."""
        head = UPerNetHead(in_channels=256, hidden_dim=64, classes=1)
        features = [torch.randn(2, 256, 8, 8, requires_grad=True) for _ in range(4)]
        out = head(features, output_size=(112, 112))
        loss = out.sum()
        loss.backward()
        for f in features:
            assert f.grad is not None


# ---------------------------------------------------------------------------
# DINOv2Segmenter tests (mocked backbone)
# ---------------------------------------------------------------------------


def _make_mock_backbone(embed_dim: int, num_layers: int):
    """Create a mock Dinov2Model that returns fake hidden states."""
    mock = MagicMock()

    def mock_forward(pixel_values, output_hidden_states=True, return_dict=True):
        B = pixel_values.shape[0]
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        h, w = H // 14, W // 14
        n_tokens = h * w + 1  # CLS + patches

        result = MagicMock()
        # hidden_states[0] = embeddings, then one per layer
        result.hidden_states = [
            torch.randn(B, n_tokens, embed_dim)
            for _ in range(num_layers + 1)
        ]
        return result

    mock.side_effect = mock_forward
    mock.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    return mock


class TestDINOv2SegmenterMocked:
    """Tests for the full segmenter with mocked HF backbone."""

    @pytest.fixture()
    def model(self):
        """Build a DINOv2Segmenter with mocked encoder backbone."""
        with patch("transformers.Dinov2Model") as MockModel:
            MockModel.from_pretrained.return_value = _make_mock_backbone(1024, 24)
            m = DINOv2Segmenter(
                encoder_name="dinov2-large",
                use_lora=False,
                frozen_encoder=False,
            )
        # Replace the backbone call with our mock
        m.encoder.backbone = _make_mock_backbone(1024, 24)
        m.eval()  # eval mode avoids BatchNorm batch_size=1 issue in PSP pooling
        return m

    def test_forward_shape(self, model):
        """Forward pass produces (B, 1, H, W) for 518x518 input."""
        x = torch.randn(1, 3, 518, 518)
        out = model(x)
        assert out.shape == (1, 1, 518, 518)

    def test_forward_batch(self, model):
        """Forward pass handles batch size > 1."""
        x = torch.randn(2, 3, 518, 518)
        out = model(x)
        assert out.shape == (2, 1, 518, 518)

    def test_forward_448(self, model):
        """Forward pass works at 448x448 (also divisible by 14)."""
        x = torch.randn(1, 3, 448, 448)
        out = model(x)
        assert out.shape == (1, 1, 448, 448)

    def test_input_not_divisible_by_14(self, model):
        """Input not divisible by 14 raises AssertionError."""
        x = torch.randn(1, 3, 480, 480)
        with pytest.raises(AssertionError, match="divisible by patch_size"):
            model(x)

    def test_trainable_parameters(self, model):
        """trainable_parameters returns a non-empty list."""
        params = model.trainable_parameters()
        assert isinstance(params, list)
        assert len(params) > 0

    def test_param_counts(self, model):
        """Param count methods return positive integers."""
        assert model.trainable_param_count() > 0
        assert model.total_param_count() > 0


# ---------------------------------------------------------------------------
# _build_model integration
# ---------------------------------------------------------------------------


class TestBuildModelDINOv2:
    """Test _build_model with dinov2 arch."""

    def test_unknown_encoder_raises(self):
        """Unknown DINOv2 encoder raises ValueError."""
        from mayascan.train import _build_model

        with pytest.raises(ValueError, match="Unknown DINOv2 encoder"):
            _build_model("dinov2", "resnet101")

    def test_valid_encoder_accepted(self):
        """Valid DINOv2 encoder name is accepted (model construction)."""
        from mayascan.train import _build_model

        with patch("transformers.Dinov2Model") as MockModel:
            MockModel.from_pretrained.return_value = _make_mock_backbone(1024, 24)
            model = _build_model("dinov2", "dinov2-large", use_lora=False)
            assert isinstance(model, DINOv2Segmenter)


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Test foundation model constants."""

    def test_foundation_archs_contains_dinov2(self):
        from mayascan.config import FOUNDATION_ARCHS
        assert "dinov2" in FOUNDATION_ARCHS

    def test_dinov2_tile_size_divisible_by_14(self):
        from mayascan.config import DINOV2_TILE_SIZE
        assert DINOV2_TILE_SIZE % 14 == 0

    def test_dinov2_models_dict(self):
        assert "dinov2-large" in DINOV2_MODELS
        assert "dinov2-small" in DINOV2_MODELS

    def test_dinov2_dims_dict(self):
        assert DINOV2_DIMS["dinov2-large"] == 1024
        assert DINOV2_DIMS["dinov2-small"] == 384

    def test_feature_layers_correct_count(self):
        for name, layers in DINOV2_FEATURE_LAYERS.items():
            assert len(layers) == 4, f"{name} should have 4 feature layers"

    def test_imagenet_normalization_values(self):
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < v < 1 for v in IMAGENET_MEAN)
        assert all(0 < v < 1 for v in IMAGENET_STD)
