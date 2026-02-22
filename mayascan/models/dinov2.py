"""DINOv2 foundation model for archaeological LiDAR segmentation.

Provides a frozen DINOv2 encoder with optional LoRA fine-tuning,
plus a UPerNet decoder head for dense binary prediction. Leverages
rich self-supervised visual representations to outperform CNN encoders
on terrain feature detection.

Usage::

    model = DINOv2Segmenter(encoder_name="dinov2-large", use_lora=True)
    logits = model(images)  # (B, 3, 518, 518) -> (B, 1, 518, 518)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DINOV2_MODELS: dict[str, str] = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
}

DINOV2_DIMS: dict[str, int] = {
    "dinov2-small": 384,
    "dinov2-base": 768,
    "dinov2-large": 1024,
    "dinov2-giant": 1536,
}

#: Intermediate layers to extract multi-scale features from
#: (4 evenly-spaced layers for UPerNet decoder).
DINOV2_FEATURE_LAYERS: dict[str, list[int]] = {
    "dinov2-small": [2, 5, 8, 11],
    "dinov2-base": [2, 5, 8, 11],
    "dinov2-large": [4, 11, 17, 23],
    "dinov2-giant": [9, 19, 29, 39],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class DINOv2Encoder(nn.Module):
    """DINOv2 backbone wrapper with multi-scale feature extraction.

    Loads a pretrained DINOv2 model from HuggingFace, extracts features
    from 4 intermediate transformer blocks, reshapes token sequences
    to spatial feature maps.

    Parameters
    ----------
    encoder_name : str
        One of ``"dinov2-small"``, ``"dinov2-base"``, ``"dinov2-large"``,
        ``"dinov2-giant"``.
    frozen : bool
        If True, freeze all base encoder parameters.
    use_lora : bool
        If True, add LoRA adapters to attention query/value projections.
    lora_rank : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling factor.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2-large",
        frozen: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.embed_dim = DINOV2_DIMS[encoder_name]
        self.feature_layers = DINOV2_FEATURE_LAYERS[encoder_name]
        self.patch_size = 14

        from transformers import Dinov2Model

        hub_name = DINOV2_MODELS[encoder_name]
        self.backbone = Dinov2Model.from_pretrained(hub_name)

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

    def _apply_lora(self, rank: int, alpha: int) -> None:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, config)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Parameters
        ----------
        x : (B, 3, H, W), values in [0, 1]. H, W must be divisible by 14.

        Returns
        -------
        list of 4 tensors, each (B, embed_dim, h, w) where h = H//14.
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Input {H}x{W} must be divisible by patch_size={self.patch_size}. "
            f"Use 448 or 518."
        )

        x = (x - self.pixel_mean) / self.pixel_std

        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states[0] = embeddings, hidden_states[i+1] = layer i output
        hidden_states = outputs.hidden_states
        h = H // self.patch_size
        w = W // self.patch_size

        features = []
        for layer_idx in self.feature_layers:
            tokens = hidden_states[layer_idx + 1]  # (B, 1+h*w, C)
            patch_tokens = tokens[:, 1:, :]  # remove CLS
            feat = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, self.embed_dim, h, w)
            features.append(feat)

        return features


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _MPSSafeAdaptivePool(nn.Module):
    """AdaptiveAvgPool2d replacement that works on MPS.

    MPS requires input sizes divisible by output sizes for adaptive pooling.
    This uses F.interpolate(mode="area") instead, which is equivalent and
    MPS-compatible for any input/output size combination.
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_size == 1:
            # Global average pooling — works on any backend
            return x.mean(dim=(-2, -1), keepdim=True)
        return F.interpolate(
            x, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False,
        )


class UPerNetHead(nn.Module):
    """UPerNet decoder: PSP on deepest features + FPN lateral fusion.

    Parameters
    ----------
    in_channels : int
        Channels in each input feature map (same for all DINOv2 scales).
    hidden_dim : int
        Internal decoder width.
    classes : int
        Number of output classes (1 for binary).
    pool_scales : tuple
        PSP pooling scales.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 256,
        classes: int = 1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()

        # PSP module on deepest features (MPS-safe pooling)
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    _MPSSafeAdaptivePool(scale),
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + hidden_dim * len(pool_scales),
                hidden_dim,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Lateral convolutions (one per scale)
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        )

        # FPN smoothing convolutions
        self.fpn_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        )

        # Bottleneck: merge all 4 scales
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(hidden_dim, classes, 1)

    def forward(
        self, features: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Decode multi-scale features to segmentation logits.

        Parameters
        ----------
        features : list of 4 tensors, each (B, C, h, w).
        output_size : (H, W) target spatial resolution.

        Returns
        -------
        (B, classes, H, W) logits.
        """
        # PSP on deepest
        psp_input = features[-1]
        psp_outs = [psp_input]
        for mod in self.psp_modules:
            pooled = mod(psp_input)
            up = F.interpolate(
                pooled, size=psp_input.shape[2:], mode="bilinear", align_corners=False
            )
            psp_outs.append(up)
        psp_out = self.psp_bottleneck(torch.cat(psp_outs, dim=1))

        # Lateral connections
        laterals = [conv(feat) for feat, conv in zip(features, self.lateral_convs)]
        laterals[-1] = psp_out

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # FPN smoothing
        fpn_outs = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]

        # Upsample all to finest resolution
        target = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target, mode="bilinear", align_corners=False
            )

        fused = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        logits = self.cls_head(fused)

        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------


class DINOv2Segmenter(nn.Module):
    """Complete DINOv2 segmentation model (encoder + UPerNet decoder).

    Drop-in replacement for smp models in the MayaScan pipeline.
    Input: (B, 3, H, W) in [0, 1].  Output: (B, classes, H, W) logits.

    Parameters
    ----------
    encoder_name : str
        DINOv2 variant.
    use_lora : bool
        Enable LoRA adapters on attention layers.
    lora_rank, lora_alpha : int
        LoRA hyperparameters.
    frozen_encoder : bool
        Freeze base encoder weights.
    hidden_dim : int
        Decoder hidden dimension.
    classes : int
        Output channels (1 for binary segmentation).
    """

    def __init__(
        self,
        encoder_name: str = "dinov2-large",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        frozen_encoder: bool = True,
        hidden_dim: int = 256,
        classes: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        self.encoder = DINOv2Encoder(
            encoder_name=encoder_name,
            frozen=frozen_encoder,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        self.decoder = UPerNetHead(
            in_channels=DINOV2_DIMS[encoder_name],
            hidden_dim=hidden_dim,
            classes=classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        features = self.encoder(x)
        return self.decoder(features, output_size=(H, W))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only trainable parameters (decoder + LoRA adapters)."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
