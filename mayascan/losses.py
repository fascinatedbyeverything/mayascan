"""Loss functions for archaeological segmentation training.

Implements competition-winning loss combinations from ECML PKDD 2021 Maya
Challenge. The Focal + Dice combo handles extreme class imbalance (aguadas
represent only 0.3% of pixels).
"""

from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance.

    Down-weights well-classified examples so the model focuses on hard negatives.
    Essential for rare classes like aguadas.

    Parameters
    ----------
    alpha : float
        Weighting factor for the rare class.
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="none"
        )
        p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss — directly optimizes an IoU-like metric.

    Parameters
    ----------
    smooth : float
        Smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_f.sum() + self.smooth
        )


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss (competition-winning combination).

    Parameters
    ----------
    focal_weight : float
        Weight for the Focal Loss component.
    dice_weight : float
        Weight for the Dice Loss component.
    alpha : float
        Focal Loss alpha parameter.
    gamma : float
        Focal Loss gamma parameter.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)
