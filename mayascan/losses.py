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


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszLoss(nn.Module):
    """Lovász-Hinge loss for binary segmentation.

    Directly optimizes the IoU metric via its tight convex surrogate.
    Often outperforms Dice loss for segmentation tasks.

    Based on: Berman, Triki, Blaschko (2018) "The Lovász-Softmax loss:
    A tractable surrogate for the optimization of the intersection-over-union
    measure in neural networks" (CVPR 2018).
    """

    def _lovasz_single(self, logits_flat: torch.Tensor, targets_flat: torch.Tensor) -> torch.Tensor:
        """Compute Lovász loss for a single flattened sample."""
        signs = 2.0 * targets_flat - 1.0
        errors = 1.0 - logits_flat * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = targets_flat[perm]
        grad = _lovasz_grad(gt_sorted)
        return torch.dot(torch.relu(errors_sorted), grad)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample Lovász, averaged across batch
        if logits.dim() > 1:
            batch_size = logits.shape[0]
            losses = []
            for i in range(batch_size):
                losses.append(self._lovasz_single(
                    logits[i].reshape(-1), targets[i].reshape(-1).float()
                ))
            return torch.stack(losses).mean()
        return self._lovasz_single(logits.reshape(-1), targets.reshape(-1).float())


class FocalLovaszLoss(nn.Module):
    """Combined Focal + Lovász loss.

    Focal handles pixel-level class imbalance while Lovász directly
    optimizes IoU. This combination can outperform Focal + Dice.

    Parameters
    ----------
    focal_weight : float
        Weight for the Focal Loss component.
    lovasz_weight : float
        Weight for the Lovász Loss component.
    alpha : float
        Focal Loss alpha parameter.
    gamma : float
        Focal Loss gamma parameter.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.lovasz = LovaszLoss()
        self.fw = focal_weight
        self.lw = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.lw * self.lovasz(logits, targets)
