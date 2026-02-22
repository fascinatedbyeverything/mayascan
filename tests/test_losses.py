"""Tests for mayascan.losses — training loss functions."""

import numpy as np
import pytest
import torch

from mayascan.losses import DiceLoss, FocalDiceLoss, FocalLoss, LovaszLoss, FocalLovaszLoss


class TestFocalLoss:
    def test_perfect_prediction(self):
        loss_fn = FocalLoss()
        # Confident correct prediction -> low loss
        logits = torch.tensor([[[[5.0]]]])  # sigmoid -> ~0.993
        targets = torch.tensor([[[[1.0]]]])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_wrong_prediction(self):
        loss_fn = FocalLoss()
        # Confident wrong prediction -> high loss
        logits = torch.tensor([[[[5.0]]]])
        targets = torch.tensor([[[[0.0]]]])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.1

    def test_returns_scalar(self):
        loss_fn = FocalLoss()
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gamma_effect(self):
        logits = torch.tensor([[[[0.0]]]])  # uncertain prediction
        targets = torch.tensor([[[[1.0]]]])
        low_gamma = FocalLoss(gamma=0.5)(logits, targets).item()
        high_gamma = FocalLoss(gamma=4.0)(logits, targets).item()
        # Higher gamma should down-weight easy examples more
        assert high_gamma != low_gamma


class TestDiceLoss:
    def test_perfect_overlap(self):
        loss_fn = DiceLoss()
        logits = torch.tensor([[[[10.0, 10.0], [10.0, 10.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.02

    def test_no_overlap(self):
        loss_fn = DiceLoss()
        logits = torch.tensor([[[[-10.0, -10.0], [-10.0, -10.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.7

    def test_empty_target_empty_pred(self):
        loss_fn = DiceLoss()
        logits = torch.tensor([[[[-10.0]]]])
        targets = torch.tensor([[[[0.0]]]])
        loss = loss_fn(logits, targets)
        # Both nearly zero -> Dice ~ 1 - smooth/(smooth) ~ 0
        assert loss.item() < 0.1


class TestFocalDiceLoss:
    def test_combines_both(self):
        loss_fn = FocalDiceLoss()
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_weights(self):
        logits = torch.randn(1, 1, 8, 8)
        targets = torch.randint(0, 2, (1, 1, 8, 8)).float()
        focal_only = FocalDiceLoss(focal_weight=1.0, dice_weight=0.0)(logits, targets)
        dice_only = FocalDiceLoss(focal_weight=0.0, dice_weight=1.0)(logits, targets)
        combined = FocalDiceLoss(focal_weight=1.0, dice_weight=1.0)(logits, targets)
        np.testing.assert_allclose(
            combined.item(), focal_only.item() + dice_only.item(), atol=1e-5
        )

    def test_backward(self):
        loss_fn = FocalDiceLoss()
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestLovaszLoss:
    def test_perfect_prediction(self):
        loss_fn = LovaszLoss()
        logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_wrong_prediction(self):
        loss_fn = LovaszLoss()
        logits = torch.tensor([-5.0, -5.0, 5.0, 5.0])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.5

    def test_returns_scalar(self):
        loss_fn = LovaszLoss()
        logits = torch.randn(64)
        targets = torch.randint(0, 2, (64,)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_backward(self):
        loss_fn = LovaszLoss()
        logits = torch.randn(64, requires_grad=True)
        targets = torch.randint(0, 2, (64,)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestFocalLovaszLoss:
    def test_combines_both(self):
        loss_fn = FocalLovaszLoss()
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_backward(self):
        loss_fn = FocalLovaszLoss()
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
