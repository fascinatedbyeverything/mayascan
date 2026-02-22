"""U-Net wrapper for archaeological feature segmentation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MayaScanUNet(nn.Module):
    """U-Net model for 3-channel input, multi-class archaeological segmentation.

    Wraps segmentation-models-pytorch's Unet with sensible defaults
    for LiDAR-derived feature detection (e.g. mounds, platforms,
    causeways, aguadas).

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 4).
    encoder : str
        Backbone encoder name (default ``"resnet34"``).
    pretrained : bool
        Whether to load ImageNet-pretrained encoder weights.
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder: str = "resnet34",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes, H, W)``."""
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return per-pixel class and confidence.

        Switches the model to eval mode, runs a forward pass, and
        converts the logits to class predictions via softmax + argmax.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.  Only the first
            sample in the batch is returned.

        Returns
        -------
        classes : torch.Tensor
            ``(H, W)`` long tensor of predicted class indices.
        confidence : torch.Tensor
            ``(H, W)`` float tensor of softmax confidence for the
            predicted class, values in ``[0, 1]``.
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        confidence, classes = probs[0].max(dim=0)  # both (H, W)
        return classes.long(), confidence
