"""Detection module: tiled inference pipeline for archaeological feature segmentation.

Tiles visualization rasters, runs U-Net inference on each tile, stitches
results back together, and applies a confidence threshold to produce a
final segmentation map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from mayascan.models.unet import MayaScanUNet
from mayascan.tile import slice_tiles, stitch_tiles

CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "building",
    2: "platform",
    3: "aguada",
}


@dataclass
class DetectionResult:
    """Result of running detection on a visualization raster.

    Attributes
    ----------
    classes : np.ndarray
        (H, W) integer array of predicted class indices.
    confidence : np.ndarray
        (H, W) float array of per-pixel confidence values in [0, 1].
    class_names : dict[int, str]
        Mapping from class index to human-readable name.
    """

    classes: np.ndarray
    confidence: np.ndarray
    class_names: dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))


def run_detection(
    visualization: np.ndarray,
    model_path: str | None = None,
    tile_size: int = 480,
    overlap: float = 0.5,
    confidence_threshold: float = 0.5,
    device: torch.device | str | None = None,
) -> DetectionResult:
    """Run tiled U-Net inference on a visualization raster.

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)`` where *C* is typically 3.
    model_path : str or None
        Path to a ``.pt`` / ``.pth`` file containing saved model weights.
        If *None*, the model is used with its initial (random/pretrained)
        weights -- useful for testing.
    tile_size : int
        Side length of each square tile fed to the model (default 480).
    overlap : float
        Fractional overlap between adjacent tiles, in ``[0, 1)``
        (default 0.5).
    confidence_threshold : float
        Pixels whose confidence is below this value *and* whose predicted
        class is not background (0) are reset to background.
    device : torch.device, str, or None
        Device for inference.  If *None*, CUDA is used when available,
        otherwise CPU.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    # --- device selection ---
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # --- model setup ---
    num_classes = len(CLASS_NAMES)
    model = MayaScanUNet(num_classes=num_classes, encoder="resnet34", pretrained=False)

    if model_path is not None:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    # --- tile the input ---
    C, H, W = visualization.shape
    tiles, origins = slice_tiles(visualization, tile_size=tile_size, overlap=overlap)

    # --- run inference per tile, collecting probability maps ---
    prob_tiles: list[np.ndarray] = []

    with torch.no_grad():
        for tile in tiles:
            # tile: (C, tile_size, tile_size) ndarray → torch (1, C, ts, ts)
            x = torch.from_numpy(tile.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x)  # (1, num_classes, ts, ts)
            probs = torch.softmax(logits, dim=1)  # (1, num_classes, ts, ts)
            prob_tiles.append(probs[0].cpu().numpy())  # (num_classes, ts, ts)

    # --- stitch probability maps ---
    prob_output_shape = (num_classes, H, W)
    prob_map = stitch_tiles(prob_tiles, origins, prob_output_shape, overlap=overlap)
    # prob_map: (num_classes, H, W)

    # --- derive classes and confidence ---
    classes = np.argmax(prob_map, axis=0)  # (H, W)
    confidence = np.max(prob_map, axis=0)  # (H, W)

    # --- apply confidence threshold ---
    low_conf_mask = (confidence < confidence_threshold) & (classes != 0)
    classes[low_conf_mask] = 0
    confidence[low_conf_mask] = prob_map[0][low_conf_mask]  # background prob

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )
