"""Detection module: tiled inference pipeline for archaeological feature segmentation.

Supports two model formats:
  - **v1** (single multi-class model): one U-Net producing 4-class softmax output.
  - **v2** (per-class binary models): separate DeepLabV3+/U-Net++ models per class,
    producing sigmoid probabilities.  Achieves higher accuracy via competition-grade
    techniques (Focal+Dice loss, TTA, post-processing).

Tiles visualization rasters, runs inference on each tile, stitches results back
together, and applies a confidence threshold to produce a final segmentation map.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch

from mayascan.models.unet import MayaScanUNet
from mayascan.tile import slice_tiles, stitch_tiles

CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "building",
    2: "platform",
    3: "aguada",
}

# v2 model naming convention: mayascan_v2_{class}_{arch}_{encoder}.pth
V2_CLASSES = {1: "building", 2: "platform", 3: "aguada"}


@dataclass
class GeoInfo:
    """Georeferencing information extracted from an input raster.

    Attributes
    ----------
    crs : str or None
        Coordinate reference system (e.g. ``"EPSG:32616"``).
    transform : tuple or None
        Affine transform as a 6-element tuple ``(a, b, c, d, e, f)``
        mapping pixel coordinates to map coordinates.
    bounds : tuple or None
        Bounding box ``(left, bottom, right, top)`` in map coordinates.
    resolution : float
        Pixel size in map units (metres). Falls back to 0.5 if unknown.
    """

    crs: str | None = None
    transform: tuple[float, ...] | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: float = 0.5


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
    geo : GeoInfo or None
        Georeferencing from the source raster, if available.
    """

    classes: np.ndarray
    confidence: np.ndarray
    class_names: dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    geo: GeoInfo | None = None


def _select_device(device: torch.device | str | None) -> torch.device:
    """Pick the best available device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict_tile_with_tta(
    model: torch.nn.Module,
    tile: np.ndarray,
    device: torch.device,
    use_tta: bool,
    binary: bool,
) -> np.ndarray:
    """Run inference on a single tile, optionally with 8-fold TTA.

    Returns probability array: (num_classes, H, W) for multi-class,
    or (1, H, W) for binary.
    """
    def _infer(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        logits = model(t)
        if binary:
            return torch.sigmoid(logits)[0].cpu().numpy()
        else:
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    if not use_tta:
        return _infer(tile)

    # 8 orientations: 4 rotations × 2 flips
    accum = np.zeros_like(_infer(tile))
    count = 0
    for k in range(4):
        rotated = np.rot90(tile, k, axes=(1, 2)).copy()
        for flip in [False, True]:
            aug = np.flip(rotated, axis=2).copy() if flip else rotated
            pred = _infer(aug)
            # Reverse augmentation
            if flip:
                pred = np.flip(pred, axis=2).copy()
            pred = np.rot90(pred, -k, axes=(1, 2)).copy()
            accum += pred
            count += 1

    return accum / count


def _load_v2_model(
    model_path: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a v2 per-class binary segmentation model."""
    arch_map = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
    }
    model_cls = arch_map.get(arch, smp.DeepLabV3Plus)
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Handle full checkpoint dict vs bare state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def discover_v2_models(
    model_dir: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
) -> dict[int, str]:
    """Find v2 per-class model files in a directory.

    Returns mapping from class_id to model file path.
    """
    found = {}
    for cls_id, cls_name in V2_CLASSES.items():
        filename = f"mayascan_v2_{cls_name}_{arch}_{encoder}.pth"
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            found[cls_id] = path
    return found


def run_detection(
    visualization: np.ndarray,
    model_path: str | None = None,
    tile_size: int = 480,
    overlap: float = 0.5,
    confidence_threshold: float = 0.5,
    device: torch.device | str | None = None,
) -> DetectionResult:
    """Run tiled v1 multi-class U-Net inference on a visualization raster.

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
        Device for inference.  If *None*, best available device is used.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    device = _select_device(device)

    # --- model setup ---
    num_classes = len(CLASS_NAMES)
    model = MayaScanUNet(num_classes=num_classes, encoder="resnet34", pretrained=False)

    if model_path is not None:
        state = torch.load(model_path, map_location=device, weights_only=False)
        # Handle state_dict saved from raw smp.Unet (no "net." prefix)
        if any(k.startswith("encoder.") for k in state):
            state = {f"net.{k}": v for k, v in state.items()}
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
            x = torch.from_numpy(tile.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            prob_tiles.append(probs[0].cpu().numpy())

    # --- stitch probability maps ---
    prob_output_shape = (num_classes, H, W)
    prob_map = stitch_tiles(prob_tiles, origins, prob_output_shape, overlap=overlap)

    # --- derive classes and confidence ---
    classes = np.argmax(prob_map, axis=0)
    confidence = np.max(prob_map, axis=0)

    # --- apply confidence threshold ---
    low_conf_mask = (confidence < confidence_threshold) & (classes != 0)
    classes[low_conf_mask] = 0
    confidence[low_conf_mask] = prob_map[0][low_conf_mask]

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )


def run_detection_v2(
    visualization: np.ndarray,
    model_dir: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
    tile_size: int = 480,
    overlap: float = 0.5,
    confidence_threshold: float = 0.5,
    use_tta: bool = True,
    min_blob_size: int = 50,
    device: torch.device | str | None = None,
) -> DetectionResult:
    """Run tiled inference using v2 per-class binary models with TTA.

    Loads separate binary segmentation models for each class, runs inference
    with optional test-time augmentation (8 orientations), merges per-class
    probabilities, and applies post-processing (blob filtering).

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)`` where *C* is typically 3.
    model_dir : str
        Directory containing per-class model files named
        ``mayascan_v2_{class}_{arch}_{encoder}.pth``.
    arch : str
        Architecture name: ``"deeplabv3plus"``, ``"unetplusplus"``, or ``"unet"``.
    encoder : str
        Encoder backbone name (e.g. ``"resnet101"``).
    tile_size : int
        Side length of each square tile fed to the model (default 480).
    overlap : float
        Fractional overlap between adjacent tiles, in ``[0, 1)``
        (default 0.5).
    confidence_threshold : float
        Per-class probability threshold. Pixels below this are set to background.
    use_tta : bool
        If True, use 8-fold test-time augmentation (4 rotations x 2 flips).
    min_blob_size : int
        Minimum connected component size (in pixels). Smaller blobs are removed.
    device : torch.device, str, or None
        Device for inference.

    Returns
    -------
    DetectionResult
        Dataclass with ``classes``, ``confidence``, and ``class_names``.
    """
    device = _select_device(device)

    # Discover available per-class models
    model_paths = discover_v2_models(model_dir, arch, encoder)
    if not model_paths:
        raise FileNotFoundError(
            f"No v2 models found in {model_dir} for arch={arch}, encoder={encoder}"
        )

    C, H, W = visualization.shape
    tiles, origins = slice_tiles(visualization, tile_size=tile_size, overlap=overlap)

    # Per-class probability maps: class_id -> (H, W) probability
    class_probs: dict[int, np.ndarray] = {}

    for cls_id, mpath in model_paths.items():
        model = _load_v2_model(mpath, arch=arch, encoder=encoder, device=device)

        # Run inference per tile
        prob_tiles: list[np.ndarray] = []
        with torch.no_grad():
            for tile in tiles:
                prob = _predict_tile_with_tta(model, tile, device, use_tta, binary=True)
                prob_tiles.append(prob)

        # Stitch per-class probability
        prob_map = stitch_tiles(prob_tiles, origins, (1, H, W), overlap=overlap)
        class_probs[cls_id] = prob_map[0]  # (H, W)

        # Free model memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Build combined probability map: (num_classes, H, W)
    num_classes = len(CLASS_NAMES)
    full_prob = np.zeros((num_classes, H, W), dtype=np.float32)
    for cls_id, prob in class_probs.items():
        full_prob[cls_id] = prob

    # Background probability = 1 - max(class probabilities)
    max_fg = np.max(full_prob[1:], axis=0) if len(class_probs) > 0 else np.zeros((H, W))
    full_prob[0] = 1.0 - max_fg

    # Derive classes: highest probability wins, but only if above threshold
    classes = np.argmax(full_prob, axis=0).astype(np.int64)
    confidence = np.max(full_prob, axis=0).astype(np.float32)

    # Apply threshold: reset low-confidence non-background to background
    for cls_id in class_probs:
        cls_mask = classes == cls_id
        low_conf = full_prob[cls_id] < confidence_threshold
        classes[cls_mask & low_conf] = 0

    # Post-processing: remove small blobs
    if min_blob_size > 0:
        try:
            from scipy import ndimage
            for cls_id in class_probs:
                cls_mask = classes == cls_id
                if not cls_mask.any():
                    continue
                labeled, n_features = ndimage.label(cls_mask)
                for i in range(1, n_features + 1):
                    blob = labeled == i
                    if blob.sum() < min_blob_size:
                        classes[blob] = 0
        except ImportError:
            pass  # scipy not available, skip blob filtering

    # Recalculate confidence for final class assignments
    confidence = np.take_along_axis(
        full_prob, classes[np.newaxis].astype(np.intp), axis=0
    )[0]

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=dict(CLASS_NAMES),
    )
