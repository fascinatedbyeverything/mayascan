"""Training module for archaeological segmentation models.

Provides a reusable training loop using the library's data, losses,
augment, and metrics modules. Can be invoked via ``mayascan train``
or imported for custom training scripts.

Based on competition-winning techniques from ECML PKDD 2021:
  - Per-class binary models (not multi-class)
  - Focal + Dice combo loss
  - Heavy augmentation + oversampling
  - Warmup + cosine annealing
  - Test-Time Augmentation at evaluation
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mayascan.config import (
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    DINOV2_TILE_SIZE,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOUNDATION_ARCHS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MIN_BLOB_SIZE,
    V2_ARCH,
    V2_CLASSES,
    V2_ENCODER,
)
from mayascan.data import BinarySegmentationDataset
from mayascan.losses import FocalDiceLoss


def _build_model(
    arch: str = V2_ARCH,
    encoder: str = V2_ENCODER,
    in_channels: int = 3,
    classes: int = 1,
    use_lora: bool = True,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    frozen_encoder: bool = True,
) -> nn.Module:
    """Create a segmentation model.

    Parameters
    ----------
    arch : str
        Architecture name: ``"deeplabv3plus"``, ``"unetplusplus"``, ``"unet"``,
        ``"segformer"``, ``"upernet"``, ``"manet"``, ``"fpn"``, ``"dinov2"``.
    encoder : str
        Encoder backbone name (e.g. ``"resnet101"``, ``"dinov2-large"``).
    in_channels : int
        Number of input channels.
    classes : int
        Number of output classes.
    use_lora : bool
        Enable LoRA fine-tuning (foundation models only).
    lora_rank : int
        LoRA rank (foundation models only).
    lora_alpha : int
        LoRA alpha (foundation models only).
    frozen_encoder : bool
        Freeze encoder weights (foundation models only).

    Returns
    -------
    nn.Module
        Segmentation model.
    """
    # Foundation model path
    if arch == "dinov2":
        from mayascan.models.dinov2 import DINOV2_MODELS, DINOv2Segmenter

        if encoder not in DINOV2_MODELS:
            raise ValueError(
                f"Unknown DINOv2 encoder: {encoder}. "
                f"Choose from {list(DINOV2_MODELS)}"
            )
        return DINOv2Segmenter(
            encoder_name=encoder,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            frozen_encoder=frozen_encoder,
            classes=classes,
        )

    # smp path
    import segmentation_models_pytorch as smp

    builders = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "segformer": smp.Segformer,
        "upernet": smp.UPerNet,
        "manet": smp.MAnet,
        "fpn": smp.FPN,
    }
    if arch not in builders:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(builders)}")

    return builders[arch](
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Test-Time Augmentation: average predictions over 8 orientations.

    Parameters
    ----------
    model : nn.Module
        Segmentation model in eval mode.
    images : torch.Tensor
        Batch of images, shape (B, C, H, W).
    device : str or torch.device
        Device for inference.

    Returns
    -------
    torch.Tensor
        Averaged probability map, shape (B, 1, H, W).
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for k in range(4):
            for flip in [False, True]:
                x = torch.rot90(images, k, dims=[2, 3])
                if flip:
                    x = torch.flip(x, dims=[3])

                pred = torch.sigmoid(model(x.to(device)))

                if flip:
                    pred = torch.flip(pred, dims=[3])
                pred = torch.rot90(pred, -k, dims=[2, 3])

                predictions.append(pred.cpu())

    return torch.stack(predictions).mean(dim=0)


def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = CONFIDENCE_THRESHOLD,
    min_blob_size: int = MIN_BLOB_SIZE,
) -> np.ndarray:
    """Threshold probability map and remove small blobs.

    Parameters
    ----------
    prob_map : np.ndarray
        Probability map, shape (H, W), values in [0, 1].
    threshold : float
        Probability threshold for binarization.
    min_blob_size : int
        Minimum connected component size to keep.

    Returns
    -------
    np.ndarray
        Binary mask, shape (H, W), dtype uint8.
    """
    from scipy.ndimage import label

    binary = (prob_map > threshold).astype(np.uint8)
    labeled, num_features = label(binary)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_blob_size:
            binary[labeled == i] = 0
    return binary


def compute_binary_iou(preds: np.ndarray, masks: np.ndarray) -> float | None:
    """Compute IoU for binary predictions.

    Returns None for tiles where both prediction and ground truth are empty
    (true negatives), so they can be excluded from averaging. This prevents
    rare classes like aguada (3.7% positive tiles) from having their IoU
    dragged to near-zero by empty tiles.

    Parameters
    ----------
    preds : np.ndarray
        Binary prediction mask.
    masks : np.ndarray
        Binary ground truth mask.

    Returns
    -------
    float or None
        Intersection over Union, or None if both are empty.
    """
    pred_pos = preds > 0
    mask_pos = masks > 0
    intersection = (pred_pos & mask_pos).sum()
    union = (pred_pos | mask_pos).sum()
    if union == 0:
        # Both empty: true negative — exclude from IoU average
        return None
    return float(intersection / union)


def _make_criterion(loss_type: str = "focal_dice") -> nn.Module:
    """Create loss function by name.

    Parameters
    ----------
    loss_type : str
        ``"focal_dice"`` or ``"focal_lovasz"``.
    """
    if loss_type == "focal_lovasz":
        from mayascan.losses import FocalLovaszLoss
        return FocalLovaszLoss(focal_weight=1.0, lovasz_weight=1.0,
                               alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    return FocalDiceLoss(
        focal_weight=1.0, dice_weight=1.0,
        alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
    )


def train_class(
    cls_name: str,
    lidar_dir: str,
    mask_dir: str,
    save_dir: str,
    arch: str = V2_ARCH,
    encoder: str = V2_ENCODER,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: str = "cpu",
    use_tta: bool = True,
    warmup_epochs: int = 5,
    num_workers: int = 4,
    use_amp: bool = False,
    loss_type: str = "focal_dice",
    grad_accum_steps: int = 1,
    use_lora: bool = True,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    frozen_encoder: bool = True,
    tile_size: int | None = None,
) -> tuple[float, str]:
    """Train a binary segmentation model for a single class.

    Parameters
    ----------
    cls_name : str
        Target class name (e.g. ``"building"``).
    lidar_dir : str
        Directory containing lidar tiles.
    mask_dir : str
        Directory containing mask tiles.
    save_dir : str
        Directory to save model checkpoints.
    arch : str
        Model architecture.
    encoder : str
        Encoder backbone.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    device : str
        Device string (``"cuda"``, ``"mps"``, ``"cpu"``).
    use_tta : bool
        Whether to use TTA during validation (last 10 epochs).
    warmup_epochs : int
        Number of warmup epochs.
    num_workers : int
        DataLoader worker count.
    use_amp : bool
        Whether to use automatic mixed precision for faster training.

    Returns
    -------
    tuple of (float, str)
        Best IoU achieved and path to saved checkpoint.
    """
    # Auto-set tile size for foundation models
    is_foundation = arch in FOUNDATION_ARCHS
    if tile_size is None and is_foundation:
        tile_size = DINOV2_TILE_SIZE

    print(f"\n{'=' * 60}")
    print(f"Training binary model: {cls_name}")
    print(f"Architecture: {arch} ({encoder}), Epochs: {epochs}, AMP: {use_amp}")
    if is_foundation:
        print(f"Foundation model: LoRA={use_lora} (rank={lora_rank}), "
              f"tile_size={tile_size}")
    print(f"{'=' * 60}\n")

    # Datasets
    train_ds = BinarySegmentationDataset(
        lidar_dir, mask_dir, cls_name,
        split="train", augment=True, oversample_positive=True,
    )
    val_ds = BinarySegmentationDataset(
        lidar_dir, mask_dir, cls_name,
        split="val", augment=False, oversample_positive=False,
    )

    stats = train_ds.stats
    print(f"[train] {cls_name}: {stats['total_tiles']} tiles "
          f"({stats['positive_tiles']} positive, {stats['negative_tiles']} negative)")
    stats_v = val_ds.stats
    print(f"[val]   {cls_name}: {stats_v['total_tiles']} tiles "
          f"({stats_v['positive_tiles']} positive, {stats_v['negative_tiles']} negative)")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    # Model
    model = _build_model(
        arch, encoder,
        use_lora=use_lora, lora_rank=lora_rank,
        lora_alpha=lora_alpha, frozen_encoder=frozen_encoder,
    ).to(device)

    # Loss
    criterion = _make_criterion(loss_type)

    # Optimizer: only trainable params for foundation models
    if is_foundation:
        train_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(train_params, lr=lr, weight_decay=1e-4)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in train_params)
        print(f"  Trainable params: {trainable:,} / {total:,} total "
              f"({100 * trainable / total:.1f}%)")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_amp and device != "cpu")
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    best_iou = 0.0
    start_epoch = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mayascan_v2_{cls_name}_{arch}_{encoder}.pth")

    # Resume from checkpoint
    if os.path.isfile(save_path):
        print(f"  Resuming from checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            best_iou = checkpoint.get("best_iou", 0.0)
            start_epoch = checkpoint.get("epoch", 0)
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"  Resumed at epoch {start_epoch}, best IoU={best_iou:.4f}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (images, masks) in enumerate(tqdm(
            train_dl, desc=f"[{cls_name}] Epoch {epoch + 1}/{epochs}", leave=False
        )):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            # Resize tiles for foundation models (e.g. 480->518 for DINOv2)
            if tile_size is not None and images.shape[-1] != tile_size:
                images = torch.nn.functional.interpolate(
                    images, size=(tile_size, tile_size), mode="bilinear",
                    align_corners=False,
                )
                masks = torch.nn.functional.interpolate(
                    masks, size=(tile_size, tile_size), mode="nearest",
                )

            # CutMix: 30% of batches get a random rectangular patch swapped
            if np.random.rand() < 0.3 and images.shape[0] > 1:
                perm = torch.randperm(images.shape[0])
                lam = np.random.beta(1.0, 1.0)
                _, _, h, w = images.shape
                cut_h = int(h * np.sqrt(1 - lam))
                cut_w = int(w * np.sqrt(1 - lam))
                cy, cx = np.random.randint(h), np.random.randint(w)
                y1 = max(0, cy - cut_h // 2)
                y2 = min(h, cy + cut_h // 2)
                x1 = max(0, cx - cut_w // 2)
                x2 = min(w, cx + cut_w // 2)
                images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
                masks[:, :, y1:y2, x1:x2] = masks[perm, :, y1:y2, x1:x2]

            with torch.amp.autocast(
                device_type=device if device in ("cuda", "cpu") else "cpu",
                dtype=amp_dtype, enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, masks) / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dl):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.detach().item() * grad_accum_steps

        scheduler.step()
        avg_loss = train_loss / max(len(train_dl), 1)

        # --- Validate ---
        model.eval()
        all_ious = []
        all_ious_pp = []

        with torch.no_grad():
            for images, masks in val_dl:
                # Resize for foundation models
                val_images = images
                if tile_size is not None and images.shape[-1] != tile_size:
                    val_images = torch.nn.functional.interpolate(
                        images, size=(tile_size, tile_size), mode="bilinear",
                        align_corners=False,
                    )
                if use_tta and epoch >= epochs - 10:
                    probs = predict_with_tta(model, val_images, device).squeeze(1).numpy()
                else:
                    logits = model(val_images.to(device))
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                # Resize probs back to original mask size if needed
                if tile_size is not None and probs.shape[-1] != masks.shape[-1]:
                    probs = torch.nn.functional.interpolate(
                        torch.from_numpy(probs).unsqueeze(1),
                        size=masks.shape[-2:], mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).numpy()

                masks_np = masks.numpy()

                for i in range(probs.shape[0]):
                    pred_raw = (probs[i] > 0.5).astype(np.uint8)
                    iou = compute_binary_iou(pred_raw, masks_np[i])
                    if iou is not None:
                        all_ious.append(iou)

                    pred_pp = postprocess_mask(probs[i])
                    iou_pp = compute_binary_iou(pred_pp, masks_np[i])
                    if iou_pp is not None:
                        all_ious_pp.append(iou_pp)

        mean_iou = np.mean(all_ious) if all_ious else 0.0
        mean_iou_pp = np.mean(all_ious_pp) if all_ious_pp else 0.0
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{cls_name}] Epoch {epoch + 1}/{epochs}: "
            f"loss={avg_loss:.4f}, IoU={mean_iou:.4f}, "
            f"IoU+PP={mean_iou_pp:.4f}, lr={current_lr:.6f}, "
            f"time={elapsed:.0f}s"
        )

        eval_iou = mean_iou_pp if mean_iou_pp > 0 else mean_iou
        if eval_iou > best_iou:
            best_iou = eval_iou
            ckpt = {
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "cls_name": cls_name,
                    "arch": arch,
                    "encoder": encoder,
                    "best_iou": best_iou,
                    "epoch": epoch + 1,
                }
            if is_foundation:
                    ckpt.update({
                        "use_lora": use_lora,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                        "frozen_encoder": frozen_encoder,
                    })
            torch.save(ckpt, save_path)
            print(f"  >>> Saved best {cls_name} model (IoU={best_iou:.4f})")

    print(f"\n{cls_name} training complete! Best IoU: {best_iou:.4f}")
    return best_iou, save_path


def train_all(
    lidar_dir: str,
    mask_dir: str,
    save_dir: str,
    classes: list[str] | None = None,
    **kwargs,
) -> dict[str, dict]:
    """Train models for all (or selected) classes.

    Parameters
    ----------
    lidar_dir : str
        Directory containing lidar tiles.
    mask_dir : str
        Directory containing mask tiles.
    save_dir : str
        Directory to save model checkpoints.
    classes : list of str or None
        Class names to train. If None, trains all V2_CLASSES.
    **kwargs
        Additional arguments passed to :func:`train_class`.

    Returns
    -------
    dict
        Mapping from class name to ``{"iou": float, "path": str}``.
    """
    if classes is None:
        classes = list(V2_CLASSES.values())

    results = {}
    for cls_name in classes:
        iou, path = train_class(
            cls_name=cls_name,
            lidar_dir=lidar_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            **kwargs,
        )
        results[cls_name] = {"iou": iou, "path": path}

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for cls_name, info in results.items():
        print(f"  {cls_name}: IoU={info['iou']:.4f}")
    if len(results) > 1:
        mean = np.mean([r["iou"] for r in results.values()])
        print(f"  Mean IoU: {mean:.4f}")

    return results
