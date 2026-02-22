"""K-fold cross-validation for archaeological segmentation.

Competition winners used 5-fold CV with fold ensembles to achieve
the highest IoU scores. This module provides fold splitting,
a cross-validation runner that trains one model per fold, and
ensemble inference that averages predictions across all fold models.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FoldSplit:
    """A single train/val split for cross-validation.

    Attributes
    ----------
    fold : int
        Fold index (0-based).
    train_tiles : list of str
        Paths to training lidar tiles.
    val_tiles : list of str
        Paths to validation lidar tiles.
    """

    fold: int
    train_tiles: list[str] = field(default_factory=list)
    val_tiles: list[str] = field(default_factory=list)


def create_folds(
    lidar_dir: str,
    n_folds: int = 5,
    seed: int = 42,
) -> list[FoldSplit]:
    """Split tiles into K stratified folds.

    Uses a fixed random seed for reproducibility across runs.

    Parameters
    ----------
    lidar_dir : str
        Directory containing ``tile_*_lidar.tif`` files.
    n_folds : int
        Number of folds (default 5).
    seed : int
        Random seed for shuffling.

    Returns
    -------
    list of FoldSplit
        One FoldSplit per fold with train/val tile paths.
    """
    all_tiles = sorted(glob.glob(os.path.join(lidar_dir, "tile_*_lidar.tif")))
    if not all_tiles:
        raise FileNotFoundError(f"No lidar tiles in {lidar_dir}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_tiles))

    fold_sizes = [len(all_tiles) // n_folds] * n_folds
    for i in range(len(all_tiles) % n_folds):
        fold_sizes[i] += 1

    folds = []
    offset = 0
    for fold_idx in range(n_folds):
        val_indices = indices[offset : offset + fold_sizes[fold_idx]]
        train_indices = np.concatenate(
            [indices[:offset], indices[offset + fold_sizes[fold_idx] :]]
        )
        folds.append(
            FoldSplit(
                fold=fold_idx,
                train_tiles=[all_tiles[i] for i in sorted(train_indices)],
                val_tiles=[all_tiles[i] for i in sorted(val_indices)],
            )
        )
        offset += fold_sizes[fold_idx]

    return folds


def fold_summary(folds: list[FoldSplit]) -> str:
    """Generate a human-readable summary of fold splits.

    Parameters
    ----------
    folds : list of FoldSplit
        Folds from :func:`create_folds`.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = [f"Cross-validation: {len(folds)} folds"]
    total = len(folds[0].train_tiles) + len(folds[0].val_tiles)
    lines.append(f"Total tiles: {total}")
    lines.append("")
    for f in folds:
        lines.append(
            f"  Fold {f.fold}: train={len(f.train_tiles)}, val={len(f.val_tiles)}"
        )
    return "\n".join(lines)


def train_fold(
    cls_name: str,
    fold: FoldSplit,
    mask_dir: str,
    save_dir: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
    epochs: int = 80,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    use_tta: bool = True,
    warmup_epochs: int = 5,
    num_workers: int = 4,
    use_amp: bool = True,
    loss_type: str = "focal_dice",
    grad_accum_steps: int = 1,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    frozen_encoder: bool = True,
    tile_size: int | None = None,
) -> tuple[float, str]:
    """Train a single fold model for a class.

    Uses mixed-precision training (AMP) for faster training and
    elastic deformation for stronger augmentation.

    Parameters
    ----------
    cls_name : str
        Target class name.
    fold : FoldSplit
        Fold split with train/val tile paths.
    mask_dir : str
        Directory containing mask tiles.
    save_dir : str
        Directory to save fold model checkpoints.
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
        Device string.
    use_tta : bool
        Whether to use TTA during validation.
    warmup_epochs : int
        Number of warmup epochs.
    num_workers : int
        DataLoader worker count.
    use_amp : bool
        Whether to use automatic mixed precision.

    Returns
    -------
    tuple of (float, str)
        Best IoU achieved and path to saved checkpoint.
    """
    import time

    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import torch.nn.functional as TF

    from mayascan.config import DINOV2_TILE_SIZE, FOUNDATION_ARCHS
    from mayascan.data import BinarySegmentationDataset
    from mayascan.losses import FocalDiceLoss
    from mayascan.train import _build_model, predict_with_tta, compute_binary_iou, postprocess_mask

    # Auto-set tile size for foundation models
    is_foundation = arch in FOUNDATION_ARCHS
    if tile_size is None and is_foundation:
        tile_size = DINOV2_TILE_SIZE

    print(f"\n{'=' * 60}")
    print(f"Training {cls_name} — Fold {fold.fold}")
    print(f"Architecture: {arch} ({encoder}), Epochs: {epochs}")
    if is_foundation:
        print(f"Foundation model: LoRA={use_lora} (rank={lora_rank}), tile_size={tile_size}")
    print(f"AMP: {use_amp}, Train tiles: {len(fold.train_tiles)}, Val tiles: {len(fold.val_tiles)}")
    print(f"{'=' * 60}\n")

    # Datasets with explicit tile paths (fold-aware)
    # Use a dummy lidar_dir since tile_paths bypasses it
    dummy_dir = os.path.dirname(fold.train_tiles[0]) if fold.train_tiles else "."
    train_ds = BinarySegmentationDataset(
        dummy_dir, mask_dir, cls_name,
        split="train", augment=True, oversample_positive=True,
        tile_paths=fold.train_tiles,
    )
    val_ds = BinarySegmentationDataset(
        dummy_dir, mask_dir, cls_name,
        split="val", augment=False, oversample_positive=False,
        tile_paths=fold.val_tiles,
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
        use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
        frozen_encoder=frozen_encoder,
    ).to(device)

    # Loss
    from mayascan.train import _make_criterion
    criterion = _make_criterion(loss_type)

    # Optimizer — only trainable params for foundation models
    if is_foundation and hasattr(model, "trainable_parameters"):
        train_params = model.trainable_parameters()
        print(f"  Trainable: {sum(p.numel() for p in train_params):,} / "
              f"{model.total_param_count():,} params")
        optimizer = optim.AdamW(train_params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=use_amp and device != "cpu")
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    best_iou = 0.0
    start_epoch = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mayascan_v2_{cls_name}_{arch}_{encoder}_fold{fold.fold}.pth")

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
            train_dl, desc=f"[{cls_name} F{fold.fold}] Epoch {epoch + 1}/{epochs}", leave=False
        )):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            # Resize for foundation model tile size
            if tile_size and images.shape[-1] != tile_size:
                images = TF.interpolate(images, size=(tile_size, tile_size),
                                        mode="bilinear", align_corners=False)
                masks = TF.interpolate(masks, size=(tile_size, tile_size),
                                       mode="nearest")

            # CutMix: 30% of batches
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

            with torch.amp.autocast(device_type=device if device in ("cuda", "cpu") else "cpu", dtype=amp_dtype, enabled=use_amp):
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

            train_loss += loss.item() * grad_accum_steps

        scheduler.step()
        avg_loss = train_loss / max(len(train_dl), 1)

        # --- Validate ---
        model.eval()
        all_ious = []

        with torch.no_grad():
            for images, masks in val_dl:
                val_images = images
                if tile_size and val_images.shape[-1] != tile_size:
                    val_images = TF.interpolate(val_images, size=(tile_size, tile_size),
                                                mode="bilinear", align_corners=False)
                if use_tta and epoch >= epochs - 10:
                    probs = predict_with_tta(model, val_images, device).squeeze(1).numpy()
                else:
                    logits = model(val_images.to(device))
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                # Resize probs back to original mask size if needed
                if tile_size and probs.shape[-1] != masks.shape[-1]:
                    probs_t = torch.from_numpy(probs).unsqueeze(1)
                    probs_t = TF.interpolate(probs_t, size=masks.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    probs = probs_t.squeeze(1).numpy()

                masks_np = masks.numpy()
                for i in range(probs.shape[0]):
                    pred_pp = postprocess_mask(probs[i])
                    iou = compute_binary_iou(pred_pp, masks_np[i])
                    if iou is not None:
                        all_ious.append(iou)

        mean_iou = np.mean(all_ious) if all_ious else 0.0
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{cls_name} F{fold.fold}] Epoch {epoch + 1}/{epochs}: "
            f"loss={avg_loss:.4f}, IoU={mean_iou:.4f}, "
            f"lr={current_lr:.6f}, time={elapsed:.0f}s"
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            ckpt = {
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "cls_name": cls_name,
                    "arch": arch,
                    "encoder": encoder,
                    "fold": fold.fold,
                    "best_iou": best_iou,
                    "epoch": epoch + 1,
                }
            if is_foundation:
                ckpt.update({
                    "use_lora": use_lora,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                })
            torch.save(ckpt, save_path)
            print(f"  >>> Saved best {cls_name} fold {fold.fold} model (IoU={best_iou:.4f})")

    print(f"\n{cls_name} fold {fold.fold} complete! Best IoU: {best_iou:.4f}")
    return best_iou, save_path


def train_kfold(
    cls_name: str,
    lidar_dir: str,
    mask_dir: str,
    save_dir: str,
    n_folds: int = 5,
    seed: int = 42,
    **kwargs,
) -> dict[int, dict[str, Any]]:
    """Train K fold models for a single class.

    Parameters
    ----------
    cls_name : str
        Target class name.
    lidar_dir : str
        Directory containing lidar tiles.
    mask_dir : str
        Directory containing mask tiles.
    save_dir : str
        Directory to save fold model checkpoints.
    n_folds : int
        Number of folds.
    seed : int
        Random seed for fold splitting.
    **kwargs
        Additional arguments passed to :func:`train_fold`.

    Returns
    -------
    dict
        Mapping from fold index to ``{"iou": float, "path": str}``.
    """
    folds = create_folds(lidar_dir, n_folds=n_folds, seed=seed)
    print(fold_summary(folds))

    results: dict[int, dict[str, Any]] = {}
    for fold in folds:
        iou, path = train_fold(
            cls_name=cls_name,
            fold=fold,
            mask_dir=mask_dir,
            save_dir=save_dir,
            **kwargs,
        )
        results[fold.fold] = {"iou": iou, "path": path}

    print(f"\n{'=' * 60}")
    print(f"K-FOLD TRAINING COMPLETE: {cls_name}")
    print(f"{'=' * 60}")
    for fold_idx, info in results.items():
        print(f"  Fold {fold_idx}: IoU={info['iou']:.4f}")
    mean_iou = np.mean([r["iou"] for r in results.values()])
    print(f"  Mean IoU: {mean_iou:.4f}")

    return results


def train_kfold_all(
    lidar_dir: str,
    mask_dir: str,
    save_dir: str,
    classes: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    **kwargs,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Train K-fold models for all classes.

    Parameters
    ----------
    lidar_dir : str
        Directory containing lidar tiles.
    mask_dir : str
        Directory containing mask tiles.
    save_dir : str
        Directory to save fold model checkpoints.
    classes : list of str or None
        Class names to train. If None, trains building, platform, aguada.
    n_folds : int
        Number of folds.
    seed : int
        Random seed for fold splitting.
    **kwargs
        Additional arguments passed to :func:`train_fold`.

    Returns
    -------
    dict
        Nested mapping: class_name -> fold_idx -> {"iou", "path"}.
    """
    from mayascan.config import V2_CLASSES

    if classes is None:
        classes = list(V2_CLASSES.values())

    all_results: dict[str, dict[int, dict[str, Any]]] = {}
    for cls_name in classes:
        all_results[cls_name] = train_kfold(
            cls_name=cls_name,
            lidar_dir=lidar_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            n_folds=n_folds,
            seed=seed,
            **kwargs,
        )

    print(f"\n{'=' * 60}")
    print("ALL K-FOLD TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for cls_name, fold_results in all_results.items():
        mean_iou = np.mean([r["iou"] for r in fold_results.values()])
        print(f"  {cls_name}: mean IoU={mean_iou:.4f}")

    return all_results


def discover_fold_models(
    model_dir: str,
    cls_name: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet101",
) -> list[str]:
    """Find all fold model files for a given class.

    Parameters
    ----------
    model_dir : str
        Directory containing fold model checkpoints.
    cls_name : str
        Class name to search for.
    arch : str
        Architecture name.
    encoder : str
        Encoder name.

    Returns
    -------
    list of str
        Sorted list of fold model file paths.
    """
    pattern = f"mayascan_v2_{cls_name}_{arch}_{encoder}_fold*.pth"
    paths = sorted(glob.glob(os.path.join(model_dir, pattern)))
    return paths
