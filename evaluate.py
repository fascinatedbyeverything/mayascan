"""MayaScan evaluation — benchmark model performance on the Chactun validation set.

Supports both v1 (single multi-class U-Net) and v2 (per-class binary DeepLabV3+)
model formats. Computes per-class IoU, precision, recall, F1, and optionally
saves prediction overlay visualizations on LiDAR tiles.

Usage:
    python evaluate.py --model-dir models/                        # evaluate v2 models
    python evaluate.py --model mayascan_unet_best.pth             # evaluate v1 model
    python evaluate.py --model-dir models/ --save-viz results/    # save visualizations
    python evaluate.py --model-dir models/ --tta                  # with test-time augmentation
    python evaluate.py --model-dir models/ --arch unet --encoder resnet34  # custom arch
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from scipy import ndimage

from mayascan.config import CLASS_NAMES, V2_CLASSES, TILE_SIZE, V2_ARCH, V2_ENCODER
from mayascan.metrics import ClassMetrics, format_metrics_table, mean_iou as compute_mean_iou

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/Volumes/macos4tb/Projects/mayascan/chactun_data/extracted"


def update_metrics(
    metrics: dict[int, ClassMetrics],
    pred: np.ndarray,
    target: np.ndarray,
    cls_id: int,
) -> None:
    """Update accumulated metrics for one class on a single sample."""
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    m = metrics[cls_id]
    m.tp += int((pred_bool & target_bool).sum())
    m.fp += int((pred_bool & ~target_bool).sum())
    m.fn += int((~pred_bool & target_bool).sum())
    m.tn += int((~pred_bool & ~target_bool).sum())


# ---------------------------------------------------------------------------
# Data loading — same split logic as train_v2.py
# ---------------------------------------------------------------------------
def load_validation_set(
    data_dir: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return (lidar_paths, tile_ids, mask_dir) for the 20% validation split.

    Mirrors train_v2.py: sorted lidar files, last 20% is validation.
    """
    lidar_dir = os.path.join(data_dir, "lidar")
    mask_dir = os.path.join(data_dir, "masks")
    all_lidar = sorted(glob.glob(os.path.join(lidar_dir, "tile_*_lidar.tif")))
    if not all_lidar:
        raise FileNotFoundError(f"No lidar tiles found in {lidar_dir}")

    n = len(all_lidar)
    split_idx = int(n * 0.8)
    val_lidar = all_lidar[split_idx:]

    tile_ids = []
    for path in val_lidar:
        tid = os.path.basename(path).replace("tile_", "").replace("_lidar.tif", "")
        tile_ids.append(tid)

    return val_lidar, tile_ids, mask_dir


def load_lidar_tile(path: str) -> np.ndarray:
    """Load a LiDAR tile as (3, H, W) float32 in [0, 1]."""
    img = np.array(Image.open(path), dtype=np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    else:
        img = img.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return img


def load_mask(mask_dir: str, tile_id: str, cls_name: str) -> np.ndarray:
    """Load a binary mask for one class. Returns (H, W) with 1 = feature present."""
    mask_path = os.path.join(mask_dir, f"tile_{tile_id}_mask_{cls_name}.tif")
    if not os.path.exists(mask_path):
        return None
    m = np.array(Image.open(mask_path), dtype=np.float32)
    if m.ndim > 2:
        m = m[:, :, 0]
    return (m < 128).astype(np.float32)  # inverted: 0 = feature, 255 = no feature


# ---------------------------------------------------------------------------
# TTA (same logic as train_v2.py / detect.py)
# ---------------------------------------------------------------------------
def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    binary: bool = True,
) -> np.ndarray:
    """8-fold TTA: 4 rotations x 2 flips. Returns (B, C, H, W) probabilities."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for k in range(4):
            for flip in [False, True]:
                x = torch.rot90(images, k, dims=[2, 3])
                if flip:
                    x = torch.flip(x, dims=[3])

                logits = model(x.to(device))
                if binary:
                    pred = torch.sigmoid(logits)
                else:
                    pred = torch.softmax(logits, dim=1)

                # Undo augmentation
                if flip:
                    pred = torch.flip(pred, dims=[3])
                pred = torch.rot90(pred, -k, dims=[2, 3])

                predictions.append(pred.cpu())

    return torch.stack(predictions).mean(dim=0).numpy()


# ---------------------------------------------------------------------------
# Post-processing (same as train_v2.py)
# ---------------------------------------------------------------------------
def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_blob_size: int = 50,
) -> np.ndarray:
    """Threshold + remove small connected components."""
    binary = (prob_map > threshold).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_blob_size:
            binary[labeled == i] = 0
    return binary


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_v1_model(
    model_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a v1 multi-class U-Net model.

    v1 state dict keys start with 'encoder.' (raw smp.Unet), so we prefix
    with 'net.' to match the MayaScanUNet wrapper.
    """
    from mayascan.models.unet import MayaScanUNet

    model = MayaScanUNet(num_classes=4, encoder="resnet34", pretrained=False)
    state = torch.load(model_path, map_location=device, weights_only=False)

    # Handle raw smp.Unet state dict (keys start with "encoder." not "net.encoder.")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("encoder.") for k in state):
        state = {f"net.{k}": v for k, v in state.items()}

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def load_v2_model(
    model_path: str,
    arch: str,
    encoder: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load a v2 per-class binary model. Returns (model, checkpoint_meta)."""
    arch_map = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "segformer": smp.Segformer,
        "upernet": smp.UPerNet,
        "manet": smp.MAnet,
        "fpn": smp.FPN,
    }

    # Auto-detect arch/encoder from checkpoint metadata
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "arch" in checkpoint:
        arch = checkpoint["arch"]
    if isinstance(checkpoint, dict) and "encoder" in checkpoint:
        encoder = checkpoint["encoder"]

    model_cls = arch_map.get(arch, smp.DeepLabV3Plus)
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    meta = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    else:
        state = checkpoint

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, meta


def discover_v2_models(
    model_dir: str,
    arch: str = V2_ARCH,
    encoder: str = V2_ENCODER,
) -> dict[int, str]:
    """Find v2 per-class model files. Returns {cls_id: model_path}."""
    found = {}
    for cls_id, cls_name in V2_CLASSES.items():
        filename = f"mayascan_v2_{cls_name}_{arch}_{encoder}.pth"
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            found[cls_id] = path
    return found


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Class colors for overlays (RGBA, semi-transparent)
CLASS_COLORS = {
    1: (255, 0, 0, 128),     # building = red
    2: (0, 255, 0, 128),     # platform = green
    3: (0, 0, 255, 128),     # aguada = blue
}

# Ground truth outline colors (brighter, for comparison)
GT_COLORS = {
    1: (255, 128, 128, 180),  # building GT = light red
    2: (128, 255, 128, 180),  # platform GT = light green
    3: (128, 128, 255, 180),  # aguada GT = light blue
}


def save_visualization(
    lidar_path: str,
    tile_id: str,
    predictions: dict[int, np.ndarray],
    ground_truths: dict[int, np.ndarray],
    save_dir: str,
) -> str:
    """Save a prediction overlay visualization on the LiDAR tile.

    Creates a side-by-side image: left = prediction overlay, right = ground truth overlay.
    Returns the path to the saved image.
    """
    # Load original LiDAR as RGB for display
    img = np.array(Image.open(lidar_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    h, w = img.shape[:2]

    # Create prediction overlay
    pred_overlay = Image.fromarray(img.copy())
    pred_alpha = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for cls_id, pred_mask in predictions.items():
        color = CLASS_COLORS.get(cls_id, (255, 255, 0, 128))
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        layer[pred_mask > 0] = color
        layer_img = Image.fromarray(layer, "RGBA")
        pred_alpha = Image.alpha_composite(pred_alpha, layer_img)
    pred_overlay = Image.alpha_composite(pred_overlay.convert("RGBA"), pred_alpha)

    # Create ground truth overlay
    gt_overlay = Image.fromarray(img.copy())
    gt_alpha = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for cls_id, gt_mask in ground_truths.items():
        if gt_mask is None:
            continue
        color = GT_COLORS.get(cls_id, (255, 255, 0, 180))
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        layer[gt_mask > 0] = color
        layer_img = Image.fromarray(layer, "RGBA")
        gt_alpha = Image.alpha_composite(gt_alpha, layer_img)
    gt_overlay = Image.alpha_composite(gt_overlay.convert("RGBA"), gt_alpha)

    # Side-by-side: prediction | ground truth
    combined = Image.new("RGBA", (w * 2 + 10, h + 30), (32, 32, 32, 255))
    combined.paste(pred_overlay, (0, 30))
    combined.paste(gt_overlay, (w + 10, 30))

    # Add labels
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((5, 5), f"Prediction — tile {tile_id}", fill=(255, 255, 255, 255), font=font)
        draw.text((w + 15, 5), f"Ground Truth — tile {tile_id}", fill=(255, 255, 255, 255), font=font)
    except ImportError:
        pass  # PIL without ImageDraw, skip labels

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"eval_tile_{tile_id}.png")
    combined.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Evaluation: v2 per-class binary models
# ---------------------------------------------------------------------------
def evaluate_v2(
    model_dir: str,
    arch: str,
    encoder: str,
    device: torch.device,
    use_tta: bool,
    threshold: float,
    min_blob_size: int,
    save_viz: str | None,
) -> dict[int, ClassMetrics]:
    """Evaluate v2 per-class binary models on the validation set."""
    model_paths = discover_v2_models(model_dir, arch, encoder)
    if not model_paths:
        print(f"ERROR: No v2 models found in {model_dir} "
              f"for arch={arch}, encoder={encoder}")
        print(f"  Expected filenames like: mayascan_v2_building_{arch}_{encoder}.pth")
        sys.exit(1)

    val_lidar, tile_ids, mask_dir = load_validation_set(DATA_DIR)
    metrics = {cls_id: ClassMetrics() for cls_id in V2_CLASSES}

    print(f"\nEvaluating v2 models ({arch}/{encoder})")
    print(f"  Models found: {', '.join(V2_CLASSES[c] for c in sorted(model_paths))}")
    print(f"  Missing: {', '.join(V2_CLASSES[c] for c in sorted(set(V2_CLASSES) - set(model_paths)))}"
          if set(V2_CLASSES) - set(model_paths) else "  All classes present")
    print(f"  Validation tiles: {len(val_lidar)}")
    print(f"  TTA: {'yes (8-fold)' if use_tta else 'no'}")
    print(f"  Threshold: {threshold}, Min blob size: {min_blob_size}")

    # Print checkpoint metadata
    for cls_id, mpath in sorted(model_paths.items()):
        ckpt = torch.load(mpath, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            epoch = ckpt.get("epoch", "?")
            best_iou = ckpt.get("best_iou", "?")
            if isinstance(best_iou, float):
                best_iou = f"{best_iou:.4f}"
            print(f"  {V2_CLASSES[cls_id]:>10s}: epoch {epoch}, train-best IoU {best_iou}")

    print()

    # Load all per-class models into memory
    models: dict[int, torch.nn.Module] = {}
    for cls_id, mpath in model_paths.items():
        model, meta = load_v2_model(mpath, arch, encoder, device)
        models[cls_id] = model

    # Evaluate tile by tile
    viz_count = 0
    for i, (lidar_path, tid) in enumerate(zip(val_lidar, tile_ids)):
        img = load_lidar_tile(lidar_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)

        tile_preds: dict[int, np.ndarray] = {}
        tile_gts: dict[int, np.ndarray] = {}

        for cls_id, model in models.items():
            cls_name = V2_CLASSES[cls_id]

            # Ground truth
            gt = load_mask(mask_dir, tid, cls_name)
            if gt is None:
                # No mask file means no features of this class
                gt = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
            tile_gts[cls_id] = gt

            # Prediction
            with torch.no_grad():
                if use_tta:
                    probs = predict_with_tta(model, img_tensor, device, binary=True)
                    prob = probs[0, 0]  # (H, W)
                else:
                    logits = model(img_tensor.to(device))
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            pred = postprocess_mask(prob, threshold=threshold, min_blob_size=min_blob_size)
            tile_preds[cls_id] = pred

            # Accumulate metrics
            update_metrics(metrics, pred, gt, cls_id)

        # Progress
        status = f"  [{i+1}/{len(val_lidar)}] tile {tid}"
        for cls_id in sorted(tile_preds):
            n_pred = int(tile_preds[cls_id].sum())
            n_gt = int(tile_gts[cls_id].sum())
            status += f"  {V2_CLASSES[cls_id][:4]}={n_pred}/{n_gt}"
        print(status)

        # Save visualization if requested
        if save_viz is not None:
            out_path = save_visualization(
                lidar_path, tid, tile_preds, tile_gts, save_viz
            )
            viz_count += 1

    if save_viz is not None:
        print(f"\n  Saved {viz_count} visualization(s) to {save_viz}/")

    # Clean up models
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Evaluation: v1 multi-class U-Net
# ---------------------------------------------------------------------------
def evaluate_v1(
    model_path: str,
    device: torch.device,
    use_tta: bool,
    threshold: float,
    save_viz: str | None,
) -> dict[int, ClassMetrics]:
    """Evaluate v1 multi-class U-Net on the validation set."""
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    val_lidar, tile_ids, mask_dir = load_validation_set(DATA_DIR)
    metrics = {cls_id: ClassMetrics() for cls_id in V2_CLASSES}

    print(f"\nEvaluating v1 model: {model_path}")
    print(f"  Validation tiles: {len(val_lidar)}")
    print(f"  TTA: {'yes (8-fold)' if use_tta else 'no'}")
    print(f"  Threshold: {threshold}")
    print()

    model = load_v1_model(model_path, device)

    viz_count = 0
    for i, (lidar_path, tid) in enumerate(zip(val_lidar, tile_ids)):
        img = load_lidar_tile(lidar_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)

        # Inference: v1 produces (1, 4, H, W) logits → softmax → class probabilities
        with torch.no_grad():
            if use_tta:
                probs = predict_with_tta(model, img_tensor, device, binary=False)
                # probs shape: (1, 4, H, W)
            else:
                logits = model(img_tensor.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                # probs shape: (1, 4, H, W)

        tile_preds: dict[int, np.ndarray] = {}
        tile_gts: dict[int, np.ndarray] = {}

        for cls_id, cls_name in V2_CLASSES.items():
            # Ground truth
            gt = load_mask(mask_dir, tid, cls_name)
            if gt is None:
                gt = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
            tile_gts[cls_id] = gt

            # Prediction: class probability for this class
            cls_prob = probs[0, cls_id]  # (H, W)
            pred = (cls_prob > threshold).astype(np.uint8)
            tile_preds[cls_id] = pred

            update_metrics(metrics, pred, gt, cls_id)

        # Progress
        status = f"  [{i+1}/{len(val_lidar)}] tile {tid}"
        for cls_id in sorted(tile_preds):
            n_pred = int(tile_preds[cls_id].sum())
            n_gt = int(tile_gts[cls_id].sum())
            status += f"  {V2_CLASSES[cls_id][:4]}={n_pred}/{n_gt}"
        print(status)

        if save_viz is not None:
            out_path = save_visualization(
                lidar_path, tid, tile_preds, tile_gts, save_viz
            )
            viz_count += 1

    if save_viz is not None:
        print(f"\n  Saved {viz_count} visualization(s) to {save_viz}/")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
def print_results(metrics: dict[int, ClassMetrics], model_label: str) -> None:
    """Print a formatted results table."""
    print(f"\n{'=' * 72}")
    print(f"  EVALUATION RESULTS — {model_label}")
    print(f"{'=' * 72}")

    # Use library formatter for the standard table
    table = format_metrics_table(metrics, class_names=V2_CLASSES)
    for line in table.splitlines():
        print(f"  {line}")

    # Extended info: TP/FP/FN counts
    print()
    print(f"  {'Class':>12s}  {'TP':>10s}  {'FP':>10s}  {'FN':>10s}")
    print(f"  {'-' * 48}")
    for cls_id in sorted(metrics):
        m = metrics[cls_id]
        cls_name = V2_CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_name:>12s}  {m.tp:10d}  {m.fp:10d}  {m.fn:10d}")

    miou = compute_mean_iou(metrics)
    print(f"\n{'=' * 72}")
    print(f"  mIoU: {miou:.4f}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Threshold sweep: find optimal per-class thresholds in one pass
# ---------------------------------------------------------------------------
def sweep_thresholds_v2(
    model_dir: str,
    arch: str,
    encoder: str,
    device: torch.device,
    use_tta: bool,
    min_blob_size: int,
    thresholds: list[float] | None = None,
) -> dict[int, dict[str, float]]:
    """Find optimal threshold per class by evaluating at multiple thresholds.

    Returns dict: cls_id -> {"best_threshold": float, "best_iou": float, ...}
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

    model_paths = discover_v2_models(model_dir, arch, encoder)
    if not model_paths:
        print(f"ERROR: No v2 models found in {model_dir}")
        sys.exit(1)

    val_lidar, tile_ids, mask_dir = load_validation_set(DATA_DIR)

    # Multi-threshold metrics: {threshold -> {cls_id -> ClassMetrics}}
    multi_metrics: dict[float, dict[int, ClassMetrics]] = {
        t: {cls_id: ClassMetrics() for cls_id in V2_CLASSES}
        for t in thresholds
    }

    # Load models
    models: dict[int, torch.nn.Module] = {}
    for cls_id, mpath in model_paths.items():
        model, _ = load_v2_model(mpath, arch, encoder, device)
        models[cls_id] = model

    print(f"\nThreshold sweep: {thresholds}")
    print(f"  Models: {len(models)}, Tiles: {len(val_lidar)}, TTA: {use_tta}")

    # Single pass through all tiles
    for i, (lidar_path, tid) in enumerate(zip(val_lidar, tile_ids)):
        img = load_lidar_tile(lidar_path)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        for cls_id, model in models.items():
            cls_name = V2_CLASSES[cls_id]
            gt = load_mask(mask_dir, tid, cls_name)
            if gt is None:
                gt = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

            with torch.no_grad():
                if use_tta:
                    probs = predict_with_tta(model, img_tensor, device, binary=True)
                    prob = probs[0, 0]
                else:
                    logits = model(img_tensor.to(device))
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            # Evaluate at each threshold
            for t in thresholds:
                pred = postprocess_mask(prob, threshold=t, min_blob_size=min_blob_size)
                update_metrics(multi_metrics[t], pred, gt, cls_id)

        if (i + 1) % 50 == 0 or i == len(val_lidar) - 1:
            print(f"  [{i+1}/{len(val_lidar)}]")

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Find best threshold per class
    results: dict[int, dict[str, float]] = {}
    print(f"\n{'=' * 72}")
    print(f"  THRESHOLD SWEEP RESULTS")
    print(f"{'=' * 72}")

    for cls_id in sorted(V2_CLASSES):
        cls_name = V2_CLASSES[cls_id]
        best_t = 0.5
        best_iou = 0.0
        print(f"\n  {cls_name}:")
        print(f"    {'Threshold':>10s}  {'IoU':>8s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}")
        for t in thresholds:
            m = multi_metrics[t][cls_id]
            iou = m.iou
            prec = m.precision
            rec = m.recall
            f1 = m.f1
            marker = ""
            if iou > best_iou:
                best_iou = iou
                best_t = t
                marker = " <-- best"
            print(f"    {t:>10.2f}  {iou:>8.4f}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}{marker}")

        results[cls_id] = {
            "best_threshold": best_t,
            "best_iou": best_iou,
            "best_precision": multi_metrics[best_t][cls_id].precision,
            "best_recall": multi_metrics[best_t][cls_id].recall,
            "best_f1": multi_metrics[best_t][cls_id].f1,
        }

    print(f"\n{'=' * 72}")
    print(f"  OPTIMAL THRESHOLDS:")
    for cls_id in sorted(results):
        r = results[cls_id]
        cls_name = V2_CLASSES[cls_id]
        print(f"    {cls_name:>10s}: threshold={r['best_threshold']:.2f}, "
              f"IoU={r['best_iou']:.4f}, F1={r['best_f1']:.4f}")
    mean_best = np.mean([r["best_iou"] for r in results.values()])
    print(f"    {'mIoU':>10s}: {mean_best:.4f}")
    print(f"{'=' * 72}\n")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(
        description="Evaluate MayaScan models on the Chactun validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate.py --model-dir models/                        # v2 models (default)
  python evaluate.py --model mayascan_unet_best.pth             # v1 model
  python evaluate.py --model-dir models/ --save-viz results/    # save overlays
  python evaluate.py --model-dir models/ --tta                  # with TTA
  python evaluate.py --model-dir models/ --threshold 0.4        # custom threshold
""",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Directory containing v2 per-class binary models",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a v1 multi-class U-Net model (.pth)",
    )
    parser.add_argument(
        "--arch", type=str, default="deeplabv3plus",
        choices=["deeplabv3plus", "unetplusplus", "unet", "segformer", "upernet", "manet", "fpn"],
        help="v2 model architecture (default: deeplabv3plus)",
    )
    parser.add_argument(
        "--encoder", type=str, default="resnet101",
        help="v2 encoder backbone (default: resnet101)",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable test-time augmentation (8-fold, slower but more accurate)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for positive prediction (default: 0.5)",
    )
    parser.add_argument(
        "--min-blob-size", type=int, default=50,
        help="Minimum connected-component size in pixels (v2 only, default: 50)",
    )
    parser.add_argument(
        "--save-viz", type=str, default=None,
        help="Directory to save prediction overlay visualizations",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference: cuda, mps, or cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help=f"Chactun data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--sweep-threshold", action="store_true",
        help="Find optimal per-class thresholds (evaluates at 9 thresholds)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.model is None and args.model_dir is None:
        parser.error("Either --model (v1) or --model-dir (v2) is required")
    if args.model is not None and args.model_dir is not None:
        parser.error("Specify either --model (v1) or --model-dir (v2), not both")

    # Override data dir if specified
    DATA_DIR = args.data_dir

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"MayaScan Evaluation")
    print(f"  Device: {device}")
    print(f"  Data:   {DATA_DIR}")

    t0 = time.time()

    if args.sweep_threshold and args.model_dir is not None:
        # Threshold sweep mode
        sweep_results = sweep_thresholds_v2(
            model_dir=args.model_dir,
            arch=args.arch,
            encoder=args.encoder,
            device=device,
            use_tta=args.tta,
            min_blob_size=args.min_blob_size,
        )
        elapsed = time.time() - t0
        print(f"  Threshold sweep completed in {elapsed:.1f}s")
    elif args.model_dir is not None:
        # v2 evaluation
        metrics = evaluate_v2(
            model_dir=args.model_dir,
            arch=args.arch,
            encoder=args.encoder,
            device=device,
            use_tta=args.tta,
            threshold=args.threshold,
            min_blob_size=args.min_blob_size,
            save_viz=args.save_viz,
        )
        model_label = f"v2 {args.arch}/{args.encoder}"
        elapsed = time.time() - t0
        print(f"\n  Evaluation completed in {elapsed:.1f}s")
        print_results(metrics, model_label)
    else:
        # v1 evaluation
        metrics = evaluate_v1(
            model_path=args.model,
            device=device,
            use_tta=args.tta,
            threshold=args.threshold,
            save_viz=args.save_viz,
        )
        model_label = f"v1 U-Net (resnet34)"
        elapsed = time.time() - t0
        print(f"\n  Evaluation completed in {elapsed:.1f}s")
        print_results(metrics, model_label)


if __name__ == "__main__":
    main()
