"""Upload trained MayaScan models to HuggingFace Hub.

Uploads per-class binary models with a generated model card containing
training metadata, architecture details, and performance metrics.

Usage:
    python upload_models.py                            # upload all models
    python upload_models.py --dry-run                  # show what would be uploaded
    python upload_models.py --repo fascinated23/mayascan  # custom repo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from mayascan.config import (
    CLASS_NAMES,
    V2_CLASSES,
    V2_ARCH,
    V2_ENCODER,
    HF_REPO_ID,
    TILE_SIZE,
)


MODEL_DIR = Path("/Volumes/macos4tb/Projects/mayascan/models")


def gather_model_info(model_dir: Path) -> list[dict]:
    """Gather metadata from all model checkpoints."""
    models = []
    for cls_id, cls_name in V2_CLASSES.items():
        filename = f"mayascan_v2_{cls_name}_{V2_ARCH}_{V2_ENCODER}.pth"
        path = model_dir / filename
        if not path.exists():
            continue

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        info = {
            "cls_id": cls_id,
            "cls_name": cls_name,
            "filename": filename,
            "path": str(path),
            "size_mb": path.stat().st_size / (1024 * 1024),
        }
        if isinstance(ckpt, dict):
            info["epoch"] = ckpt.get("epoch", "?")
            info["best_iou"] = ckpt.get("best_iou", None)
            info["arch"] = ckpt.get("arch", V2_ARCH)
            info["encoder"] = ckpt.get("encoder", V2_ENCODER)
        models.append(info)

    # Also check for v1 model
    v1_path = model_dir.parent / "mayascan_unet_best.pth"
    if v1_path.exists():
        models.append({
            "cls_id": -1,
            "cls_name": "v1_multiclass",
            "filename": v1_path.name,
            "path": str(v1_path),
            "size_mb": v1_path.stat().st_size / (1024 * 1024),
            "arch": "unet",
            "encoder": "resnet34",
        })

    return models


def generate_model_card(models: list[dict]) -> str:
    """Generate a HuggingFace model card (README.md) from model metadata."""
    v2_models = [m for m in models if m["cls_id"] > 0]
    v1_models = [m for m in models if m["cls_id"] == -1]

    # Compute mean IoU
    ious = [m["best_iou"] for m in v2_models if m.get("best_iou")]
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    card = f"""---
license: mit
tags:
  - archaeology
  - lidar
  - segmentation
  - maya
  - deep-learning
  - remote-sensing
  - deeplabv3plus
  - pytorch
datasets:
  - custom
pipeline_tag: image-segmentation
---

# MayaScan: Archaeological LiDAR Feature Detection

Open-source deep learning models for detecting ancient Maya archaeological
structures from LiDAR-derived terrain visualizations.

## Model Description

MayaScan uses **per-class binary segmentation** with DeepLabV3+ (ResNet-101)
to detect three types of archaeological features:

| Class | Description | Architecture | IoU |
|-------|------------|-------------|-----|
"""

    for m in v2_models:
        iou_str = f"{m['best_iou']:.4f}" if m.get("best_iou") else "N/A"
        card += f"| **{m['cls_name'].capitalize()}** | Ancient Maya {m['cls_name']}s | {m.get('arch', V2_ARCH)} ({m.get('encoder', V2_ENCODER)}) | {iou_str} |\n"

    if mean_iou > 0:
        card += f"\n**Mean IoU: {mean_iou:.4f}**\n"

    card += f"""
## Training Details

- **Architecture**: DeepLabV3+ with ResNet-101 encoder (ImageNet pretrained)
- **Loss**: Focal + Dice combined loss (handles extreme class imbalance)
- **Augmentation**: 90-degree rotation, flip, brightness, Gaussian noise, channel shuffle
- **Oversampling**: Up to 6x for positive tiles (rare classes like aguada)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4) with cosine annealing
- **Input**: 3-channel terrain visualization (SVF, openness, slope), {TILE_SIZE}x{TILE_SIZE} tiles
- **TTA**: 8-fold test-time augmentation (4 rotations x 2 flips)
- **Dataset**: Chactun archaeological site (Belize), 2090 tiles

## Model Files

"""

    for m in v2_models:
        epoch_str = f"epoch {m.get('epoch', '?')}" if m.get("epoch") else ""
        iou_str = f"IoU={m['best_iou']:.4f}" if m.get("best_iou") else ""
        card += f"- `{m['filename']}` — {m['cls_name']} ({m['size_mb']:.0f} MB, {epoch_str}, {iou_str})\n"

    for m in v1_models:
        card += f"- `{m['filename']}` — v1 multi-class U-Net ({m['size_mb']:.0f} MB)\n"

    card += f"""
## Usage

```python
import mayascan

# Process a DEM
result = mayascan.process_dem(dem_array, resolution=0.5)

# Or use v2 models directly
viz = mayascan.visualize(dem, resolution=0.5)
result = mayascan.detect_v2(viz, model_dir="models/")

# Extract features
features = mayascan.extract_features(result, pixel_size=0.5)
```

## CLI

```bash
# Download models
mayascan download

# Run detection
mayascan scan input.tif -o results/
```

## Citation

If you use MayaScan in your research, please cite:

```bibtex
@software{{mayascan,
  title = {{MayaScan: Open-source Archaeological LiDAR Feature Detection}},
  author = {{Holmes, Chris}},
  year = {{{datetime.now().year}}},
  url = {{https://github.com/fascinatedbyeverything/mayascan}}
}}
```

## License

MIT License
"""

    return card


def upload(repo_id: str, model_dir: Path, dry_run: bool = False) -> None:
    """Upload models and model card to HuggingFace Hub."""
    models = gather_model_info(model_dir)

    if not models:
        print(f"No models found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(models)} model(s):")
    for m in models:
        iou_str = f"IoU={m['best_iou']:.4f}" if m.get("best_iou") else ""
        print(f"  {m['filename']:50s}  {m['size_mb']:6.0f} MB  {iou_str}")

    # Generate model card
    card = generate_model_card(models)

    if dry_run:
        print(f"\n--- Model Card (dry run) ---")
        print(card)
        print(f"--- End Model Card ---\n")
        print(f"Would upload to: {repo_id}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface-hub is required. Install with: pip install mayascan[hub]")
        sys.exit(1)

    api = HfApi()

    # Upload model card
    print(f"\nUploading model card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload model files
    for m in models:
        path = Path(m["path"])
        if not path.exists():
            print(f"  Skipping {m['filename']} (file not found)")
            continue
        print(f"  Uploading {m['filename']} ({m['size_mb']:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=m["filename"],
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nDone! Models uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload MayaScan models to HuggingFace")
    parser.add_argument("--repo", default=HF_REPO_ID, help=f"HF repo ID (default: {HF_REPO_ID})")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Model directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    upload(args.repo, Path(args.model_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
