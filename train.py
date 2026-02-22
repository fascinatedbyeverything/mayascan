"""MayaScan training script — train U-Net on Chactún ML-ready dataset.

Runs on Apple Silicon (MPS), CUDA, or CPU.

Usage:
    python train.py
"""

import glob
import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/Volumes/macos4tb/Projects/mayascan/chactun_data/extracted"
SAVE_PATH = "/Volumes/macos4tb/Projects/mayascan/mayascan_unet_best.pth"
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChactunDataset(Dataset):
    """Chactún ML-ready dataset.

    Each sample is:
      - input:  tile_XXXX_lidar.tif  (3-band: SVF, openness, slope, 480x480, 8-bit)
      - target: tile_XXXX_mask_{building,platform,aguada}.tif  (480x480, binary)

    Masks are merged into a single multi-class label:
      0 = background, 1 = building, 2 = platform, 3 = aguada
    """

    def __init__(self, lidar_dir: str, mask_dir: str, split: str = "train", augment: bool = True):
        # Find all lidar tiles
        all_lidar = sorted(glob.glob(os.path.join(lidar_dir, "tile_*_lidar.tif")))
        if not all_lidar:
            raise FileNotFoundError(f"No lidar tiles found in {lidar_dir}")

        # 80/20 split (deterministic — sorted by name)
        n = len(all_lidar)
        split_idx = int(n * 0.8)
        if split == "train":
            self.lidar_files = all_lidar[:split_idx]
        else:
            self.lidar_files = all_lidar[split_idx:]

        self.mask_dir = mask_dir
        self.augment = augment and (split == "train")
        print(f"[{split}] {len(self.lidar_files)} tiles (of {n} total)")

    def _tile_id(self, path: str) -> str:
        """Extract tile ID from path: tile_1469_lidar.tif -> 1469"""
        base = os.path.basename(path)
        # tile_XXXX_lidar.tif
        return base.replace("tile_", "").replace("_lidar.tif", "")

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, idx):
        lidar_path = self.lidar_files[idx]
        tile_id = self._tile_id(lidar_path)

        # Load 3-band LiDAR visualization (SVF, openness, slope)
        img = np.array(Image.open(lidar_path), dtype=np.float32)
        # Normalize uint8 -> [0, 1]
        img = img / 255.0

        if img.ndim == 2:
            # Single band — stack 3 times
            img = np.stack([img, img, img], axis=0)
        else:
            # (H, W, 3) -> (3, H, W)
            img = img.transpose(2, 0, 1)

        # Load masks -> multi-class target
        h, w = img.shape[1], img.shape[2]
        mask = np.zeros((h, w), dtype=np.int64)

        for cls_id, cls_name in [(1, "building"), (2, "platform"), (3, "aguada")]:
            mask_path = os.path.join(
                self.mask_dir, f"tile_{tile_id}_mask_{cls_name}.tif"
            )
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path), dtype=np.float32)
                if m.ndim > 2:
                    m = m[:, :, 0]
                mask[m < 128] = cls_id  # 0 = feature present, 255 = no feature

        # Augmentation: random rotation + flip (direction-invariant for LiDAR viz)
        if self.augment:
            k = np.random.randint(4)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

        return torch.from_numpy(img), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou_per_class(preds: torch.Tensor, masks: torch.Tensor, num_classes: int = 4):
    """Compute IoU for each class (skip background=0)."""
    ious = {}
    for c in range(1, num_classes):
        pred_c = preds == c
        mask_c = masks == c
        intersection = (pred_c & mask_c).sum().float()
        union = (pred_c | mask_c).sum().float()
        if union > 0:
            ious[c] = (intersection / union).item()
    return ious


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Directories
    lidar_dir = os.path.join(DATA_DIR, "lidar")
    mask_dir = os.path.join(DATA_DIR, "masks")

    if not os.path.isdir(lidar_dir):
        # Maybe files are directly in DATA_DIR
        if glob.glob(os.path.join(DATA_DIR, "tile_*_lidar.tif")):
            lidar_dir = DATA_DIR
            mask_dir = DATA_DIR
        else:
            raise FileNotFoundError(
                f"Cannot find lidar tiles. Checked {lidar_dir} and {DATA_DIR}"
            )

    # Datasets
    train_ds = ChactunDataset(lidar_dir, mask_dir, split="train", augment=True)
    val_ds = ChactunDataset(lidar_dir, mask_dir, split="val", augment=False)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device != "cpu"),
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device != "cpu"),
    )

    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    ).to(device)

    # Class weights (inverse frequency: background very common, aguadas very rare)
    class_weights = torch.tensor([0.1, 1.0, 2.0, 50.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training: {len(train_ds)} tiles, Validation: {len(val_ds)} tiles")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Model: U-Net (ResNet34), Classes: 4")
    print()

    best_iou = 0.0
    class_names = {1: "building", 2: "platform", 3: "aguada"}

    for epoch in range(EPOCHS):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / max(len(train_dl), 1)

        # --- Validate ---
        model.eval()
        all_ious = {1: [], 2: [], 3: []}
        with torch.no_grad():
            for images, masks in tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images).argmax(dim=1)

                batch_ious = compute_iou_per_class(preds, masks)
                for c, iou in batch_ious.items():
                    all_ious[c].append(iou)

        # Per-class mean IoU
        class_ious = {}
        for c in range(1, 4):
            if all_ious[c]:
                class_ious[c] = np.mean(all_ious[c])
            else:
                class_ious[c] = 0.0

        mean_iou = np.mean(list(class_ious.values()))
        elapsed = time.time() - t0

        # Log
        iou_str = " | ".join(
            f"{class_names[c]}: {class_ious[c]:.4f}" for c in range(1, 4)
        )
        print(
            f"Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, "
            f"mIoU={mean_iou:.4f} [{iou_str}], "
            f"lr={scheduler.get_last_lr()[0]:.6f}, "
            f"time={elapsed:.0f}s"
        )

        # Save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  >>> Saved best model (mIoU={best_iou:.4f}) -> {SAVE_PATH}")

    print(f"\nTraining complete! Best validation mIoU: {best_iou:.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
