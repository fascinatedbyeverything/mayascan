"""Dataset loading for archaeological segmentation.

Provides a reusable PyTorch Dataset for per-class binary segmentation,
with oversampling for rare classes and integration with the augment module.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch is required for data loading: pip install torch")

from mayascan.augment import augment_sample
from mayascan.config import V2_CLASSES


class BinarySegmentationDataset(Dataset):
    """Per-class binary segmentation dataset.

    Competition insight: per-class binary models outperform multi-class when
    dealing with extreme class imbalance (e.g. aguadas at 0.3% of pixels).

    Parameters
    ----------
    lidar_dir : str
        Directory containing ``tile_*_lidar.tif`` files.
    mask_dir : str
        Directory containing ``tile_*_mask_{class_name}.tif`` files.
    cls_name : str
        Target class name (e.g. ``"building"``, ``"platform"``, ``"aguada"``).
    split : str
        ``"train"`` or ``"val"``. Uses 80/20 split.
    augment : bool
        Whether to apply augmentation (only effective for ``"train"``).
    oversample_positive : bool
        Whether to duplicate tiles containing the target class.
    max_oversample : int
        Maximum oversampling factor for positive tiles.
    val_fraction : float
        Fraction of tiles reserved for validation.
    """

    def __init__(
        self,
        lidar_dir: str,
        mask_dir: str,
        cls_name: str,
        split: str = "train",
        augment: bool = True,
        oversample_positive: bool = True,
        max_oversample: int = 6,
        val_fraction: float = 0.2,
        tile_paths: list[str] | None = None,
    ):
        self.cls_name = cls_name
        self.mask_dir = mask_dir
        self.augment = augment and (split == "train")

        if tile_paths is not None:
            # Explicit tile list (for fold-based training)
            tiles = list(tile_paths)
        else:
            all_lidar = sorted(glob.glob(os.path.join(lidar_dir, "tile_*_lidar.tif")))
            if not all_lidar:
                raise FileNotFoundError(f"No lidar tiles in {lidar_dir}")

            n = len(all_lidar)
            split_idx = int(n * (1 - val_fraction))
            tiles = all_lidar[:split_idx] if split == "train" else all_lidar[split_idx:]

        self.tiles: list[str] = []
        self.has_positive: list[bool] = []
        for t in tiles:
            tid = _extract_tile_id(t)
            mask_path = os.path.join(mask_dir, f"tile_{tid}_mask_{cls_name}.tif")
            has_pos = False
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path))
                has_pos = bool((m < 128).any())
            self.tiles.append(t)
            self.has_positive.append(has_pos)

        pos_count = sum(self.has_positive)
        neg_count = len(self.tiles) - pos_count

        if oversample_positive and split == "train" and pos_count > 0:
            ratio = max(1, neg_count // max(pos_count, 1))
            ratio = min(ratio, max_oversample)
            if ratio > 1:
                extra = []
                for i, has_pos in enumerate(self.has_positive):
                    if has_pos:
                        extra.extend([self.tiles[i]] * (ratio - 1))
                self.tiles.extend(extra)
                self.has_positive.extend([True] * len(extra))

        self._pos_count = pos_count
        self._neg_count = neg_count
        self._split = split

    @property
    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        return {
            "class": self.cls_name,
            "split": self._split,
            "total_tiles": len(self.tiles),
            "positive_tiles": self._pos_count,
            "negative_tiles": self._neg_count,
        }

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lidar_path = self.tiles[idx]
        tile_id = _extract_tile_id(lidar_path)

        img = np.array(Image.open(lidar_path), dtype=np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=0)
        else:
            img = img.transpose(2, 0, 1)

        h, w = img.shape[1], img.shape[2]
        mask_path = os.path.join(
            self.mask_dir, f"tile_{tile_id}_mask_{self.cls_name}.tif"
        )
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path), dtype=np.float32)
            if m.ndim > 2:
                m = m[:, :, 0]
            mask = (m < 128).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        if self.augment:
            img, mask = augment_sample(img, mask, use_elastic=True)

        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())


def _extract_tile_id(path: str) -> str:
    """Extract numeric tile ID from a tile filename."""
    return os.path.basename(path).replace("tile_", "").replace("_lidar.tif", "")


def list_available_classes(mask_dir: str) -> list[str]:
    """Discover which class masks are available in a directory.

    Parameters
    ----------
    mask_dir : str
        Directory containing mask files.

    Returns
    -------
    list of str
        Sorted list of class names found.
    """
    masks = glob.glob(os.path.join(mask_dir, "tile_*_mask_*.tif"))
    classes = set()
    for m in masks:
        base = os.path.basename(m)
        # tile_0001_mask_building.tif -> building
        parts = base.replace(".tif", "").split("_mask_")
        if len(parts) == 2:
            classes.add(parts[1])
    return sorted(classes)


def count_tiles(lidar_dir: str) -> int:
    """Count the number of lidar tiles in a directory.

    Parameters
    ----------
    lidar_dir : str
        Directory containing ``tile_*_lidar.tif`` files.

    Returns
    -------
    int
        Number of tiles found.
    """
    return len(glob.glob(os.path.join(lidar_dir, "tile_*_lidar.tif")))
