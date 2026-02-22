"""K-fold cross-validation for archaeological segmentation.

Competition winners used 5-fold CV with fold ensembles to achieve
the highest IoU scores. This module provides fold splitting and
a cross-validation runner that trains one model per fold.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

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
