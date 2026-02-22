"""Model ensemble utilities for combining multiple detection results.

Competition-winning approaches use ensembles of different architectures
(DeepLabV3+, UNet++, HRNet) to improve segmentation quality. This module
provides tools for combining multiple detection results via probability
averaging or voting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mayascan.config import CLASS_NAMES
from mayascan.detect import DetectionResult


def average_probabilities(
    prob_maps: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Average probability maps from multiple models.

    Parameters
    ----------
    prob_maps : list of np.ndarray
        Each array has shape (num_classes, H, W) with values in [0, 1].
    weights : list of float or None
        Optional per-model weights. If None, equal weighting is used.

    Returns
    -------
    np.ndarray
        Averaged probability map, shape (num_classes, H, W).
    """
    if not prob_maps:
        raise ValueError("At least one probability map is required")

    if weights is None:
        weights = [1.0 / len(prob_maps)] * len(prob_maps)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    result = np.zeros_like(prob_maps[0], dtype=np.float64)
    for prob, w in zip(prob_maps, weights):
        result += prob.astype(np.float64) * w

    return result.astype(np.float32)


def majority_vote(
    class_maps: list[np.ndarray],
    num_classes: int = 4,
) -> np.ndarray:
    """Combine class predictions via per-pixel majority voting.

    Parameters
    ----------
    class_maps : list of np.ndarray
        Each array has shape (H, W) with integer class indices.
    num_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray
        Voted class map, shape (H, W), dtype int64.
    """
    if not class_maps:
        raise ValueError("At least one class map is required")

    h, w = class_maps[0].shape
    votes = np.zeros((num_classes, h, w), dtype=np.int32)

    for cmap in class_maps:
        for cls_id in range(num_classes):
            votes[cls_id] += (cmap == cls_id).astype(np.int32)

    return np.argmax(votes, axis=0).astype(np.int64)


def merge_results(
    results: list[DetectionResult],
    method: str = "average",
    weights: list[float] | None = None,
    confidence_threshold: float = 0.5,
) -> DetectionResult:
    """Merge multiple DetectionResults into one ensemble result.

    Parameters
    ----------
    results : list of DetectionResult
        Detection results from different models.
    method : str
        ``"average"`` for probability averaging, ``"vote"`` for majority voting.
    weights : list of float or None
        Per-model weights (only used with ``"average"``).
    confidence_threshold : float
        Threshold applied after merging (for ``"average"`` method).

    Returns
    -------
    DetectionResult
        Merged detection result.
    """
    if not results:
        raise ValueError("At least one result is required")

    if len(results) == 1:
        return results[0]

    # Get class names and geo from first result
    class_names = results[0].class_names
    geo = results[0].geo

    if method == "vote":
        classes = majority_vote([r.classes for r in results], len(class_names))
        # Average the confidence maps
        confidence = np.mean(
            [r.confidence for r in results], axis=0
        ).astype(np.float32)
    elif method == "average":
        # Build per-class probability maps from each result
        h, w = results[0].classes.shape
        n_classes = len(class_names)
        prob_maps = []

        for r in results:
            # Reconstruct approximate probability from classes + confidence
            prob = np.zeros((n_classes, h, w), dtype=np.float32)
            for cls_id in range(n_classes):
                mask = r.classes == cls_id
                prob[cls_id][mask] = r.confidence[mask]
            # Background gets remaining probability
            fg_max = np.max(prob[1:], axis=0) if n_classes > 1 else np.zeros((h, w))
            prob[0] = np.where(r.classes == 0, r.confidence, 1.0 - fg_max)
            prob_maps.append(prob)

        merged_prob = average_probabilities(prob_maps, weights)
        classes = np.argmax(merged_prob, axis=0).astype(np.int64)
        confidence = np.max(merged_prob, axis=0).astype(np.float32)

        # Apply threshold
        for cls_id in range(1, n_classes):
            cls_mask = classes == cls_id
            low_conf = merged_prob[cls_id] < confidence_threshold
            classes[cls_mask & low_conf] = 0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average' or 'vote'.")

    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names=class_names,
        geo=geo,
    )
