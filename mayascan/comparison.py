"""Comparison of detection results across models or time periods.

Provides tools for computing detection differences, agreement maps,
and change analysis — useful for model evaluation and temporal
monitoring of archaeological sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label

from mayascan.detect import DetectionResult


@dataclass
class ComparisonResult:
    """Result of comparing two detection maps.

    Attributes
    ----------
    agreement : np.ndarray
        Boolean mask where both detections agree (same non-background class).
    only_a : np.ndarray
        Boolean mask for pixels detected only in result A.
    only_b : np.ndarray
        Boolean mask for pixels detected only in result B.
    changed_class : np.ndarray
        Boolean mask for pixels where both detect a feature but different class.
    jaccard : float
        Jaccard index (IoU) between the two detection maps.
    dice : float
        Dice coefficient (F1) between the two detection maps.
    per_class_jaccard : dict[str, float]
        Per-class Jaccard index.
    """

    agreement: np.ndarray
    only_a: np.ndarray
    only_b: np.ndarray
    changed_class: np.ndarray
    jaccard: float
    dice: float
    per_class_jaccard: dict[str, float]


def compare_detections(
    result_a: DetectionResult,
    result_b: DetectionResult,
) -> ComparisonResult:
    """Compare two detection results pixel-by-pixel.

    Parameters
    ----------
    result_a : DetectionResult
        First detection result (e.g., model A or time period 1).
    result_b : DetectionResult
        Second detection result (e.g., model B or time period 2).

    Returns
    -------
    ComparisonResult
        Detailed comparison with agreement, differences, and metrics.
    """
    ca = result_a.classes
    cb = result_b.classes

    mask_a = ca > 0
    mask_b = cb > 0

    agreement = (ca == cb) & mask_a
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    changed_class = mask_a & mask_b & (ca != cb)

    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    jaccard = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / (mask_a.sum() + mask_b.sum())) if (mask_a.sum() + mask_b.sum()) > 0 else 1.0

    class_names = result_a.class_names
    per_class: dict[str, float] = {}
    for cls_id, cls_name in class_names.items():
        if cls_id == 0:
            continue
        a_cls = ca == cls_id
        b_cls = cb == cls_id
        inter = (a_cls & b_cls).sum()
        uni = (a_cls | b_cls).sum()
        per_class[cls_name] = float(inter / uni) if uni > 0 else 1.0

    return ComparisonResult(
        agreement=agreement,
        only_a=only_a,
        only_b=only_b,
        changed_class=changed_class,
        jaccard=round(jaccard, 4),
        dice=round(dice, 4),
        per_class_jaccard={k: round(v, 4) for k, v in per_class.items()},
    )


def comparison_summary(comp: ComparisonResult) -> dict[str, Any]:
    """Generate a summary of a comparison result.

    Parameters
    ----------
    comp : ComparisonResult
        Comparison output from :func:`compare_detections`.

    Returns
    -------
    dict
        Summary with pixel counts, percentages, and metrics.
    """
    agreement_px = int(comp.agreement.sum())
    only_a_px = int(comp.only_a.sum())
    only_b_px = int(comp.only_b.sum())
    changed_px = int(comp.changed_class.sum())
    detected_px = agreement_px + only_a_px + only_b_px + changed_px

    return {
        "agreement_pixels": agreement_px,
        "only_a_pixels": only_a_px,
        "only_b_pixels": only_b_px,
        "changed_class_pixels": changed_px,
        "total_detected_pixels": detected_px,
        "agreement_pct": round(100 * agreement_px / detected_px, 2) if detected_px > 0 else 100.0,
        "jaccard": comp.jaccard,
        "dice": comp.dice,
        "per_class_jaccard": comp.per_class_jaccard,
    }


def difference_map(
    result_a: DetectionResult,
    result_b: DetectionResult,
) -> np.ndarray:
    """Create a color-coded difference map.

    Returns an (H, W, 4) RGBA image where:
    - Green = agreement (both detect same class)
    - Red = only in A (lost detections if B is newer)
    - Blue = only in B (new detections if B is newer)
    - Yellow = class changed

    Parameters
    ----------
    result_a : DetectionResult
        First result.
    result_b : DetectionResult
        Second result.

    Returns
    -------
    np.ndarray
        RGBA uint8 image (H, W, 4).
    """
    comp = compare_detections(result_a, result_b)
    h, w = comp.agreement.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    rgba[comp.agreement] = [60, 200, 60, 200]
    rgba[comp.only_a] = [255, 60, 60, 200]
    rgba[comp.only_b] = [60, 120, 255, 200]
    rgba[comp.changed_class] = [255, 255, 60, 200]

    return rgba


def count_feature_changes(
    result_a: DetectionResult,
    result_b: DetectionResult,
) -> dict[str, Any]:
    """Count features gained and lost between two detections.

    Parameters
    ----------
    result_a : DetectionResult
        Earlier detection result.
    result_b : DetectionResult
        Later detection result.

    Returns
    -------
    dict
        Feature counts per class: gained, lost, stable.
    """
    class_names = result_a.class_names
    changes: dict[str, dict[str, int]] = {}

    for cls_id, cls_name in class_names.items():
        if cls_id == 0:
            continue

        mask_a = result_a.classes == cls_id
        mask_b = result_b.classes == cls_id

        _, n_a = label(mask_a) if mask_a.any() else (None, 0)
        _, n_b = label(mask_b) if mask_b.any() else (None, 0)

        overlap = mask_a & mask_b
        if overlap.any():
            _, n_overlap = label(overlap)
            stable = n_overlap
        else:
            stable = 0

        changes[cls_name] = {
            "count_a": n_a,
            "count_b": n_b,
            "stable": stable,
            "gained": max(0, n_b - stable),
            "lost": max(0, n_a - stable),
        }

    return changes
