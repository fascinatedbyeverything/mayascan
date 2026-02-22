"""Metrics and evaluation utilities for detection results.

Computes per-class IoU, precision, recall, F1, and confusion matrices
for comparing predictions against ground truth masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassMetrics:
    """Accumulated pixel-level confusion matrix counts for a single class."""

    tp: int = 0  # true positive pixels
    fp: int = 0  # false positive pixels
    fn: int = 0  # false negative pixels
    tn: int = 0  # true negative pixels

    @property
    def iou(self) -> float:
        """Intersection over Union."""
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a dictionary."""
        return {
            "iou": round(self.iou, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> ClassMetrics:
    """Compute binary classification metrics between prediction and target.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (0 or 1).
    target : np.ndarray
        Ground truth binary mask (0 or 1).

    Returns
    -------
    ClassMetrics
        Accumulated TP/FP/FN/TN counts.
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    return ClassMetrics(
        tp=int((pred_bool & target_bool).sum()),
        fp=int((pred_bool & ~target_bool).sum()),
        fn=int((~pred_bool & target_bool).sum()),
        tn=int((~pred_bool & ~target_bool).sum()),
    )


def compute_multiclass_metrics(
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> dict[int, ClassMetrics]:
    """Compute per-class metrics for a multi-class segmentation.

    Parameters
    ----------
    pred_classes : np.ndarray
        Predicted class indices (H, W).
    target_classes : np.ndarray
        Ground truth class indices (H, W).
    class_names : dict or None
        Mapping from class ID to name. If None, auto-detected from arrays.

    Returns
    -------
    dict[int, ClassMetrics]
        Per-class metrics (excludes background class 0).
    """
    if class_names is None:
        all_ids = set(np.unique(pred_classes)) | set(np.unique(target_classes))
        class_names = {i: f"class_{i}" for i in all_ids}

    metrics: dict[int, ClassMetrics] = {}
    for cls_id in class_names:
        if cls_id == 0:
            continue
        pred_mask = pred_classes == cls_id
        target_mask = target_classes == cls_id
        metrics[cls_id] = compute_binary_metrics(pred_mask, target_mask)

    return metrics


def mean_iou(metrics: dict[int, ClassMetrics]) -> float:
    """Compute mean IoU across all classes.

    Parameters
    ----------
    metrics : dict[int, ClassMetrics]
        Per-class metrics.

    Returns
    -------
    float
        Mean IoU.
    """
    ious = [m.iou for m in metrics.values()]
    return sum(ious) / len(ious) if ious else 0.0


def confusion_matrix(
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """Compute a confusion matrix.

    Parameters
    ----------
    pred_classes : np.ndarray
        Predicted class indices (H, W).
    target_classes : np.ndarray
        Ground truth class indices (H, W).
    num_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Shape (num_classes, num_classes). Row = target, column = prediction.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            cm[true_cls, pred_cls] = int(
                ((target_classes == true_cls) & (pred_classes == pred_cls)).sum()
            )
    return cm


def format_metrics_table(
    metrics: dict[int, ClassMetrics],
    class_names: dict[int, str] | None = None,
) -> str:
    """Format per-class metrics as a readable table string.

    Parameters
    ----------
    metrics : dict[int, ClassMetrics]
        Per-class metrics.
    class_names : dict or None
        Optional class name mapping.

    Returns
    -------
    str
        Formatted table.
    """
    if class_names is None:
        class_names = {i: f"class_{i}" for i in metrics}

    header = f"{'Class':>12s}  {'IoU':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}"
    sep = "-" * len(header)

    lines = [header, sep]
    ious = []
    f1s = []

    for cls_id in sorted(metrics):
        m = metrics[cls_id]
        name = class_names.get(cls_id, f"class_{cls_id}")
        lines.append(
            f"{name:>12s}  {m.iou:8.4f}  {m.precision:8.4f}  "
            f"{m.recall:8.4f}  {m.f1:8.4f}"
        )
        ious.append(m.iou)
        f1s.append(m.f1)

    lines.append(sep)
    n = len(ious)
    if n > 0:
        lines.append(
            f"{'Mean':>12s}  {sum(ious)/n:8.4f}  {'':>8s}  "
            f"{'':>8s}  {sum(f1s)/n:8.4f}"
        )

    return "\n".join(lines)
