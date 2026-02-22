"""Tests for mayascan.metrics — evaluation metrics."""

import numpy as np

from mayascan.metrics import (
    ClassMetrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
    confusion_matrix,
    format_metrics_table,
    mean_iou,
)


class TestClassMetrics:
    def test_perfect_prediction(self):
        m = ClassMetrics(tp=100, fp=0, fn=0, tn=900)
        assert m.iou == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_no_predictions(self):
        m = ClassMetrics(tp=0, fp=0, fn=100, tn=900)
        assert m.iou == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_positives(self):
        m = ClassMetrics(tp=0, fp=100, fn=0, tn=900)
        assert m.iou == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_partial(self):
        m = ClassMetrics(tp=50, fp=10, fn=20, tn=920)
        assert 0 < m.iou < 1
        assert m.precision == 50 / 60
        assert m.recall == 50 / 70

    def test_to_dict(self):
        m = ClassMetrics(tp=50, fp=10, fn=20, tn=920)
        d = m.to_dict()
        assert "iou" in d
        assert "precision" in d
        assert "f1" in d
        assert d["tp"] == 50


class TestComputeBinaryMetrics:
    def test_exact_match(self):
        pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        target = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        m = compute_binary_metrics(pred, target)
        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 2

    def test_no_overlap(self):
        pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        target = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        m = compute_binary_metrics(pred, target)
        assert m.tp == 0
        assert m.fp == 2
        assert m.fn == 2
        assert m.tn == 0


class TestComputeMulticlassMetrics:
    def test_perfect_multiclass(self):
        pred = np.array([[0, 1], [2, 3]], dtype=np.int64)
        target = np.array([[0, 1], [2, 3]], dtype=np.int64)
        class_names = {0: "bg", 1: "a", 2: "b", 3: "c"}
        metrics = compute_multiclass_metrics(pred, target, class_names)
        for cls_id, m in metrics.items():
            assert m.iou == 1.0

    def test_miou(self):
        pred = np.array([[0, 1], [2, 3]], dtype=np.int64)
        target = np.array([[0, 1], [2, 3]], dtype=np.int64)
        class_names = {0: "bg", 1: "a", 2: "b", 3: "c"}
        metrics = compute_multiclass_metrics(pred, target, class_names)
        assert mean_iou(metrics) == 1.0


class TestConfusionMatrix:
    def test_perfect(self):
        pred = np.array([[0, 1], [2, 3]], dtype=np.int64)
        target = np.array([[0, 1], [2, 3]], dtype=np.int64)
        cm = confusion_matrix(pred, target, num_classes=4)
        assert cm.shape == (4, 4)
        assert np.all(np.diag(cm) == 1)  # one pixel per class
        assert cm.sum() == 4

    def test_misclassification(self):
        pred = np.array([[1, 1], [1, 1]], dtype=np.int64)
        target = np.array([[0, 0], [0, 0]], dtype=np.int64)
        cm = confusion_matrix(pred, target, num_classes=4)
        assert cm[0, 1] == 4  # all bg classified as class 1


class TestFormatMetricsTable:
    def test_produces_string(self):
        metrics = {1: ClassMetrics(50, 10, 20, 920)}
        class_names = {1: "building"}
        table = format_metrics_table(metrics, class_names)
        assert "building" in table
        assert "IoU" in table
        assert "Mean" in table
