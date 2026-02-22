"""Tests for mayascan.benchmark — inference benchmarking."""

import numpy as np
import pytest

from mayascan.benchmark import BenchmarkResult, format_benchmark, run_benchmark


class TestBenchmarkResult:
    def test_fields(self):
        r = BenchmarkResult(
            config_name="test",
            image_size=(100, 200),
            elapsed_seconds=1.5,
            pixels_per_second=13333.3,
            num_features=42,
        )
        assert r.config_name == "test"
        assert r.image_size == (100, 200)
        assert r.elapsed_seconds == 1.5


class TestFormatBenchmark:
    def test_produces_table(self):
        results = [
            BenchmarkResult("fast", (100, 100), 0.5, 20000, 10),
            BenchmarkResult("slow", (100, 100), 2.0, 5000, 15),
        ]
        table = format_benchmark(results)
        assert "fast" in table
        assert "slow" in table
        assert "Configuration" in table

    def test_single_result(self):
        results = [BenchmarkResult("only", (50, 50), 1.0, 2500, 5)]
        table = format_benchmark(results)
        assert "only" in table


class TestRunBenchmark:
    def test_with_mock(self, tmp_path, monkeypatch):
        import mayascan.benchmark as bm_mod

        call_count = [0]

        def mock_detect(visualization, **kwargs):
            from mayascan.detect import DetectionResult
            from mayascan.config import CLASS_NAMES

            call_count[0] += 1
            h, w = visualization.shape[1], visualization.shape[2]
            classes = np.zeros((h, w), dtype=np.int64)
            classes[5:15, 5:15] = 1
            return DetectionResult(
                classes=classes,
                confidence=np.full((h, w), 0.8, dtype=np.float32),
                class_names=dict(CLASS_NAMES),
            )

        def mock_multiscale(visualization, **kwargs):
            return mock_detect(visualization, **kwargs)

        import importlib

        det_mod = importlib.import_module("mayascan.detect")
        ms_mod = importlib.import_module("mayascan.multiscale")

        monkeypatch.setattr(det_mod, "run_detection_v2", mock_detect)
        monkeypatch.setattr(ms_mod, "run_multiscale_detection", mock_multiscale)

        viz = np.random.rand(3, 32, 32).astype(np.float32)
        results = run_benchmark(
            viz, model_dir=str(tmp_path),
            configs=[
                {"name": "basic", "use_tta": False},
                {"name": "multi", "multiscale": True},
            ],
        )
        assert len(results) == 2
        assert results[0].config_name == "basic"
        assert results[1].config_name == "multi"
        assert all(r.elapsed_seconds > 0 for r in results)
        assert all(r.num_features > 0 for r in results)
