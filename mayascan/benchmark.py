"""Benchmark inference speed for different configurations.

Measures throughput and latency to help users choose optimal settings
for their hardware and accuracy requirements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mayascan.config import (
    CONFIDENCE_THRESHOLD,
    MIN_BLOB_SIZE,
    TILE_SIZE,
    V2_ARCH,
    V2_ENCODER,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes
    ----------
    config_name : str
        Human-readable configuration label.
    image_size : tuple of int
        Input image dimensions (H, W).
    elapsed_seconds : float
        Total wall-clock time.
    pixels_per_second : float
        Throughput in pixels/second.
    num_features : int
        Number of features detected.
    """

    config_name: str
    image_size: tuple[int, int]
    elapsed_seconds: float
    pixels_per_second: float
    num_features: int


def run_benchmark(
    visualization: np.ndarray,
    model_dir: str,
    configs: list[dict] | None = None,
    device: str | None = None,
) -> list[BenchmarkResult]:
    """Benchmark multiple inference configurations.

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)``.
    model_dir : str
        Directory containing per-class model files.
    configs : list of dict or None
        List of configuration dicts. Each should have a ``"name"`` key
        and any kwargs accepted by ``run_detection_v2`` or
        ``run_multiscale_detection``. If None, uses default configs.
    device : str or None
        Device for inference.

    Returns
    -------
    list of BenchmarkResult
        Benchmark results for each configuration.
    """
    from mayascan.detect import run_detection_v2
    from mayascan.multiscale import run_multiscale_detection
    from scipy.ndimage import label

    _, H, W = visualization.shape

    if configs is None:
        configs = [
            {"name": "v2 no-TTA", "use_tta": False},
            {"name": "v2 TTA", "use_tta": True},
            {"name": "v2 multiscale", "multiscale": True, "use_tta": True},
        ]

    results = []
    for cfg in configs:
        name = cfg.pop("name", "unnamed")
        is_multiscale = cfg.pop("multiscale", False)

        t0 = time.time()
        if is_multiscale:
            result = run_multiscale_detection(
                visualization, model_dir=model_dir, device=device, **cfg
            )
        else:
            result = run_detection_v2(
                visualization, model_dir=model_dir, device=device, **cfg
            )
        elapsed = time.time() - t0

        # Count features
        num_features = 0
        for cls_id in range(1, 4):
            mask = result.classes == cls_id
            if mask.any():
                _, n = label(mask)
                num_features += n

        pixels = H * W
        results.append(
            BenchmarkResult(
                config_name=name,
                image_size=(H, W),
                elapsed_seconds=elapsed,
                pixels_per_second=pixels / elapsed if elapsed > 0 else 0,
                num_features=num_features,
            )
        )

    return results


def format_benchmark(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a human-readable table.

    Parameters
    ----------
    results : list of BenchmarkResult
        Results from :func:`run_benchmark`.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = [
        f"{'Configuration':<25s} {'Time':>8s} {'Mpx/s':>8s} {'Features':>10s}",
        "-" * 55,
    ]
    for r in results:
        mpx = r.pixels_per_second / 1e6
        lines.append(
            f"{r.config_name:<25s} {r.elapsed_seconds:>7.1f}s {mpx:>7.2f} {r.num_features:>10d}"
        )
    return "\n".join(lines)
