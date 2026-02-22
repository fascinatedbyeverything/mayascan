"""Multi-scale inference for improved detection quality.

Competition-winning approaches run inference at multiple tile sizes and
merge the results. Larger tiles capture context (large platforms), while
smaller tiles preserve fine detail (small aguadas). This module provides
a convenience wrapper around the standard detection + ensemble pipeline.
"""

from __future__ import annotations

import numpy as np

from mayascan.config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    MIN_BLOB_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
    V2_ARCH,
    V2_ENCODER,
)
from mayascan.detect import DetectionResult, run_detection_v2
from mayascan.ensemble import merge_results


#: Default multi-scale tile sizes (small, medium, large)
DEFAULT_SCALES = [320, 480, 640]


def run_multiscale_detection(
    visualization: np.ndarray,
    model_dir: str,
    scales: list[int] | None = None,
    arch: str = V2_ARCH,
    encoder: str = V2_ENCODER,
    overlap: float = TILE_OVERLAP,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_tta: bool = True,
    min_blob_size: int = MIN_BLOB_SIZE,
    merge_method: str = "average",
    device: str | None = None,
) -> DetectionResult:
    """Run detection at multiple tile sizes and merge results.

    Runs ``run_detection_v2`` at each scale in *scales*, then combines
    the results using the ensemble module. This typically improves IoU
    by 1-3% compared to single-scale inference.

    Parameters
    ----------
    visualization : np.ndarray
        Input raster with shape ``(C, H, W)``.
    model_dir : str
        Directory containing per-class model files.
    scales : list of int or None
        Tile sizes to use. If None, uses ``[320, 480, 640]``.
    arch : str
        Model architecture.
    encoder : str
        Encoder backbone.
    overlap : float
        Tile overlap fraction.
    confidence_threshold : float
        Confidence threshold after merging.
    use_tta : bool
        Whether to use test-time augmentation at each scale.
    min_blob_size : int
        Minimum blob size after post-processing.
    merge_method : str
        ``"average"`` for probability averaging or ``"vote"`` for majority voting.
    device : str or None
        Device for inference.

    Returns
    -------
    DetectionResult
        Merged detection result across all scales.
    """
    if scales is None:
        scales = list(DEFAULT_SCALES)

    if len(scales) == 1:
        return run_detection_v2(
            visualization,
            model_dir=model_dir,
            arch=arch,
            encoder=encoder,
            tile_size=scales[0],
            overlap=overlap,
            confidence_threshold=confidence_threshold,
            use_tta=use_tta,
            min_blob_size=min_blob_size,
            device=device,
        )

    results = []
    for scale in scales:
        result = run_detection_v2(
            visualization,
            model_dir=model_dir,
            arch=arch,
            encoder=encoder,
            tile_size=scale,
            overlap=overlap,
            confidence_threshold=confidence_threshold,
            use_tta=use_tta,
            min_blob_size=min_blob_size,
            device=device,
        )
        results.append(result)

    merged = merge_results(
        results,
        method=merge_method,
        confidence_threshold=confidence_threshold,
    )
    return merged
