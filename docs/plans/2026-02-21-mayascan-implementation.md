# MayaScan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an open-source pipeline that takes raw LiDAR point clouds and outputs annotated maps of probable ancient Maya structures using deep learning.

**Architecture:** Four-stage pipeline (ingest → ground classification → raster visualization → AI detection) exposed as both a `pip install`-able Python library and a Gradio web app. U-Net with ResNet34 encoder trained on the Chactún ML-ready dataset (9,303 buildings, 2,110 platforms, 95 aguadas).

**Tech Stack:** Python 3.12, PyTorch, segmentation-models-pytorch, PDAL, rvt-py, rasterio, Gradio, geopandas

---

### Task 1: Project Scaffolding + Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `mayascan/__init__.py`
- Create: `LICENSE`
- Create: `README.md`
- Create: `tests/__init__.py`

**Step 1: Initialize git repo**

```bash
cd /Volumes/macos4tb/Projects/mayascan
git init
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mayascan"
version = "0.1.0"
description = "Open-source archaeological LiDAR feature detection for Maya archaeology"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    {name = "Chris Holmes"},
]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "rasterio>=1.3",
    "geopandas>=0.13",
    "shapely>=2.0",
    "torch>=2.0",
    "torchvision>=0.15",
    "segmentation-models-pytorch>=0.3.3",
    "rvt-py>=2.2",
    "gradio>=4.0",
    "Pillow>=10.0",
    "tqdm>=4.65",
    "huggingface-hub>=0.20",
]

[project.optional-dependencies]
pdal = ["python-pdal>=3.2"]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]
all = ["mayascan[pdal,dev]"]

[project.scripts]
mayascan = "mayascan.cli:main"

[tool.setuptools.packages.find]
include = ["mayascan*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Note: PDAL is optional because it requires conda or system-level install on many platforms. The web app can accept pre-processed DEMs if PDAL isn't available.

**Step 3: Create package init**

Create `mayascan/__init__.py`:
```python
"""MayaScan: Open-source archaeological LiDAR feature detection."""

__version__ = "0.1.0"
```

**Step 4: Create LICENSE (MIT)**

Standard MIT license text.

**Step 5: Create minimal README.md**

```markdown
# MayaScan

Open-source archaeological LiDAR feature detection for Maya archaeology.

Upload a LiDAR scan, get back a map of probable ancient structures.

## Quick Start

```bash
pip install mayascan
```

```python
import mayascan
results = mayascan.process("survey.laz")
```

## License

MIT
```

**Step 6: Create tests/__init__.py**

Empty file.

**Step 7: Install dev dependencies**

```bash
cd /Volumes/macos4tb/Projects/mayascan
pip3 install -e ".[dev]"
```

Note: Skip `[pdal]` for now — PDAL requires conda. We'll handle that separately. If `rvt-py` fails to install, install it individually: `pip3 install rvt-py`.

**Step 8: Verify pytest runs**

```bash
cd /Volumes/macos4tb/Projects/mayascan
python3 -m pytest tests/ -v
```

Expected: "no tests ran" (0 collected), exit 5, but no import errors.

**Step 9: Commit**

```bash
git add pyproject.toml mayascan/__init__.py LICENSE README.md tests/__init__.py docs/
git commit -m "feat: scaffold MayaScan project with dependencies and design docs"
```

---

### Task 2: Tiling Module (tile.py)

The tiling module is foundational — slicing large rasters into model-sized patches and stitching results back. No external geo-dependencies beyond numpy, so we start here.

**Files:**
- Create: `mayascan/tile.py`
- Create: `tests/test_tile.py`

**Step 1: Write the failing tests**

Create `tests/test_tile.py`:
```python
import numpy as np
from mayascan.tile import slice_tiles, stitch_tiles


def test_slice_tiles_no_overlap():
    """A 960x960 image should produce 4 tiles of 480x480 with no overlap."""
    image = np.arange(960 * 960, dtype=np.float32).reshape(1, 960, 960)
    tiles, origins = slice_tiles(image, tile_size=480, overlap=0)
    assert len(tiles) == 4
    assert all(t.shape == (1, 480, 480) for t in tiles)
    assert origins == [(0, 0), (0, 480), (480, 0), (480, 480)]


def test_slice_tiles_with_overlap():
    """With 50% overlap, a 960x960 image produces overlapping tiles."""
    image = np.random.rand(3, 960, 960).astype(np.float32)
    tiles, origins = slice_tiles(image, tile_size=480, overlap=0.5)
    # 50% overlap = stride of 240. Positions: 0, 240, 480 in each dim = 3x3 = 9
    assert len(tiles) == 9
    assert all(t.shape == (3, 480, 480) for t in tiles)


def test_slice_tiles_pads_remainder():
    """An image not divisible by tile_size should be padded."""
    image = np.ones((1, 500, 700), dtype=np.float32)
    tiles, origins = slice_tiles(image, tile_size=480, overlap=0)
    # Should pad to at least 480x480 and cover the whole image
    assert len(tiles) >= 2
    assert all(t.shape == (1, 480, 480) for t in tiles)


def test_stitch_tiles_roundtrip():
    """Slicing and stitching should recover the original image."""
    original = np.random.rand(3, 960, 960).astype(np.float32)
    tiles, origins = slice_tiles(original, tile_size=480, overlap=0)
    # Simulate single-channel output masks (class predictions)
    masks = [np.random.rand(4, 480, 480).astype(np.float32) for _ in tiles]
    stitched = stitch_tiles(masks, origins, output_shape=(4, 960, 960), overlap=0)
    assert stitched.shape == (4, 960, 960)


def test_stitch_tiles_overlap_blending():
    """Overlapping regions should be averaged (blended)."""
    # Create uniform tiles — stitching overlaps should still produce uniform output
    output_shape = (1, 960, 960)
    tiles_data = [np.ones((1, 480, 480), dtype=np.float32) * 5.0] * 9
    origins = []
    for r in range(0, 960, 240):
        for c in range(0, 960, 240):
            if r + 480 <= 960 and c + 480 <= 960:
                origins.append((r, c))
    tiles_data = tiles_data[:len(origins)]
    stitched = stitch_tiles(tiles_data, origins, output_shape=output_shape, overlap=0.5)
    np.testing.assert_allclose(stitched, 5.0, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_tile.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mayascan.tile'`

**Step 3: Implement tile.py**

Create `mayascan/tile.py`:
```python
"""Tile slicing and stitching for large raster inference."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def slice_tiles(
    image: np.ndarray,
    tile_size: int = 480,
    overlap: float = 0.0,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Slice a (C, H, W) image into tiles.

    Args:
        image: Array of shape (C, H, W).
        tile_size: Size of each square tile in pixels.
        overlap: Fraction of overlap between adjacent tiles (0.0 to 0.5).

    Returns:
        tiles: List of (C, tile_size, tile_size) arrays.
        origins: List of (row, col) top-left positions in the original image.
    """
    _, h, w = image.shape
    stride = max(1, int(tile_size * (1.0 - overlap)))

    # Pad image so tiles cover the entire area
    pad_h = max(0, tile_size - h) if h < tile_size else (stride - ((h - tile_size) % stride)) % stride
    pad_w = max(0, tile_size - w) if w < tile_size else (stride - ((w - tile_size) % stride)) % stride
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    _, h_pad, w_pad = image.shape
    tiles = []
    origins = []

    for r in range(0, h_pad - tile_size + 1, stride):
        for c in range(0, w_pad - tile_size + 1, stride):
            tile = image[:, r : r + tile_size, c : c + tile_size]
            tiles.append(tile)
            origins.append((r, c))

    return tiles, origins


def stitch_tiles(
    tiles: List[np.ndarray],
    origins: List[Tuple[int, int]],
    output_shape: Tuple[int, int, int],
    overlap: float = 0.0,
) -> np.ndarray:
    """Stitch tiles back into a full image, averaging overlapping regions.

    Args:
        tiles: List of (C, H, W) tile arrays.
        origins: List of (row, col) positions matching each tile.
        output_shape: (C, H, W) shape of the output image.
        overlap: Overlap fraction used during slicing (for documentation).

    Returns:
        Stitched array of shape output_shape.
    """
    output = np.zeros(output_shape, dtype=np.float32)
    counts = np.zeros(output_shape, dtype=np.float32)

    for tile, (r, c) in zip(tiles, origins):
        _, th, tw = tile.shape
        out_r = min(r, output_shape[1] - th)
        out_c = min(c, output_shape[2] - tw)
        h_end = out_r + th
        w_end = out_c + tw
        # Clip to output bounds
        h_end = min(h_end, output_shape[1])
        w_end = min(w_end, output_shape[2])
        actual_h = h_end - out_r
        actual_w = w_end - out_c
        output[:, out_r:h_end, out_c:w_end] += tile[:, :actual_h, :actual_w]
        counts[:, out_r:h_end, out_c:w_end] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    output /= counts

    return output
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_tile.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add mayascan/tile.py tests/test_tile.py
git commit -m "feat: add tile slicing and stitching module"
```

---

### Task 3: Visualization Module (visualize.py)

Generate the three raster channels (SVF, openness, slope) from a DEM using rvt-py.

**Files:**
- Create: `mayascan/visualize.py`
- Create: `tests/test_visualize.py`

**Step 1: Write the failing tests**

Create `tests/test_visualize.py`:
```python
import numpy as np
import pytest
from mayascan.visualize import compute_slope, compute_visualizations


def test_compute_slope_flat():
    """A flat DEM should produce near-zero slope everywhere."""
    dem = np.ones((100, 100), dtype=np.float32) * 50.0
    slope = compute_slope(dem, resolution=0.5)
    assert slope.shape == (100, 100)
    np.testing.assert_allclose(slope, 0.0, atol=1e-3)


def test_compute_slope_ramp():
    """A linear ramp DEM should produce uniform non-zero slope."""
    dem = np.tile(np.arange(100, dtype=np.float32), (100, 1))
    slope = compute_slope(dem, resolution=1.0)
    assert slope.shape == (100, 100)
    # Interior points should have consistent slope
    interior = slope[1:-1, 1:-1]
    assert interior.mean() > 0


def test_compute_visualizations_shape():
    """compute_visualizations should return 3-channel (C, H, W) array."""
    dem = np.random.rand(200, 200).astype(np.float32) * 100
    result = compute_visualizations(dem, resolution=0.5)
    assert result.shape == (3, 200, 200)
    assert result.dtype == np.float32


def test_compute_visualizations_normalized():
    """Each channel should be normalized to [0, 1]."""
    dem = np.random.rand(200, 200).astype(np.float32) * 100
    result = compute_visualizations(dem, resolution=0.5)
    for c in range(3):
        assert result[c].min() >= 0.0 - 1e-6
        assert result[c].max() <= 1.0 + 1e-6
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_visualize.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement visualize.py**

Create `mayascan/visualize.py`:
```python
"""Generate archaeological visualization rasters from DEMs."""

from __future__ import annotations

import numpy as np


def compute_slope(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute slope in degrees from a DEM.

    Args:
        dem: 2D array of elevation values.
        resolution: Ground resolution in meters per pixel.

    Returns:
        2D array of slope values in degrees.
    """
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad).astype(np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(arr)
    return ((arr - vmin) / (vmax - vmin)).astype(np.float32)


def compute_svf(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute Sky-View Factor from a DEM.

    Uses rvt-py if available, falls back to a simplified approximation.

    Args:
        dem: 2D array of elevation values.
        resolution: Ground resolution in meters per pixel.

    Returns:
        2D array of SVF values.
    """
    try:
        import rvt.vis

        svf_dict = rvt.vis.sky_view_factor(
            dem=dem,
            resolution=resolution,
            compute_svf=True,
            compute_opns=False,
            svf_n_dir=16,
            svf_r_max=10,
            svf_noise=0,
        )
        return svf_dict["svf"].astype(np.float32)
    except ImportError:
        # Fallback: approximate SVF using mean elevation difference to neighbors
        from scipy.ndimage import uniform_filter
        mean_elev = uniform_filter(dem, size=21)
        svf_approx = 1.0 - np.clip((mean_elev - dem) / (dem.ptp() + 1e-10), 0, 1)
        return svf_approx.astype(np.float32)


def compute_openness(dem: np.ndarray, resolution: float = 0.5) -> np.ndarray:
    """Compute positive openness from a DEM.

    Uses rvt-py if available, falls back to a simplified approximation.

    Args:
        dem: 2D array of elevation values.
        resolution: Ground resolution in meters per pixel.

    Returns:
        2D array of positive openness values.
    """
    try:
        import rvt.vis

        svf_dict = rvt.vis.sky_view_factor(
            dem=dem,
            resolution=resolution,
            compute_svf=False,
            compute_opns=True,
            svf_n_dir=16,
            svf_r_max=10,
            svf_noise=0,
        )
        return svf_dict["opns"].astype(np.float32)
    except ImportError:
        # Fallback: approximate openness using max filter vs elevation
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(dem, size=21)
        openness_approx = np.clip(
            (local_max - dem) / (dem.ptp() + 1e-10), 0, 1
        )
        return (1.0 - openness_approx).astype(np.float32)


def compute_visualizations(
    dem: np.ndarray,
    resolution: float = 0.5,
) -> np.ndarray:
    """Generate 3-channel visualization stack from a DEM.

    Channels: [sky-view factor, positive openness, slope]
    All normalized to [0, 1].

    Args:
        dem: 2D array of elevation values.
        resolution: Ground resolution in meters per pixel.

    Returns:
        Array of shape (3, H, W), float32, values in [0, 1].
    """
    svf = _normalize(compute_svf(dem, resolution))
    openness = _normalize(compute_openness(dem, resolution))
    slope = _normalize(compute_slope(dem, resolution))

    return np.stack([svf, openness, slope], axis=0)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_visualize.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add mayascan/visualize.py tests/test_visualize.py
git commit -m "feat: add DEM visualization module (SVF, openness, slope)"
```

---

### Task 4: U-Net Model Wrapper (models/unet.py)

Wrap segmentation-models-pytorch's U-Net for our specific 3-channel input, 4-class output task.

**Files:**
- Create: `mayascan/models/__init__.py`
- Create: `mayascan/models/unet.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing tests**

Create `tests/test_model.py`:
```python
import numpy as np
import pytest
import torch
from mayascan.models.unet import MayaScanUNet


def test_model_output_shape():
    """Model should output (B, 4, H, W) for (B, 3, H, W) input."""
    model = MayaScanUNet(num_classes=4, encoder="resnet34")
    x = torch.randn(2, 3, 480, 480)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4, 480, 480)


def test_model_output_logits():
    """Output should be raw logits (not softmaxed) — can be negative."""
    model = MayaScanUNet(num_classes=4, encoder="resnet34")
    x = torch.randn(1, 3, 480, 480)
    with torch.no_grad():
        out = model(x)
    # Raw logits can be negative
    assert out.min() < 0 or out.max() > 1  # not clamped


def test_model_predict():
    """predict() should return class indices and confidence map."""
    model = MayaScanUNet(num_classes=4, encoder="resnet34")
    x = torch.randn(1, 3, 480, 480)
    classes, confidence = model.predict(x)
    assert classes.shape == (480, 480)
    assert confidence.shape == (480, 480)
    assert classes.dtype == torch.long
    assert confidence.min() >= 0.0
    assert confidence.max() <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the model**

Create `mayascan/models/__init__.py`:
```python
"""Model architectures for MayaScan."""
```

Create `mayascan/models/unet.py`:
```python
"""U-Net model wrapper for archaeological feature segmentation."""

from __future__ import annotations

from typing import Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MayaScanUNet(nn.Module):
    """U-Net semantic segmentation model for Maya archaeological features.

    Input: 3-channel raster (SVF, openness, slope), 480x480 px.
    Output: 4-class segmentation (background, buildings, platforms, aguadas).
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: (B, 3, H, W) float tensor.

        Returns:
            (B, num_classes, H, W) logit tensor.
        """
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference and return predicted classes + confidence.

        Args:
            x: (1, 3, H, W) float tensor.

        Returns:
            classes: (H, W) long tensor of class indices.
            confidence: (H, W) float tensor of max softmax probability.
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, classes = probs.max(dim=1)
        return classes.squeeze(0), confidence.squeeze(0)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: All 3 tests PASS. (First run may be slow — downloads ResNet34 weights.)

**Step 5: Commit**

```bash
git add mayascan/models/ tests/test_model.py
git commit -m "feat: add U-Net model wrapper with predict method"
```

---

### Task 5: Detection Module (detect.py)

The main inference pipeline: takes visualization rasters, tiles them, runs the model, stitches results, and returns structured predictions.

**Files:**
- Create: `mayascan/detect.py`
- Create: `tests/test_detect.py`

**Step 1: Write the failing tests**

Create `tests/test_detect.py`:
```python
import numpy as np
import pytest
from mayascan.detect import DetectionResult, run_detection


def test_detection_result_fields():
    """DetectionResult should have classes, confidence, and class names."""
    result = DetectionResult(
        classes=np.zeros((100, 100), dtype=np.int64),
        confidence=np.ones((100, 100), dtype=np.float32),
        class_names={0: "background", 1: "building", 2: "platform", 3: "aguada"},
    )
    assert result.classes.shape == (100, 100)
    assert result.class_names[1] == "building"


def test_run_detection_output_shape():
    """run_detection should return a DetectionResult matching input spatial dims."""
    viz = np.random.rand(3, 480, 480).astype(np.float32)
    result = run_detection(viz, model_path=None, confidence_threshold=0.0)
    assert isinstance(result, DetectionResult)
    assert result.classes.shape == (480, 480)
    assert result.confidence.shape == (480, 480)


def test_run_detection_large_image():
    """Detection on an image larger than tile size should still work."""
    viz = np.random.rand(3, 960, 960).astype(np.float32)
    result = run_detection(viz, model_path=None, confidence_threshold=0.0)
    assert result.classes.shape == (960, 960)


def test_run_detection_confidence_threshold():
    """Pixels below confidence threshold should be classified as background (0)."""
    viz = np.random.rand(3, 480, 480).astype(np.float32)
    result = run_detection(viz, model_path=None, confidence_threshold=0.99)
    # With random weights, most predictions will be low confidence
    background_count = (result.classes == 0).sum()
    total = result.classes.size
    assert background_count / total > 0.5  # Most should be filtered to background
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_detect.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement detect.py**

Create `mayascan/detect.py`:
```python
"""Archaeological feature detection using semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from mayascan.models.unet import MayaScanUNet
from mayascan.tile import slice_tiles, stitch_tiles

CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "building",
    2: "platform",
    3: "aguada",
}


@dataclass
class DetectionResult:
    """Result of archaeological feature detection."""

    classes: np.ndarray  # (H, W) int array of class indices
    confidence: np.ndarray  # (H, W) float array of confidence values
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))


def run_detection(
    visualization: np.ndarray,
    model_path: Optional[str] = None,
    tile_size: int = 480,
    overlap: float = 0.5,
    confidence_threshold: float = 0.5,
    device: Optional[str] = None,
) -> DetectionResult:
    """Run feature detection on a visualization raster.

    Args:
        visualization: (3, H, W) float32 array (SVF, openness, slope in [0,1]).
        model_path: Path to saved model weights. None uses random weights.
        tile_size: Size of tiles for inference.
        overlap: Overlap fraction between tiles.
        confidence_threshold: Minimum confidence to keep a non-background prediction.
        device: PyTorch device string. None auto-detects.

    Returns:
        DetectionResult with class map and confidence map.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MayaScanUNet(num_classes=4, pretrained=(model_path is None))
    if model_path is not None:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model = model.to(device).eval()

    _, h, w = visualization.shape
    tiles, origins = slice_tiles(visualization, tile_size=tile_size, overlap=overlap)

    prob_tiles = []
    for tile in tiles:
        x = torch.from_numpy(tile).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        prob_tiles.append(probs)

    prob_map = stitch_tiles(prob_tiles, origins, output_shape=(4, h, w), overlap=overlap)

    classes = prob_map.argmax(axis=0).astype(np.int64)
    confidence = prob_map.max(axis=0)

    # Apply confidence threshold: low-confidence non-background → background
    low_conf = confidence < confidence_threshold
    classes[low_conf & (classes != 0)] = 0

    return DetectionResult(classes=classes, confidence=confidence)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_detect.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add mayascan/detect.py tests/test_detect.py
git commit -m "feat: add detection module with tiled inference pipeline"
```

---

### Task 6: Export Module (export.py)

Convert detection results to GeoJSON, CSV, and GeoTIFF formats.

**Files:**
- Create: `mayascan/export.py`
- Create: `tests/test_export.py`

**Step 1: Write the failing tests**

Create `tests/test_export.py`:
```python
import json
import os
import tempfile

import numpy as np
import pytest
from mayascan.detect import DetectionResult
from mayascan.export import to_csv, to_geojson, to_geotiff


@pytest.fixture
def sample_result():
    classes = np.zeros((100, 100), dtype=np.int64)
    classes[20:40, 30:50] = 1  # building
    classes[60:70, 70:80] = 2  # platform
    confidence = np.ones((100, 100), dtype=np.float32) * 0.9
    return DetectionResult(
        classes=classes,
        confidence=confidence,
        class_names={0: "background", 1: "building", 2: "platform", 3: "aguada"},
    )


def test_to_csv(sample_result, tmp_path):
    """to_csv should write a CSV with feature centroids."""
    path = tmp_path / "features.csv"
    to_csv(sample_result, str(path))
    assert path.exists()
    content = path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 feature
    assert "class" in lines[0]
    assert "building" in content


def test_to_geojson(sample_result, tmp_path):
    """to_geojson should write valid GeoJSON with feature polygons."""
    path = tmp_path / "features.geojson"
    to_geojson(sample_result, str(path))
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 1


def test_to_geotiff(sample_result, tmp_path):
    """to_geotiff should write a single-band GeoTIFF."""
    path = tmp_path / "classes.tif"
    to_geotiff(sample_result, str(path))
    assert path.exists()
    assert path.stat().st_size > 0
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_export.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement export.py**

Create `mayascan/export.py`:
```python
"""Export detection results to GIS-compatible formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from mayascan.detect import DetectionResult


def to_csv(
    result: DetectionResult,
    output_path: str,
    pixel_size: float = 0.5,
) -> None:
    """Export detected features as CSV with centroids.

    Args:
        result: Detection result.
        output_path: Path to write CSV file.
        pixel_size: Ground resolution in meters/pixel for coordinate conversion.
    """
    rows = []
    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue
        mask = result.classes == class_id
        if not mask.any():
            continue
        labeled, n_features = ndimage.label(mask)
        for feat_id in range(1, n_features + 1):
            feat_mask = labeled == feat_id
            ys, xs = np.where(feat_mask)
            centroid_y = ys.mean() * pixel_size
            centroid_x = xs.mean() * pixel_size
            area = feat_mask.sum() * pixel_size * pixel_size
            conf = result.confidence[feat_mask].mean()
            rows.append({
                "class": class_name,
                "class_id": class_id,
                "centroid_x": round(centroid_x, 2),
                "centroid_y": round(centroid_y, 2),
                "area_m2": round(area, 2),
                "confidence": round(float(conf), 4),
                "pixel_count": int(feat_mask.sum()),
            })

    with open(output_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("class,class_id,centroid_x,centroid_y,area_m2,confidence,pixel_count\n")


def to_geojson(
    result: DetectionResult,
    output_path: str,
    pixel_size: float = 0.5,
) -> None:
    """Export detected features as GeoJSON polygons.

    Note: Coordinates are in pixel-space scaled by pixel_size (meters).
    For geo-referenced output, a CRS transform would need to be applied.

    Args:
        result: Detection result.
        output_path: Path to write GeoJSON file.
        pixel_size: Ground resolution in meters/pixel.
    """
    features = []

    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue
        mask = (result.classes == class_id).astype(np.uint8)
        if not mask.any():
            continue
        labeled, n_features = ndimage.label(mask)
        for feat_id in range(1, n_features + 1):
            feat_mask = labeled == feat_id
            ys, xs = np.where(feat_mask)
            min_x, max_x = xs.min() * pixel_size, (xs.max() + 1) * pixel_size
            min_y, max_y = ys.min() * pixel_size, (ys.max() + 1) * pixel_size
            conf = float(result.confidence[feat_mask].mean())
            bbox_polygon = [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]
            features.append({
                "type": "Feature",
                "properties": {
                    "class": class_name,
                    "class_id": int(class_id),
                    "area_m2": round(feat_mask.sum() * pixel_size * pixel_size, 2),
                    "confidence": round(conf, 4),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [bbox_polygon],
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def to_geotiff(
    result: DetectionResult,
    output_path: str,
    pixel_size: float = 0.5,
) -> None:
    """Export class map as a single-band GeoTIFF.

    Args:
        result: Detection result.
        output_path: Path to write GeoTIFF file.
        pixel_size: Ground resolution in meters/pixel.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        h, w = result.classes.shape
        transform = from_bounds(0, 0, w * pixel_size, h * pixel_size, w, h)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="uint8",
            transform=transform,
        ) as dst:
            dst.write(result.classes.astype(np.uint8), 1)
    except ImportError:
        # Fallback: write raw numpy as a minimal TIFF using PIL
        from PIL import Image

        img = Image.fromarray(result.classes.astype(np.uint8))
        img.save(output_path)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_export.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add mayascan/export.py tests/test_export.py
git commit -m "feat: add export module (CSV, GeoJSON, GeoTIFF)"
```

---

### Task 7: Ground Classification Module (classify.py)

PDAL-based ground filtering. This has an optional dependency on python-pdal, so we provide a graceful fallback.

**Files:**
- Create: `mayascan/classify.py`
- Create: `tests/test_classify.py`

**Step 1: Write the failing tests**

Create `tests/test_classify.py`:
```python
import numpy as np
import pytest
from mayascan.classify import dem_from_array, PdalNotAvailableError


def test_dem_from_array():
    """dem_from_array should generate a DEM from x, y, z arrays."""
    n = 10000
    x = np.random.uniform(0, 100, n).astype(np.float64)
    y = np.random.uniform(0, 100, n).astype(np.float64)
    z = np.sin(x / 10) + np.cos(y / 10)  # smooth surface
    z = z.astype(np.float64)
    dem, extent = dem_from_array(x, y, z, resolution=1.0)
    assert dem.ndim == 2
    assert dem.shape[0] > 0 and dem.shape[1] > 0
    assert extent["xmin"] == pytest.approx(x.min(), abs=1)
    assert not np.all(np.isnan(dem))


def test_dem_from_array_resolution():
    """Higher resolution should produce a larger array."""
    n = 5000
    x = np.random.uniform(0, 50, n).astype(np.float64)
    y = np.random.uniform(0, 50, n).astype(np.float64)
    z = np.zeros(n, dtype=np.float64)
    dem_low, _ = dem_from_array(x, y, z, resolution=2.0)
    dem_high, _ = dem_from_array(x, y, z, resolution=0.5)
    assert dem_high.size > dem_low.size
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_classify.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement classify.py**

Create `mayascan/classify.py`:
```python
"""Ground classification and DEM generation from LiDAR point clouds."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata


class PdalNotAvailableError(ImportError):
    """Raised when PDAL is required but not installed."""

    def __init__(self):
        super().__init__(
            "python-pdal is required for LAS/LAZ processing. "
            "Install it with: conda install -c conda-forge python-pdal"
        )


def classify_ground(
    input_path: str,
    output_dem_path: Optional[str] = None,
    resolution: float = 0.5,
) -> np.ndarray:
    """Classify ground points and generate a bare-earth DEM from a LAS/LAZ file.

    Requires python-pdal to be installed (conda install -c conda-forge python-pdal).

    Args:
        input_path: Path to LAS/LAZ file.
        output_dem_path: Optional path to write DEM as GeoTIFF.
        resolution: DEM resolution in meters.

    Returns:
        2D numpy array of the bare-earth DEM.

    Raises:
        PdalNotAvailableError: If python-pdal is not installed.
    """
    try:
        import pdal
    except ImportError:
        raise PdalNotAvailableError()

    pipeline_json = {
        "pipeline": [
            {"type": "readers.las", "filename": input_path},
            {"type": "filters.outlier", "method": "statistical", "mean_k": 12, "multiplier": 2.2},
            {"type": "filters.smrf", "cell": 1.0, "slope": 0.15, "window": 30, "threshold": 0.5},
            {"type": "filters.range", "limits": "Classification[2:2]"},
        ]
    }

    if output_dem_path:
        pipeline_json["pipeline"].append({
            "type": "writers.gdal",
            "filename": output_dem_path,
            "resolution": resolution,
            "output_type": "idw",
        })

    pipeline = pdal.Pipeline(json_module.dumps(pipeline_json))
    pipeline.execute()
    arrays = pipeline.arrays
    if len(arrays) == 0:
        raise ValueError(f"No ground points found in {input_path}")

    points = arrays[0]
    return dem_from_array(
        points["X"], points["Y"], points["Z"], resolution=resolution
    )[0]


def dem_from_array(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Generate a DEM grid from x, y, z point arrays using IDW interpolation.

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        z: 1D array of z (elevation) values.
        resolution: Grid cell size.

    Returns:
        dem: 2D array of interpolated elevations.
        extent: Dict with xmin, xmax, ymin, ymax.
    """
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    cols = max(1, int(math.ceil((xmax - xmin) / resolution)))
    rows = max(1, int(math.ceil((ymax - ymin) / resolution)))

    grid_x = np.linspace(xmin, xmax, cols)
    grid_y = np.linspace(ymin, ymax, rows)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    dem = griddata(
        np.column_stack([x, y]),
        z,
        (grid_xx, grid_yy),
        method="linear",
        fill_value=np.nan,
    ).astype(np.float32)

    # Fill remaining NaN with nearest neighbor
    if np.any(np.isnan(dem)):
        dem_nearest = griddata(
            np.column_stack([x, y]),
            z,
            (grid_xx, grid_yy),
            method="nearest",
        ).astype(np.float32)
        dem = np.where(np.isnan(dem), dem_nearest, dem)

    extent = {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    return dem, extent
```

Note: `classify_ground` uses PDAL (optional). `dem_from_array` uses scipy only and is always available. Fix the import: add `import json as json_module` to the imports or use `json.dumps`. Let me correct — use `import json` at the top and use `json.dumps`.

Actually, correct the implementation: replace `json_module.dumps` with proper import. The final file should have `import json` at the top and use `json.dumps(pipeline_json)`.

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_classify.py -v
```

Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add mayascan/classify.py tests/test_classify.py
git commit -m "feat: add ground classification module with PDAL pipeline and scipy fallback"
```

---

### Task 8: Public API (__init__.py)

Wire up the top-level `mayascan.process()`, `mayascan.classify()`, `mayascan.visualize()`, `mayascan.detect()` convenience functions.

**Files:**
- Modify: `mayascan/__init__.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing tests**

Create `tests/test_api.py`:
```python
import numpy as np
import pytest
import mayascan


def test_version():
    assert hasattr(mayascan, "__version__")


def test_visualize_api():
    """mayascan.visualize() should accept a DEM and return 3-channel array."""
    dem = np.random.rand(200, 200).astype(np.float32) * 100
    result = mayascan.visualize(dem)
    assert result.shape == (3, 200, 200)


def test_detect_api():
    """mayascan.detect() should accept visualization and return DetectionResult."""
    viz = np.random.rand(3, 480, 480).astype(np.float32)
    result = mayascan.detect(viz, confidence_threshold=0.0)
    assert hasattr(result, "classes")
    assert hasattr(result, "confidence")
    assert result.classes.shape == (480, 480)


def test_process_from_dem():
    """mayascan.process_dem() should run full pipeline from a DEM array."""
    dem = np.random.rand(480, 480).astype(np.float32) * 100
    result = mayascan.process_dem(dem)
    assert hasattr(result, "classes")
    assert result.classes.shape == (480, 480)
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_api.py -v
```

Expected: FAIL — `AttributeError: module 'mayascan' has no attribute 'visualize'`

**Step 3: Update __init__.py**

Modify `mayascan/__init__.py`:
```python
"""MayaScan: Open-source archaeological LiDAR feature detection."""

__version__ = "0.1.0"

from mayascan.visualize import compute_visualizations
from mayascan.detect import run_detection, DetectionResult


def visualize(dem, resolution=0.5):
    """Generate visualization rasters from a DEM.

    Args:
        dem: 2D numpy array of elevation values.
        resolution: Ground resolution in meters/pixel.

    Returns:
        (3, H, W) float32 array: [SVF, openness, slope], normalized to [0, 1].
    """
    return compute_visualizations(dem, resolution=resolution)


def detect(visualization, model_path=None, confidence_threshold=0.5):
    """Run archaeological feature detection on visualization rasters.

    Args:
        visualization: (3, H, W) float32 array from visualize().
        model_path: Path to trained model weights. None uses default.
        confidence_threshold: Minimum confidence for non-background predictions.

    Returns:
        DetectionResult with classes and confidence arrays.
    """
    return run_detection(
        visualization,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )


def process_dem(dem, resolution=0.5, model_path=None, confidence_threshold=0.5):
    """Full pipeline from DEM array to detection results.

    Args:
        dem: 2D numpy array of elevation values.
        resolution: Ground resolution in meters/pixel.
        model_path: Path to trained model weights.
        confidence_threshold: Minimum confidence threshold.

    Returns:
        DetectionResult.
    """
    viz = visualize(dem, resolution=resolution)
    return detect(viz, model_path=model_path, confidence_threshold=confidence_threshold)
```

**Step 4: Run ALL tests to verify everything passes**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests across all modules PASS.

**Step 5: Commit**

```bash
git add mayascan/__init__.py tests/test_api.py
git commit -m "feat: add public API with process_dem, visualize, detect"
```

---

### Task 9: Gradio Web Application (app.py)

Build the cloud-hosted web interface.

**Files:**
- Create: `app.py`

**Step 1: Implement app.py**

Create `app.py`:
```python
"""MayaScan Gradio web application."""

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

import mayascan
from mayascan.detect import CLASS_NAMES
from mayascan.export import to_csv, to_geojson, to_geotiff


COLORS = {
    0: [0, 0, 0, 0],        # background: transparent
    1: [255, 60, 60, 180],   # building: red
    2: [255, 220, 50, 180],  # platform: yellow
    3: [50, 120, 255, 180],  # aguada: blue
}


def colorize_classes(classes: np.ndarray) -> np.ndarray:
    """Convert class index map to RGBA image."""
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in COLORS.items():
        mask = classes == cls_id
        rgba[mask] = color
    return rgba


def load_dem_from_file(file_path: str) -> np.ndarray:
    """Load a DEM from GeoTIFF or numpy file."""
    path = Path(file_path)
    if path.suffix in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                dem = src.read(1).astype(np.float32)
            return dem
        except ImportError:
            from PIL import Image
            img = Image.open(file_path)
            return np.array(img, dtype=np.float32)
    elif path.suffix == ".npy":
        return np.load(file_path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .tif or .npy")


def process_upload(file, confidence_threshold, resolution):
    """Main processing function for the Gradio interface."""
    if file is None:
        return None, None, None, "Please upload a DEM file (.tif or .npy)"

    try:
        dem = load_dem_from_file(file.name)
    except Exception as e:
        return None, None, None, f"Error loading file: {e}"

    # Generate visualizations
    viz = mayascan.visualize(dem, resolution=resolution)

    # Create RGB preview of the visualization channels
    viz_rgb = np.stack([viz[0], viz[1], viz[2]], axis=-1)
    viz_rgb = (viz_rgb * 255).astype(np.uint8)

    # Run detection
    result = mayascan.detect(viz, confidence_threshold=confidence_threshold)

    # Colorize results
    overlay = colorize_classes(result.classes)

    # Count features
    from scipy import ndimage
    stats_lines = []
    total_features = 0
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        mask = result.classes == cls_id
        if mask.any():
            _, n = ndimage.label(mask)
            total_features += n
            stats_lines.append(f"  {cls_name}: {n} detected")

    stats = f"Detection complete!\n\nFeatures found: {total_features}\n"
    stats += "\n".join(stats_lines)
    stats += f"\n\nDEM size: {dem.shape[0]}x{dem.shape[1]} pixels"
    stats += f"\nResolution: {resolution}m/pixel"
    stats += f"\nGround area: {dem.shape[1]*resolution:.0f}m x {dem.shape[0]*resolution:.0f}m"

    # Export files
    export_dir = tempfile.mkdtemp()
    csv_path = str(Path(export_dir) / "features.csv")
    geojson_path = str(Path(export_dir) / "features.geojson")
    tiff_path = str(Path(export_dir) / "classes.tif")
    to_csv(result, csv_path, pixel_size=resolution)
    to_geojson(result, geojson_path, pixel_size=resolution)
    to_geotiff(result, tiff_path, pixel_size=resolution)

    return viz_rgb, overlay, [csv_path, geojson_path, tiff_path], stats


with gr.Blocks(
    title="MayaScan — Archaeological LiDAR Feature Detection",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # MayaScan
        ### Open-Source Archaeological LiDAR Feature Detection

        Upload a Digital Elevation Model (DEM) derived from LiDAR to detect
        ancient Maya structures: **buildings**, **platforms**, and **aguadas** (reservoirs).

        Accepts GeoTIFF (.tif) or NumPy (.npy) DEM files.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload DEM (.tif or .npy)", file_types=[".tif", ".tiff", ".npy"])
            confidence_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                label="Confidence Threshold",
                info="Higher = fewer but more confident detections",
            )
            resolution_input = gr.Number(
                value=0.5, label="Resolution (m/pixel)",
                info="Ground resolution of your DEM",
            )
            run_btn = gr.Button("Detect Structures", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tab("Visualization"):
                viz_output = gr.Image(label="DEM Visualizations (SVF / Openness / Slope)")
            with gr.Tab("Detection Results"):
                detection_output = gr.Image(label="Detected Features (Red=Building, Yellow=Platform, Blue=Aguada)")
            stats_output = gr.Textbox(label="Detection Summary", lines=8)
            export_output = gr.File(label="Download Results", file_count="multiple")

    run_btn.click(
        fn=process_upload,
        inputs=[file_input, confidence_slider, resolution_input],
        outputs=[viz_output, detection_output, export_output, stats_output],
    )

    gr.Markdown(
        """
        ---
        **MayaScan** is open-source (MIT). No proprietary software required.

        Built on: PDAL, rvt-py, PyTorch, segmentation-models-pytorch

        Training data: [Chactún ML-ready dataset](https://doi.org/10.1038/s41597-023-02455-x) (Kokalj et al., 2023)
        """
    )


if __name__ == "__main__":
    demo.launch()
```

**Step 2: Test the app launches**

```bash
cd /Volumes/macos4tb/Projects/mayascan
python3 app.py
```

Expected: Gradio prints a local URL. Open it in browser, verify the UI renders. Ctrl+C to stop.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio web interface for DEM upload and detection"
```

---

### Task 10: Training Notebook (notebooks/train.ipynb)

Colab-ready notebook to download the Chactún dataset and train the U-Net.

**Files:**
- Create: `notebooks/train.ipynb`

**Step 1: Create the training notebook**

Create `notebooks/train.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# MayaScan Model Training

Train the U-Net archaeological feature detection model on the Chactún ML-ready dataset.

This notebook is designed to run on Google Colab with a free T4 GPU.
```

**Cell 2 (code): Install dependencies**
```python
# Install MayaScan and dependencies
!pip install -q segmentation-models-pytorch torch torchvision rasterio rvt-py tqdm
!pip install -q gdown  # for dataset download
```

**Cell 3 (code): Download dataset**
```python
import os
import zipfile
import gdown

DATA_DIR = "/content/chactun_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Chactún ML-ready dataset from Figshare
# https://figshare.com/articles/dataset/22202395
# Files: ALS visualizations (SVF, openness, slope) + annotation masks
FIGSHARE_URL = "https://figshare.com/ndownloader/articles/22202395/versions/3"
ZIP_PATH = f"{DATA_DIR}/chactun.zip"

if not os.path.exists(f"{DATA_DIR}/extracted"):
    print("Downloading Chactún dataset from Figshare...")
    gdown.download(FIGSHARE_URL, ZIP_PATH, quiet=False, fuzzy=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(f"{DATA_DIR}/extracted")
    print("Dataset extracted!")
else:
    print("Dataset already downloaded.")
```

**Cell 4 (code): Dataset loader**
```python
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ChactunDataset(Dataset):
    """Load Chactún tiles: 3-channel viz input + 3 binary masks → multi-class."""

    def __init__(self, data_dir, split="train", augment=True):
        self.data_dir = data_dir
        self.augment = augment

        # Find all SVF tiles (use as index)
        svf_pattern = os.path.join(data_dir, "**", "*svf*.*")
        self.svf_files = sorted(glob.glob(svf_pattern, recursive=True))

        # Split 80/20
        n = len(self.svf_files)
        split_idx = int(n * 0.8)
        if split == "train":
            self.svf_files = self.svf_files[:split_idx]
        else:
            self.svf_files = self.svf_files[split_idx:]

        print(f"{split}: {len(self.svf_files)} tiles")

    def _find_matching(self, svf_path, keyword):
        """Find the matching openness/slope/mask file for a given SVF file."""
        directory = os.path.dirname(svf_path)
        basename = os.path.basename(svf_path)
        # Replace 'svf' with the target keyword
        target = basename.replace("svf", keyword).replace("SVF", keyword)
        candidate = os.path.join(directory, target)
        if os.path.exists(candidate):
            return candidate
        # Fallback: search in same directory
        for f in os.listdir(directory):
            if keyword.lower() in f.lower():
                return os.path.join(directory, f)
        return None

    def __len__(self):
        return len(self.svf_files)

    def __getitem__(self, idx):
        svf_path = self.svf_files[idx]

        # Load channels
        svf = np.array(Image.open(svf_path), dtype=np.float32)
        opns_path = self._find_matching(svf_path, "opns")
        slope_path = self._find_matching(svf_path, "slope")

        opns = np.array(Image.open(opns_path), dtype=np.float32) if opns_path else np.zeros_like(svf)
        slope = np.array(Image.open(slope_path), dtype=np.float32) if slope_path else np.zeros_like(svf)

        # Normalize to [0, 1]
        for arr in [svf, opns, slope]:
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr[:] = (arr - vmin) / (vmax - vmin)

        # Stack channels
        image = np.stack([svf, opns, slope], axis=0)  # (3, H, W)

        # Load masks → combine into multi-class
        mask = np.zeros(svf.shape, dtype=np.int64)
        for cls_id, keyword in [(1, "building"), (2, "platform"), (3, "aguada")]:
            mask_path = self._find_matching(svf_path, keyword)
            if mask_path and os.path.exists(mask_path):
                m = np.array(Image.open(mask_path), dtype=np.float32)
                mask[m > 0.5] = cls_id

        # Augmentation
        if self.augment:
            k = np.random.randint(4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

        return torch.from_numpy(image), torch.from_numpy(mask)
```

**Cell 5 (code): Training loop**
```python
import segmentation_models_pytorch as smp
from torch import nn, optim
from tqdm import tqdm

# Model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4,
).cuda()

# Class weights (inverse frequency: bg common, aguadas rare)
class_weights = torch.tensor([0.1, 1.0, 2.0, 50.0], dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Data
train_ds = ChactunDataset(f"{DATA_DIR}/extracted", split="train")
val_ds = ChactunDataset(f"{DATA_DIR}/extracted", split="val", augment=False)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

# Training
EPOCHS = 50
best_iou = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.cuda(), masks.cuda()
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_iou_sum = 0
    val_count = 0
    with torch.no_grad():
        for images, masks in val_dl:
            images, masks = images.cuda(), masks.cuda()
            preds = model(images).argmax(dim=1)
            # Compute mean IoU (excluding background)
            for c in range(1, 4):
                pred_c = preds == c
                mask_c = masks == c
                intersection = (pred_c & mask_c).sum().float()
                union = (pred_c | mask_c).sum().float()
                if union > 0:
                    val_iou_sum += (intersection / union).item()
                    val_count += 1

    mean_iou = val_iou_sum / max(val_count, 1)
    print(f"Epoch {epoch+1}: loss={train_loss/len(train_dl):.4f}, val_mIoU={mean_iou:.4f}")

    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "mayascan_unet_best.pth")
        print(f"  Saved best model (mIoU={best_iou:.4f})")

print(f"\nTraining complete! Best mIoU: {best_iou:.4f}")
```

**Cell 6 (code): Upload to Hugging Face Hub**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="mayascan_unet_best.pth",
    path_in_repo="mayascan_unet_v1.pth",
    repo_id="fascinated23/mayascan",  # Change to your HF username
    repo_type="model",
)
print("Model uploaded to Hugging Face Hub!")
```

**Step 2: Commit**

```bash
git add notebooks/train.ipynb
git commit -m "feat: add Colab-ready training notebook for Chactún dataset"
```

---

### Task 11: Integration Test + Final Wiring

End-to-end test: synthetic DEM → visualize → detect → export.

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:
```python
"""End-to-end integration test for the full MayaScan pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import mayascan
from mayascan.export import to_csv, to_geojson, to_geotiff


def test_full_pipeline_synthetic():
    """Run the full pipeline on a synthetic DEM with fake structures."""
    # Create a DEM with mound-like features
    dem = np.zeros((480, 480), dtype=np.float32)
    # Add some "mounds"
    for cy, cx, r, h in [(100, 100, 20, 5), (300, 200, 15, 3), (200, 350, 25, 7)]:
        yy, xx = np.ogrid[-cy:480-cy, -cx:480-cx]
        mask = xx**2 + yy**2 <= r**2
        dem[mask] += h

    # Visualize
    viz = mayascan.visualize(dem, resolution=0.5)
    assert viz.shape == (3, 480, 480)

    # Detect (with random weights — just testing the pipeline runs)
    result = mayascan.detect(viz, confidence_threshold=0.0)
    assert result.classes.shape == (480, 480)
    assert result.confidence.shape == (480, 480)

    # Export
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "features.csv")
        geojson_path = str(Path(tmpdir) / "features.geojson")
        tiff_path = str(Path(tmpdir) / "classes.tif")

        to_csv(result, csv_path)
        to_geojson(result, geojson_path)
        to_geotiff(result, tiff_path)

        assert Path(csv_path).exists()
        assert Path(geojson_path).exists()
        assert Path(tiff_path).exists()

        with open(geojson_path) as f:
            geojson = json.load(f)
        assert geojson["type"] == "FeatureCollection"


def test_process_dem_convenience():
    """Test the top-level convenience function."""
    dem = np.random.rand(480, 480).astype(np.float32) * 50
    result = mayascan.process_dem(dem, confidence_threshold=0.0)
    assert result.classes.shape == (480, 480)
```

**Step 2: Run ALL tests**

```bash
python3 -m pytest tests/ -v
```

Expected: All tests across all modules PASS.

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add end-to-end integration tests"
```

---

### Task 12: .gitignore + Final Commit

**Files:**
- Create: `.gitignore`

**Step 1: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.pth
*.pt
.ipynb_checkpoints/
.env
*.tif
*.las
*.laz
```

**Step 2: Final commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

**Step 3: Verify clean state**

```bash
git log --oneline
python3 -m pytest tests/ -v
```

Expected: 10+ commits, all tests pass.
