# MayaScan

Open-source archaeological LiDAR feature detection for Maya archaeology.

Upload a LiDAR DEM, get back a map of probable ancient structures — buildings, platforms, and aguadas (water reservoirs).

## Features

- **Per-class binary segmentation** — separate DeepLabV3+ (ResNet-101) models for each feature class, trained with competition-winning techniques from ECML PKDD 2021 Maya Challenge
- **Terrain visualization** — automatic SVF, openness, and slope computation from raw DEMs (using rvt-py or scipy fallback)
- **Test-Time Augmentation (TTA)** — 8-fold augmentation (4 rotations x 2 flips) for robust predictions
- **Georeferenced output** — CRS and affine transform propagated from input GeoTIFFs to all exports
- **Multiple export formats** — GeoTIFF, GeoJSON (with polygon contours), CSV
- **Batch processing** — process entire survey directories in one command
- **Web interface** — Gradio app with interactive overlay visualization
- **HuggingFace integration** — download pre-trained models with one command

## Quick Start

### Install

```bash
pip install mayascan
```

With optional dependencies:

```bash
pip install mayascan[geo]     # rasterio, geopandas — for georeferenced I/O
pip install mayascan[rvt]     # rvt-py — high-quality terrain visualizations
pip install mayascan[web]     # gradio — web interface
pip install mayascan[hub]     # huggingface-hub — model download
pip install mayascan[all]     # everything
```

### Download Models

```bash
python -m mayascan download
```

### Run Detection

```bash
# Single file
python -m mayascan scan input.tif -o results/

# Directory of DEMs (batch mode)
python -m mayascan scan survey_data/ -o results/

# With options
python -m mayascan scan input.tif --threshold 0.6 --no-tta --device mps
```

### Python API

```python
import mayascan

# Full pipeline: DEM → visualization → detection
result = mayascan.process_dem(dem_array, resolution=0.5)

# Step by step
viz = mayascan.visualize(dem, resolution=0.5)        # (3, H, W) float32
result = mayascan.detect(viz, confidence_threshold=0.6)

# v2 per-class models (higher accuracy)
result = mayascan.detect_v2(viz, model_dir="models/")

# With georeferencing
data, geo = mayascan.read_raster("input.tif")        # auto-reads CRS
viz = mayascan.visualize(data, resolution=geo.resolution)
result = mayascan.detect_v2(viz)
result.geo = geo  # propagate for georeferenced export

# Export
from mayascan.export import to_csv, to_geojson, to_geotiff
to_csv(result, "detections.csv")
to_geojson(result, "detections.geojson")     # polygon contours
to_geotiff(result, "detections.tif")         # georeferenced class map
```

### Web App

```bash
python app.py
```

Opens a Gradio interface where you can upload DEMs, adjust confidence thresholds, and visualize detections with adjustable overlay opacity.

## Architecture

### Detection Pipeline

```
Raw DEM (GeoTIFF)
    ↓
Terrain Visualization (SVF, Openness, Slope)
    ↓  → (3, H, W) float32
Tiled Inference (480×480 patches, 50% overlap)
    ↓
Per-class Binary Models (DeepLabV3+ ResNet-101)
    ↓  → per-class probability maps
Test-Time Augmentation (8 orientations)
    ↓
Post-processing (threshold + blob filtering)
    ↓
DetectionResult (classes, confidence, geo)
    ↓
Export (GeoTIFF, GeoJSON, CSV)
```

### Model Architecture

**v2 (recommended):** Three separate binary segmentation models, one per class:
- Building/mound detector — DeepLabV3+ with ResNet-101
- Platform detector — DeepLabV3+ with ResNet-101
- Aguada detector — DeepLabV3+ with ResNet-101

Each model uses:
- **Focal + Dice combo loss** for extreme class imbalance (aguadas are <0.3% of pixels)
- **ImageNet-pretrained encoder** with AdamW optimizer
- **Cosine annealing** learning rate schedule with warmup
- **Heavy augmentation**: rotation, flip, brightness, noise, channel shuffle, oversampling

**v1 (legacy):** Single multi-class U-Net (ResNet-34) producing 4-class softmax output.

### Feature Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Background | No archaeological feature |
| 1 | Building | Residential mounds, temples, pyramids |
| 2 | Platform | Elevated platforms, plazas |
| 3 | Aguada | Water reservoirs, depressions |

### Terrain Visualizations

MayaScan computes three complementary terrain features from raw DEMs:

1. **Sky-View Factor (SVF)** — openness of the sky hemisphere above each point. Depressions (aguadas, building foundations) show low SVF.
2. **Positive Openness** — angular measure of terrain openness. Ridges and elevated features show high openness.
3. **Slope** — terrain gradient in degrees. Artificial structures typically have steeper slopes than natural terrain.

These are stacked as a 3-channel image fed to the segmentation models.

## Training

### Dataset: Chactun LiDAR

Training uses the [Chactun archaeological LiDAR dataset](https://figshare.com/articles/dataset/chactun/22202395) — 2,094 tiles at 480×480 pixels with per-class binary masks for buildings, platforms, and aguadas.

### Train Your Own Models

```bash
# Train all classes (building, platform, aguada)
python train_v2.py --epochs 80 --batch-size 4

# Train a single class
python train_v2.py --cls building --epochs 100

# Resume from checkpoint (automatic if model file exists)
python train_v2.py --cls platform --epochs 80
```

Training supports automatic checkpoint resume — if a model file exists, training resumes from the last saved epoch with optimizer and scheduler state restored.

## CLI Reference

```
mayascan scan <input> [-o OUTPUT] [--model-dir DIR] [--threshold T]
                      [--resolution R] [--no-tta] [--min-blob N] [--device D]

mayascan info [MODEL_DIR]

mayascan download [--model-dir DIR] [--repo REPO_ID] [--force]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | 0.5 | Confidence threshold for detections |
| `--resolution` | 0.5 | DEM pixel size in metres (auto-detected from GeoTIFF) |
| `--no-tta` | — | Disable test-time augmentation (faster, less accurate) |
| `--min-blob` | 50 | Minimum feature size in pixels |
| `--device` | auto | Compute device: `cuda`, `mps`, or `cpu` |
| `--model-dir` | `models/` | Directory containing v2 per-class models |
| `--model` | — | Path to v1 single model (forces v1 mode) |

## Project Structure

```
mayascan/
├── __init__.py          # Public API (visualize, detect, detect_v2, process_dem)
├── detect.py            # Detection pipeline (v1 multi-class, v2 per-class)
├── visualize.py         # SVF, openness, slope from DEMs
├── tile.py              # Tile slicing and stitching
├── export.py            # CSV, GeoJSON, GeoTIFF export
├── cli.py               # CLI (scan, info, download)
├── models/
│   └── unet.py          # U-Net wrapper
├── app.py               # Gradio web interface
├── train_v2.py          # v2 training pipeline
├── evaluate.py          # Model evaluation and benchmarking
└── tests/               # Test suite (47 tests)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- segmentation-models-pytorch >= 0.3.3
- numpy, scipy, Pillow, tqdm

Optional: rasterio (georeferenced I/O), rvt-py (high-quality visualizations), gradio (web UI), huggingface-hub (model download).

## Links

- [GitHub](https://github.com/fascinatedbyeverything/mayascan)
- [HuggingFace Models](https://huggingface.co/fascinatedbyeverything/mayascan)
- [HuggingFace Demo](https://huggingface.co/spaces/fascinatedbyeverything/mayascan)
- [Chactun Dataset](https://figshare.com/articles/dataset/chactun/22202395)

## References

- Somrak et al. (2020) — "Learning to classify structures in ALS-derived visualizations of ancient Maya settlements." *Remote Sensing*
- ECML PKDD 2021 Discovery Challenge — Maya archaeological feature detection competition
- Kokalj & Somrak (2019) — "Why not a single image? Combining visualizations to facilitate fieldwork and on-screen mapping." *Remote Sensing*

## License

MIT
