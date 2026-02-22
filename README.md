# MayaScan

Open-source archaeological LiDAR feature detection for Maya archaeology.

Upload a LiDAR DEM, get back a map of probable ancient structures — buildings, platforms, and aguadas (water reservoirs).

## Features

- **Per-class binary segmentation** — separate DeepLabV3+ (ResNet-101) models for each feature class, trained with competition-winning techniques from ECML PKDD 2021 Maya Challenge
- **Terrain visualization** — automatic SVF, openness, and slope computation from raw DEMs (using rvt-py or scipy fallback)
- **Test-Time Augmentation (TTA)** — 8-fold augmentation (4 rotations x 2 flips) for robust predictions
- **Multi-scale inference** — run at multiple tile sizes and merge via ensemble for higher accuracy
- **Model ensemble** — probability averaging and majority voting for combining multiple models
- **K-fold cross-validation** — reproducible fold splitting for competition-grade training
- **Georeferenced output** — CRS and affine transform propagated from input GeoTIFFs to all exports
- **Multiple export formats** — GeoTIFF, GeoJSON (polygon contours), CSV, Shapefile, KML, overlay PNG, HTML/JSON reports
- **Feature analysis** — extract, filter, and summarize individual features by area, confidence, and class
- **Competition-grade augmentation** — elastic deformation, CutMix, rotation, flip, brightness, noise, channel shuffle
- **Focal + Dice loss** — handles extreme class imbalance (aguadas are <0.3% of pixels)
- **Morphological post-processing** — closing/opening to clean feature boundaries + blob filtering
- **Batch processing** — process entire survey directories in one command
- **Web interface** — Gradio app with interactive overlay visualization and multi-scale option
- **HuggingFace integration** — auto-download pre-trained models, upload with model card generation
- **Reusable training pipeline** — modular data loading, losses, augmentation, and training loop

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

# Multi-scale inference (higher accuracy, slower)
python -m mayascan scan input.tif --multiscale

# With options
python -m mayascan scan input.tif --threshold 0.6 --no-tta --device mps
```

### Train Models

```bash
# Train all classes using the library modules
python -m mayascan train --data-dir chactun_data/extracted --save-dir models/

# Train a single class
python -m mayascan train --data-dir chactun_data/extracted --cls building --epochs 100

# Legacy standalone training script
python train_v2.py --epochs 80 --batch-size 4
```

### Python API

```python
import mayascan

# Full pipeline: DEM → visualization → detection
result = mayascan.process_dem(dem_array, resolution=0.5)

# Step by step
viz = mayascan.visualize(dem, resolution=0.5)        # (3, H, W) float32
result = mayascan.detect_v2(viz, model_dir="models/")

# Multi-scale detection (higher accuracy)
result = mayascan.run_multiscale_detection(viz, model_dir="models/")

# With georeferencing
data, geo = mayascan.read_raster("input.tif")        # auto-reads CRS
viz = mayascan.visualize(data, resolution=geo.resolution)
result = mayascan.detect_v2(viz)
result.geo = geo  # propagate for georeferenced export

# Export
from mayascan.export import to_csv, to_geojson, to_geotiff, to_kml, to_overlay_png
to_csv(result, "detections.csv")
to_geojson(result, "detections.geojson")     # polygon contours
to_geotiff(result, "detections.tif")         # georeferenced class map
to_kml(result, "detections.kml")             # Google Earth
to_overlay_png(result, viz, "overlay.png")   # blended visualization

# Feature-level analysis
features = mayascan.extract_features(result)
large = mayascan.filter_features(features, min_area=50.0, min_confidence=0.8)
summary = mayascan.feature_summary(large)

# Model ensemble
from mayascan.ensemble import merge_results
merged = merge_results([result1, result2], method="average")

# Reports
mayascan.save_report(result, "report.html", format="html")
```

### Web App

```bash
python app.py
```

Opens a Gradio interface where you can upload DEMs, adjust confidence thresholds, enable multi-scale inference, and visualize detections with adjustable overlay opacity. Downloads include all export formats.

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
Multi-Scale Merge (optional: 320/480/640 tiles)
    ↓
Post-processing (threshold + morphological cleanup + blob filtering)
    ↓
DetectionResult (classes, confidence, geo)
    ↓
Export (GeoTIFF, GeoJSON, CSV, Shapefile, KML, Overlay PNG, Reports)
```

### Model Architecture

**v2 (recommended):** Three separate binary segmentation models, one per class:
- Building/mound detector — DeepLabV3+ with ResNet-101
- Platform detector — DeepLabV3+ with ResNet-101
- Aguada detector — DeepLabV3+ with ResNet-101

Each model uses:
- **Focal + Dice combo loss** for extreme class imbalance
- **ImageNet-pretrained encoder** with AdamW optimizer
- **Cosine annealing** learning rate schedule with warmup
- **Heavy augmentation**: rotation, flip, brightness, noise, channel shuffle, elastic deformation, CutMix, oversampling

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

Training uses the [Chactun archaeological LiDAR dataset](https://figshare.com/articles/dataset/chactun/22202395) — 2,094 tiles at 480x480 pixels with per-class binary masks for buildings, platforms, and aguadas.

### Train Your Own Models

```bash
# Using the CLI (recommended)
python -m mayascan train --data-dir chactun_data/extracted --epochs 80

# Using the standalone script (advanced)
python train_v2.py --cls all --epochs 80 --batch-size 4

# Cross-validation
from mayascan.crossval import create_folds, fold_summary
folds = create_folds("chactun_data/extracted/lidar", n_folds=5)
print(fold_summary(folds))
```

Training supports automatic checkpoint resume — if a model file exists, training resumes from the last saved epoch with optimizer and scheduler state restored.

### Custom Augmentation

```python
from mayascan.augment import augment_sample, cutmix

# Full augmentation pipeline
aug_img, aug_mask = augment_sample(image, mask, use_elastic=True)

# CutMix: paste crops between samples
mixed_img, mixed_mask = cutmix(img1, mask1, img2, mask2)
```

## CLI Reference

```
mayascan scan <input> [-o OUTPUT] [--model-dir DIR] [--threshold T]
                      [--resolution R] [--no-tta] [--multiscale]
                      [--min-blob N] [--device D]

mayascan train --data-dir DIR [--save-dir DIR] [--cls CLASS]
               [--arch ARCH] [--encoder ENC] [--epochs N]
               [--batch-size N] [--lr LR] [--device D]

mayascan evaluate [--model-dir DIR] [--arch ARCH] [--encoder ENC]
                  [--threshold T] [--save-viz DIR] [--device D]

mayascan info [MODEL_DIR]

mayascan download [--model-dir DIR] [--repo REPO_ID] [--force]

mayascan version
```

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | 0.5 | Confidence threshold for detections |
| `--resolution` | 0.5 | DEM pixel size in metres (auto-detected from GeoTIFF) |
| `--no-tta` | — | Disable test-time augmentation (faster, less accurate) |
| `--multiscale` | — | Multi-scale inference at 3 tile sizes (slower, more accurate) |
| `--min-blob` | 50 | Minimum feature size in pixels |
| `--device` | auto | Compute device: `cuda`, `mps`, or `cpu` |
| `--model-dir` | `models/` | Directory containing v2 per-class models |
| `--model` | — | Path to v1 single model (forces v1 mode) |

## Project Structure

```
mayascan/
├── __init__.py          # Public API
├── config.py            # Centralized configuration constants
├── detect.py            # Detection pipeline (v1 multi-class, v2 per-class binary)
├── visualize.py         # SVF, openness, slope from DEMs
├── tile.py              # Tile slicing and stitching with overlap blending
├── export.py            # CSV, GeoJSON, GeoTIFF, Shapefile, KML, overlay PNG
├── report.py            # Report generation (text, JSON, HTML)
├── features.py          # Feature extraction, filtering, and summary
├── metrics.py           # IoU, F1, precision, recall, confusion matrix
├── ensemble.py          # Model ensemble (probability averaging, majority voting)
├── multiscale.py        # Multi-scale inference with ensemble merge
├── augment.py           # Competition-grade data augmentation
├── losses.py            # Focal, Dice, and FocalDice loss functions
├── data.py              # PyTorch dataset with oversampling
├── train.py             # Reusable training loop
├── crossval.py          # K-fold cross-validation
├── crs.py               # Coordinate reference system utilities
├── classify.py          # Ground-point classification (PDAL)
├── cli.py               # CLI (scan, train, evaluate, info, download, version)
├── models/
│   └── unet.py          # U-Net wrapper
├── app.py               # Gradio web interface
├── train_v2.py          # Standalone training script
├── evaluate.py          # Model evaluation
├── upload_models.py     # HuggingFace model upload with model card
└── tests/               # 182 tests
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- segmentation-models-pytorch >= 0.3.3
- numpy, scipy, Pillow, tqdm

Optional: rasterio (georeferenced I/O), rvt-py (high-quality visualizations), gradio (web UI), huggingface-hub (model download), geopandas (Shapefile export).

## Links

- [GitHub](https://github.com/fascinatedbyeverything/mayascan)
- [HuggingFace Models](https://huggingface.co/fascinated23/mayascan)
- [HuggingFace Demo](https://huggingface.co/spaces/fascinated23/mayascan)
- [Chactun Dataset](https://figshare.com/articles/dataset/chactun/22202395)

## References

- Somrak et al. (2020) — "Learning to classify structures in ALS-derived visualizations of ancient Maya settlements." *Remote Sensing*
- ECML PKDD 2021 Discovery Challenge — Maya archaeological feature detection competition
- Kokalj & Somrak (2019) — "Why not a single image? Combining visualizations to facilitate fieldwork and on-screen mapping." *Remote Sensing*

## License

MIT
