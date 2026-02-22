# MayaScan — Open-Source Archaeological LiDAR Feature Detection

**Date:** 2026-02-21
**Status:** Approved
**Author:** Chris Holmes + Claude

## Problem Statement

Archaeologists studying ancient Maya civilization have access to massive LiDAR datasets covering thousands of square kilometers of jungle canopy in Mexico and Guatemala. However, processing these scans into actionable archaeological maps is bottlenecked by:

1. **Proprietary software** — Professional LiDAR processing tools (TerraScan, LAStools Pro, ArcGIS) cost $5,000–$20,000/year per license
2. **Centralized expertise** — NCALM (National Center for Airborne Laser Mapping) is the near-monopoly provider for Maya-region LiDAR processing
3. **Massive backlogs** — Mexico's national forest carbon LiDAR survey sat unprocessed for years until an archaeologist stumbled on it; the PACUNAM Initiative scanned 2,144 km² but analysis is ongoing years later
4. **Manual interpretation** — Even after processing, identifying structures in DEM visualizations is done by eye, taking months per survey area

MayaScan is an open-source system that automates the full pipeline from raw LiDAR point cloud to annotated archaeological feature map, accessible to any archaeologist with a web browser.

## Inspiration

Inspired by the work of archaeologist Ed Barnhart (Maya Exploration Center) and the growing body of research demonstrating that deep learning can detect Maya structures in LiDAR data with 60–95% accuracy.

## Architecture

### Four-Stage Pipeline

```
LAS/LAZ Upload → Ground Classification → Raster Visualization → AI Feature Detection
```

**Stage 1: Ingest**
- Accept LAS/LAZ point cloud files via web upload or Python API
- Validate file format, compute point cloud statistics (count, extent, density, CRS)
- Reject files that are too small or malformed

**Stage 2: Ground Classification (PDAL + SMRF)**
- Strip vegetation and above-ground objects using Simple Morphological Filter
- Generate bare-earth Digital Elevation Model at 0.5m resolution
- Parameters tuned for archaeological feature preservation (window=30m, low slope threshold)
- PDAL pipeline:
  - `filters.outlier` (statistical, mean_k=12) — remove noise
  - `filters.smrf` (cell=1.0, slope=0.15, window=30, threshold=0.5) — ground classification
  - `writers.gdal` (resolution=0.5, output_type=idw) — DEM rasterization

**Stage 3: Raster Visualization (rvt-py)**
- Generate three direction-independent visualization channels from the DEM:
  - Sky-View Factor (SVF) — reveals depressions and elevated features
  - Positive Openness — highlights convex terrain (mounds, platforms)
  - Slope — edges of structures produce sharp gradients
- These three channels were validated by the Chactún dataset research as optimal for CNN-based detection

**Stage 4: AI Feature Detection (U-Net)**
- Tile visualizations into 480×480 px patches with 50% overlap
- Run U-Net semantic segmentation inference
- Stitch tiles with overlap blending to eliminate edge artifacts
- Output 4-class feature map: background, buildings, platforms, aguadas (reservoirs)
- Generate confidence heatmap

### Model Architecture

**U-Net with ResNet34 encoder** (via segmentation-models-pytorch)

- **Input:** 3-channel float32 tensor (SVF, openness, slope), 480×480 px
- **Output:** 4-class segmentation mask, 480×480 px
- **Encoder:** ResNet34 pre-trained on ImageNet (transfer learning)
- **Loss:** Weighted cross-entropy (class weights inversely proportional to frequency)
- **Augmentation:** Random rotation (0/90/180/270°), horizontal/vertical flip, random crop

**Training Data:** Chactún ML-ready dataset (Kokalj et al., 2023)
- 2,094 annotated tiles, 130 km² coverage
- 9,303 buildings, 2,110 platforms, 95 aguadas
- 480×480 px tiles at 0.5m resolution
- Source: https://figshare.com/articles/dataset/22202395

**Class Imbalance Mitigation:**
- Weighted loss function (aguadas ~100× weight vs buildings)
- Oversampling tiles containing rare classes during training
- Data augmentation (direction-independent visualizations allow full rotation)

### Web Interface (Gradio)

Three-panel cloud-hosted web application:

1. **Upload Panel** — Drag-and-drop LAS/LAZ, point cloud stats display
2. **Processing Panel** — Progress indicators per stage, intermediate previews (DEM, visualizations)
3. **Results Panel** — Interactive map with:
   - Hillshade base layer
   - Color-coded feature overlay (buildings=red, platforms=yellow, aguadas=blue)
   - Confidence threshold slider
   - Export: GeoTIFF, GeoJSON (polygons), CSV (centroids + class + confidence)

### Python Library (`mayascan`)

```python
import mayascan

# Full pipeline
results = mayascan.process("survey.laz")

# Step by step
dem = mayascan.classify("survey.laz", resolution=0.5)
viz = mayascan.visualize(dem, methods=["svf", "openness", "slope"])
features = mayascan.detect(viz, confidence_threshold=0.6)
features.to_geojson("structures.geojson")
```

## Deployment

- **Web app:** Hugging Face Spaces (Gradio), free hosting with optional GPU
- **Model weights:** Hugging Face Hub
- **Training:** Google Colab notebook (free T4 GPU)
- **Package:** PyPI (`pip install mayascan`)

## Repository Structure

```
mayascan/
├── mayascan/               # Python package
│   ├── __init__.py
│   ├── classify.py         # PDAL ground classification
│   ├── visualize.py        # RVT raster generation
│   ├── detect.py           # Model inference + tiling
│   ├── tile.py             # Tile slicing + stitching
│   ├── export.py           # GeoJSON/CSV/GeoTIFF export
│   └── models/
│       └── unet.py         # U-Net architecture wrapper
├── app.py                  # Gradio web interface
├── notebooks/
│   ├── train.ipynb         # Training notebook (Colab-ready)
│   └── quickstart.ipynb    # Usage tutorial
├── tests/
│   ├── test_classify.py
│   ├── test_visualize.py
│   ├── test_detect.py
│   └── fixtures/           # Small test LAS files
├── pyproject.toml
├── README.md
└── LICENSE                 # MIT
```

## Training Data Sources

| Dataset | Region | Size | Classes | Access |
|---------|--------|------|---------|--------|
| Chactún ML-ready (Kokalj 2023) | Yucatan, Mexico | 130 km², 2,094 tiles | Buildings, platforms, aguadas | Figshare (open) |
| PACUNAM LiDAR (2016) | Guatemala | 2,144 km² | Manually labeled structures | OpenTopography / tDAR |
| Mayapan (NCALM) | Yucatan, Mexico | ~4 km² | Point cloud + rasters | OpenTopography (open) |

## Key Dependencies

- `pdal` — Point cloud processing and ground classification
- `rvt-py` — Relief Visualization Toolbox Python bindings
- `segmentation-models-pytorch` — U-Net with pre-trained encoders
- `torch` — PyTorch deep learning framework
- `rasterio` — GeoTIFF I/O
- `gradio` — Web interface
- `geopandas` — Vector data export (GeoJSON)
- `numpy`, `scipy` — Numerical processing

## What This Unlocks

An archaeologist with a LiDAR scan can:
1. Open the MayaScan web app in any browser
2. Upload their LAS/LAZ file
3. Receive an annotated map showing probable buildings, platforms, and reservoirs
4. Export results to GIS software for field verification planning

No proprietary software. No expensive licenses. No queue. The model improves over time as archaeologists contribute verified detections back as training data.

## References

- Kokalj et al. (2023). "Machine learning-ready remote sensing data for Maya archaeology." Scientific Data. https://doi.org/10.1038/s41597-023-02455-x
- Bundzel et al. (2020). "Semantic Segmentation of Airborne LiDAR Data in Maya Archaeology." Remote Sensing 12(22):3685
- Zhang et al. (2024). "Unveiling Ancient Maya Settlements Using Aerial LiDAR Image Segmentation." arXiv:2403.05773
- Štular et al. (2021). "Airborne LiDAR Point Cloud Processing for Archaeology." Remote Sensing 13(16):3225
- Canuto et al. (2018). "Ancient lowland Maya complexity as revealed by airborne laser scanning." Science 361(6409)
- Archaeoscape (2024). "Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era." arXiv:2412.05203
- Richards-Rissetto & Newton. "LiDAR deep learning for ancient Maya archaeology." GIM International
