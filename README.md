# MayaScan

Open-source archaeological LiDAR feature detection for Maya archaeology.

Upload a LiDAR scan, get back a map of probable ancient structures.

## Quick Start

```bash
pip install mayascan
```

```python
import mayascan

# From a DEM array
results = mayascan.process_dem(dem_array)

# Step by step
viz = mayascan.visualize(dem)
features = mayascan.detect(viz, confidence_threshold=0.6)
```

## Web App

```bash
python app.py
```

## License

MIT
