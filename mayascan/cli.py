"""MayaScan CLI: run archaeological feature detection on LiDAR DEMs.

Usage:
    python -m mayascan scan input.tif -o results/
    python -m mayascan scan input.tif --model-dir models/ --threshold 0.6
    python -m mayascan info models/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _load_raster(path: str) -> np.ndarray:
    """Load a raster from GeoTIFF or .npy file. Returns raw array."""
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                if src.count == 1:
                    return src.read(1).astype(np.float32)
                else:
                    return src.read().astype(np.float32)
        except ImportError:
            from PIL import Image

            return np.array(Image.open(path), dtype=np.float32)
    elif suffix == ".npy":
        return np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .tif or .npy")


def cmd_scan(args: argparse.Namespace) -> None:
    """Run detection on a DEM file."""
    import mayascan
    from mayascan.detect import discover_v2_models, run_detection_v2
    from mayascan.export import to_csv, to_geojson, to_geotiff

    t0 = time.time()

    # Load input
    print(f"Loading: {args.input}")
    data = _load_raster(args.input)

    # Detect if this is a pre-computed visualization (3-band, uint8-range)
    # or a raw DEM (single-band, float elevation values)
    is_viz = False
    if data.ndim == 3 and data.shape[-1] == 3:
        # (H, W, 3) — likely a pre-computed visualization tile
        is_viz = True
        viz = (data / 255.0).transpose(2, 0, 1).astype(np.float32)  # -> (3, H, W) [0,1]
        print(f"  Pre-computed visualization: {data.shape[0]} x {data.shape[1]} pixels (3-band)")
    elif data.ndim == 3 and data.shape[0] == 3:
        # (3, H, W) — already in CHW format
        is_viz = True
        viz = data / 255.0 if data.max() > 1.0 else data
        print(f"  Pre-computed visualization: {data.shape[1]} x {data.shape[2]} pixels (3-band)")
    else:
        dem = data if data.ndim == 2 else data[0] if data.ndim == 3 else data
        print(f"  Raw DEM: {dem.shape[0]} x {dem.shape[1]} pixels")
        print(f"  Resolution: {args.resolution} m/px")
        print(f"  Coverage: {dem.shape[1] * args.resolution:.0f} x {dem.shape[0] * args.resolution:.0f} m")

    if not is_viz:
        # Compute visualizations from raw DEM
        print("\nComputing terrain visualizations (SVF, openness, slope)...")
        viz = mayascan.visualize(dem, resolution=args.resolution)
        print(f"  Done ({time.time() - t0:.1f}s)")
    else:
        print("  Skipping visualization (already pre-computed)")

    # Run detection
    t1 = time.time()
    model_dir = args.model_dir

    # Explicit --model flag forces v1; otherwise auto-detect v2
    if args.model:
        print(f"\nRunning v1 detection (single model)...")
        result = mayascan.detect(
            viz,
            model_path=args.model,
            confidence_threshold=args.threshold,
        )
    elif model_dir and discover_v2_models(model_dir):
        v2_models = discover_v2_models(model_dir)
        print(f"\nRunning v2 detection ({len(v2_models)} per-class models)...")
        print(f"  TTA: {'enabled (8x)' if args.tta else 'disabled'}")
        result = run_detection_v2(
            viz,
            model_dir=model_dir,
            confidence_threshold=args.threshold,
            use_tta=args.tta,
            min_blob_size=args.min_blob,
            device=args.device,
        )
    else:
        print("\nNo model specified — running with random weights (demo mode)")
        result = mayascan.detect(viz, confidence_threshold=args.threshold)

    print(f"  Done ({time.time() - t1:.1f}s)")

    # Statistics
    from scipy.ndimage import label

    print("\nDetection Results:")
    print("=" * 40)
    total = 0
    for cls_id, cls_name in result.class_names.items():
        if cls_id == 0:
            continue
        mask = result.classes == cls_id
        if not mask.any():
            print(f"  {cls_name:>10s}: 0 features")
            continue
        labeled, n = label(mask)
        area = int(mask.sum()) * args.resolution * args.resolution
        print(f"  {cls_name:>10s}: {n} features ({area:.0f} m\u00b2)")
        total += n
    print(f"  {'TOTAL':>10s}: {total} features")

    # Export
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    csv_path = to_csv(result, out_dir / f"{stem}_detections.csv", pixel_size=args.resolution)
    geojson_path = to_geojson(result, out_dir / f"{stem}_detections.geojson", pixel_size=args.resolution)
    tiff_path = to_geotiff(result, out_dir / f"{stem}_detections.tif", pixel_size=args.resolution)

    # Save visualization as PNG
    viz_rgb = (np.transpose(viz, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image

    Image.fromarray(viz_rgb).save(str(out_dir / f"{stem}_visualization.png"))

    print(f"\nExported to {out_dir}/:")
    print(f"  {csv_path.name}")
    print(f"  {geojson_path.name}")
    print(f"  {tiff_path.name}")
    print(f"  {stem}_visualization.png")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


def cmd_info(args: argparse.Namespace) -> None:
    """Show info about available models."""
    from mayascan.detect import discover_v2_models

    model_dir = args.model_dir
    print(f"Model directory: {model_dir}")

    v2_models = discover_v2_models(model_dir)
    if v2_models:
        print(f"\nv2 per-class binary models found:")
        for cls_id, path in sorted(v2_models.items()):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Class {cls_id} ({['', 'building', 'platform', 'aguada'][cls_id]}): "
                  f"{os.path.basename(path)} ({size_mb:.0f} MB)")
    else:
        print("\nNo v2 models found.")

    # Check for v1 model
    v1_path = os.path.join(model_dir, "..", "mayascan_unet_best.pth")
    if os.path.isfile(v1_path):
        size_mb = os.path.getsize(v1_path) / (1024 * 1024)
        print(f"\nv1 multi-class model: {os.path.basename(v1_path)} ({size_mb:.0f} MB)")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mayascan",
        description="MayaScan: Archaeological LiDAR feature detection",
    )
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_p = subparsers.add_parser("scan", help="Run detection on a DEM file")
    scan_p.add_argument("input", help="Input DEM file (.tif or .npy)")
    scan_p.add_argument("-o", "--output", default="results", help="Output directory")
    scan_p.add_argument("--model-dir", default="models", help="v2 model directory")
    scan_p.add_argument("--model", default=None, help="v1 model path (.pth)")
    scan_p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    scan_p.add_argument("--resolution", type=float, default=0.5, help="DEM resolution (m/px)")
    scan_p.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    scan_p.add_argument("--min-blob", type=int, default=50, help="Min blob size (pixels)")
    scan_p.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")

    # info command
    info_p = subparsers.add_parser("info", help="Show model info")
    info_p.add_argument("model_dir", nargs="?", default="models", help="Model directory")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
