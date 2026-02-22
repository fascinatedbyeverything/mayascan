"""MayaScan CLI: run archaeological feature detection on LiDAR DEMs.

Usage:
    python -m mayascan scan input.tif -o results/
    python -m mayascan scan input.tif --model-dir models/ --threshold 0.6
    python -m mayascan scan directory/ -o results/    # batch processing
    python -m mayascan info models/
    python -m mayascan download                        # download models from HF
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _scan_single(input_path: str, args: argparse.Namespace, out_dir: Path) -> None:
    """Process a single DEM file."""
    import mayascan
    from mayascan.detect import discover_v2_models, run_detection_v2, GeoInfo
    from mayascan.export import to_csv, to_geojson, to_geotiff, to_overlay_png
    from mayascan.report import save_report

    t0 = time.time()

    # Load input with georeferencing
    print(f"\nLoading: {input_path}")
    data, geo = mayascan.read_raster(input_path)

    if geo.crs:
        print(f"  CRS: {geo.crs}")
        print(f"  Resolution: {geo.resolution:.4f} m/px")
        resolution = geo.resolution
    else:
        resolution = args.resolution

    # Detect if this is a pre-computed visualization (3-band, uint8-range)
    # or a raw DEM (single-band, float elevation values)
    is_viz = False
    if data.ndim == 3 and data.shape[-1] == 3:
        is_viz = True
        viz = (data / 255.0).transpose(2, 0, 1).astype(np.float32)
        print(f"  Pre-computed visualization: {data.shape[0]} x {data.shape[1]} pixels (3-band)")
    elif data.ndim == 3 and data.shape[0] == 3:
        is_viz = True
        viz = data / 255.0 if data.max() > 1.0 else data
        print(f"  Pre-computed visualization: {data.shape[1]} x {data.shape[2]} pixels (3-band)")
    else:
        dem = data if data.ndim == 2 else data[0] if data.ndim == 3 else data
        print(f"  Raw DEM: {dem.shape[0]} x {dem.shape[1]} pixels")
        print(f"  Resolution: {resolution} m/px")
        print(f"  Coverage: {dem.shape[1] * resolution:.0f} x {dem.shape[0] * resolution:.0f} m")

    if not is_viz:
        print("  Computing terrain visualizations (SVF, openness, slope)...")
        viz = mayascan.visualize(dem, resolution=resolution)
        print(f"  Done ({time.time() - t0:.1f}s)")
    else:
        print("  Skipping visualization (already pre-computed)")

    # Run detection
    t1 = time.time()
    model_dir = args.model_dir

    if args.model:
        print(f"  Running v1 detection (single model)...")
        result = mayascan.detect(
            viz,
            model_path=args.model,
            confidence_threshold=args.threshold,
        )
    elif model_dir and discover_v2_models(model_dir):
        v2_models = discover_v2_models(model_dir)
        print(f"  Running v2 detection ({len(v2_models)} per-class models)...")
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
        print("  No model specified — running with random weights (demo mode)")
        result = mayascan.detect(viz, confidence_threshold=args.threshold)

    # Attach georeferencing to result
    result.geo = geo

    print(f"  Inference done ({time.time() - t1:.1f}s)")

    # Statistics
    from scipy.ndimage import label

    print(f"\n  Detection Results:")
    print(f"  {'=' * 38}")
    total = 0
    for cls_id, cls_name in result.class_names.items():
        if cls_id == 0:
            continue
        mask = result.classes == cls_id
        if not mask.any():
            print(f"    {cls_name:>10s}: 0 features")
            continue
        labeled, n = label(mask)
        area = int(mask.sum()) * resolution * resolution
        print(f"    {cls_name:>10s}: {n} features ({area:.0f} m\u00b2)")
        total += n
    print(f"    {'TOTAL':>10s}: {total} features")

    # Export
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    csv_path = to_csv(result, out_dir / f"{stem}_detections.csv", pixel_size=resolution)
    geojson_path = to_geojson(result, out_dir / f"{stem}_detections.geojson", pixel_size=resolution)
    tiff_path = to_geotiff(result, out_dir / f"{stem}_detections.tif", pixel_size=resolution)

    # Save visualization and overlay PNGs
    viz_rgb = (np.transpose(viz, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image

    Image.fromarray(viz_rgb).save(str(out_dir / f"{stem}_visualization.png"))
    overlay_path = to_overlay_png(result, viz, out_dir / f"{stem}_overlay.png")

    # Save reports (text + JSON + HTML)
    report_txt = save_report(result, out_dir / f"{stem}_report.txt",
                             input_path=input_path, pixel_size=resolution, format="text")
    report_json = save_report(result, out_dir / f"{stem}_report.json",
                              input_path=input_path, pixel_size=resolution, format="json")
    report_html = save_report(result, out_dir / f"{stem}_report.html",
                              input_path=input_path, pixel_size=resolution, format="html")

    print(f"\n  Exported to {out_dir}/:")
    print(f"    {csv_path.name}")
    print(f"    {geojson_path.name}")
    print(f"    {tiff_path.name}")
    print(f"    {stem}_visualization.png")
    print(f"    {overlay_path.name}")
    print(f"    {report_txt.name}")
    print(f"    {report_json.name}")
    print(f"    {report_html.name}")
    if geo.crs:
        print(f"    (georeferenced: {geo.crs})")
    print(f"  Time: {time.time() - t0:.1f}s")

    return total


def cmd_scan(args: argparse.Namespace) -> None:
    """Run detection on a DEM file or directory of files."""
    input_path = Path(args.input)
    out_dir = Path(args.output)

    if input_path.is_dir():
        # Batch mode: process all raster files in directory
        raster_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".tif", ".tiff", ".npy")
        )
        if not raster_files:
            print(f"No raster files found in {input_path}")
            sys.exit(1)

        t_start = time.time()
        print(f"Batch processing: {len(raster_files)} files from {input_path}")
        print("=" * 50)

        grand_total = 0
        for i, fp in enumerate(raster_files, 1):
            print(f"\n[{i}/{len(raster_files)}] {fp.name}")
            count = _scan_single(str(fp), args, out_dir)
            grand_total += count

        print(f"\n{'=' * 50}")
        print(f"Batch complete: {len(raster_files)} files, {grand_total} total features")
        print(f"Total time: {time.time() - t_start:.1f}s")
    else:
        # Single file mode
        t_start = time.time()
        _scan_single(str(input_path), args, out_dir)
        print(f"\nTotal time: {time.time() - t_start:.1f}s")


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


from mayascan.config import (
    HF_REPO_ID, CONFIDENCE_THRESHOLD, MIN_BLOB_SIZE, DEFAULT_RESOLUTION,
    V2_ARCH, V2_ENCODER, EPOCHS, BATCH_SIZE, LEARNING_RATE,
)


def cmd_download(args: argparse.Namespace) -> None:
    """Download pre-trained models from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("huggingface-hub is required. Install with:")
        print("  pip install mayascan[hub]")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo or HF_REPO_ID
    print(f"Downloading models from {repo_id}...")

    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Failed to list files from {repo_id}: {e}")
        sys.exit(1)

    model_files = [f for f in files if f.endswith(".pth")]
    if not model_files:
        print("No model files found in the repository.")
        sys.exit(1)

    print(f"Found {len(model_files)} model files:")
    for mf in model_files:
        print(f"  {mf}")

    for mf in model_files:
        dest = model_dir / os.path.basename(mf)
        if dest.exists() and not args.force:
            print(f"  Skipping {mf} (already exists, use --force to overwrite)")
            continue
        print(f"  Downloading {mf}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=mf,
            local_dir=str(model_dir),
        )
        print(f"    -> {downloaded}")

    print(f"\nModels saved to {model_dir}/")


def cmd_train(args: argparse.Namespace) -> None:
    """Train per-class binary segmentation models."""
    from mayascan.train import train_all, train_class
    from mayascan.data import list_available_classes, count_tiles

    import torch

    # Device selection
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    lidar_dir = os.path.join(args.data_dir, "lidar")
    mask_dir = os.path.join(args.data_dir, "masks")

    n_tiles = count_tiles(lidar_dir)
    available = list_available_classes(mask_dir)
    print(f"MayaScan Training")
    print(f"  Device:  {device}")
    print(f"  Data:    {args.data_dir} ({n_tiles} tiles)")
    print(f"  Classes: {', '.join(available)}")
    print(f"  Arch:    {args.arch} ({args.encoder})")
    print(f"  Epochs:  {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    classes = None if args.cls == "all" else [args.cls]

    train_all(
        lidar_dir=lidar_dir,
        mask_dir=mask_dir,
        save_dir=args.save_dir,
        classes=classes,
        arch=args.arch,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_tta=args.tta,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run model evaluation on the validation set."""
    # Delegate to the standalone evaluate.py script's logic
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate import evaluate_v2, evaluate_v1, print_results, DATA_DIR

    import torch

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_dir = args.data_dir or DATA_DIR

    print(f"MayaScan Evaluation")
    print(f"  Device: {device}")
    print(f"  Data:   {data_dir}")

    t0 = time.time()

    if args.model:
        metrics = evaluate_v1(
            model_path=args.model,
            device=device,
            use_tta=args.tta,
            threshold=args.threshold,
            save_viz=args.save_viz,
        )
        model_label = "v1 U-Net (resnet34)"
    else:
        metrics = evaluate_v2(
            model_dir=args.model_dir,
            arch=args.arch,
            encoder=args.encoder,
            device=device,
            use_tta=args.tta,
            threshold=args.threshold,
            min_blob_size=args.min_blob,
            save_viz=args.save_viz,
        )
        model_label = f"v2 {args.arch}/{args.encoder}"

    print(f"\n  Evaluation completed in {time.time() - t0:.1f}s")
    print_results(metrics, model_label)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mayascan",
        description="MayaScan: Archaeological LiDAR feature detection",
    )
    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_p = subparsers.add_parser("scan", help="Run detection on a DEM file or directory")
    scan_p.add_argument("input", help="Input DEM file (.tif or .npy) or directory")
    scan_p.add_argument("-o", "--output", default="results", help="Output directory")
    scan_p.add_argument("--model-dir", default="models", help="v2 model directory")
    scan_p.add_argument("--model", default=None, help="v1 model path (.pth)")
    scan_p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    scan_p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="DEM resolution (m/px)")
    scan_p.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    scan_p.add_argument("--min-blob", type=int, default=MIN_BLOB_SIZE, help="Min blob size (pixels)")
    scan_p.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")

    # info command
    info_p = subparsers.add_parser("info", help="Show model info")
    info_p.add_argument("model_dir", nargs="?", default="models", help="Model directory")

    # download command
    dl_p = subparsers.add_parser("download", help="Download pre-trained models from HuggingFace")
    dl_p.add_argument("--model-dir", default="models", help="Directory to save models")
    dl_p.add_argument("--repo", default=None, help=f"HuggingFace repo ID (default: {HF_REPO_ID})")
    dl_p.add_argument("--force", action="store_true", help="Overwrite existing models")

    # train command
    train_p = subparsers.add_parser("train", help="Train per-class binary segmentation models")
    train_p.add_argument("--data-dir", required=True, help="Data directory (with lidar/ and masks/ subdirs)")
    train_p.add_argument("--save-dir", default="models", help="Directory to save model checkpoints")
    train_p.add_argument("--cls", default="all", help="Class to train (building/platform/aguada/all)")
    train_p.add_argument("--arch", default=V2_ARCH, help="Model architecture")
    train_p.add_argument("--encoder", default=V2_ENCODER, help="Encoder backbone")
    train_p.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    train_p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    train_p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    train_p.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    train_p.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")

    # evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Evaluate models on validation set")
    eval_p.add_argument("--model-dir", default="models", help="v2 model directory")
    eval_p.add_argument("--model", default=None, help="v1 model path (.pth)")
    eval_p.add_argument("--arch", default="deeplabv3plus", help="v2 architecture")
    eval_p.add_argument("--encoder", default="resnet101", help="v2 encoder backbone")
    eval_p.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    eval_p.add_argument("--min-blob", type=int, default=MIN_BLOB_SIZE, help="Min blob size (pixels)")
    eval_p.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    eval_p.add_argument("--save-viz", default=None, help="Save overlay visualizations to directory")
    eval_p.add_argument("--data-dir", default=None, help="Validation data directory")
    eval_p.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")

    # version command
    subparsers.add_parser("version", help="Show MayaScan version")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "version":
        import mayascan
        print(f"MayaScan v{mayascan.__version__}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
