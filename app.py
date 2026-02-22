"""MayaScan Gradio web interface for archaeological LiDAR feature detection.

Upload a DEM (GeoTIFF or .npy) or pre-computed visualization tile,
run deep learning segmentation, and explore results interactively.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from scipy.ndimage import label

import mayascan
from mayascan.detect import CLASS_NAMES, GeoInfo, discover_v2_models, run_detection_v2
from mayascan.export import to_csv, to_geojson, to_geotiff, to_confidence_geotiff

# Default model directory for v2 per-class models
V2_MODEL_DIR = Path(__file__).parent / "models"

# Temp directory — use external drive if available
TMPDIR = os.environ.get("TMPDIR", tempfile.gettempdir())

# ---------------------------------------------------------------------------
# Class colours (RGBA)
# ---------------------------------------------------------------------------

CLASS_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),           # background — transparent
    1: (255, 60, 60, 180),     # building — red
    2: (60, 200, 60, 180),     # platform — green
    3: (50, 120, 255, 180),    # aguada — blue
}


def colorize_classes(classes: np.ndarray) -> np.ndarray:
    """Convert a (H, W) class-index map to an RGBA image."""
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        mask = classes == class_id
        rgba[mask] = color
    return rgba


def blend_overlay(
    base_rgb: np.ndarray,
    overlay_rgba: np.ndarray,
    opacity: float = 0.6,
) -> np.ndarray:
    """Blend a detection overlay onto a visualization base image."""
    base = base_rgb.astype(np.float32) / 255.0
    overlay = overlay_rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0 * opacity)[:, :, np.newaxis]

    blended = base * (1 - alpha) + overlay * alpha
    return (blended * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_upload(
    file: str,
    confidence_threshold: float,
    resolution: float,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    """Process an uploaded DEM file through the full MayaScan pipeline.

    Returns
    -------
    tuple
        (viz_rgb, overlay, blended, export_files, stats)
    """
    # --- Load with georeferencing ---
    data, geo = mayascan.read_raster(file)

    if geo.crs:
        resolution = geo.resolution

    # Detect pre-computed visualization vs raw DEM
    is_viz = False
    if data.ndim == 3 and data.shape[-1] == 3:
        is_viz = True
        viz = (data / 255.0).transpose(2, 0, 1).astype(np.float32)
    elif data.ndim == 3 and data.shape[0] == 3:
        is_viz = True
        viz = data / 255.0 if data.max() > 1.0 else data
    else:
        dem = data if data.ndim == 2 else data[0] if data.ndim == 3 else data

    if not is_viz:
        viz = mayascan.visualize(dem, resolution=resolution)

    # Convert viz to RGB display
    viz_rgb = (np.transpose(viz, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

    # --- Detect ---
    v2_models = discover_v2_models(str(V2_MODEL_DIR))
    if v2_models:
        result = run_detection_v2(
            viz,
            model_dir=str(V2_MODEL_DIR),
            confidence_threshold=confidence_threshold,
        )
    else:
        result = mayascan.detect(viz, confidence_threshold=confidence_threshold)

    result.geo = geo

    # --- Colorize detection ---
    overlay = colorize_classes(result.classes)
    blended = blend_overlay(viz_rgb, overlay, opacity=opacity)

    # --- Statistics ---
    stats_lines = [
        "MayaScan Detection Results",
        "=" * 35,
        "",
    ]

    total_features = 0
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        mask = result.classes == class_id
        if not mask.any():
            stats_lines.append(f"  {class_name.capitalize():>10s}: 0 features")
            continue
        labeled_array, num_features = label(mask)
        pixel_count = int(mask.sum())
        area_m2 = pixel_count * resolution * resolution
        avg_conf = float(result.confidence[mask].mean())
        stats_lines.append(
            f"  {class_name.capitalize():>10s}: {num_features} features "
            f"({area_m2:.0f} m\u00b2, conf: {avg_conf:.2f})"
        )
        total_features += num_features

    h, w = result.classes.shape
    stats_lines.extend([
        "",
        f"  {'Total':>10s}: {total_features} features",
        f"  {'Size':>10s}: {h} x {w} px",
        f"  {'Resolution':>10s}: {resolution:.4f} m/px",
        f"  {'Coverage':>10s}: {w * resolution:.0f} x {h * resolution:.0f} m",
        f"  {'Threshold':>10s}: >= {confidence_threshold}",
    ])
    if geo.crs:
        stats_lines.append(f"  {'CRS':>10s}: {geo.crs}")

    stats_text = "\n".join(stats_lines)

    # --- Export files ---
    tmpdir = tempfile.mkdtemp(prefix="mayascan_", dir=TMPDIR)
    stem = Path(file).stem
    csv_path = to_csv(result, Path(tmpdir) / f"{stem}_detections.csv", pixel_size=resolution)
    geojson_path = to_geojson(result, Path(tmpdir) / f"{stem}_detections.geojson", pixel_size=resolution)
    geotiff_path = to_geotiff(result, Path(tmpdir) / f"{stem}_detections.tif", pixel_size=resolution)
    conf_path = to_confidence_geotiff(result, Path(tmpdir) / f"{stem}_confidence.tif", pixel_size=resolution)
    export_files = [str(csv_path), str(geojson_path), str(geotiff_path), str(conf_path)]

    return viz_rgb, overlay, blended, export_files, stats_text


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct the Gradio application."""

    css = """
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .subtitle { text-align: center; color: #666; margin-top: 0; }
    """

    with gr.Blocks(
        title="MayaScan — Archaeological LiDAR Feature Detection",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        gr.HTML(
            '<h1 class="main-title">MayaScan</h1>'
            '<p class="subtitle">Open-source archaeological feature detection from LiDAR DEMs</p>'
        )

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload DEM or Visualization",
                    file_types=[".tif", ".tiff", ".npy"],
                    type="filepath",
                )
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Higher = fewer but more confident detections.",
                )
                resolution_input = gr.Number(
                    value=0.5,
                    label="Resolution (m/px)",
                    info="Auto-detected from GeoTIFFs. Set manually for .npy.",
                    precision=4,
                )
                opacity_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Overlay Opacity",
                )
                detect_btn = gr.Button(
                    "Detect Structures",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    "### Legend\n"
                    "- **Red**: Buildings / mounds\n"
                    "- **Green**: Platforms\n"
                    "- **Blue**: Aguadas (reservoirs)\n"
                )

            # --- Right column: outputs ---
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Overlay"):
                        blended_image = gr.Image(
                            label="Detections overlaid on visualization",
                            type="numpy",
                            interactive=False,
                        )
                    with gr.TabItem("Visualization"):
                        viz_image = gr.Image(
                            label="SVF / Openness / Slope",
                            type="numpy",
                            interactive=False,
                        )
                    with gr.TabItem("Detection Mask"):
                        det_image = gr.Image(
                            label="Raw detection classes",
                            type="numpy",
                            interactive=False,
                        )

                stats_box = gr.Textbox(
                    label="Statistics",
                    lines=12,
                    interactive=False,
                )
                download_files = gr.File(
                    label="Download Results (CSV, GeoJSON, GeoTIFF, Confidence)",
                    file_count="multiple",
                    interactive=False,
                )

        # --- Wire up the button ---
        detect_btn.click(
            fn=process_upload,
            inputs=[file_input, confidence_slider, resolution_input, opacity_slider],
            outputs=[viz_image, det_image, blended_image, download_files, stats_box],
        )

        gr.Markdown(
            "---\n"
            f"**MayaScan** v{mayascan.__version__} | "
            "[GitHub](https://github.com/fascinatedbyeverything/mayascan) | "
            "[Models](https://huggingface.co/fascinatedbyeverything/mayascan) | "
            "Built with [Gradio](https://gradio.app)\n\n"
            "Per-class binary segmentation (DeepLabV3+ ResNet-101) with "
            "test-time augmentation. Detects ancient Maya buildings, "
            "platforms, and aguadas from LiDAR-derived terrain visualizations."
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
