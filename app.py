"""MayaScan Gradio web interface for archaeological LiDAR feature detection.

Upload a DEM (GeoTIFF or .npy), visualize SVF / openness / slope, run
U-Net segmentation, and download results as CSV, GeoJSON, or GeoTIFF.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from scipy.ndimage import label

import mayascan
from mayascan.detect import CLASS_NAMES, discover_v2_models, run_detection_v2
from mayascan.export import to_csv, to_geojson, to_geotiff

# Default model directory for v2 per-class models
V2_MODEL_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Class colours (RGBA)
# ---------------------------------------------------------------------------

COLORS: dict[int, list[int]] = {
    0: [0, 0, 0, 0],           # background — transparent
    1: [255, 60, 60, 180],     # building — red
    2: [255, 220, 50, 180],    # platform — yellow
    3: [50, 120, 255, 180],    # aguada — blue
}


def colorize_classes(classes: np.ndarray) -> np.ndarray:
    """Convert a (H, W) class-index map to an RGBA image.

    Parameters
    ----------
    classes : np.ndarray
        Integer array of shape (H, W) with values in ``COLORS.keys()``.

    Returns
    -------
    np.ndarray
        uint8 RGBA image of shape (H, W, 4).
    """
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id, color in COLORS.items():
        mask = classes == class_id
        rgba[mask] = color
    return rgba


# ---------------------------------------------------------------------------
# DEM loading
# ---------------------------------------------------------------------------

def load_dem_from_file(file_path: str) -> np.ndarray:
    """Load a DEM from a file path.

    Supports GeoTIFF (.tif / .tiff) and NumPy (.npy) formats.

    Parameters
    ----------
    file_path : str
        Path to the DEM file.

    Returns
    -------
    np.ndarray
        2-D float32 elevation array.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore[import-untyped]

            with rasterio.open(str(path)) as src:
                dem = src.read(1).astype(np.float32)
        except ImportError:
            from PIL import Image  # noqa: E402

            img = Image.open(str(path))
            dem = np.array(img, dtype=np.float32)
        return dem

    if suffix == ".npy":
        dem = np.load(str(path)).astype(np.float32)
        if dem.ndim != 2:
            raise ValueError(f"Expected 2-D array in .npy file, got shape {dem.shape}")
        return dem

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Please upload a .tif, .tiff, or .npy file."
    )


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_upload(
    file: str,
    confidence_threshold: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Process an uploaded DEM file through the full MayaScan pipeline.

    Parameters
    ----------
    file : str
        Path to the uploaded file (Gradio provides this as a string).
    confidence_threshold : float
        Minimum confidence for non-background predictions.
    resolution : float
        Pixel size in metres.

    Returns
    -------
    tuple
        (viz_rgb, overlay, export_files, stats)
        - viz_rgb: uint8 RGB image (H, W, 3) of the visualization stack
        - overlay: uint8 RGBA image (H, W, 4) of colorized detection classes
        - export_files: list of file paths for download
        - stats: human-readable statistics string
    """
    # --- Load DEM ---
    dem = load_dem_from_file(file)

    # --- Visualize ---
    viz = mayascan.visualize(dem, resolution=resolution)  # (3, H, W) float32

    # Convert (3, H, W) float32 [0,1] to (H, W, 3) uint8 for display
    viz_rgb = (np.transpose(viz, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

    # --- Detect (v2 per-class models if available, else v1 multi-class) ---
    v2_models = discover_v2_models(str(V2_MODEL_DIR))
    if v2_models:
        result = run_detection_v2(
            viz,
            model_dir=str(V2_MODEL_DIR),
            confidence_threshold=confidence_threshold,
        )
    else:
        result = mayascan.detect(viz, confidence_threshold=confidence_threshold)

    # --- Colorize detection ---
    overlay = colorize_classes(result.classes)

    # --- Feature statistics ---
    stats_lines = ["MayaScan Detection Results", "=" * 30, ""]

    total_features = 0
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        mask = result.classes == class_id
        if not mask.any():
            stats_lines.append(f"{class_name.capitalize():>10s}: 0 features")
            continue
        labeled_array, num_features = label(mask)
        pixel_count = int(mask.sum())
        area_m2 = pixel_count * resolution * resolution
        stats_lines.append(
            f"{class_name.capitalize():>10s}: {num_features} features "
            f"({pixel_count} px, {area_m2:.1f} m\u00b2)"
        )
        total_features += num_features

    stats_lines.append("")
    stats_lines.append(f"{'Total':>10s}: {total_features} features detected")
    stats_lines.append(f"{'DEM size':>10s}: {dem.shape[0]} x {dem.shape[1]} px")
    stats_lines.append(f"{'Resolution':>10s}: {resolution} m/px")
    stats_lines.append(f"{'Confidence':>10s}: >= {confidence_threshold}")
    stats_text = "\n".join(stats_lines)

    # --- Export files ---
    tmpdir = tempfile.mkdtemp(prefix="mayascan_")
    csv_path = to_csv(result, Path(tmpdir) / "detections.csv", pixel_size=resolution)
    geojson_path = to_geojson(
        result, Path(tmpdir) / "detections.geojson", pixel_size=resolution
    )
    geotiff_path = to_geotiff(
        result, Path(tmpdir) / "detections.tif", pixel_size=resolution
    )
    export_files = [str(csv_path), str(geojson_path), str(geotiff_path)]

    return viz_rgb, overlay, export_files, stats_text


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        title="MayaScan",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# MayaScan\n"
            "**Open-source archaeological feature detection from LiDAR DEMs.**\n\n"
            "Upload a digital elevation model to visualize terrain features "
            "(sky-view factor, openness, slope) and detect ancient Maya "
            "structures using deep learning segmentation."
        )

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload DEM",
                    file_types=[".tif", ".tiff", ".npy"],
                    type="filepath",
                )
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence for non-background detections.",
                )
                resolution_input = gr.Number(
                    value=0.5,
                    label="Resolution (m/px)",
                    info="Ground sampling distance of the DEM in metres.",
                    precision=2,
                )
                detect_btn = gr.Button(
                    "Detect Structures",
                    variant="primary",
                )

            # --- Right column: outputs ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Visualization"):
                        viz_image = gr.Image(
                            label="SVF / Openness / Slope",
                            type="numpy",
                            interactive=False,
                        )
                    with gr.TabItem("Detection Results"):
                        det_image = gr.Image(
                            label="Detected Features",
                            type="numpy",
                            interactive=False,
                        )
                stats_box = gr.Textbox(
                    label="Statistics",
                    lines=10,
                    interactive=False,
                )
                download_files = gr.File(
                    label="Download Results",
                    file_count="multiple",
                    interactive=False,
                )

        # --- Wire up the button ---
        detect_btn.click(
            fn=process_upload,
            inputs=[file_input, confidence_slider, resolution_input],
            outputs=[viz_image, det_image, download_files, stats_box],
        )

        gr.Markdown(
            "---\n"
            "**MayaScan** v0.2.0 | "
            "[GitHub](https://github.com/fascinatedbyeverything/mayascan) | "
            "Built with [Gradio](https://gradio.app)\n\n"
            "Archaeological LiDAR feature detection for Maya archaeology. "
            "Per-class binary segmentation (DeepLabV3+ ResNet-101) with "
            "test-time augmentation for buildings, platforms, and aguadas."
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
