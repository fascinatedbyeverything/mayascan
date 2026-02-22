"""Report generation for MayaScan detection results.

Produces a human-readable summary report (text or HTML) of detected
archaeological features, suitable for field documentation and publication.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from mayascan.detect import DetectionResult


def generate_report(
    result: DetectionResult,
    input_path: str | None = None,
    pixel_size: float = 0.5,
) -> dict[str, Any]:
    """Generate a structured report of detection results.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    input_path : str or None
        Path to the original input file (for display purposes).
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    dict
        Structured report with summary statistics per class.
    """
    geo = result.geo
    h, w = result.classes.shape

    if geo and geo.resolution:
        pixel_size = geo.resolution

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "software": "MayaScan",
        "input": str(input_path) if input_path else None,
        "dimensions": {"height": h, "width": w},
        "resolution_m": pixel_size,
        "coverage_m2": h * w * pixel_size * pixel_size,
        "crs": geo.crs if geo else None,
        "classes": {},
        "total_features": 0,
        "total_feature_area_m2": 0.0,
    }

    for class_id, class_name in result.class_names.items():
        if class_id == 0:
            continue

        mask = result.classes == class_id
        if not mask.any():
            report["classes"][class_name] = {
                "count": 0,
                "total_area_m2": 0.0,
                "mean_confidence": 0.0,
                "features": [],
            }
            continue

        labeled_arr, num_features = label(mask)
        total_pixels = int(mask.sum())
        total_area = total_pixels * pixel_size * pixel_size
        mean_conf = float(result.confidence[mask].mean())

        features = []
        for feat_idx in range(1, num_features + 1):
            feat_mask = labeled_arr == feat_idx
            px_count = int(feat_mask.sum())
            area = px_count * pixel_size * pixel_size
            conf = float(result.confidence[feat_mask].mean())

            rows, cols = np.where(feat_mask)
            cy, cx = float(rows.mean()), float(cols.mean())

            feat_info: dict[str, Any] = {
                "id": feat_idx,
                "pixel_count": px_count,
                "area_m2": round(area, 2),
                "confidence": round(conf, 4),
                "centroid_px": [round(cx, 1), round(cy, 1)],
                "bbox_px": [int(rows.min()), int(cols.min()),
                            int(rows.max()), int(cols.max())],
            }

            # Add geo coordinates if available
            if geo and geo.transform:
                a, b, c, d, e, f = geo.transform
                gx = c + cx * a + cy * b
                gy = f + cx * d + cy * e
                feat_info["centroid_geo"] = [round(gx, 6), round(gy, 6)]

            features.append(feat_info)

        # Sort by area descending
        features.sort(key=lambda f: f["area_m2"], reverse=True)

        report["classes"][class_name] = {
            "count": num_features,
            "total_area_m2": round(total_area, 2),
            "mean_confidence": round(mean_conf, 4),
            "coverage_pct": round(100 * total_pixels / (h * w), 4),
            "features": features,
        }
        report["total_features"] += num_features
        report["total_feature_area_m2"] += total_area

    report["total_feature_area_m2"] = round(report["total_feature_area_m2"], 2)
    report["feature_density_per_km2"] = round(
        report["total_features"] / (report["coverage_m2"] / 1e6), 2
    ) if report["coverage_m2"] > 0 else 0

    return report


def report_to_text(report: dict[str, Any]) -> str:
    """Format a structured report as human-readable text."""
    lines = [
        "=" * 60,
        "MAYASCAN DETECTION REPORT",
        "=" * 60,
        "",
        f"Date:       {report['timestamp'][:19]}",
    ]

    if report["input"]:
        lines.append(f"Input:      {report['input']}")

    lines.extend([
        f"Dimensions: {report['dimensions']['height']} x {report['dimensions']['width']} px",
        f"Resolution: {report['resolution_m']} m/px",
        f"Coverage:   {report['coverage_m2']:,.0f} m\u00b2 ({report['coverage_m2'] / 1e6:.3f} km\u00b2)",
    ])

    if report["crs"]:
        lines.append(f"CRS:        {report['crs']}")

    lines.extend([
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
        f"Total features:    {report['total_features']}",
        f"Total feature area:{report['total_feature_area_m2']:>10,.0f} m\u00b2",
        f"Feature density:   {report['feature_density_per_km2']:,.1f} / km\u00b2",
        "",
    ])

    for class_name, cls_data in report["classes"].items():
        lines.extend([
            "-" * 60,
            f"{class_name.upper()} ({cls_data['count']} features)",
            "-" * 60,
        ])

        if cls_data["count"] == 0:
            lines.append("  No features detected.")
            lines.append("")
            continue

        lines.extend([
            f"  Total area:    {cls_data['total_area_m2']:>10,.0f} m\u00b2",
            f"  Coverage:      {cls_data['coverage_pct']:.2f}%",
            f"  Mean confidence: {cls_data['mean_confidence']:.2f}",
            "",
        ])

        # Top features
        top_n = min(10, len(cls_data["features"]))
        if top_n > 0:
            lines.append(f"  Top {top_n} features by area:")
            for feat in cls_data["features"][:top_n]:
                geo_str = ""
                if "centroid_geo" in feat:
                    geo_str = f" @ ({feat['centroid_geo'][0]:.2f}, {feat['centroid_geo'][1]:.2f})"
                lines.append(
                    f"    #{feat['id']:>3d}: {feat['area_m2']:>8,.1f} m\u00b2  "
                    f"conf={feat['confidence']:.2f}{geo_str}"
                )
            lines.append("")

    lines.extend([
        "=" * 60,
        f"Generated by MayaScan",
        "=" * 60,
    ])

    return "\n".join(lines)


def save_report(
    result: DetectionResult,
    output_path: str | Path,
    input_path: str | None = None,
    pixel_size: float = 0.5,
    format: str = "text",
) -> Path:
    """Save a detection report to a file.

    Parameters
    ----------
    result : DetectionResult
        Detection output.
    output_path : str or Path
        Output file path.
    input_path : str or None
        Path to the original input file.
    pixel_size : float
        Ground resolution in metres per pixel.
    format : str
        Output format: ``"text"`` or ``"json"``.

    Returns
    -------
    Path
        The written report file path.
    """
    output_path = Path(output_path)
    report = generate_report(result, input_path=input_path, pixel_size=pixel_size)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    else:
        text = report_to_text(report)
        with open(output_path, "w") as f:
            f.write(text)

    return output_path
