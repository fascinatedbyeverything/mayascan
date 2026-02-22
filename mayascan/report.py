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


def report_to_html(report: dict[str, Any]) -> str:
    """Format a structured report as a self-contained HTML document."""
    class_colors = {
        "building": "#ff3c3c",
        "platform": "#3cc83c",
        "aguada": "#3278ff",
    }

    rows = []
    for class_name, cls_data in report["classes"].items():
        color = class_colors.get(class_name, "#888")
        rows.append(
            f'<tr><td><span style="color:{color}; font-weight:bold">'
            f'{class_name.capitalize()}</span></td>'
            f'<td>{cls_data["count"]}</td>'
            f'<td>{cls_data["total_area_m2"]:,.0f}</td>'
            f'<td>{cls_data["mean_confidence"]:.2f}</td>'
            f'<td>{cls_data.get("coverage_pct", 0):.2f}%</td></tr>'
        )

    feature_rows = []
    for class_name, cls_data in report["classes"].items():
        color = class_colors.get(class_name, "#888")
        for feat in cls_data["features"][:10]:
            geo_str = ""
            if "centroid_geo" in feat:
                geo_str = f'{feat["centroid_geo"][0]:.4f}, {feat["centroid_geo"][1]:.4f}'
            feature_rows.append(
                f'<tr><td style="color:{color}">{class_name}</td>'
                f'<td>#{feat["id"]}</td>'
                f'<td>{feat["area_m2"]:,.1f}</td>'
                f'<td>{feat["confidence"]:.2f}</td>'
                f'<td>{feat["centroid_px"][0]:.0f}, {feat["centroid_px"][1]:.0f}</td>'
                f'<td>{geo_str}</td></tr>'
            )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>MayaScan Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 2em auto; background: #0a0a0a; color: #e8e0d4; }}
  h1 {{ text-align: center; color: #c4a882; }}
  h2 {{ color: #c4a882; border-bottom: 1px solid #333; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #c4a882; }}
  .meta {{ color: #888; font-size: 0.9em; }}
  .stat {{ font-size: 1.4em; font-weight: bold; color: #c4a882; }}
  .card {{ background: #1a1a1a; border-radius: 8px; padding: 1.2em; margin: 1em 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em; }}
  .grid .card {{ text-align: center; }}
</style>
</head><body>
<h1>MayaScan Detection Report</h1>
<p class="meta" style="text-align:center">{report['timestamp'][:19]}
{' | ' + report['input'] if report['input'] else ''}</p>

<div class="grid">
  <div class="card"><div class="stat">{report['total_features']}</div>Total Features</div>
  <div class="card"><div class="stat">{report['total_feature_area_m2']:,.0f} m&sup2;</div>Feature Area</div>
  <div class="card"><div class="stat">{report['feature_density_per_km2']:,.1f}/km&sup2;</div>Density</div>
</div>

<div class="card">
  <b>Dimensions:</b> {report['dimensions']['height']} x {report['dimensions']['width']} px |
  <b>Resolution:</b> {report['resolution_m']} m/px |
  <b>Coverage:</b> {report['coverage_m2']:,.0f} m&sup2; ({report['coverage_m2']/1e6:.3f} km&sup2;)
  {f'| <b>CRS:</b> {report["crs"]}' if report['crs'] else ''}
</div>

<h2>Class Summary</h2>
<table>
<tr><th>Class</th><th>Features</th><th>Area (m&sup2;)</th><th>Mean Conf.</th><th>Coverage</th></tr>
{''.join(rows)}
</table>

<h2>Top Features</h2>
<table>
<tr><th>Class</th><th>ID</th><th>Area (m&sup2;)</th><th>Conf.</th><th>Centroid (px)</th><th>Centroid (geo)</th></tr>
{''.join(feature_rows)}
</table>

<p class="meta" style="text-align:center; margin-top:3em">Generated by MayaScan v{report.get('software', 'MayaScan')}</p>
</body></html>"""

    return html


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
        Output format: ``"text"``, ``"json"``, or ``"html"``.

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
    elif format == "html":
        html = report_to_html(report)
        with open(output_path, "w") as f:
            f.write(html)
    else:
        text = report_to_text(report)
        with open(output_path, "w") as f:
            f.write(text)

    return output_path
