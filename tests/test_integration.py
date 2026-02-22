"""End-to-end integration tests for the full MayaScan pipeline.

Tests the complete workflow: synthetic DEM -> visualize -> detect -> export.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import mayascan
from mayascan.detect import DetectionResult
from mayascan.export import to_csv, to_geojson, to_geotiff, to_kml
from mayascan.report import generate_report, report_to_text, report_to_html, save_report
from mayascan.features import extract_features, filter_features, feature_summary
from mayascan.metrics import compute_binary_metrics, compute_multiclass_metrics, mean_iou


def _make_mound_dem(size: int = 480, num_mounds: int = 5, seed: int = 42) -> np.ndarray:
    """Create a synthetic DEM with circular mound features.

    Uses np.ogrid to generate Gaussian-like mounds on a flat terrain,
    simulating archaeological mound structures.
    """
    rng = np.random.default_rng(seed)
    dem = np.zeros((size, size), dtype=np.float64)

    # Add gentle background terrain
    y, x = np.ogrid[0:size, 0:size]
    dem += 50.0 + 2.0 * np.sin(2 * np.pi * x / size) + 1.5 * np.cos(2 * np.pi * y / size)

    # Add circular mound features
    for _ in range(num_mounds):
        cx = rng.integers(60, size - 60)
        cy = rng.integers(60, size - 60)
        radius = rng.integers(15, 40)
        height = rng.uniform(3.0, 10.0)

        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mound = height * np.exp(-dist_sq / (2 * radius ** 2))
        dem += mound

    return dem


class TestFullPipelineSynthetic:
    """End-to-end test: synthetic DEM -> visualize -> detect -> export."""

    def test_full_pipeline_synthetic(self, tmp_path: Path):
        """Run the complete pipeline on a synthetic DEM with mound features."""
        # --- Step 1: Create synthetic DEM ---
        dem = _make_mound_dem(size=480, num_mounds=5, seed=42)
        assert dem.shape == (480, 480)

        # --- Step 2: Visualize ---
        viz = mayascan.visualize(dem, resolution=0.5)
        assert viz.shape == (3, 480, 480)
        assert viz.dtype == np.float32
        # Each channel should be in [0, 1]
        for ch in range(3):
            assert viz[ch].min() >= 0.0
            assert viz[ch].max() <= 1.0 + 1e-6

        # --- Step 3: Detect ---
        result = mayascan.detect(viz, confidence_threshold=0.0)
        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert result.classes.dtype in (np.int64, np.int32, np.intp)
        assert result.confidence.dtype == np.float32 or np.issubdtype(
            result.confidence.dtype, np.floating
        )
        assert result.classes.min() >= 0
        assert result.classes.max() < len(result.class_names)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0 + 1e-6

        # --- Step 4: Export ---
        csv_path = tmp_path / "features.csv"
        geojson_path = tmp_path / "features.geojson"
        geotiff_path = tmp_path / "classmap.tif"

        to_csv(result, csv_path)
        to_geojson(result, geojson_path)
        to_geotiff(result, geotiff_path)

        # Verify all files exist and are non-empty
        assert csv_path.exists(), "CSV file was not created"
        assert csv_path.stat().st_size > 0, "CSV file is empty"

        assert geojson_path.exists(), "GeoJSON file was not created"
        assert geojson_path.stat().st_size > 0, "GeoJSON file is empty"

        assert geotiff_path.exists(), "GeoTIFF file was not created"
        assert geotiff_path.stat().st_size > 0, "GeoTIFF file is empty"

        # Verify GeoJSON is valid JSON with correct structure
        with open(geojson_path) as f:
            geojson_data = json.load(f)

        assert geojson_data["type"] == "FeatureCollection"
        assert "features" in geojson_data
        assert isinstance(geojson_data["features"], list)


class TestFullPipelineWithAnalysis:
    """End-to-end: DEM -> detect -> features + report + metrics + all exports."""

    def test_pipeline_features_report_export(self, tmp_path: Path):
        """Full pipeline exercises features, report, and all export formats."""
        dem = _make_mound_dem(size=480, num_mounds=5, seed=42)
        viz = mayascan.visualize(dem, resolution=0.5)
        result = mayascan.detect(viz, confidence_threshold=0.0)

        # --- Feature extraction ---
        features = extract_features(result, pixel_size=0.5)
        assert isinstance(features, list)
        # Features should be sorted by area descending
        if len(features) > 1:
            areas = [f.pixel_count for f in features]
            assert areas == sorted(areas, reverse=True)

        # Every feature has valid fields
        for feat in features:
            assert feat.class_id > 0  # no background features
            assert feat.class_name in result.class_names.values()
            assert feat.pixel_count > 0
            assert feat.area_m2 > 0
            assert 0.0 <= feat.confidence <= 1.0
            assert feat.mask.shape == result.classes.shape
            assert feat.mask.sum() == feat.pixel_count

        # --- Feature filtering ---
        if features:
            min_area = features[-1].area_m2 + 0.01  # filter out smallest
            filtered = filter_features(features, min_area=min_area)
            assert len(filtered) < len(features) or len(features) == 0

        # --- Feature summary ---
        summary = feature_summary(features)
        assert summary["total_count"] == len(features)
        assert summary["total_area_m2"] >= 0
        if features:
            assert 0.0 <= summary["mean_confidence"] <= 1.0

        # --- Report generation ---
        report = generate_report(result, pixel_size=0.5)
        assert report["software"] == "MayaScan"
        assert report["dimensions"] == {"height": 480, "width": 480}
        assert report["total_features"] >= 0

        text = report_to_text(report)
        assert "MAYASCAN DETECTION REPORT" in text

        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html

        # --- Save all report formats ---
        save_report(result, tmp_path / "report.txt", format="text")
        save_report(result, tmp_path / "report.json", format="json")
        save_report(result, tmp_path / "report.html", format="html")
        assert (tmp_path / "report.txt").stat().st_size > 0
        assert (tmp_path / "report.json").stat().st_size > 0
        assert (tmp_path / "report.html").stat().st_size > 0

        # JSON report is valid and round-trips
        with open(tmp_path / "report.json") as f:
            report_rt = json.load(f)
        assert report_rt["software"] == "MayaScan"

        # --- All export formats ---
        to_csv(result, tmp_path / "features.csv")
        to_geojson(result, tmp_path / "features.geojson")
        to_geotiff(result, tmp_path / "classmap.tif")
        to_kml(result, tmp_path / "features.kml")

        for name in ["features.csv", "features.geojson", "classmap.tif", "features.kml"]:
            path = tmp_path / name
            assert path.exists(), f"{name} was not created"
            assert path.stat().st_size > 0, f"{name} is empty"


class TestMetricsIntegration:
    """Test metrics computation on synthetic detection results."""

    def test_self_comparison_perfect_score(self):
        """Comparing a result to itself should give IoU=1.0."""
        classes = np.zeros((100, 100), dtype=np.int64)
        classes[20:40, 20:40] = 1
        classes[60:80, 60:80] = 2

        class_names = {0: "background", 1: "building", 2: "platform"}
        metrics = compute_multiclass_metrics(classes, classes, class_names)

        for cls_id, m in metrics.items():
            assert abs(m.iou - 1.0) < 1e-6, f"class {cls_id} IoU should be 1.0"
            assert abs(m.f1 - 1.0) < 1e-6

        assert abs(mean_iou(metrics) - 1.0) < 1e-6

    def test_binary_metrics_partial_overlap(self):
        """Binary metrics for partially overlapping predictions."""
        pred = np.zeros((50, 50), dtype=bool)
        target = np.zeros((50, 50), dtype=bool)

        # pred covers rows 10-30, target covers rows 20-40
        pred[10:30, 10:30] = True
        target[20:40, 10:30] = True

        m = compute_binary_metrics(pred, target)
        # Overlap is 10x20 = 200 pixels
        # Union is 30x20 = 600 pixels
        expected_iou = 200.0 / 600.0
        assert abs(m.iou - expected_iou) < 1e-6
        assert 0.0 < m.precision < 1.0
        assert 0.0 < m.recall < 1.0
        assert 0.0 < m.f1 < 1.0

    def test_multiclass_with_misclassification(self):
        """Multiclass metrics handle misclassified pixels correctly."""
        pred = np.zeros((50, 50), dtype=np.int64)
        target = np.zeros((50, 50), dtype=np.int64)

        # Target: building at top-left, platform at bottom-right
        target[0:20, 0:20] = 1
        target[30:50, 30:50] = 2

        # Pred: building correct, but platform region predicted as building
        pred[0:20, 0:20] = 1
        pred[30:50, 30:50] = 1  # misclassified as building

        class_names = {0: "background", 1: "building", 2: "platform"}
        metrics = compute_multiclass_metrics(pred, target, class_names)

        # Building should have recall=1.0 but precision < 1.0 (extra FPs from misclassified platform)
        assert metrics[1].recall == 1.0
        assert metrics[1].precision < 1.0

        # Platform should have recall=0.0 (nothing predicted as platform)
        assert metrics[2].recall == 0.0
        assert metrics[2].iou == 0.0


class TestProcessDemConvenience:
    """Test the convenience function mayascan.process_dem."""

    def test_process_dem_convenience(self):
        """process_dem runs the full pipeline in one call and returns correct shapes."""
        rng = np.random.default_rng(99)
        dem = rng.random((480, 480)).astype(np.float64) * 100.0

        result = mayascan.process_dem(dem, confidence_threshold=0.0)

        assert isinstance(result, DetectionResult)
        assert result.classes.shape == (480, 480)
        assert result.confidence.shape == (480, 480)
        assert isinstance(result.class_names, dict)
        assert 0 in result.class_names
        assert result.class_names[0] == "background"
        assert result.classes.min() >= 0
        assert result.classes.max() < len(result.class_names)
        assert result.confidence.min() >= 0.0
        assert result.confidence.max() <= 1.0 + 1e-6
