"""Morphological feature analysis for archaeological structure characterization.

Computes shape descriptors, spatial relationships, and settlement pattern
metrics to help archaeologists classify and interpret detected features.

Shape metrics follow standards from landscape ecology and GIS-based
archaeological analysis (e.g., compactness, elongation, rectangularity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation

from mayascan.features import Feature


@dataclass
class ShapeDescriptors:
    """Shape metrics for a single archaeological feature.

    Attributes
    ----------
    compactness : float
        Polsby-Popper compactness (4*pi*area / perimeter^2).
        1.0 = perfect circle, lower = more irregular.
    elongation : float
        Ratio of major to minor axis (from PCA of pixel coords).
        1.0 = circular, higher = more elongated.
    rectangularity : float
        Area / oriented bounding box area.
        1.0 = perfect rectangle, lower = less rectangular.
    solidity : float
        Area / convex hull area.
        1.0 = convex shape, lower = concave or irregular.
    perimeter_px : float
        Perimeter in pixels (boundary pixel count).
    orientation_deg : float
        Major axis orientation in degrees (0-180).
    equivalent_diameter_px : float
        Diameter of circle with same area.
    """

    compactness: float
    elongation: float
    rectangularity: float
    solidity: float
    perimeter_px: float
    orientation_deg: float
    equivalent_diameter_px: float


@dataclass
class FeatureProfile:
    """Extended feature analysis with shape and spatial context.

    Attributes
    ----------
    feature : Feature
        The original detected feature.
    shape : ShapeDescriptors
        Computed shape metrics.
    structure_type : str
        Inferred structure type based on shape and class.
    nearest_neighbor_px : float
        Distance to nearest feature of any class (pixels).
    nearest_same_class_px : float
        Distance to nearest feature of same class (pixels).
    """

    feature: Feature
    shape: ShapeDescriptors
    structure_type: str = "unknown"
    nearest_neighbor_px: float = float("inf")
    nearest_same_class_px: float = float("inf")


def compute_perimeter(mask: np.ndarray) -> float:
    """Compute feature perimeter as count of boundary pixels.

    A boundary pixel is one that is True in the mask but adjacent
    (4-connected) to at least one False pixel.
    """
    eroded = binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    boundary = mask & ~eroded
    return float(boundary.sum())


def compute_shape_descriptors(mask: np.ndarray) -> ShapeDescriptors:
    """Compute shape metrics for a binary feature mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) for a single feature.

    Returns
    -------
    ShapeDescriptors
        Computed shape metrics.
    """
    area = float(mask.sum())
    if area < 2:
        return ShapeDescriptors(
            compactness=0.0,
            elongation=1.0,
            rectangularity=0.0,
            solidity=0.0,
            perimeter_px=0.0,
            orientation_deg=0.0,
            equivalent_diameter_px=0.0,
        )

    # Perimeter
    perimeter = compute_perimeter(mask)

    # Compactness (Polsby-Popper)
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
    else:
        compactness = 0.0

    # Get pixel coordinates
    rows, cols = np.where(mask)
    coords = np.column_stack([cols, rows]).astype(np.float64)

    # PCA for orientation and elongation
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Elongation
    if eigenvalues[1] > 0:
        elongation = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))
    else:
        elongation = float("inf")

    # Orientation (angle of major axis)
    major_axis = eigenvectors[:, 0]
    orientation = float(np.degrees(np.arctan2(major_axis[1], major_axis[0]))) % 180

    # Rectangularity (area / oriented bounding box area)
    # Project onto principal axes
    projected = centered @ eigenvectors
    obb_ranges = projected.max(axis=0) - projected.min(axis=0)
    obb_area = float(obb_ranges[0] * obb_ranges[1])
    rectangularity = area / obb_area if obb_area > 0 else 0.0

    # Solidity (area / convex hull area)
    # Use a simple raster-based convex hull approximation
    solidity = _compute_solidity(mask, area)

    # Equivalent diameter
    equivalent_diameter = float(2 * np.sqrt(area / np.pi))

    return ShapeDescriptors(
        compactness=round(min(compactness, 1.0), 4),
        elongation=round(elongation, 4),
        rectangularity=round(min(rectangularity, 1.0), 4),
        solidity=round(min(solidity, 1.0), 4),
        perimeter_px=round(perimeter, 1),
        orientation_deg=round(orientation, 1),
        equivalent_diameter_px=round(equivalent_diameter, 1),
    )


def _compute_solidity(mask: np.ndarray, area: float) -> float:
    """Approximate solidity as area / convex hull area.

    Uses a scan-line approach for the convex hull of the pixel set.
    """
    rows, cols = np.where(mask)
    if len(rows) < 3:
        return 1.0

    # Simple approximation: for each row, fill between min and max column
    hull_area = 0
    unique_rows = np.unique(rows)
    for r in unique_rows:
        row_cols = cols[rows == r]
        hull_area += row_cols.max() - row_cols.min() + 1

    return area / hull_area if hull_area > 0 else 0.0


def classify_structure(
    feature: Feature,
    shape: ShapeDescriptors,
    pixel_size: float = 0.5,
) -> str:
    """Infer archaeological structure sub-type from shape and class.

    Parameters
    ----------
    feature : Feature
        The detected feature.
    shape : ShapeDescriptors
        Computed shape metrics.
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    str
        Inferred structure type (e.g., "temple_pyramid", "residential_mound",
        "plaza_platform", "reservoir").
    """
    area = feature.area_m2
    cls = feature.class_name.lower()

    if cls == "building":
        # Maya building classification by area and shape
        if area > 2000 and shape.compactness > 0.5:
            return "temple_pyramid"
        elif area > 500:
            return "elite_residence"
        elif shape.rectangularity > 0.6:
            return "residential_mound"
        else:
            return "small_mound"

    elif cls == "platform":
        if area > 5000:
            return "grand_plaza"
        elif area > 1000 and shape.rectangularity > 0.6:
            return "plaza_platform"
        elif shape.elongation > 3.0:
            return "causeway_segment"
        else:
            return "raised_platform"

    elif cls == "aguada":
        if area > 2000:
            return "large_reservoir"
        elif shape.compactness > 0.6:
            return "circular_reservoir"
        else:
            return "irregular_depression"

    return "unknown"


def analyze_features(
    features: list[Feature],
    pixel_size: float = 0.5,
) -> list[FeatureProfile]:
    """Compute shape descriptors and spatial context for all features.

    Parameters
    ----------
    features : list[Feature]
        Extracted features.
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    list[FeatureProfile]
        Feature profiles with shape and spatial analysis.
    """
    profiles: list[FeatureProfile] = []

    # Compute centroids for nearest-neighbor analysis
    centroids = np.array(
        [[f.centroid_col, f.centroid_row] for f in features],
        dtype=np.float64,
    ) if features else np.empty((0, 2))

    for i, feat in enumerate(features):
        shape = compute_shape_descriptors(feat.mask)
        structure_type = classify_structure(feat, shape, pixel_size)

        # Nearest neighbor distances
        nn_dist = float("inf")
        nn_same = float("inf")
        if len(centroids) > 1:
            dists = np.sqrt(((centroids - centroids[i]) ** 2).sum(axis=1))
            dists[i] = float("inf")  # exclude self
            nn_dist = float(dists.min())

            # Same class
            same_class_mask = np.array(
                [j != i and features[j].class_id == feat.class_id
                 for j in range(len(features))]
            )
            if same_class_mask.any():
                same_dists = dists.copy()
                same_dists[~same_class_mask] = float("inf")
                nn_same = float(same_dists.min())

        profiles.append(FeatureProfile(
            feature=feat,
            shape=shape,
            structure_type=structure_type,
            nearest_neighbor_px=round(nn_dist, 1),
            nearest_same_class_px=round(nn_same, 1),
        ))

    return profiles


def settlement_summary(profiles: list[FeatureProfile], pixel_size: float = 0.5) -> dict[str, Any]:
    """Summarize settlement pattern metrics.

    Parameters
    ----------
    profiles : list[FeatureProfile]
        Analyzed feature profiles.
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    dict
        Settlement pattern summary with structure type counts,
        spatial statistics, and density metrics.
    """
    if not profiles:
        return {
            "total_features": 0,
            "structure_types": {},
            "spatial": {},
        }

    # Count structure types
    type_counts: dict[str, int] = {}
    for p in profiles:
        type_counts[p.structure_type] = type_counts.get(p.structure_type, 0) + 1

    # Spatial statistics
    nn_dists = [p.nearest_neighbor_px * pixel_size for p in profiles
                if p.nearest_neighbor_px < float("inf")]

    spatial: dict[str, Any] = {}
    if nn_dists:
        spatial["mean_nn_distance_m"] = round(float(np.mean(nn_dists)), 1)
        spatial["median_nn_distance_m"] = round(float(np.median(nn_dists)), 1)
        spatial["min_nn_distance_m"] = round(float(np.min(nn_dists)), 1)
        spatial["max_nn_distance_m"] = round(float(np.max(nn_dists)), 1)

    # Shape distribution
    compactness_vals = [p.shape.compactness for p in profiles]
    elongation_vals = [p.shape.elongation for p in profiles]

    return {
        "total_features": len(profiles),
        "structure_types": type_counts,
        "spatial": spatial,
        "shape_stats": {
            "mean_compactness": round(float(np.mean(compactness_vals)), 4),
            "mean_elongation": round(float(np.mean(elongation_vals)), 4),
        },
    }
