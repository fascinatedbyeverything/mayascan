"""Spatial clustering and settlement hierarchy analysis.

Identifies archaeological site cores, clusters, and settlement patterns
using density-based clustering (DBSCAN) on detected feature centroids.
This helps archaeologists delineate site boundaries and understand
settlement organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mayascan.features import Feature


@dataclass
class Cluster:
    """A spatial cluster of archaeological features.

    Attributes
    ----------
    cluster_id : int
        Cluster index (0-based). -1 = noise/outlier.
    features : list[Feature]
        Features belonging to this cluster.
    centroid_px : tuple[float, float]
        Cluster centroid (col, row) in pixel coordinates.
    extent_px : float
        Maximum distance from centroid to any member feature.
    density : float
        Features per unit area (features per 10000 px²).
    """

    cluster_id: int
    features: list[Feature] = field(default_factory=list)
    centroid_px: tuple[float, float] = (0.0, 0.0)
    extent_px: float = 0.0
    density: float = 0.0


def _euclidean_distances(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def dbscan(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Minimal DBSCAN implementation (no sklearn dependency).

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) array of coordinates.
    eps : float
        Maximum distance between two samples in same neighborhood.
    min_samples : int
        Minimum number of samples in a neighborhood to form a core point.

    Returns
    -------
    np.ndarray
        Cluster labels for each point. -1 means noise.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int64)

    dist_matrix = _euclidean_distances(points)
    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = np.where(dist_matrix[i] <= eps)[0]

        if len(neighbors) < min_samples:
            continue  # noise point (may be claimed later)

        # Start a new cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist_matrix[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


def cluster_features(
    features: list[Feature],
    eps_px: float = 100.0,
    min_features: int = 3,
) -> list[Cluster]:
    """Cluster features by spatial proximity using DBSCAN.

    Parameters
    ----------
    features : list[Feature]
        Detected features with centroid coordinates.
    eps_px : float
        Maximum distance (pixels) between features in same cluster.
    min_features : int
        Minimum features to form a cluster.

    Returns
    -------
    list[Cluster]
        Identified clusters, sorted by feature count (largest first).
        Noise features are grouped in a cluster with id=-1.
    """
    if not features:
        return []

    points = np.array(
        [[f.centroid_col, f.centroid_row] for f in features],
        dtype=np.float64,
    )

    labels = dbscan(points, eps=eps_px, min_samples=min_features)

    # Group features by cluster label
    cluster_map: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        cluster_map.setdefault(int(lbl), []).append(i)

    clusters = []
    for cid, indices in sorted(cluster_map.items()):
        cluster_features_list = [features[i] for i in indices]
        cluster_points = points[indices]

        centroid = cluster_points.mean(axis=0)
        dists = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1))
        extent = float(dists.max()) if len(dists) > 0 else 0.0

        # Density: features per 10000 px² (in the bounding circle)
        area = np.pi * max(extent, 1.0) ** 2
        density = len(cluster_features_list) / (area / 10000)

        clusters.append(Cluster(
            cluster_id=cid,
            features=cluster_features_list,
            centroid_px=(float(centroid[0]), float(centroid[1])),
            extent_px=round(extent, 1),
            density=round(density, 4),
        ))

    # Sort: real clusters first (by size desc), then noise last
    real = sorted([c for c in clusters if c.cluster_id >= 0],
                  key=lambda c: len(c.features), reverse=True)
    noise = [c for c in clusters if c.cluster_id < 0]
    return real + noise


def identify_site_core(clusters: list[Cluster]) -> Cluster | None:
    """Identify the most likely site core (densest cluster).

    Parameters
    ----------
    clusters : list[Cluster]
        Output of :func:`cluster_features`.

    Returns
    -------
    Cluster or None
        The cluster with highest density, or None if no clusters.
    """
    real_clusters = [c for c in clusters if c.cluster_id >= 0]
    if not real_clusters:
        return None
    return max(real_clusters, key=lambda c: c.density)


def settlement_hierarchy(
    clusters: list[Cluster],
    pixel_size: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank clusters into a settlement hierarchy.

    Parameters
    ----------
    clusters : list[Cluster]
        Output of :func:`cluster_features`.
    pixel_size : float
        Ground resolution in metres per pixel.

    Returns
    -------
    list[dict]
        Ranked list of site components with metrics.
    """
    real_clusters = [c for c in clusters if c.cluster_id >= 0]
    if not real_clusters:
        return []

    hierarchy = []
    for rank, cluster in enumerate(sorted(real_clusters,
                                          key=lambda c: c.density,
                                          reverse=True)):
        total_area = sum(f.area_m2 for f in cluster.features)
        class_counts: dict[str, int] = {}
        for f in cluster.features:
            class_counts[f.class_name] = class_counts.get(f.class_name, 0) + 1

        extent_m = cluster.extent_px * pixel_size

        hierarchy.append({
            "rank": rank + 1,
            "cluster_id": cluster.cluster_id,
            "feature_count": len(cluster.features),
            "total_area_m2": round(total_area, 1),
            "extent_m": round(extent_m, 1),
            "density": cluster.density,
            "class_counts": class_counts,
            "centroid_px": cluster.centroid_px,
            "role": _infer_role(rank, cluster, class_counts),
        })

    return hierarchy


def _infer_role(rank: int, cluster: Cluster, class_counts: dict[str, int]) -> str:
    """Infer settlement role from rank and composition."""
    has_buildings = class_counts.get("building", 0) > 0
    has_platforms = class_counts.get("platform", 0) > 0
    has_aguadas = class_counts.get("aguada", 0) > 0
    n = len(cluster.features)

    if rank == 0 and has_buildings and has_platforms:
        return "site_core"
    elif has_platforms and has_buildings and n >= 5:
        return "secondary_center"
    elif has_buildings and n >= 3:
        return "residential_group"
    elif has_aguadas:
        return "water_management"
    else:
        return "peripheral"
