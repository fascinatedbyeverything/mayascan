"""Tests for mayascan.spatial — spatial clustering and settlement hierarchy."""

import numpy as np
import pytest

from mayascan.features import Feature
from mayascan.spatial import (
    Cluster,
    cluster_features,
    dbscan,
    identify_site_core,
    settlement_hierarchy,
)


def _make_feature(col, row, class_id=1, class_name="building", area_m2=100.0):
    """Create a Feature at given pixel coordinates."""
    mask = np.zeros((100, 100), dtype=bool)
    r, c = int(row), int(col)
    mask[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = True
    return Feature(
        feature_id=0,
        class_id=class_id,
        class_name=class_name,
        pixel_count=int(mask.sum()),
        area_m2=area_m2,
        confidence=0.9,
        centroid_row=float(row),
        centroid_col=float(col),
        centroid_geo=None,
        bbox=(max(0, r - 2), max(0, c - 2), min(99, r + 2), min(99, c + 2)),
        mask=mask,
    )


class TestDBSCAN:
    def test_two_clusters(self):
        # Two tight groups well separated
        points = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # cluster 0
            [20, 20], [21, 20], [20, 21], [21, 21],  # cluster 1
        ], dtype=np.float64)
        labels = dbscan(points, eps=3.0, min_samples=3)
        assert len(set(labels)) == 2
        assert labels[0] == labels[1] == labels[2] == labels[3]
        assert labels[4] == labels[5] == labels[6] == labels[7]
        assert labels[0] != labels[4]

    def test_noise_points(self):
        points = np.array([
            [0, 0], [1, 0], [0, 1],  # cluster
            [50, 50],  # noise
        ], dtype=np.float64)
        labels = dbscan(points, eps=3.0, min_samples=3)
        assert labels[3] == -1
        assert labels[0] >= 0

    def test_single_cluster(self):
        points = np.array([[i, i] for i in range(5)], dtype=np.float64)
        labels = dbscan(points, eps=3.0, min_samples=2)
        assert (labels >= 0).all()
        assert len(set(labels)) == 1

    def test_all_noise(self):
        points = np.array([[0, 0], [100, 100]], dtype=np.float64)
        labels = dbscan(points, eps=3.0, min_samples=3)
        assert (labels == -1).all()

    def test_empty_input(self):
        labels = dbscan(np.empty((0, 2)), eps=3.0, min_samples=3)
        assert len(labels) == 0


class TestClusterFeatures:
    def test_two_clusters(self):
        features = [
            _make_feature(10, 10),
            _make_feature(12, 10),
            _make_feature(10, 12),
            _make_feature(80, 80),
            _make_feature(82, 80),
            _make_feature(80, 82),
        ]
        clusters = cluster_features(features, eps_px=20, min_features=3)
        real = [c for c in clusters if c.cluster_id >= 0]
        assert len(real) == 2

    def test_returns_sorted_by_size(self):
        # First cluster: 5 features, second: 3 features
        features = [
            _make_feature(10, 10), _make_feature(12, 10),
            _make_feature(10, 12), _make_feature(12, 12),
            _make_feature(11, 11),  # 5-member cluster
            _make_feature(80, 80), _make_feature(82, 80),
            _make_feature(80, 82),  # 3-member cluster
        ]
        clusters = cluster_features(features, eps_px=20, min_features=3)
        real = [c for c in clusters if c.cluster_id >= 0]
        assert len(real) == 2
        assert len(real[0].features) >= len(real[1].features)

    def test_noise_cluster_last(self):
        features = [
            _make_feature(10, 10), _make_feature(12, 10),
            _make_feature(10, 12),  # cluster
            _make_feature(90, 90),  # noise
        ]
        clusters = cluster_features(features, eps_px=20, min_features=3)
        noise = [c for c in clusters if c.cluster_id < 0]
        if noise:
            assert clusters[-1].cluster_id == -1

    def test_cluster_has_centroid(self):
        features = [
            _make_feature(10, 10), _make_feature(12, 10),
            _make_feature(10, 12),
        ]
        clusters = cluster_features(features, eps_px=20, min_features=3)
        real = [c for c in clusters if c.cluster_id >= 0]
        assert len(real) == 1
        cx, cy = real[0].centroid_px
        assert 9 < cx < 13
        assert 9 < cy < 13

    def test_cluster_density(self):
        features = [
            _make_feature(10, 10), _make_feature(12, 10),
            _make_feature(10, 12),
        ]
        clusters = cluster_features(features, eps_px=20, min_features=3)
        real = [c for c in clusters if c.cluster_id >= 0]
        assert real[0].density > 0

    def test_empty_features(self):
        clusters = cluster_features([])
        assert clusters == []


class TestIdentifySiteCore:
    def test_returns_densest(self):
        c1 = Cluster(cluster_id=0, density=5.0, features=[_make_feature(10, 10)] * 3)
        c2 = Cluster(cluster_id=1, density=10.0, features=[_make_feature(50, 50)] * 3)
        core = identify_site_core([c1, c2])
        assert core.cluster_id == 1

    def test_ignores_noise(self):
        c1 = Cluster(cluster_id=0, density=5.0, features=[_make_feature(10, 10)] * 3)
        noise = Cluster(cluster_id=-1, density=100.0, features=[_make_feature(50, 50)])
        core = identify_site_core([c1, noise])
        assert core.cluster_id == 0

    def test_no_clusters_returns_none(self):
        noise = Cluster(cluster_id=-1, density=1.0, features=[_make_feature(50, 50)])
        core = identify_site_core([noise])
        assert core is None

    def test_empty_returns_none(self):
        assert identify_site_core([]) is None


class TestSettlementHierarchy:
    def test_ranking(self):
        c1 = Cluster(
            cluster_id=0, density=10.0,
            centroid_px=(10.0, 10.0), extent_px=20.0,
            features=[
                _make_feature(10, 10, class_name="building"),
                _make_feature(12, 10, class_name="platform"),
                _make_feature(10, 12, class_name="building"),
            ],
        )
        c2 = Cluster(
            cluster_id=1, density=5.0,
            centroid_px=(80.0, 80.0), extent_px=15.0,
            features=[
                _make_feature(80, 80, class_name="building"),
                _make_feature(82, 80, class_name="building"),
                _make_feature(80, 82, class_name="building"),
            ],
        )
        h = settlement_hierarchy([c1, c2], pixel_size=0.5)
        assert len(h) == 2
        assert h[0]["rank"] == 1
        assert h[1]["rank"] == 2

    def test_site_core_role(self):
        c = Cluster(
            cluster_id=0, density=10.0,
            centroid_px=(10.0, 10.0), extent_px=20.0,
            features=[
                _make_feature(10, 10, class_name="building"),
                _make_feature(12, 10, class_name="platform"),
                _make_feature(10, 12, class_name="building"),
            ],
        )
        h = settlement_hierarchy([c])
        assert h[0]["role"] == "site_core"

    def test_residential_group_role(self):
        c = Cluster(
            cluster_id=0, density=5.0,
            centroid_px=(50.0, 50.0), extent_px=10.0,
            features=[
                _make_feature(50, 50, class_name="building"),
                _make_feature(52, 50, class_name="building"),
                _make_feature(50, 52, class_name="building"),
            ],
        )
        # Not rank 0 so won't be site_core, but only buildings so residential
        h = settlement_hierarchy([c])
        # This is rank 0 with buildings but no platforms → actually it'll be site_core
        # Let me add two clusters
        c2 = Cluster(
            cluster_id=1, density=20.0,
            centroid_px=(10.0, 10.0), extent_px=5.0,
            features=[
                _make_feature(10, 10, class_name="building"),
                _make_feature(12, 10, class_name="platform"),
                _make_feature(10, 12, class_name="building"),
            ],
        )
        h = settlement_hierarchy([c2, c])
        residential = [x for x in h if x["role"] == "residential_group"]
        assert len(residential) >= 1

    def test_empty_hierarchy(self):
        h = settlement_hierarchy([])
        assert h == []

    def test_noise_excluded(self):
        noise = Cluster(cluster_id=-1, density=1.0,
                        centroid_px=(50.0, 50.0), extent_px=0.0,
                        features=[_make_feature(50, 50)])
        h = settlement_hierarchy([noise])
        assert h == []
