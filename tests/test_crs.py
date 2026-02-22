"""Tests for mayascan.crs — coordinate reference system utilities."""

import numpy as np
import pytest

from mayascan.detect import GeoInfo
from mayascan.crs import pixel_to_latlon, get_bounds_latlon


class TestPixelToLatlon:
    def test_returns_none_without_geo(self):
        assert pixel_to_latlon(0, 0, GeoInfo()) is None

    def test_returns_none_without_crs(self):
        geo = GeoInfo(transform=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
        assert pixel_to_latlon(0, 0, geo) is None

    def test_passthrough_wgs84(self):
        """If CRS is already WGS84, returns map coordinates directly."""
        geo = GeoInfo(
            crs="EPSG:4326",
            transform=(0.001, 0.0, -89.0, 0.0, -0.001, 18.0),
        )
        result = pixel_to_latlon(0, 0, geo)
        assert result is not None
        lon, lat = result
        assert abs(lon - (-89.0)) < 0.01
        assert abs(lat - 18.0) < 0.01

    def test_utm_to_latlon(self):
        """Converts UTM coordinates to lat/lon."""
        try:
            from pyproj import Transformer  # noqa: F401
        except ImportError:
            pytest.skip("pyproj not installed")

        # UTM Zone 16N — center of Yucatan area
        geo = GeoInfo(
            crs="EPSG:32616",
            transform=(0.5, 0.0, 250000.0, 0.0, -0.5, 2000000.0),
            resolution=0.5,
        )
        result = pixel_to_latlon(100, 100, geo)
        assert result is not None
        lon, lat = result
        # Should be roughly in Central America
        assert -95 < lon < -80
        assert 10 < lat < 25


class TestGetBoundsLatlon:
    def test_returns_none_without_bounds(self):
        assert get_bounds_latlon(GeoInfo()) is None

    def test_passthrough_wgs84(self):
        geo = GeoInfo(
            crs="EPSG:4326",
            bounds=(-90.0, 17.0, -89.0, 18.0),
        )
        result = get_bounds_latlon(geo)
        assert result is not None
        assert abs(result[0] - (-90.0)) < 0.001

    def test_utm_bounds(self):
        try:
            from pyproj import Transformer  # noqa: F401
        except ImportError:
            pytest.skip("pyproj not installed")

        geo = GeoInfo(
            crs="EPSG:32616",
            bounds=(200000, 1900000, 300000, 2100000),
        )
        result = get_bounds_latlon(geo)
        assert result is not None
        min_lon, min_lat, max_lon, max_lat = result
        assert min_lon < max_lon
        assert min_lat < max_lat
