"""Tests for volatility surface implementations."""

import pytest

from trellis.models.vol_surface import FlatVol, GridVolSurface, VolSurface


class TestFlatVol:

    def test_constant(self):
        fv = FlatVol(0.20)
        assert fv.black_vol(1.0, 0.05) == 0.20
        assert fv.black_vol(5.0, 0.10) == 0.20
        assert fv.black_vol(0.25, 0.01) == 0.20

    def test_satisfies_protocol(self):
        assert isinstance(FlatVol(0.20), VolSurface)


class TestGridVolSurface:

    def test_exact_grid_point(self):
        surface = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        )
        assert surface.black_vol(1.0, 90.0) == pytest.approx(0.25)
        assert surface.black_vol(2.0, 110.0) == pytest.approx(0.24)

    def test_bilinear_interpolation(self):
        surface = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        )
        assert surface.black_vol(1.5, 100.0) == pytest.approx(0.245, abs=1e-12)

    def test_flat_extrapolation_at_boundaries(self):
        surface = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        )
        assert surface.black_vol(0.25, 100.0) == pytest.approx(0.235, abs=1e-12)
        assert surface.black_vol(3.0, 120.0) == pytest.approx(0.24, abs=1e-12)

    def test_grid_surface_satisfies_protocol(self):
        surface = GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        )
        assert isinstance(surface, VolSurface)
