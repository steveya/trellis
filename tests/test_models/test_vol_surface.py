"""Tests for VolSurface protocol and FlatVol."""

from trellis.models.vol_surface import FlatVol, VolSurface


class TestFlatVol:

    def test_constant(self):
        fv = FlatVol(0.20)
        assert fv.black_vol(1.0, 0.05) == 0.20
        assert fv.black_vol(5.0, 0.10) == 0.20
        assert fv.black_vol(0.25, 0.01) == 0.20

    def test_satisfies_protocol(self):
        assert isinstance(FlatVol(0.20), VolSurface)
