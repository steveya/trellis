"""Tests for reusable volatility-surface bucket shocks."""

from __future__ import annotations

import pytest

from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.models.vol_surface_shocks import build_vol_surface_shock_surface


def test_bucket_surface_reexpresses_grid_surface_on_requested_bucket_grid():
    surface = GridVolSurface(
        expiries=(1.0, 2.0),
        strikes=(90.0, 110.0),
        vols=((0.25, 0.22), (0.27, 0.24)),
    )

    shock_surface = build_vol_surface_shock_surface(
        surface,
        expiries=(1.0, 1.5, 2.0),
        strikes=(90.0, 100.0, 110.0),
    )
    bucketed = shock_surface.bucketed_surface()

    assert bucketed.black_vol(1.5, 100.0) == pytest.approx(surface.black_vol(1.5, 100.0), abs=1e-12)
    bucket = shock_surface.bucket_for(1.5, 100.0)
    assert bucket.is_exact_surface_node is False


def test_apply_bumps_bumps_requested_bucket_nodes():
    surface = GridVolSurface(
        expiries=(1.0, 2.0),
        strikes=(90.0, 110.0),
        vols=((0.25, 0.22), (0.27, 0.24)),
    )

    shock_surface = build_vol_surface_shock_surface(
        surface,
        expiries=(1.0, 1.5, 2.0),
        strikes=(90.0, 100.0, 110.0),
    )
    shocked = shock_surface.apply_bumps({(1.5, 100.0): 100.0})

    assert shocked.black_vol(1.5, 100.0) == pytest.approx(surface.black_vol(1.5, 100.0) + 0.01, abs=1e-12)
    assert shocked.black_vol(1.0, 90.0) == pytest.approx(surface.black_vol(1.0, 90.0), abs=1e-12)


def test_flat_vol_expands_to_bucket_surface_with_warning():
    surface = FlatVol(0.20)

    shock_surface = build_vol_surface_shock_surface(
        surface,
        expiries=(1.0, 2.0),
        strikes=(90.0, 110.0),
    )
    bucketed = shock_surface.bucketed_surface()

    assert bucketed.black_vol(1.0, 90.0) == pytest.approx(0.20)
    assert bucketed.black_vol(2.0, 110.0) == pytest.approx(0.20)
    assert {warning.code for warning in shock_surface.warnings} == {"flat_surface_expanded"}
