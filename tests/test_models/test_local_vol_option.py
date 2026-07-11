"""Tests for bounded local-vol vanilla option helpers."""

from __future__ import annotations

import pytest


def test_local_vol_option_pde_and_mc_agree_on_flat_surface():
    from trellis.models.local_vol_option import (
        LocalVolVanillaOptionSpec,
        price_local_vol_option_monte_carlo,
        price_local_vol_option_pde,
    )

    spec = LocalVolVanillaOptionSpec(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        discount_rate=0.04,
        local_vol_level=0.20,
        option_type="call",
    )

    pde_price = price_local_vol_option_pde(None, spec, n_x=161, n_t=180)
    mc_price = price_local_vol_option_monte_carlo(
        None,
        spec,
        n_paths=80_000,
        n_steps=120,
        seed=59,
    )

    assert pde_price > 0.0
    assert mc_price == pytest.approx(pde_price, rel=0.035)


def test_local_vol_option_spec_fails_closed_on_invalid_inputs():
    from trellis.models.local_vol_option import LocalVolVanillaOptionSpec

    with pytest.raises(ValueError, match="option_type"):
        LocalVolVanillaOptionSpec(option_type="payer")

    with pytest.raises(ValueError, match="local_vol_level"):
        LocalVolVanillaOptionSpec(local_vol_level=-0.01)
