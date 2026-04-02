from __future__ import annotations

import math

import numpy as raw_np
import pytest

from trellis.curves.yield_curve import YieldCurve
from trellis.models.monte_carlo.local_vol import local_vol_european_vanilla_price
from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.trees.algebra import (
    LATTICE_MODEL_REGISTRY,
    LOG_SPOT_MESH,
    TRINOMIAL_1F_TOPOLOGY,
    VolSurfaceTarget,
    build_lattice,
    compile_lattice_recipe,
    equity_tree,
    price_on_lattice,
    with_control,
)


def _smooth_local_vol(spot: float, time: float) -> float:
    moneyness = math.log(max(float(spot), 1e-12) / 100.0)
    return max(0.12, 0.19 + 0.03 * time + 0.08 * moneyness * moneyness)


def _local_vol_pde_price(
    *,
    option_type: str,
    american: bool = False,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
    rate: float = 0.03,
    n_x: int = 401,
    n_t: int = 401,
) -> float:
    s_max = 4.0 * max(spot, strike)
    grid = Grid(x_min=0.0, x_max=s_max, n_x=n_x, T=maturity, n_t=n_t)
    operator = BlackScholesOperator(_smooth_local_vol, lambda t: rate)
    if option_type == "call":
        terminal = raw_np.maximum(grid.x - strike, 0.0)
        lower_bc_fn = lambda t: 0.0
        upper_bc_fn = lambda t: s_max - strike * raw_np.exp(-rate * (maturity - t))
    else:
        terminal = raw_np.maximum(strike - grid.x, 0.0)
        lower_bc_fn = lambda t: strike * raw_np.exp(-rate * (maturity - t))
        upper_bc_fn = lambda t: 0.0
    values = theta_method_1d(
        grid,
        operator,
        terminal,
        theta=1.0 if american else 0.5,
        lower_bc_fn=lower_bc_fn,
        upper_bc_fn=upper_bc_fn,
        exercise_values=terminal if american else None,
    )
    idx = raw_np.searchsorted(grid.x, spot)
    idx = max(1, min(idx, len(grid.x) - 1))
    weight = (spot - grid.x[idx - 1]) / (grid.x[idx] - grid.x[idx - 1])
    return float(values[idx - 1] * (1.0 - weight) + values[idx] * weight)


def _build_local_vol_lattice(*, n_steps: int = 160):
    curve = YieldCurve.flat(0.03)
    return build_lattice(
        TRINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["local_vol"],
        calibration_target=VolSurfaceTarget(
            _smooth_local_vol,
            discount_curve=curve,
            smoothing_policy="none",
            arbitrage_checks=("probability_bounds",),
        ),
        spot=100.0,
        rate=0.03,
        maturity=1.0,
        n_steps=n_steps,
        sigma=0.20,
    )


def test_local_vol_lattice_matches_local_vol_pde_for_european_call():
    lattice = _build_local_vol_lattice()
    _, _, _, contract = compile_lattice_recipe(
        equity_tree(model_family="local_vol", branching=3, strike=100.0, option_type="call")
    )

    lattice_price = price_on_lattice(lattice, contract)
    pde_price = _local_vol_pde_price(option_type="call", american=False)

    assert lattice_price == pytest.approx(pde_price, rel=0.01)


def test_local_vol_american_put_is_bracketed_by_mc_and_pde():
    lattice = _build_local_vol_lattice()
    _, _, _, contract = compile_lattice_recipe(
        with_control(
            equity_tree(model_family="local_vol", branching=3, strike=100.0, option_type="put"),
            "american",
        )
    )

    lattice_price = price_on_lattice(lattice, contract)
    mc_price = local_vol_european_vanilla_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        discount_curve=YieldCurve.flat(0.03),
        local_vol_surface=_smooth_local_vol,
        option_type="put",
        n_paths=8_000,
        n_steps=120,
        seed=1234,
    )
    pde_american = _local_vol_pde_price(option_type="put", american=True)

    assert lattice_price >= mc_price
    assert lattice_price == pytest.approx(pde_american, rel=0.08)


def test_local_vol_lattice_records_probability_diagnostics():
    lattice = _build_local_vol_lattice(n_steps=80)
    diagnostics = lattice._lattice_calibration_diagnostics

    assert diagnostics.positivity_violations >= 0
    assert diagnostics.residuals["max_probability"] <= 1.0 + 1e-12
    assert diagnostics.residuals["min_probability"] >= -1e-12
    assert diagnostics.iterations["strategy"] == "vol_surface"
    if diagnostics.positivity_violations:
        assert len(diagnostics.fallback_flags) == diagnostics.positivity_violations
