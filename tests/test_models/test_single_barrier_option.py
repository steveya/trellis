"""Tests for reusable single-barrier option primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.barrier import barrier_option_price
from trellis.models.monte_carlo.path_state import MonteCarloPathState
from trellis.models.single_barrier_option import (
    SingleBarrierMonteCarloConfig,
    SingleBarrierPDEConfig,
    price_single_barrier_option_monte_carlo_result,
    price_single_barrier_option_pde_result,
    single_barrier_state_payoff,
)
from trellis.models.vol_surface import FlatVol


@dataclass(frozen=True)
class BarrierOptionSpec:
    notional: float = 2.0
    spot: float = 100.0
    strike: float = 100.0
    barrier: float = 80.0
    expiry_date: date = date(2025, 11, 15)
    barrier_type: str = "down_and_out"
    option_type: str = "call"
    rebate: float = 0.0
    observations_per_year: int | None = None


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.2),
        spot=101.0,
    )


def test_single_barrier_state_payoff_declares_one_monitor():
    spec = BarrierOptionSpec(barrier_type="down_and_in")
    payoff = single_barrier_state_payoff(spec)
    state = MonteCarloPathState(
        initial_value=100.0,
        n_steps=2,
        terminal_values=np.asarray([120.0, 115.0, 95.0]),
        barrier_hits={"barrier": np.asarray([False, True, True])},
    )

    monitors = payoff.path_requirement.barrier_monitors
    assert [(monitor.name, monitor.direction, monitor.level) for monitor in monitors] == [
        ("barrier", "down", 80.0)
    ]
    assert payoff.evaluate_state(state).tolist() == [0.0, 30.0, 0.0]


def test_single_barrier_pde_agrees_with_rubinstein_down_and_out_call():
    market_state = _market_state()
    spec = BarrierOptionSpec()
    pde = price_single_barrier_option_pde_result(
        market_state,
        spec,
        config=SingleBarrierPDEConfig(spot_steps=241, time_steps=420),
    )
    reference = spec.notional * barrier_option_price(
        spec.spot,
        spec.strike,
        spec.barrier,
        market_state.discount.zero_rate(1.0),
        market_state.vol_surface.black_vol(1.0, spec.strike),
        1.0,
        barrier_type=spec.barrier_type,
        option_type=spec.option_type,
    )

    assert pde.validation_bundle == "single_barrier:pde_theta_1d"
    assert pde.boundary_conditions == "absorbing_at_barrier"
    assert pde.price == pytest.approx(reference, rel=0.06, abs=0.35)
    assert 0.0 < pde.price < pde.vanilla_price


def test_single_barrier_monte_carlo_uses_one_monitor_and_agrees_with_pde():
    market_state = _market_state()
    spec = BarrierOptionSpec(barrier=75.0)
    pde = price_single_barrier_option_pde_result(
        market_state,
        spec,
        config=SingleBarrierPDEConfig(spot_steps=201, time_steps=320),
    )
    mc = price_single_barrier_option_monte_carlo_result(
        market_state,
        spec,
        config=SingleBarrierMonteCarloConfig(n_paths=50_000, n_steps=180, seed=7),
    )

    assert mc.validation_bundle == "single_barrier:monte_carlo_gbm"
    assert mc.path_contract == ("barrier:down",)
    assert mc.price == pytest.approx(pde.price, rel=0.15, abs=0.85)


def test_single_barrier_in_out_parity_uses_vanilla_contract():
    market_state = _market_state()
    out = price_single_barrier_option_pde_result(
        market_state,
        BarrierOptionSpec(barrier_type="down_and_out"),
        config=SingleBarrierPDEConfig(spot_steps=201, time_steps=320),
    )
    inn = price_single_barrier_option_pde_result(
        market_state,
        BarrierOptionSpec(barrier_type="down_and_in"),
        config=SingleBarrierPDEConfig(spot_steps=201, time_steps=320),
    )

    assert out.price + inn.price == pytest.approx(out.vanilla_price, rel=1e-10)
