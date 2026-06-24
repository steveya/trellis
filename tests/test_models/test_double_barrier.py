"""Tests for reusable double-barrier payoff primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.support.barriers import (
    DoubleBarrierSpec,
    double_barrier_hit_mask,
    double_barrier_path_payoff,
    double_barrier_state_payoff,
    resolve_double_barrier_inputs,
)
from trellis.models.monte_carlo.path_state import MonteCarloPathState
from trellis.models.vol_surface import FlatVol


@dataclass(frozen=True)
class BarrierOptionSpec:
    notional: float = 2.0
    spot: float = 100.0
    strike: float = 100.0
    lower_barrier: float = 70.0
    upper_barrier: float = 140.0
    expiry_date: date = date(2025, 11, 15)
    option_type: str = "call"
    knock: str = "out"


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.2),
        spot=101.0,
    )


def test_resolve_double_barrier_inputs_binds_market_without_pricing_route():
    resolved = resolve_double_barrier_inputs(_market_state(), BarrierOptionSpec())

    assert resolved.spot == 100.0
    assert resolved.notional == 2.0
    assert resolved.maturity == 1.0
    assert resolved.rate > 0.0
    assert resolved.sigma == 0.2


def test_double_barrier_path_payoff_tracks_lower_and_upper_hits():
    spec = DoubleBarrierSpec(
        notional=1.0,
        strike=100.0,
        lower_barrier=80.0,
        upper_barrier=130.0,
        option_type="call",
        knock="out",
    )
    paths = np.asarray(
        [
            [100.0, 112.0, 125.0],
            [100.0, 79.0, 120.0],
            [100.0, 110.0, 131.0],
        ]
    )

    assert double_barrier_hit_mask(paths, spec).tolist() == [False, True, True]
    assert double_barrier_path_payoff(paths, spec).tolist() == [25.0, 0.0, 0.0]


def test_double_barrier_state_payoff_declares_two_monitors():
    spec = DoubleBarrierSpec(
        notional=1.0,
        strike=100.0,
        lower_barrier=80.0,
        upper_barrier=130.0,
        option_type="call",
        knock="in",
    )
    payoff = double_barrier_state_payoff(spec)
    state = MonteCarloPathState(
        initial_value=100.0,
        n_steps=2,
        terminal_values=np.asarray([125.0, 120.0, 131.0]),
        barrier_hits={
            "lower_barrier": np.asarray([False, True, False]),
            "upper_barrier": np.asarray([False, False, True]),
        },
    )

    monitors = payoff.path_requirement.barrier_monitors
    assert [monitor.name for monitor in monitors] == ["lower_barrier", "upper_barrier"]
    assert payoff.evaluate_state(state).tolist() == [0.0, 20.0, 31.0]
