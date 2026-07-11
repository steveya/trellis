"""Tests for reusable double-barrier payoff primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.support.barriers import (
    DoubleBarrierSpec,
    double_barrier_hit_mask,
    double_barrier_path_payoff,
    double_barrier_state_payoff,
    resolve_double_barrier_inputs,
)
from trellis.models.double_barrier_option import (
    DoubleBarrierMonteCarloConfig,
    DoubleBarrierPDEConfig,
    price_double_barrier_option_monte_carlo_result,
    price_double_barrier_option_pde_result,
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


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("knock_out", "out"),
        ("knock-out", "out"),
        ("knock_in", "in"),
        ("knock-in", "in"),
    ],
)
def test_double_barrier_spec_normalizes_knock_aliases(alias, expected):
    spec = DoubleBarrierSpec(knock=alias)

    assert spec.knock == expected


def test_double_barrier_knock_out_aliases_preserve_knock_out_semantics():
    class BarrierStyleSpec:
        strike = 100.0
        lower_barrier = 80.0
        upper_barrier = 130.0
        barrier_style = "knock_out"

    direct = DoubleBarrierSpec(
        strike=100.0,
        lower_barrier=80.0,
        upper_barrier=130.0,
        knock="knock_out",
    )
    aliased = DoubleBarrierSpec.from_spec(BarrierStyleSpec())
    paths = np.asarray([[100.0, 131.0, 125.0]])

    assert direct.knock == aliased.knock == "out"
    assert double_barrier_path_payoff(paths, direct).tolist() == [0.0]
    assert double_barrier_path_payoff(paths, aliased).tolist() == [0.0]


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


def test_double_barrier_pde_uses_bounded_absorbing_contract():
    spec = BarrierOptionSpec(lower_barrier=75.0, upper_barrier=150.0)
    result = price_double_barrier_option_pde_result(
        _market_state(),
        spec,
        config=DoubleBarrierPDEConfig(spot_steps=121, time_steps=220),
    )

    assert result.validation_bundle == "double_barrier:pde_theta_1d"
    assert result.grid_bounds == (75.0, 150.0)
    assert result.boundary_conditions == "absorbing"
    assert result.operator_signature == "BlackScholesOperator(sigma_fn, r_fn)"
    assert result.price > 0.0
    assert result.price < result.vanilla_price


def test_double_barrier_monte_carlo_uses_two_barrier_monitors():
    result = price_double_barrier_option_monte_carlo_result(
        _market_state(),
        BarrierOptionSpec(lower_barrier=75.0, upper_barrier=150.0),
        config=DoubleBarrierMonteCarloConfig(n_paths=20_000, n_steps=126, seed=7),
    )

    assert result.validation_bundle == "double_barrier:monte_carlo_gbm"
    assert result.path_contract == ("lower_barrier:down", "upper_barrier:up")
    assert result.n_paths == 20_000
    assert result.price > 0.0


def test_double_barrier_pde_and_mc_agree_on_non_degenerate_fixture():
    spec = BarrierOptionSpec(lower_barrier=70.0, upper_barrier=160.0)
    pde = price_double_barrier_option_pde_result(
        _market_state(),
        spec,
        config=DoubleBarrierPDEConfig(spot_steps=141, time_steps=260),
    )
    mc = price_double_barrier_option_monte_carlo_result(
        _market_state(),
        spec,
        config=DoubleBarrierMonteCarloConfig(n_paths=60_000, n_steps=180, seed=19),
    )

    assert mc.price == pytest.approx(pde.price, rel=0.12, abs=0.75)


def test_double_barrier_in_out_parity_uses_vanilla_contract():
    out_spec = BarrierOptionSpec(
        lower_barrier=75.0,
        upper_barrier=150.0,
        knock="out",
    )
    in_spec = BarrierOptionSpec(
        lower_barrier=75.0,
        upper_barrier=150.0,
        knock="in",
    )
    out = price_double_barrier_option_pde_result(
        _market_state(),
        out_spec,
        config=DoubleBarrierPDEConfig(spot_steps=121, time_steps=220),
    )
    inn = price_double_barrier_option_pde_result(
        _market_state(),
        in_spec,
        config=DoubleBarrierPDEConfig(spot_steps=121, time_steps=220),
    )

    assert out.price + inn.price == pytest.approx(out.vanilla_price, rel=1e-10)
