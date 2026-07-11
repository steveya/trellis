from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.fx_barrier_option import (
    FXBarrierOptionSpec,
    price_fx_barrier_option_analytical,
    price_fx_barrier_option_monte_carlo_result,
    resolve_fx_barrier_inputs,
)
from trellis.models.fx_vanilla import price_fx_vanilla_analytical
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.045),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.025)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        vol_surface=FlatVol(0.14),
    )


@dataclass(frozen=True)
class _Spec:
    notional: float = 1_000_000.0
    strike: float = 1.10
    barrier: float = 1.02
    expiry_date: date = date(2025, 11, 15)
    fx_pair: str = "EURUSD"
    foreign_discount_key: str = "EUR-DISC"
    option_type: str = "call"
    barrier_type: str = "down_and_in"
    n_paths: int = 90_000
    n_steps: int = 252
    seed: int = 7


def test_resolve_fx_barrier_inputs_reads_fx_spot_curves_and_vol():
    resolved = resolve_fx_barrier_inputs(_market_state(), _Spec())

    assert resolved.spot == pytest.approx(1.10)
    assert resolved.strike == pytest.approx(1.10)
    assert resolved.barrier == pytest.approx(1.02)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.domestic_rate == pytest.approx(0.045)
    assert resolved.foreign_rate == pytest.approx(0.025)
    assert resolved.sigma == pytest.approx(0.14)
    assert resolved.barrier_type == "down_and_in"
    assert resolved.observations_per_year == 252


def test_resolve_fx_barrier_inputs_honors_explicit_observation_frequency():
    spec = FXBarrierOptionSpec.from_spec(_Spec(), observations_per_year=52)

    resolved = resolve_fx_barrier_inputs(_market_state(), spec)

    assert resolved.observations_per_year == 52


def test_fx_barrier_mc_agrees_with_analytical_for_down_and_in_call():
    market_state = _market_state()
    spec = _Spec()

    analytical = price_fx_barrier_option_analytical(market_state, spec)
    mc = price_fx_barrier_option_monte_carlo_result(market_state, spec)

    assert mc.validation_bundle == "fx_barrier:monte_carlo_gbm"
    assert mc.price == pytest.approx(analytical, rel=0.03, abs=1_000.0)


def test_fx_barrier_in_out_parity_matches_fx_vanilla():
    market_state = _market_state()
    knock_in = FXBarrierOptionSpec.from_spec(_Spec(barrier_type="down_and_in"))
    knock_out = FXBarrierOptionSpec.from_spec(_Spec(barrier_type="down_and_out"))

    in_price = price_fx_barrier_option_analytical(market_state, knock_in)
    out_price = price_fx_barrier_option_analytical(market_state, knock_out)
    vanilla = price_fx_vanilla_analytical(market_state, knock_in)

    assert in_price + out_price == pytest.approx(vanilla, rel=1e-10)
