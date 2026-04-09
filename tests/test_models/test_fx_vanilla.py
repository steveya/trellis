from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    domestic = YieldCurve.flat(0.05)
    foreign = YieldCurve.flat(0.03)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=domestic,
        forecast_curves={"EUR-DISC": foreign},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        vol_surface=FlatVol(0.20),
    )


class _Spec:
    notional = 100_000
    strike = 1.05
    expiry_date = date(2025, 11, 15)
    fx_pair = "EURUSD"
    foreign_discount_key = "EUR-DISC"
    option_type = "call"
    n_paths = 12_000
    n_steps = 64
    seed = 7


def test_resolve_fx_vanilla_inputs_reads_spot_curves_and_vol():
    from trellis.models.fx_vanilla import resolve_fx_vanilla_inputs

    resolved = resolve_fx_vanilla_inputs(_market_state(), _Spec())

    assert resolved.garman_kohlhagen.spot == pytest.approx(1.10)
    assert resolved.garman_kohlhagen.strike == pytest.approx(1.05)
    assert resolved.garman_kohlhagen.T == pytest.approx(1.0)
    assert resolved.garman_kohlhagen.sigma == pytest.approx(0.20)
    assert resolved.domestic_rate == pytest.approx(0.05)
    assert resolved.foreign_rate == pytest.approx(0.03)


def test_fx_vanilla_helpers_match_adapter_prices():
    from trellis.instruments._agent.fxvanillaanalytical import (
        FXVanillaAnalyticalPayoff,
        FXVanillaOptionSpec as AnalyticalSpec,
    )
    from trellis.instruments._agent.fxvanillamontecarlo import (
        FXVanillaMonteCarloPayoff,
        FXVanillaOptionSpec as MonteCarloSpec,
    )
    from trellis.models.fx_vanilla import (
        price_fx_vanilla_analytical,
        price_fx_vanilla_monte_carlo,
    )

    analytical_spec = AnalyticalSpec(
        notional=_Spec.notional,
        strike=_Spec.strike,
        expiry_date=_Spec.expiry_date,
        fx_pair=_Spec.fx_pair,
        foreign_discount_key=_Spec.foreign_discount_key,
        option_type=_Spec.option_type,
    )
    mc_spec = MonteCarloSpec(
        notional=_Spec.notional,
        strike=_Spec.strike,
        expiry_date=_Spec.expiry_date,
        fx_pair=_Spec.fx_pair,
        foreign_discount_key=_Spec.foreign_discount_key,
        option_type=_Spec.option_type,
        n_paths=_Spec.n_paths,
        n_steps=_Spec.n_steps,
        seed=_Spec.seed,
    )
    market_state = _market_state()

    assert price_fx_vanilla_analytical(market_state, analytical_spec) == pytest.approx(
        FXVanillaAnalyticalPayoff(analytical_spec).evaluate(market_state),
        rel=1e-12,
    )
    assert price_fx_vanilla_monte_carlo(market_state, mc_spec) == pytest.approx(
        FXVanillaMonteCarloPayoff(mc_spec).evaluate(market_state),
        rel=1e-12,
    )
