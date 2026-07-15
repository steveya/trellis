from __future__ import annotations

from datetime import date
from math import isfinite
from pathlib import Path

import pytest

from trellis.analytics.measures import Delta
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.lookbackoption import (
    LookbackOptionPayoff,
    LookbackOptionSpec,
)
from trellis.models.analytical.equity_exotics import (
    price_equity_fixed_lookback_option_analytical,
)
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)


def _market_state(
    *,
    spot: float | None = None,
    rate: float = 0.05,
    volatility: float = 0.2,
    settlement: date = SETTLEMENT,
) -> MarketState:
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(volatility),
        spot=spot,
    )


def _spec(
    *,
    option_type: str = "call",
    strike: float = 100.0,
    running_extreme: float = 100.0,
    dividend_yield: float = 0.01,
) -> LookbackOptionSpec:
    return LookbackOptionSpec(
        notional=2.0,
        spot=100.0,
        strike=strike,
        expiry_date=EXPIRY,
        option_type=option_type,
        running_extreme=running_extreme,
        dividend_yield=dividend_yield,
        monitoring_style="continuous",
        exercise_style="european",
    )


@pytest.mark.parametrize(
    ("option_type", "strike", "running_extreme"),
    [
        ("call", 110.0, 105.0),
        ("call", 90.0, 110.0),
        ("put", 110.0, 90.0),
        ("put", 80.0, 90.0),
    ],
)
def test_checked_lookback_adapter_matches_retained_reference_across_formula_branches(
    option_type,
    strike,
    running_extreme,
):
    spec = _spec(
        option_type=option_type,
        strike=strike,
        running_extreme=running_extreme,
    )

    actual = LookbackOptionPayoff(spec).evaluate(_market_state())
    expected = price_equity_fixed_lookback_option_analytical(_market_state(), spec)

    assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_checked_lookback_adapter_has_continuous_zero_carry_limit(option_type):
    running_extreme = 110.0 if option_type == "call" else 90.0
    exact = LookbackOptionPayoff(
        _spec(
            option_type=option_type,
            running_extreme=running_extreme,
            dividend_yield=0.05,
        )
    ).evaluate(_market_state(rate=0.05))
    below = LookbackOptionPayoff(
        _spec(
            option_type=option_type,
            running_extreme=running_extreme,
            dividend_yield=0.05 - 1e-7,
        )
    ).evaluate(_market_state(rate=0.05))
    above = LookbackOptionPayoff(
        _spec(
            option_type=option_type,
            running_extreme=running_extreme,
            dividend_yield=0.05 + 1e-7,
        )
    ).evaluate(_market_state(rate=0.05))

    assert isfinite(exact)
    assert exact == pytest.approx((below + above) / 2.0, rel=2e-7, abs=2e-7)


def test_checked_lookback_adapter_honors_runtime_spot_for_generic_delta():
    payoff = LookbackOptionPayoff(_spec(option_type="call", running_extreme=100.0))
    market_state = _market_state(spot=100.0)

    delta = float(Delta(bump_pct=0.1).compute(payoff, market_state))

    assert delta > 0.0
    assert payoff.evaluate(_market_state(spot=101.0)) != pytest.approx(
        payoff.evaluate(_market_state(spot=99.0))
    )


@pytest.mark.parametrize(
    ("option_type", "running_extreme", "message"),
    [
        ("call", 99.0, "running maximum"),
        ("put", 101.0, "running minimum"),
    ],
)
def test_checked_lookback_adapter_rejects_invalid_historical_extreme(
    option_type,
    running_extreme,
    message,
):
    with pytest.raises(ValueError, match=message):
        LookbackOptionPayoff(
            _spec(option_type=option_type, running_extreme=running_extreme)
        ).evaluate(_market_state())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lookback_type", "floating_strike", "fixed_strike"),
        ("monitoring_style", "discrete", "continuous"),
        ("exercise_style", "american", "European"),
        ("option_type", "straddle", "call or put"),
    ],
)
def test_checked_lookback_adapter_rejects_unsupported_contract_semantics(
    field,
    value,
    message,
):
    spec = _spec()
    payload = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        LookbackOptionPayoff(LookbackOptionSpec(**payload)).evaluate(_market_state())


@pytest.mark.parametrize(
    ("option_type", "spot", "running_extreme", "expected"),
    [
        ("call", 115.0, 110.0, 30.0),
        ("put", 85.0, 90.0, 30.0),
    ],
)
def test_checked_lookback_adapter_uses_intrinsic_settlement_at_expiry(
    option_type,
    spot,
    running_extreme,
    expected,
):
    spec = _spec(
        option_type=option_type,
        strike=100.0,
        running_extreme=running_extreme,
    )

    price = LookbackOptionPayoff(spec).evaluate(
        _market_state(spot=spot, settlement=EXPIRY)
    )

    assert price == pytest.approx(expected)


def test_checked_lookback_adapter_rejects_nonpositive_volatility_before_expiry():
    with pytest.raises(ValueError, match="positive volatility"):
        LookbackOptionPayoff(_spec()).evaluate(_market_state(volatility=0.0))


def test_checked_lookback_adapter_source_composes_reusable_primitives():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/lookbackoption.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "price_equity_fixed_lookback_option_analytical" not in source
    for symbol in (
        "resolve_scalar_diffusion_market_inputs",
        "year_fraction",
        "normalized_option_type",
        "discount_factor_from_zero_rate",
        "standard_normal_cdf",
    ):
        assert symbol in source
