from __future__ import annotations

from datetime import date
from math import exp, sqrt
from pathlib import Path

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.varianceswap import (
    VarianceSwapPayoff,
    VarianceSwapSpec,
)
from trellis.models.analytical.equity_exotics import (
    equity_variance_swap_outputs_analytical,
)
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)


def _market_state(
    *,
    spot: float | None = None,
    rate: float = 0.04,
    volatility: float | None = 0.22,
    settlement: date = SETTLEMENT,
) -> MarketState:
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(rate),
        vol_surface=None if volatility is None else FlatVol(volatility),
        spot=spot,
    )


def _spec(
    *,
    notional: float = 10_000.0,
    spot: float = 100.0,
    strike_variance: float = 0.04,
    replication_strikes: str | None = "60,80,100,120,140",
    replication_volatilities: str | None = "0.26,0.24,0.22,0.23,0.25",
) -> VarianceSwapSpec:
    return VarianceSwapSpec(
        notional=notional,
        spot=spot,
        strike_variance=strike_variance,
        expiry_date=EXPIRY,
        replication_strikes=replication_strikes,
        replication_volatilities=replication_volatilities,
    )


@pytest.mark.parametrize(
    "volatilities",
    [
        "0.22,0.22,0.22,0.22,0.22",
        "0.26,0.24,0.22,0.23,0.25",
    ],
)
def test_checked_variance_swap_adapter_matches_retained_approximation(volatilities):
    spec = _spec(replication_volatilities=volatilities)
    market_state = _market_state()

    actual = VarianceSwapPayoff(spec).benchmark_outputs(market_state)
    expected = equity_variance_swap_outputs_analytical(market_state, spec)

    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert VarianceSwapPayoff(spec).evaluate(market_state) == pytest.approx(
        expected["price"],
        rel=1e-13,
        abs=1e-13,
    )


def test_checked_variance_swap_adapter_uses_black_surface_when_quotes_are_absent():
    spec = _spec(
        replication_strikes=None,
        replication_volatilities=None,
    )

    outputs = VarianceSwapPayoff(spec).benchmark_outputs(
        _market_state(volatility=0.23)
    )

    assert outputs["fair_strike_variance"] == pytest.approx(0.23**2)


def test_checked_variance_swap_adapter_uses_runtime_spot_as_smile_coordinate():
    payoff = VarianceSwapPayoff(_spec())

    lower = payoff.benchmark_outputs(_market_state(spot=90.0))
    higher = payoff.benchmark_outputs(_market_state(spot=110.0))

    assert lower["fair_strike_variance"] != pytest.approx(
        higher["fair_strike_variance"]
    )


def test_checked_variance_swap_adapter_treats_near_zero_span_as_flat_smile():
    spec = _spec(
        replication_strikes=(100.0, 100.0 + 5e-13),
        replication_volatilities=(0.20, 0.30),
    )
    market_state = _market_state()

    actual = VarianceSwapPayoff(spec).benchmark_outputs(market_state)
    expected = equity_variance_swap_outputs_analytical(market_state, spec)

    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert actual["fair_strike_variance"] == pytest.approx(0.20**2)


@pytest.mark.parametrize(
    ("strikes", "volatilities", "message"),
    [
        ("80,100,120", "0.20,0.21", "same length"),
        ("80,100,100", "0.20,0.21,0.22", "strictly increasing"),
        ("80,120,100", "0.20,0.21,0.22", "strictly increasing"),
        ("0,100,120", "0.20,0.21,0.22", "positive"),
        ("80,100,120", "0.20,0.00,0.22", "positive"),
        ("80,100,120", "0.20,nan,0.22", "finite"),
        ("100", "0.20", "at least two"),
    ],
)
def test_checked_variance_swap_adapter_rejects_invalid_quote_grids(
    strikes,
    volatilities,
    message,
):
    with pytest.raises(ValueError, match=message):
        VarianceSwapPayoff(
            _spec(
                replication_strikes=strikes,
                replication_volatilities=volatilities,
            )
        ).evaluate(_market_state())


def test_checked_variance_swap_adapter_requires_surface_or_explicit_volatilities():
    with pytest.raises(ValueError, match="vol surface or explicit"):
        VarianceSwapPayoff(
            _spec(replication_volatilities=None)
        ).evaluate(_market_state(volatility=None))


def test_checked_variance_swap_adapter_has_explicit_expiry_outputs():
    spec = _spec(strike_variance=0.07)

    outputs = VarianceSwapPayoff(spec).benchmark_outputs(
        _market_state(settlement=EXPIRY)
    )

    assert outputs == {"price": 0.0, "fair_strike_variance": 0.07}


def test_checked_variance_swap_adapter_applies_discount_and_notional_once():
    rate = 0.04
    market_state = _market_state(rate=rate)
    unit_spec = _spec(notional=1.0)
    outputs = VarianceSwapPayoff(unit_spec).benchmark_outputs(market_state)
    maturity = 1.0
    atm_vol = 0.22
    slope = 100.0 * (0.25 - 0.26) / (140.0 - 60.0)
    fair = atm_vol**2 * sqrt(1.0 + 3.0 * maturity * slope * slope)
    expected = exp(-rate * maturity) * (fair - unit_spec.strike_variance)

    assert outputs["fair_strike_variance"] == pytest.approx(fair)
    assert outputs["price"] == pytest.approx(expected)
    scaled = VarianceSwapPayoff(_spec(notional=7.0)).evaluate(market_state)
    assert scaled == pytest.approx(7.0 * outputs["price"])


def test_checked_variance_swap_adapter_source_composes_reusable_primitives():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/varianceswap.py"
    )
    source = source_path.read_text(encoding="utf-8")

    for wrapper in (
        "price_equity_variance_swap_analytical",
        "equity_variance_swap_outputs_analytical",
    ):
        assert wrapper not in source
    for primitive in (
        "year_fraction",
        "linear_interp",
        "discount_factor_from_zero_rate",
    ):
        assert primitive in source
