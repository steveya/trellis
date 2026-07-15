from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import trellis.instruments._agent.compoundoption as compound_module
from trellis.analytics.measures import Delta
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.compoundoption import (
    CompoundOptionPayoff,
    CompoundOptionSpec,
)
from trellis.models.analytical.equity_exotics import (
    price_equity_compound_option_analytical,
)
from trellis.models.calibration.solve_request import SolveResult
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)


def _market_state(*, spot: float | None = None) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.2),
        spot=spot,
    )


def _spec(*, outer_option_type: str = "call", inner_option_type: str = "call"):
    return CompoundOptionSpec(
        notional=2.0,
        spot=100.0,
        outer_expiry_date=date(2025, 5, 15),
        inner_expiry_date=date(2025, 11, 15),
        outer_strike=10.0,
        inner_strike=100.0,
        outer_option_type=outer_option_type,
        inner_option_type=inner_option_type,
        dividend_yield=0.01,
    )


@pytest.mark.parametrize(
    ("outer_option_type", "inner_option_type"),
    [("call", "call"), ("put", "call"), ("call", "put"), ("put", "put")],
)
def test_checked_compound_adapter_matches_reference_for_all_subtypes(
    outer_option_type,
    inner_option_type,
):
    spec = _spec(
        outer_option_type=outer_option_type,
        inner_option_type=inner_option_type,
    )

    actual = CompoundOptionPayoff(spec).evaluate(_market_state())
    expected = price_equity_compound_option_analytical(_market_state(), spec)

    assert actual == pytest.approx(expected, rel=2e-8, abs=2e-8)


def test_checked_compound_adapter_honors_runtime_spot_for_generic_delta():
    payoff = CompoundOptionPayoff(_spec())
    market_state = _market_state(spot=100.0)

    delta = float(Delta(bump_pct=0.1).compute(payoff, market_state))

    assert delta > 0.0
    assert payoff.evaluate(_market_state(spot=101.0)) != pytest.approx(
        payoff.evaluate(_market_state(spot=99.0))
    )


@pytest.mark.parametrize(
    ("outer_expiry", "inner_expiry"),
    [
        (SETTLEMENT, date(2025, 11, 15)),
        (date(2025, 11, 15), date(2025, 11, 15)),
        (date(2026, 5, 15), date(2025, 11, 15)),
    ],
)
def test_checked_compound_adapter_rejects_degenerate_date_ordering(
    outer_expiry,
    inner_expiry,
):
    spec = CompoundOptionSpec(
        notional=1.0,
        spot=100.0,
        outer_expiry_date=outer_expiry,
        inner_expiry_date=inner_expiry,
        outer_strike=10.0,
        inner_strike=100.0,
    )

    with pytest.raises(
        ValueError,
        match="outer expiry must lie strictly between settlement and inner expiry",
    ):
        CompoundOptionPayoff(spec).evaluate(_market_state())


def test_checked_compound_adapter_rejects_invalid_solved_state(monkeypatch):
    monkeypatch.setattr(
        compound_module,
        "execute_solve_request",
        lambda request: SolveResult(
            solution=(float("nan"),),
            objective_value=0.0,
            success=True,
        ),
    )

    with pytest.raises(ValueError, match="positive finite critical stock state"):
        CompoundOptionPayoff(_spec()).evaluate(_market_state())


def test_checked_compound_adapter_rejects_unsuccessful_solve(monkeypatch):
    monkeypatch.setattr(
        compound_module,
        "execute_solve_request",
        lambda request: SolveResult(
            solution=(100.0,),
            objective_value=1.0,
            success=False,
        ),
    )

    with pytest.raises(ValueError, match="successful critical-stock solve"):
        CompoundOptionPayoff(_spec()).evaluate(_market_state())


def test_checked_compound_adapter_source_composes_reusable_primitives():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/compoundoption.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "price_equity_compound_option_analytical" not in source
    for symbol in (
        "resolve_scalar_diffusion_market_inputs",
        "year_fraction",
        "forward_from_dividend_yield",
        "discount_factor_from_zero_rate",
        "discounted_value",
        "black76_call",
        "black76_put",
        "standard_normal_cdf",
        "bivariate_standard_normal_cdf",
        "ObjectiveBundle",
        "SolveBounds",
        "SolveRequest",
        "execute_solve_request",
    ):
        assert symbol in source
