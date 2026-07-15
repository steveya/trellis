from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import trellis.instruments._agent.chooseroption as chooser_module
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.chooseroption import (
    ChooserOptionPayoff,
    ChooserOptionSpec,
)
from trellis.models.analytical.equity_exotics import (
    price_equity_chooser_option_analytical,
)
from trellis.models.vol_surface import FlatVol
from trellis.analytics.measures import Delta
from trellis.models.calibration.solve_request import SolveResult


SETTLEMENT = date(2024, 11, 15)


def _market_state(*, spot: float | None = None) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.2),
        spot=spot,
    )


def test_checked_chooser_adapter_matches_retained_reference_for_unequal_terms():
    spec = ChooserOptionSpec(
        notional=2.0,
        spot=100.0,
        choose_date=date(2025, 5, 15),
        call_expiry_date=date(2025, 11, 15),
        put_expiry_date=date(2026, 5, 15),
        call_strike=105.0,
        put_strike=95.0,
        dividend_yield=0.01,
    )

    actual = ChooserOptionPayoff(spec).evaluate(_market_state())
    expected = price_equity_chooser_option_analytical(_market_state(), spec)

    assert actual == pytest.approx(expected, rel=2e-8, abs=2e-8)


def test_checked_chooser_adapter_honors_runtime_spot_for_generic_delta():
    spec = ChooserOptionSpec(
        notional=1.0,
        spot=100.0,
        choose_date=date(2025, 5, 15),
        call_expiry_date=date(2025, 11, 15),
        put_expiry_date=date(2025, 11, 15),
        call_strike=100.0,
        put_strike=100.0,
    )
    payoff = ChooserOptionPayoff(spec)
    market_state = _market_state(spot=100.0)

    delta = float(Delta(bump_pct=0.1).compute(payoff, market_state))

    assert delta > 0.0
    assert payoff.evaluate(_market_state(spot=101.0)) != pytest.approx(
        payoff.evaluate(_market_state(spot=99.0))
    )


def test_checked_chooser_adapter_rejects_invalid_solved_state(monkeypatch):
    spec = ChooserOptionSpec(
        notional=1.0,
        spot=100.0,
        choose_date=date(2025, 5, 15),
        call_expiry_date=date(2025, 11, 15),
        put_expiry_date=date(2025, 11, 15),
        call_strike=100.0,
        put_strike=100.0,
    )
    monkeypatch.setattr(
        chooser_module,
        "execute_solve_request",
        lambda request: SolveResult(
            solution=(float("nan"),),
            objective_value=0.0,
            success=True,
        ),
    )

    with pytest.raises(ValueError, match="positive finite critical stock state"):
        ChooserOptionPayoff(spec).evaluate(_market_state())


@pytest.mark.parametrize(
    ("choose_date", "call_expiry", "put_expiry"),
    [
        (SETTLEMENT, date(2025, 11, 15), date(2025, 11, 15)),
        (date(2025, 11, 15), date(2025, 11, 15), date(2026, 5, 15)),
        (date(2025, 5, 15), date(2025, 4, 15), date(2025, 11, 15)),
    ],
)
def test_checked_chooser_adapter_rejects_degenerate_date_ordering(
    choose_date,
    call_expiry,
    put_expiry,
):
    spec = ChooserOptionSpec(
        notional=1.0,
        spot=100.0,
        choose_date=choose_date,
        call_expiry_date=call_expiry,
        put_expiry_date=put_expiry,
        call_strike=100.0,
        put_strike=100.0,
    )

    with pytest.raises(ValueError, match="strictly between settlement and both expiries"):
        ChooserOptionPayoff(spec).evaluate(_market_state())


def test_checked_chooser_adapter_source_composes_reusable_primitives():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/chooseroption.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "price_equity_chooser_option_analytical" not in source
    for symbol in (
        "resolve_scalar_diffusion_market_inputs",
        "year_fraction",
        "forward_from_dividend_yield",
        "discount_factor_from_zero_rate",
        "discounted_value",
        "black76_call",
        "black76_put",
        "bivariate_standard_normal_cdf",
        "ObjectiveBundle",
        "SolveBounds",
        "SolveRequest",
        "execute_solve_request",
    ):
        assert symbol in source
