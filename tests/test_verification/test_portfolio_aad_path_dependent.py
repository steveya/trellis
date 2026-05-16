"""Verification tests for bounded path-dependent portfolio-AAD lanes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.book import Book, portfolio_aad_arithmetic_asian_vol_risk
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)


@dataclass(frozen=True)
class _ArithmeticAsianSpec:
    spot: float
    strike: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"
    averaging_type: str = "arithmetic"
    dividend_yield: float = 0.0


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _asian_book() -> Book:
    return Book(
        {
            "asian_call": _ArithmeticAsianSpec(
                spot=100.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                observation_dates=(
                    date(2025, 2, 15),
                    date(2025, 5, 15),
                    date(2025, 8, 15),
                    date(2025, 11, 15),
                ),
                option_type="call",
                notional=2.0,
            ),
            "asian_put": _ArithmeticAsianSpec(
                spot=95.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                observation_dates=(
                    date(2025, 2, 15),
                    date(2025, 5, 15),
                    date(2025, 8, 15),
                    date(2025, 11, 15),
                ),
                option_type="put",
                notional=1.5,
            ),
        },
        notionals={"asian_call": 10.0, "asian_put": 4.0},
    )


def _book_value(book: Book, vol: float) -> float:
    return float(portfolio_aad_arithmetic_asian_vol_risk(book, _market_state(vol)).portfolio_value)


def test_arithmetic_asian_flat_vol_vjp_matches_finite_difference_bump():
    book = _asian_book()
    result = portfolio_aad_arithmetic_asian_vol_risk(
        book,
        _market_state(0.20),
        vol_surface_name="spx_flat",
        currency="USD",
    )
    selected_factor = tuple(result.risk_vector)[0]
    bump = 1.0e-5

    finite_difference = (
        _book_value(book, 0.20 + bump) - _book_value(book, 0.20 - bump)
    ) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.method_metadata["path_derivative_policy"] == (
        "lognormal_moment_matching_smooth_path_summary"
    )
    assert result.risk_vector[selected_factor] == pytest.approx(
        finite_difference,
        rel=5.0e-6,
        abs=1.0e-6,
    )
