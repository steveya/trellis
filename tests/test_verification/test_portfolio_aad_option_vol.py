"""Verification tests for portfolio-AAD flat-vol option risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.analytics.portfolio_aad import PortfolioAADRequest, PortfolioAADResult
from trellis.analytics.risk_factors import RiskFactorId
from trellis.book import Book, portfolio_aad_equity_option_vol_risk
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)


@dataclass(frozen=True)
class _VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def _factorized_result(book: Book, vol: float = 0.20) -> PortfolioAADResult:
    return portfolio_aad_equity_option_vol_risk(
        book,
        _market_state(vol),
        vol_surface_name="spx_flat",
        currency="USD",
    )


def _black76_equity_value(spec: _VanillaEquitySpec, vol: float) -> float:
    maturity = year_fraction(SETTLEMENT, spec.expiry_date)
    discount = YieldCurve.flat(0.05).discount(maturity)
    forward = spec.spot / max(float(discount), 1e-12)
    if spec.option_type == "put":
        return float(
            spec.notional * discount * black76_put(forward, spec.strike, vol, maturity)
        )
    return float(
        spec.notional * discount * black76_call(forward, spec.strike, vol, maturity)
    )


def _book_value(book: Book, vol: float) -> float:
    return sum(
        book.notional(name) * _black76_equity_value(book[name], vol)
        for name in book
        if getattr(book[name], "exercise_style", "european") == "european"
    )


def _option_book() -> Book:
    return Book(
        {
            "call": _VanillaEquitySpec(
                spot=100.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                option_type="call",
                notional=2.0,
            ),
            "put": _VanillaEquitySpec(
                spot=95.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                option_type="put",
                notional=1.5,
            ),
        },
        notionals={"call": 100.0, "put": 40.0},
    )


def test_shared_flat_vol_factor_aggregates_across_multiple_options():
    book = _option_book()

    result = _factorized_result(book)

    assert result.support_status == "supported"
    assert len(result.risk_vector) == 1
    assert result.portfolio_value == pytest.approx(_book_value(book, 0.20))
    assert result.method_metadata["supported_position_names"] == ["call", "put"]
    assert all(
        coordinate.factor_id.object_name == "spx_flat"
        for coordinate in result.coordinates
    )


def test_selected_factor_filtering_does_not_mutate_full_option_result():
    result = _factorized_result(_option_book())
    selected_factor = tuple(result.risk_vector)[0]

    filtered = result.apply_request(PortfolioAADRequest(selected_factors=(selected_factor,)))

    assert len(result.risk_vector) == 1
    assert tuple(filtered.risk_vector) == (selected_factor,)
    assert tuple(filtered.coordinates)[0].factor_id == selected_factor


def test_vjp_sparse_flat_vol_factor_matches_independent_finite_difference_bump():
    book = _option_book()
    result = _factorized_result(book, vol=0.20)
    selected_factor = tuple(result.risk_vector)[0]
    bump = 1.0e-5

    finite_difference = (
        _book_value(book, 0.20 + bump) - _book_value(book, 0.20 - bump)
    ) / (2.0 * bump)

    assert result.risk_vector[selected_factor] == pytest.approx(
        finite_difference,
        rel=2.0e-6,
        abs=1.0e-6,
    )


def test_missing_selected_flat_vol_factor_is_reported_not_guessed():
    result = _factorized_result(_option_book())
    missing_factor = RiskFactorId(
        object_type="vol_surface",
        object_name="other_flat",
        coordinate_type="flat_vol",
        currency="USD",
    )
    request = PortfolioAADRequest(selected_factors=(missing_factor,))

    assert result.missing_selected_factors(request) == (missing_factor,)
    filtered = portfolio_aad_equity_option_vol_risk(
        _option_book(),
        _market_state(),
        request=request,
        vol_surface_name="spx_flat",
        currency="USD",
    )
    assert len(filtered.risk_vector) == 0


def test_unsupported_option_shapes_are_reported_not_guessed():
    book = Book(
        {
            "european": _VanillaEquitySpec(
                spot=100.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
            ),
            "american": _VanillaEquitySpec(
                spot=100.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                exercise_style="american",
            ),
        }
    )

    result = _factorized_result(book)

    assert result.support_status == "partial"
    assert len(result.risk_vector) == 1
    assert result.unsupported_positions[0].position_name == "american"
    assert result.unsupported_positions[0].reason == "unsupported_exercise_style"
