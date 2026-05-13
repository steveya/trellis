"""Verification tests for bounded hybrid portfolio-AAD lanes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.analytics.portfolio_aad import QuantoCorrelationAADMarketContext
from trellis.book import Book, portfolio_aad_quanto_correlation_risk
from trellis.models.resolution.quanto import ResolvedQuantoInputs


@dataclass(frozen=True)
class _QuantoSpec:
    strike: float
    option_type: str = "call"
    notional: float = 1.0


def _resolved_inputs(corr: float = 0.25) -> ResolvedQuantoInputs:
    return ResolvedQuantoInputs(
        spot=100.0,
        fx_spot=1.10,
        valuation_date=date(2024, 11, 15),
        T=1.0,
        domestic_df=0.95,
        foreign_df=0.97,
        sigma_underlier=0.22,
        sigma_fx=0.12,
        corr=corr,
    )


def _context(corr: float = 0.25) -> QuantoCorrelationAADMarketContext:
    return QuantoCorrelationAADMarketContext(
        resolved_inputs=_resolved_inputs(corr),
        correlation_name="sx5e_eurusd",
        factor_a="SX5E",
        factor_b="EURUSD",
        currency="EUR",
    )


def _quanto_book() -> Book:
    return Book(
        {
            "call": _QuantoSpec(strike=100.0, option_type="call", notional=2.0),
            "put": _QuantoSpec(strike=95.0, option_type="put", notional=1.5),
        },
        notionals={"call": 10.0, "put": 4.0},
    )


def _book_value(book: Book, corr: float) -> float:
    return float(portfolio_aad_quanto_correlation_risk(book, _context(corr)).portfolio_value)


def test_quanto_scalar_correlation_vjp_matches_finite_difference_bump():
    book = _quanto_book()
    result = portfolio_aad_quanto_correlation_risk(book, _context(0.25))
    factor = tuple(result.risk_vector)[0]
    bump = 1.0e-5

    finite_difference = (
        _book_value(book, 0.25 + bump) - _book_value(book, 0.25 - bump)
    ) / (2.0 * bump)

    assert result.support_status == "supported"
    assert result.method_metadata["product_family"] == "quanto_option"
    assert result.method_metadata["hybrid_derivative_policy"] == (
        "bounded_quanto_scalar_correlation_vjp"
    )
    assert result.risk_vector[factor] == pytest.approx(
        finite_difference,
        rel=5.0e-6,
        abs=1.0e-6,
    )
