"""Verification tests for factorized portfolio-AAD curve risk."""

from datetime import date

import pytest

from trellis.analytics.portfolio_aad import PortfolioAADRequest, PortfolioAADResult
from trellis.analytics.risk_factors import RiskFactorId
from trellis.book import Book, portfolio_aad_curve_risk
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond


SETTLEMENT = date(2024, 11, 15)


def _bond(**kwargs):
    defaults = dict(
        face=100.0,
        coupon=0.05,
        maturity_date=date(2034, 11, 15),
        maturity=10,
        frequency=2,
    )
    defaults.update(kwargs)
    return Bond(**defaults)


def _book_value(book: Book, curve: YieldCurve) -> float:
    return sum(
        book.notional(name) * book[name].price(curve, SETTLEMENT)
        for name in book
    )


def _curve_with_bumped_node(curve: YieldCurve, index: int, bump: float) -> YieldCurve:
    rates = list(curve.rates)
    rates[index] = rates[index] + bump
    return YieldCurve(curve.tenors, rates)


def _factorized_result(book: Book, curve: YieldCurve) -> PortfolioAADResult:
    risk = portfolio_aad_curve_risk(book, curve, SETTLEMENT)
    return PortfolioAADResult.from_payload(risk.metadata["portfolio_aad_result"])


def test_shared_curve_factors_aggregate_across_multiple_bonds():
    curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
    book = Book(
        {
            "long": _bond(coupon=0.05),
            "short": _bond(coupon=0.04, maturity_date=date(2030, 11, 15), maturity=6),
        },
        notionals={"long": 1_000_000, "short": 750_000},
    )

    result = _factorized_result(book, curve)

    assert len(result.risk_vector) == len(curve.tenors)
    assert len({factor.key for factor in result.risk_vector}) == len(curve.tenors)
    assert result.portfolio_value == pytest.approx(_book_value(book, curve))
    assert all(
        factor.object_name == "shared_curve"
        for factor in result.risk_vector
    )


def test_selected_factor_filtering_does_not_mutate_full_result():
    curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
    book = Book({"long": _bond()}, notionals={"long": 1_000_000})
    result = _factorized_result(book, curve)
    selected_factor = tuple(result.risk_vector)[2]

    filtered = result.apply_request(PortfolioAADRequest(selected_factors=(selected_factor,)))

    assert len(result.risk_vector) == 4
    assert tuple(filtered.risk_vector) == (selected_factor,)
    assert tuple(filtered.coordinates)[0].factor_id == selected_factor


def test_vjp_sparse_factor_matches_independent_finite_difference_bump():
    curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
    book = Book(
        {
            "long": _bond(coupon=0.05),
            "belly": _bond(coupon=0.035, maturity_date=date(2029, 11, 15), maturity=5),
        },
        notionals={"long": 1_000_000, "belly": 500_000},
    )
    result = _factorized_result(book, curve)
    node_index = 2
    selected_factor = result.coordinates[node_index].factor_id
    bump = 1.0e-5

    bumped_up = _curve_with_bumped_node(curve, node_index, bump)
    bumped_down = _curve_with_bumped_node(curve, node_index, -bump)
    finite_difference = (
        _book_value(book, bumped_up) - _book_value(book, bumped_down)
    ) / (2.0 * bump)

    assert result.risk_vector[selected_factor] == pytest.approx(
        finite_difference,
        rel=2.0e-5,
        abs=1.0e-3,
    )


def test_missing_selected_factor_is_reported_not_guessed():
    curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
    book = Book({"long": _bond()}, notionals={"long": 1_000_000})
    result = _factorized_result(book, curve)
    missing_factor = RiskFactorId(
        object_type="curve",
        object_name="other_curve",
        coordinate_type="zero_rate",
        axes={"tenor_years": 30.0},
    )
    request = PortfolioAADRequest(selected_factors=(missing_factor,))

    assert result.missing_selected_factors(request) == (missing_factor,)
    assert len(result.apply_request(request).risk_vector) == 0
