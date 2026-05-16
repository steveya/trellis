"""Verification tests for portfolio-AAD grid-vol option node risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.analytics.portfolio_aad import PortfolioAADRequest
from trellis.analytics.risk_factors import RiskFactorId
from trellis.book import Book, portfolio_aad_equity_option_vol_risk
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import GridVolSurface


SETTLEMENT = date(2024, 11, 15)


@dataclass(frozen=True)
class _VanillaEquitySpec:
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    notional: float = 1.0
    exercise_style: str = "european"


def _grid_surface(
    vols: tuple[tuple[float, ...], ...] = ((0.18, 0.21), (0.24, 0.27)),
) -> GridVolSurface:
    return GridVolSurface(
        expiries=(0.5, 1.5),
        strikes=(90.0, 110.0),
        vols=vols,
    )


def _market_state(surface: GridVolSurface | None = None) -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=surface or _grid_surface(),
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
                spot=100.0,
                strike=100.0,
                expiry_date=date(2025, 11, 15),
                option_type="put",
                notional=1.5,
            ),
        },
        notionals={"call": 100.0, "put": 40.0},
    )


def _black76_equity_value(spec: _VanillaEquitySpec, surface: GridVolSurface) -> float:
    maturity = year_fraction(SETTLEMENT, spec.expiry_date)
    discount = YieldCurve.flat(0.05).discount(maturity)
    forward = spec.spot / max(float(discount), 1e-12)
    vol = surface.black_vol(maturity, spec.strike)
    if spec.option_type == "put":
        return float(
            spec.notional * discount * black76_put(forward, spec.strike, vol, maturity)
        )
    return float(
        spec.notional * discount * black76_call(forward, spec.strike, vol, maturity)
    )


def _book_value(book: Book, surface: GridVolSurface) -> float:
    return sum(
        book.notional(name) * _black76_equity_value(book[name], surface)
        for name in book
    )


def _surface_with_bumped_node(
    surface: GridVolSurface,
    factor: RiskFactorId,
    bump: float,
) -> GridVolSurface:
    axes = dict(factor.axes)
    expiry = float(axes["expiry_years"])
    strike = float(axes["strike"])
    expiry_index = tuple(float(value) for value in surface.expiries).index(expiry)
    strike_index = tuple(float(value) for value in surface.strikes).index(strike)
    vols = [list(row) for row in surface.vols]
    vols[expiry_index][strike_index] += bump
    return GridVolSurface(
        expiries=surface.expiries,
        strikes=surface.strikes,
        vols=tuple(tuple(row) for row in vols),
    )


def test_grid_vol_node_vjp_matches_independent_finite_difference_bumps():
    book = _option_book()
    surface = _grid_surface()
    result = portfolio_aad_equity_option_vol_risk(
        book,
        _market_state(surface),
        vol_surface_name="spx_grid",
        currency="USD",
    )
    bump = 1.0e-5

    assert result.support_status == "supported"
    assert len(result.risk_vector) == 4
    for factor in result.risk_vector:
        bumped_up = _surface_with_bumped_node(surface, factor, bump)
        bumped_down = _surface_with_bumped_node(surface, factor, -bump)
        finite_difference = (
            _book_value(book, bumped_up) - _book_value(book, bumped_down)
        ) / (2.0 * bump)

        assert result.risk_vector[factor] == pytest.approx(
            finite_difference,
            rel=3.0e-6,
            abs=1.0e-6,
        )


def test_selected_grid_vol_factor_filtering_reports_missing_nodes():
    result = portfolio_aad_equity_option_vol_risk(
        _option_book(),
        _market_state(),
        vol_surface_name="spx_grid",
        currency="USD",
    )
    selected_factor = tuple(result.risk_vector)[0]
    missing_factor = RiskFactorId(
        object_type="vol_surface",
        object_name="other_grid",
        coordinate_type="black_vol",
        currency="USD",
        axes={"expiry_years": 0.5, "strike": 90.0},
    )

    filtered = portfolio_aad_equity_option_vol_risk(
        _option_book(),
        _market_state(),
        request=PortfolioAADRequest(selected_factors=(selected_factor, missing_factor)),
        vol_surface_name="spx_grid",
        currency="USD",
    )

    assert tuple(filtered.risk_vector) == (selected_factor,)
    assert filtered.missing_selected_factors(
        PortfolioAADRequest(selected_factors=(selected_factor, missing_factor))
    ) == (missing_factor,)
