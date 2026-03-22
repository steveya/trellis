"""Ready-to-use sample objects for demos, notebooks, and testing.

All dates are pinned so results are deterministic.  No network or API keys
required — everything uses :class:`~trellis.data.mock.MockDataProvider`.

Usage::

    from trellis.samples import sample_session, sample_bond_10y
    result = sample_session().price(sample_bond_10y())
"""

from __future__ import annotations

from datetime import date

from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond


# Pinned settlement — all samples are priced as-of this date.
SETTLEMENT = date(2024, 11, 15)


def sample_bond_2y() -> Bond:
    """2-year Treasury note, 4.25% coupon."""
    return Bond(
        face=100, coupon=0.0425,
        maturity_date=date(2026, 11, 15), maturity=2, frequency=2,
    )


def sample_bond_5y() -> Bond:
    """5-year Treasury note, 4.375% coupon."""
    return Bond(
        face=100, coupon=0.04375,
        maturity_date=date(2029, 11, 15), maturity=5, frequency=2,
    )


def sample_bond_10y() -> Bond:
    """10-year Treasury note, 4.5% coupon."""
    return Bond(
        face=100, coupon=0.045,
        maturity_date=date(2034, 11, 15), maturity=10, frequency=2,
    )


def sample_bond_30y() -> Bond:
    """30-year Treasury bond, 4.75% coupon."""
    return Bond(
        face=100, coupon=0.0475,
        maturity_date=date(2054, 11, 15), maturity=30, frequency=2,
    )


def sample_curve() -> YieldCurve:
    """YieldCurve from the 2024-11-15 mock snapshot."""
    from trellis.data.mock import MockDataProvider
    provider = MockDataProvider()
    yields = provider.fetch_yields(SETTLEMENT)
    return YieldCurve.from_treasury_yields(yields)


def sample_book():
    """A four-bond book with realistic notionals."""
    from trellis.book import Book
    return Book(
        {
            "2Y": sample_bond_2y(),
            "5Y": sample_bond_5y(),
            "10Y": sample_bond_10y(),
            "30Y": sample_bond_30y(),
        },
        notionals={
            "2Y": 5_000_000,
            "5Y": 10_000_000,
            "10Y": 25_000_000,
            "30Y": 10_000_000,
        },
    )


def sample_session():
    """A Session ready for pricing — mock curve, pinned settlement."""
    from trellis.session import Session
    return Session(curve=sample_curve(), settlement=SETTLEMENT)
