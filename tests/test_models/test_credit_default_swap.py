"""Tests for reusable single-name CDS pricing helpers."""

from datetime import date

import pytest

from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.credit_default_swap import (
    build_cds_schedule,
    interval_default_probability,
    normalize_cds_running_spread,
    price_cds_analytical,
    price_cds_monte_carlo,
)
from trellis.core.types import DayCountConvention, Frequency


SETTLE = date(2024, 11, 15)
MATURITY = date(2029, 11, 15)


def _schedule():
    return build_cds_schedule(
        SETTLE,
        MATURITY,
        Frequency.QUARTERLY,
        DayCountConvention.ACT_360,
    )


class TestCreditDefaultSwapHelpers:

    def test_build_cds_schedule_accepts_legacy_time_origin_keyword(self):
        schedule = build_cds_schedule(
            SETTLE,
            MATURITY,
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
            time_origin=SETTLE,
        )

        assert schedule.time_origin == SETTLE

    def test_normalize_cds_running_spread_accepts_bps_and_decimal(self):
        assert normalize_cds_running_spread(100.0) == pytest.approx(0.01)
        assert normalize_cds_running_spread(0.01) == pytest.approx(0.01)

    def test_interval_default_probability_uses_survival_ratio(self):
        credit_curve = CreditCurve.flat(0.02)
        t0 = 1.0
        t1 = 1.5
        observed = interval_default_probability(credit_curve, t0, t1)
        expected = 1.0 - (
            float(credit_curve.survival_probability(t1))
            / float(credit_curve.survival_probability(t0))
        )
        assert observed == pytest.approx(expected)

    def test_cds_analytical_treats_equivalent_spread_quotes_the_same(self):
        schedule = _schedule()
        credit_curve = CreditCurve.flat(0.02)
        discount_curve = YieldCurve.flat(0.05)

        pv_bps = price_cds_analytical(
            notional=100.0,
            spread_quote=100.0,
            recovery=0.4,
            schedule=schedule,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
        )
        pv_decimal = price_cds_analytical(
            notional=100.0,
            spread_quote=0.01,
            recovery=0.4,
            schedule=schedule,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
        )

        assert pv_bps == pytest.approx(pv_decimal, rel=1e-12, abs=1e-12)

    def test_cds_monte_carlo_tracks_analytical_price(self):
        schedule = _schedule()
        credit_curve = CreditCurve.flat(0.02)
        discount_curve = YieldCurve.flat(0.05)

        analytical = price_cds_analytical(
            notional=100.0,
            spread_quote=100.0,
            recovery=0.4,
            schedule=schedule,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
        )
        monte_carlo = price_cds_monte_carlo(
            notional=100.0,
            spread_quote=100.0,
            recovery=0.4,
            schedule=schedule,
            credit_curve=credit_curve,
            discount_curve=discount_curve,
            n_paths=500000,
            seed=42,
        )

        assert monte_carlo == pytest.approx(analytical, abs=0.02)
