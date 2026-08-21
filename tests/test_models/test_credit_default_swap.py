"""Tests for reusable single-name CDS pricing helpers."""

from datetime import date
from pathlib import Path

import pytest
import numpy as np

from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.core.market_state import MarketState
from trellis.conventions.calendar import BusinessDayAdjustment, US_SETTLEMENT
from trellis.conventions.schedule import RollConvention, StubType
from trellis.models.credit_default_swap import (
    build_cds_schedule,
    interval_default_probability,
    normalize_cds_running_spread,
    normalize_cds_upfront_quote,
    price_cds_analytical,
    price_cds_monte_carlo,
)
from trellis.core.types import DayCountConvention, Frequency
from trellis.instruments._agent.cds import CDSPayoff, CDSSpec


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

    def test_checked_cds_adapter_composes_generic_default_event_primitives(self):
        source = (
            Path(__file__).parents[2]
            / "trellis"
            / "instruments"
            / "_agent"
            / "cds.py"
        ).read_text(encoding="utf-8")

        assert "build_period_schedule(" in source
        assert "build_default_event_grid(" in source
        assert "conditional_event_probabilities_from_curve(" in source
        assert "expected_first_event_weights(" in source
        assert "sample_first_event_weights(" in source
        assert "CouponAccrual(" in source
        assert "ProtectionPayment(" in source
        assert "build_cds_schedule" not in source
        assert "price_cds_analytical" not in source
        assert "price_cds_monte_carlo" not in source

    def test_build_cds_schedule_accepts_legacy_time_origin_keyword(self):
        schedule = build_cds_schedule(
            SETTLE,
            MATURITY,
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
            time_origin=SETTLE,
        )

        assert schedule.time_origin == SETTLE

    def test_build_cds_schedule_applies_standard_business_day_adjustment(self):
        schedule = build_cds_schedule(
            date(2024, 9, 20),
            date(2029, 12, 20),
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
            time_origin=SETTLE,
        )

        assert schedule.periods[3].payment_date == date(2025, 9, 22)
        assert schedule.periods[4].payment_date == date(2025, 12, 22)

    def test_build_cds_schedule_supports_imm_roll_with_holiday_calendar_and_payment_lag(self):
        schedule = build_cds_schedule(
            date(2024, 9, 20),
            date(2025, 9, 22),
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
            time_origin=SETTLE,
            calendar=US_SETTLEMENT,
            business_day_adjustment=BusinessDayAdjustment.MODIFIED_FOLLOWING,
            roll_convention=RollConvention.IMM,
            stub=StubType.SHORT_FIRST,
            payment_lag_days=1,
        )

        assert schedule.period_end_dates == (
            date(2024, 12, 20),
            date(2025, 3, 20),
            date(2025, 6, 20),
            date(2025, 9, 22),
        )
        assert schedule.payment_dates[-1] == date(2025, 9, 23)

    def test_normalize_cds_running_spread_accepts_bps_and_decimal(self):
        assert normalize_cds_running_spread(100.0) == pytest.approx(0.01)
        assert normalize_cds_running_spread(0.01) == pytest.approx(0.01)

    def test_normalize_cds_upfront_quote_accepts_points_and_decimal(self):
        assert normalize_cds_upfront_quote(5.25) == pytest.approx(0.0525)
        assert normalize_cds_upfront_quote(0.0525) == pytest.approx(0.0525)
        assert normalize_cds_upfront_quote(-2.0) == pytest.approx(-0.02)

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

    def test_cds_analytical_matches_financepy_flat_hazard_benchmark(self):
        financepy = pytest.importorskip("financepy")
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.products.credit.cds import CDS
        from financepy.products.credit.cds_curve import CDSCurve
        from financepy.utils.date import Date
        from financepy.utils.day_count import DayCountTypes
        from financepy.utils.frequency import FrequencyTypes

        value_dt = Date(15, 11, 2024)
        discount_curve = DiscountCurveFlat(value_dt, 0.04)
        cds = CDS(
            value_dt,
            "5Y",
            0.015,
            notional=10_000_000.0,
            long_protect=True,
            freq_type=FrequencyTypes.QUARTERLY,
            dc_type=DayCountTypes.ACT_360,
        )
        issuer_curve = CDSCurve(value_dt, [], discount_curve, 0.4)
        maturity_years = max((cds.maturity_dt - value_dt) / 365.0, 1.0)
        issuer_curve._times = np.asarray([0.0, float(maturity_years)], dtype=float)
        issuer_curve._qs = np.asarray([1.0, np.exp(-0.025 * maturity_years)], dtype=float)
        expected = float(cds.value(value_dt, issuer_curve, 0.4)["clean_pv"])

        schedule = build_cds_schedule(
            date(2024, 9, 20),
            date(2029, 12, 20),
            Frequency.QUARTERLY,
            DayCountConvention.ACT_360,
            time_origin=SETTLE,
        )
        observed = price_cds_analytical(
            notional=10_000_000.0,
            spread_quote=0.015,
            recovery=0.4,
            schedule=schedule,
            credit_curve=CreditCurve.flat(0.025),
            discount_curve=YieldCurve.flat(0.04),
        )

        assert observed == pytest.approx(expected, rel=0.02)

    def test_cds_agent_payoff_uses_valuation_date_for_analytical_benchmarks(self):
        spec = CDSSpec(
            notional=10_000_000.0,
            spread=0.015,
            recovery=0.4,
            valuation_date=SETTLE,
            start_date=date(2024, 9, 20),
            end_date=date(2029, 12, 20),
            pricing_method="analytical",
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.04),
            credit_curve=CreditCurve.flat(0.025),
        )

        payoff = CDSPayoff(spec)
        observed = payoff.evaluate(market_state)

        assert observed == pytest.approx(-5650.965650176979, rel=0.02)

    def test_cds_agent_payoff_explicit_analytical_composition_matches_reference(self):
        spec = CDSSpec(
            notional=10_000_000.0,
            spread=0.015,
            recovery=0.4,
            valuation_date=SETTLE,
            start_date=date(2024, 9, 20),
            end_date=date(2029, 12, 20),
            pricing_method="analytical",
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.04),
            credit_curve=CreditCurve.flat(0.025),
        )
        reference_schedule = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            spec.day_count,
            time_origin=spec.valuation_date,
        )
        reference = price_cds_analytical(
            notional=spec.notional,
            spread_quote=spec.spread,
            recovery=spec.recovery,
            schedule=reference_schedule,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
        )

        observed = CDSPayoff(spec).evaluate(market_state)

        assert observed == pytest.approx(reference, rel=1e-12, abs=1e-8)

    def test_cds_agent_payoff_forward_start_includes_survival_to_start(self):
        forward_start = date(2025, 11, 15)
        spec = CDSSpec(
            notional=10_000_000.0,
            spread=0.015,
            recovery=0.4,
            valuation_date=SETTLE,
            start_date=forward_start,
            end_date=date(2030, 11, 15),
            pricing_method="analytical",
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.04),
            credit_curve=CreditCurve.flat(0.025),
        )
        reference_schedule = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            spec.day_count,
            time_origin=spec.valuation_date,
        )
        reference = price_cds_analytical(
            notional=spec.notional,
            spread_quote=spec.spread,
            recovery=spec.recovery,
            schedule=reference_schedule,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
        )

        observed = CDSPayoff(spec).evaluate(market_state)

        assert observed == pytest.approx(reference, rel=1e-12, abs=1e-8)

    def test_cds_agent_payoff_sampled_composition_matches_reference_monte_carlo(self):
        spec = CDSSpec(
            notional=100.0,
            spread=0.01,
            recovery=0.4,
            valuation_date=SETTLE,
            start_date=SETTLE,
            end_date=MATURITY,
            pricing_method="monte_carlo",
            n_paths=200_000,
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            credit_curve=CreditCurve.flat(0.02),
        )
        reference_schedule = build_cds_schedule(
            spec.start_date,
            spec.end_date,
            spec.frequency,
            spec.day_count,
            time_origin=spec.valuation_date,
        )
        reference = price_cds_monte_carlo(
            notional=spec.notional,
            spread_quote=spec.spread,
            recovery=spec.recovery,
            schedule=reference_schedule,
            credit_curve=market_state.credit_curve,
            discount_curve=market_state.discount,
            n_paths=spec.n_paths,
            seed=42,
        )

        observed = CDSPayoff(spec).evaluate(market_state)

        assert observed == pytest.approx(reference, abs=0.02)

    def test_cds_agent_payoff_rejects_qmc_instead_of_using_pseudo_random_weights(self):
        spec = CDSSpec(
            notional=100.0,
            spread=0.01,
            recovery=0.4,
            valuation_date=SETTLE,
            start_date=SETTLE,
            end_date=MATURITY,
            pricing_method="qmc",
            n_paths=20_000,
        )
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            credit_curve=CreditCurve.flat(0.02),
        )

        with pytest.raises(ValueError, match="unsupported CDS pricing_method.*qmc"):
            CDSPayoff(spec).evaluate(market_state)
