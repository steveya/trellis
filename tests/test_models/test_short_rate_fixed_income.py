"""Tests for generic short-rate fixed-income helper kits."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.callable_bond import CallableBondSpec
from trellis.models.callable_bond_tree import (
    build_callable_bond_lattice,
    compile_callable_bond_contract_spec,
    price_callable_bond_on_lattice,
    straight_bond_present_value,
)
from trellis.models.short_rate_fixed_income import (
    EmbeddedFixedIncomeEventTimeline,
    EmbeddedFixedIncomeExerciseConfig,
    FixedIncomeCouponCashflow,
    build_embedded_fixed_income_event_timeline,
    build_embedded_fixed_income_pde_event_buckets,
    compile_embedded_fixed_income_lattice_contract_spec,
    present_value_fixed_coupon_bond,
)
from trellis.models.vol_surface import FlatVol


def _callable_spec() -> CallableBondSpec:
    return CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=[date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15)],
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )


class _PuttableSpec:
    def __init__(self) -> None:
        self.notional = 100.0
        self.coupon = 0.05
        self.start_date = date(2025, 1, 15)
        self.end_date = date(2035, 1, 15)
        self.put_dates = (date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15))
        self.put_price = 100.0
        self.frequency = Frequency.SEMI_ANNUAL
        self.day_count = DayCountConvention.ACT_365


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


def test_build_embedded_fixed_income_event_timeline_preserves_control_semantics():
    callable_timeline = build_embedded_fixed_income_event_timeline(
        _callable_spec(),
        settlement=date(2024, 11, 15),
    )
    puttable_timeline = build_embedded_fixed_income_event_timeline(
        _PuttableSpec(),
        settlement=date(2024, 11, 15),
    )

    assert callable_timeline.exercise.control_style == "issuer_min"
    assert callable_timeline.exercise.projection_kind == "project_min"
    assert puttable_timeline.exercise.control_style == "holder_max"
    assert puttable_timeline.exercise.projection_kind == "project_max"
    assert callable_timeline.terminal_redemption_cash > 100.0


def test_generic_lattice_contract_spec_matches_callable_wrapper_price():
    market_state = _market_state()
    spec = _callable_spec()
    lattice = build_callable_bond_lattice(
        market_state,
        spec,
        model="hull_white",
        n_steps=80,
    )

    generic_contract = compile_embedded_fixed_income_lattice_contract_spec(
        spec,
        settlement=market_state.settlement,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
    )
    wrapper_contract = compile_callable_bond_contract_spec(
        spec,
        settlement=market_state.settlement,
        dt=lattice.dt,
        n_steps=lattice.n_steps,
    )

    assert generic_contract.control.objective == wrapper_contract.control.objective
    assert generic_contract.metadata["coupon_by_step"] == wrapper_contract.metadata["coupon_by_step"]
    assert price_callable_bond_on_lattice(lattice, contract_spec=generic_contract) == pytest.approx(
        price_callable_bond_on_lattice(lattice, contract_spec=wrapper_contract),
        rel=1e-12,
    )


def test_generic_lattice_contract_spec_includes_terminal_coupon_in_terminal_payoff():
    spec = _callable_spec()
    contract = compile_embedded_fixed_income_lattice_contract_spec(
        spec,
        settlement=date(2024, 11, 15),
        dt=0.125,
        n_steps=80,
    )

    terminal_value = contract.claim.terminal_payoff(80, 0, None, None)
    assert terminal_value == pytest.approx(102.52054794520548)


def test_generic_lattice_exercise_policy_excludes_maturity_step():
    spec = CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=[date(2028, 1, 15), date(2035, 1, 15)],
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )
    contract = compile_embedded_fixed_income_lattice_contract_spec(
        spec,
        settlement=date(2024, 11, 15),
        dt=0.125,
        n_steps=80,
    )

    assert contract.control.exercise_steps
    assert max(contract.control.exercise_steps) < 80


def test_generic_pde_event_buckets_support_callable_and_puttable_projection():
    callable_timeline = build_embedded_fixed_income_event_timeline(
        _callable_spec(),
        settlement=date(2024, 11, 15),
    )
    puttable_timeline = build_embedded_fixed_income_event_timeline(
        _PuttableSpec(),
        settlement=date(2024, 11, 15),
    )

    callable_buckets = build_embedded_fixed_income_pde_event_buckets(
        callable_timeline,
        day_count=DayCountConvention.ACT_365,
        maturity_date=date(2035, 1, 15),
    )
    puttable_buckets = build_embedded_fixed_income_pde_event_buckets(
        puttable_timeline,
        day_count=DayCountConvention.ACT_365,
        maturity_date=date(2035, 1, 15),
    )

    assert any(
        transform.kind == "project_min"
        for bucket in callable_buckets
        for transform in bucket.transforms
    )
    assert any(
        transform.kind == "project_max"
        for bucket in puttable_buckets
        for transform in bucket.transforms
    )


def test_generic_pde_event_buckets_accumulate_duplicate_coupon_dates():
    timeline = EmbeddedFixedIncomeEventTimeline(
        settlement=date(2024, 11, 15),
        coupon_cashflows=(
            FixedIncomeCouponCashflow(
                payment_date=date(2028, 1, 15),
                amount=1.25,
                accrual_fraction=0.25,
                time_to_payment=3.0,
            ),
            FixedIncomeCouponCashflow(
                payment_date=date(2028, 1, 15),
                amount=0.75,
                accrual_fraction=0.15,
                time_to_payment=3.0,
            ),
        ),
        exercise=EmbeddedFixedIncomeExerciseConfig(
            schedule_dates=(),
            exercise_price_cash=100.0,
            exercise_style="issuer_call",
            control_style="issuer_min",
            reference_bound="upper",
            projection_kind="project_min",
        ),
        terminal_coupon_cash=0.0,
        terminal_redemption_cash=100.0,
    )

    buckets = build_embedded_fixed_income_pde_event_buckets(
        timeline,
        day_count=DayCountConvention.ACT_365,
        maturity_date=date(2035, 1, 15),
    )

    assert len(buckets) == 1
    add_cashflow = [t for t in buckets[0].transforms if t.kind == "add_cashflow"]
    assert len(add_cashflow) == 1
    assert float(add_cashflow[0].payload) == pytest.approx(2.0)


def test_present_value_fixed_coupon_bond_matches_callable_wrapper_reference():
    market_state = _market_state()
    spec = _callable_spec()

    generic = present_value_fixed_coupon_bond(
        market_state,
        spec,
        settlement=market_state.settlement,
    )
    wrapper = straight_bond_present_value(
        market_state,
        spec,
        settlement=market_state.settlement,
    )

    assert generic == pytest.approx(wrapper, rel=1e-12)
