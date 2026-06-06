from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from trellis.agent.semantic_observables import (
    AndPredicate,
    BetweenPredicate,
    CmsRateObservable,
    GreaterThanPredicate,
    LessThanPredicate,
    ObservationMetadata,
    PredicateGrammarValidationError,
    RateIndexObservable,
    SpreadObservable,
    predicate_support_blockers,
    validate_conditional_accrual_predicate,
)
from trellis.agent.static_leg_contract import (
    ConditionalAccrualLeg,
    ConditionalAccrualPeriod,
    FixedCouponFormula,
    NotionalSchedule,
    NotionalStep,
    StaticLegIRWellFormednessError,
)


def _notional() -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(
                start_date=date(2026, 1, 15),
                end_date=date(2026, 7, 15),
                amount=1_000_000.0,
            ),
        )
    )


def _periods() -> tuple[ConditionalAccrualPeriod, ...]:
    return (
        ConditionalAccrualPeriod(
            accrual_start=date(2026, 1, 15),
            accrual_end=date(2026, 4, 15),
            observation_date=date(2026, 4, 15),
            payment_date=date(2026, 4, 17),
            fixing_date=date(2026, 4, 15),
        ),
    )


def _sofr_observable() -> RateIndexObservable:
    return RateIndexObservable(
        observable_id="sofr_3m",
        index_name="SOFR",
        tenor="3M",
        observation=ObservationMetadata(
            schedule_role="observation_dates",
            fixing_date_role="fixing_dates",
            missing_fixing_policy="project_forward_for_future_only",
        ),
    )


def _ff_observable() -> RateIndexObservable:
    return RateIndexObservable(
        observable_id="ff_3m",
        index_name="FF",
        tenor="3M",
    )


def test_single_index_rate_between_predicate_is_frozen_and_admitted():
    predicate = BetweenPredicate(
        observable=_sofr_observable(),
        lower_bound=0.015,
        upper_bound=0.0325,
    )

    assert validate_conditional_accrual_predicate(predicate) == ()
    assert predicate_support_blockers(predicate) == ()
    assert predicate.observable.observable_family == "rate_index"

    leg = ConditionalAccrualLeg(
        currency="USD",
        notional_schedule=_notional(),
        accrual_periods=_periods(),
        coupon_formula=FixedCouponFormula(0.0525),
        day_count="ACT/365",
        payment_frequency="quarterly",
        accrual_condition_ref="sofr_in_range",
        accrual_counter_ref="in_range_coupon_count",
        accrual_condition=predicate,
    )

    assert leg.accrual_condition is predicate
    with pytest.raises(FrozenInstanceError):
        predicate.lower_bound = 0.01


def test_predicate_grammar_rejects_invalid_bounds_and_missing_fixing_policy():
    with pytest.raises(PredicateGrammarValidationError, match="lower_bound <= upper_bound"):
        BetweenPredicate(
            observable=_sofr_observable(),
            lower_bound=0.04,
            upper_bound=0.03,
        )

    with pytest.raises(PredicateGrammarValidationError, match="missing_fixing_policy"):
        ObservationMetadata(missing_fixing_policy="guess")


def test_boolean_composition_requires_predicates_and_preserves_order():
    lower = GreaterThanPredicate(observable=_sofr_observable(), threshold=0.015)
    upper = BetweenPredicate(
        observable=_sofr_observable(),
        lower_bound=0.0,
        upper_bound=0.0325,
    )
    predicate = AndPredicate((lower, upper))

    assert predicate.predicates == (lower, upper)
    assert validate_conditional_accrual_predicate(predicate) == ()

    with pytest.raises(PredicateGrammarValidationError, match="must be non-empty"):
        AndPredicate(())


def test_multi_index_rate_predicate_fails_closed():
    predicate = AndPredicate(
        (
            GreaterThanPredicate(observable=_sofr_observable(), threshold=0.015),
            LessThanPredicate(observable=_ff_observable(), threshold=0.0325),
        )
    )

    blockers = predicate_support_blockers(predicate)
    assert tuple(blocker.observable_family for blocker in blockers) == ("multi_index",)
    assert blockers[0].blocker_id == "conditional_accrual_multi_index_predicate_pending"

    with pytest.raises(PredicateGrammarValidationError, match="multi_index"):
        validate_conditional_accrual_predicate(predicate)

    with pytest.raises(StaticLegIRWellFormednessError, match="multi_index"):
        ConditionalAccrualLeg(
            currency="USD",
            notional_schedule=_notional(),
            accrual_periods=_periods(),
            coupon_formula=FixedCouponFormula(0.0525),
            day_count="ACT/365",
            payment_frequency="quarterly",
            accrual_condition_ref="mixed_rate_index_in_range",
            accrual_counter_ref="in_range_coupon_count",
            accrual_condition=predicate,
        )


def test_conditional_accrual_leg_preserves_positional_defaults():
    leg = ConditionalAccrualLeg(
        "USD",
        _notional(),
        _periods(),
        FixedCouponFormula(0.0525),
        "ACT/365",
        "quarterly",
        "sofr_in_range",
        "in_range_coupon_count",
        "coupon_period_cash_settlement",
        "legacy_label",
        {"semantic_family": "range_accrual"},
    )

    assert leg.settlement_rule == "coupon_period_cash_settlement"
    assert leg.label == "legacy_label"
    assert leg.metadata["semantic_family"] == "range_accrual"
    assert leg.accrual_condition is None


def test_cms_spread_predicate_emits_blockers_and_fails_closed_for_leg():
    predicate = BetweenPredicate(
        observable=SpreadObservable(
            left=CmsRateObservable(
                observable_id="usd_cms_10y",
                curve_id="USD_SWAP",
                tenor="10Y",
            ),
            right=RateIndexObservable(
                observable_id="sofr_3m",
                index_name="SOFR",
                tenor="3M",
            ),
            spread_id="cms_minus_sofr",
        ),
        lower_bound=0.0,
        upper_bound=0.025,
    )

    blockers = predicate_support_blockers(predicate)
    assert tuple(blocker.observable_family for blocker in blockers) == (
        "spread",
        "cms_rate",
    )
    assert all(blocker.blocker_id for blocker in blockers)
    with pytest.raises(
        PredicateGrammarValidationError,
        match="unsupported observable family",
    ):
        validate_conditional_accrual_predicate(predicate)

    with pytest.raises(
        StaticLegIRWellFormednessError,
        match="unsupported observable family",
    ):
        ConditionalAccrualLeg(
            currency="USD",
            notional_schedule=_notional(),
            accrual_periods=_periods(),
            coupon_formula=FixedCouponFormula(0.0525),
            day_count="ACT/365",
            payment_frequency="quarterly",
            accrual_condition_ref="cms_spread_in_range",
            accrual_counter_ref="in_range_coupon_count",
            accrual_condition=predicate,
        )
