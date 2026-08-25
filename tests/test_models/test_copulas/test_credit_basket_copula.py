from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve


SETTLE = date(2024, 11, 15)


class _CDOSpec:
    notional = 100_000_000.0
    n_names = 100
    attachment = 0.03
    detachment = 0.07
    end_date = date(2029, 11, 15)
    correlation = 0.3
    recovery = 0.4


class _NthSpec:
    notional = 10_000_000.0
    n_names = 5
    n_th = 1
    end_date = date(2029, 11, 15)
    correlation = 0.3
    recovery = 0.4


class _ZeroRecoveryNthSpec(_NthSpec):
    recovery = 0.0


class _WeightedNthSpec(_NthSpec):
    n_names = 4
    n_th = 2
    basket_names = ("A", "B", "C", "D")
    basket_weights = (0.4, 0.2, 0.2, 0.2)
    spread = 0.025


class _DuplicateNameWeightedNthSpec(_WeightedNthSpec):
    basket_names = ("A", "A", "C", "D")


class _MismatchedWeightWeightedNthSpec(_WeightedNthSpec):
    basket_weights = (0.4, 0.3, 0.3)


class _UnnormalizedWeightedNthSpec(_WeightedNthSpec):
    basket_weights = (0.4, 0.2, 0.2, 0.1)


class _LossDistributionSpec:
    notional = 100_000_000.0
    n_names = 75
    end_date = date(2029, 11, 15)
    correlation = 0.28
    recovery = 0.4


def _market_state(*, hazard: float = 0.02, rate: float = 0.04) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(rate, max_tenor=10.0),
        credit_curve=CreditCurve.flat(hazard, max_tenor=10.0),
    )


def test_price_credit_basket_tranche_result_supports_gaussian_and_student_t():
    from trellis.models.credit_basket_copula import price_credit_basket_tranche_result

    market_state = _market_state()
    gaussian = price_credit_basket_tranche_result(
        market_state,
        _CDOSpec(),
        copula_family="gaussian",
    )
    student_t = price_credit_basket_tranche_result(
        market_state,
        _CDOSpec(),
        copula_family="student_t",
        n_paths=20_000,
        seed=42,
    )

    assert gaussian.price > 0.0
    assert gaussian.expected_loss_fraction > 0.0
    assert gaussian.fair_spread_bp > 0.0
    assert student_t.price > 0.0
    assert student_t.expected_loss_fraction > 0.0
    assert student_t.fair_spread_bp > 0.0
    assert student_t.expected_loss_fraction != pytest.approx(gaussian.expected_loss_fraction)


def test_price_credit_basket_nth_to_default_preserves_compatibility():
    from trellis.models.credit_basket_copula import price_credit_basket_nth_to_default
    from trellis.instruments.nth_to_default import price_nth_to_default_basket

    market_state = _market_state(hazard=0.03)
    spec = _NthSpec()

    helper_price = price_credit_basket_nth_to_default(
        market_state,
        spec,
        copula_family="gaussian",
    )
    reference_price = price_nth_to_default_basket(
        notional=spec.notional,
        n_names=spec.n_names,
        n_th=spec.n_th,
        horizon=5.0,
        correlation=spec.correlation,
        recovery=spec.recovery,
        credit_curve=market_state.credit_curve,
        discount_curve=market_state.discount,
    )

    assert helper_price == pytest.approx(reference_price)


def test_resolve_credit_basket_inputs_preserves_explicit_zero_recovery():
    from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

    resolved = resolve_credit_basket_inputs(
        _market_state(hazard=0.03),
        _ZeroRecoveryNthSpec(),
    )

    assert resolved.recovery == 0.0


@pytest.mark.parametrize("recovery", (-0.1, 1.0, float("nan"), float("inf")))
def test_resolve_credit_basket_inputs_rejects_invalid_curve_quoted_recovery(
    recovery,
):
    from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

    spec_type = type("InvalidRecoveryNthSpec", (_NthSpec,), {"recovery": recovery})

    with pytest.raises(ValueError, match=r"recovery.*\[0, 1\)"):
        resolve_credit_basket_inputs(_market_state(), spec_type())


def test_resolve_credit_basket_inputs_preserves_weight_and_decimal_spread_contract():
    from math import exp

    from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

    resolved = resolve_credit_basket_inputs(_market_state(hazard=0.01), _WeightedNthSpec())
    bumped = resolve_credit_basket_inputs(
        _market_state(hazard=0.01),
        _WeightedNthSpec(),
        credit_spread_shift=1.0e-4,
    )

    assert resolved.reference_names == ("A", "B", "C", "D")
    assert resolved.exposure_weights == pytest.approx((0.4, 0.2, 0.2, 0.2))
    assert resolved.credit_spread == pytest.approx(0.025)
    assert resolved.hazard_rate == pytest.approx(0.025 / 0.6)
    assert resolved.survival_probability == pytest.approx(exp(-(0.025 / 0.6) * 5.0))
    assert bumped.credit_spread == pytest.approx(0.0251)
    assert bumped.hazard_rate == pytest.approx(0.0251 / 0.6)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (_DuplicateNameWeightedNthSpec(), "unique"),
        (_MismatchedWeightWeightedNthSpec(), "same length"),
        (_UnnormalizedWeightedNthSpec(), "sum to 1"),
    ],
)
def test_resolve_credit_basket_inputs_rejects_invalid_weight_contract(spec, message):
    from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

    with pytest.raises(ValueError, match=message):
        resolve_credit_basket_inputs(_market_state(), spec)


def test_credit_loss_distribution_helpers_agree_on_discounted_expected_loss():
    from trellis.models.credit_basket_copula import (
        price_credit_portfolio_loss_distribution_monte_carlo,
        price_credit_portfolio_loss_distribution_recursive,
        price_credit_portfolio_loss_distribution_transform_proxy,
    )

    market_state = _market_state(hazard=0.025)
    spec = _LossDistributionSpec()

    recursive_price = price_credit_portfolio_loss_distribution_recursive(
        market_state,
        spec,
    )
    transform_price = price_credit_portfolio_loss_distribution_transform_proxy(
        market_state,
        spec,
    )
    mc_price = price_credit_portfolio_loss_distribution_monte_carlo(
        market_state,
        spec,
        n_paths=30_000,
        seed=42,
    )

    assert recursive_price > 0.0
    assert transform_price > 0.0
    assert mc_price > 0.0
    assert transform_price == pytest.approx(recursive_price, rel=1e-10)
    assert mc_price == pytest.approx(recursive_price, rel=0.08)
