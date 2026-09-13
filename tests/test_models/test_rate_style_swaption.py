from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.differentiable import gradient
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.data.resolver import resolve_market_snapshot
from trellis.models.rate_style_swaption import (
    ResolvedSwaptionBlack76Inputs,
    price_bermudan_swaption_black76_lower_bound,
    price_swaption_monte_carlo,
    price_swaption_black76_raw,
    price_swaption_black76,
    resolve_swaption_black76_inputs,
    resolve_swaption_curve_basis_spread,
    resolve_swaption_monte_carlo_problem,
)
from trellis.models.rate_style_swaption_tree import (
    build_swaption_tree_spec,
    price_swaption_tree,
)
from trellis.models.black import black76_put
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _EuropeanSpec:
    notional = 100.0
    strike = 0.05
    expiry_date = date(2029, 11, 15)
    swap_start = expiry_date
    swap_end = date(2034, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = None
    is_payer = True


class _EuropeanCurveSpec:
    notional = 1_000_000.0
    strike = 0.045
    expiry_date = date(2025, 11, 15)
    swap_start = expiry_date
    swap_end = date(2030, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = "USD-SOFR-3M"
    is_payer = True


class _DualLegEuropeanCurveSpec:
    """Bounded T73 economics with distinct coupon and model-time conventions."""

    notional = 1_000_000.0
    strike = 0.03
    expiry_date = date(2025, 11, 15)
    swap_start = expiry_date
    swap_end = date(2030, 11, 15)
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.THIRTY_360
    float_frequency = Frequency.QUARTERLY
    float_day_count = DayCountConvention.ACT_360
    model_time_day_count = DayCountConvention.THIRTY_360
    rate_index = "USD-SOFR-3M"
    is_payer = True


class _ForwardStartEuropeanCurveSpec:
    notional = 1_000_000.0
    strike = 0.045
    expiry_date = date(2025, 11, 15)
    swap_start = date(2026, 11, 15)
    swap_end = date(2030, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = "USD-SOFR-3M"
    is_payer = True


class _BermudanSpec:
    notional = 100.0
    strike = 0.05
    exercise_dates = (
        date(2025, 11, 15),
        date(2026, 11, 15),
        date(2027, 11, 15),
        date(2028, 11, 15),
        date(2029, 11, 15),
    )
    swap_end = date(2030, 11, 15)
    from trellis.core.types import DayCountConvention, Frequency
    swap_frequency = Frequency.SEMI_ANNUAL
    day_count = DayCountConvention.ACT_360
    rate_index = None
    is_payer = True


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(vol),
    )


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def _payment_timeline_with_model_times(
    *,
    start: date,
    end: date,
    frequency: Frequency,
    accrual_day_count: DayCountConvention,
    model_time_day_count: DayCountConvention,
):
    """Build an independent expected leg timeline for dual-convention tests."""
    return tuple(
        replace(
            period,
            t_start=year_fraction(SETTLE, period.start_date, model_time_day_count),
            t_end=year_fraction(SETTLE, period.end_date, model_time_day_count),
            t_payment=year_fraction(SETTLE, period.payment_date, model_time_day_count),
        )
        for period in build_payment_timeline(
            start,
            end,
            frequency,
            day_count=accrual_day_count,
        )
    )


def test_price_swaption_black76_is_positive():
    price = price_swaption_black76(_market_state(), _EuropeanSpec())
    assert price > 0.0


def test_price_swaption_black76_raw_matches_public_wrapper():
    market_state = _market_state()
    resolved = resolve_swaption_black76_inputs(market_state, _EuropeanSpec())

    assert price_swaption_black76_raw(resolved) == pytest.approx(
        price_swaption_black76(market_state, _EuropeanSpec()),
        abs=1e-12,
    )


def test_resolve_swaption_black76_inputs_respects_explicit_swap_start():
    market_state = _market_state()

    spot_start = resolve_swaption_black76_inputs(market_state, _EuropeanCurveSpec())
    forward_start = resolve_swaption_black76_inputs(
        market_state,
        _ForwardStartEuropeanCurveSpec(),
    )

    assert forward_start.payment_count < spot_start.payment_count
    assert forward_start.annuity < spot_start.annuity
    assert price_swaption_black76(market_state, _ForwardStartEuropeanCurveSpec()) != pytest.approx(
        price_swaption_black76(market_state, _EuropeanCurveSpec()),
        rel=1e-9,
        abs=1e-9,
    )


def test_resolve_swaption_black76_inputs_uses_dual_leg_conventions_on_one_model_clock():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )
    spec = _DualLegEuropeanCurveSpec()

    resolved = resolve_swaption_black76_inputs(market_state, spec)
    fixed_timeline = _payment_timeline_with_model_times(
        start=spec.swap_start,
        end=spec.swap_end,
        frequency=spec.swap_frequency,
        accrual_day_count=spec.day_count,
        model_time_day_count=spec.model_time_day_count,
    )
    floating_timeline = _payment_timeline_with_model_times(
        start=spec.swap_start,
        end=spec.swap_end,
        frequency=spec.float_frequency,
        accrual_day_count=spec.float_day_count,
        model_time_day_count=spec.model_time_day_count,
    )
    expected_annuity = sum(
        float(period.accrual_fraction)
        * float(market_state.discount.discount(float(period.t_payment)))
        for period in fixed_timeline
    )
    forecast_curve = market_state.forecast_curves[spec.rate_index]
    expected_float_pv = sum(
        (
            float(forecast_curve.discount(float(period.t_start)))
            / float(forecast_curve.discount(float(period.t_end)))
            - 1.0
        )
        * float(market_state.discount.discount(float(period.t_payment)))
        for period in floating_timeline
    )

    assert len(fixed_timeline) == 10
    assert len(floating_timeline) == 20
    assert fixed_timeline[0].accrual_fraction == pytest.approx(0.5)
    assert floating_timeline[0].accrual_fraction == pytest.approx(92.0 / 360.0)
    assert fixed_timeline[0].t_payment == pytest.approx(1.5)
    assert floating_timeline[0].t_payment == pytest.approx(1.25)
    assert resolved.expiry_years == pytest.approx(1.0)
    assert resolved.payment_count == len(fixed_timeline)
    assert resolved.annuity == pytest.approx(expected_annuity, rel=1e-13, abs=1e-13)
    assert resolved.forward_swap_rate == pytest.approx(
        expected_float_pv / expected_annuity,
        rel=1e-13,
        abs=1e-13,
    )


def test_dual_leg_same_curve_float_pv_telescopes_without_artificial_basis():
    curve = YieldCurve.flat(0.04, max_tenor=10.0)
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        forecast_curves={"USD-SOFR-3M": curve},
        vol_surface=FlatVol(0.20),
    )
    spec = _DualLegEuropeanCurveSpec()

    resolved = resolve_swaption_black76_inputs(market_state, spec)

    # With matching curves and payment at reset end, simple floating coupons
    # telescope regardless of the coupon's annualization convention.
    expected_float_pv = float(curve.discount(1.0) - curve.discount(6.0))
    assert resolved.forward_swap_rate * resolved.annuity == pytest.approx(
        expected_float_pv, rel=1e-13, abs=1e-13
    )
    assert resolve_swaption_curve_basis_spread(market_state, spec) == pytest.approx(
        0.0, abs=1e-13
    )


def test_resolve_swaption_black76_inputs_preserves_legacy_single_timeline_fallback():
    class _ExplicitLegacyEquivalent(_EuropeanCurveSpec):
        float_frequency = _EuropeanCurveSpec.swap_frequency
        float_day_count = _EuropeanCurveSpec.day_count
        model_time_day_count = _EuropeanCurveSpec.day_count

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )

    legacy = resolve_swaption_black76_inputs(market_state, _EuropeanCurveSpec())
    explicit = resolve_swaption_black76_inputs(market_state, _ExplicitLegacyEquivalent())

    assert legacy == explicit


def test_price_swaption_black76_raw_receiver_branch_matches_black76_put():
    resolved = resolve_swaption_black76_inputs(_market_state(), _EuropeanSpec())
    receiver = replace(resolved, is_payer=False)

    expected = (
        receiver.notional
        * receiver.annuity
        * black76_put(
            receiver.forward_swap_rate,
            receiver.strike,
            receiver.vol,
            receiver.expiry_years,
        )
    )

    assert price_swaption_black76_raw(receiver) == pytest.approx(expected, abs=1e-12)


def test_price_swaption_black76_raw_vega_matches_finite_difference():
    resolved = resolve_swaption_black76_inputs(_market_state(), _EuropeanSpec())

    autodiff_vega = gradient(
        lambda vol: price_swaption_black76_raw(replace(resolved, vol=vol))
    )(resolved.vol)
    fd_vega = _finite_difference(
        lambda vol: price_swaption_black76_raw(replace(resolved, vol=vol)),
        resolved.vol,
    )

    assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)


def test_price_swaption_black76_raw_returns_zero_when_payment_count_is_zero():
    resolved = ResolvedSwaptionBlack76Inputs(
        expiry_date=_EuropeanSpec.expiry_date,
        expiry_years=5.0,
        annuity=4.2,
        forward_swap_rate=0.051,
        strike=0.05,
        vol=0.20,
        notional=100.0,
        is_payer=True,
        payment_count=0,
    )

    assert price_swaption_black76_raw(resolved) == 0.0


def test_bermudan_lower_bound_uses_final_exercise_date():
    market_state = _market_state()
    lower_bound = price_bermudan_swaption_black76_lower_bound(market_state, _BermudanSpec())
    final_exercise = price_swaption_black76(
        market_state,
        _BermudanSpec(),
        expiry_date=date(2029, 11, 15),
    )

    assert lower_bound == final_exercise


def test_bermudan_lower_bound_increases_with_vol():
    low = price_bermudan_swaption_black76_lower_bound(_market_state(vol=0.10), _BermudanSpec())
    high = price_bermudan_swaption_black76_lower_bound(_market_state(vol=0.30), _BermudanSpec())
    assert high > low


def test_build_swaption_tree_spec_maps_single_exercise_surface():
    tree_spec = build_swaption_tree_spec(_EuropeanSpec())

    assert tree_spec.notional == pytest.approx(_EuropeanSpec.notional)
    assert tree_spec.strike == pytest.approx(_EuropeanSpec.strike)
    assert tree_spec.exercise_dates == (_EuropeanSpec.expiry_date,)
    assert tree_spec.swap_end == _EuropeanSpec.swap_end
    assert tree_spec.rate_index == _EuropeanSpec.rate_index
    assert tree_spec.is_payer is _EuropeanSpec.is_payer
    assert tree_spec.model_time_day_count is None

    dual_leg_tree_spec = build_swaption_tree_spec(_DualLegEuropeanCurveSpec())

    assert (
        dual_leg_tree_spec.model_time_day_count
        == DayCountConvention.THIRTY_360
    )


def test_price_swaption_tree_is_positive():
    price = price_swaption_tree(_market_state(), _EuropeanSpec(), model="hull_white")
    assert price > 0.0


def test_price_swaption_tree_executes_with_explicit_shared_model_clock():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )

    price = price_swaption_tree(
        market_state,
        _DualLegEuropeanCurveSpec(),
        model="hull_white",
        mean_reversion=0.05,
        sigma=0.01,
        n_steps=120,
    )

    assert price > 0.0


def test_price_swaption_monte_carlo_stays_close_to_black76_and_tree():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.042, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.046, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
        selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
    )
    spec = _EuropeanCurveSpec()

    mc_price = price_swaption_monte_carlo(
        market_state,
        spec,
        n_paths=12_000,
        seed=11,
        n_steps=64,
    )
    black76_price = price_swaption_black76(market_state, spec)
    tree_price = price_swaption_tree(market_state, spec, model="hull_white")

    assert mc_price > 0.0
    assert mc_price == pytest.approx(tree_price, rel=0.15)
    assert mc_price == pytest.approx(black76_price, rel=0.35)


def test_swaption_monte_carlo_problem_starts_payment_schedule_at_explicit_swap_start():
    class _DelayedStartSpec(_EuropeanSpec):
        swap_start = date(2030, 5, 15)

    problem = resolve_swaption_monte_carlo_problem(
        _market_state(),
        _DelayedStartSpec(),
        n_steps=64,
        mean_reversion=0.05,
        sigma=0.01,
    )

    assert problem.event_timeline is not None
    settlement_event = next(
        event
        for event in problem.event_timeline.events
        if event.name == "swaption_settlement"
    )
    assert settlement_event.payload["payment_times"][0] == pytest.approx(
        year_fraction(
            SETTLE,
            date(2030, 11, 15),
            _DelayedStartSpec.day_count,
        )
    )


def test_swaption_monte_carlo_payload_exposes_and_uses_separate_floating_timeline():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )
    spec = _DualLegEuropeanCurveSpec()
    problem = resolve_swaption_monte_carlo_problem(
        market_state,
        spec,
        n_steps=64,
        mean_reversion=0.05,
        sigma=0.01,
    )
    settlement_event = next(
        event
        for event in problem.event_timeline.events
        if event.name == "swaption_settlement"
    )
    payload = settlement_event.payload
    fixed_timeline = _payment_timeline_with_model_times(
        start=spec.swap_start,
        end=spec.swap_end,
        frequency=spec.swap_frequency,
        accrual_day_count=spec.day_count,
        model_time_day_count=spec.model_time_day_count,
    )
    floating_timeline = _payment_timeline_with_model_times(
        start=spec.swap_start,
        end=spec.swap_end,
        frequency=spec.float_frequency,
        accrual_day_count=spec.float_day_count,
        model_time_day_count=spec.model_time_day_count,
    )

    assert payload["payment_times"] == pytest.approx(
        tuple(float(period.t_payment) for period in fixed_timeline)
    )
    assert payload["accrual_fractions"] == pytest.approx(
        tuple(float(period.accrual_fraction) for period in fixed_timeline)
    )
    assert payload["floating_start_times"] == pytest.approx(
        tuple(float(period.t_start) for period in floating_timeline)
    )
    assert payload["floating_end_times"] == pytest.approx(
        tuple(float(period.t_end) for period in floating_timeline)
    )
    assert payload["floating_payment_times"] == pytest.approx(
        tuple(float(period.t_payment) for period in floating_timeline)
    )
    assert payload["floating_accrual_fractions"] == pytest.approx(
        tuple(float(period.accrual_fraction) for period in floating_timeline)
    )
    assert len(payload["payment_times"]) == 10
    assert len(payload["floating_payment_times"]) == 20

    discount_to_expiry = float(market_state.discount.discount(1.0))
    fixed_annuity = sum(
        float(period.accrual_fraction)
        * float(market_state.discount.discount(float(period.t_payment)))
        / discount_to_expiry
        for period in fixed_timeline
    )
    forecast_curve = market_state.forecast_curves[spec.rate_index]
    floating_leg = sum(
        (
            float(forecast_curve.discount(float(period.t_start)))
            / float(forecast_curve.discount(float(period.t_end)))
            - 1.0
        )
        * float(market_state.discount.discount(float(period.t_payment)))
        / discount_to_expiry
        for period in floating_timeline
    )
    discount_par_rate = (
        1.0
        - float(market_state.discount.discount(float(fixed_timeline[-1].t_payment)))
        / discount_to_expiry
    ) / fixed_annuity
    expected_curve_basis = floating_leg / fixed_annuity - discount_par_rate

    assert payload["curve_basis_spread"] == pytest.approx(
        expected_curve_basis,
        rel=1e-13,
        abs=1e-13,
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_swaption_monte_carlo_defaults_missing_start_to_expiry(explicit_none):
    spec_fields = {
        field: getattr(_DualLegEuropeanCurveSpec, field)
        for field in (
            "notional", "strike", "expiry_date", "swap_end", "swap_frequency",
            "day_count", "float_frequency", "float_day_count",
            "model_time_day_count", "rate_index", "is_payer",
        )
    }
    if explicit_none:
        spec_fields["swap_start"] = None
    spec = SimpleNamespace(**spec_fields)

    problem = resolve_swaption_monte_carlo_problem(
        _market_state(), spec, mean_reversion=0.05, sigma=0.01
    )
    settlement = next(
        event for event in problem.event_timeline.events
        if event.name == "swaption_settlement"
    )

    assert settlement.payload["exercise_time"] == pytest.approx(1.0)
    assert len(settlement.payload["payment_times"]) == 10
    assert settlement.payload["payment_times"][0] == pytest.approx(1.5)
    assert len(settlement.payload["floating_payment_times"]) == 20
    assert settlement.payload["floating_start_times"][0] == pytest.approx(1.0)
    assert settlement.payload["floating_payment_times"][0] == pytest.approx(1.25)
    assert settlement.payload["floating_accrual_fractions"][0] == pytest.approx(92 / 360)


def test_dual_leg_swaption_monte_carlo_is_exactly_repeatable_for_fixed_seed():
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0)},
        vol_surface=FlatVol(0.20),
    )
    controls = {
        "n_paths": 2_048,
        "n_steps": 64,
        "seed": 42,
        "mean_reversion": 0.05,
        "sigma": 0.01,
    }

    first = price_swaption_monte_carlo(
        market_state,
        _DualLegEuropeanCurveSpec(),
        **controls,
    )
    second = price_swaption_monte_carlo(
        market_state,
        _DualLegEuropeanCurveSpec(),
        **controls,
    )

    assert first == second


def test_price_swaption_black76_with_hull_white_comparison_vol_matches_tree_and_mc():
    snapshot = resolve_market_snapshot(as_of=SETTLE, source="mock")
    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        discount_curve="usd_ois",
        forecast_curve="USD-SOFR-3M",
        vol_surface="usd_rates_atm",
        fixing_history="USD-SOFR-3M",
    )

    class _TaskLikeSpec:
        notional = 1_000_000.0
        strike = 0.03
        expiry_date = date(2025, 11, 15)
        swap_start = expiry_date
        swap_end = date(2030, 11, 15)
        swap_frequency = _EuropeanSpec.swap_frequency
        day_count = DayCountConvention.THIRTY_360
        rate_index = "USD-SOFR-3M"
        is_payer = True

    spec = _TaskLikeSpec()
    black76_price = price_swaption_black76(
        market_state,
        spec,
        mean_reversion=0.05,
        sigma=0.01,
    )
    tree_price = price_swaption_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=0.05,
        sigma=0.01,
    )
    mc_price = price_swaption_monte_carlo(
        market_state,
        spec,
        mean_reversion=0.05,
        sigma=0.01,
        n_paths=10_000,
        seed=42,
    )

    assert black76_price > 0.0
    assert black76_price == pytest.approx(tree_price, rel=0.05)
    assert black76_price == pytest.approx(mc_price, rel=0.08)
