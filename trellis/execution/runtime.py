"""Runtime visitors for bounded execution IR artifacts."""

from __future__ import annotations

from datetime import date
from types import MappingProxyType
from typing import Mapping

from trellis.conventions.day_count import DayCountConvention
from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.execution.ir import (
    ConditionalAccrualLegExecution,
    ContractExecutionIR,
    CouponLegExecution,
    KnownCashflowObligation,
    PeriodRateOptionStripExecution,
)


def price_dynamic_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    method: str | None = None,
    terms: Mapping[str, object] | None = None,
) -> float:
    """Price one bounded dynamic execution IR through its admitted engine lane."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    if ir.source_track.source_kind != "dynamic_contract_ir":
        raise ValueError("price_dynamic_execution_ir requires a dynamic execution IR")
    if ir.execution_metadata.unsupported_reasons:
        raise ValueError(
            "Cannot price unsupported execution IR: "
            f"{ir.execution_metadata.unsupported_reasons!r}"
        )

    if ir.source_track.product_family == "callable_bond":
        return _price_callable_bond_execution_ir(
            ir,
            market_state,
            method=method,
            terms=terms,
        )
    if ir.source_track.product_family == "callable_range_accrual":
        return _price_callable_range_accrual_execution_ir(
            ir,
            market_state,
            method=method,
        )
    raise ValueError(
        "Unsupported dynamic execution product family "
        f"{ir.source_track.product_family!r}"
    )


def price_static_leg_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    method: str | None = None,
    terms: Mapping[str, object] | None = None,
) -> float:
    """Price a bounded static-leg execution IR from its lowered obligations."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    if ir.source_track.source_kind != "static_leg_contract_ir":
        raise ValueError("price_static_leg_execution_ir requires a static-leg execution IR")
    if ir.execution_metadata.unsupported_reasons:
        raise ValueError(
            "Cannot price unsupported execution IR: "
            f"{ir.execution_metadata.unsupported_reasons!r}"
        )

    total = 0.0
    for obligation in ir.obligations:
        if isinstance(obligation, KnownCashflowObligation):
            total += _price_known_cashflow(obligation, market_state, ir)
        elif isinstance(obligation, ConditionalAccrualLegExecution):
            total += _price_conditional_accrual_leg(obligation, market_state)
        elif isinstance(obligation, CouponLegExecution):
            total += _price_coupon_leg(obligation, market_state, ir)
        elif isinstance(obligation, PeriodRateOptionStripExecution):
            total += _price_period_rate_option_strip(
                obligation,
                market_state,
                method=method or _source_metadata(ir).get("requested_method"),
                terms=terms,
            )
        else:
            raise ValueError(
                f"Unsupported static-leg execution obligation {type(obligation).__name__}"
            )
    return float(total)


def _price_conditional_accrual_leg(
    obligation: ConditionalAccrualLegExecution,
    market_state: MarketState,
) -> float:
    from trellis.models.range_accrual import price_range_accrual

    metadata = _metadata_dict(obligation.metadata)
    reference_index = _conditional_accrual_reference_index(metadata)
    spec = _range_accrual_spec_from_metadata(metadata)
    result = price_range_accrual(
        spec,
        as_of=market_state.settlement,
        discount_curve=_required_discount_curve(market_state),
        forecast_curve=_range_accrual_forecast_curve(
            market_state,
            reference_index=reference_index,
        ),
        fixing_history=_range_accrual_fixing_history(
            market_state,
            reference_index=reference_index,
        ),
        scenario_shifts_bps=(),
    )
    return _direction_sign(str(metadata.get("direction") or "receive")) * float(result.price)


def _price_callable_range_accrual_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    method: str | None = None,
) -> float:
    from trellis.models.range_accrual import project_range_accrual_cashflows

    normalized_method = str(method or "deterministic").strip().lower().replace("-", "_")
    if normalized_method not in {"deterministic", "deterministic_call", "analytical"}:
        raise ValueError(
            "callable range-accrual dynamic execution currently admits only "
            f"deterministic call-decision pricing; method={method!r}"
        )
    obligation = _single_conditional_accrual_obligation(ir)
    metadata = _metadata_dict(obligation.metadata)
    reference_index = _conditional_accrual_reference_index(metadata)
    spec = _range_accrual_spec_from_metadata(metadata)
    discount_curve = _required_discount_curve(market_state)
    cashflows = project_range_accrual_cashflows(
        spec,
        as_of=market_state.settlement,
        forecast_curve=_range_accrual_forecast_curve(
            market_state,
            reference_index=reference_index,
        ),
        fixing_history=_range_accrual_fixing_history(
            market_state,
            reference_index=reference_index,
        ),
    )
    no_call_price = _discount_projected_cashflows(
        cashflows,
        as_of=market_state.settlement,
        discount_curve=discount_curve,
        day_count=spec.day_count,
    )
    source_metadata = _source_metadata(ir)
    call_price_cash = float(source_metadata.get("call_price_cash", spec.notional))
    call_path_prices = tuple(
        _discount_projected_cashflows(
            tuple(cashflow for cashflow in cashflows if cashflow.payment_date <= call_date),
            as_of=market_state.settlement,
            discount_curve=discount_curve,
            day_count=spec.day_count,
        )
        + call_price_cash
        * _discount_factor_from_curve(
            discount_curve,
            market_state.settlement,
            call_date,
            day_count=spec.day_count,
        )
        for call_date in tuple(source_metadata.get("call_dates") or ())
        if market_state.settlement < call_date < spec.payment_dates[-1]
    )
    candidates = (no_call_price, *call_path_prices)
    return float(min(candidates))


def _price_callable_bond_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    method: str | None = None,
    terms: Mapping[str, object] | None = None,
) -> float:
    from trellis.execution.visitors.event_compile import (
        compile_callable_bond_spec_from_execution_ir,
    )
    from trellis.models.callable_bond_pde import price_callable_bond_pde
    from trellis.models.callable_bond_tree import price_callable_bond_tree

    spec = compile_callable_bond_spec_from_execution_ir(ir)
    normalized_terms = dict(terms or {})
    normalized_method = _normalized_dynamic_method(method)
    if normalized_method == "lattice":
        return float(
            price_callable_bond_tree(
                market_state,
                spec,
                model=str(normalized_terms.get("model") or "hull_white"),
                mean_reversion=_optional_float(normalized_terms.get("mean_reversion")),
                sigma=_optional_float(normalized_terms.get("sigma")),
                n_steps=_optional_int(normalized_terms.get("n_steps")),
            )
        )
    if normalized_method == "pde":
        return float(
            price_callable_bond_pde(
                market_state,
                spec,
                mean_reversion=_optional_float(normalized_terms.get("mean_reversion")),
                sigma=_optional_float(normalized_terms.get("sigma")),
                theta=(
                    0.5
                    if normalized_terms.get("theta") is None
                    else float(normalized_terms["theta"])
                ),
                n_r=_optional_int(normalized_terms.get("n_r")),
                n_t=_optional_int(normalized_terms.get("n_t")),
                r_min=_optional_float(normalized_terms.get("r_min")),
                r_max=_optional_float(normalized_terms.get("r_max")),
            )
        )
    raise ValueError(
        "callable-bond dynamic execution currently admits only lattice and pde methods; "
        f"method={method!r}"
    )


def _price_known_cashflow(
    obligation: KnownCashflowObligation,
    market_state: MarketState,
    ir: ContractExecutionIR,
) -> float:
    if obligation.payment_date <= market_state.settlement:
        return 0.0
    sign = 1.0 if obligation.receiver == "holder" else -1.0
    day_count = _discount_day_count(
        ir,
        fallback=_contract_coupon_day_count(ir) or DayCountConvention.ACT_365,
    )
    return sign * obligation.amount * _discount_factor(
        market_state,
        obligation.payment_date,
        day_count=day_count,
    )


def _price_coupon_leg(
    obligation: CouponLegExecution,
    market_state: MarketState,
    ir: ContractExecutionIR,
) -> float:
    metadata = _metadata_dict(obligation.metadata)
    direction = str(metadata.get("direction") or "receive")
    sign = 1.0 if direction == "receive" else -1.0
    notional = float(metadata["notional"])
    formula_kind = str(metadata.get("formula_kind") or "")
    day_count = _day_count(metadata.get("day_count"))
    periods = tuple(metadata.get("periods") or ())
    total = 0.0
    for period in periods:
        accrual_start, accrual_end, payment_date, fixing_date = period
        if payment_date <= market_state.settlement:
            continue
        if (
            ir.source_track.product_family == "fixed_coupon_bond"
            and formula_kind == "fixed"
        ):
            accrual = 1.0 / _frequency_per_year(metadata.get("payment_frequency"))
        else:
            accrual = year_fraction(accrual_start, accrual_end, day_count)

        if formula_kind == "fixed":
            rate = float(metadata["fixed_rate"])
        elif formula_kind == "floating":
            rate_index = str(metadata.get("rate_index") or "")
            if fixing_date is not None and fixing_date < market_state.settlement:
                forward = _required_historical_fixing(
                    market_state,
                    rate_index=rate_index,
                    fixing_date=fixing_date,
                )
            else:
                forward_curve = market_state.forecast_forward_curve(rate_index or None)
                time_based_forward_day_count = (
                    DayCountConvention.ACT_365
                    if ir.source_track.product_family == "basis_swap"
                    else day_count
                )
                forward_rate_dates = getattr(forward_curve, "forward_rate_dates", None)
                if callable(forward_rate_dates):
                    try:
                        # Date-aware checked rate helpers use the leg accrual day count.
                        # The basis-swap ACT/365 override matches their time-based fallback.
                        forward = float(
                            forward_rate_dates(
                                accrual_start,
                                accrual_end,
                                day_count=day_count,
                            )
                        )
                    except AttributeError:
                        forward = _forward_rate_by_time(
                            forward_curve,
                            market_state,
                            accrual_start,
                            accrual_end,
                            day_count=time_based_forward_day_count,
                        )
                else:
                    forward = _forward_rate_by_time(
                        forward_curve,
                        market_state,
                        accrual_start,
                        accrual_end,
                        day_count=time_based_forward_day_count,
                    )
            rate = float(metadata.get("gearing", 1.0)) * forward + float(
                metadata.get("spread", 0.0)
            )
        else:
            raise ValueError(f"Unsupported coupon formula kind {formula_kind!r}")

        discount_day_count = _discount_day_count(ir, fallback=day_count)
        total += (
            sign
            * notional
            * rate
            * accrual
            * _discount_factor(
                market_state,
                payment_date,
                day_count=discount_day_count,
            )
        )
    return float(total)


def _required_historical_fixing(
    market_state: MarketState,
    *,
    rate_index: str,
    fixing_date: date,
) -> float:
    histories = market_state.fixing_histories or {}
    keys = tuple(dict.fromkeys((rate_index, *_rate_index_lookup_keys(rate_index))))
    for key in keys:
        history = histories.get(key)
        if history is not None and fixing_date in history:
            return float(history[fixing_date])
    raise ValueError(
        "Floating coupon requires historical fixing "
        f"{rate_index!r} on {fixing_date.isoformat()}."
    )


def _price_period_rate_option_strip(
    obligation: PeriodRateOptionStripExecution,
    market_state: MarketState,
    *,
    method: object,
    terms: Mapping[str, object] | None = None,
) -> float:
    from trellis.models.rate_cap_floor import (
        CapFloorPeriod,
        price_rate_cap_floor_strip_analytical,
        price_rate_cap_floor_strip_monte_carlo,
    )

    metadata = _metadata_dict(obligation.metadata)
    periods = tuple(
        CapFloorPeriod(
            start_date=period[0],
            end_date=period[1],
            fixing_date=period[2],
            payment_date=period[3],
        )
        for period in tuple(metadata.get("periods") or ())
    )
    kwargs = {
        "instrument_class": "cap" if obligation.option_style == "call" else "floor",
        "periods": periods,
        "notional": float(metadata["notional"]),
        "strike": float(metadata["strike"]),
        "start_date": periods[0].start_date if periods else None,
        "end_date": periods[-1].end_date if periods else None,
        "frequency": _frequency_enum(metadata.get("payment_frequency")),
        "day_count": _day_count(metadata.get("day_count")),
        "rate_index": str(metadata.get("rate_index") or ""),
    }
    normalized_terms = dict(terms or {})
    for key in ("calendar_name", "business_day_adjustment"):
        if normalized_terms.get(key) is not None:
            kwargs[key] = normalized_terms[key]
    if normalized_terms.get("reconstruct_from_spec_schedule"):
        kwargs.pop("periods", None)
    normalized_method = str(method or "analytical").strip().lower().replace("-", "_")
    if normalized_method == "monte_carlo":
        for key in (
            "n_paths",
            "seed",
            "n_steps",
            "mean_reversion",
            "sigma",
            "discount_curve",
            "forward_curve",
            "vol",
        ):
            if normalized_terms.get(key) is not None:
                kwargs[key] = normalized_terms[key]
        price = price_rate_cap_floor_strip_monte_carlo(market_state, **kwargs)
    else:
        for key in ("model", "shift", "sabr"):
            if normalized_terms.get(key) is not None:
                kwargs[key] = normalized_terms[key]
        price = price_rate_cap_floor_strip_analytical(market_state, **kwargs)
    return _direction_sign(str(metadata.get("direction") or "receive")) * float(price)


def _forward_rate_by_time(
    forward_curve,
    market_state: MarketState,
    accrual_start: date,
    accrual_end: date,
    *,
    day_count: DayCountConvention = DayCountConvention.ACT_365,
) -> float:
    start_years = max(
        year_fraction(market_state.settlement, accrual_start, day_count),
        0.0,
    )
    end_years = max(
        year_fraction(market_state.settlement, accrual_end, day_count),
        start_years + 1e-6,
    )
    return float(forward_curve.forward_rate(start_years, end_years))


def _single_conditional_accrual_obligation(
    ir: ContractExecutionIR,
) -> ConditionalAccrualLegExecution:
    matches = tuple(
        obligation
        for obligation in ir.obligations
        if isinstance(obligation, ConditionalAccrualLegExecution)
    )
    if len(matches) != 1:
        raise ValueError(
            "callable range-accrual execution requires one conditional-accrual obligation"
        )
    return matches[0]


def _range_accrual_spec_from_metadata(metadata: Mapping[str, object]):
    from trellis.models.range_accrual import RangeAccrualSpec

    periods = tuple(metadata.get("periods") or ())
    return RangeAccrualSpec(
        reference_index=_conditional_accrual_reference_index(metadata),
        notional=float(metadata["notional"]),
        coupon_rate=float(metadata["coupon_rate"]),
        lower_bound=float(metadata["lower_bound"]),
        upper_bound=float(metadata["upper_bound"]),
        observation_dates=tuple(period[2] for period in periods),
        accrual_start_dates=tuple(period[0] for period in periods),
        payment_dates=tuple(period[4] for period in periods),
        principal_redemption=float(metadata.get("principal_redemption", 0.0)),
        inclusive_lower=bool(metadata.get("inclusive_lower", True)),
        inclusive_upper=bool(metadata.get("inclusive_upper", True)),
        day_count=_day_count(metadata.get("day_count")),
    )


def _conditional_accrual_reference_index(metadata: Mapping[str, object]) -> str:
    return str(metadata.get("reference_index") or "").strip()


def _discount_projected_cashflows(
    cashflows,
    *,
    as_of: date,
    discount_curve,
    day_count: DayCountConvention,
) -> float:
    return float(
        sum(
            cashflow.amount
            * _discount_factor_from_curve(
                discount_curve,
                as_of,
                cashflow.payment_date,
                day_count=day_count,
            )
            for cashflow in cashflows
            if cashflow.payment_date > as_of
        )
    )


def _discount_factor_from_curve(
    discount_curve,
    as_of: date,
    payment_date: date,
    *,
    day_count: DayCountConvention,
) -> float:
    discount_date = getattr(discount_curve, "discount_date", None)
    if callable(discount_date):
        return float(discount_date(payment_date))
    t = max(year_fraction(as_of, payment_date, day_count), 0.0)
    return float(discount_curve.discount(t))


def _required_discount_curve(market_state: MarketState):
    if market_state.discount is None:
        raise ValueError("execution pricing requires market_state.discount")
    return market_state.discount


def _range_accrual_forecast_curve(
    market_state: MarketState,
    *,
    reference_index: str,
):
    forecast_curves = market_state.forecast_curves or {}
    for key in _rate_index_lookup_keys(reference_index):
        if key in forecast_curves:
            return forecast_curves[key]
    if market_state.discount is not None:
        return market_state.discount
    raise ValueError("range-accrual execution pricing requires a forecast curve")


def _range_accrual_fixing_history(
    market_state: MarketState,
    *,
    reference_index: str,
):
    fixing_histories = market_state.fixing_histories or {}
    for key in _rate_index_lookup_keys(reference_index):
        if key in fixing_histories:
            return fixing_histories[key]
    return {}


def _rate_index_lookup_keys(reference_index: str) -> tuple[str, ...]:
    text = str(reference_index or "").strip().upper()
    if not text:
        return ()
    keys = [text]
    if "-" in text:
        keys.append(text.rsplit("-", 1)[0])
    return tuple(dict.fromkeys(keys))


def _discount_factor(
    market_state: MarketState,
    payment_date: date,
    *,
    day_count: DayCountConvention,
) -> float:
    discount_curve = market_state.discount
    if discount_curve is None:
        raise ValueError("static-leg execution pricing requires market_state.discount")
    discount_date = getattr(discount_curve, "discount_date", None)
    if callable(discount_date):
        return float(discount_date(payment_date))
    t = max(year_fraction(market_state.settlement, payment_date, day_count), 0.0)
    return float(discount_curve.discount(t))


def _discount_day_count(
    ir: ContractExecutionIR,
    *,
    fallback: DayCountConvention,
) -> DayCountConvention:
    if ir.source_track.product_family == "basis_swap":
        return DayCountConvention.ACT_365
    return fallback


def _contract_coupon_day_count(ir: ContractExecutionIR) -> DayCountConvention | None:
    for obligation in ir.obligations:
        if not isinstance(obligation, CouponLegExecution):
            continue
        metadata = _metadata_dict(obligation.metadata)
        if metadata.get("day_count"):
            return _day_count(metadata["day_count"])
    return None


def _day_count(value: object) -> DayCountConvention:
    text = str(value or "ACT/365").strip().upper().replace("_", "/")
    mapping = {
        "ACT/360": DayCountConvention.ACT_360,
        "ACT/365": DayCountConvention.ACT_365,
        "ACT/ACT": DayCountConvention.ACT_ACT,
        "30/360": DayCountConvention.THIRTY_360,
    }
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution day count {value!r}") from exc


def _frequency_per_year(value: object) -> int:
    text = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "annual": 1,
        "semiannual": 2,
        "semi_annual": 2,
        "quarterly": 4,
        "monthly": 12,
    }
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution payment frequency {value!r}") from exc


def _direction_sign(direction: str) -> float:
    return 1.0 if direction == "receive" else -1.0


def _frequency_enum(value: object) -> Frequency:
    text = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "annual": Frequency.ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "semi_annual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution payment frequency {value!r}") from exc


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalized_dynamic_method(method: object) -> str:
    text = str(method or "lattice").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"rate_tree", "tree", "lattice"}:
        return "lattice"
    if text in {"pde", "pde_solver", "event_aware_pde"}:
        return "pde"
    return text


def _metadata_dict(items: object) -> Mapping[str, object]:
    try:
        return MappingProxyType(dict(items or ()))
    except (TypeError, ValueError):
        return MappingProxyType({})


def _source_metadata(ir: ContractExecutionIR) -> Mapping[str, object]:
    try:
        return MappingProxyType(dict(ir.source_track.source_metadata or ()))
    except (TypeError, ValueError):
        return MappingProxyType({})


__all__ = ["price_dynamic_execution_ir", "price_static_leg_execution_ir"]
