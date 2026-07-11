"""Shared benchmark-contract normalization for FinancePy and related task corpora."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
import re
from typing import Any

from trellis.agent.knowledge.methods import is_known_method, normalize_method
from trellis.agent.market_scenarios import MarketScenarioContract, market_scenario_contract_from_task
from trellis.conventions.calendar import (
    BRAZIL,
    SYDNEY,
    TARGET,
    TOKYO,
    TORONTO,
    UK_SETTLEMENT,
    US_SETTLEMENT,
    WEEKEND_ONLY,
    ZURICH,
    BusinessDayAdjustment,
    Calendar,
)
from trellis.core.date_utils import add_months
from trellis.core.types import DayCountConvention, Frequency
from trellis.execution import (
    BermudanBestOfBasketLatticeControls,
    BermudanBestOfBasketLatticeResult,
    BermudanBestOfBasketMCControls,
    BermudanBestOfBasketMCInputs,
    BermudanBestOfBasketMCResult,
    ContractExecutionIR,
    ExecutionCapabilityAdmission,
    admit_execution_capabilities,
    compile_bermudan_best_of_basket_execution_ir,
    price_bermudan_best_of_basket_lattice,
    price_bermudan_best_of_basket_monte_carlo,
)


_DIRECT_OVERRIDE_KEYS: tuple[str, ...] = (
    "notional",
    "spot",
    "strike",
    "strike_variance",
    "option_type",
    "payout_type",
    "barrier",
    "barrier_type",
    "cash_payoff",
    "global_floor",
    "global_cap",
    "local_floor",
    "local_cap",
    "running_coupon",
    "observations_per_year",
    "fixed_coupon",
    "recovery_rate",
    "lookback_type",
    "payoff",
    "style",
    "currency_pair",
    "fx_pair",
    "foreign_discount_key",
    "rebate",
    "call_strike",
    "put_strike",
    "outer_strike",
    "inner_strike",
    "outer_option_type",
    "inner_option_type",
    "running_extreme",
    "realized_variance",
    "cap_strike",
    "floor_strike",
    "call_price",
    "spread",
)

_FREQUENCY_MAP: dict[str, Frequency] = {
    "annual": Frequency.ANNUAL,
    "semi_annual": Frequency.SEMI_ANNUAL,
    "semiannual": Frequency.SEMI_ANNUAL,
    "semi-annual": Frequency.SEMI_ANNUAL,
    "quarterly": Frequency.QUARTERLY,
    "monthly": Frequency.MONTHLY,
}

_DAY_COUNT_MAP: dict[str, DayCountConvention] = {
    "act/360": DayCountConvention.ACT_360,
    "act_360": DayCountConvention.ACT_360,
    "act/365": DayCountConvention.ACT_365,
    "act_365": DayCountConvention.ACT_365,
    "30e/360": DayCountConvention.THIRTY_E_360,
    "30e_360": DayCountConvention.THIRTY_E_360,
    "30/360": DayCountConvention.THIRTY_360,
    "30_360": DayCountConvention.THIRTY_360,
}

_BUSINESS_DAY_ADJUSTMENT_MAP: dict[str, BusinessDayAdjustment] = {
    "unadjusted": BusinessDayAdjustment.UNADJUSTED,
    "following": BusinessDayAdjustment.FOLLOWING,
    "modified_following": BusinessDayAdjustment.MODIFIED_FOLLOWING,
    "modified-following": BusinessDayAdjustment.MODIFIED_FOLLOWING,
    "preceding": BusinessDayAdjustment.PRECEDING,
    "modified_preceding": BusinessDayAdjustment.MODIFIED_PRECEDING,
    "modified-preceding": BusinessDayAdjustment.MODIFIED_PRECEDING,
}

_CALENDAR_MAP: dict[str, Calendar] = {
    "weekend_only": WEEKEND_ONLY,
    "weekendonly": WEEKEND_ONLY,
    "us_settlement": US_SETTLEMENT,
    "ussettlement": US_SETTLEMENT,
    "uk_settlement": UK_SETTLEMENT,
    "uksettlement": UK_SETTLEMENT,
    "target": TARGET,
    "tokyo": TOKYO,
    "sydney": SYDNEY,
    "toronto": TORONTO,
    "zurich": ZURICH,
    "brazil": BRAZIL,
}

_BENCHMARK_PRODUCT_INSTRUMENT_TYPES: dict[str, str] = {
    "equity_vanilla": "european_option",
    "fx_vanilla": "european_option",
    "fx_barrier_option": "barrier_option",
    "swaption": "swaption",
    "cds": "cds",
    "barrier_option": "barrier_option",
    "digital_option": "digital_option",
    "lookback_option": "lookback_option",
    "chooser_option": "chooser_option",
    "compound_option": "compound_option",
    "cliquet_option": "cliquet_option",
    "variance_swap": "variance_swap",
    "rainbow_option": "basket_option",
}


def benchmark_contract(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized benchmark contract payload for one task."""
    raw = task.get("benchmark_contract")
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def _spec_override_contract(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return the declared contract payload used for runtime spec overrides."""
    contract = benchmark_contract(task)
    if contract:
        return contract
    raw = task.get("extension_contract")
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def canonical_benchmark_instrument_type(task: Mapping[str, Any]) -> str | None:
    """Return the canonical runtime instrument family for one benchmark-backed task."""
    contract = benchmark_contract(task)
    product = str(contract.get("product") or "").strip().lower()
    if product == "period_rate_option_strip":
        cap_floor = str(contract.get("cap_floor") or "").strip().lower()
        if cap_floor in {"cap", "floor"}:
            return cap_floor
        return "period_rate_option_strip"
    return _BENCHMARK_PRODUCT_INSTRUMENT_TYPES.get(product)


def benchmark_preferred_method(task: Mapping[str, Any]) -> str | None:
    """Return one normalized benchmark-preferred method when the task declares it."""
    construct = task.get("construct")
    if construct is None:
        return None
    raw_methods = construct if isinstance(construct, (list, tuple)) else [construct]
    normalized = [
        normalize_method(str(item))
        for item in raw_methods
        if str(item).strip()
    ]
    normalized = [method for method in normalized if is_known_method(method)]
    unique = list(dict.fromkeys(normalized))
    if len(unique) == 1:
        return unique[0]
    return None


def benchmark_request_description(
    task: Mapping[str, Any],
    *,
    root=None,
) -> str | None:
    """Render one deterministic contract-rich request description for a benchmark task."""
    contract = _spec_override_contract(task)
    if not contract:
        return None

    title = str(task.get("title") or "Benchmark pricing task").strip()
    product = str(contract.get("product") or "").strip().lower()
    lines = [f"Build a pricer for: {title}", ""]

    summary = _benchmark_summary_line(contract)
    if summary:
        lines.extend([summary, ""])

    detail_lines = _benchmark_detail_lines(
        contract,
        scenario_contract=market_scenario_contract_from_task(task, root=root) if root is not None else market_scenario_contract_from_task(task),
    )
    if detail_lines:
        lines.extend(detail_lines)
        lines.append("")

    preferred_method = benchmark_preferred_method(task)
    if preferred_method:
        lines.append(f"Preferred method family: {preferred_method}")
    financepy_binding_id = str(task.get("financepy_binding_id") or "").strip()
    if financepy_binding_id:
        lines.append(f"FinancePy binding: {financepy_binding_id}")
    if product:
        lines.append(f"Benchmark product: {product}")
    return "\n".join(line for line in lines if line is not None).strip()


def benchmark_spec_overrides(
    task: Mapping[str, Any],
    *,
    root=None,
) -> dict[str, Any]:
    """Return shared product-aware spec overrides implied by a declared task contract."""
    contract = _spec_override_contract(task)
    if not contract:
        return {}
    scenario_contract = (
        market_scenario_contract_from_task(task, root=root)
        if root is not None
        else market_scenario_contract_from_task(task)
    )
    benchmark_inputs = {} if scenario_contract is None else scenario_contract.financepy_inputs()
    overrides: dict[str, Any] = {}

    for key in _DIRECT_OVERRIDE_KEYS:
        value = contract.get(key)
        if value not in {None, ""}:
            overrides[key] = value

    if contract.get("dividend_rate") not in {None, ""}:
        overrides["dividend_yield"] = float(contract["dividend_rate"])

    for key in ("spots", "volatilities", "dividend_rates", "weights", "underliers", "constituents"):
        value = contract.get(key)
        if isinstance(value, (list, tuple)) and value:
            target = {
                "volatilities": "vols",
                "dividend_rates": "dividend_yields",
            }.get(key, key)
            overrides[target] = ",".join(str(item) for item in value)

    correlation = contract.get("correlation")
    if isinstance(correlation, (list, tuple)) and correlation:
        if isinstance(correlation[0], (list, tuple)):
            overrides["correlation"] = ";".join(
                ",".join(str(item) for item in row)
                for row in correlation
            )
        else:
            overrides["correlation"] = ",".join(str(item) for item in correlation)

    valuation_date = _valuation_date(contract, scenario_contract)
    expiry_years = contract.get("expiry_years")
    if expiry_years not in {None, ""}:
        overrides["expiry_date"] = valuation_date + timedelta(days=round(float(expiry_years) * 365.0))

    for contract_key, spec_key in (
        ("exercise_date", "exercise_date"),
        ("maturity_date", "maturity_date"),
        ("start_date", "start_date"),
        ("step_in_date", "step_in_date"),
        ("settle_date", "settle_date"),
    ):
        parsed = _parse_date(contract.get(contract_key))
        if parsed is not None:
            overrides[spec_key] = parsed

    for contract_key, spec_key in (
        ("callable_dates", "exercise_dates"),
        ("exercise_dates", "exercise_dates"),
        ("coupon_dates", "coupon_dates"),
        ("accrual_dates", "accrual_dates"),
    ):
        parsed_dates = _parse_date_sequence(contract.get(contract_key))
        if parsed_dates:
            overrides[spec_key] = parsed_dates

    product = str(contract.get("product") or "").strip().lower()
    if product in {"fx_vanilla", "fx_barrier_option"}:
        overrides.update(
            _fx_vanilla_overrides(
                contract,
                scenario_contract=scenario_contract,
                valuation_date=valuation_date,
            )
        )
        if product == "fx_barrier_option":
            overrides.update(_barrier_option_overrides(contract, valuation_date=valuation_date))
    elif product in {"period_rate_option_strip", "rate_cap_floor_collar"}:
        overrides.update(
            _rate_cap_floor_overrides(
                contract,
                scenario_contract=scenario_contract,
            )
        )
    elif product == "swaption":
        overrides.update(
            _swaption_overrides(
                contract,
                scenario_contract=scenario_contract,
            )
        )
    elif product == "barrier_option":
        overrides.update(_barrier_option_overrides(contract, valuation_date=valuation_date))
    elif product == "digital_option":
        overrides.update(_digital_option_overrides(contract, valuation_date=valuation_date))
    elif product == "lookback_option":
        overrides.update(_lookback_option_overrides(contract, valuation_date=valuation_date))
    elif product == "chooser_option":
        overrides.update(_chooser_option_overrides(contract, valuation_date=valuation_date))
    elif product == "compound_option":
        overrides.update(_compound_option_overrides(contract, valuation_date=valuation_date))
    elif product == "cds":
        overrides.update(_cds_overrides(contract, valuation_date=valuation_date))
    elif product == "cliquet_option":
        overrides.update(_cliquet_option_overrides(contract, valuation_date=valuation_date))
    elif product == "variance_swap":
        overrides.update(_variance_swap_overrides(contract, valuation_date=valuation_date))
    elif product == "rainbow_option":
        overrides.update(
            _rainbow_option_overrides(
                contract,
                valuation_date=valuation_date,
                scenario_contract=scenario_contract,
            )
        )
    elif product == "nth_to_default":
        overrides.update(_nth_to_default_overrides(contract, valuation_date=valuation_date))

    return {key: value for key, value in overrides.items() if value is not None}


def benchmark_contract_execution_ir(
    task: Mapping[str, Any],
    *,
    root=None,
) -> ContractExecutionIR:
    """Compile a supported benchmark task into route-free execution IR."""
    contract = _spec_override_contract(task)
    product = str(contract.get("product") or "").strip().lower()
    style = str(contract.get("style") or "").strip().lower()
    payoff = str(contract.get("payoff") or "").strip().lower()
    if (product, style, payoff) != ("rainbow_option", "bermudan", "best_of_call"):
        raise ValueError(
            "benchmark execution IR currently supports only the P001 "
            "Bermudan best-of rainbow proof shape"
        )

    overrides = benchmark_spec_overrides(task, root=root)
    underliers = _string_sequence(
        overrides.get("underliers") or overrides.get("constituents")
    )
    if not underliers:
        raise ValueError("benchmark execution IR requires named underliers")

    strike = overrides.get("strike")
    if strike is None:
        raise ValueError("benchmark execution IR requires a strike")

    return compile_bermudan_best_of_basket_execution_ir(
        semantic_id=str(task.get("id") or product),
        underliers=underliers,
        strike=float(strike),
        expiry_date=overrides.get("expiry_date"),
        observation_dates=tuple(overrides.get("observation_dates") or ()),
        exercise_dates=tuple(overrides.get("exercise_dates") or ()),
        notional=float(overrides.get("notional") or 1.0),
        currency=str(contract.get("currency") or "USD"),
        requested_outputs=_requested_outputs_from_task(task),
        validation_policy=str(task.get("validation_policy") or ""),
        source_ref=f"benchmark_task:{task.get('id') or product}",
    )


def benchmark_contract_execution_admission(
    task: Mapping[str, Any],
    *,
    method: str,
    root=None,
) -> ExecutionCapabilityAdmission:
    """Compile a supported benchmark task and admit one execution method."""
    return admit_execution_capabilities(
        benchmark_contract_execution_ir(task, root=root),
        method=method,
    )


def benchmark_contract_monte_carlo_execution(
    task: Mapping[str, Any],
    *,
    root=None,
    controls: BermudanBestOfBasketMCControls | None = None,
) -> BermudanBestOfBasketMCResult:
    """Price the supported benchmark execution IR through the MC visitor."""
    ir = benchmark_contract_execution_ir(task, root=root)
    scenario_contract = (
        market_scenario_contract_from_task(task, root=root)
        if root is not None
        else market_scenario_contract_from_task(task)
    )
    result = price_bermudan_best_of_basket_monte_carlo(
        ir,
        _benchmark_bermudan_basket_mc_inputs(
            task,
            root=root,
            scenario_contract=scenario_contract,
        ),
        controls=controls,
    )
    if scenario_contract is None:
        return result
    return result.with_provenance(
        {
            "scenario_id": scenario_contract.scenario_id,
            "scenario_digest": scenario_contract.scenario_digest,
        }
    )


def benchmark_contract_lattice_execution(
    task: Mapping[str, Any],
    *,
    root=None,
    controls: BermudanBestOfBasketLatticeControls | None = None,
) -> BermudanBestOfBasketLatticeResult:
    """Price the supported benchmark execution IR through the lattice visitor."""
    ir = benchmark_contract_execution_ir(task, root=root)
    scenario_contract = (
        market_scenario_contract_from_task(task, root=root)
        if root is not None
        else market_scenario_contract_from_task(task)
    )
    result = price_bermudan_best_of_basket_lattice(
        ir,
        _benchmark_bermudan_basket_mc_inputs(
            task,
            root=root,
            scenario_contract=scenario_contract,
        ),
        controls=controls,
    )
    if scenario_contract is None:
        return result
    return result.with_provenance(
        {
            "scenario_id": scenario_contract.scenario_id,
            "scenario_digest": scenario_contract.scenario_digest,
        }
    )


def _benchmark_bermudan_basket_mc_inputs(
    task: Mapping[str, Any],
    *,
    root=None,
    scenario_contract: MarketScenarioContract | None,
) -> BermudanBestOfBasketMCInputs:
    contract = _spec_override_contract(task)
    overrides = benchmark_spec_overrides(task, root=root)
    underliers = _string_sequence(
        overrides.get("underliers") or overrides.get("constituents")
    )
    if not underliers:
        raise ValueError("benchmark MC execution requires named underliers")

    valuation_date = _valuation_date(contract, scenario_contract)
    scenario_underliers = {
        underlier.name: underlier
        for underlier in (scenario_contract.underliers if scenario_contract is not None else ())
    }
    spots = _named_float_payload(
        underliers,
        overrides.get("spots"),
        scenario_values={
            name: scenario_underliers[name].spot
            for name in underliers
            if name in scenario_underliers
        },
        label="spots",
    )
    vols = _named_float_payload(
        underliers,
        overrides.get("vols"),
        scenario_values={
            name: scenario_underliers[name].volatility
            for name in underliers
            if name in scenario_underliers
            and scenario_underliers[name].volatility is not None
        },
        label="vols",
    )
    carries = _named_float_payload(
        underliers,
        overrides.get("dividend_yields"),
        scenario_values={
            name: scenario_underliers[name].carry_rate
            for name in underliers
            if name in scenario_underliers
        },
        label="dividend_yields",
        default=0.0,
    )
    correlation = _correlation_matrix_payload(
        overrides.get("correlation"),
        scenario_contract=scenario_contract,
        n_assets=len(underliers),
    )
    risk_free_rate = (
        scenario_contract.domestic_rate
        if scenario_contract is not None and scenario_contract.domestic_rate is not None
        else _float_or_none(overrides.get("risk_free_rate"))
    )
    return BermudanBestOfBasketMCInputs(
        valuation_date=valuation_date,
        spot_values=spots,
        volatilities=vols,
        carry_rates=carries,
        correlation_matrix=correlation,
        risk_free_rate=float(risk_free_rate or 0.0),
    )


def _named_float_payload(
    names: tuple[str, ...],
    value: object,
    *,
    scenario_values: Mapping[str, object],
    label: str,
    default: float | None = None,
) -> dict[str, float]:
    vector = _float_sequence(value)
    result: dict[str, float] = {}
    if vector:
        if len(vector) != len(names):
            raise ValueError(
                f"{label} vector length {len(vector)} does not match underlier count {len(names)}"
            )
        result.update(dict(zip(names, vector, strict=True)))
    for name in names:
        if name in result:
            continue
        scenario_value = scenario_values.get(name)
        if scenario_value is not None:
            result[name] = float(scenario_value)
        elif default is not None:
            result[name] = float(default)
    missing = tuple(name for name in names if name not in result)
    if missing:
        raise ValueError(f"benchmark MC execution missing {label} for {missing!r}")
    return result


def _float_sequence(value: object | None) -> tuple[float, ...]:
    return tuple(float(item) for item in _raw_sequence(value))


def _correlation_matrix_payload(
    value: object,
    *,
    scenario_contract: MarketScenarioContract | None,
    n_assets: int,
) -> tuple[tuple[float, ...], ...]:
    matrix = _parse_correlation_matrix(value)
    if matrix:
        return matrix
    if scenario_contract is not None and scenario_contract.correlation_source:
        scenario_matrix = scenario_contract.correlation_source.get("matrix")
        matrix = _parse_correlation_matrix(scenario_matrix)
        if matrix:
            return matrix
    return tuple(
        tuple(1.0 if row == col else 0.0 for col in range(n_assets))
        for row in range(n_assets)
    )


def _parse_correlation_matrix(value: object | None) -> tuple[tuple[float, ...], ...]:
    if value is None:
        return ()
    if isinstance(value, str) and not value.strip():
        return ()
    if isinstance(value, str):
        rows = tuple(row.strip() for row in value.split(";") if row.strip())
        if rows:
            return tuple(
                tuple(float(cell.strip()) for cell in row.split(",") if cell.strip())
                for row in rows
            )
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], (list, tuple)):
            return tuple(tuple(float(cell) for cell in row) for row in value)
        values = tuple(float(cell) for cell in value)
        return tuple((cell,) for cell in values)
    return ()


def _requested_outputs_from_task(task: Mapping[str, Any]) -> tuple[str, ...]:
    text = " ".join(
        str(task.get(key) or "")
        for key in ("title", "description", "validation_policy")
    ).lower()
    outputs: list[str] = []
    if "price" in text:
        outputs.append("price")
    if "greek" in text:
        outputs.append("greeks")
    if "bound" in text:
        outputs.append("bounds")
    if "compare" in text or "cross_method" in text:
        outputs.append("comparison")
    return tuple(output for output in outputs if output)


def _benchmark_summary_line(contract: Mapping[str, Any]) -> str:
    product = str(contract.get("product") or "").strip().lower()
    if product == "fx_vanilla":
        return (
            f"Price a {contract.get('option_type', 'call')} FX vanilla option on "
            f"{contract.get('currency_pair', 'EURUSD')} under the analytical benchmark surface."
        )
    if product == "fx_barrier_option":
        return (
            f"Price a {contract.get('barrier_type', 'barrier')} FX option on "
            f"{contract.get('currency_pair', 'EURUSD')} with domestic/foreign discounting."
        )
    if product == "period_rate_option_strip":
        kind = str(contract.get("cap_floor") or "cap").strip().lower()
        return f"Price a {kind} strip under the declared benchmark rates surface."
    if product == "swaption":
        style = str(contract.get("style") or "european").strip().lower()
        if style == "bermudan":
            return "Price a Bermudan swaption under the declared benchmark rates surface."
        return "Price a European-style swaption under the declared Black benchmark surface."
    if product == "barrier_option":
        monitoring = str(contract.get("monitoring") or "").strip().lower()
        if monitoring == "discrete":
            obs = int(contract.get("observations_per_year") or 252)
            return f"Price a discretely monitored barrier option ({obs} observations/year) under the declared analytical benchmark surface."
        return "Price a continuously monitored barrier option under the declared analytical benchmark surface."
    if product == "cds":
        return "Price a single-name CDS under the declared benchmark credit surface."
    if product == "cliquet_option":
        return "Price a reset-style cliquet option under the declared FinancePy-compatible Black-Scholes surface."
    if product == "variance_swap":
        return "Price a variance swap under the declared variance-replication benchmark surface."
    if product == "rainbow_option":
        return "Price a rainbow basket option under the declared benchmark surface."
    return str(contract.get("product") or "Price the declared benchmark contract.")


def _benchmark_detail_lines(
    contract: Mapping[str, Any],
    *,
    scenario_contract: MarketScenarioContract | None,
) -> list[str]:
    product = str(contract.get("product") or "").strip().lower()
    lines: list[str] = []
    if product == "fx_vanilla":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        lines.extend(
            [
                f"Currency pair: {contract.get('currency_pair', 'EURUSD')}.",
                f"Option type: {contract.get('option_type', 'call')}.",
                f"Strike: {contract.get('strike')}.",
                f"Spot: {contract.get('spot')}.",
                f"Expiry date: {expiry_date.isoformat()}.",
                f"Notional: {contract.get('notional')}.",
            ]
        )
        if scenario_contract is not None and scenario_contract.foreign_curve_name:
            lines.append(f"Foreign discount key: {scenario_contract.foreign_curve_name}.")
        return lines
    if product == "fx_barrier_option":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        lines.extend(
            [
                f"Currency pair: {contract.get('currency_pair', 'EURUSD')}.",
                f"Option type: {contract.get('option_type', 'call')}.",
                f"Strike: {contract.get('strike')}.",
                f"Spot: {contract.get('spot')}.",
                f"Barrier: {contract.get('barrier')}.",
                f"Barrier type: {contract.get('barrier_type')}.",
                f"Expiry date: {expiry_date.isoformat()}.",
                f"Notional: {contract.get('notional')}.",
            ]
        )
        if scenario_contract is not None and scenario_contract.foreign_curve_name:
            lines.append(f"Foreign discount key: {scenario_contract.foreign_curve_name}.")
        return lines
    if product == "period_rate_option_strip":
        start_date = _parse_date(contract.get("start_date"))
        end_date = _end_date_from_contract(contract, start_date=start_date)
        lines.extend(
            [
                f"Instrument class: {contract.get('cap_floor', 'cap')}.",
                f"Strike: {contract.get('strike')}.",
                f"Notional: {contract.get('notional')}.",
            ]
        )
        if start_date is not None:
            lines.append(f"Start date: {start_date.isoformat()}.")
        if end_date is not None:
            lines.append(f"End date: {end_date.isoformat()}.")
        if contract.get("payment_frequency"):
            lines.append(f"Payment frequency: {contract['payment_frequency']}.")
        if contract.get("day_count"):
            lines.append(f"Day count: {contract['day_count']}.")
        if scenario_contract is not None and scenario_contract.forecast_curve_name:
            lines.append(f"Rate index: {scenario_contract.forecast_curve_name}.")
        model = str(contract.get("model") or "").strip().lower()
        if model:
            lines.append(f"Pricing model: {model}.")
        if contract.get("shift") not in {None, ""}:
            lines.append(f"Shift: {contract.get('shift')}.")
        sabr = dict(contract.get("sabr") or {})
        if sabr:
            lines.append(
                "SABR parameters: "
                + ", ".join(f"{key}={value}" for key, value in sorted(sabr.items()))
                + "."
            )
        return lines
    if product == "swaption":
        style = str(contract.get("style") or "european").strip().lower()
        lines.extend(
            [
                f"Style: {style}.",
                f"Payer/receiver: {contract.get('payer_receiver', 'payer')}.",
                f"Fixed coupon: {contract.get('fixed_coupon')}.",
                f"Notional: {contract.get('notional')}.",
            ]
        )
        exercise_dates = _parse_date_sequence(contract.get("exercise_dates"))
        if exercise_dates:
            lines.append(
                "Exercise dates: "
                + ", ".join(item.isoformat() for item in exercise_dates)
                + "."
            )
        for key in ("settle_date", "exercise_date", "maturity_date"):
            parsed = _parse_date(contract.get(key))
            if parsed is not None:
                lines.append(f"{key.replace('_', ' ').capitalize()}: {parsed.isoformat()}.")
        if contract.get("fixed_frequency"):
            lines.append(f"Fixed frequency: {contract['fixed_frequency']}.")
        if contract.get("float_frequency"):
            lines.append(f"Floating frequency: {contract['float_frequency']}.")
        if contract.get("fixed_day_count"):
            lines.append(f"Fixed day count: {contract['fixed_day_count']}.")
        if scenario_contract is not None and scenario_contract.forecast_curve_name:
            lines.append(f"Rate index: {scenario_contract.forecast_curve_name}.")
        return lines
    if product == "barrier_option":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        lines = [
            f"Spot: {contract.get('spot')}.",
            f"Strike: {contract.get('strike')}.",
            f"Barrier: {contract.get('barrier')}.",
            f"Barrier type: {contract.get('barrier_type')}.",
            f"Option type: {contract.get('option_type', 'call')}.",
            f"Expiry date: {expiry_date.isoformat()}.",
        ]
        monitoring = str(contract.get("monitoring") or "").strip().lower()
        if monitoring == "discrete":
            obs = int(contract.get("observations_per_year") or 252)
            lines.append(f"Monitoring: discrete ({obs} observations/year).")
        else:
            lines.append("Monitoring: continuous.")
        return lines
    if product == "digital_option":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        return [
            f"Spot: {contract.get('spot')}.",
            f"Strike: {contract.get('strike')}.",
            f"Option type: {contract.get('option_type', 'call')}.",
            f"Payout type: {contract.get('payout_type', 'cash_or_nothing')}.",
            f"Cash payoff: {contract.get('cash_payoff', 1.0)}.",
            f"Expiry date: {expiry_date.isoformat()}.",
        ]
    if product == "lookback_option":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        return [
            f"Spot: {contract.get('spot')}.",
            f"Strike: {contract.get('strike')}.",
            f"Option type: {contract.get('option_type', 'call')}.",
            f"Lookback type: {contract.get('lookback_type', 'fixed_strike')}.",
            f"Running extreme: {contract.get('running_extreme', contract.get('spot'))}.",
            f"Expiry date: {expiry_date.isoformat()}.",
        ]
    if product == "chooser_option":
        choose_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("choose_time_years") or 0.0) * 365.0)
        )
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        return [
            f"Spot: {contract.get('spot')}.",
            f"Choose date: {choose_date.isoformat()}.",
            f"Call expiry date: {expiry_date.isoformat()}.",
            f"Put expiry date: {expiry_date.isoformat()}.",
            f"Call strike: {contract.get('call_strike')}.",
            f"Put strike: {contract.get('put_strike')}.",
        ]
    if product == "compound_option":
        outer_expiry = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("outer_expiry_years") or 0.0) * 365.0)
        )
        inner_expiry = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("inner_expiry_years") or 0.0) * 365.0)
        )
        return [
            f"Spot: {contract.get('spot')}.",
            f"Outer option type: {contract.get('outer_option_type', 'call')}.",
            f"Inner option type: {contract.get('inner_option_type', 'call')}.",
            f"Outer strike: {contract.get('outer_strike')}.",
            f"Inner strike: {contract.get('inner_strike')}.",
            f"Outer expiry date: {outer_expiry.isoformat()}.",
            f"Inner expiry date: {inner_expiry.isoformat()}.",
        ]
    if product == "cds":
        start_date, end_date = _standard_cds_contract_dates(
            contract,
            valuation_date=_valuation_date(contract, scenario_contract),
        )
        lines.extend(
            [
                f"Side: {contract.get('side')}.",
                f"Running coupon: {contract.get('running_coupon')}.",
                f"Notional: {contract.get('notional')}.",
                f"Recovery rate: {contract.get('recovery_rate')}.",
            ]
        )
        if start_date is not None:
            lines.append(f"Start date: {start_date.isoformat()}.")
        if end_date is not None:
            lines.append(f"End date: {end_date.isoformat()}.")
        return lines
    if product == "cliquet_option":
        observation_dates = _observation_dates_from_contract(
            contract,
            valuation_date=_valuation_date(contract, scenario_contract),
        )
        expiry_date = observation_dates[-1] if observation_dates else _valuation_date(contract, scenario_contract)
        lines = [
            f"Spot: {contract.get('spot')}.",
            f"Option type: {contract.get('option_type', 'call')}.",
            f"Final expiry date: {expiry_date.isoformat()}.",
            f"Reset frequency: {contract.get('reset_frequency', 'quarterly')}.",
        ]
        if contract.get("day_count"):
            lines.append(f"Day count: {contract['day_count']}.")
        if observation_dates:
            lines.append("Observation dates: " + ", ".join(dt.isoformat() for dt in observation_dates) + ".")
        return lines
    if product == "variance_swap":
        expiry_date = _valuation_date(contract, scenario_contract) + timedelta(
            days=round(float(contract.get("expiry_years") or 0.0) * 365.0)
        )
        lines = [
            f"Spot: {contract.get('spot')}.",
            f"Variance strike: {contract.get('strike_variance')}.",
            f"Notional: {contract.get('notional')}.",
            f"Expiry date: {expiry_date.isoformat()}.",
        ]
        if contract.get("realized_variance") not in {None, ""}:
            lines.append(f"Realized variance to date: {contract.get('realized_variance')}.")
        if contract.get("replication_strikes"):
            lines.append("Replication strikes: " + ", ".join(str(item) for item in contract.get("replication_strikes") or ()) + ".")
        return lines
    return [f"{key.replace('_', ' ').capitalize()}: {value}." for key, value in contract.items()]


def _fx_vanilla_overrides(
    contract: Mapping[str, Any],
    *,
    scenario_contract: MarketScenarioContract | None,
    valuation_date: date,
) -> dict[str, Any]:
    pair = str(contract.get("currency_pair") or "EURUSD").strip().upper()
    expiry_date = valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0))
    foreign_key = None
    if scenario_contract is not None:
        foreign_key = scenario_contract.foreign_curve_name or scenario_contract.forecast_curve_name
    if not foreign_key and len(pair) == 6:
        foreign_key = f"{pair[:3]}-DISC"
    return {
        "expiry_date": expiry_date,
        "fx_pair": pair,
        "foreign_discount_key": foreign_key,
        "option_type": str(contract.get("option_type") or "call").strip().lower(),
    }


def _rate_cap_floor_overrides(
    contract: Mapping[str, Any],
    *,
    scenario_contract: MarketScenarioContract | None,
) -> dict[str, Any]:
    start_date = _parse_date(contract.get("start_date"))
    return {
        "notional": _float_or_none(contract.get("notional")),
        "strike": _float_or_none(contract.get("strike")),
        "start_date": start_date,
        "end_date": _end_date_from_contract(contract, start_date=start_date),
        "frequency": _frequency(contract.get("payment_frequency")),
        "day_count": _day_count(contract.get("day_count")),
        "calendar_name": str(contract.get("calendar_name") or "weekend_only").strip().lower() or None,
        "business_day_adjustment": str(contract.get("business_day_adjustment") or "following").strip().lower() or None,
        "rate_index": (
            scenario_contract.forecast_curve_name
            if scenario_contract is not None and scenario_contract.forecast_curve_name
            else None
        ),
        "instrument_class": str(contract.get("cap_floor") or "").strip().lower() or None,
        "model": str(contract.get("model") or "black").strip().lower() or None,
        "shift": _float_or_none(contract.get("shift")),
        "sabr": dict(contract.get("sabr") or {}) or None,
    }


def _swaption_overrides(
    contract: Mapping[str, Any],
    *,
    scenario_contract: MarketScenarioContract | None,
) -> dict[str, Any]:
    exercise_date = _parse_date(contract.get("exercise_date"))
    maturity_date = _parse_date(contract.get("maturity_date"))
    payer_receiver = str(contract.get("payer_receiver") or "payer").strip().lower()
    return {
        "notional": _float_or_none(contract.get("notional")),
        "strike": _float_or_none(contract.get("fixed_coupon")),
        "expiry_date": exercise_date,
        "swap_start": exercise_date,
        "swap_end": maturity_date,
        "swap_frequency": _frequency(contract.get("fixed_frequency")),
        "day_count": _day_count(contract.get("fixed_day_count")),
        "rate_index": (
            scenario_contract.forecast_curve_name
            if scenario_contract is not None and scenario_contract.forecast_curve_name
            else None
        ),
        "is_payer": payer_receiver == "payer",
    }


def _barrier_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "strike": _float_or_none(contract.get("strike")),
        "barrier": _float_or_none(contract.get("barrier")),
        "expiry_date": valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0)),
        "barrier_type": str(contract.get("barrier_type") or "").strip().lower() or None,
        "option_type": str(contract.get("option_type") or "call").strip().lower(),
        "rebate": _float_or_none(contract.get("rebate")) or 0.0,
        "observations_per_year": int(contract["observations_per_year"]) if contract.get("observations_per_year") not in {None, ""} else None,
    }


def _digital_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "strike": _float_or_none(contract.get("strike")),
        "expiry_date": valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0)),
        "option_type": str(contract.get("option_type") or "call").strip().lower(),
        "payout_type": str(contract.get("payout_type") or "cash_or_nothing").strip().lower(),
        "cash_payoff": _float_or_none(contract.get("cash_payoff")) or 1.0,
    }


def _lookback_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    running_extreme = _float_or_none(contract.get("running_extreme"))
    spot = _float_or_none(contract.get("spot"))
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": spot,
        "strike": _float_or_none(contract.get("strike")),
        "expiry_date": valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0)),
        "option_type": str(contract.get("option_type") or "call").strip().lower(),
        "lookback_type": str(contract.get("lookback_type") or "fixed_strike").strip().lower(),
        "running_extreme": running_extreme if running_extreme is not None else spot,
    }


def _chooser_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    expiry_date = valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0))
    choose_date = valuation_date + timedelta(days=round(float(contract.get("choose_time_years") or 0.0) * 365.0))
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "choose_date": choose_date,
        "call_expiry_date": expiry_date,
        "put_expiry_date": expiry_date,
        "call_strike": _float_or_none(contract.get("call_strike")),
        "put_strike": _float_or_none(contract.get("put_strike")),
    }


def _compound_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "outer_expiry_date": valuation_date + timedelta(days=round(float(contract.get("outer_expiry_years") or 0.0) * 365.0)),
        "inner_expiry_date": valuation_date + timedelta(days=round(float(contract.get("inner_expiry_years") or 0.0) * 365.0)),
        "outer_strike": _float_or_none(contract.get("outer_strike")),
        "inner_strike": _float_or_none(contract.get("inner_strike")),
        "outer_option_type": str(contract.get("outer_option_type") or "call").strip().lower(),
        "inner_option_type": str(contract.get("inner_option_type") or "call").strip().lower(),
    }


def _cds_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    start_date, end_date = _standard_cds_contract_dates(contract, valuation_date=valuation_date)
    step_in_date = _parse_date(contract.get("step_in_date")) or valuation_date
    return {
        "notional": _float_or_none(contract.get("notional")),
        "spread": _float_or_none(contract.get("running_coupon")),
        "recovery": _float_or_none(contract.get("recovery_rate")),
        "valuation_date": step_in_date,
        "pricing_method": "analytical",
        "start_date": start_date,
        "end_date": end_date,
        "frequency": _frequency(contract.get("payment_frequency")),
        "day_count": _day_count(contract.get("day_count")),
    }


def _nth_to_default_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    basket_names = contract.get("basket_names") or contract.get("names")
    n_names = len(basket_names) if isinstance(basket_names, (list, tuple)) else None
    return {
        "notional": _float_or_none(contract.get("notional")),
        "n_names": n_names,
        "n_th": int(contract.get("nth") or contract.get("n_th") or 1),
        "end_date": _end_date_from_contract(contract, start_date=valuation_date),
        "correlation": _float_or_none(contract.get("correlation")),
        "recovery": _float_or_none(contract.get("recovery_rate") or contract.get("recovery")),
    }


def _rainbow_option_overrides(
    contract: Mapping[str, Any],
    *,
    valuation_date: date,
    scenario_contract: MarketScenarioContract | None,
) -> dict[str, Any]:
    payoff = str(contract.get("payoff") or "").strip().lower()
    basket_style = {
        "best_of_call": "best_of",
        "worst_of_call": "worst_of",
        "spread_call": "spread",
    }.get(payoff, "best_of")
    observation_dates = _parse_date_sequence(contract.get("observation_dates"))
    exercise_dates = _parse_date_sequence(contract.get("exercise_dates"))
    expiry_years = _float_or_none(contract.get("expiry_years"))
    if expiry_years not in {None, 0.0}:
        expiry_date = valuation_date + timedelta(days=round(float(expiry_years) * 365.0))
    else:
        expiry_candidates = observation_dates + exercise_dates
        expiry_date = max(expiry_candidates) if expiry_candidates else valuation_date
    underliers = _rainbow_underlier_names(contract, scenario_contract=scenario_contract)
    _validate_rainbow_vector_dimensions(contract, underliers=underliers)
    underlier_csv = ",".join(underliers)
    risk_free_rate = _float_or_none(contract.get("domestic_rate"))
    if risk_free_rate is None:
        risk_free_rate = _float_or_none(contract.get("risk_free_rate"))
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "underliers": underlier_csv,
        "constituents": underlier_csv,
        "spots": ",".join(str(value) for value in (contract.get("spots") or ())),
        "strike": _float_or_none(contract.get("strike")),
        "expiry_date": expiry_date,
        "observation_dates": observation_dates or None,
        "exercise_dates": exercise_dates or None,
        "correlation": _correlation_text(contract.get("correlation")),
        "dividend_yields": ",".join(str(value) for value in (contract.get("dividend_rates") or ())),
        "basket_style": basket_style,
        "option_type": "call",
        "risk_free_rate": risk_free_rate,
    }


def _rainbow_underlier_names(
    contract: Mapping[str, Any],
    *,
    scenario_contract: MarketScenarioContract | None,
) -> tuple[str, ...]:
    explicit = _string_sequence(contract.get("underliers"))
    if not explicit:
        explicit = _string_sequence(contract.get("constituents"))
    if explicit:
        return explicit

    scenario_underliers = (
        tuple(underlier.name for underlier in scenario_contract.underliers)
        if scenario_contract is not None
        else ()
    )
    if scenario_underliers:
        return scenario_underliers

    raise ValueError(
        "rainbow_option requires named underliers from the contract or market scenario"
    )


def _validate_rainbow_vector_dimensions(
    contract: Mapping[str, Any],
    *,
    underliers: tuple[str, ...],
) -> None:
    n_underliers = len(underliers)
    if n_underliers == 0:
        raise ValueError("rainbow_option underlier count must be greater than zero")

    num_assets = contract.get("num_assets")
    if num_assets not in {None, ""} and int(num_assets) != n_underliers:
        raise ValueError(
            "rainbow_option underlier count mismatch: "
            f"num_assets={int(num_assets)} but underliers={n_underliers}"
        )

    for key in ("spots", "volatilities", "dividend_rates", "weights"):
        values = _raw_sequence(contract.get(key))
        if values and len(values) != n_underliers:
            raise ValueError(
                "rainbow_option underlier count mismatch for "
                f"{key}: {len(values)} values for {n_underliers} underliers"
            )

    correlation = contract.get("correlation")
    if isinstance(correlation, (list, tuple)) and correlation:
        if isinstance(correlation[0], (list, tuple)):
            row_count = len(correlation)
            col_counts = {len(row) for row in correlation if isinstance(row, (list, tuple))}
            if row_count != n_underliers or col_counts != {n_underliers}:
                raise ValueError(
                    "rainbow_option underlier count mismatch for correlation: "
                    f"expected {n_underliers}x{n_underliers}"
                )
        elif len(correlation) != n_underliers:
            raise ValueError(
                "rainbow_option underlier count mismatch for correlation: "
                f"{len(correlation)} values for {n_underliers} underliers"
            )


def _cliquet_option_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    observation_dates = _observation_dates_from_contract(contract, valuation_date=valuation_date)
    expiry_date = observation_dates[-1] if observation_dates else valuation_date
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "expiry_date": expiry_date,
        "observation_dates": observation_dates or None,
        "option_type": str(contract.get("option_type") or "call").strip().lower(),
        "day_count": _day_count(contract.get("day_count")) or DayCountConvention.THIRTY_E_360,
        "time_day_count": _day_count(contract.get("time_day_count")) or DayCountConvention.ACT_365,
    }


def _variance_swap_overrides(contract: Mapping[str, Any], *, valuation_date: date) -> dict[str, Any]:
    replication_strikes = contract.get("replication_strikes")
    replication_volatilities = contract.get("replication_volatilities")
    def _maybe_csv(value: object | None) -> str | None:
        if value is None or value == "":
            return None
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        text = str(value).strip()
        return text or None
    return {
        "notional": _float_or_none(contract.get("notional")) or 1.0,
        "spot": _float_or_none(contract.get("spot")),
        "strike_variance": _float_or_none(contract.get("strike_variance")),
        "expiry_date": valuation_date + timedelta(days=round(float(contract.get("expiry_years") or 0.0) * 365.0)),
        "realized_variance": _float_or_none(contract.get("realized_variance")) or 0.0,
        "replication_strikes": _maybe_csv(replication_strikes),
        "replication_volatilities": _maybe_csv(replication_volatilities),
        "day_count": _day_count(contract.get("day_count")) or DayCountConvention.ACT_365,
    }


def _correlation_text(value: object | None) -> str | None:
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], (list, tuple)):
            return ";".join(
                ",".join(str(item) for item in row)
                for row in value
            )
        return ",".join(str(item) for item in value)
    if value in {None, ""}:
        return None
    return str(value)


def _raw_sequence(value: object | None) -> tuple[object, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
        return tuple(item.strip() for item in items if item.strip())
    return (value,)


def _string_sequence(value: object | None) -> tuple[str, ...]:
    return tuple(
        str(item).strip()
        for item in _raw_sequence(value)
        if str(item).strip()
    )


def _observation_dates_from_contract(contract: Mapping[str, Any], *, valuation_date: date) -> tuple[date, ...]:
    explicit_dates = _parse_date_sequence(contract.get("observation_dates"))
    if explicit_dates:
        return explicit_dates

    observation_times = contract.get("observation_times")
    if isinstance(observation_times, (list, tuple)) and observation_times:
        return tuple(
            valuation_date + timedelta(days=round(float(item) * 365.0))
            for item in observation_times
        )

    expiry_years = _float_or_none(contract.get("expiry_years"))
    if expiry_years in {None, 0.0}:
        return ()

    end_date = valuation_date + timedelta(days=round(float(expiry_years) * 365.0))
    frequency = _frequency(contract.get("reset_frequency"))
    if frequency is None:
        return (end_date,)
    return _generate_schedule_dates(
        start_date=valuation_date,
        end_date=end_date,
        frequency=frequency,
        calendar=_calendar(contract.get("calendar_name")),
        business_day_adjustment=_business_day_adjustment(contract.get("business_day_adjustment")),
        schedule_generation=str(contract.get("schedule_generation") or "forward").strip().lower(),
        adjust_anchors=bool(contract.get("adjust_schedule_anchors")),
    )


def _parse_date_sequence(value: object | None) -> tuple[date, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in (_parse_date(entry) for entry in value) if item is not None)


def _frequency_to_months(value: str) -> int | None:
    normalized = value.strip().lower().replace("-", "_")
    return {
        "monthly": 1,
        "quarterly": 3,
        "semi_annual": 6,
        "semiannual": 6,
        "annual": 12,
    }.get(normalized)


def _generate_schedule_dates(
    *,
    start_date: date,
    end_date: date,
    frequency: Frequency,
    calendar: Calendar | None,
    business_day_adjustment: BusinessDayAdjustment,
    schedule_generation: str,
    adjust_anchors: bool,
) -> tuple[date, ...]:
    months = 12 // frequency.value
    direction = "backward" if schedule_generation == "backward" else "forward"

    def _adjust(input_date: date) -> date:
        if calendar is None or business_day_adjustment == BusinessDayAdjustment.UNADJUSTED:
            return input_date
        return calendar.adjust(input_date, business_day_adjustment)

    start_anchor = _adjust(start_date) if adjust_anchors else start_date
    end_anchor = _adjust(end_date) if adjust_anchors else end_date

    if direction == "backward":
        dates: list[date] = []
        index = 0
        while True:
            candidate = add_months(end_anchor, -(months * index))
            adjusted = _adjust(candidate)
            if adjusted <= start_date:
                break
            dates.insert(0, adjusted)
            index += 1
        return tuple(sorted(dict.fromkeys(dates)))

    dates = []
    index = 1
    while True:
        candidate = add_months(start_anchor, months * index)
        adjusted = _adjust(candidate)
        if adjusted > end_anchor:
            break
        if adjusted > start_date:
            dates.append(adjusted)
        index += 1
    if end_anchor > start_date and (not dates or dates[-1] != end_anchor):
        dates.append(end_anchor)
    return tuple(sorted(dict.fromkeys(dates)))


def _valuation_date(contract: Mapping[str, Any], scenario_contract: MarketScenarioContract | None) -> date:
    if scenario_contract is not None:
        return scenario_contract.valuation_date or scenario_contract.as_of
    parsed = _parse_date(contract.get("valuation_date"))
    return parsed if parsed is not None else date(2024, 11, 15)


def _end_date_from_contract(contract: Mapping[str, Any], *, start_date: date | None) -> date | None:
    explicit_end = _parse_date(contract.get("end_date")) or _parse_date(contract.get("maturity_date"))
    if explicit_end is not None:
        return explicit_end
    tenor = str(contract.get("maturity_tenor") or "").strip()
    if not tenor or start_date is None:
        return None
    return _add_simple_tenor(start_date, tenor)


def _standard_cds_contract_dates(
    contract: Mapping[str, Any],
    *,
    valuation_date: date,
) -> tuple[date, date | None]:
    """Return FinancePy-compatible standard CDS start/end dates."""
    step_in_date = _parse_date(contract.get("step_in_date")) or valuation_date
    explicit_end = _parse_date(contract.get("end_date")) or _parse_date(contract.get("maturity_date"))
    if explicit_end is not None:
        return step_in_date, explicit_end
    tenor = str(contract.get("maturity_tenor") or "").strip()
    if not tenor:
        return step_in_date, None
    accrual_start = _previous_standard_cds_roll(step_in_date)
    maturity = _add_simple_tenor(_next_standard_cds_roll(step_in_date), tenor)
    return accrual_start, maturity


def _previous_standard_cds_roll(as_of: date) -> date:
    month_day = [(3, 20), (6, 20), (9, 20), (12, 20)]
    candidates = [date(as_of.year, month, day) for month, day in month_day]
    prior = [candidate for candidate in candidates if candidate <= as_of]
    if prior:
        return prior[-1]
    return date(as_of.year - 1, 12, 20)


def _next_standard_cds_roll(as_of: date) -> date:
    month_day = [(3, 20), (6, 20), (9, 20), (12, 20)]
    candidates = [date(as_of.year, month, day) for month, day in month_day]
    future = [candidate for candidate in candidates if candidate >= as_of]
    if future:
        return future[0]
    return date(as_of.year + 1, 3, 20)


def _add_simple_tenor(start: date, tenor: str) -> date:
    matched = re.fullmatch(r"(\d+)\s*([YyMm])", tenor.strip())
    if matched is None:
        raise ValueError(f"Unsupported tenor format: {tenor!r}")
    value = int(matched.group(1))
    unit = matched.group(2).upper()
    if unit == "Y":
        return date(start.year + value, start.month, start.day)
    month = start.month - 1 + value
    year = start.year + month // 12
    month = month % 12 + 1
    day = min(start.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    if month in {4, 6, 9, 11}:
        return 30
    return 31


def _float_or_none(value: object | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _parse_date(value: object | None) -> date | None:
    if value in {None, ""}:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _frequency(value: object | None) -> Frequency | None:
    if value in {None, ""}:
        return None
    return _FREQUENCY_MAP.get(str(value).strip().lower().replace("-", "_"))


def _day_count(value: object | None) -> DayCountConvention | None:
    if value in {None, ""}:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    return _DAY_COUNT_MAP.get(normalized)


def _calendar(value: object | None) -> Calendar | None:
    if value in {None, ""}:
        return None
    return _CALENDAR_MAP.get(str(value).strip().lower().replace("-", "_"))


def _business_day_adjustment(value: object | None) -> BusinessDayAdjustment:
    if value in {None, ""}:
        return BusinessDayAdjustment.UNADJUSTED
    normalized = str(value).strip().lower().replace("-", "_")
    return _BUSINESS_DAY_ADJUSTMENT_MAP.get(normalized, BusinessDayAdjustment.UNADJUSTED)


__all__ = [
    "benchmark_contract",
    "benchmark_contract_execution_admission",
    "benchmark_contract_execution_ir",
    "benchmark_contract_lattice_execution",
    "benchmark_contract_monte_carlo_execution",
    "benchmark_preferred_method",
    "benchmark_request_description",
    "benchmark_spec_overrides",
    "canonical_benchmark_instrument_type",
]
