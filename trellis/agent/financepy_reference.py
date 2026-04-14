"""Deterministic FinancePy reference adapters for benchmark tasks."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np

from trellis.agent.task_manifests import load_financepy_bindings, load_market_scenarios


def price_financepy_reference(
    task: Mapping[str, Any],
    *,
    root=None,
) -> dict[str, Any]:
    """Price one benchmark task with the configured FinancePy binding."""
    binding_id = str(task.get("financepy_binding_id") or "").strip()
    if not binding_id:
        raise ValueError(f"Task {task.get('id')} is missing financepy_binding_id")

    bindings = load_financepy_bindings(root=root) if root is not None else load_financepy_bindings()
    scenarios = load_market_scenarios(root=root) if root is not None else load_market_scenarios()
    binding = dict(bindings.get(binding_id) or {})
    if not binding:
        raise KeyError(f"Unknown FinancePy binding: {binding_id}")

    scenario_id = str(task.get("market_scenario_id") or "").strip()
    scenario = dict(scenarios.get(scenario_id) or {})
    contract = dict(task.get("benchmark_contract") or {})

    adapter = _BINDING_ADAPTERS.get(binding_id)
    if adapter is None:
        raise NotImplementedError(f"No FinancePy adapter implemented for {binding_id}")

    started = perf_counter()
    outputs = adapter(contract=contract, scenario=scenario)
    elapsed = round(perf_counter() - started, 6)
    overlapping = list(binding.get("overlapping_outputs") or ())
    return {
        "binding_id": binding_id,
        "instrument_family": binding.get("instrument_family"),
        "method_family": binding.get("method_family"),
        "overlapping_outputs": overlapping,
        "outputs": outputs,
        "elapsed_seconds": elapsed,
    }


def _fp_date(value: str):
    from financepy.utils.date import Date

    year, month, day = (int(part) for part in str(value).split("-"))
    return Date(day, month, year)


def _flat_curve(rate: float, value_date):
    from financepy.market.curves.discount_curve_flat import DiscountCurveFlat

    return DiscountCurveFlat(value_date, float(rate))


def _maybe_method_outputs(option, methods: list[str], *args) -> dict[str, float]:
    outputs: dict[str, float] = {}
    for method_name in methods:
        method = getattr(option, method_name, None)
        if method is None:
            continue
        try:
            value = method(*args)
        except TypeError:
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(item, (int, float, np.floating)):
                    continue
                normalized = str(key).strip().lower()
                if method_name == "delta" and "delta" in normalized:
                    outputs.setdefault("delta", float(item))
                elif normalized:
                    outputs.setdefault(normalized, float(item))
            continue
        outputs[method_name] = float(value)
    return outputs


def _equity_vanilla_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_vanilla_option import EquityVanillaOption
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    model = BlackScholes(float(contract["volatility"]))
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    option = EquityVanillaOption(
        expiry_dt,
        float(contract["strike"]),
        OptionTypes.EUROPEAN_CALL if contract["option_type"] == "call" else OptionTypes.EUROPEAN_PUT,
        float(contract.get("notional", 1.0)),
    )
    args = (value_dt, float(contract["spot"]), discount_curve, dividend_curve, model)
    outputs = {"price": float(option.value(*args))}
    outputs.update(_maybe_method_outputs(option, ["delta", "gamma", "vega", "theta"], *args))
    return outputs


def _fx_vanilla_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.fx.fx_vanilla_option import FXVanillaOption
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    model = BlackScholes(float(contract["volatility"]))
    domestic_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    foreign_curve = _flat_curve(float(contract["foreign_rate"]), value_dt)
    option = FXVanillaOption(
        expiry_dt,
        float(contract["strike"]),
        str(contract["currency_pair"]),
        OptionTypes.EUROPEAN_CALL if contract["option_type"] == "call" else OptionTypes.EUROPEAN_PUT,
        float(contract["notional"]),
        str(contract["premium_currency"]),
    )
    args = (value_dt, float(contract["spot"]), domestic_curve, foreign_curve, model)
    price_payload = option.value(*args)
    outputs = {"price": float(price_payload["v"]) if isinstance(price_payload, Mapping) and "v" in price_payload else float(price_payload)}
    outputs.update(_maybe_method_outputs(option, ["delta", "vega", "theta", "gamma"], *args))
    return outputs


def _cap_floor_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black import Black
    from financepy.models.black_shifted import BlackShifted
    from financepy.models.sabr import SABR
    from financepy.products.rates.ibor_cap_floor import IborCapFloor
    from financepy.utils.day_count import DayCountTypes
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.global_types import FinCapFloorTypes

    inputs = dict(scenario.get("benchmark_inputs") or {})
    value_dt = _fp_date(inputs["valuation_date"])
    libor_curve = _flat_curve(float(inputs["flat_forward_rate"]), value_dt)
    model_name = str(contract.get("model") or "black")
    if model_name == "shifted_black":
        model = BlackShifted(float(inputs["shifted_black_vol"]), float(contract.get("shift") or inputs.get("shift") or 0.0))
    elif model_name == "sabr":
        sabr = dict(contract.get("sabr") or inputs.get("sabr") or {})
        model = SABR(float(sabr["alpha"]), float(sabr["beta"]), float(sabr["rho"]), float(sabr["nu"]))
    else:
        model = Black(float(inputs["black_vol"]))
    option = IborCapFloor(
        value_dt,
        str(contract["maturity_tenor"]),
        FinCapFloorTypes.CAP if contract["cap_floor"] == "cap" else FinCapFloorTypes.FLOOR,
        float(contract["strike"]),
        freq_type=FrequencyTypes.QUARTERLY,
        dc_type=DayCountTypes.ACT_360,
        notional=float(contract["notional"]),
    )
    return {"price": float(option.value(value_dt, libor_curve, model))}


def _swaption_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black import Black
    from financepy.products.rates.ibor_swaption import IborSwaption
    from financepy.utils.day_count import DayCountTypes
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.global_types import SwapTypes

    inputs = dict(scenario.get("benchmark_inputs") or {})
    value_dt = _fp_date(inputs["valuation_date"])
    exercise_dt = _fp_date(contract["exercise_date"])
    maturity_dt = _fp_date(contract["maturity_date"])
    option = IborSwaption(
        value_dt,
        exercise_dt,
        maturity_dt,
        SwapTypes.PAY if contract["payer_receiver"] == "payer" else SwapTypes.RECEIVE,
        float(contract["fixed_coupon"]),
        FrequencyTypes.SEMI_ANNUAL,
        DayCountTypes.THIRTY_E_360,
        notional=float(contract["notional"]),
        float_freq_type=FrequencyTypes.QUARTERLY,
        float_dc_type=DayCountTypes.THIRTY_E_360,
    )
    discount_curve = _flat_curve(float(inputs["flat_discount_rate"]), value_dt)
    model = Black(float(inputs["black_vol"]))
    return {"price": float(option.value(value_dt, discount_curve, model))}


def _cds_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.products.credit.cds import CDS
    from financepy.products.credit.cds_curve import CDSCurve
    from financepy.utils.day_count import DayCountTypes
    from financepy.utils.frequency import FrequencyTypes

    inputs = dict(scenario.get("benchmark_inputs") or {})
    value_dt = _fp_date(inputs["valuation_date"])
    discount_curve = _flat_curve(float(inputs["flat_discount_rate"]), value_dt)
    cds = CDS(
        value_dt,
        str(contract["maturity_tenor"]),
        float(contract["running_coupon"]),
        notional=float(contract["notional"]),
        long_protect=contract["side"] == "protection_buyer",
        freq_type=FrequencyTypes.QUARTERLY,
        dc_type=DayCountTypes.ACT_360,
    )
    issuer_curve = CDSCurve(
        value_dt,
        [cds],
        discount_curve,
        float(contract["recovery_rate"]),
    )
    value = cds.value(value_dt, issuer_curve, float(contract["recovery_rate"]))
    outputs = {"price": float(value["clean_pv"])}
    try:
        outputs["credit_delta"] = float(
            cds.credit_dv01(value_dt, issuer_curve, float(contract["recovery_rate"]))
        )
    except Exception:
        pass
    return outputs


def _rainbow_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.products.equity.equity_rainbow_option import (
        EquityRainbowOption,
        EquityRainbowOptionTypes,
    )

    inputs = dict(scenario.get("benchmark_inputs") or {})
    value_dt = _fp_date(inputs["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    option = EquityRainbowOption(
        expiry_dt,
        EquityRainbowOptionTypes.CALL_ON_MAXIMUM,
        [float(contract["strike"])],
        int(contract["num_assets"]),
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curves = [
        _flat_curve(float(rate), value_dt)
        for rate in contract["dividend_rates"]
    ]
    return {
        "price": float(
            option.value(
                value_dt,
                np.asarray(contract["spots"], dtype=float),
                discount_curve,
                dividend_curves,
                np.asarray(contract["volatilities"], dtype=float),
                np.asarray(contract["correlation"], dtype=float),
            )
        )
    }


def _barrier_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_barrier_option import EquityBarrierOption
    from financepy.utils.global_types import EquityBarrierTypes

    barrier_types = {
        "up_and_out": EquityBarrierTypes.UP_AND_OUT_CALL,
        "up_and_in": EquityBarrierTypes.UP_AND_IN_CALL,
        "down_and_out": EquityBarrierTypes.DOWN_AND_OUT_CALL,
        "down_and_in": EquityBarrierTypes.DOWN_AND_IN_CALL,
    }
    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    option = EquityBarrierOption(
        expiry_dt,
        float(contract["strike"]),
        barrier_types[str(contract["barrier_type"])],
        float(contract["barrier"]),
        notional=float(contract.get("notional", 1.0)),
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    model = BlackScholes(float(contract["volatility"]))
    return {
        "price": float(option.value(value_dt, float(contract["spot"]), discount_curve, dividend_curve, model))
    }


def _digital_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_digital_option import (
        EquityDigitalOption,
        FinDigitalOptionTypes,
    )
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    option = EquityDigitalOption(
        expiry_dt,
        float(contract["strike"]),
        OptionTypes.EUROPEAN_CALL if contract["option_type"] == "call" else OptionTypes.EUROPEAN_PUT,
        FinDigitalOptionTypes.CASH_OR_NOTHING,
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    model = BlackScholes(float(contract["volatility"]))
    return {
        "price": float(option.value(value_dt, float(contract["spot"]), discount_curve, dividend_curve, model))
    }


def _lookback_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.products.equity.equity_fixed_lookback_option import EquityFixedLookbackOption
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    option = EquityFixedLookbackOption(
        expiry_dt,
        OptionTypes.EUROPEAN_CALL if contract["option_type"] == "call" else OptionTypes.EUROPEAN_PUT,
        float(contract["strike"]),
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    return {
        "price": float(
            option.value(
                value_dt,
                float(contract["spot"]),
                discount_curve,
                dividend_curve,
                float(contract["volatility"]),
                float(contract["spot"]),
            )
        )
    }


def _chooser_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_chooser_option import EquityChooserOption

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    choose_dt = value_dt.add_tenor(f"{int(contract['choose_time_years'] * 12)}M")
    expiry_dt = value_dt.add_tenor(f"{int(contract['expiry_years'] * 12)}M")
    option = EquityChooserOption(
        choose_dt,
        expiry_dt,
        expiry_dt,
        float(contract["call_strike"]),
        float(contract["put_strike"]),
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    model = BlackScholes(float(contract["volatility"]))
    return {
        "price": float(option.value(value_dt, float(contract["spot"]), discount_curve, dividend_curve, model))
    }


def _compound_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_compound_option import EquityCompoundOption
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    outer_expiry = value_dt.add_tenor(f"{int(contract['outer_expiry_years'] * 12)}M")
    inner_expiry = value_dt.add_tenor(f"{int(contract['inner_expiry_years'] * 12)}M")
    option = EquityCompoundOption(
        outer_expiry,
        OptionTypes.EUROPEAN_CALL,
        float(contract["outer_strike"]),
        inner_expiry,
        OptionTypes.EUROPEAN_CALL,
        float(contract["inner_strike"]),
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    model = BlackScholes(float(contract["volatility"]))
    return {
        "price": float(option.value(value_dt, float(contract["spot"]), discount_curve, dividend_curve, model))
    }


def _cliquet_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.models.black_scholes import BlackScholes
    from financepy.products.equity.equity_cliquet_option import EquityCliquetOption
    from financepy.utils.day_count import DayCountTypes
    from financepy.utils.frequency import FrequencyTypes
    from financepy.utils.global_types import OptionTypes

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    final_expiry = value_dt.add_tenor("1Y")
    option = EquityCliquetOption(
        value_dt,
        final_expiry,
        OptionTypes.EUROPEAN_CALL,
        FrequencyTypes.QUARTERLY,
        DayCountTypes.THIRTY_E_360,
    )
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    dividend_curve = _flat_curve(float(contract["dividend_rate"]), value_dt)
    model = BlackScholes(
        float(contract["volatility"]),
        num_steps_per_year=52,
        num_paths=int(contract.get("num_paths") or 10000),
    )
    return {
        "price": float(option.value(value_dt, float(contract["spot"]), discount_curve, dividend_curve, model))
    }


def _variance_swap_reference(*, contract: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    from financepy.products.equity.equity_variance_swap import EquityVarianceSwap

    value_dt = _fp_date(scenario["benchmark_inputs"]["valuation_date"])
    discount_curve = _flat_curve(float(contract["domestic_rate"]), value_dt)
    swap = EquityVarianceSwap(
        value_dt,
        "1Y",
        float(contract["strike_variance"]),
        notional=float(contract["notional"]),
    )
    strike_grid = np.asarray([60.0, 80.0, 100.0, 120.0, 140.0], dtype=float)
    vol_grid = np.asarray([0.26, 0.24, 0.22, 0.23, 0.25], dtype=float)
    fair_strike_var = float(
        swap.fair_strike_approx(
            value_dt,
            float(contract["spot"]),
            strike_grid,
            vol_grid,
        )
    )
    return {
        "price": float(swap.value(value_dt, 0.0, fair_strike_var, discount_curve)),
        "fair_strike_variance": fair_strike_var,
    }


_BINDING_ADAPTERS: dict[str, Any] = {
    "financepy.equity.vanilla.black_scholes": _equity_vanilla_reference,
    "financepy.fx.vanilla.garman_kohlhagen": _fx_vanilla_reference,
    "financepy.rates.cap_floor.black": _cap_floor_reference,
    "financepy.rates.cap_floor.shifted_black": _cap_floor_reference,
    "financepy.rates.cap_floor.sabr": _cap_floor_reference,
    "financepy.rates.swaption.black": _swaption_reference,
    "financepy.credit.cds.analytical": _cds_reference,
    "financepy.equity.rainbow.stulz": _rainbow_reference,
    "financepy.equity.barrier.black_scholes": _barrier_reference,
    "financepy.equity.digital.black_scholes": _digital_reference,
    "financepy.equity.lookback.fixed": _lookback_reference,
    "financepy.equity.chooser.black_scholes": _chooser_reference,
    "financepy.equity.compound.black_scholes": _compound_reference,
    "financepy.equity.cliquet.monte_carlo": _cliquet_reference,
    "financepy.equity.variance_swap.analytical": _variance_swap_reference,
}
