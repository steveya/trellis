"""Deterministic paired FpML/native conformance task execution.

This module is an evaluation harness, not a pricing authority.  It constructs
existing generic contract IR values from structured oracle terms and compares
them with contracts admitted through the production FpML compiler boundary.
It is intentionally excluded from the import registry used to authorize
generated pricing code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
_NO_AGENT_CALLS = {
    "builder": False,
    "codegen": False,
    "quant_review": False,
    "model_validator": False,
    "recovery": False,
}


def build_native_conformance_contract(spec: Mapping[str, object]):
    """Build an existing generic IR value from one structured task oracle."""

    if not isinstance(spec, Mapping):
        raise TypeError("native conformance contract must be a mapping")
    kind = _required_text(spec, "kind")
    if kind == "fixed_float_swap":
        return _build_fixed_float_swap(spec)
    if kind == "period_rate_option_strip":
        return _build_period_rate_option_strip(spec)
    if kind == "european_swaption":
        return _build_european_swaption(spec)
    raise ValueError(f"unsupported native conformance contract kind {kind!r}")


def run_fpml_conformance_task(
    task: dict[str, Any],
    market_state,
    *,
    timer: Callable[[], float],
    now_fn: Callable[[], datetime],
    task_run_storage_root: Path | None = None,
    task_run_storage_layout: str = "repo",
) -> dict[str, Any]:
    """Run one deterministic FpML conformance task without agent execution."""

    from trellis.agent.evals import (
        task_result_outcome_class,
        task_result_passed_expectation,
    )

    started = now_fn()
    t0 = timer()
    result: dict[str, Any] = {
        "task_id": str(task["id"]),
        "title": str(task["title"]),
        "task_kind": "fpml_conformance",
        "task_corpus": str(task.get("task_corpus") or "fpml_conformance"),
        "task_definition_version": task.get("task_definition_version"),
        "task_definition_manifest": str(task.get("task_definition_manifest") or ""),
        "market_scenario_id": str(task.get("market_scenario_id") or ""),
        "start_time": started.isoformat(),
        "run_started_at": started.isoformat(),
        "execution_mode": "deterministic_import_conformance",
        "generation_policy": "deterministic_allowed",
        "generation_evidence": {
            "policy": "deterministic_allowed",
            "artifact_origins": ["existing_ir", "existing_route"],
            "agent_synthesis_attempted": False,
            "agent_synthesis_observed": False,
        },
        "agent_calls": dict(_NO_AGENT_CALLS),
        "recovery_attempts": [],
        "token_usage_summary": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    }

    try:
        compiled, import_summary = _compile_fpml_task(task, market_state)
        result["import_report"] = import_summary
        expected_outcome = str(task.get("expected_outcome") or "pricing_success")
        if expected_outcome == "honest_block":
            _record_expected_block(result, task, compiled)
        elif expected_outcome == "pricing_success":
            _record_positive_pair(result, task, compiled, market_state)
        else:
            raise ValueError(
                f"unsupported FpML conformance expected_outcome {expected_outcome!r}"
            )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "passed_expectation": False,
                "outcome_class": "build_failure",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "failures": [str(exc)[:500]],
            }
        )

    result["elapsed_seconds"] = round(timer() - t0, 3)
    result["run_completed_at"] = now_fn().isoformat()
    result["outcome_class"] = task_result_outcome_class(result)
    result["passed_expectation"] = task_result_passed_expectation(result)
    _persist_result(
        task,
        result,
        task_run_storage_root=task_run_storage_root,
        task_run_storage_layout=task_run_storage_layout,
    )
    return result


def _record_positive_pair(result, task, compiled, market_state) -> None:
    from trellis.io.fpml import fpml_import_report_summary

    report = compiled.import_report
    if report is None or report.normalized_contract is None:
        blockers = _blocker_report_payload(compiled.blocker_report)
        raise ValueError(
            "positive FpML conformance task did not normalize: "
            + ", ".join(item["id"] for item in blockers["blockers"])
        )

    imported = report.normalized_contract
    native = build_native_conformance_contract(task["native_contract"])
    imported_identity, imported_projection = _economic_evidence(imported)
    native_identity, native_projection = _economic_evidence(native)
    imported_selection = _selection_evidence(imported)
    native_selection = _selection_evidence(native)
    imported_binding = _binding_evidence(imported, market_state)
    native_binding = _binding_evidence(native, market_state)

    imported_price = float(compiled.request.instrument.evaluate(market_state))
    native_price = float(_price_contract(native, market_state))
    tolerance = _price_tolerance(task, imported_price, native_price)
    difference = abs(imported_price - native_price)
    price_equal = difference <= tolerance["allowed_difference"]
    envelope_variants = [
        _evaluate_envelope_variant(
            task,
            market_state,
            variant,
            baseline_identity=imported_identity,
            baseline_selection=imported_selection,
            baseline_binding=imported_binding,
            baseline_price=imported_price,
        )
        for variant in task.get("fpml", {}).get("envelope_variants", ())
    ]

    gates = {
        "identity": imported_identity == native_identity,
        "economic_projection": imported_projection == native_projection,
        "selection": imported_selection == native_selection,
        "binding": imported_binding == native_binding,
        "price": price_equal,
        "envelope_variants": all(
            bool(variant.get("passed")) for variant in envelope_variants
        ),
    }
    success = all(gates.values())
    result.update(
        {
            "success": success,
            "outcome_class": "compare_ready",
            "passed_expectation": success,
            "conformance": {
                "identity": {
                    "equal": gates["identity"],
                    "fpml": imported_identity,
                    "native": native_identity,
                },
                "economic_projection_equal": gates["economic_projection"],
                "selection": {
                    "equal": gates["selection"],
                    "fpml": imported_selection,
                    "native": native_selection,
                },
                "binding": {
                    "equal": gates["binding"],
                    "fpml": imported_binding,
                    "native": native_binding,
                },
                "price": {
                    "within_tolerance": price_equal,
                    "fpml": imported_price,
                    "native": native_price,
                    "absolute_difference": difference,
                    **tolerance,
                },
                "envelope_variants": envelope_variants,
                "gates": gates,
            },
            "cross_validation": {
                "status": "passed" if success else "failed",
                "reference_target": "native_contract_oracle",
                "prices": {"fpml": imported_price, "native": native_price},
                "absolute_difference": difference,
                **tolerance,
            },
            "import_report": fpml_import_report_summary(report),
        }
    )
    if not success:
        failed = [name for name, passed in gates.items() if not passed]
        result["failures"] = [
            "FpML/native conformance gates failed: " + ", ".join(failed)
        ]


def _record_expected_block(result, task, compiled) -> None:
    expected = [str(item) for item in task.get("expected_blocker_ids", ())]
    observed = [
        str(blocker.id) for blocker in getattr(compiled.blocker_report, "blockers", ())
    ]
    matched = observed == expected and bool(observed)
    blocker_payload = _blocker_report_payload(compiled.blocker_report)
    result.update(
        {
            "success": False,
            "expected_honest_block": True,
            "outcome_class": "honest_block",
            "passed_expectation": matched,
            "conformance": {
                "expected_blocker_ids": expected,
                "observed_blocker_ids": observed,
                "blockers_equal": matched,
                "price": None,
            },
            "blocker_details": {"blocker_report": blocker_payload},
            "failures": []
            if matched
            else [f"Expected FpML blocker ids {expected!r}, observed {observed!r}."],
        }
    )


def _compile_fpml_task(
    task: Mapping[str, object],
    market_state,
    *,
    envelope_variant: Mapping[str, object] | None = None,
):
    from trellis.agent.platform_requests import (
        compile_platform_request,
        make_fpml_request,
    )
    from trellis.io.fpml import fpml_import_report_summary

    fpml = task.get("fpml")
    if not isinstance(fpml, Mapping):
        raise ValueError("FpML conformance task requires an fpml mapping")
    fixture = ROOT / _required_text(fpml, "fixture")
    xml = fixture.read_bytes()
    envelope = _trade_envelope(fpml, envelope_variant=envelope_variant)
    request = make_fpml_request(
        xml,
        source_view=_optional_text(fpml.get("source_view")),
        source_version=_optional_text(fpml.get("source_version")),
        source_reference=str(fixture.relative_to(ROOT)),
        trade_envelope=envelope,
        market_snapshot=market_state,
        settlement=(
            getattr(market_state, "settlement", None)
            or getattr(market_state, "as_of", None)
        ),
        request_type="price",
        metadata={"task_kind": "fpml_conformance"},
    )
    compiled = compile_platform_request(request)
    if compiled.import_report is not None:
        summary = fpml_import_report_summary(compiled.import_report)
    else:
        summary = _preflight_import_summary(compiled.blocker_report)
    return compiled, summary


def _trade_envelope(
    fpml: Mapping[str, object],
    *,
    envelope_variant: Mapping[str, object] | None,
):
    from trellis.agent.trade_envelope import TradeEnvelope, TradeParty

    variant = dict(envelope_variant or {})
    valuation_party_id = _optional_text(fpml.get("valuation_party_id"))
    parties = (
        (
            TradeParty(
                valuation_party_id,
                role="valuation_party",
                name=_optional_text(variant.get("valuation_party_name")),
            ),
        )
        if valuation_party_id
        else ()
    )
    return TradeEnvelope(
        source_format="fpml",
        source_view=_optional_text(fpml.get("source_view")),
        source_version=_optional_text(fpml.get("source_version")),
        parties=parties,
        identifiers={
            str(key): str(value)
            for key, value in dict(variant.get("identifiers") or {}).items()
        },
        metadata=dict(variant.get("metadata") or {}),
    )


def _evaluate_envelope_variant(
    task,
    market_state,
    variant,
    *,
    baseline_identity,
    baseline_selection,
    baseline_binding,
    baseline_price,
) -> dict[str, object]:
    compiled, summary = _compile_fpml_task(
        task,
        market_state,
        envelope_variant=variant,
    )
    report = compiled.import_report
    if report is None or report.normalized_contract is None:
        return {
            "name": str(variant.get("name") or "unnamed"),
            "passed": False,
            "observed_blocker_ids": [
                item["id"]
                for item in _blocker_report_payload(compiled.blocker_report)["blockers"]
            ],
            "import_status": summary.get("status"),
        }
    contract = report.normalized_contract
    identity, _projection = _economic_evidence(contract)
    selection = _selection_evidence(contract)
    binding = _binding_evidence(contract, market_state)
    price = float(compiled.request.instrument.evaluate(market_state))
    tolerance = _price_tolerance(task, baseline_price, price)
    gates = {
        "identity": identity == baseline_identity,
        "selection": selection == baseline_selection,
        "binding": binding == baseline_binding,
        "price": abs(price - baseline_price) <= tolerance["allowed_difference"],
    }
    return {
        "name": str(variant.get("name") or "unnamed"),
        "passed": all(gates.values()),
        "gates": gates,
        "economic_identity": identity,
        "price": price,
    }


def _build_fixed_float_swap(spec: Mapping[str, object]):
    from trellis.agent.static_leg_contract import (
        CouponLeg,
        CouponPeriod,
        FixedCouponFormula,
        FloatingCouponFormula,
        NotionalSchedule,
        NotionalStep,
        SettlementRule,
        SignedLeg,
        StaticLegContractIR,
        TermRateIndex,
    )
    from trellis.conventions.schedule import generate_schedule

    start = _required_date(spec, "start_date")
    end = _required_date(spec, "end_date")
    fixed_direction = _required_text(spec, "fixed_direction")
    notional = NotionalSchedule(
        (NotionalStep(start, end, _required_float(spec, "notional")),)
    )

    def periods(frequency_key: str, *, floating: bool):
        frequency = _frequency(_required_text(spec, frequency_key))
        ends = tuple(generate_schedule(start, end, frequency))
        starts = (start, *ends[:-1])
        return tuple(
            CouponPeriod(
                accrual_start=left,
                accrual_end=right,
                payment_date=right,
                fixing_date=left if floating else None,
            )
            for left, right in zip(starts, ends)
        )

    return StaticLegContractIR(
        legs=(
            SignedLeg(
                fixed_direction,
                CouponLeg(
                    currency=_required_text(spec, "currency"),
                    notional_schedule=notional,
                    coupon_periods=periods("fixed_frequency", floating=False),
                    coupon_formula=FixedCouponFormula(
                        _required_float(spec, "fixed_rate")
                    ),
                    day_count=_required_text(spec, "fixed_day_count"),
                    payment_frequency=_required_text(spec, "fixed_frequency"),
                    label="fixed_leg",
                ),
            ),
            SignedLeg(
                _opposite_direction(fixed_direction),
                CouponLeg(
                    currency=_required_text(spec, "currency"),
                    notional_schedule=notional,
                    coupon_periods=periods("floating_frequency", floating=True),
                    coupon_formula=FloatingCouponFormula(
                        TermRateIndex(
                            _required_text(spec, "rate_index"),
                            _required_text(spec, "index_tenor"),
                        )
                    ),
                    day_count=_required_text(spec, "floating_day_count"),
                    payment_frequency=_required_text(spec, "floating_frequency"),
                    label="floating_leg",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=_required_text(spec, "currency")),
        metadata={"semantic_family": "fixed_float_swap"},
    )


def _build_period_rate_option_strip(spec: Mapping[str, object]):
    from trellis.agent.static_leg_contract import (
        NotionalSchedule,
        NotionalStep,
        PeriodRateOptionPeriod,
        PeriodRateOptionStripLeg,
        SettlementRule,
        SignedLeg,
        StaticLegContractIR,
        TermRateIndex,
    )
    from trellis.conventions.schedule import generate_schedule

    start = _required_date(spec, "start_date")
    end = _required_date(spec, "end_date")
    frequency_text = _required_text(spec, "payment_frequency")
    ends = tuple(generate_schedule(start, end, _frequency(frequency_text)))
    starts = (start, *ends[:-1])
    option_side = _required_text(spec, "option_side")
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                _required_text(spec, "direction"),
                PeriodRateOptionStripLeg(
                    currency=_required_text(spec, "currency"),
                    notional_schedule=NotionalSchedule(
                        (
                            NotionalStep(
                                start,
                                end,
                                _required_float(spec, "notional"),
                            ),
                        )
                    ),
                    option_periods=tuple(
                        PeriodRateOptionPeriod(
                            accrual_start=left,
                            accrual_end=right,
                            fixing_date=left,
                            payment_date=right,
                        )
                        for left, right in zip(starts, ends)
                    ),
                    rate_index=TermRateIndex(
                        _required_text(spec, "rate_index"),
                        _required_text(spec, "index_tenor"),
                    ),
                    strike=_required_float(spec, "strike"),
                    option_side=option_side,
                    day_count=_required_text(spec, "day_count"),
                    payment_frequency=frequency_text,
                    label="cap_strip" if option_side == "call" else "floor_strip",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=_required_text(spec, "currency")),
        metadata={"semantic_family": "period_rate_option_strip"},
    )


def _build_european_swaption(spec: Mapping[str, object]):
    from trellis.agent.contract_ir import (
        Annuity,
        Constant,
        ContractIR,
        Exercise,
        FiniteSchedule,
        ForwardRate,
        Max,
        Observation,
        Scaled,
        Singleton,
        Strike,
        Sub,
        SwapRate,
        Underlying,
    )
    from trellis.agent.static_leg_contract import SettlementRule

    underlying_spec = spec.get("underlying_swap")
    if not isinstance(underlying_spec, Mapping):
        raise ValueError("european swaption requires underlying_swap terms")
    underlying = _build_fixed_float_swap(underlying_spec)
    fixed_leg = next(
        signed_leg.leg
        for signed_leg in underlying.legs
        if type(signed_leg.leg.coupon_formula).__name__ == "FixedCouponFormula"
    )
    underlier_id = _required_text(spec, "underlier_id")
    fixed_dates = FiniteSchedule(
        tuple(period.payment_date for period in fixed_leg.coupon_periods)
    )
    expiry = Singleton(_required_date(spec, "exercise_date"))
    payer_receiver = _required_text(spec, "payer_receiver")
    if payer_receiver != "payer":
        raise ValueError("the admitted conformance swaption oracle is payer-only")
    return ContractIR(
        payoff=Scaled(
            Annuity(underlier_id, fixed_dates),
            Max(
                (
                    Sub(
                        SwapRate(underlier_id, fixed_dates),
                        Strike(_required_float(spec, "strike")),
                    ),
                    Constant(0.0),
                )
            ),
        ),
        exercise=Exercise("european", expiry),
        observation=Observation("terminal", expiry),
        underlying=Underlying(ForwardRate(underlier_id, "lognormal_forward")),
        position=_required_text(spec, "position"),
        settlement=SettlementRule(
            settlement_kind=_required_text(spec, "settlement_kind"),
            payout_currency=_required_text(underlying_spec, "currency"),
        ),
        underlying_contract=underlying,
    )


def _economic_evidence(contract) -> tuple[str, dict[str, object]]:
    from trellis.agent.contract_ir import (
        ContractIR,
        contract_ir_economic_identity,
        contract_ir_economic_summary,
    )
    from trellis.agent.static_leg_contract import (
        StaticLegContractIR,
        static_leg_economic_identity,
        static_leg_economic_summary,
    )

    if isinstance(contract, StaticLegContractIR):
        return (
            static_leg_economic_identity(contract),
            static_leg_economic_summary(contract),
        )
    if isinstance(contract, ContractIR):
        return contract_ir_economic_identity(contract), contract_ir_economic_summary(
            contract
        )
    raise TypeError(f"unsupported conformance contract type {type(contract).__name__}")


def _selection_evidence(contract) -> dict[str, object]:
    from trellis.agent.contract_ir import ContractIR
    from trellis.agent.contract_ir_solver_compiler import select_contract_ir_solver
    from trellis.agent.static_leg_admission import select_static_leg_lowering
    from trellis.agent.static_leg_contract import StaticLegContractIR

    if isinstance(contract, StaticLegContractIR):
        selection = select_static_leg_lowering(contract)
        return {
            "contract_type": "StaticLegContractIR",
            "declaration_id": selection.declaration_id,
            "callable_ref": selection.callable_ref,
            "validation_bundle_id": selection.validation_bundle_id,
            "method": selection.method,
        }
    if isinstance(contract, ContractIR):
        selection = select_contract_ir_solver(contract)
        return {
            "contract_type": "ContractIR",
            "declaration_id": selection.declaration_id,
            "callable_ref": selection.callable_ref,
            "validation_bundle_id": selection.validation_bundle_id,
            "requested_method": selection.requested_method,
            "requested_outputs": list(selection.requested_outputs),
            "match_bindings": _summary_value(selection.match_bindings),
        }
    raise TypeError(f"unsupported conformance contract type {type(contract).__name__}")


def _binding_evidence(contract, market_state) -> dict[str, object]:
    from trellis.agent.contract_ir import ContractIR
    from trellis.agent.contract_ir_solver_compiler import compile_contract_ir_solver
    from trellis.agent.static_leg_contract import StaticLegContractIR
    from trellis.execution import compile_static_leg_execution_ir

    if isinstance(contract, StaticLegContractIR):
        execution_ir = compile_static_leg_execution_ir(
            contract,
            fail_on_unsupported=True,
        )
        return {
            "binding_kind": "static_execution_ir",
            "execution_ir": _summary_value(execution_ir),
        }
    if isinstance(contract, ContractIR):
        decision = compile_contract_ir_solver(contract, market_state=market_state)
        return {
            "binding_kind": "contract_ir_solver_decision",
            "decision": _summary_value(decision),
        }
    raise TypeError(f"unsupported conformance contract type {type(contract).__name__}")


def _price_contract(contract, market_state) -> float:
    from trellis.agent.contract_ir import ContractIR
    from trellis.agent.contract_ir_solver_compiler import ContractIRPricingPayoff
    from trellis.agent.static_leg_contract import StaticLegContractIR
    from trellis.core.payoff import ExecutionBackedPayoff
    from trellis.execution import compile_static_leg_execution_ir

    if isinstance(contract, StaticLegContractIR):
        return ExecutionBackedPayoff(
            compile_static_leg_execution_ir(contract, fail_on_unsupported=True)
        ).evaluate(market_state)
    if isinstance(contract, ContractIR):
        return ContractIRPricingPayoff(contract).evaluate(market_state)
    raise TypeError(f"unsupported conformance contract type {type(contract).__name__}")


def _price_tolerance(task, left: float, right: float) -> dict[str, float]:
    spec = dict(task.get("tolerance") or {})
    absolute = float(spec.get("absolute") or 0.0)
    relative = float(spec.get("relative") or 0.0)
    return {
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "allowed_difference": max(
            absolute,
            relative * max(abs(left), abs(right)),
        ),
    }


def _blocker_report_payload(report) -> dict[str, object]:
    blockers = []
    for blocker in getattr(report, "blockers", ()):
        blockers.append(
            {
                "id": str(blocker.id),
                "category": str(blocker.category),
                "severity": str(blocker.severity),
                "summary": str(blocker.summary),
                "target_package": blocker.target_package,
            }
        )
    return {
        "should_block": bool(getattr(report, "should_block", blockers)),
        "summary": str(getattr(report, "summary", "")),
        "blockers": blockers,
    }


def _preflight_import_summary(report) -> dict[str, object]:
    blockers = _blocker_report_payload(report)["blockers"]
    missing_fields = [
        str(item["id"]).split(":", 1)[1]
        for item in blockers
        if str(item["id"]).startswith("missing_contract_field:")
    ]
    ambiguous_fields = [
        str(item["id"]).split(":", 1)[1]
        for item in blockers
        if str(item["id"]).startswith("contract_ambiguity:")
    ]
    return {
        "status": "blocked",
        "profile": None,
        "document": None,
        "trade": None,
        "trade_envelope": None,
        "economic_identity": None,
        "normalized_contract": None,
        "mapping_provenance": [],
        "premium_metadata": [],
        "blockers": blockers,
        "clarification": {
            "requires_clarification": bool(missing_fields or ambiguous_fields),
            "missing_fields": missing_fields,
            "ambiguous_fields": ambiguous_fields,
            "messages": [str(item["summary"]) for item in blockers],
        },
    }


def _persist_result(
    task,
    result,
    *,
    task_run_storage_root,
    task_run_storage_layout,
) -> None:
    from trellis.agent.task_run_store import persist_task_run_record

    try:
        persist_root = task_run_storage_root or ROOT
        persist_layout = (
            task_run_storage_layout if task_run_storage_root is not None else "repo"
        )
        persisted = persist_task_run_record(
            task,
            result,
            root=persist_root,
            storage_layout=persist_layout,
        )
        result["task_run_history_path"] = persisted["history_path"]
        result["task_run_latest_path"] = persisted["latest_path"]
        result["task_run_latest_index_path"] = persisted["latest_index_path"]
        result["task_diagnosis_persist_skipped"] = persisted.get(
            "diagnosis_persist_skipped"
        )
    except Exception as exc:  # pragma: no cover - persistence is best effort
        result["task_run_persist_error"] = str(exc)[:200]


def _summary_value(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{
                item.name: _summary_value(getattr(value, item.name))
                for item in fields(value)
                if item.name != "callable"
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _summary_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_summary_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _frequency(value: str):
    from trellis.core.types import Frequency

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "annual": Frequency.ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "semi_annual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported conformance frequency {value!r}") from exc


def _opposite_direction(direction: str) -> str:
    if direction == "pay":
        return "receive"
    if direction == "receive":
        return "pay"
    raise ValueError("leg direction must be 'pay' or 'receive'")


def _required_text(spec: Mapping[str, object], key: str) -> str:
    value = _optional_text(spec.get(key))
    if value is None:
        raise ValueError(f"native conformance contract requires {key}")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_float(spec: Mapping[str, object], key: str) -> float:
    if key not in spec or spec[key] is None:
        raise ValueError(f"native conformance contract requires {key}")
    return float(spec[key])


def _required_date(spec: Mapping[str, object], key: str) -> date:
    value = spec.get(key)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _required_text(spec, key)
    return date.fromisoformat(text)


__all__ = [
    "build_native_conformance_contract",
    "run_fpml_conformance_task",
]
