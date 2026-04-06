"""Governed MCP pricing orchestration over typed parse, match, policy, and ledger layers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from trellis.data.file_snapshot import (
    load_snapshot_from_record,
    manifest_warnings,
    serialize_market_snapshot,
)
from trellis.platform.audits import build_run_audit_bundle
from trellis.platform.context import ExecutionContext
from trellis.platform.models import evaluate_model_execution_gate
from trellis.platform.policies import evaluate_execution_policy
from trellis.platform.providers import (
    MockDataNotAllowedError,
    ProviderBindingRequiredError,
    ProviderResolutionError,
    resolve_governed_market_snapshot,
)
from trellis.platform.requests import CompiledPlatformRequest, ExecutionPlan, PlatformRequest
from trellis.platform.results import ExecutionResult
from trellis.platform.runs import build_run_record
from trellis.platform.storage import SnapshotRecord


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _new_request_id() -> str:
    return f"mcp_price_trade_{uuid4().hex[:12]}"


class PricingService:
    """Thin governed `trellis.price.trade` orchestration for approved-model execution."""

    def __init__(
        self,
        *,
        session_service,
        trade_service,
        model_service,
        provider_registry,
        model_registry,
        run_ledger,
        snapshot_store,
        audit_service,
    ):
        self.session_service = session_service
        self.trade_service = trade_service
        self.model_service = model_service
        self.provider_registry = provider_registry
        self.model_registry = model_registry
        self.run_ledger = run_ledger
        self.snapshot_store = snapshot_store
        self.audit_service = audit_service

    def price_trade(
        self,
        *,
        session_id: str | None = None,
        description: str | None = None,
        instrument_type: str | None = None,
        structured_trade: Mapping[str, object] | None = None,
        normalization_profile: str = "canonical",
        output_mode: str | None = None,
        valuation_date: str | date | None = None,
    ) -> dict[str, object]:
        """Run the narrow approved-model governed pricing flow for one MCP trade request."""
        session_record = self.session_service.ensure_record(session_id)
        request_id = _new_request_id()
        resolved_output_mode = str(output_mode or session_record.default_output_mode or "structured").strip() or "structured"
        execution_context = replace(
            session_record.execution_context(),
            default_output_mode=resolved_output_mode,
            requested_persistence="persisted",
            metadata={
                **dict(session_record.metadata),
                "mcp_tool": "trellis.price.trade",
            },
        )

        parsed = self.trade_service.parse_trade(
            description=description,
            instrument_type=instrument_type,
            structured_trade=structured_trade,
            normalization_profile=normalization_profile,
        )
        trade_identity = self._trade_identity(parsed)
        warnings = list(parsed.warnings)

        if parsed.parse_status != "parsed":
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                warnings=warnings,
                result_summary={
                    "reason": "trade_parse_incomplete",
                    "missing_fields": list(parsed.missing_fields),
                },
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        match = self.model_service.match_trade(parsed)
        selected_candidate = dict(match.selected_candidate)
        if match.match_type != "exact_approved_match" or not selected_candidate:
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_candidate,
                selected_engine=dict(selected_candidate.get("engine_binding", {})),
                warnings=warnings,
                result_summary={
                    "reason": "no_approved_model_match",
                    "match_type": match.match_type,
                    "selected_candidate": selected_candidate,
                    "candidate_count": len(match.candidates),
                },
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        gate = evaluate_model_execution_gate(
            registry=self.model_registry,
            model_id=str(selected_candidate.get("model_id", "")).strip(),
            execution_context=execution_context,
        )
        if not gate.allowed:
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=dict(gate.selected_model),
                selected_engine=dict(gate.selected_model.get("engine_binding", {})),
                warnings=warnings,
                result_summary={
                    "reason": "model_execution_not_allowed",
                    "rejection_codes": list(gate.rejection_codes),
                },
                validation_summary={"model_execution_gate": gate.to_dict()},
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        selected_model = dict(gate.selected_model)
        selected_engine = dict(selected_model.get("engine_binding", {}))
        market_snapshot = None
        active_snapshot_id = self._active_market_snapshot_id(session_record)
        if active_snapshot_id:
            try:
                market_snapshot = self._load_imported_market_snapshot(
                    active_snapshot_id,
                    valuation_date=valuation_date,
                )
            except ValueError as exc:
                run = self._persist_preexecution_run(
                    run_id=self._new_run_id(),
                    request_id=request_id,
                    status="blocked",
                    execution_context=execution_context,
                    trade_identity=trade_identity,
                    selected_model=selected_model,
                    selected_engine=selected_engine,
                    warnings=warnings,
                    result_summary={
                        "reason": "market_snapshot_unavailable",
                        "error": str(exc),
                        "market_snapshot_id": active_snapshot_id,
                    },
                    validation_summary={"model_execution_gate": gate.to_dict()},
                )
                return self._response_for_run(run, output_mode=resolved_output_mode)
        else:
            try:
                market_snapshot = resolve_governed_market_snapshot(
                    execution_context=execution_context,
                    as_of=valuation_date,
                    registry=self.provider_registry,
                )
            except (ProviderBindingRequiredError, MockDataNotAllowedError, ProviderResolutionError) as exc:
                policy_outcome = evaluate_execution_policy(
                    execution_context=execution_context,
                    selected_model=selected_model,
                ).to_dict()
                blocker_codes = list(policy_outcome.get("blocker_codes") or ())
                if isinstance(exc, ProviderBindingRequiredError) and "provider_binding_required" not in blocker_codes:
                    blocker_codes.append("provider_binding_required")
                if isinstance(exc, MockDataNotAllowedError) and "mock_data_not_allowed" not in blocker_codes:
                    blocker_codes.append("mock_data_not_allowed")
                if isinstance(exc, ProviderResolutionError) and "provider_resolution_failed" not in blocker_codes:
                    blocker_codes.append("provider_resolution_failed")
                run = self._persist_preexecution_run(
                    run_id=self._new_run_id(),
                    request_id=request_id,
                    status="blocked",
                    execution_context=execution_context,
                    trade_identity=trade_identity,
                    selected_model=selected_model,
                    selected_engine=selected_engine,
                    warnings=warnings,
                    result_summary={
                        "reason": "provider_resolution_blocked",
                        "error": str(exc),
                        "blocker_codes": blocker_codes,
                    },
                    validation_summary={"model_execution_gate": gate.to_dict()},
                    policy_outcome=policy_outcome,
                )
                return self._response_for_run(run, output_mode=resolved_output_mode)
        if market_snapshot is None:
            policy_outcome = evaluate_execution_policy(
                execution_context=execution_context,
                selected_model=selected_model,
            ).to_dict()
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_model,
                selected_engine=selected_engine,
                warnings=warnings,
                result_summary={
                    "reason": "market_snapshot_unavailable",
                },
                validation_summary={"model_execution_gate": gate.to_dict()},
                policy_outcome=policy_outcome,
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)
        warnings = list(
            dict.fromkeys(
                [
                    *warnings,
                    *tuple(market_snapshot.metadata.get("import_warnings", ()) or ()),
                ]
            )
        )
        settlement_date = self._resolved_settlement_date(
            valuation_date,
            default=market_snapshot.as_of,
        )

        policy_outcome = evaluate_execution_policy(
            execution_context=execution_context,
            market_snapshot=market_snapshot,
            selected_model=selected_model,
        ).to_dict()
        if not policy_outcome.get("allowed", False):
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_model,
                selected_engine=selected_engine,
                market_snapshot=market_snapshot,
                warnings=warnings,
                result_summary={
                    "reason": "policy_blocked",
                    "blocker_codes": list(policy_outcome.get("blocker_codes") or ()),
                },
                valuation_timestamp=settlement_date.isoformat(),
                validation_summary={"model_execution_gate": gate.to_dict()},
                policy_outcome=policy_outcome,
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        snapshot_record = self._persist_snapshot(market_snapshot)
        pricing_input = self._pricing_input(structured_trade, description)
        try:
            payoff, payoff_warnings = self._build_supported_payoff(
                selected_engine=selected_engine,
                parsed_trade=parsed,
                pricing_input=pricing_input,
                market_snapshot=market_snapshot,
            )
            warnings = list(dict.fromkeys([*warnings, *payoff_warnings]))
        except ValueError as exc:
            run = self._persist_preexecution_run(
                run_id=self._new_run_id(),
                request_id=request_id,
                status="blocked",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_model,
                selected_engine=selected_engine,
                market_snapshot=market_snapshot,
                warnings=warnings,
                result_summary={
                    "reason": "pricing_input_incomplete",
                    "error": str(exc),
                },
                valuation_timestamp=settlement_date.isoformat(),
                validation_summary={"model_execution_gate": gate.to_dict()},
                policy_outcome=policy_outcome,
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        compiled_request = self._compiled_pricing_request(
            request_id=request_id,
            parsed_trade=parsed,
            execution_context=execution_context,
            market_snapshot=market_snapshot,
            payoff=payoff,
            description=description,
            selected_model=selected_model,
            settlement=settlement_date,
        )
        from trellis.platform.executor import execute_compiled_request

        result = execute_compiled_request(
            compiled_request,
            execution_context,
            handlers={"price_trade": self._price_trade_handler},
        )
        run = self._persist_execution_result(
            result,
            execution_context=execution_context,
            trade_identity=trade_identity,
            selected_model=selected_model,
            selected_engine=selected_engine,
            market_snapshot_id=snapshot_record.snapshot_id,
            valuation_timestamp=result.provenance.get("valuation_timestamp") or snapshot_record.as_of,
            warnings=tuple(dict.fromkeys([*warnings, *result.warnings])),
            validation_summary={"model_execution_gate": gate.to_dict()},
            policy_outcome=policy_outcome,
        )
        return self._response_for_run(run, output_mode=resolved_output_mode)

    @staticmethod
    def _new_run_id() -> str:
        return f"run_{uuid4().hex[:12]}"

    @staticmethod
    def _trade_identity(parsed_trade) -> dict[str, object]:
        payload = parsed_trade.to_dict()
        parsed_contract = payload.pop("parsed_contract", {})
        return {
            **parsed_contract,
            "parse_status": parsed_trade.parse_status,
            "semantic_id": parsed_trade.semantic_id,
            "semantic_version": parsed_trade.semantic_version,
            "trade_type": parsed_trade.trade_type,
        }

    @staticmethod
    def _pricing_input(structured_trade, description: str | None) -> dict[str, object]:
        payload = dict(structured_trade or {})
        if description and "description" not in payload:
            payload["description"] = description
        return payload

    @staticmethod
    def _compiled_pricing_request(
        *,
        request_id: str,
        parsed_trade,
        execution_context: ExecutionContext,
        market_snapshot,
        payoff,
        description: str | None,
        selected_model: Mapping[str, object],
        settlement: date,
    ) -> CompiledPlatformRequest:
        request = PlatformRequest(
            request_id=request_id,
            request_type="price",
            entry_point="mcp",
            settlement=settlement,
            market_snapshot=market_snapshot,
            description=description or str(parsed_trade.parsed_contract.get("description", "")).strip() or None,
            instrument_type=parsed_trade.trade_type,
            measures=("price",),
            instrument=payoff,
            metadata={
                "selected_model": dict(selected_model),
                "discount_curve_name": market_snapshot.default_discount_curve,
                "vol_surface_name": market_snapshot.default_vol_surface,
            },
        )
        route_method = str(
            selected_model.get("methodology_summary", {}).get("method_family")
            or (parsed_trade.candidate_methods[0] if parsed_trade.candidate_methods else "")
        ).strip()
        return CompiledPlatformRequest(
            request=request,
            market_snapshot=market_snapshot,
            execution_plan=ExecutionPlan(
                action="price_trade",
                reason="approved_model_mcp_trade",
                requested_outputs=("price",),
                route_method=route_method or "approved_model",
                requires_build=False,
            ),
            semantic_contract=parsed_trade.semantic_contract,
            semantic_blueprint=parsed_trade.semantic_blueprint,
            product_ir=parsed_trade.product_ir_object,
        )

    @staticmethod
    def _price_trade_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
        from trellis.analytics.measures import CallableScenarioExplain, OASDuration
        from trellis.engine.payoff_pricer import price_payoff
        from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
        from trellis.models.bermudan_swaption_tree import (
            BermudanSwaptionTreeSpec,
            price_bermudan_swaption_tree,
        )
        from trellis.models.callable_bond_tree import (
            price_callable_bond_tree,
            straight_bond_present_value,
        )
        from trellis.models.range_accrual import RangeAccrualSpec, price_range_accrual

        payoff = compiled_request.request.instrument
        if payoff is None:
            raise ValueError("Compiled trade request has no payoff adapter")
        market_snapshot = compiled_request.market_snapshot
        if market_snapshot is None:
            raise ValueError("Compiled trade request has no market snapshot")
        selected_model = dict(compiled_request.request.metadata.get("selected_model", {}) or {})
        adapter_id = str(selected_model.get("engine_binding", {}).get("adapter_id", "")).strip()
        settlement = compiled_request.request.settlement or market_snapshot.as_of
        if adapter_id == "callable_bond_tree" or isinstance(payoff, CallableBondSpec):
            spec = payoff if isinstance(payoff, CallableBondSpec) else None
            if spec is None:
                raise ValueError("Callable bond adapter requires a CallableBondSpec instrument.")
            callable_payoff = CallableBondPayoff(spec)
            market_state = market_snapshot.to_market_state(
                settlement=settlement,
                discount_curve=market_snapshot.default_discount_curve,
                vol_surface=market_snapshot.default_vol_surface,
            )
            price = float(price_callable_bond_tree(market_state, spec, model="hull_white"))
            straight_price = float(
                straight_bond_present_value(
                    market_state,
                    spec,
                    settlement=market_state.settlement,
                )
            )
            analytics_context = {"_cache": {"base_price": price}}
            oas_duration = float(
                OASDuration(bump_bps=25.0).compute(
                    callable_payoff,
                    market_state,
                    **analytics_context,
                )
            )
            callable_scenario_explain = CallableScenarioExplain().compute(
                callable_payoff,
                market_state,
                **analytics_context,
            )
            analytics_assumptions = [
                "OAS duration uses effective-duration style repricing with constant OAS at the current callable-tree price.",
                *list(callable_scenario_explain.metadata.get("assumptions") or ()),
            ]
            return {
                "status": "succeeded",
                "result_payload": {
                    "price": price,
                    "straight_bond_price": straight_price,
                    "call_option_value": max(straight_price - price, 0.0),
                    "oas_duration": oas_duration,
                    "callable_scenario_explain": callable_scenario_explain.to_payload(),
                    "exercise_dates": [exercise_date.isoformat() for exercise_date in spec.call_dates],
                    "schedule_role": "decision_dates",
                    "projected_events": PricingService._project_callable_bond_events(spec),
                    "validation_bundle": PricingService._callable_bond_validation_bundle(
                        price=price,
                        straight_price=straight_price,
                        exercise_dates=spec.call_dates,
                        assumptions=analytics_assumptions,
                    ),
                },
                "warnings": (),
                "provenance": {
                    "engine_id": str(
                        selected_model.get("engine_binding", {}).get("engine_id", "")
                    ).strip(),
                },
                "audit_summary": {
                    "adapter": "callable_bond_tree",
                    "route_id": "callable_bond_tree_v1",
                },
            }
        if adapter_id == "bermudan_swaption_tree" or isinstance(payoff, BermudanSwaptionTreeSpec):
            spec = payoff if isinstance(payoff, BermudanSwaptionTreeSpec) else None
            if spec is None:
                raise ValueError("Bermudan swaption adapter requires a BermudanSwaptionTreeSpec instrument.")
            market_state = market_snapshot.to_market_state(
                settlement=settlement,
                discount_curve=market_snapshot.default_discount_curve,
                vol_surface=market_snapshot.default_vol_surface,
            )
            price = float(price_bermudan_swaption_tree(market_state, spec, model="hull_white"))
            return {
                "status": "succeeded",
                "result_payload": {
                    "price": price,
                    "exercise_dates": [exercise_date.isoformat() for exercise_date in spec.exercise_dates],
                    "schedule_role": "exercise_dates",
                    "projected_events": PricingService._project_bermudan_swaption_events(spec),
                    "validation_bundle": PricingService._bermudan_swaption_validation_bundle(
                        price=price,
                        exercise_dates=spec.exercise_dates,
                        swap_end=spec.swap_end,
                    ),
                },
                "warnings": (),
                "provenance": {
                    "engine_id": str(
                        selected_model.get("engine_binding", {}).get("engine_id", "")
                    ).strip(),
                },
                "audit_summary": {
                    "adapter": "bermudan_swaption_tree",
                    "route_id": "bermudan_swaption_tree_v1",
                },
            }
        if adapter_id == "range_accrual_discounted" or isinstance(payoff, RangeAccrualSpec):
            spec = payoff if isinstance(payoff, RangeAccrualSpec) else None
            if spec is None:
                raise ValueError("Range accrual adapter requires a RangeAccrualSpec instrument.")
            forecast_curve, assumptions = PricingService._resolve_range_accrual_forecast_curve(
                market_snapshot,
                reference_index=spec.reference_index,
            )
            fixing_history = PricingService._resolve_range_accrual_fixing_history(
                market_snapshot,
                reference_index=spec.reference_index,
            )
            result = price_range_accrual(
                spec,
                as_of=settlement,
                discount_curve=market_snapshot.discount_curve(),
                forecast_curve=forecast_curve,
                fixing_history=fixing_history,
                assumptions=assumptions,
            )
            return {
                "status": "succeeded",
                "result_payload": result.to_dict(),
                "warnings": assumptions,
                "provenance": {
                    "engine_id": str(
                        selected_model.get("engine_binding", {}).get("engine_id", "")
                    ).strip(),
                },
                "audit_summary": {
                    "adapter": "range_accrual_discounted",
                    "route_id": result.validation_bundle.route_id,
                },
            }
        market_state = market_snapshot.to_market_state(
            settlement=compiled_request.request.settlement or market_snapshot.as_of,
            discount_curve=market_snapshot.default_discount_curve,
            vol_surface=market_snapshot.default_vol_surface,
            underlier_spot=market_snapshot.default_underlier_spot,
        )
        price = float(price_payoff(payoff, market_state))
        return {
            "status": "succeeded",
            "result_payload": {"price": price},
            "provenance": {
                "engine_id": str(
                    compiled_request.request.metadata.get("selected_model", {}).get("engine_binding", {}).get("engine_id", "")
                ).strip(),
            },
            "audit_summary": {
                "adapter": "approved_model_payoff",
            },
        }

    def _persist_execution_result(
        self,
        result: ExecutionResult,
        *,
        execution_context: ExecutionContext,
        trade_identity: Mapping[str, object],
        selected_model: Mapping[str, object],
        selected_engine: Mapping[str, object],
        market_snapshot_id: str,
        valuation_timestamp: str,
        warnings,
        validation_summary: Mapping[str, object],
        policy_outcome: Mapping[str, object],
    ):
        run = self.run_ledger.save_run(
            build_run_record(
                run_id=result.run_id,
                request_id=result.request_id,
                status=result.status,
                action="price_trade",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_model,
                selected_engine=selected_engine,
                market_snapshot_id=market_snapshot_id,
                valuation_timestamp=str(valuation_timestamp or "").strip(),
                warnings=warnings,
                result_summary=dict(result.result_payload),
                provenance={
                    **dict(result.provenance),
                    "engine_id": str(selected_engine.get("engine_id", "")).strip(),
                    "engine_version": str(selected_engine.get("version", "")).strip(),
                    "model_id": str(selected_model.get("model_id", "")).strip(),
                    "model_version": str(selected_model.get("version", "")).strip(),
                    "model_status": str(selected_model.get("status", "")).strip(),
                },
                validation_summary=validation_summary,
                policy_outcome=policy_outcome,
                artifacts=result.artifacts,
            )
        )
        return run

    def _persist_preexecution_run(
        self,
        *,
        run_id: str,
        request_id: str,
        status: str,
        execution_context: ExecutionContext,
        trade_identity: Mapping[str, object],
        warnings,
        result_summary: Mapping[str, object],
        selected_model: Mapping[str, object] | None = None,
        selected_engine: Mapping[str, object] | None = None,
        market_snapshot=None,
        valuation_timestamp: str | None = None,
        validation_summary: Mapping[str, object] | None = None,
        policy_outcome: Mapping[str, object] | None = None,
    ):
        return self.run_ledger.save_run(
            build_run_record(
                run_id=run_id,
                request_id=request_id,
                status=status,
                action="price_trade",
                execution_context=execution_context,
                trade_identity=trade_identity,
                selected_model=selected_model or {},
                selected_engine=selected_engine or {},
                market_snapshot_id=(
                    "" if market_snapshot is None else str(getattr(market_snapshot, "market_snapshot_id", "")).strip()
                ),
                valuation_timestamp=(
                    str(valuation_timestamp or "").strip()
                    or (
                        ""
                        if market_snapshot is None
                        else str(getattr(market_snapshot, "as_of", "")).strip()
                    )
                ),
                warnings=warnings,
                result_summary=result_summary,
                provenance={
                    "provider_id": str(
                        execution_context.provider_bindings.market_data.primary.provider_id
                    ).strip()
                    if execution_context.provider_bindings.market_data.primary is not None
                    else "",
                    "engine_id": str((selected_engine or {}).get("engine_id", "")).strip(),
                    "engine_version": str((selected_engine or {}).get("version", "")).strip(),
                    "model_id": str((selected_model or {}).get("model_id", "")).strip(),
                    "model_version": str((selected_model or {}).get("version", "")).strip(),
                    "model_status": str((selected_model or {}).get("status", "")).strip(),
                },
                validation_summary=validation_summary or {},
                policy_outcome=policy_outcome or {},
            )
        )

    def _persist_snapshot(self, market_snapshot) -> SnapshotRecord:
        existing_id = str(getattr(market_snapshot, "market_snapshot_id", "")).strip()
        if existing_id:
            existing = self.snapshot_store.get_snapshot(existing_id)
            if existing is not None and self._has_rehydratable_snapshot_payload(existing):
                return existing
        record = SnapshotRecord(
            snapshot_id=str(getattr(market_snapshot, "market_snapshot_id", "")).strip(),
            provider_id=str(getattr(market_snapshot, "provider_id", "")).strip(),
            as_of=market_snapshot.as_of.isoformat(),
            source=str(getattr(market_snapshot, "source", "")).strip(),
            payload={
                "bundle_type": "governed_market_snapshot",
                "snapshot_contract": serialize_market_snapshot(market_snapshot),
            },
            provenance=dict(market_snapshot.provenance),
        )
        return self.snapshot_store.save_snapshot(record)

    @staticmethod
    def _active_market_snapshot_id(session_record) -> str:
        return str(session_record.metadata.get("active_market_snapshot_id", "")).strip()

    @staticmethod
    def _has_rehydratable_snapshot_payload(record) -> bool:
        payload = dict(getattr(record, "payload", {}) or {})
        return bool(
            payload.get("manifest")
            or payload.get("snapshot_contract")
            or (
                str(payload.get("bundle_type", "")).strip() == "reproducibility_bundle"
                and payload.get("market_snapshot")
            )
        )

    @staticmethod
    def _resolved_settlement_date(
        valuation_date: str | date | None,
        *,
        default: date,
    ) -> date:
        if isinstance(valuation_date, date):
            return valuation_date
        text = str(valuation_date or "").strip()
        if not text:
            return default
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("valuation_date must be an ISO date.") from exc

    def _load_imported_market_snapshot(
        self,
        snapshot_id: str,
        *,
        valuation_date: str | date | None = None,
    ):
        record = self.snapshot_store.get_snapshot(str(snapshot_id or "").strip())
        if record is None:
            raise ValueError(f"Unknown imported market snapshot id: {snapshot_id!r}")
        market_snapshot = load_snapshot_from_record(record)
        warnings = manifest_warnings(
            dict(record.payload.get("manifest") or {}),
            reference_date=valuation_date or market_snapshot.as_of,
        )
        market_snapshot_warnings = tuple(dict.fromkeys([*warnings]))
        return replace(
            market_snapshot,
            metadata={
                **dict(market_snapshot.metadata),
                "import_warnings": market_snapshot_warnings,
            },
        )

    def _response_for_run(self, run, *, output_mode: str) -> dict[str, object]:
        audit_uri = f"trellis://runs/{run.run_id}/audit"
        desk_review = self._desk_review_bundle(run, audit_uri=audit_uri)
        structured = {
            "run_id": run.run_id,
            "status": run.status,
            "result": dict(run.result_summary),
            "warnings": list(run.warnings),
            "provenance": self._response_provenance(run),
            "audit_uri": audit_uri,
            "audit_reference": {
                "tool_name": "trellis.run.get_audit",
                "run_id": run.run_id,
            },
            "desk_review": desk_review,
        }
        normalized_mode = str(output_mode or "structured").strip().lower()
        if normalized_mode == "concise":
            return {
                "status": run.status,
                "result": dict(run.result_summary),
                "warnings": list(run.warnings),
                "provenance": {
                    "run_id": run.run_id,
                    "audit_uri": audit_uri,
                },
            }
        if normalized_mode == "audit":
            structured["audit"] = build_run_audit_bundle(run).to_dict()
            structured["execution_stages"] = [
                "parse",
                "match",
                "policy",
                "execute" if run.status == "succeeded" else "blocked",
                "persist",
            ]
        return structured

    @staticmethod
    def _response_provenance(run) -> dict[str, object]:
        return {
            "model_id": str(run.selected_model.get("model_id", "")).strip(),
            "model_version": str(run.selected_model.get("version", "")).strip(),
            "model_status": str(run.selected_model.get("status", "")).strip(),
            "engine_id": str(run.selected_engine.get("engine_id", "")).strip(),
            "engine_version": str(run.selected_engine.get("version", "")).strip(),
            "provider_id": str(run.provenance.get("provider_id", "")).strip(),
            "market_snapshot_id": run.market_snapshot_id,
            "valuation_timestamp": run.valuation_timestamp,
            "policy_id": run.policy_id,
            "run_mode": run.run_mode,
        }

    @staticmethod
    def _desk_review_bundle(run, *, audit_uri: str) -> dict[str, object]:
        result = dict(run.result_summary)
        validation_bundle = dict(result.get("validation_bundle") or {})
        trade_summary = PricingService._trade_summary(run.trade_identity)
        route_summary = PricingService._route_summary(run)
        assumptions = PricingService._assumptions_summary(run, validation_bundle)
        warning_pack = PricingService._warning_pack(run.warnings)
        schedule_summary = PricingService._schedule_summary(run.trade_identity, result=result)
        scenario_summary = PricingService._scenario_summary(result)
        return {
            "trade_summary": trade_summary,
            "route_summary": route_summary,
            "assumptions": assumptions,
            "warning_pack": warning_pack,
            "schedule_summary": schedule_summary,
            "scenario_summary": scenario_summary,
            "driver_narrative": PricingService._driver_narrative(
                trade_summary=trade_summary,
                route_summary=route_summary,
                assumptions=assumptions,
                warning_pack=warning_pack,
                schedule_summary=schedule_summary,
                result=result,
                run_status=run.status,
            ),
            "scenario_commentary": PricingService._scenario_commentary(
                trade_summary=trade_summary,
                route_summary=route_summary,
                assumptions=assumptions,
                warning_pack=warning_pack,
                scenario_summary=scenario_summary,
                result=result,
            ),
            "audit_refs": PricingService._audit_refs(run, audit_uri=audit_uri),
        }

    @staticmethod
    def _trade_summary(trade_identity: Mapping[str, object]) -> dict[str, object]:
        product = dict(trade_identity.get("product") or {})
        term_fields = dict(product.get("term_fields") or {})
        return {
            "semantic_id": str(trade_identity.get("semantic_id", "")).strip(),
            "instrument_class": str(trade_identity.get("trade_type", "")).strip()
            or str(product.get("instrument_class", "")).strip(),
            "exercise_style": str(product.get("exercise_style", "")).strip(),
            "reference_index": str(term_fields.get("reference_index", "")).strip(),
            "coupon_definition": dict(term_fields.get("coupon_definition") or {}),
            "range_condition": dict(term_fields.get("range_condition") or {}),
            "term_fields": term_fields,
        }

    @staticmethod
    def _route_summary(run) -> dict[str, object]:
        selected_model = dict(run.selected_model)
        methodology_summary = dict(selected_model.get("methodology_summary", {}) or {})
        return {
            "method_family": str(methodology_summary.get("method_family", "")).strip()
            or str(run.provenance.get("method_family", "")).strip(),
            "adapter_id": str(run.selected_engine.get("adapter_id", "")).strip(),
            "engine_id": str(run.selected_engine.get("engine_id", "")).strip(),
            "model_id": str(selected_model.get("model_id", "")).strip(),
            "model_version": str(selected_model.get("version", "")).strip(),
            "model_status": str(selected_model.get("status", "")).strip(),
        }

    @staticmethod
    def _assumptions_summary(run, validation_bundle: Mapping[str, object]) -> dict[str, object]:
        assumptions = {
            "explicit_inputs": [],
            "defaulted_inputs": [],
            "synthetic_inputs": [],
            "missing_inputs": [],
        }
        if str(run.provenance.get("provider_id", "")).strip() == "market_data.file_import":
            assumptions["explicit_inputs"].append(
                "Market data came from the active explicit imported market snapshot."
            )

        notes = list(validation_bundle.get("assumptions") or []) + list(run.warnings)
        for note in dict.fromkeys(str(item).strip() for item in notes if str(item).strip()):
            lowered = note.lower()
            if "defaulted" in lowered:
                assumptions["defaulted_inputs"].append(note)
            elif any(token in lowered for token in ("proxy", "synthetic", "inferred")):
                assumptions["synthetic_inputs"].append(note)
            elif "missing" in lowered:
                assumptions["missing_inputs"].append(note)
            elif "explicit" in lowered:
                assumptions["explicit_inputs"].append(note)
        return assumptions

    @staticmethod
    def _warning_pack(warnings) -> dict[str, object]:
        items = []
        for warning in dict.fromkeys(str(item).strip() for item in warnings if str(item).strip()):
            lowered = warning.lower()
            if "defaulted" in lowered:
                category = "defaulted_input"
            elif any(token in lowered for token in ("proxy", "synthetic", "inferred")):
                category = "synthetic_input"
            elif "missing" in lowered:
                category = "missing_input"
            else:
                category = "general"
            items.append(
                {
                    "category": category,
                    "message": warning,
                }
            )
        return {
            "count": len(items),
            "items": items,
        }

    @staticmethod
    def _schedule_summary(trade_identity: Mapping[str, object], *, result: Mapping[str, object]) -> dict[str, object]:
        product = dict(trade_identity.get("product") or {})
        term_fields = dict(product.get("term_fields") or {})
        callability = dict(term_fields.get("callability") or {})
        observation_dates = list(product.get("observation_schedule") or [])
        call_dates = list(callability.get("call_schedule") or [])
        exercise_style = str(product.get("exercise_style", "")).strip()
        exercise_dates = list(
            result.get("exercise_dates")
            or term_fields.get("exercise_schedule")
            or (
                observation_dates
                if exercise_style in {"issuer_call", "bermudan", "european"}
                else ()
            )
        )
        projected_events = list(result.get("projected_events") or [])
        event_dates = list(
            dict.fromkeys(
                [
                    *observation_dates,
                    *call_dates,
                    *exercise_dates,
                ]
            )
        )
        schedule_role = str(result.get("schedule_role", "")).strip()
        if not schedule_role:
            if exercise_style == "issuer_call":
                schedule_role = "decision_dates"
            elif exercise_style in {"bermudan", "european"}:
                schedule_role = "exercise_dates"
            else:
                schedule_role = "observation_dates"
        return {
            "observation_dates": observation_dates,
            "observation_count": len(observation_dates),
            "call_dates": call_dates,
            "call_event_count": len(call_dates),
            "exercise_dates": exercise_dates,
            "exercise_event_count": len(exercise_dates),
            "schedule_role": schedule_role,
            "projected_events": projected_events,
            "event_count": len(projected_events) or len(event_dates),
        }

    @staticmethod
    def _scenario_summary(result: Mapping[str, object]) -> dict[str, object]:
        callable_explain = dict(result.get("callable_scenario_explain") or {})
        if callable_explain:
            values = dict(callable_explain.get("values") or {})
            return {
                "scenario_count": len(values),
                "ladder": [
                    {
                        "scenario": str(name),
                        **dict(payload or {}),
                    }
                    for name, payload in values.items()
                ],
                "base_price": result.get("price"),
            }
        scenarios = list(result.get("scenarios") or [])
        return {
            "scenario_count": len(scenarios),
            "ladder": scenarios,
            "base_price": result.get("price"),
        }

    @staticmethod
    def _driver_narrative(
        *,
        trade_summary: Mapping[str, object],
        route_summary: Mapping[str, object],
        assumptions: Mapping[str, object],
        warning_pack: Mapping[str, object],
        schedule_summary: Mapping[str, object],
        result: Mapping[str, object],
        run_status: str,
    ) -> dict[str, object]:
        if str(run_status or "").strip().lower() != "succeeded":
            return PricingService._blocked_driver_narrative(
                route_summary=route_summary,
                assumptions=assumptions,
                warning_pack=warning_pack,
                result=result,
            )

        semantic_id = str(trade_summary.get("semantic_id", "")).strip()
        instrument_class = str(trade_summary.get("instrument_class", "")).strip()
        adapter_id = str(route_summary.get("adapter_id", "")).strip() or "selected_route"
        method_family = str(route_summary.get("method_family", "")).strip() or "selected"
        linked_assumptions = PricingService._linked_assumptions(assumptions)
        linked_warnings = PricingService._linked_warning_items(warning_pack)
        bullets: list[str] = []

        if semantic_id == "range_accrual":
            range_condition = dict(trade_summary.get("range_condition") or {})
            reference_index = str(trade_summary.get("reference_index", "")).strip()
            lower_bound = range_condition.get("lower_bound")
            upper_bound = range_condition.get("upper_bound")
            headline = (
                "Range accrual PV is driven by discounted coupon accrual inside the configured rate range plus principal redemption."
            )
            bullets.append(
                f"Route {adapter_id} used the {method_family} family and priced "
                f"{int(result.get('observed_coupon_count') or 0)} observed plus "
                f"{int(result.get('projected_coupon_count') or 0)} projected coupon periods."
            )
            bullets.append(
                "Coupon leg contributes "
                f"{PricingService._format_number(result.get('coupon_leg_pv'))} and principal leg contributes "
                f"{PricingService._format_number(result.get('principal_leg_pv'))}."
            )
            if reference_index and lower_bound is not None and upper_bound is not None:
                bullets.append(
                    f"{reference_index} coupons accrue only when fixings stay between "
                    f"{PricingService._format_percent(lower_bound)} and "
                    f"{PricingService._format_percent(upper_bound)}."
                )
        elif semantic_id == "callable_bond":
            explain_metadata = dict(
                dict(result.get("callable_scenario_explain") or {}).get("metadata") or {}
            )
            headline = (
                "Callable-bond PV reflects straight-bond value reduced by issuer call optionality on the decision schedule."
            )
            bullets.append(
                f"Route {adapter_id} used the {method_family} family across "
                f"{int(schedule_summary.get('exercise_event_count') or 0)} issuer decision dates."
            )
            base_straight = explain_metadata.get("base_straight_bond_price")
            base_option = explain_metadata.get("base_call_option_value")
            if base_straight is not None and base_option is not None:
                bullets.append(
                    "Base callable price is "
                    f"{PricingService._format_number(result.get('price'))} versus straight-bond reference "
                    f"{PricingService._format_number(base_straight)}, implying embedded call option value "
                    f"{PricingService._format_number(base_option)}."
                )
        elif semantic_id == "bermudan_swaption" or instrument_class == "bermudan_swaption":
            option_side = "payer" if bool(dict(trade_summary.get("term_fields") or {}).get("is_payer", True)) else "receiver"
            headline = (
                "Bermudan swaption PV is driven by the right to enter the underlying swap across the exercise schedule."
            )
            bullets.append(
                f"Route {adapter_id} used the {method_family} family across "
                f"{int(schedule_summary.get('exercise_event_count') or 0)} {option_side} exercise dates."
            )
            bullets.append(
                f"Reported option PV is {PricingService._format_number(result.get('price'))} "
                "and exercise remains bounded by the underlying swap maturity schedule."
            )
        else:
            label = instrument_class or semantic_id or "trade"
            headline = (
                f"{label.replace('_', ' ').title()} PV is explained by the approved route, the configured schedule, and the linked assumptions."
            )
            bullets.append(
                f"Route {adapter_id} used the {method_family} family for the approved pricing path."
            )
            if "price" in result:
                bullets.append(
                    f"Reported price is {PricingService._format_number(result.get('price'))}."
                )

        context_bullet = PricingService._linked_context_bullet(
            linked_assumptions,
            linked_warnings,
        )
        if context_bullet:
            bullets.append(context_bullet)

        return {
            "headline": headline,
            "bullets": bullets,
            "linked_route": dict(route_summary),
            "linked_assumptions": linked_assumptions,
            "linked_warnings": linked_warnings,
        }

    @staticmethod
    def _blocked_driver_narrative(
        *,
        route_summary: Mapping[str, object],
        assumptions: Mapping[str, object],
        warning_pack: Mapping[str, object],
        result: Mapping[str, object],
    ) -> dict[str, object]:
        reason = str(result.get("reason", "")).strip()
        linked_assumptions = PricingService._linked_assumptions(assumptions)
        linked_warnings = PricingService._linked_warning_items(warning_pack)
        blocker_codes = [str(code).strip() for code in result.get("blocker_codes") or () if str(code).strip()]
        missing_fields = [str(field).strip() for field in result.get("missing_fields") or () if str(field).strip()]
        error_text = str(result.get("error", "")).strip()
        bullets: list[str] = []

        if reason == "trade_parse_incomplete":
            headline = "Pricing is blocked because the trade request is incomplete."
            if missing_fields:
                bullets.append(
                    "Pricing stopped before route selection because required trade fields are missing: "
                    + ", ".join(missing_fields)
                    + "."
                )
            else:
                bullets.append(
                    "Pricing stopped before route selection because typed trade normalization did not complete."
                )
        elif reason == "no_approved_model_match":
            headline = "Pricing is blocked because no approved governed model matched the trade."
            bullets.append(
                "Pricing stopped before pricing execution because no approved model/route was eligible to run."
            )
            match_type = str(result.get("match_type", "")).strip()
            if match_type:
                bullets.append(f"Deterministic match result: {match_type}.")
        elif reason == "model_execution_not_allowed":
            headline = "Pricing is blocked because the matched model is not execution-eligible."
            bullets.append(
                "Pricing stopped before pricing execution because the model execution gate rejected the selected version."
            )
        elif reason == "provider_resolution_blocked":
            headline = "Pricing is blocked because governed market data could not be resolved."
            bullets.append(
                "Pricing stopped before pricing execution because required provider bindings or market-data resolution failed."
            )
        elif reason == "policy_blocked":
            headline = "Pricing is blocked by governed execution policy."
            bullets.append(
                "Pricing stopped before pricing execution because policy blockers prevented the selected route from running."
            )
        elif reason == "market_snapshot_unavailable":
            headline = "Pricing is blocked because the required market snapshot is unavailable."
            bullets.append(
                "Pricing stopped before pricing execution because the requested market snapshot could not be loaded."
            )
        elif reason == "pricing_input_incomplete":
            headline = "Pricing is blocked because the selected adapter input is incomplete."
            bullets.append(
                "Pricing stopped before pricing execution because the approved adapter could not be built from the structured trade input."
            )
        else:
            headline = "Pricing is blocked before execution."
            bullets.append(
                "This run did not reach a successful approved pricing path."
            )

        if blocker_codes:
            bullets.append(f"Blocker codes: {', '.join(blocker_codes)}.")
        if error_text:
            bullets.append(error_text)

        context_bullet = PricingService._linked_context_bullet(
            linked_assumptions,
            linked_warnings,
        )
        if context_bullet:
            bullets.append(context_bullet)

        return {
            "headline": headline,
            "bullets": bullets,
            "linked_route": dict(route_summary),
            "linked_assumptions": linked_assumptions,
            "linked_warnings": linked_warnings,
        }

    @staticmethod
    def _scenario_commentary(
        *,
        trade_summary: Mapping[str, object],
        route_summary: Mapping[str, object],
        assumptions: Mapping[str, object],
        warning_pack: Mapping[str, object],
        scenario_summary: Mapping[str, object],
        result: Mapping[str, object],
    ) -> dict[str, object]:
        semantic_id = str(trade_summary.get("semantic_id", "")).strip()
        linked_assumptions = PricingService._linked_assumptions(assumptions)
        linked_warnings = PricingService._linked_warning_items(warning_pack)
        ladder = list(scenario_summary.get("ladder") or [])
        scenario_count = int(scenario_summary.get("scenario_count") or len(ladder))
        base_price = scenario_summary.get("base_price")
        dominant = PricingService._dominant_scenario(ladder, base_price=base_price)

        if not ladder:
            return {
                "headline": "No route-specific scenario ladder is attached to this trade yet.",
                "bullets": [
                    "Use the linked schedule, assumptions, and warnings before extrapolating scenario moves from this route."
                ],
                "availability": "unavailable",
                "scenario_count": scenario_count,
                "dominant_scenario": None,
                "linked_route": dict(route_summary),
                "linked_assumptions": linked_assumptions,
                "linked_warnings": linked_warnings,
            }

        if semantic_id == "callable_bond":
            headline = (
                "Rates-down scenarios lift the straight-bond reference, but call optionality caps part of that upside."
            )
        else:
            headline = PricingService._generic_scenario_headline(
                ladder,
                base_price=base_price,
            )

        bullets: list[str] = []
        if dominant is not None:
            shift = dominant.get("shift_bps")
            shift_label = (
                f"{float(shift):+g} bp"
                if isinstance(shift, (int, float))
                else str(dominant.get("scenario", "")).strip() or "the dominant scenario"
            )
            bullets.append(
                f"Largest modeled move is {shift_label}: scenario price "
                f"{PricingService._format_number(dominant.get('price'))} with P&L "
                f"{PricingService._format_signed_number(dominant.get('pnl'))}."
            )
            if semantic_id == "callable_bond" and dominant.get("call_option_value_change") is not None:
                bullets.append(
                    "Embedded call optionality changes by "
                    f"{PricingService._format_signed_number(dominant.get('call_option_value_change'))} "
                    "in that scenario."
                )
        bullets.append(
            f"The route returned {scenario_count} scenario point"
            f"{'' if scenario_count == 1 else 's'} around base price "
            f"{PricingService._format_number(base_price)}."
        )
        context_bullet = PricingService._linked_context_bullet(
            linked_assumptions,
            linked_warnings,
        )
        if context_bullet:
            bullets.append(context_bullet)

        return {
            "headline": headline,
            "bullets": bullets,
            "availability": "available",
            "scenario_count": scenario_count,
            "dominant_scenario": dominant,
            "linked_route": dict(route_summary),
            "linked_assumptions": linked_assumptions,
            "linked_warnings": linked_warnings,
        }

    @staticmethod
    def _linked_assumptions(assumptions: Mapping[str, object]) -> dict[str, list[str]]:
        return {
            str(category): [str(item) for item in items]
            for category, items in dict(assumptions).items()
            if items
        }

    @staticmethod
    def _linked_warning_items(warning_pack: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            dict(item)
            for item in list(warning_pack.get("items") or [])
        ]

    @staticmethod
    def _linked_context_bullet(
        linked_assumptions: Mapping[str, object],
        linked_warnings,
    ) -> str:
        tokens: list[str] = []
        if linked_assumptions.get("synthetic_inputs"):
            tokens.append("synthetic/proxy inputs")
        if linked_assumptions.get("defaulted_inputs"):
            tokens.append("defaulted inputs")
        if linked_assumptions.get("missing_inputs"):
            tokens.append("missing inputs")
        warning_count = len(linked_warnings or [])
        if warning_count:
            tokens.append(f"{warning_count} linked warning item{'s' if warning_count != 1 else ''}")
        if not tokens:
            return ""
        if len(tokens) == 1:
            detail = tokens[0]
        else:
            detail = ", ".join(tokens[:-1]) + f", and {tokens[-1]}"
        return f"Review the linked {detail} before using the output as a desk talking point."

    @staticmethod
    def _dominant_scenario(ladder, *, base_price) -> dict[str, object] | None:
        dominant = None
        dominant_magnitude = -1.0
        for raw_entry in ladder:
            entry = dict(raw_entry or {})
            pnl = PricingService._scenario_pnl(entry, base_price=base_price)
            if pnl is None:
                continue
            magnitude = abs(float(pnl))
            if magnitude <= dominant_magnitude:
                continue
            dominant_magnitude = magnitude
            dominant = {
                **entry,
                "scenario": str(entry.get("scenario") or entry.get("name") or "").strip(),
                "shift_bps": PricingService._scenario_shift(entry),
                "pnl": float(pnl),
                "price": (
                    None
                    if not isinstance(entry.get("price"), (int, float))
                    else float(entry.get("price"))
                ),
            }
        return dominant

    @staticmethod
    def _generic_scenario_headline(ladder, *, base_price) -> str:
        down_pnls = []
        up_pnls = []
        for raw_entry in ladder:
            entry = dict(raw_entry or {})
            shift = PricingService._scenario_shift(entry)
            pnl = PricingService._scenario_pnl(entry, base_price=base_price)
            if shift is None or pnl is None:
                continue
            if shift < 0:
                down_pnls.append(float(pnl))
            elif shift > 0:
                up_pnls.append(float(pnl))
        if down_pnls and up_pnls:
            avg_down = sum(down_pnls) / len(down_pnls)
            avg_up = sum(up_pnls) / len(up_pnls)
            if avg_down > 0.0 and avg_up < 0.0:
                return (
                    "Rates-down scenarios lift PV while rates-up scenarios reduce PV, consistent with a long-duration profile."
                )
            if avg_down < 0.0 and avg_up > 0.0:
                return (
                    "Rates-down scenarios reduce PV while rates-up scenarios lift PV, consistent with a short-rates exposure."
                )
        return "Scenario ladder shows the trade's modeled move across the configured shocks."

    @staticmethod
    def _scenario_shift(entry: Mapping[str, object]) -> float | None:
        shift = entry.get("shift_bps")
        if isinstance(shift, (int, float)):
            return float(shift)
        scenario_name = str(entry.get("scenario") or entry.get("name") or "").strip()
        if not scenario_name:
            return None
        try:
            return float(scenario_name)
        except ValueError:
            return None

    @staticmethod
    def _scenario_pnl(entry: Mapping[str, object], *, base_price) -> float | None:
        pnl = entry.get("pnl")
        if isinstance(pnl, (int, float)):
            return float(pnl)
        price = entry.get("price")
        if isinstance(price, (int, float)) and isinstance(base_price, (int, float)):
            return float(price) - float(base_price)
        return None

    @staticmethod
    def _format_number(value) -> str:
        if not isinstance(value, (int, float)):
            return "n/a"
        formatted = f"{float(value):.4f}"
        return formatted.rstrip("0").rstrip(".")

    @staticmethod
    def _format_signed_number(value) -> str:
        if not isinstance(value, (int, float)):
            return "n/a"
        formatted = f"{float(value):+.4f}"
        return formatted.rstrip("0").rstrip(".")

    @staticmethod
    def _format_percent(value) -> str:
        if not isinstance(value, (int, float)):
            return "n/a"
        return f"{float(value):.2%}"

    @staticmethod
    def _audit_refs(run, *, audit_uri: str) -> dict[str, object]:
        snapshot_uri = (
            f"trellis://market-snapshots/{run.market_snapshot_id}"
            if run.market_snapshot_id
            else ""
        )
        return {
            "run_uri": f"trellis://runs/{run.run_id}",
            "audit_uri": audit_uri,
            "snapshot_uri": snapshot_uri,
            "audit_tool": {
                "tool_name": "trellis.run.get_audit",
                "run_id": run.run_id,
            },
        }

    @staticmethod
    def _build_supported_payoff(
        *,
        selected_engine: Mapping[str, object],
        parsed_trade,
        pricing_input: Mapping[str, object],
        market_snapshot,
    ):
        adapter_id = str(selected_engine.get("adapter_id", "")).strip()
        if adapter_id in {"european_option_analytical", "vanilla_option_analytical"}:
            return PricingService._build_european_option_analytical_payoff(
                pricing_input=pricing_input,
                market_snapshot=market_snapshot,
            ), ()
        if adapter_id == "callable_bond_tree":
            return PricingService._build_callable_bond_spec(
                parsed_trade=parsed_trade,
                pricing_input=pricing_input,
            )
        if adapter_id == "bermudan_swaption_tree":
            return PricingService._build_bermudan_swaption_spec(
                parsed_trade=parsed_trade,
                pricing_input=pricing_input,
            )
        if adapter_id == "range_accrual_discounted":
            return PricingService._build_range_accrual_spec(
                parsed_trade=parsed_trade,
                pricing_input=pricing_input,
            ), ()
        raise ValueError(
            f"MCP price.trade MVP has no supported execution adapter for {adapter_id or 'this model'}."
        )

    @staticmethod
    def _build_european_option_analytical_payoff(
        *,
        pricing_input: Mapping[str, object],
        market_snapshot,
    ):
        from datetime import date as _date

        from trellis.instruments._agent.europeanoptionanalytical import (
            EuropeanOptionAnalyticalPayoff,
            EuropeanOptionSpec,
        )

        strike = pricing_input.get("strike")
        if strike in {None, ""}:
            raise ValueError("Structured trade must provide `strike` for the European option adapter.")
        notional = pricing_input.get("notional", 1.0)
        expiry = pricing_input.get("expiry_date")
        if expiry in {None, ""}:
            schedule = pricing_input.get("observation_schedule") or ()
            expiry = schedule[0] if schedule else None
        if expiry in {None, ""}:
            raise ValueError("Structured trade must provide `expiry_date` or `observation_schedule`.")
        if isinstance(expiry, str):
            expiry = _date.fromisoformat(expiry)
        if not isinstance(expiry, _date):
            raise ValueError("European option expiry must resolve to one ISO date.")
        spot = pricing_input.get("spot")
        if spot in {None, ""}:
            underliers = pricing_input.get("underliers") or pricing_input.get("constituents") or ()
            name = underliers[0] if underliers else None
            try:
                spot = market_snapshot.underlier_spot(name)
            except Exception:
                spot = market_snapshot.underlier_spot()
        if spot in {None, ""}:
            raise ValueError("Structured trade must provide `spot` or resolve one from the market snapshot.")
        return EuropeanOptionAnalyticalPayoff(
            EuropeanOptionSpec(
                notional=float(notional),
                spot=float(spot),
                strike=float(strike),
                expiry_date=expiry,
                option_type=str(pricing_input.get("option_type", "call")).strip() or "call",
            )
        )

    @staticmethod
    def _build_callable_bond_spec(
        *,
        parsed_trade,
        pricing_input: Mapping[str, object],
    ):
        from trellis.instruments.callable_bond import CallableBondSpec

        contract = getattr(parsed_trade, "semantic_contract", None)
        if contract is None:
            raise ValueError("Callable bond pricing requires a parsed semantic contract.")

        term_fields = dict(getattr(contract.product, "term_fields", {}) or {})
        schedule = tuple(term_fields.get("exercise_schedule") or contract.product.observation_schedule or ())
        if not schedule:
            raise ValueError("Structured trade must provide `call_dates` for the callable bond adapter.")

        assumptions: list[str] = []
        if pricing_input.get("call_price") in {None, ""}:
            assumptions.append("Defaulted call_price to 100.0 for callable bond tree route.")
        if pricing_input.get("frequency") in {None, ""}:
            assumptions.append("Defaulted frequency to semi_annual for callable bond tree route.")
        if pricing_input.get("day_count") in {None, ""}:
            assumptions.append("Defaulted day_count to ACT_365 for callable bond tree route.")

        return (
            CallableBondSpec(
                notional=float(term_fields.get("notional", pricing_input.get("notional"))),
                coupon=float(term_fields.get("coupon", pricing_input.get("coupon"))),
                start_date=PricingService._coerce_iso_date(
                    term_fields.get("start_date", pricing_input.get("start_date")),
                    field_name="start_date",
                ),
                end_date=PricingService._coerce_iso_date(
                    term_fields.get("end_date", pricing_input.get("end_date")),
                    field_name="end_date",
                ),
                call_dates=[
                    PricingService._coerce_iso_date(value, field_name="call_dates")
                    for value in schedule
                ],
                call_price=float(term_fields.get("call_price", pricing_input.get("call_price", 100.0)) or 100.0),
                frequency=PricingService._coerce_frequency(
                    term_fields.get("frequency", pricing_input.get("frequency")),
                    field_name="frequency",
                    default="SEMI_ANNUAL",
                ),
                day_count=PricingService._coerce_day_count(
                    term_fields.get("day_count", pricing_input.get("day_count")),
                    field_name="day_count",
                    default="ACT_365",
                ),
            ),
            tuple(assumptions),
        )

    @staticmethod
    def _build_bermudan_swaption_spec(
        *,
        parsed_trade,
        pricing_input: Mapping[str, object],
    ):
        from trellis.models.bermudan_swaption_tree import BermudanSwaptionTreeSpec

        contract = getattr(parsed_trade, "semantic_contract", None)
        if contract is None:
            raise ValueError("Bermudan swaption pricing requires a parsed semantic contract.")

        term_fields = dict(getattr(contract.product, "term_fields", {}) or {})
        schedule = tuple(term_fields.get("exercise_schedule") or contract.product.observation_schedule or ())
        if not schedule:
            raise ValueError("Structured trade must provide `exercise_schedule` for the Bermudan swaption adapter.")

        assumptions: list[str] = []
        if pricing_input.get("swap_frequency") in {None, ""}:
            assumptions.append("Defaulted swap_frequency to semi_annual for Bermudan swaption tree route.")
        if pricing_input.get("day_count") in {None, ""}:
            assumptions.append("Defaulted day_count to ACT_360 for Bermudan swaption tree route.")
        if pricing_input.get("is_payer") in {None, ""}:
            assumptions.append("Defaulted is_payer to True for Bermudan swaption tree route.")

        return (
            BermudanSwaptionTreeSpec(
                notional=float(term_fields.get("notional", pricing_input.get("notional"))),
                strike=float(term_fields.get("strike", pricing_input.get("strike"))),
                exercise_dates=tuple(
                    PricingService._coerce_iso_date(value, field_name="exercise_schedule")
                    for value in schedule
                ),
                swap_end=PricingService._coerce_iso_date(
                    term_fields.get("swap_end", pricing_input.get("swap_end")),
                    field_name="swap_end",
                ),
                swap_frequency=PricingService._coerce_frequency(
                    term_fields.get("swap_frequency", pricing_input.get("swap_frequency")),
                    field_name="swap_frequency",
                    default="SEMI_ANNUAL",
                ),
                day_count=PricingService._coerce_day_count(
                    term_fields.get("day_count", pricing_input.get("day_count")),
                    field_name="day_count",
                    default="ACT_360",
                ),
                is_payer=PricingService._coerce_bool(
                    term_fields.get("is_payer", pricing_input.get("is_payer")),
                    field_name="is_payer",
                    default=True,
                ),
            ),
            tuple(assumptions),
        )

    @staticmethod
    def _build_range_accrual_spec(
        *,
        parsed_trade,
        pricing_input: Mapping[str, object],
    ):
        from trellis.models.range_accrual import RangeAccrualSpec

        contract = getattr(parsed_trade, "semantic_contract", None)
        if contract is None:
            raise ValueError("Range accrual pricing requires a parsed semantic contract.")

        term_fields = dict(getattr(contract.product, "term_fields", {}) or {})
        coupon_definition = dict(term_fields.get("coupon_definition") or {})
        range_condition = dict(term_fields.get("range_condition") or {})
        notional = (
            pricing_input.get("notional")
            or pricing_input.get("principal_amount")
            or pricing_input.get("face_amount")
        )
        if notional in {None, ""}:
            raise ValueError("Structured trade must provide `notional` for the range accrual adapter.")

        observation_dates = tuple(contract.product.observation_schedule)
        accrual_start_dates = pricing_input.get("accrual_start_dates") or ()
        payment_dates = pricing_input.get("payment_dates") or pricing_input.get("payment_schedule") or ()

        return RangeAccrualSpec(
            reference_index=str(term_fields.get("reference_index", "")).strip(),
            notional=float(notional),
            coupon_rate=float(coupon_definition.get("coupon_rate", 0.0)),
            lower_bound=float(range_condition.get("lower_bound", 0.0)),
            upper_bound=float(range_condition.get("upper_bound", 0.0)),
            observation_dates=observation_dates,
            accrual_start_dates=accrual_start_dates,
            payment_dates=payment_dates,
            principal_redemption=float(pricing_input.get("principal_redemption", 1.0) or 1.0),
            inclusive_lower=bool(range_condition.get("inclusive_lower", True)),
            inclusive_upper=bool(range_condition.get("inclusive_upper", True)),
        )

    @staticmethod
    def _coerce_iso_date(value, *, field_name: str):
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"Structured trade must provide `{field_name}` for this adapter.")
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Structured trade field `{field_name}` must be an ISO date.") from exc

    @staticmethod
    def _coerce_frequency(value, *, field_name: str, default: str):
        from trellis.core.types import Frequency

        raw_value = default if value in {None, ""} else getattr(value, "name", value)
        text = str(raw_value or default).strip().lower().replace("-", "_").replace(" ", "_")
        if text.startswith("frequency."):
            text = text.split(".", 1)[1]
        aliases = {
            "annual": Frequency.ANNUAL,
            "annually": Frequency.ANNUAL,
            "semi_annual": Frequency.SEMI_ANNUAL,
            "semiannual": Frequency.SEMI_ANNUAL,
            "semi_annually": Frequency.SEMI_ANNUAL,
            "quarterly": Frequency.QUARTERLY,
            "monthly": Frequency.MONTHLY,
        }
        resolved = aliases.get(text)
        if resolved is None:
            raise ValueError(f"Structured trade field `{field_name}` has unsupported frequency {value!r}.")
        return resolved

    @staticmethod
    def _coerce_day_count(value, *, field_name: str, default: str):
        from trellis.conventions.day_count import DayCountConvention

        raw_value = default if value in {None, ""} else getattr(value, "name", value)
        text = (
            str(raw_value or default)
            .strip()
            .lower()
            .replace("-", "_")
            .replace("/", "_")
            .replace(" ", "_")
        )
        if text.startswith("daycountconvention."):
            text = text.split(".", 1)[1]
        aliases = {
            "act_360": DayCountConvention.ACT_360,
            "act360": DayCountConvention.ACT_360,
            "actual_360": DayCountConvention.ACT_360,
            "act_365": DayCountConvention.ACT_365,
            "act365": DayCountConvention.ACT_365,
            "actual_365": DayCountConvention.ACT_365,
            "act_365_fixed": DayCountConvention.ACT_365,
            "act_act": DayCountConvention.ACT_ACT,
            "act_act_isda": DayCountConvention.ACT_ACT,
            "actual_actual": DayCountConvention.ACT_ACT,
            "30_360": DayCountConvention.THIRTY_360,
            "thirty_360": DayCountConvention.THIRTY_360,
            "30_360_us": DayCountConvention.THIRTY_360,
            "thirty_360_us": DayCountConvention.THIRTY_360,
        }
        resolved = aliases.get(text)
        if resolved is None:
            raise ValueError(f"Structured trade field `{field_name}` has unsupported day_count {value!r}.")
        return resolved

    @staticmethod
    def _coerce_bool(value, *, field_name: str, default: bool) -> bool:
        if value in {None, ""}:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "payer"}:
            return True
        if text in {"false", "0", "no", "receiver"}:
            return False
        raise ValueError(f"Structured trade field `{field_name}` must be boolean-like.")

    @staticmethod
    def _project_callable_bond_events(spec) -> list[dict[str, object]]:
        events = [
            {
                "date": exercise_date.isoformat(),
                "event_type": "issuer_call_decision",
                "schedule_role": "decision_dates",
                "call_price": float(spec.call_price),
            }
            for exercise_date in spec.call_dates
        ]
        events.append(
            {
                "date": spec.end_date.isoformat(),
                "event_type": "bond_maturity_settlement",
                "schedule_role": "settlement_dates",
            }
        )
        return events

    @staticmethod
    def _project_bermudan_swaption_events(spec) -> list[dict[str, object]]:
        option_side = "payer" if bool(spec.is_payer) else "receiver"
        events = [
            {
                "date": exercise_date.isoformat(),
                "event_type": "holder_exercise_decision",
                "schedule_role": "exercise_dates",
                "option_side": option_side,
            }
            for exercise_date in spec.exercise_dates
        ]
        events.append(
            {
                "date": spec.swap_end.isoformat(),
                "event_type": "underlying_swap_maturity_boundary",
                "schedule_role": "settlement_dates",
            }
        )
        return events

    @staticmethod
    def _callable_bond_validation_bundle(*, price: float, straight_price: float, exercise_dates, assumptions=()) -> dict[str, object]:
        capped = float(price) <= float(straight_price) + 1e-9
        return {
            "route_id": "callable_bond_tree_v1",
            "checks": [
                {
                    "check_id": "exercise_schedule_present",
                    "status": "passed" if exercise_dates else "failed",
                },
                {
                    "check_id": "callable_price_bounded_by_straight_bond",
                    "status": "passed" if capped else "failed",
                },
            ],
            "assumptions": [str(item).strip() for item in assumptions if str(item).strip()],
        }

    @staticmethod
    def _bermudan_swaption_validation_bundle(*, price: float, exercise_dates, swap_end: date) -> dict[str, object]:
        ordered = all(exercise_date < swap_end for exercise_date in exercise_dates)
        return {
            "route_id": "bermudan_swaption_tree_v1",
            "checks": [
                {
                    "check_id": "exercise_schedule_present",
                    "status": "passed" if exercise_dates else "failed",
                },
                {
                    "check_id": "exercise_dates_before_swap_end",
                    "status": "passed" if ordered else "failed",
                },
                {
                    "check_id": "non_negative_option_value",
                    "status": "passed" if float(price) >= -1e-9 else "failed",
                },
            ],
        }

    @staticmethod
    def _resolve_range_accrual_forecast_curve(market_snapshot, *, reference_index: str):
        forecast_curves = dict(getattr(market_snapshot, "forecast_curves", {}) or {})
        metadata = dict(getattr(market_snapshot, "metadata", {}) or {})
        default_name = str(metadata.get("default_forecast_curve", "")).strip()
        candidates = (
            str(reference_index or "").strip(),
            str(reference_index or "").strip().upper(),
            str(reference_index or "").strip().lower(),
            default_name,
        )
        for name in candidates:
            if name and name in forecast_curves:
                return forecast_curves[name], ()
        if len(forecast_curves) == 1:
            return next(iter(forecast_curves.values())), ()
        return market_snapshot.discount_curve(), (
            "Used the discount curve as the forward projection proxy because no dedicated forecast curve was configured.",
        )

    @staticmethod
    def _resolve_range_accrual_fixing_history(market_snapshot, *, reference_index: str):
        fixing_histories = {
            key: dict(value)
            for key, value in dict(getattr(market_snapshot, "fixing_histories", {}) or {}).items()
        }
        metadata = dict(getattr(market_snapshot, "metadata", {}) or {})
        if not fixing_histories:
            fixing_histories = dict(metadata.get("fixing_histories") or {})
        if not fixing_histories:
            return {}
        default_name = str(
            getattr(market_snapshot, "default_fixing_history", None)
            or metadata.get("default_fixing_history", "")
        ).strip()
        candidates = (
            str(reference_index or "").strip(),
            str(reference_index or "").strip().upper(),
            str(reference_index or "").strip().lower(),
            default_name,
        )
        for name in candidates:
            if name and name in fixing_histories:
                return fixing_histories[name]
        if len(fixing_histories) == 1:
            return next(iter(fixing_histories.values()))
        return {}


__all__ = [
    "PricingService",
]
