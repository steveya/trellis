"""Governed MCP pricing orchestration over typed parse, match, policy, and ledger layers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

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
                validation_summary={"model_execution_gate": gate.to_dict()},
                policy_outcome=policy_outcome,
            )
            return self._response_for_run(run, output_mode=resolved_output_mode)

        snapshot_record = self._persist_snapshot(market_snapshot)
        pricing_input = self._pricing_input(structured_trade, description)
        try:
            payoff = self._build_supported_payoff(
                selected_engine=selected_engine,
                pricing_input=pricing_input,
                market_snapshot=market_snapshot,
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
                market_snapshot=market_snapshot,
                warnings=warnings,
                result_summary={
                    "reason": "pricing_input_incomplete",
                    "error": str(exc),
                },
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
    ) -> CompiledPlatformRequest:
        request = PlatformRequest(
            request_id=request_id,
            request_type="price",
            entry_point="mcp",
            settlement=market_snapshot.as_of,
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
        from trellis.engine.payoff_pricer import price_payoff

        payoff = compiled_request.request.instrument
        if payoff is None:
            raise ValueError("Compiled trade request has no payoff adapter")
        market_snapshot = compiled_request.market_snapshot
        if market_snapshot is None:
            raise ValueError("Compiled trade request has no market snapshot")
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
                    "" if market_snapshot is None else str(getattr(market_snapshot, "as_of", "")).strip()
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
        record = SnapshotRecord(
            snapshot_id=str(getattr(market_snapshot, "market_snapshot_id", "")).strip(),
            provider_id=str(getattr(market_snapshot, "provider_id", "")).strip(),
            as_of=market_snapshot.as_of.isoformat(),
            source=str(getattr(market_snapshot, "source", "")).strip(),
            payload={
                "discount_curves": sorted(market_snapshot.discount_curves),
                "forecast_curves": sorted(market_snapshot.forecast_curves),
                "vol_surfaces": sorted(market_snapshot.vol_surfaces),
                "credit_curves": sorted(market_snapshot.credit_curves),
                "fx_rates": sorted(market_snapshot.fx_rates),
                "underlier_spots": dict(market_snapshot.underlier_spots),
                "default_discount_curve": market_snapshot.default_discount_curve,
                "default_vol_surface": market_snapshot.default_vol_surface,
                "default_underlier_spot": market_snapshot.default_underlier_spot,
            },
            provenance=dict(market_snapshot.provenance),
        )
        return self.snapshot_store.save_snapshot(record)

    def _response_for_run(self, run, *, output_mode: str) -> dict[str, object]:
        audit_uri = f"trellis://runs/{run.run_id}/audit"
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
    def _build_supported_payoff(
        *,
        selected_engine: Mapping[str, object],
        pricing_input: Mapping[str, object],
        market_snapshot,
    ):
        adapter_id = str(selected_engine.get("adapter_id", "")).strip()
        if adapter_id in {"european_option_analytical", "vanilla_option_analytical"}:
            return PricingService._build_european_option_analytical_payoff(
                pricing_input=pricing_input,
                market_snapshot=market_snapshot,
            )
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


__all__ = [
    "PricingService",
]
