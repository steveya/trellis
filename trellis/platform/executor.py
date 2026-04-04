"""Authoritative compiled-request dispatcher skeleton for governed execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from trellis.platform.context import ExecutionContext
from trellis.platform.policies import evaluate_execution_policy
from trellis.platform.results import ExecutionResult


REQUIRED_EXECUTION_ACTIONS = (
    "price_book",
    "price_existing_instrument",
    "compute_greeks",
    "analyze_existing_instrument",
    "price_existing_payoff",
    "build_then_price",
    "compile_only",
    "block",
    "compare_methods",
)

ExecutionHandler = Callable[[object, ExecutionContext, str], Mapping[str, object] | ExecutionResult]


def execute_compiled_request(
    compiled_request,
    execution_context: ExecutionContext,
    *,
    handlers: Mapping[str, ExecutionHandler] | None = None,
) -> ExecutionResult:
    """Dispatch one compiled request through the governed executor boundary."""
    run_id = _new_run_id()
    action = str(compiled_request.execution_plan.action or "").strip()
    provenance = _base_provenance(compiled_request, execution_context, run_id=run_id)
    policy_outcome = evaluate_execution_policy(
        execution_context=execution_context,
        provenance=provenance,
    ).to_dict()

    if not policy_outcome.get("allowed", False):
        return _build_result(
            compiled_request,
            execution_context,
            run_id=run_id,
            status="blocked",
            result_payload={
                "reason": "policy_blocked",
                "blocker_codes": list(policy_outcome.get("blocker_codes") or []),
            },
            warnings=("policy_blocked",),
            provenance=provenance,
            audit_summary={
                "dispatch_status": "policy_blocked",
                "handler_action": action,
            },
            policy_outcome=policy_outcome,
        )

    handler_table = dict(default_execution_handlers())
    handler_table.update(handlers or {})
    handler = handler_table.get(action)
    if handler is None:
        return _build_result(
            compiled_request,
            execution_context,
            run_id=run_id,
            status="failed",
            result_payload={
                "error_type": "UnknownExecutionAction",
                "error": f"Unsupported execution action: {action}",
            },
            warnings=("unknown_execution_action",),
            provenance=provenance,
            audit_summary={
                "dispatch_status": "unknown_action",
                "handler_action": action,
            },
            policy_outcome=policy_outcome,
        )

    try:
        outcome = handler(compiled_request, execution_context, run_id)
    except Exception as exc:
        return _build_result(
            compiled_request,
            execution_context,
            run_id=run_id,
            status="failed",
            result_payload={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            warnings=("handler_exception",),
            provenance=provenance,
            audit_summary={
                "dispatch_status": "handler_exception",
                "handler_action": action,
            },
            policy_outcome=policy_outcome,
        )

    if isinstance(outcome, ExecutionResult):
        return outcome

    return _build_result(
        compiled_request,
        execution_context,
        run_id=run_id,
        status=str(outcome.get("status", "succeeded")).strip(),
        result_payload=outcome.get("result_payload") or {},
        warnings=outcome.get("warnings") or (),
        provenance={**provenance, **dict(outcome.get("provenance") or {})},
        artifacts=outcome.get("artifacts") or (),
        audit_summary={
            "dispatch_status": str(outcome.get("status", "succeeded")).strip(),
            "handler_action": action,
            **dict(outcome.get("audit_summary") or {}),
        },
        trace_path=str(outcome.get("trace_path", "") or "").strip(),
        policy_outcome=policy_outcome,
        output_mode=str(
            outcome.get("output_mode") or execution_context.default_output_mode
        ).strip(),
    )


def default_execution_handlers() -> dict[str, ExecutionHandler]:
    """Return the default handler table for the current compiled action space."""
    return {
        "price_book": _price_book_handler,
        "price_existing_instrument": _price_existing_instrument_handler,
        "compute_greeks": _compute_greeks_handler,
        "analyze_existing_instrument": _analyze_existing_instrument_handler,
        "price_existing_payoff": _price_existing_payoff_handler,
        "build_then_price": _build_then_price_handler,
        "compile_only": _compile_only_handler,
        "block": _block_handler,
        "compare_methods": _pending_route_adapter_handler,
    }


def _compile_only_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Return a success envelope for compile-only requests."""
    request = compiled_request.request
    return {
        "status": "succeeded",
        "result_payload": {
            "compiled": True,
            "request_type": request.request_type,
            "reason": compiled_request.execution_plan.reason,
            "route_method": compiled_request.execution_plan.route_method,
            "requested_outputs": list(request.requested_outputs),
        },
        "audit_summary": {
            "mode": "compile_only",
            "request_type": request.request_type,
        },
    }


def _block_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Return a structured blocked envelope from a compiled blocker report."""
    blocker_report = getattr(compiled_request, "blocker_report", None)
    blocker_codes = [
        str(blocker.id).strip()
        for blocker in getattr(blocker_report, "blockers", ())
    ]
    return {
        "status": "blocked",
        "result_payload": {
            "reason": "compiled_request_blocked",
            "blocker_codes": blocker_codes,
            "blocker_summary": getattr(blocker_report, "summary", "") if blocker_report is not None else "",
        },
        "warnings": ("compiled_request_blocked",),
        "audit_summary": {
            "mode": "blocked",
            "blocker_count": len(blocker_codes),
        },
    }


def _pending_route_adapter_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Return a temporary blocked envelope until route adapters land."""
    action = str(compiled_request.execution_plan.action or "").strip()
    return {
        "status": "blocked",
        "result_payload": {
            "reason": "route_adapter_pending",
            "action": action,
            "route_method": compiled_request.execution_plan.route_method,
        },
        "warnings": ("route_adapter_pending",),
        "audit_summary": {
            "mode": "pending_route_adapter",
            "pending_action": action,
        },
    }


def _price_existing_instrument_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Price one direct instrument through the existing deterministic pricer."""
    from trellis.engine.pricer import price_instrument

    instrument = compiled_request.request.instrument
    if instrument is None:
        raise ValueError("Compiled request has no instrument")
    result = price_instrument(
        instrument,
        _discount_curve(compiled_request),
        _settlement(compiled_request),
        greeks=_pricing_greeks_spec(compiled_request.request),
    )
    return {
        "status": "succeeded",
        "result_payload": {"result": result},
        "audit_summary": {
            "adapter": "direct_instrument",
            "result_type": type(result).__name__,
        },
    }


def _price_book_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Price or analyze a book through thin executor-owned adapters."""
    from trellis.analytics.result import BookAnalyticsResult
    from trellis.book import BookResult
    from trellis.engine.pricer import price_instrument

    book = compiled_request.request.book
    if book is None:
        raise ValueError("Compiled request has no book")
    if compiled_request.request.request_type == "analytics":
        result = _analyze_book(
            book,
            compiled_request,
            requested_outputs=compiled_request.request.requested_outputs,
        )
        result_type = BookAnalyticsResult.__name__
        adapter = "book_analytics"
    else:
        greeks = _pricing_greeks_spec(compiled_request.request)
        results = {
            name: price_instrument(
                book[name],
                _discount_curve(compiled_request),
                _settlement(compiled_request),
                greeks=greeks,
            )
            for name in book
        }
        result = BookResult(results, book)
        result_type = BookResult.__name__
        adapter = "book_pricing"
    return {
        "status": "succeeded",
        "result_payload": {"result": result},
        "audit_summary": {
            "adapter": adapter,
            "result_type": result_type,
        },
    }


def _compute_greeks_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Compute instrument Greeks through the existing deterministic pricer."""
    from trellis.engine.pricer import price_instrument

    instrument = compiled_request.request.instrument
    if instrument is None:
        raise ValueError("Compiled request has no instrument")
    pricing_result = price_instrument(
        instrument,
        _discount_curve(compiled_request),
        _settlement(compiled_request),
        greeks=_greeks_spec(compiled_request.request, default="all"),
    )
    return {
        "status": "succeeded",
        "result_payload": {"result": pricing_result.greeks},
        "audit_summary": {
            "adapter": "direct_greeks",
            "measure_count": len(pricing_result.greeks),
        },
    }


def _analyze_existing_instrument_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Analyze one direct instrument or payoff through existing analytics helpers."""
    instrument = compiled_request.request.instrument
    if instrument is None:
        raise ValueError("Compiled request has no instrument")
    result = _analyze_instrument(
        instrument,
        compiled_request,
        requested_outputs=compiled_request.request.requested_outputs,
    )
    return {
        "status": "succeeded",
        "result_payload": {"result": result},
        "audit_summary": {
            "adapter": "direct_analytics",
            "result_type": type(result).__name__,
        },
    }


def _price_existing_payoff_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Price a matched existing payoff for ask-style requests."""
    from trellis.agent.ask import match_payoff
    from trellis.engine.payoff_pricer import price_payoff

    request = compiled_request.request
    term_sheet = request.term_sheet
    if term_sheet is None:
        raise ValueError("Compiled request has no term sheet")
    match = match_payoff(term_sheet, _settlement(compiled_request))
    if match is None:
        raise ValueError("No existing payoff match is available")
    payoff, requirements = match
    market_state = _market_state(compiled_request)
    price = price_payoff(payoff, market_state)
    analytics = None
    if any(output != "price" for output in request.requested_outputs):
        analytics = _analyze_instrument(
            payoff,
            compiled_request,
            requested_outputs=request.requested_outputs,
        )
    return {
        "status": "succeeded",
        "result_payload": {
            "price": price,
            "analytics": analytics,
            "payoff": payoff,
            "payoff_class": type(payoff).__name__,
            "matched_existing": True,
            "term_sheet": term_sheet,
            "requirements": sorted(str(item) for item in requirements),
        },
        "audit_summary": {
            "adapter": "matched_existing_payoff",
            "payoff_class": type(payoff).__name__,
        },
    }


def _build_then_price_handler(compiled_request, execution_context: ExecutionContext, run_id: str):
    """Route candidate-generation pricing through the governed executor spine."""
    from trellis.agent.config import get_default_model
    from trellis.agent.executor import _make_test_payoff, build_payoff
    from trellis.agent.planner import plan_build
    from trellis.agent.ask import _infer_requirements
    from trellis.engine.payoff_pricer import price_payoff

    request = compiled_request.request
    term_sheet = request.term_sheet
    if term_sheet is None:
        raise ValueError("Compiled request has no term sheet")

    market_state = _market_state(compiled_request)
    resolved_model = request.model or get_default_model()
    payoff_cls = build_payoff(
        term_sheet.raw_description or request.description or term_sheet.instrument_type,
        requirements=None,
        model=resolved_model,
        market_state=market_state,
        instrument_type=term_sheet.instrument_type,
        compiled_request=compiled_request,
    )
    plan = plan_build(
        term_sheet.raw_description or request.description or term_sheet.instrument_type,
        _infer_requirements(term_sheet),
        model=resolved_model,
        instrument_type=term_sheet.instrument_type,
    )
    if plan.spec_schema is not None:
        payoff = _make_test_payoff(
            payoff_cls,
            plan.spec_schema,
            _settlement(compiled_request),
        )
    else:
        payoff = payoff_cls(term_sheet.parameters)

    price = price_payoff(payoff, market_state)
    analytics = None
    if any(output != "price" for output in request.requested_outputs):
        analytics = _analyze_instrument(
            payoff,
            compiled_request,
            requested_outputs=request.requested_outputs,
        )
    return {
        "status": "succeeded",
        "result_payload": {
            "price": price,
            "analytics": analytics,
            "payoff": payoff,
            "payoff_class": type(payoff).__name__,
            "matched_existing": False,
            "term_sheet": term_sheet,
            "details": {
                "route_method": compiled_request.execution_plan.route_method,
                "candidate_generation": True,
            },
        },
        "warnings": ("candidate_generation",),
        "audit_summary": {
            "adapter": "build_then_price_candidate",
            "payoff_class": type(payoff).__name__,
        },
    }


def _build_result(
    compiled_request,
    execution_context: ExecutionContext,
    *,
    run_id: str,
    status: str,
    result_payload: Mapping[str, object] | None = None,
    warnings=(),
    provenance: Mapping[str, object] | None = None,
    artifacts=(),
    audit_summary: Mapping[str, object] | None = None,
    trace_path: str = "",
    policy_outcome: Mapping[str, object] | None = None,
    output_mode: str | None = None,
) -> ExecutionResult:
    """Normalize one dispatch outcome into the canonical result envelope."""
    return ExecutionResult(
        run_id=run_id,
        request_id=compiled_request.request.request_id,
        status=status,
        action=compiled_request.execution_plan.action,
        output_mode=str(output_mode or execution_context.default_output_mode).strip(),
        result_payload=result_payload or {},
        warnings=warnings,
        provenance=provenance or {},
        artifacts=artifacts,
        audit_summary=audit_summary or {},
        trace_path=trace_path,
        policy_outcome=policy_outcome or {},
    )


def _base_provenance(
    compiled_request,
    execution_context: ExecutionContext,
    *,
    run_id: str,
) -> dict[str, object]:
    """Build the default provenance block for one dispatch attempt."""
    request = compiled_request.request
    market_snapshot = getattr(compiled_request, "market_snapshot", None)
    snapshot_id = (
        getattr(market_snapshot, "market_snapshot_id", None)
        or getattr(market_snapshot, "snapshot_id", None)
        or ""
    )
    valuation_timestamp = _valuation_timestamp(request, market_snapshot)
    market_data_binding = execution_context.provider_bindings.market_data.primary
    return {
        "run_id": run_id,
        "request_id": request.request_id,
        "action": compiled_request.execution_plan.action,
        "run_mode": execution_context.run_mode.value,
        "session_id": execution_context.session_id,
        "policy_id": execution_context.policy_bundle_id,
        "request_type": request.request_type,
        "entry_point": request.entry_point,
        "route_method": compiled_request.execution_plan.route_method or "",
        "provider_bindings": execution_context.provider_bindings.to_dict(),
        "provider_id": (
            market_data_binding.provider_id
            if market_data_binding is not None
            else ""
        ),
        "market_snapshot_id": str(snapshot_id or "").strip(),
        "valuation_timestamp": valuation_timestamp,
    }


def _valuation_timestamp(request, market_snapshot) -> str:
    """Infer the best available valuation timestamp for one compiled request."""
    settlement = getattr(request, "settlement", None)
    if isinstance(settlement, date):
        return settlement.isoformat()
    as_of = getattr(market_snapshot, "as_of", None)
    if isinstance(as_of, datetime):
        return as_of.isoformat()
    if isinstance(as_of, date):
        return as_of.isoformat()
    return ""


def _settlement(compiled_request) -> date:
    """Return the effective settlement date for one compiled request."""
    settlement = getattr(compiled_request.request, "settlement", None)
    if isinstance(settlement, date):
        return settlement
    market_snapshot = getattr(compiled_request, "market_snapshot", None)
    as_of = getattr(market_snapshot, "as_of", None)
    if isinstance(as_of, date):
        return as_of
    return date.today()


def _discount_curve(compiled_request):
    """Resolve the active discount curve from the compiled request snapshot."""
    market_snapshot = getattr(compiled_request, "market_snapshot", None)
    if market_snapshot is None:
        raise ValueError("Compiled request has no market snapshot")
    metadata = getattr(compiled_request.request, "metadata", None) or {}
    curve = market_snapshot.discount_curve(metadata.get("discount_curve_name"))
    if curve is None:
        raise ValueError("Compiled request market snapshot has no discount curve")
    return curve


def _market_state(compiled_request):
    """Compile the request snapshot into a runtime market state."""
    market_snapshot = getattr(compiled_request, "market_snapshot", None)
    if market_snapshot is None:
        raise ValueError("Compiled request has no market snapshot")
    metadata = getattr(compiled_request.request, "metadata", None) or {}
    return market_snapshot.to_market_state(
        settlement=_settlement(compiled_request),
        discount_curve=metadata.get("discount_curve_name"),
        vol_surface=metadata.get("vol_surface_name"),
    )


def _greeks_spec(request, *, default: str | None = None):
    """Map normalized requested outputs onto the greeks spec expected by the pricer."""
    requested_outputs = [output for output in request.requested_outputs if output != "price"]
    if requested_outputs:
        return requested_outputs
    return default


def _pricing_greeks_spec(request):
    """Resolve the pricing greeks mode from request metadata."""
    metadata = getattr(request, "metadata", None) or {}
    greeks_mode = str(metadata.get("greeks_mode", "")).strip().lower()
    if greeks_mode == "none":
        return None
    return _greeks_spec(request, default="all")


def _analyze_instrument(instrument, compiled_request, *, requested_outputs):
    """Compute analytics for one direct instrument or payoff."""
    from trellis.analytics.measures import resolve_measures
    from trellis.analytics.result import AnalyticsResult

    request = compiled_request.request
    resolved = resolve_measures(_measure_specs(request, requested_outputs=requested_outputs))
    market_state = _market_state(compiled_request)
    context = _measure_context(request)
    data = {}
    for measure in resolved:
        data[measure.name] = measure.compute(instrument, market_state, **context)
    return AnalyticsResult(data)


def _analyze_book(book, compiled_request, *, requested_outputs):
    """Compute book analytics through the same deterministic measure helpers."""
    from trellis.analytics.measures import resolve_measures
    from trellis.analytics.result import AnalyticsResult, BookAnalyticsResult

    request = compiled_request.request
    resolved = resolve_measures(_measure_specs(request, requested_outputs=requested_outputs))
    market_state = _market_state(compiled_request)
    positions = {}
    for name in book:
        context = _measure_context(request)
        data = {}
        for measure in resolved:
            data[measure.name] = measure.compute(book[name], market_state, **context)
        positions[name] = AnalyticsResult(data)
    notionals = {name: book.notional(name) for name in book}
    return BookAnalyticsResult(positions, notionals)


def _new_run_id() -> str:
    """Generate a governed executor run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"run_{stamp}_{uuid4().hex[:8]}"


def _measure_specs(request, *, requested_outputs):
    """Return the raw analytics measure specs when the request carried them."""
    specs = tuple(getattr(request, "measure_specs", ()) or ())
    if specs:
        return list(specs)
    return list(requested_outputs) or ["price", "dv01", "duration"]


def _measure_context(request) -> dict[str, object]:
    """Build the shared measure context for one analytics request."""
    context = dict(getattr(request, "measure_context", {}) or {})
    context.setdefault("_cache", {})
    return context
