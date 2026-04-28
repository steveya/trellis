"""Stable summaries for execution IR diagnostics."""

from __future__ import annotations

from trellis.execution.ir import ContractExecutionIR


def _ordered(values: object) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(values, str):
        values = (values,)
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(sorted(result))


def _ordered_preserve(values: object) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(values, str):
        values = (values,)
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def contract_execution_summary(ir: ContractExecutionIR) -> dict[str, object]:
    """Return a route-free, model-free summary of a contract execution IR."""
    unsupported_reasons = _ordered_preserve(
        (
            *ir.requirement_hints.unsupported_reasons,
            *ir.execution_metadata.unsupported_reasons,
        )
    )
    return {
        "schema_version": ir.execution_metadata.schema_version,
        "source_kind": ir.source_track.source_kind,
        "semantic_id": ir.source_track.semantic_id,
        "source_ref": ir.source_track.source_ref,
        "obligation_count": len(ir.obligations),
        "obligation_kinds": _ordered(
            getattr(obligation, "obligation_kind", "")
            for obligation in ir.obligations
        ),
        "observable_count": len(ir.observables),
        "observable_kinds": _ordered(
            getattr(observable, "observable_kind", "")
            for observable in ir.observables
        ),
        "event_count": len(ir.event_plan.events),
        "state_field_count": len(ir.state_schema.fields),
        "decision_action_count": len(ir.decision_program.actions),
        "settlement_step_count": len(ir.settlement_program.steps),
        "requirement_markets": _ordered(ir.requirement_hints.market_inputs),
        "requirement_states": _ordered(ir.requirement_hints.state_variables),
        "timeline_roles": _ordered(ir.requirement_hints.timeline_roles),
        "unsupported_reasons": unsupported_reasons,
        "route_ids": (),
        "model_families": (),
        "tags": _ordered_preserve(ir.execution_metadata.tags),
    }
