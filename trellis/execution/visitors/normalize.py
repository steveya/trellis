"""Normalization visitors for execution IR artifacts."""

from __future__ import annotations

from dataclasses import replace

from trellis.execution.ir import ContractExecutionIR, ExecutionEventPlan


def normalize_execution_ir(ir: ContractExecutionIR) -> ContractExecutionIR:
    """Return a deterministic ordering-normalized execution artifact."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    return replace(
        ir,
        obligations=tuple(
            sorted(ir.obligations, key=lambda item: (item.obligation_kind, item.obligation_id))
        ),
        observables=tuple(
            sorted(ir.observables, key=lambda item: (item.observable_kind, item.observable_id))
        ),
        event_plan=replace(
            ir.event_plan,
            events=tuple(
                sorted(
                    ir.event_plan.events,
                    key=lambda item: (item.event_date is None, item.event_date, item.phase, item.event_id),
                )
            ),
        )
        if isinstance(ir.event_plan, ExecutionEventPlan)
        else ir.event_plan,
    )


__all__ = ["normalize_execution_ir"]
