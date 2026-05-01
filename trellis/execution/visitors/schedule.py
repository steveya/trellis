"""Schedule visitors for execution IR artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.execution.ir import ContractExecutionIR


@dataclass(frozen=True)
class ExecutionScheduleEntry:
    """Stable event schedule row derived from a ContractExecutionIR."""

    event_id: str
    event_kind: str
    event_date: object
    schedule_role: str
    phase: str


def execution_event_schedule(ir: ContractExecutionIR) -> tuple[ExecutionScheduleEntry, ...]:
    """Return a deterministic event schedule projection."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    return tuple(
        ExecutionScheduleEntry(
            event_id=event.event_id,
            event_kind=event.event_kind,
            event_date=event.event_date,
            schedule_role=event.schedule_role,
            phase=event.phase,
        )
        for event in sorted(
            ir.event_plan.events,
            key=lambda item: (item.event_date is None, item.event_date, item.phase, item.event_id),
        )
    )


__all__ = ["ExecutionScheduleEntry", "execution_event_schedule"]
