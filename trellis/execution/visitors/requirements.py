"""Requirement visitors for execution IR artifacts."""

from __future__ import annotations

from trellis.execution.ir import ContractExecutionIR, RequirementHints


def derive_requirement_hints(ir: ContractExecutionIR) -> RequirementHints:
    """Derive route-free requirement hints from an execution artifact."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")

    market_inputs = set(ir.requirement_hints.market_inputs)
    state_variables = set(ir.requirement_hints.state_variables)
    timeline_roles = set(ir.requirement_hints.timeline_roles)

    for observable in ir.observables:
        if observable.observable_id:
            market_inputs.add(observable.observable_id)
        if observable.observable_kind == "spot":
            state_variables.add("spot_state")
    for event in ir.event_plan.events:
        if event.schedule_role:
            timeline_roles.add(event.schedule_role)

    return RequirementHints(
        market_inputs=frozenset(market_inputs),
        state_variables=frozenset(state_variables),
        timeline_roles=frozenset(timeline_roles),
        unsupported_reasons=ir.requirement_hints.unsupported_reasons,
    )


__all__ = ["derive_requirement_hints"]
