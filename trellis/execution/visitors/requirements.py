"""Requirement visitors for execution IR artifacts."""

from __future__ import annotations

from datetime import date

from trellis.execution.ir import (
    ContractExecutionIR,
    CouponLegExecution,
    RequirementHints,
)


def derive_requirement_hints(
    ir: ContractExecutionIR,
    *,
    valuation_date: date | None = None,
) -> RequirementHints:
    """Derive route-free requirement hints from an execution artifact."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    if valuation_date is not None and not isinstance(valuation_date, date):
        raise TypeError("valuation_date must be a date or None")

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
    if valuation_date is not None:
        _remove_future_coupon_fixing_inputs(
            market_inputs,
            ir,
            valuation_date=valuation_date,
        )

    return RequirementHints(
        market_inputs=frozenset(market_inputs),
        state_variables=frozenset(state_variables),
        timeline_roles=frozenset(timeline_roles),
        unsupported_reasons=ir.requirement_hints.unsupported_reasons,
    )


def _remove_future_coupon_fixing_inputs(
    market_inputs: set[str],
    ir: ContractExecutionIR,
    *,
    valuation_date: date,
) -> None:
    coupon_inputs: set[str] = set()
    historical_inputs: set[str] = set()
    for obligation in ir.obligations:
        if not isinstance(obligation, CouponLegExecution):
            continue
        metadata = dict(obligation.metadata or ())
        if str(metadata.get("formula_kind") or "") != "floating":
            continue
        index_name = str(metadata.get("rate_index") or "").strip()
        if not index_name:
            continue
        requirement = f"fixing_history:{index_name}"
        coupon_inputs.add(requirement)
        for period in tuple(metadata.get("periods") or ()):
            if len(period) < 4:
                continue
            payment_date = period[2]
            fixing_date = period[3]
            if (
                isinstance(payment_date, date)
                and isinstance(fixing_date, date)
                and fixing_date < valuation_date < payment_date
            ):
                historical_inputs.add(requirement)
                break

    non_coupon_inputs = {
        observable.observable_id
        for observable in ir.observables
        if observable.observable_kind == "fixing_history"
        and "coupon_obligation" not in observable.tags
    }
    market_inputs.difference_update(
        coupon_inputs - historical_inputs - non_coupon_inputs
    )


__all__ = ["derive_requirement_hints"]
