"""Aggregation and xVA-precursor visitors over execution IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Mapping

from trellis.core.market_state import MarketState
from trellis.execution.ir import ContractExecutionIR
from trellis.execution.runtime import (
    price_dynamic_execution_ir,
    price_static_leg_execution_ir,
)
from trellis.execution.visitors.simulation_bridge import (
    build_future_value_cube_from_execution_ir,
)


def _mapping_proxy(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(sorted(dict(values or {}).items())))


def _tuple_date(values: object) -> tuple[date, ...]:
    return tuple(value for value in (values or ()) if isinstance(value, date))


def _tuple_text(values: object) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _currency_from_execution_ir(ir: ContractExecutionIR) -> str:
    for step in ir.settlement_program.steps:
        if step.currency:
            return str(step.currency).strip().upper()
    for obligation in ir.obligations:
        currency = getattr(obligation, "currency", "")
        if str(currency or "").strip():
            return str(currency).strip().upper()
    return "USD"


@dataclass(frozen=True)
class DiscountedExecutionSummary:
    """Deterministic discounted summary derived from one execution artifact."""

    source_kind: str
    product_family: str
    currency: str
    present_value: float
    payment_dates: tuple[date, ...] = ()
    market_inputs: tuple[str, ...] = ()
    timeline_roles: tuple[str, ...] = ()
    obligation_kinds: tuple[str, ...] = ()
    compute_plan: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", str(self.source_kind or "").strip().lower())
        object.__setattr__(
            self,
            "product_family",
            str(self.product_family or "").strip().lower(),
        )
        object.__setattr__(self, "currency", str(self.currency or "").strip().upper())
        object.__setattr__(self, "present_value", float(self.present_value))
        object.__setattr__(self, "payment_dates", _tuple_date(self.payment_dates))
        object.__setattr__(self, "market_inputs", _tuple_text(self.market_inputs))
        object.__setattr__(self, "timeline_roles", _tuple_text(self.timeline_roles))
        object.__setattr__(self, "obligation_kinds", _tuple_text(self.obligation_kinds))
        object.__setattr__(self, "compute_plan", _mapping_proxy(self.compute_plan))


@dataclass(frozen=True)
class FutureValueExecutionSummary:
    """Future-value and exposure-shape summary derived from execution IR."""

    source_kind: str
    product_family: str
    currency: str
    position_name: str
    observation_dates: tuple[date, ...]
    expected_portfolio_value: tuple[float, ...]
    expected_positive_exposure: tuple[float, ...]
    potential_future_exposure: tuple[tuple[float, tuple[float, ...]], ...] = ()
    current_value: float = 0.0
    terminal_value: float = 0.0
    compute_plan: Mapping[str, object] = field(default_factory=dict)
    position_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", str(self.source_kind or "").strip().lower())
        object.__setattr__(
            self,
            "product_family",
            str(self.product_family or "").strip().lower(),
        )
        object.__setattr__(self, "currency", str(self.currency or "").strip().upper())
        object.__setattr__(self, "position_name", str(self.position_name or "").strip())
        object.__setattr__(self, "observation_dates", _tuple_date(self.observation_dates))
        object.__setattr__(
            self,
            "expected_portfolio_value",
            tuple(float(value) for value in (self.expected_portfolio_value or ())),
        )
        object.__setattr__(
            self,
            "expected_positive_exposure",
            tuple(float(value) for value in (self.expected_positive_exposure or ())),
        )
        object.__setattr__(
            self,
            "potential_future_exposure",
            tuple(
                (
                    float(level),
                    tuple(float(value) for value in values),
                )
                for level, values in (self.potential_future_exposure or ())
            ),
        )
        object.__setattr__(self, "current_value", float(self.current_value))
        object.__setattr__(self, "terminal_value", float(self.terminal_value))
        object.__setattr__(self, "compute_plan", _mapping_proxy(self.compute_plan))
        object.__setattr__(
            self,
            "position_provenance",
            _mapping_proxy(self.position_provenance),
        )


def summarize_discounted_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    method: str | None = None,
    terms: Mapping[str, object] | None = None,
) -> DiscountedExecutionSummary:
    """Summarize the discounted execution artifact through the checked runtime."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")

    source_kind = ir.source_track.source_kind
    if source_kind == "static_leg_contract_ir":
        present_value = price_static_leg_execution_ir(
            ir,
            market_state,
            method=method,
            terms=terms,
        )
    elif source_kind == "dynamic_contract_ir":
        present_value = price_dynamic_execution_ir(
            ir,
            market_state,
            method=method,
            terms=terms,
        )
    else:
        raise ValueError(
            "discounted execution summary requires a static-leg or dynamic execution IR"
        )

    payment_dates = tuple(
        sorted(
            {
                event.event_date
                for event in ir.event_plan.events
                if event.event_kind == "payment" and isinstance(event.event_date, date)
            }
        )
    )
    return DiscountedExecutionSummary(
        source_kind=source_kind,
        product_family=ir.source_track.product_family,
        currency=_currency_from_execution_ir(ir),
        present_value=float(present_value),
        payment_dates=payment_dates,
        market_inputs=tuple(sorted(ir.requirement_hints.market_inputs)),
        timeline_roles=tuple(sorted(ir.requirement_hints.timeline_roles)),
        obligation_kinds=tuple(
            sorted(
                {
                    str(getattr(obligation, "obligation_kind", "") or "").strip()
                    for obligation in ir.obligations
                }
            )
        ),
        compute_plan={
            "aggregation_family": "discounted_execution_summary",
            "source_kind": source_kind,
            "product_family": ir.source_track.product_family,
            "pricing_method": str(method or "").strip().lower() or "default",
        },
    )


def summarize_future_value_execution_ir(
    ir: ContractExecutionIR,
    market_state: MarketState,
    *,
    position_name: str | None = None,
    n_paths: int = 10_000,
    n_steps: int = 120,
    seed: int | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    pfe_levels: tuple[float, ...] = (0.95,),
) -> FutureValueExecutionSummary:
    """Summarize execution-backed future values into xVA-precursor shapes."""
    discounted_summary = summarize_discounted_execution_ir(ir, market_state)
    cube = build_future_value_cube_from_execution_ir(
        ir,
        market_state,
        position_name=position_name,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    portfolio_values = cube.portfolio_values()
    expected_portfolio_value = tuple(float(value) for value in portfolio_values.mean(axis=1))
    exposure_shape = tuple(float(value) for value in cube.expected_positive_exposure())
    pfe = tuple(
        (
            float(level),
            tuple(float(value) for value in cube.potential_future_exposure(level)),
        )
        for level in pfe_levels
    )
    resolved_name = str(position_name or cube.position_names[0]).strip()
    return FutureValueExecutionSummary(
        source_kind=ir.source_track.source_kind,
        product_family=ir.source_track.product_family,
        currency=_currency_from_execution_ir(ir),
        position_name=resolved_name,
        observation_dates=cube.observation_dates,
        expected_portfolio_value=expected_portfolio_value,
        expected_positive_exposure=exposure_shape,
        potential_future_exposure=pfe,
        current_value=discounted_summary.present_value,
        terminal_value=expected_portfolio_value[-1] if expected_portfolio_value else 0.0,
        compute_plan={
            **cube.compute_plan,
            "aggregation_family": "future_value_execution_summary",
        },
        position_provenance=cube.position_provenance.get(resolved_name, {}),
    )


__all__ = [
    "DiscountedExecutionSummary",
    "FutureValueExecutionSummary",
    "summarize_discounted_execution_ir",
    "summarize_future_value_execution_ir",
]
