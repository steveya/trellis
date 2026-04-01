"""Valuation-context normalization for semantic-contract compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

from trellis.agent.sensitivity_support import normalize_requested_outputs


def _string_tuple(values) -> tuple[str, ...]:
    """Return a deduplicated tuple of normalized strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize_label(value: str | None, *, fallback: str) -> str:
    """Normalize a policy/source label to a stable lowercase token."""
    text = str(value or "").strip().lower().replace(" ", "_")
    return text or fallback


def _market_source_label(snapshot) -> str:
    """Return a stable source label for one market snapshot binding."""
    if snapshot is None:
        return "unbound_market_snapshot"
    source = getattr(snapshot, "source", None) or getattr(snapshot, "data_source", None)
    if source:
        return _normalize_label(str(source), fallback="provided_market_snapshot")
    return f"provided_{type(snapshot).__name__.lower()}"


def _market_snapshot_handle(snapshot) -> str:
    """Return a lightweight handle string for traceable snapshot provenance."""
    if snapshot is None:
        return ""
    for attr in ("snapshot_id", "market_snapshot_id", "trace_id", "source"):
        value = getattr(snapshot, attr, None)
        if value:
            return str(value).strip()
    return type(snapshot).__name__


@dataclass(frozen=True)
class ReportingPolicy:
    """Tranche-1 reporting and FX conversion policy."""

    reporting_currency: str = ""
    fx_conversion_policy: str = "native_payout"
    include_native_currency: bool = True


@dataclass(frozen=True)
class ValuationContext:
    """Valuation policy and market-binding context compiled separately from semantics."""

    market_source: str = "unbound_market_snapshot"
    market_snapshot_handle: str = ""
    market_snapshot: object | None = None
    model_spec: str | None = None
    measure_spec: str = "risk_neutral"
    discounting_policy: str = "contract_convention_discounting"
    collateral_policy: str | None = None
    reporting_policy: ReportingPolicy = field(default_factory=ReportingPolicy)
    requested_outputs: tuple[str, ...] = ()

    def __post_init__(self):
        """Normalize policy labels and requested outputs into stable tuples."""
        object.__setattr__(
            self,
            "market_source",
            _normalize_label(
                self.market_source,
                fallback=_market_source_label(self.market_snapshot),
            ),
        )
        object.__setattr__(
            self,
            "market_snapshot_handle",
            str(self.market_snapshot_handle or _market_snapshot_handle(self.market_snapshot)).strip(),
        )
        object.__setattr__(
            self,
            "measure_spec",
            _normalize_label(self.measure_spec, fallback="risk_neutral"),
        )
        object.__setattr__(
            self,
            "discounting_policy",
            _normalize_label(self.discounting_policy, fallback="contract_convention_discounting"),
        )
        object.__setattr__(
            self,
            "requested_outputs",
            normalize_requested_outputs(self.requested_outputs),
        )


def build_valuation_context(
    *,
    market_snapshot=None,
    model_spec: str | None = None,
    measure_spec: str | None = None,
    discounting_policy: str | None = None,
    collateral_policy: str | None = None,
    reporting_currency: str | None = None,
    requested_outputs=None,
) -> ValuationContext:
    """Construct a normalized valuation context from front-door request data."""
    return ValuationContext(
        market_source=_market_source_label(market_snapshot),
        market_snapshot_handle=_market_snapshot_handle(market_snapshot),
        market_snapshot=market_snapshot,
        model_spec=model_spec,
        measure_spec=measure_spec or "risk_neutral",
        discounting_policy=discounting_policy or "contract_convention_discounting",
        collateral_policy=collateral_policy,
        reporting_policy=ReportingPolicy(
            reporting_currency=str(reporting_currency or "").strip(),
        ),
        requested_outputs=_string_tuple(requested_outputs),
    )


def normalize_valuation_context(
    valuation_context: ValuationContext | None = None,
    *,
    market_snapshot=None,
    model_spec: str | None = None,
    measure_spec: str | None = None,
    discounting_policy: str | None = None,
    collateral_policy: str | None = None,
    reporting_currency: str | None = None,
    requested_outputs=None,
    requested_measures=None,
) -> ValuationContext:
    """Return one normalized valuation context, merging legacy request inputs."""
    normalized_outputs = normalize_requested_outputs(
        requested_outputs if requested_outputs is not None else requested_measures
    )
    if valuation_context is None:
        return build_valuation_context(
            market_snapshot=market_snapshot,
            model_spec=model_spec,
            measure_spec=measure_spec,
            discounting_policy=discounting_policy,
            collateral_policy=collateral_policy,
            reporting_currency=reporting_currency,
            requested_outputs=normalized_outputs,
        )

    merged_outputs = valuation_context.requested_outputs or ()
    for output in normalized_outputs:
        if output not in merged_outputs:
            merged_outputs += (output,)
    return ValuationContext(
        market_source=valuation_context.market_source,
        market_snapshot_handle=valuation_context.market_snapshot_handle,
        market_snapshot=valuation_context.market_snapshot or market_snapshot,
        model_spec=valuation_context.model_spec or model_spec,
        measure_spec=valuation_context.measure_spec or measure_spec or "risk_neutral",
        discounting_policy=valuation_context.discounting_policy or discounting_policy or "contract_convention_discounting",
        collateral_policy=valuation_context.collateral_policy or collateral_policy,
        reporting_policy=ReportingPolicy(
            reporting_currency=valuation_context.reporting_policy.reporting_currency or str(reporting_currency or "").strip(),
            fx_conversion_policy=valuation_context.reporting_policy.fx_conversion_policy,
            include_native_currency=valuation_context.reporting_policy.include_native_currency,
        ),
        requested_outputs=merged_outputs,
    )


def valuation_context_summary(context: ValuationContext) -> dict[str, object]:
    """Return a compact YAML-safe summary of one valuation context."""
    return {
        "market_source": context.market_source,
        "market_snapshot_handle": context.market_snapshot_handle,
        "model_spec": context.model_spec,
        "measure_spec": context.measure_spec,
        "discounting_policy": context.discounting_policy,
        "collateral_policy": context.collateral_policy,
        "reporting_policy": {
            "reporting_currency": context.reporting_policy.reporting_currency,
            "fx_conversion_policy": context.reporting_policy.fx_conversion_policy,
            "include_native_currency": context.reporting_policy.include_native_currency,
        },
        "requested_outputs": list(context.requested_outputs),
    }
