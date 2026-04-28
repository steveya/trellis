"""Compiler entrypoints for the XIR.0 execution seam.

These entrypoints intentionally do not perform product lowering yet.  They
accept upstream semantic objects and return an explicit empty/unsupported
execution artifact so downstream code can wire against the seam without making
pricing-behavior claims.
"""

from __future__ import annotations

from trellis.execution.ir import ContractExecutionIR, SourceTrack


class UnsupportedExecutionSemantics(ValueError):
    """Raised when fail-closed execution lowering is requested for XIR.0."""


def compile_contract_execution_ir(
    source: object,
    *,
    source_track: SourceTrack | None = None,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile an upstream semantic object into a conservative execution IR.

    XIR.0 is an authority boundary only.  Until later XIR tickets add concrete
    visitors/lowerers, the compiler returns an empty execution IR with an
    explicit unsupported reason instead of inventing schedules, obligations, or
    pricing semantics.
    """
    resolved_source_track = source_track or infer_source_track(source)
    reason = (
        f"execution lowering not implemented for {resolved_source_track.source_kind}"
    )
    if fail_on_unsupported:
        raise UnsupportedExecutionSemantics(reason)
    return ContractExecutionIR.empty(
        source_track=resolved_source_track,
        unsupported_reasons=(reason,),
    )


def compile_semantic_execution_ir(
    semantic_contract: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a semantic contract onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        semantic_contract,
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_contract_ir_execution_ir(
    contract_ir: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a payoff-expression ContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        contract_ir,
        source_track=infer_source_track(contract_ir, default_source_kind="contract_ir"),
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_static_leg_execution_ir(
    static_leg_contract_ir: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a StaticLegContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        static_leg_contract_ir,
        source_track=infer_source_track(
            static_leg_contract_ir,
            default_source_kind="static_leg_contract_ir",
        ),
        fail_on_unsupported=fail_on_unsupported,
    )


def compile_dynamic_execution_ir(
    dynamic_contract_ir: object,
    *,
    fail_on_unsupported: bool = False,
) -> ContractExecutionIR:
    """Compile a DynamicContractIR onto the XIR.0 seam."""
    return compile_contract_execution_ir(
        dynamic_contract_ir,
        source_track=infer_source_track(
            dynamic_contract_ir,
            default_source_kind="dynamic_contract_ir",
        ),
        fail_on_unsupported=fail_on_unsupported,
    )


def infer_source_track(
    source: object,
    *,
    default_source_kind: str | None = None,
) -> SourceTrack:
    """Infer minimal source metadata without importing agent-owned modules."""
    source_kind = (
        _attr_text(source, "source_kind")
        or _attr_text(source, "track")
        or default_source_kind
        or _source_kind_from_type(source)
    )
    semantic_id = (
        _attr_text(source, "semantic_id")
        or _attr_text(source, "declaration_id")
        or _attr_text(source, "contract_id")
    )
    product = getattr(source, "product", None)
    instrument_class = (
        _attr_text(product, "instrument_class")
        or _attr_text(source, "instrument_class")
        or _attr_text(source, "instrument_type")
    )
    product_family = (
        _attr_text(product, "payoff_family")
        or _attr_text(source, "payoff_family")
        or _attr_text(source, "product_family")
    )
    source_ref = (
        _attr_text(source, "source_ref")
        or (f"{source_kind}:{semantic_id}" if semantic_id else source_kind)
    )
    return SourceTrack(
        source_kind=source_kind,
        semantic_id=semantic_id,
        product_family=product_family,
        instrument_class=instrument_class,
        source_ref=source_ref,
    )


def _attr_text(source: object, name: str) -> str:
    if source is None:
        return ""
    value = getattr(source, name, "")
    if callable(value):
        return ""
    return str(value or "").strip()


def _source_kind_from_type(source: object) -> str:
    name = type(source).__name__.lower()
    if "staticleg" in name or "static_leg" in name:
        return "static_leg_contract_ir"
    if "dynamic" in name:
        return "dynamic_contract_ir"
    if "contractir" in name or "contract_ir" in name:
        return "contract_ir"
    if "semantic" in name:
        return "semantic_contract"
    return "unknown"
