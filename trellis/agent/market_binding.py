"""Required-data and market-binding compilation for semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.agent.valuation_context import ValuationContext


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


@dataclass(frozen=True)
class DataRequirement:
    """One contract-level market-data requirement."""

    input_id: str
    capability: str = ""
    aliases: tuple[str, ...] = ()
    connector_hint: str = ""
    allowed_provenance: tuple[str, ...] = ()
    optional: bool = False


@dataclass(frozen=True)
class RequiredDataSpec:
    """Compiled market-data requirements independent of route codegen."""

    required_inputs: tuple[DataRequirement, ...] = ()
    optional_inputs: tuple[DataRequirement, ...] = ()
    derivable_inputs: tuple[str, ...] = ()
    estimation_policy: tuple[str, ...] = ()
    provenance_requirements: tuple[str, ...] = ()
    missing_data_error_policy: tuple[str, ...] = ()

    @property
    def required_input_ids(self) -> tuple[str, ...]:
        """Return the canonical ordered required-input ids."""
        return tuple(item.input_id for item in self.required_inputs)

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        """Return the ordered capability set required at valuation time."""
        capabilities: list[str] = []
        for item in self.required_inputs:
            if item.capability and item.capability not in capabilities:
                capabilities.append(item.capability)
        return tuple(capabilities)

    def to_estimation_hints(self) -> dict[str, object]:
        """Project the spec into the legacy estimation-hint mapping."""
        return {
            "derivable_inputs": list(self.derivable_inputs),
            "estimation_policy": list(self.estimation_policy),
            "provenance_requirements": list(self.provenance_requirements),
            "missing_data_error_policy": list(self.missing_data_error_policy),
        }


@dataclass(frozen=True)
class MarketBinding:
    """One binding target from required data onto the valuation context."""

    input_id: str
    capability: str = ""
    binding_source: str = "runtime_connector_resolution"
    aliases: tuple[str, ...] = ()
    connector_hint: str = ""
    allowed_provenance: tuple[str, ...] = ()
    optional: bool = False


@dataclass(frozen=True)
class MarketBindingSpec:
    """Compiled market-binding policy ready before route codegen."""

    market_source: str
    market_snapshot_handle: str = ""
    model_spec: str | None = None
    measure_spec: str = "risk_neutral"
    discounting_policy: str = "contract_convention_discounting"
    collateral_policy: str | None = None
    reporting_currency: str = ""
    requested_outputs: tuple[str, ...] = ()
    bindings: tuple[MarketBinding, ...] = ()
    derivable_inputs: tuple[str, ...] = ()

    def to_connector_binding_hints(self) -> dict[str, object]:
        """Project the binding spec into the legacy connector-hint mapping."""
        return {
            item.input_id: {
                "capability": item.capability,
                "aliases": list(item.aliases),
                "connector_hint": item.connector_hint,
                "allowed_provenance": list(item.allowed_provenance),
                "binding_source": item.binding_source,
            }
            for item in self.bindings
        }


def build_required_data_spec(contract) -> RequiredDataSpec:
    """Compile the contract's market-data section into a stable requirement spec."""
    return RequiredDataSpec(
        required_inputs=tuple(
            DataRequirement(
                input_id=item.input_id,
                capability=item.capability or "",
                aliases=_string_tuple(item.aliases),
                connector_hint=item.connector_hint,
                allowed_provenance=_string_tuple(item.allowed_provenance),
                optional=False,
            )
            for item in contract.market_data.required_inputs
        ),
        optional_inputs=tuple(
            DataRequirement(
                input_id=item.input_id,
                capability=item.capability or "",
                aliases=_string_tuple(item.aliases),
                connector_hint=item.connector_hint,
                allowed_provenance=_string_tuple(item.allowed_provenance),
                optional=True,
            )
            for item in contract.market_data.optional_inputs
        ),
        derivable_inputs=_string_tuple(contract.market_data.derivable_inputs),
        estimation_policy=_string_tuple(contract.market_data.estimation_policy),
        provenance_requirements=_string_tuple(contract.market_data.provenance_requirements),
        missing_data_error_policy=_string_tuple(contract.market_data.missing_data_error_policy),
    )


def build_market_binding_spec(
    contract,
    *,
    valuation_context: ValuationContext,
    required_data_spec: RequiredDataSpec | None = None,
) -> MarketBindingSpec:
    """Compile route-independent market-binding policy from contract plus valuation context."""
    required_spec = required_data_spec or build_required_data_spec(contract)
    binding_source = (
        "valuation_context.market_snapshot"
        if valuation_context.market_snapshot is not None
        else "runtime_connector_resolution"
    )
    bindings = tuple(
        MarketBinding(
            input_id=item.input_id,
            capability=item.capability,
            binding_source=binding_source,
            aliases=item.aliases,
            connector_hint=item.connector_hint,
            allowed_provenance=item.allowed_provenance,
            optional=item.optional,
        )
        for item in (*required_spec.required_inputs, *required_spec.optional_inputs)
    )
    return MarketBindingSpec(
        market_source=valuation_context.market_source,
        market_snapshot_handle=valuation_context.market_snapshot_handle,
        model_spec=valuation_context.model_spec,
        measure_spec=valuation_context.measure_spec,
        discounting_policy=valuation_context.discounting_policy,
        collateral_policy=valuation_context.collateral_policy,
        reporting_currency=valuation_context.reporting_policy.reporting_currency,
        requested_outputs=valuation_context.requested_outputs,
        bindings=bindings,
        derivable_inputs=required_spec.derivable_inputs,
    )


def required_data_spec_summary(spec: RequiredDataSpec) -> dict[str, object]:
    """Return a compact YAML-safe summary of required-data compilation."""
    return {
        "required_inputs": [
            {
                "input_id": item.input_id,
                "capability": item.capability,
                "aliases": list(item.aliases),
                "optional": item.optional,
            }
            for item in spec.required_inputs
        ],
        "optional_inputs": [
            {
                "input_id": item.input_id,
                "capability": item.capability,
                "aliases": list(item.aliases),
                "optional": item.optional,
            }
            for item in spec.optional_inputs
        ],
        "derivable_inputs": list(spec.derivable_inputs),
        "estimation_policy": list(spec.estimation_policy),
        "provenance_requirements": list(spec.provenance_requirements),
        "missing_data_error_policy": list(spec.missing_data_error_policy),
    }


def market_binding_spec_summary(spec: MarketBindingSpec) -> dict[str, object]:
    """Return a compact YAML-safe summary of compiled market bindings."""
    return {
        "market_source": spec.market_source,
        "market_snapshot_handle": spec.market_snapshot_handle,
        "model_spec": spec.model_spec,
        "measure_spec": spec.measure_spec,
        "discounting_policy": spec.discounting_policy,
        "collateral_policy": spec.collateral_policy,
        "reporting_currency": spec.reporting_currency,
        "requested_outputs": list(spec.requested_outputs),
        "bindings": [
            {
                "input_id": item.input_id,
                "capability": item.capability,
                "binding_source": item.binding_source,
                "optional": item.optional,
            }
            for item in spec.bindings
        ],
        "derivable_inputs": list(spec.derivable_inputs),
    }
