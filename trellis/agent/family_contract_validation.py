"""Deterministic validation for typed family contracts."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.agent.family_contracts import FamilyContract, parse_family_contract
from trellis.agent.knowledge.methods import is_known_method
from trellis.agent.sensitivity_support import support_for_method
from trellis.core.capabilities import MARKET_DATA


_KNOWN_CAPABILITIES = frozenset(cap.name for cap in MARKET_DATA)
_ALLOWED_PROVENANCE = frozenset({"observed", "derived", "estimated", "user_supplied"})
_ALLOWED_SUPPORT_LEVELS = frozenset({"unsupported", "experimental", "bump_only", "native"})

_QUANTO_REQUIRED_INPUTS = frozenset({
    "domestic_discount_curve",
    "foreign_discount_curve",
    "underlier_spot",
    "fx_spot",
    "underlier_vol",
    "fx_vol",
    "underlier_fx_correlation",
})


@dataclass(frozen=True)
class FamilyContractValidationReport:
    """Validation outcome for one family contract."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized_contract: FamilyContract | None = None

    @property
    def ok(self) -> bool:
        """Whether the contract passed validation."""
        return self.normalized_contract is not None and not self.errors


def validate_family_contract(
    spec,
) -> FamilyContractValidationReport:
    """Parse and validate a family contract."""
    try:
        contract = parse_family_contract(spec)
    except Exception as exc:
        return FamilyContractValidationReport(
            errors=(f"Could not parse family contract: {exc}",),
            warnings=(),
            normalized_contract=None,
        )

    errors: list[str] = []
    warnings: list[str] = []

    _validate_basic_structure(contract, errors, warnings)
    _validate_market_inputs(contract, errors, warnings)
    _validate_methods(contract, errors, warnings)
    _validate_sensitivities(contract, errors, warnings)
    _validate_family_specific_rules(contract, errors, warnings)

    return FamilyContractValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized_contract=contract,
    )


def _validate_basic_structure(
    contract: FamilyContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate top-level required sections."""
    if not contract.family_id:
        errors.append("Product family_id must be provided.")
    if not contract.product.instrument:
        errors.append("Product instrument must be provided.")
    if not contract.product.payoff_family:
        errors.append("Product payoff_family must be provided.")
    if not contract.methods.candidate_methods:
        errors.append("Method contract must declare at least one candidate method.")
    if not contract.market_data.required_inputs:
        errors.append("Market-data contract must declare required_inputs.")
    if not contract.validation.bundle_hints:
        warnings.append(
            f"Family `{contract.family_id}` has no explicit validation bundle hint."
        )


def _validate_market_inputs(
    contract: FamilyContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate input ids, capabilities, and provenance."""
    seen_ids: set[str] = set()
    for input_spec in (*contract.market_data.required_inputs, *contract.market_data.optional_inputs):
        if not input_spec.input_id:
            errors.append(f"Family `{contract.family_id}` has a market input with no input_id.")
            continue
        if input_spec.input_id in seen_ids:
            errors.append(
                f"Family `{contract.family_id}` defines market input `{input_spec.input_id}` more than once."
            )
        seen_ids.add(input_spec.input_id)

        if input_spec.capability and input_spec.capability not in _KNOWN_CAPABILITIES:
            errors.append(
                f"Market input `{input_spec.input_id}` references unknown capability `{input_spec.capability}`."
            )
        if any(p not in _ALLOWED_PROVENANCE for p in input_spec.allowed_provenance):
            errors.append(
                f"Market input `{input_spec.input_id}` uses unsupported provenance labels {input_spec.allowed_provenance}."
            )
        if input_spec.derivable_from and "derived" not in input_spec.allowed_provenance:
            errors.append(
                f"Market input `{input_spec.input_id}` lists derivable_from but does not allow `derived` provenance."
            )
        if not input_spec.connector_hint and input_spec.capability:
            warnings.append(
                f"Market input `{input_spec.input_id}` has capability `{input_spec.capability}` but no connector hint."
            )

    derivable = set(contract.market_data.derivable_inputs)
    missing_derivable = derivable - seen_ids
    if missing_derivable:
        errors.append(
            f"Derivable inputs {sorted(missing_derivable)} are not declared as market inputs."
        )


def _validate_methods(
    contract: FamilyContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate candidate and preferred methods."""
    candidate = set(contract.methods.candidate_methods)
    for method in contract.methods.candidate_methods:
        if not is_known_method(method):
            errors.append(f"Unknown candidate method `{method}` in family `{contract.family_id}`.")
    for method in (*contract.methods.reference_methods, *contract.methods.production_methods):
        if method not in candidate:
            errors.append(
                f"Method `{method}` is listed as reference/production but not in candidate_methods."
            )
    if contract.methods.preferred_method and contract.methods.preferred_method not in candidate:
        errors.append(
            f"Preferred method `{contract.methods.preferred_method}` is not in candidate_methods."
        )
    if not contract.methods.reference_methods and contract.methods.preferred_method is None:
        warnings.append(
            f"Family `{contract.family_id}` has no explicit preferred or reference method."
        )


def _validate_sensitivities(
    contract: FamilyContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate family-level sensitivity claims against current method support."""
    level = contract.sensitivities.support_level
    if level not in _ALLOWED_SUPPORT_LEVELS:
        errors.append(f"Unsupported sensitivity support_level `{level}`.")
    if level == "native":
        errors.append(
            f"Family `{contract.family_id}` must not claim `native` sensitivity support in C0."
        )
    supported = set(contract.sensitivities.supported_measures)
    if not supported:
        return
    union_supported: set[str] = set()
    for method in contract.methods.candidate_methods:
        union_supported.update(support_for_method(method).supported_measures)
    unsupported = sorted(supported - union_supported)
    if unsupported:
        warnings.append(
            f"Family `{contract.family_id}` lists sensitivity measures not currently covered by any candidate method: {unsupported}."
        )


def _validate_family_specific_rules(
    contract: FamilyContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Apply family-specific coherence rules."""
    family_id = contract.family_id
    required_ids = {item.input_id for item in contract.market_data.required_inputs}

    if family_id == "quanto_option":
        missing = sorted(_QUANTO_REQUIRED_INPUTS - required_ids)
        if missing:
            errors.append(
                f"Quanto contracts require the following market inputs: {missing}."
            )
        if "analytical" in contract.methods.candidate_methods and contract.product.exercise_style != "european":
            errors.append("Quanto analytical route is only valid for European exercise.")
        if contract.product.path_dependence == "path_dependent" and "analytical" in contract.methods.candidate_methods:
            errors.append("Path-dependent quanto contracts must not claim an analytical route.")
        if "underlier_fx_correlation" not in required_ids and "analytical" in contract.methods.candidate_methods:
            errors.append(
                "Quanto analytical route requires `underlier_fx_correlation`."
            )
        return

    warnings.append(f"No family-specific validator is registered for `{family_id}`.")
