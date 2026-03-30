"""Semantic validators for generated pricing code.

Public API: ``validate_generated_semantics()`` runs all three validators
and returns a merged report.  Designed to be inserted into the executor
pipeline after ``validate_semantics()`` (gate 3), sharing its retry slot.
"""

from __future__ import annotations

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec, find_route_by_id
from trellis.agent.semantic_validators.algorithm_contract import AlgorithmContractValidator
from trellis.agent.semantic_validators.base import (
    SemanticFinding,
    SemanticValidationReport,
    SemanticValidator,
)
from trellis.agent.semantic_validators.market_data import MarketDataValidator
from trellis.agent.semantic_validators.parameter_binding import ParameterBindingValidator


_VALIDATORS: tuple[SemanticValidator, ...] = (
    MarketDataValidator(),
    ParameterBindingValidator(),
    AlgorithmContractValidator(),
)

# Validator mode state — starts as "warning", promoted to "blocking"
# when false-positive rate < 5% over 50+ runs.  Stored in-memory;
# persistence handled by reflect.py auto-promotion logic.
_VALIDATOR_MODES: dict[str, str] = {
    "market_data": "warning",
    "parameter_binding": "warning",
    "algorithm_contract": "warning",
}


def validate_generated_semantics(
    source: str,
    plan: GenerationPlan,
    route_spec: RouteSpec | None = None,
    *,
    mode: str | None = None,
) -> SemanticValidationReport:
    """Run all semantic validators and return a merged report.

    Parameters
    ----------
    source : str
        The generated Python source code.
    plan : GenerationPlan
        The generation plan used for this build.
    route_spec : RouteSpec | None
        The selected route spec.  If None, attempts to look up from the
        plan's primitive_plan.route.
    mode : str | None
        Override mode for all validators.  If None, uses per-validator modes.

    Returns
    -------
    SemanticValidationReport
        Merged findings from all validators.
    """
    if route_spec is None and plan.primitive_plan is not None:
        route_spec = find_route_by_id(plan.primitive_plan.route)

    all_findings: list[SemanticFinding] = []
    effective_mode = mode or "warning"

    for validator in _VALIDATORS:
        findings = validator.validate(source, plan, route_spec)
        # Apply per-validator mode if no override
        if mode is None:
            validator_name = findings[0].validator if findings else ""
            validator_mode = _VALIDATOR_MODES.get(validator_name, "warning")
            if validator_mode == "warning":
                # Downgrade errors to warnings
                findings = tuple(
                    SemanticFinding(
                        validator=f.validator,
                        severity="warning",
                        category=f.category,
                        message=f.message,
                        line=f.line,
                        evidence=f.evidence,
                    )
                    for f in findings
                )
            else:
                effective_mode = "blocking"

        all_findings.extend(findings)

    return SemanticValidationReport(
        findings=tuple(all_findings),
        mode=effective_mode,
    )


def set_validator_mode(validator_name: str, mode: str) -> None:
    """Set the mode for a specific validator ("warning" or "blocking")."""
    if validator_name in _VALIDATOR_MODES:
        _VALIDATOR_MODES[validator_name] = mode


def get_validator_modes() -> dict[str, str]:
    """Return current validator modes."""
    return dict(_VALIDATOR_MODES)
