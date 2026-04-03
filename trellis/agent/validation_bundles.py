"""Select and run the right set of validation checks for a given pricing route.

Given a pricing method and instrument type, this module picks which checks
to run (e.g. non-negativity, volatility sensitivity, no-arbitrage bounds)
and executes them. Checks that lack required inputs are skipped gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trellis.agent.assembly_tools import normalize_comparison_relation, select_invariant_pack
from trellis.agent.knowledge.methods import normalize_method


UNIVERSAL_CHECKS = {"check_non_negativity", "check_price_sanity"}
NO_ARBITRAGE_CHECKS = {
    "check_vol_monotonicity",
    "check_vol_sensitivity",
    "check_rate_monotonicity",
    "check_zero_vol_intrinsic",
}
PRODUCT_FAMILY_CHECKS = {"check_bounded_by_reference"}
ROUTE_SPECIFIC_CHECKS = {"check_rate_style_swaption_helper_consistency"}
_KNOWN_FAMILY_CHECKS = {
    "credit_default_swap": (
        "check_cds_spread_quote_normalization",
        "check_cds_credit_curve_sensitivity",
    ),
    "cds": (
        "check_cds_spread_quote_normalization",
        "check_cds_credit_curve_sensitivity",
    ),
    "quanto_option": (
        "check_quanto_required_inputs",
        "check_quanto_cross_currency_semantics",
    ),
}


@dataclass(frozen=True)
class ValidationBundle:
    """The set of validation checks selected for a specific pricing method and instrument."""

    bundle_id: str
    instrument_type: str | None
    method: str
    checks: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ValidationBundleExecution:
    """Result of running a validation bundle: which checks passed, failed, or were skipped."""

    failures: tuple[str, ...]
    failure_details: tuple[Any, ...]
    executed_checks: tuple[str, ...]
    skipped_checks: tuple[str, ...]


def select_validation_bundle(
    *,
    instrument_type: str | None,
    method: str,
    product_ir=None,
    family_blueprint=None,
    semantic_blueprint=None,
) -> ValidationBundle:
    """Pick the validation checks that apply to the given method and instrument type."""
    normalized_method = normalize_method(method)
    normalized_instrument = _resolve_validation_instrument(
        instrument_type=instrument_type,
        product_ir=product_ir,
        family_blueprint=family_blueprint,
        semantic_blueprint=semantic_blueprint,
    )
    pack = select_invariant_pack(
        instrument_type=normalized_instrument,
        method=normalized_method,
        product_ir=product_ir,
    )
    checks = tuple(
        dict.fromkeys(
            (
                *pack.checks,
                *_family_checks_for(
                    normalized_instrument,
                    family_blueprint=family_blueprint,
                    semantic_blueprint=semantic_blueprint,
                ),
            )
        )
    )
    categories = _categorize_checks(checks)
    bundle_id = f"{normalized_method}:{normalized_instrument}"
    return ValidationBundle(
        bundle_id=bundle_id,
        instrument_type=normalized_instrument,
        method=normalized_method,
        checks=checks,
        categories=categories,
    )


def execute_validation_bundle(
    bundle: ValidationBundle,
    *,
    validation_level: str,
    test_payoff: Any,
    market_state: Any,
    payoff_factory: Callable[[], Any] | None = None,
    market_state_factory: Callable[..., Any] | None = None,
    reference_factory: Callable[[], Any] | None = None,
    intrinsic_fn: Callable[[Any], float] | None = None,
    check_relations: Mapping[str, str] | None = None,
) -> ValidationBundleExecution:
    """Execute the deterministic checks selected for a validation bundle."""
    from trellis.agent import invariants

    failures: list[str] = []
    failure_details: list[Any] = []
    executed_checks: list[str] = []
    skipped_checks: list[str] = []
    check_relations = check_relations or {}
    vol_direction = _vol_monotonicity_direction(bundle.instrument_type)

    for check in bundle.checks:
        if failures and check not in UNIVERSAL_CHECKS:
            skipped_checks.append(check)
            continue

        if check in UNIVERSAL_CHECKS:
            if test_payoff is None or market_state is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                getattr(invariants, check),
                check,
                test_payoff,
                market_state,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check == "check_bounded_by_reference":
            if validation_level not in {"standard", "thorough"}:
                skipped_checks.append(check)
                continue
            if payoff_factory is None or market_state_factory is None or reference_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            bound_relation = normalize_comparison_relation(
                check_relations.get(check),
                default="<=",
            )
            result = _run_check_with_diagnostics(
                invariants.check_bounded_by_reference,
                check,
                payoff_factory,
                reference_factory,
                market_state_factory,
                rate_range=(0.02, 0.05, 0.08),
                relation=bound_relation,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check == "check_quanto_required_inputs":
            if test_payoff is None or market_state is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                invariants.check_quanto_required_inputs,
                check,
                test_payoff,
                market_state,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check == "check_quanto_cross_currency_semantics":
            if payoff_factory is None or market_state_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                invariants.check_quanto_cross_currency_semantics,
                check,
                payoff_factory,
                market_state_factory,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check in {
            "check_cds_spread_quote_normalization",
            "check_cds_credit_curve_sensitivity",
        }:
            if payoff_factory is None or market_state_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                getattr(invariants, check),
                check,
                payoff_factory,
                market_state_factory,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check == "check_rate_style_swaption_helper_consistency":
            if payoff_factory is None or market_state_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                invariants.check_rate_style_swaption_helper_consistency,
                check,
                payoff_factory,
                market_state_factory,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check in {"check_vol_sensitivity", "check_vol_monotonicity", "check_rate_monotonicity"}:
            if validation_level == "fast":
                skipped_checks.append(check)
                continue
            if payoff_factory is None or market_state_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            if check == "check_vol_sensitivity":
                result = _run_check_with_diagnostics(
                    invariants.check_vol_sensitivity,
                    check,
                    payoff_factory,
                    market_state_factory,
                )
            elif check == "check_vol_monotonicity":
                result = _run_check_with_diagnostics(
                    invariants.check_vol_monotonicity,
                    check,
                    payoff_factory,
                    market_state_factory,
                    expected_direction=vol_direction,
                )
            else:
                result = _run_check_with_diagnostics(
                    invariants.check_rate_monotonicity,
                    check,
                    payoff_factory,
                    market_state_factory,
                )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        if check == "check_zero_vol_intrinsic":
            if validation_level != "thorough":
                skipped_checks.append(check)
                continue
            if intrinsic_fn is None or market_state_factory is None or payoff_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            result = _run_check_with_diagnostics(
                invariants.check_zero_vol_intrinsic,
                check,
                payoff_factory,
                market_state_factory,
                intrinsic_fn,
            )
            failures.extend(result[0])
            failure_details.extend(result[1])
            continue

        skipped_checks.append(check)

    return ValidationBundleExecution(
        failures=tuple(failures),
        failure_details=tuple(failure_details),
        executed_checks=tuple(executed_checks),
        skipped_checks=tuple(skipped_checks),
    )


def _run_check_with_diagnostics(check_fn, check_name: str, *args, **kwargs) -> tuple[list[str], list[Any]]:
    """Execute one invariant check and normalize structured diagnostics."""
    from trellis.agent.invariants import InvariantFailure

    try:
        raw_failures = check_fn(*args, return_diagnostics=True, **kwargs)
    except TypeError as exc:
        if "return_diagnostics" not in str(exc):
            raise
        raw_failures = check_fn(*args, **kwargs)

    failure_messages: list[str] = []
    failure_details: list[Any] = []
    for failure in raw_failures or ():
        if isinstance(failure, InvariantFailure):
            detail = failure
        else:
            detail = InvariantFailure(check=check_name, message=str(failure))
        failure_messages.append(detail.message)
        failure_details.append(detail)
    return failure_messages, failure_details


def _categorize_checks(checks: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    categories = {
        "universal": tuple(check for check in checks if check in UNIVERSAL_CHECKS),
        "no_arbitrage": tuple(check for check in checks if check in NO_ARBITRAGE_CHECKS),
        "product_family": tuple(
            check
            for check in checks
            if check in (PRODUCT_FAMILY_CHECKS | set().union(*_KNOWN_FAMILY_CHECKS.values()))
        ),
        "route_specific": tuple(check for check in checks if check in ROUTE_SPECIFIC_CHECKS),
    }
    return {
        key: value
        for key, value in categories.items()
        if value
    }


def _resolve_validation_instrument(
    *,
    instrument_type: str | None,
    product_ir=None,
    family_blueprint=None,
    semantic_blueprint=None,
) -> str:
    """Resolve the most specific instrument id available for validation."""
    normalized = (instrument_type or "").strip().lower().replace(" ", "_")
    if normalized and normalized != "unknown":
        return normalized
    if family_blueprint is not None and getattr(family_blueprint, "family_id", None):
        return str(family_blueprint.family_id).strip().lower().replace(" ", "_")
    if semantic_blueprint is not None and getattr(semantic_blueprint, "semantic_id", None):
        return str(semantic_blueprint.semantic_id).strip().lower().replace(" ", "_")
    if product_ir is not None and getattr(product_ir, "instrument", None):
        return str(product_ir.instrument).strip().lower().replace(" ", "_")
    return "unknown"


def _family_checks_for(
    normalized_instrument: str,
    *,
    family_blueprint=None,
    semantic_blueprint=None,
) -> tuple[str, ...]:
    """Return extra family-specific validation checks for known contract families.

    Supports both legacy ``FamilyImplementationBlueprint`` (``family_checks``)
    and unified ``SemanticImplementationBlueprint`` (``semantic_checks``)
    contracts.
    """
    # Legacy family blueprint path.
    contract = getattr(family_blueprint, "contract", None)
    validation = getattr(contract, "validation", None)
    if validation is not None and getattr(validation, "family_checks", None):
        return tuple(validation.family_checks)
    # Unified semantic blueprint path.
    sem_contract = getattr(semantic_blueprint, "contract", None)
    sem_validation = getattr(sem_contract, "validation", None)
    if sem_validation is not None and getattr(sem_validation, "semantic_checks", None):
        return tuple(sem_validation.semantic_checks)
    return _KNOWN_FAMILY_CHECKS.get(normalized_instrument, ())


def _vol_monotonicity_direction(instrument_type: str | None) -> str:
    normalized = (instrument_type or "").strip().lower()
    if normalized == "callable_bond":
        return "decreasing"
    return "increasing"
