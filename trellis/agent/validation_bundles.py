"""Deterministic validation-bundle selection and execution.

This module turns route/product-family validation into an explicit executable
policy layer. It is intentionally conservative: bundles select the checks that
should apply, and execution can still skip checks that lack the runtime inputs
needed for a safe deterministic evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trellis.agent.assembly_tools import select_invariant_pack
from trellis.agent.knowledge.methods import normalize_method


UNIVERSAL_CHECKS = {"check_non_negativity", "check_price_sanity"}
NO_ARBITRAGE_CHECKS = {
    "check_vol_monotonicity",
    "check_vol_sensitivity",
    "check_rate_monotonicity",
    "check_zero_vol_intrinsic",
}
PRODUCT_FAMILY_CHECKS = {"check_bounded_by_reference"}


@dataclass(frozen=True)
class ValidationBundle:
    """Deterministic validation policy for one route/product family."""

    bundle_id: str
    instrument_type: str | None
    method: str
    checks: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ValidationBundleExecution:
    """Execution result for one deterministic validation bundle."""

    failures: tuple[str, ...]
    executed_checks: tuple[str, ...]
    skipped_checks: tuple[str, ...]


def select_validation_bundle(
    *,
    instrument_type: str | None,
    method: str,
    product_ir=None,
) -> ValidationBundle:
    """Select an executable deterministic validation bundle."""
    normalized_method = normalize_method(method)
    normalized_instrument = (
        instrument_type or getattr(product_ir, "instrument", None) or "unknown"
    )
    pack = select_invariant_pack(
        instrument_type=normalized_instrument,
        method=normalized_method,
        product_ir=product_ir,
    )
    categories = _categorize_checks(pack.checks)
    bundle_id = f"{normalized_method}:{normalized_instrument}"
    return ValidationBundle(
        bundle_id=bundle_id,
        instrument_type=normalized_instrument,
        method=normalized_method,
        checks=pack.checks,
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
) -> ValidationBundleExecution:
    """Execute the deterministic checks selected for a validation bundle."""
    from trellis.agent import invariants

    failures: list[str] = []
    executed_checks: list[str] = []
    skipped_checks: list[str] = []

    for check in bundle.checks:
        if failures and check not in UNIVERSAL_CHECKS:
            skipped_checks.append(check)
            continue

        if check in UNIVERSAL_CHECKS:
            if test_payoff is None or market_state is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            failures.extend(getattr(invariants, check)(test_payoff, market_state))
            continue

        if check == "check_bounded_by_reference":
            if validation_level not in {"standard", "thorough"}:
                skipped_checks.append(check)
                continue
            if payoff_factory is None or market_state_factory is None or reference_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            failures.extend(
                invariants.check_bounded_by_reference(
                    payoff_factory,
                    reference_factory,
                    market_state_factory,
                    rate_range=(0.02, 0.05, 0.08),
                    relation="<=",
                )
            )
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
                failures.extend(invariants.check_vol_sensitivity(payoff_factory, market_state_factory))
            elif check == "check_vol_monotonicity":
                failures.extend(invariants.check_vol_monotonicity(payoff_factory, market_state_factory))
            else:
                failures.extend(invariants.check_rate_monotonicity(payoff_factory, market_state_factory))
            continue

        if check == "check_zero_vol_intrinsic":
            if validation_level != "thorough":
                skipped_checks.append(check)
                continue
            if intrinsic_fn is None or market_state_factory is None or payoff_factory is None:
                skipped_checks.append(check)
                continue
            executed_checks.append(check)
            failures.extend(
                invariants.check_zero_vol_intrinsic(
                    payoff_factory,
                    market_state_factory,
                    intrinsic_fn,
                )
            )
            continue

        skipped_checks.append(check)

    return ValidationBundleExecution(
        failures=tuple(failures),
        executed_checks=tuple(executed_checks),
        skipped_checks=tuple(skipped_checks),
    )


def _categorize_checks(checks: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    categories = {
        "universal": tuple(check for check in checks if check in UNIVERSAL_CHECKS),
        "no_arbitrage": tuple(check for check in checks if check in NO_ARBITRAGE_CHECKS),
        "product_family": tuple(check for check in checks if check in PRODUCT_FAMILY_CHECKS),
    }
    return {
        key: value
        for key, value in categories.items()
        if value
    }
