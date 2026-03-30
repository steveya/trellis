"""AlgorithmContractValidator — verifies the pricing algorithm matches the route.

Checks:
1. Engine family consistency (MC route → MonteCarloEngine instantiated)
2. Route helper usage (if route_helper primitive specified, it must be called)
3. Discount application (present-value products apply discount factors)
4. Exercise logic presence (American/Bermudan → exercise boundary or LSM)
"""

from __future__ import annotations

import re

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec
from trellis.agent.semantic_validators.base import SemanticFinding


# Engine family → expected code signatures
_ENGINE_SIGNATURES = {
    "monte_carlo": ("MonteCarloEngine", "monte_carlo"),
    "exercise": ("MonteCarloEngine", "longstaff_schwartz", "tsitsiklis_van_roy"),
    "lattice": ("build_rate_lattice", "BinomialTree", "backward_induction", "lattice_backward_induction"),
    "analytical": ("black76_call", "black76_put", "price_quanto_option_analytical"),
    "fft_pricing": ("fft_price", "cos_price"),
    "pde_solver": ("theta_method_1d", "Grid", "BlackScholesOperator"),
    "qmc": ("sobol_normals", "GBM"),
    "copula": ("FactorCopula",),
    "waterfall": ("Waterfall", "Tranche"),
}

# Discount patterns
_DISCOUNT_PATTERNS = (
    "market_state.discount",
    "discount(",
    "discount_factor",
    "df(",
    ".discount(",
)


class AlgorithmContractValidator:
    """Validates that generated code implements the correct pricing algorithm."""

    def validate(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec | None,
    ) -> tuple[SemanticFinding, ...]:
        findings: list[SemanticFinding] = []

        if route_spec is None:
            return ()

        # 1. Engine family consistency
        findings.extend(self._check_engine_family(source, route_spec))

        # 2. Route helper usage
        findings.extend(self._check_route_helper(source, route_spec))

        # 3. Discount application
        findings.extend(self._check_discount_application(source, route_spec))

        # 4. Exercise logic
        findings.extend(self._check_exercise_logic(source, plan))

        return tuple(findings)

    def _check_engine_family(
        self, source: str, route_spec: RouteSpec,
    ) -> list[SemanticFinding]:
        """Verify code uses the expected engine family signatures."""
        engine = route_spec.engine_family
        signatures = _ENGINE_SIGNATURES.get(engine, ())
        if not signatures:
            return []

        found = any(sig in source for sig in signatures)
        if not found:
            return [SemanticFinding(
                validator="algorithm_contract",
                severity="warning",
                category="engine_family_mismatch",
                message=(
                    f"Route '{route_spec.id}' expects engine family '{engine}' "
                    f"(signatures: {', '.join(signatures[:3])}), "
                    f"but none found in generated code."
                ),
            )]
        return []

    def _check_route_helper(
        self, source: str, route_spec: RouteSpec,
    ) -> list[SemanticFinding]:
        """Verify route_helper primitives are actually called."""
        findings = []
        for prim in route_spec.primitives:
            if prim.role == "route_helper" and prim.required:
                if prim.symbol not in source:
                    findings.append(SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="route_helper_not_called",
                        message=(
                            f"Route '{route_spec.id}' requires calling route helper "
                            f"'{prim.symbol}' from '{prim.module}', but it's not "
                            f"referenced in the generated code."
                        ),
                    ))
        return findings

    def _check_discount_application(
        self, source: str, route_spec: RouteSpec,
    ) -> list[SemanticFinding]:
        """Verify discount factors are applied for present-value products."""
        # Only check routes that require discount curve access
        if "discount_curve" not in route_spec.market_data_access.required:
            return []

        found = any(pattern in source for pattern in _DISCOUNT_PATTERNS)
        if not found:
            return [SemanticFinding(
                validator="algorithm_contract",
                severity="warning",
                category="missing_discount_application",
                message=(
                    f"Route '{route_spec.id}' requires discounting but no "
                    f"discount factor application found in generated code."
                ),
            )]
        return []

    def _check_exercise_logic(
        self, source: str, plan: GenerationPlan,
    ) -> list[SemanticFinding]:
        """Verify exercise logic for American/Bermudan products."""
        primitive_plan = plan.primitive_plan
        if primitive_plan is None:
            return []

        # Only relevant for exercise routes
        if primitive_plan.engine_family not in ("exercise", "lattice"):
            return []

        exercise_keywords = (
            "exercise_type", "exercise_fn", "exercise_steps",
            "longstaff_schwartz", "backward_induction",
            "exercise_boundary", "early_exercise",
            "american", "bermudan",
        )
        found = any(kw in source.lower() for kw in exercise_keywords)
        if not found:
            return [SemanticFinding(
                validator="algorithm_contract",
                severity="warning",
                category="missing_exercise_logic",
                message=(
                    "Route requires early-exercise handling but no exercise "
                    "logic (exercise_type, exercise_fn, LSM, etc.) found."
                ),
            )]
        return []
