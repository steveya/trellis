"""AlgorithmContractValidator — verifies the pricing algorithm matches the route.

Checks:
1. Engine family consistency (MC route → MonteCarloEngine instantiated)
2. Route helper usage (if route_helper primitive specified, it must be called)
3. Discount application (present-value products apply discount factors)
4. Exercise logic presence (American/Bermudan → exercise boundary or LSM)
"""

from __future__ import annotations

import ast
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

_EXACT_HELPER_SIGNATURES = {
    "price_cds_analytical": {
        "min_positional_args": 0,
        "keyword_only": True,
        "required_parameters": (
            "notional",
            "spread_quote",
            "recovery",
            "schedule",
            "credit_curve",
            "discount_curve",
        ),
        "required_keyword_groups": (
            frozenset({
                "notional",
                "spread_quote",
                "recovery",
                "schedule",
                "credit_curve",
                "discount_curve",
            }),
        ),
        "allowed_keywords": frozenset({
            "notional",
            "spread_quote",
            "recovery",
            "schedule",
            "credit_curve",
            "discount_curve",
        }),
        "message": (
            "`price_cds_analytical(...)` is a keyword-only helper expecting "
            "`notional=..., spread_quote=..., recovery=..., schedule=..., "
            "credit_curve=..., discount_curve=...`. Use the checked helper surface "
            "directly instead of rebuilding leg math or inventing alternate keywords."
        ),
    },
    "price_cds_monte_carlo": {
        "min_positional_args": 0,
        "keyword_only": True,
        "required_parameters": (
            "notional",
            "spread_quote",
            "recovery",
            "schedule",
            "credit_curve",
            "discount_curve",
        ),
        "required_keyword_groups": (
            frozenset({
                "notional",
                "spread_quote",
                "recovery",
                "schedule",
                "credit_curve",
                "discount_curve",
            }),
        ),
        "allowed_keywords": frozenset({
            "notional",
            "spread_quote",
            "recovery",
            "schedule",
            "credit_curve",
            "discount_curve",
            "n_paths",
            "seed",
        }),
        "message": (
            "`price_cds_monte_carlo(...)` is a keyword-only helper expecting "
            "`notional=..., spread_quote=..., recovery=..., schedule=..., "
            "credit_curve=..., discount_curve=...` with optional `n_paths` and `seed`."
        ),
    },
    "price_vanilla_equity_option_tree": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "model", "n_steps"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_vanilla_equity_option_tree(...)` expects `(market_state, spec_like, "
            "model=..., n_steps=...)`. Pass a spec-like object with `spot`, `strike`, "
            "`expiry_date`, and optional exercise fields instead of inventing helper keywords."
        ),
    },
    "price_fx_vanilla_analytical": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_fx_vanilla_analytical(...)` expects `(market_state, spec)`. "
            "Pass the live `market_state` and the original spec-like object instead of "
            "resolved GK inputs, option-type literals, or raw-kernel arguments."
        ),
    },
    "price_fx_vanilla_monte_carlo": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "seed"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_fx_vanilla_monte_carlo(...)` expects `(market_state, spec, seed=...)`. "
            "Pass the live `market_state` and the original spec-like object instead of "
            "resolved process inputs or raw Monte Carlo plumbing."
        ),
    },
    "price_nth_to_default_basket": {
        "min_positional_args": 0,
        "keyword_only": True,
        "required_parameters": (
            "notional",
            "n_names",
            "n_th",
            "horizon",
            "correlation",
            "recovery",
            "credit_curve",
            "discount_curve",
        ),
        "required_keyword_groups": (
            frozenset({
                "notional",
                "n_names",
                "n_th",
                "horizon",
                "correlation",
                "recovery",
                "credit_curve",
                "discount_curve",
            }),
        ),
        "allowed_keywords": frozenset({
            "notional",
            "n_names",
            "n_th",
            "horizon",
            "correlation",
            "recovery",
            "credit_curve",
            "discount_curve",
        }),
        "message": (
            "`price_nth_to_default_basket(...)` is a keyword-only helper expecting "
            "`notional=..., n_names=..., n_th=..., horizon=..., correlation=..., "
            "recovery=..., credit_curve=..., discount_curve=...`."
        ),
    },
    "price_credit_basket_tranche": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({
            "market_state",
            "spec",
            "copula_family",
            "degrees_of_freedom",
            "n_paths",
            "seed",
        }),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_credit_basket_tranche(...)` expects `(market_state, spec, *, "
            "copula_family=..., degrees_of_freedom=..., n_paths=..., seed=...)`. "
            "Pass the live market state and original tranche spec instead of "
            "rebuilding copula loss plumbing inline."
        ),
    },
    "price_quanto_option_analytical_from_market_state": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_quanto_option_analytical_from_market_state(...)` expects "
            "`(market_state, spec)`. Pass the live `market_state` and the original "
            "spec-like object instead of resolved quanto inputs or raw Black helpers."
        ),
    },
    "price_quanto_option_monte_carlo_from_market_state": {
        "min_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_quanto_option_monte_carlo_from_market_state(...)` expects "
            "`(market_state, spec)`. Pass the live `market_state` and the original "
            "spec-like object instead of resolved quanto inputs or ad hoc MC glue."
        ),
    },
}


def _calls_symbol(source: str, symbol: str) -> bool:
    """Return whether ``source`` appears to call ``symbol`` as a function."""
    return re.search(rf"\b{re.escape(symbol)}\s*\(", source) is not None


def _call_matches_symbol(node: ast.Call, symbol: str) -> bool:
    """Whether one AST call targets the requested symbol."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == symbol
    if isinstance(func, ast.Attribute):
        return func.attr == symbol
    return False


def _find_calls_for_symbol(tree: ast.AST, symbol: str) -> tuple[ast.Call, ...]:
    """Return every AST call that targets the given symbol name."""
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_matches_symbol(node, symbol)
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
        findings.extend(self._check_exact_helper_surface(source, route_spec))

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

        helper_symbols = tuple(
            prim.symbol
            for prim in route_spec.primitives
            if prim.role == "route_helper" and prim.required
        )
        if helper_symbols and any(_calls_symbol(source, symbol) for symbol in helper_symbols):
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
                if not _calls_symbol(source, prim.symbol):
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

    def _check_exact_helper_surface(
        self, source: str, route_spec: RouteSpec,
    ) -> list[SemanticFinding]:
        """Verify exact backend helpers are called with an admissible surface."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        findings: list[SemanticFinding] = []
        for prim in route_spec.primitives:
            if prim.role != "route_helper" or not prim.required:
                continue
            signature = _EXACT_HELPER_SIGNATURES.get(prim.symbol)
            if signature is None:
                continue
            for call in _find_calls_for_symbol(tree, prim.symbol):
                keyword_names = {
                    keyword.arg
                    for keyword in call.keywords
                    if keyword.arg is not None
                }
                required_keyword_groups = tuple(signature.get("required_keyword_groups", ()) or ())
                keyword_surface_ok = not required_keyword_groups or any(
                    set(group).issubset(keyword_names)
                    for group in required_keyword_groups
                )
                if not _call_satisfies_required_surface(
                    call,
                    signature=signature,
                    keyword_names=keyword_names,
                    keyword_surface_ok=keyword_surface_ok,
                ):
                    findings.append(SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="route_helper_signature_mismatch",
                        message=str(signature["message"]),
                        line=getattr(call, "lineno", None),
                    ))
                    continue
                unexpected = sorted(keyword_names - set(signature["allowed_keywords"]))
                if unexpected:
                    findings.append(SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="route_helper_signature_mismatch",
                        message=(
                            str(signature["message"])
                            + f" Unexpected keyword(s): {', '.join(unexpected)}."
                        ),
                        line=getattr(call, "lineno", None),
                    ))
                    continue
                positional_markers = tuple(signature.get("required_positional_markers", ()) or ())
                if positional_markers:
                    for index, markers in enumerate(positional_markers):
                        if index >= len(call.args):
                            break
                        if not _argument_matches_markers(call.args[index], tuple(str(marker) for marker in markers)):
                            findings.append(SemanticFinding(
                                validator="algorithm_contract",
                                severity="error",
                                category="route_helper_signature_mismatch",
                                message=str(signature["message"]),
                                line=getattr(call, "lineno", None),
                            ))
                            break
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
            "exercise_policy", "resolve_lattice_exercise_policy",
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


def _call_satisfies_required_surface(
    call: ast.Call,
    *,
    signature: dict[str, object],
    keyword_names: set[str],
    keyword_surface_ok: bool,
) -> bool:
    """Return whether one helper call satisfies the declared required surface."""
    if bool(signature.get("keyword_only")) and call.args:
        return False

    required_parameters = tuple(str(item) for item in signature.get("required_parameters", ()) or ())
    positional_markers = tuple(signature.get("required_positional_markers", ()) or ())

    if required_parameters:
        for index, parameter in enumerate(required_parameters):
            if index < len(call.args):
                if index < len(positional_markers):
                    markers = tuple(str(marker) for marker in positional_markers[index])
                    if markers and not _argument_matches_markers(call.args[index], markers):
                        return False
                continue
            if parameter in keyword_names:
                continue
            return False
        return True

    return len(call.args) >= int(signature["min_positional_args"]) or keyword_surface_ok


def _argument_matches_markers(node: ast.AST, markers: tuple[str, ...]) -> bool:
    """Return whether one AST argument resembles the expected semantic surface."""
    try:
        text = ast.unparse(node)
    except Exception:
        return False
    normalized = text.replace(" ", "").lower()
    return any(marker.lower() in normalized for marker in markers)
