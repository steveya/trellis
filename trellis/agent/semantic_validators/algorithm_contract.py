"""AlgorithmContractValidator — verifies the pricing algorithm matches the route.

Checks:
1. Engine family consistency (MC route → MonteCarloEngine instantiated)
2. Route helper usage (if route_helper primitive specified, it must be called)
3. Discount application (present-value products apply discount factors)
4. Exercise logic presence (American/Bermudan → exercise boundary or LSM)
"""

from __future__ import annotations

import ast

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec, resolve_route_primitives
from trellis.agent.semantic_validators.base import SemanticFinding


# Engine family → expected code signatures
_ENGINE_SIGNATURES = {
    "monte_carlo": ("MonteCarloEngine", "monte_carlo"),
    "exercise": ("MonteCarloEngine", "longstaff_schwartz", "tsitsiklis_van_roy"),
    "lattice": (
        "build_lattice",
        "price_on_lattice",
        "build_rate_lattice",
        "BinomialTree",
        "backward_induction",
        "lattice_backward_induction",
    ),
    "analytical": (
        "black76_call",
        "black76_put",
        "garman_kohlhagen_price_raw",
    ),
    "fft_pricing": ("fft_price", "cos_price"),
    "pde_solver": ("theta_method_1d", "Grid", "BlackScholesOperator"),
    "qmc": ("sobol_normals", "GBM"),
    "copula": ("FactorCopula",),
    "waterfall": ("Waterfall", "Tranche"),
}

_ROUTE_SIGNATURES = {
    "heston_adi_2d": ("price_heston_option_adi_pde_result", "HestonAdiPDEConfig"),
}

# Discount patterns
_DISCOUNT_PATTERNS = (
    "market_state.discount",
    "discount(",
    "discount_factor",
    "df(",
    ".discount(",
)
_CHECKED_ROUTE_HELPER_BINDINGS = {
    "price_heston_option_monte_carlo": {
        "routes": frozenset({"monte_carlo_paths"}),
        "instruments": frozenset({"heston_option", "european_option", "vanilla_option"}),
    },
}
_CHECKED_ROUTE_HELPER_SYMBOLS = frozenset(_CHECKED_ROUTE_HELPER_BINDINGS)
_HELPER_OWNED_ROUTE_SYMBOLS = _CHECKED_ROUTE_HELPER_SYMBOLS | frozenset({
    "price_double_barrier_option_pde_result",
    "price_double_barrier_option_monte_carlo_result",
})
_DECLARATIVE_PRIMITIVE_ROLES = frozenset({"mesh", "topology"})
_EXPLICIT_COMPOSITION_ROUTE_IDS = frozenset({
    "equity_quanto",
    "rate_tree_backward_induction",
})
_EXPLICIT_COMPOSITION_PRIMITIVES = frozenset({
    ("analytical_black76", "barrier_option_price"),
})
_ENGINE_OWNING_PRICING_KERNELS = frozenset({
    ("analytical_black76", "barrier_option_price"),
})

_EXACT_HELPER_SIGNATURES = {
    "price_double_barrier_option_pde_result": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "config"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_double_barrier_option_pde_result(...)` expects "
            "`(market_state, spec, *, config=...)`. Pass the live market state "
            "and original double-barrier spec-like object instead of rebuilding "
            "barrier, grid, operator, payoff, or discounting internals inline."
        ),
    },
    "price_double_barrier_option_monte_carlo_result": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "config"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_double_barrier_option_monte_carlo_result(...)` expects "
            "`(market_state, spec, *, config=...)`. Pass the live market state "
            "and original double-barrier spec-like object instead of rebuilding "
            "barrier monitors, GBM paths, payoff, or discounting internals inline."
        ),
    },
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
        "max_positional_args": 2,
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
    "price_vanilla_equity_option_monte_carlo": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({
            "market_state",
            "spec",
            "scheme",
            "variance_reduction",
            "n_paths",
            "n_steps",
            "seed",
        }),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_vanilla_equity_option_monte_carlo(...)` expects `(market_state, "
            "spec, *, scheme=..., variance_reduction=..., n_paths=..., n_steps=..., "
            "seed=...)`. Pass the live market state and original spec-like object "
            "instead of spot/strike scalars or hand-built Monte Carlo plumbing."
        ),
    },
    "price_vanilla_equity_option_transform": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({
            "market_state",
            "spec",
            "method",
            "fft_alpha",
            "fft_points",
            "fft_eta",
            "cos_points",
            "cos_truncation",
        }),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_vanilla_equity_option_transform(...)` expects `(market_state, spec, "
            "*, method=..., fft_alpha=..., fft_points=..., fft_eta=..., cos_points=..., "
            "cos_truncation=...)`. Pass the live market state and original spec-like "
            "object instead of raw transform arguments or reconstructed spot/strike inputs."
        ),
    },
    "price_vanilla_equity_option_pde": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({
            "market_state",
            "spec",
            "theta",
            "n_x",
            "n_t",
            "s_max_multiplier",
        }),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_vanilla_equity_option_pde(...)` expects `(market_state, spec, *, "
            "theta=..., n_x=..., n_t=..., s_max_multiplier=...)`. Pass the live market "
            "state and original spec-like object instead of explicit spot/strike/time "
            "keywords or manual PDE setup."
        ),
    },
    "price_callable_bond_tree": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "model", "mean_reversion", "sigma", "n_steps"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_callable_bond_tree(...)` expects `(market_state, spec, *, "
            "model=..., mean_reversion=..., sigma=..., n_steps=...)`. "
            "Pass the live market state and original callable/puttable bond spec "
            "instead of inventing lattice-builder keywords."
        ),
    },
    "price_bermudan_swaption_tree": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "model", "mean_reversion", "sigma", "n_steps"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_bermudan_swaption_tree(...)` expects `(market_state, spec, *, "
            "model=..., mean_reversion=..., sigma=..., n_steps=...)`. "
            "Pass the live market state and the original Bermudan swaption spec "
            "instead of rebuilding lattice rollback or exercise glue inline."
        ),
    },
    "price_fx_vanilla_analytical": {
        "min_positional_args": 2,
        "max_positional_args": 2,
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
        "max_positional_args": 2,
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
        "max_positional_args": 2,
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
    "price_credit_portfolio_loss_distribution_recursive": {
        "min_positional_args": 2,
        "max_positional_args": 2,
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
            "`price_credit_portfolio_loss_distribution_recursive(...)` expects "
            "`(market_state, spec, *, copula_family=..., degrees_of_freedom=..., "
            "n_paths=..., seed=...)`."
        ),
    },
    "price_credit_portfolio_loss_distribution_transform_proxy": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({
            "market_state",
            "spec",
            "copula_family",
        }),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_credit_portfolio_loss_distribution_transform_proxy(...)` expects "
            "`(market_state, spec, *, copula_family=...)`."
        ),
    },
    "price_credit_portfolio_loss_distribution_monte_carlo": {
        "min_positional_args": 2,
        "max_positional_args": 2,
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
            "`price_credit_portfolio_loss_distribution_monte_carlo(...)` expects "
            "`(market_state, spec, *, copula_family=..., degrees_of_freedom=..., "
            "n_paths=..., seed=...)`."
        ),
    },
    "price_zcb_option_tree": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "model", "mean_reversion", "sigma", "n_steps"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_zcb_option_tree(...)` expects `(market_state, spec, *, "
            "model=..., mean_reversion=..., sigma=..., n_steps=...)`. "
            "Pass the live market state and original ZCB-option spec instead of "
            "inventing direct lattice-builder keywords."
        ),
    },
    "price_zcb_option_jamshidian": {
        "min_positional_args": 2,
        "max_positional_args": 2,
        "required_parameters": ("market_state", "spec"),
        "required_keyword_groups": (frozenset({"market_state", "spec"}),),
        "allowed_keywords": frozenset({"market_state", "spec", "mean_reversion"}),
        "required_positional_markers": (
            frozenset({"market_state"}),
            frozenset({"spec", "_spec"}),
        ),
        "message": (
            "`price_zcb_option_jamshidian(...)` expects `(market_state, spec, *, "
            "mean_reversion=...)`. Pass the live market state and original ZCB-option "
            "spec instead of resolved inputs or ad hoc strike plumbing."
        ),
    },
}


def _calls_symbol(source: str, symbol: str) -> bool:
    """Return whether a Python call node targets ``symbol``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call) and _call_matches_symbol(node, symbol)
        for node in ast.walk(tree)
    )


def _references_symbol(source: str, symbol: str) -> bool:
    """Return whether parsed source calls or names ``symbol``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        (isinstance(node, ast.Name) and node.id == symbol)
        or (isinstance(node, ast.Attribute) and node.attr == symbol)
        for node in ast.walk(tree)
    )


def _calls_checked_route_helper(
    source: str,
    plan: GenerationPlan | None = None,
    route_spec: RouteSpec | None = None,
    exact_surface_primitives=(),
) -> bool:
    """Return whether source delegates to a checked wrapper for a required route helper."""
    if not any(
        prim.role == "route_helper" and prim.required
        for prim in exact_surface_primitives
    ):
        return False
    route_id = str(getattr(route_spec, "id", "") or "").strip()
    instrument_type = str(getattr(plan, "instrument_type", "") or "").strip()
    for symbol, binding in _CHECKED_ROUTE_HELPER_BINDINGS.items():
        if not _calls_symbol(source, symbol):
            continue
        routes = frozenset(binding.get("routes", frozenset()))
        if routes and route_id and route_id not in routes:
            continue
        instruments = frozenset(binding.get("instruments", frozenset()))
        if instruments and instrument_type and instrument_type not in instruments:
            continue
        return True
    return False


def _calls_helper_owned_required_route_helper(source: str, exact_surface_primitives) -> bool:
    """Return whether source calls a helper that owns its route internals."""
    return any(
        prim.role == "route_helper"
        and prim.required
        and prim.symbol in _HELPER_OWNED_ROUTE_SYMBOLS
        and _calls_symbol(source, prim.symbol)
        for prim in exact_surface_primitives
    )


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

        exact_surface_primitives = _exact_surface_primitives(plan, route_spec)
        checked_route_helper_call = _calls_checked_route_helper(
            source,
            plan,
            route_spec,
            exact_surface_primitives,
        )
        helper_owned_route = (
            _calls_helper_owned_required_route_helper(source, exact_surface_primitives)
            or checked_route_helper_call
        )

        # 1. Route helper usage and exact surface.
        findings.extend(
            self._check_route_helper(
                source,
                route_spec,
                exact_surface_primitives,
                helper_owned_route=helper_owned_route,
            )
        )
        findings.extend(self._check_exact_helper_surface(source, route_spec, exact_surface_primitives))
        findings.extend(
            self._check_required_primitive_composition(
                source,
                route_spec,
                exact_surface_primitives,
            )
        )

        # Checked route helpers own internal engine, payoff, and discounting
        # obligations, but only after the helper call surface itself validates.
        if helper_owned_route:
            return tuple(findings)

        # 2. Engine family consistency
        findings.extend(self._check_engine_family(source, route_spec, exact_surface_primitives))

        # 3. Discount application
        findings.extend(self._check_discount_application(source, route_spec))

        # 4. Exercise logic
        findings.extend(self._check_exercise_logic(source, plan))

        return tuple(findings)

    def _check_engine_family(
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
    ) -> list[SemanticFinding]:
        """Verify code uses the expected engine family signatures."""
        engine = route_spec.engine_family
        signatures = _ROUTE_SIGNATURES.get(route_spec.id, _ENGINE_SIGNATURES.get(engine, ()))
        if not signatures:
            return []

        engine_owning_symbols = tuple(
            prim.symbol
            for prim in exact_surface_primitives
            if prim.required
            and (
                prim.role == "route_helper"
                or (route_spec.id, prim.symbol) in _ENGINE_OWNING_PRICING_KERNELS
            )
        )
        if engine_owning_symbols and any(
            _calls_symbol(source, symbol) for symbol in engine_owning_symbols
        ):
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
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
        *,
        helper_owned_route: bool = False,
    ) -> list[SemanticFinding]:
        """Verify route_helper primitives are actually called."""
        if helper_owned_route:
            return []
        findings = []
        for prim in exact_surface_primitives:
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

    def _check_required_primitive_composition(
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
    ) -> list[SemanticFinding]:
        """Require explicit primitive composition for helper-retired routes."""
        enforce_whole_route = route_spec.id in _EXPLICIT_COMPOSITION_ROUTE_IDS
        required_primitives = tuple(
            primitive
            for primitive in exact_surface_primitives
            if enforce_whole_route
            or (route_spec.id, primitive.symbol) in _EXPLICIT_COMPOSITION_PRIMITIVES
        )
        if not required_primitives:
            return []

        findings: list[SemanticFinding] = []
        for primitive in required_primitives:
            if not primitive.required or primitive.excluded:
                continue
            if _calls_symbol(source, primitive.symbol) or (
                primitive.role in _DECLARATIVE_PRIMITIVE_ROLES
                and _references_symbol(source, primitive.symbol)
            ):
                continue
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="required_primitive_not_called",
                    message=(
                        f"Route '{route_spec.id}' requires explicit composition with "
                        f"'{primitive.symbol}' from '{primitive.module}', but generated "
                        "code does not call that primitive. Product pricing wrappers do "
                        "not satisfy this construction contract."
                    ),
                )
            )
        return findings

    def _check_exact_helper_surface(
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
    ) -> list[SemanticFinding]:
        """Verify exact backend helpers are called with an admissible surface."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        findings: list[SemanticFinding] = []
        for prim in exact_surface_primitives:
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


def _exact_surface_primitives(
    plan: GenerationPlan,
    route_spec: RouteSpec,
):
    """Prefer the compiled plan's resolved primitives over route-card primitives."""
    primitive_plan = getattr(plan, "primitive_plan", None)
    plan_primitives = tuple(getattr(primitive_plan, "primitives", ()) or ())
    if plan_primitives:
        return plan_primitives
    method = str(getattr(plan, "method", "") or "").strip() or None
    if method and _route_conditionals_are_method_only(route_spec):
        try:
            resolved = tuple(resolve_route_primitives(route_spec, None, method=method))
        except Exception:
            resolved = ()
        if resolved:
            return resolved
    return tuple(getattr(route_spec, "primitives", ()) or ())


def _route_conditionals_are_method_only(route_spec: RouteSpec) -> bool:
    """Return whether every non-default conditional clause depends only on method."""
    conditionals = tuple(getattr(route_spec, "conditional_primitives", ()) or ())
    if not conditionals:
        return False
    saw_method_clause = False
    for cond in conditionals:
        when = getattr(cond, "when", None)
        if when == "default":
            continue
        if not isinstance(when, dict):
            return False
        keys = {str(key).strip() for key in when.keys()}
        if keys != {"methods"}:
            return False
        saw_method_clause = True
    return saw_method_clause


def _call_satisfies_required_surface(
    call: ast.Call,
    *,
    signature: dict[str, object],
    keyword_names: set[str],
    keyword_surface_ok: bool,
) -> bool:
    """Return whether one helper call satisfies the declared required surface."""
    max_positional_args = signature.get("max_positional_args")
    if max_positional_args is not None and len(call.args) > int(max_positional_args):
        return False

    if bool(signature.get("keyword_only")) and call.args:
        return False

    required_parameters = tuple(str(item) for item in signature.get("required_parameters", ()) or ())
    positional_markers = tuple(signature.get("required_positional_markers", ()) or ())

    if required_parameters:
        for index, parameter in enumerate(required_parameters):
            if index < len(call.args):
                if parameter in keyword_names:
                    return False
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
