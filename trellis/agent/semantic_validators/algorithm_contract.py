"""AlgorithmContractValidator — verifies the pricing algorithm matches the route.

Checks:
1. Engine family consistency (MC route → MonteCarloEngine instantiated)
2. Route helper usage (if route_helper primitive specified, it must be called)
3. Discount application (present-value products apply discount factors)
4. Exercise logic presence (American/Bermudan → exercise boundary or LSM)
"""

from __future__ import annotations

import ast
from collections.abc import Callable

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
        "two_asset_extremum_option_stulz",
        "two_asset_spread_option_kirk",
        "two_asset_terminal_basket_gauss_hermite",
    ),
    "fft_pricing": ("fft_price", "cos_price", "hurd_zhou_spread_option_2d_fft"),
    "pde_solver": ("theta_method_1d", "Grid", "BlackScholesOperator"),
    "qmc": ("sobol_normals", "GBM"),
    "copula": ("FactorCopula",),
    "waterfall": ("Waterfall", "Tranche"),
}

_ROUTE_SIGNATURES = {
    "credit_default_swap": (
        "expected_first_event_weights",
        "sample_first_event_weights",
    ),
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
_DECLARATIVE_PRIMITIVE_ROLES = frozenset({"mesh", "model_registry", "topology"})
_EXPLICIT_COMPOSITION_ROUTE_IDS = frozenset({
    "credit_default_swap",
    "equity_quanto",
    "exercise_lattice",
    "rate_tree_backward_induction",
    "short_rate_bond_option",
})
_TERMINAL_BASKET_FORBIDDEN_SYMBOLS = frozenset(
    {
        "price_basket_option_analytical",
        "price_basket_option_monte_carlo",
        "price_basket_option_transform_proxy",
        "price_ranked_observation_basket_monte_carlo",
        "build_ranked_observation_basket_state_payoff",
        "terminal_ranked_observation_basket_payoff",
    }
)
_ZCB_OPTION_FORBIDDEN_SYMBOLS = frozenset(
    {
        "build_zcb_option_lattice",
        "price_zcb_option_jamshidian",
        "price_zcb_option_on_lattice",
        "price_zcb_option_tree",
        "resolve_zcb_option_hw_inputs",
    }
)
_CREDIT_DEFAULT_SWAP_FORBIDDEN_SYMBOLS = frozenset(
    {
        "build_cds_schedule",
        "price_cds_analytical",
        "price_cds_monte_carlo",
    }
)
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


def _is_exact_call_to_symbol(expression: ast.AST, symbol: str) -> bool:
    """Return whether an expression directly calls the local ``symbol`` name."""
    return (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == symbol
    )


_CDS_CASHFLOW_PRIMITIVE_MODULE = "trellis.models.contingent_cashflows"
_CDS_OPAQUE_NAMESPACE_CALLS = frozenset({
    "__import__",
    "__delattr__",
    "__delitem__",
    "__setattr__",
    "__setitem__",
    "clear",
    "delattr",
    "delitem",
    "eval",
    "exec",
    "globals",
    "locals",
    "pop",
    "popitem",
    "setdefault",
    "setattr",
    "setitem",
    "update",
    "vars",
})


def _is_direct_approved_symbol_import(node: ast.AST, symbol: str) -> bool:
    """Recognize an unaliased import of one public CDS cashflow primitive."""
    if not (
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == _CDS_CASHFLOW_PRIMITIVE_MODULE
    ):
        return False
    bindings = tuple(
        alias
        for alias in node.names
        if (alias.asname or alias.name) == symbol
    )
    return (
        len(bindings) == 1
        and bindings[0].name == symbol
        and bindings[0].asname is None
    )


def _import_binds_name(node: ast.Import | ast.ImportFrom, name: str) -> bool:
    """Return whether an import statement binds ``name`` in its current scope."""
    if isinstance(node, ast.Import):
        return any(
            (alias.asname or alias.name.split(".", 1)[0]) == name
            for alias in node.names
        )
    return any((alias.asname or alias.name) == name for alias in node.names)


def _constant_string(expression: ast.AST) -> str | None:
    """Return the exact string represented by a literal expression."""
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    return None


def _expression_is_rooted_at_name(expression: ast.AST, name: str) -> bool:
    """Recognize attribute/subscript chains rooted at one local name."""
    while isinstance(expression, (ast.Attribute, ast.Subscript)):
        expression = expression.value
    return isinstance(expression, ast.Name) and expression.id == name


def _tree_uses_opaque_namespace_access(tree: ast.AST) -> bool:
    """Reject dynamic namespace APIs that prevent proving import immutability."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _CDS_OPAQUE_NAMESPACE_CALLS:
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "builtins"
            and any(
                alias.name in _CDS_OPAQUE_NAMESPACE_CALLS
                for alias in node.names
            )
        ):
            return True
        if isinstance(node, ast.Attribute) and node.attr in {"__dict__", "modules"}:
            return True
        if (
            isinstance(node, ast.Subscript)
            and _constant_string(node.slice) in _CDS_OPAQUE_NAMESPACE_CALLS
        ):
            return True
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _CDS_OPAQUE_NAMESPACE_CALLS
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _CDS_OPAQUE_NAMESPACE_CALLS
        ):
            return True
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _constant_string(node.args[1]) in _CDS_OPAQUE_NAMESPACE_CALLS
        ):
            return True
    return False


def _node_binds_name_in_current_scope(
    node: ast.AST,
    *,
    name: str,
    ignored_node_ids: frozenset[int] = frozenset(),
) -> bool:
    """Detect a binding without descending into a genuinely nested scope."""
    if id(node) in ignored_node_ids:
        return False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Lambda):
        return False
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _import_binds_name(node, name)
    if isinstance(node, ast.ExceptHandler) and node.name == name:
        return True
    if isinstance(node, ast.MatchAs) and node.name == name:
        return True
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and (
            node.attr == name
            or _expression_is_rooted_at_name(node.value, name)
        )
    ):
        return True
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and (
            _constant_string(node.slice) == name
            or _expression_is_rooted_at_name(node.value, name)
        )
    ):
        return True
    if (
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        return True
    return any(
        _node_binds_name_in_current_scope(
            child,
            name=name,
            ignored_node_ids=ignored_node_ids,
        )
        for child in ast.iter_child_nodes(node)
    )


def _function_argument_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Return every name bound by a function signature."""
    arguments = function.args
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return frozenset(names)


def _module_has_authoritative_cds_cashflow_import(
    tree: ast.Module,
    *,
    symbol: str,
) -> bool:
    """Require the public import to own the unshadowed name used by ``evaluate``."""
    if _tree_uses_opaque_namespace_access(tree):
        return False
    evaluate_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate"
    )
    if len(evaluate_functions) != 1:
        return False
    evaluate = evaluate_functions[0]
    if symbol in _function_argument_names(evaluate):
        return False

    local_imports = tuple(
        node
        for node in evaluate.body
        if _is_direct_approved_symbol_import(node, symbol)
    )
    if local_imports:
        if len(local_imports) != 1:
            return False
        direct_calls = tuple(
            node
            for node in ast.walk(evaluate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == symbol
        )
        if not direct_calls or getattr(local_imports[0], "lineno", 0) >= min(
            getattr(call, "lineno", 0) for call in direct_calls
        ):
            return False
        ignored = frozenset({id(local_imports[0])})
        return not any(
            _node_binds_name_in_current_scope(
                statement,
                name=symbol,
                ignored_node_ids=ignored,
            )
            for statement in evaluate.body
        )

    if any(
        _node_binds_name_in_current_scope(statement, name=symbol)
        for statement in evaluate.body
    ):
        return False
    module_imports = tuple(
        node
        for node in tree.body
        if _is_direct_approved_symbol_import(node, symbol)
    )
    if len(module_imports) != 1:
        return False
    ignored = frozenset({id(module_imports[0])})
    return not any(
        _node_binds_name_in_current_scope(
            statement,
            name=symbol,
            ignored_node_ids=ignored,
        )
        for statement in tree.body
        if statement is not evaluate
    )


_NESTED_LOOP_EXIT_SCOPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _subtree_has_current_loop_exit(node: ast.AST) -> bool:
    """Detect break/continue targeting the current loop, not a nested scope."""
    if isinstance(node, _NESTED_LOOP_EXIT_SCOPES):
        return False
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NESTED_LOOP_EXIT_SCOPES):
            continue
        if isinstance(child, (ast.Break, ast.Continue)):
            return True
        if _subtree_has_current_loop_exit(child):
            return True
    return False


def _is_nonpositive_name_guard(
    expression: ast.AST,
    *,
    name: str,
) -> bool:
    """Recognize ``name <= 0`` with an integer or floating-point zero."""
    return (
        isinstance(expression, ast.Compare)
        and isinstance(expression.left, ast.Name)
        and expression.left.id == name
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], ast.LtE)
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.Constant)
        and not isinstance(expression.comparators[0].value, bool)
        and expression.comparators[0].value == 0
    )


def _is_supported_cds_early_continue_guard(
    loop: ast.For | ast.AsyncFor,
    statement: ast.AST,
) -> bool:
    """Recognize the two bounded CDS guards that may precede leg assembly."""
    if not isinstance(statement, ast.If) or statement.orelse:
        return False
    if (
        _is_nonpositive_name_guard(statement.test, name="event_weight")
        and len(statement.body) == 1
        and isinstance(statement.body[0], ast.Continue)
        and _single_immutable_assignment_value(
            loop,
            "event_weight",
            before_line=getattr(statement, "lineno", 0),
        )
        is not None
    ):
        return True
    return _is_supported_cds_empty_period_guard(
        loop,
        statement,
        start_name="interval_start",
        stop_name="interval_stop",
    )


def _is_supported_cds_empty_period_guard(
    loop: ast.For | ast.AsyncFor,
    statement: ast.AST,
    *,
    start_name: str,
    stop_name: str,
) -> bool:
    """Recognize the exact skip required before pricing a period with no events."""
    return (
        isinstance(statement, ast.If)
        and not statement.orelse
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == stop_name
        and len(statement.test.ops) == 1
        and isinstance(statement.test.ops[0], ast.LtE)
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Name)
        and statement.test.comparators[0].id == start_name
        and len(statement.body) == 2
        and isinstance(statement.body[0], ast.Assign)
        and len(statement.body[0].targets) == 1
        and isinstance(statement.body[0].targets[0], ast.Name)
        and statement.body[0].targets[0].id == start_name
        and isinstance(statement.body[0].value, ast.Name)
        and statement.body[0].value.id == stop_name
        and isinstance(statement.body[1], ast.Continue)
        and _single_immutable_assignment_value(
            loop,
            stop_name,
            before_line=getattr(statement, "lineno", 0),
        )
        is not None
        and _cds_interval_cursor_writes_are_bounded(
            loop,
            start_name=start_name,
            stop_name=stop_name,
        )
    )


def _direct_loop_body_nodes(
    loop: ast.For | ast.AsyncFor,
) -> tuple[ast.AST, ...]:
    """Return only unconditional statements directly owned by one loop body.

    CDS aggregation evidence beneath an ``if`` is not accepted here: proving
    arbitrary branch reachability or guard polarity is outside this bounded
    validator. Generated routes can use an early ``continue`` guard and keep
    the required leg accumulations unconditional after it.
    """
    reachable: list[ast.AST] = []
    for statement in loop.body:
        if isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            break
        if (
            _subtree_has_current_loop_exit(statement)
            and not _is_supported_cds_early_continue_guard(loop, statement)
        ):
            break
        reachable.append(statement)
    return tuple(reachable)


_NESTED_EVALUATE_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _subtree_has_evaluate_exit(node: ast.AST) -> bool:
    """Detect a return or raise without descending into a nested local scope."""
    if isinstance(node, _NESTED_EVALUATE_SCOPES):
        return False
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NESTED_EVALUATE_SCOPES):
            continue
        if isinstance(child, (ast.Return, ast.Raise)):
            return True
        if _subtree_has_evaluate_exit(child):
            return True
    return False


def _is_supported_cds_market_guard(statement: ast.AST) -> bool:
    """Admit exact fail-fast checks for the two required CDS market handles."""
    if not (
        isinstance(statement, ast.If)
        and len(statement.body) == 1
        and not statement.orelse
        and isinstance(statement.body[0], ast.Raise)
        and isinstance(statement.test, ast.Compare)
        and len(statement.test.ops) == 1
        and isinstance(statement.test.ops[0], ast.Is)
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Constant)
        and statement.test.comparators[0].value is None
        and isinstance(statement.test.left, ast.Attribute)
        and statement.test.left.attr in {"credit_curve", "discount"}
        and isinstance(statement.test.left.value, ast.Name)
        and statement.test.left.value.id == "market_state"
    ):
        return False
    raised = statement.body[0].exc
    return (
        isinstance(raised, ast.Call)
        and isinstance(raised.func, ast.Name)
        and raised.func.id == "ValueError"
        and not raised.keywords
    )


class _NestedEvaluateScopePruner(ast.NodeTransformer):
    """Remove local scopes whose bodies are not evidence from ``evaluate``."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> ast.Constant:
        return ast.copy_location(ast.Constant(value=None), node)


def _evaluate_definition_is_authoritative(
    tree: ast.Module,
    evaluate: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Require the inspected CDS method to remain the runtime method binding."""
    if isinstance(evaluate, ast.AsyncFunctionDef) or evaluate.decorator_list:
        return False
    owners = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and evaluate in node.body
    )
    if len(owners) > 1:
        return False
    if owners:
        owner = owners[0]
        if (
            owner not in tree.body
            or owner.bases
            or owner.keywords
            or owner.decorator_list
        ):
            return False
        if any(
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name in {"__getattr__", "__getattribute__", "__new__"}
            for statement in owner.body
        ):
            return False
    elif evaluate not in tree.body:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "evaluate"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if isinstance(node, ast.Attribute) and isinstance(
            node.ctx,
            (ast.Store, ast.Del),
        ):
            if node.attr == "evaluate" or any(
                isinstance(part, ast.Attribute) and part.attr == "evaluate"
                for part in ast.walk(node.value)
            ) or _expression_is_rooted_at_name(node.value, "evaluate"):
                return False
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (
                _constant_string(node.slice) == "evaluate"
                or any(
                    isinstance(part, ast.Attribute) and part.attr == "evaluate"
                    for part in ast.walk(node.value)
                )
                or _expression_is_rooted_at_name(node.value, "evaluate")
            )
        ):
            return False
    return True


def _reachable_evaluate_tree(tree: ast.Module) -> ast.Module | None:
    """Return one evaluate body with one direct final exit and no local scopes."""
    evaluate_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate"
    )
    if len(evaluate_functions) != 1:
        return None
    evaluate = evaluate_functions[0]
    if not _evaluate_definition_is_authoritative(tree, evaluate):
        return None

    reachable: list[ast.stmt] = []
    for statement in evaluate.body:
        if not isinstance(statement, (ast.Return, ast.Raise)) and (
            _subtree_has_evaluate_exit(statement)
        ) and not _is_supported_cds_market_guard(statement):
            return None
        reachable.append(statement)
        if isinstance(statement, (ast.Raise, ast.Return)):
            break
    pruned_tree = _NestedEvaluateScopePruner().visit(
        ast.Module(body=reachable, type_ignores=[])
    )
    return pruned_tree if isinstance(pruned_tree, ast.Module) else None


def _direct_loop_body_augments_with_call(
    loop: ast.For | ast.AsyncFor,
    symbol: str,
) -> bool:
    """Detect an accumulated call in a loop body, excluding nested loops."""
    return bool(_direct_loop_augments(loop, symbol=symbol))


def _direct_loop_augments(
    loop: ast.For | ast.AsyncFor,
    *,
    symbol: str | None = None,
) -> tuple[ast.AugAssign, ...]:
    """Return recognized additive name updates directly owned by one loop."""
    return tuple(
        node
        for node in _direct_loop_body_nodes(loop)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Name)
        and (symbol is None or _is_exact_call_to_symbol(node.value, symbol))
    )


def _direct_loop_augmented_values_with_call(
    loop: ast.For | ast.AsyncFor,
    symbol: str,
) -> tuple[ast.AST, ...]:
    """Return accumulated expressions containing a call, excluding nested loops."""
    return tuple(
        node.value
        for node in _direct_loop_body_nodes(loop)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and _is_exact_call_to_symbol(node.value, symbol)
    )


def _direct_loop_augmented_target_names(
    loop: ast.For | ast.AsyncFor,
    *,
    symbol: str | None = None,
) -> tuple[str, ...]:
    """Return direct-loop accumulator names, optionally filtered by a call."""
    return tuple(
        node.target.id
        for node in _direct_loop_augments(loop, symbol=symbol)
    )


def _simple_name_assignment(node: ast.AST) -> tuple[str, ast.AST] | None:
    """Return a simple assigned name and value, if present."""
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id, node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return node.target.id, node.value
    return None


def _assigned_values_for_name(tree: ast.AST, name: str) -> tuple[ast.AST, ...]:
    """Return simple assignment values for a local name."""
    assignments = (
        _simple_name_assignment(node)
        for node in ast.walk(tree)
    )
    return tuple(
        value
        for assignment in assignments
        if assignment is not None
        for assigned_name, value in (assignment,)
        if assigned_name == name
    )


def _single_immutable_assignment_value(
    tree: ast.AST,
    name: str,
    *,
    before_line: int | None = None,
) -> ast.AST | None:
    """Return one simple assignment value when no other binding mutates it."""
    assignments = tuple(
        (node, assignment[1])
        for node in ast.walk(tree)
        for assignment in (_simple_name_assignment(node),)
        if assignment is not None and assignment[0] == name
    )
    bindings = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    )
    if len(assignments) != 1 or len(bindings) != 1:
        return None
    assignment, value = assignments[0]
    if (
        before_line is not None
        and getattr(assignment, "lineno", 0) >= before_line
    ):
        return None
    return value


def _cds_interval_cursor_writes_are_bounded(
    period_loop: ast.For | ast.AsyncFor,
    *,
    start_name: str = "interval_start",
    stop_name: str = "interval_stop",
) -> bool:
    """Admit only the guard and tail updates of the CDS interval cursor."""
    assignments = tuple(
        assignment[1]
        for node in ast.walk(period_loop)
        for assignment in (_simple_name_assignment(node),)
        if assignment is not None and assignment[0] == start_name
    )
    bindings = tuple(
        node
        for node in ast.walk(period_loop)
        if isinstance(node, ast.Name)
        and node.id == start_name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    )
    return (
        len(assignments) == 2
        and len(bindings) == 2
        and all(
            isinstance(value, ast.Name) and value.id == stop_name
            for value in assignments
        )
    )


def _ast_equivalent(left: ast.AST, right: ast.AST) -> bool:
    """Compare expression structure while ignoring source coordinates."""
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def _subscript_uses_name(node: ast.Subscript, name: str) -> bool:
    return isinstance(node.slice, ast.Name) and node.slice.id == name


def _subscript_uses_zero(node: ast.Subscript) -> bool:
    return (
        isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and not isinstance(node.slice.value, bool)
        and node.slice.value == 0
    )


def _expression_or_alias_matches(
    tree: ast.AST,
    expression: ast.AST,
    predicate: Callable[[ast.AST], bool],
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Apply a structural predicate through simple aliases and ``float`` wrappers."""
    if predicate(expression):
        return True
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "float"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _expression_or_alias_matches(
            tree,
            expression.args[0],
            predicate,
            seen_names=seen_names,
        )
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    return any(
        _expression_or_alias_matches(
            tree,
            assigned_value,
            predicate,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in _assigned_values_for_name(tree, expression.id)
    )


def _cashflow_constructor_keyword_matches(
    tree: ast.AST,
    expression: ast.AST,
    *,
    constructor_symbol: str,
    keyword_name: str,
    predicate: Callable[[ast.AST], bool],
) -> bool:
    """Match one keyword on the constructor passed directly to a PV primitive."""
    if not (
        isinstance(expression, ast.Call)
        and len(expression.args) == 1
        and not expression.keywords
        and isinstance(expression.args[0], ast.Call)
        and isinstance(expression.args[0].func, ast.Name)
        and expression.args[0].func.id == constructor_symbol
        and not expression.args[0].args
    ):
        return False
    matching_keywords = tuple(
        keyword
        for keyword in expression.args[0].keywords
        if keyword.arg == keyword_name
    )
    return len(matching_keywords) == 1 and _expression_or_alias_matches(
        tree,
        matching_keywords[0].value,
        predicate,
    )


def _is_weight_at_name(
    tree: ast.AST,
    expression: ast.AST,
    *,
    weight_attribute: str,
    index_name: str,
    required_symbol: str,
) -> bool:
    """Recognize ``*.{weight_attribute}[index_name]``."""
    return (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == weight_attribute
        and _expression_resolves_to_first_event_weights(
            tree,
            expression.value.value,
            required_symbol=required_symbol,
        )
        and _subscript_uses_name(expression, index_name)
    )


def _is_survival_weight_at_period_stop(
    tree: ast.AST,
    expression: ast.AST,
    *,
    stop_name: str,
    required_symbol: str,
) -> bool:
    """Recognize the post-event survival mass at ``period_stop - 1``."""
    if not (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "survival_weights"
        and _expression_resolves_to_first_event_weights(
            tree,
            expression.value.value,
            required_symbol=required_symbol,
        )
        and isinstance(expression.slice, ast.BinOp)
        and isinstance(expression.slice.op, ast.Sub)
        and isinstance(expression.slice.left, ast.Name)
        and expression.slice.left.id == stop_name
    ):
        return False
    decrement = expression.slice.right
    return (
        isinstance(decrement, ast.Constant)
        and isinstance(decrement.value, int)
        and not isinstance(decrement.value, bool)
        and decrement.value == 1
    )


def _expression_resolves_to_first_event_weights(
    tree: ast.AST,
    expression: ast.AST,
    *,
    required_symbol: str,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize the selected method's validated first-event weight result."""
    if isinstance(expression, ast.Call) and _call_matches_symbol(
        expression,
        required_symbol,
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    return bool(assigned_values) and all(
        _expression_resolves_to_first_event_weights(
            tree,
            assigned_value,
            required_symbol=required_symbol,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    )


def _expression_resolves_to_discount_curve(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize the active market discount curve and unambiguous aliases."""
    if (
        isinstance(expression, ast.Attribute)
        and expression.attr == "discount"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "market_state"
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    return bool(assigned_values) and all(
        _expression_resolves_to_discount_curve(
            tree,
            assigned_value,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    )


def _is_discount_call_at_period_payment(
    tree: ast.AST,
    expression: ast.AST,
    *,
    grid_expression: ast.AST,
    period_index_name: str,
) -> bool:
    """Recognize discounting at the active period's mapped payment time."""
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "discount")
        and isinstance(expression.func, ast.Attribute)
        and _expression_resolves_to_discount_curve(tree, expression.func.value)
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return False
    payment_time = expression.args[0]
    return (
        isinstance(payment_time, ast.Subscript)
        and isinstance(payment_time.value, ast.Attribute)
        and payment_time.value.attr == "period_payment_times"
        and _ast_equivalent(payment_time.value.value, grid_expression)
        and _subscript_uses_name(payment_time, period_index_name)
    )


def _interval_alias_names(
    interval_loop: ast.For | ast.AsyncFor,
    *,
    grid_expression: ast.AST,
    interval_index_name: str,
) -> frozenset[str]:
    """Return aliases bound to the active grid interval in the nested loop."""
    names: set[str] = set()
    for node in _direct_loop_body_nodes(interval_loop):
        assignment = _simple_name_assignment(node)
        if assignment is None:
            continue
        name, value = assignment
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Attribute)
            and value.value.attr == "intervals"
            and _ast_equivalent(value.value.value, grid_expression)
            and _subscript_uses_name(value, interval_index_name)
        ):
            names.add(name)
    return frozenset(names)


def _is_discount_call_at_interval_settlement(
    tree: ast.AST,
    expression: ast.AST,
    *,
    grid_expression: ast.AST,
    interval_index_name: str,
    interval_aliases: frozenset[str],
) -> bool:
    """Recognize discounting at the active event interval's settlement time."""
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "discount")
        and isinstance(expression.func, ast.Attribute)
        and _expression_resolves_to_discount_curve(tree, expression.func.value)
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return False
    settlement_time = expression.args[0]
    if not (
        isinstance(settlement_time, ast.Attribute)
        and settlement_time.attr == "settlement_time"
    ):
        return False
    interval = settlement_time.value
    if isinstance(interval, ast.Name):
        return interval.id in interval_aliases
    return (
        isinstance(interval, ast.Subscript)
        and isinstance(interval.value, ast.Attribute)
        and interval.value.attr == "intervals"
        and _ast_equivalent(interval.value.value, grid_expression)
        and _subscript_uses_name(interval, interval_index_name)
    )


def _expression_resolves_to_credit_curve(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize a direct or simply aliased ``credit_curve`` expression."""
    if (
        isinstance(expression, ast.Attribute)
        and expression.attr == "credit_curve"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "market_state"
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    return any(
        _expression_resolves_to_credit_curve(
            tree,
            assigned_value,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in _assigned_values_for_name(tree, expression.id)
    )


def _cds_conditional_event_grid(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> ast.AST | None:
    """Return the grid used to derive one conditional-probability expression."""
    if (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(
            expression,
            "conditional_event_probabilities_from_curve",
        )
        and len(expression.args) == 2
        and not expression.keywords
        and _expression_resolves_to_credit_curve(tree, expression.args[0])
    ):
        intervals = expression.args[1]
        if isinstance(intervals, ast.Attribute) and intervals.attr == "intervals":
            return intervals.value
        return None
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return None
    assigned_values = _assigned_values_for_name(tree, expression.id)
    if not assigned_values:
        return None
    grids = tuple(
        _cds_conditional_event_grid(
            tree,
            assigned_value,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    )
    if any(grid is None for grid in grids):
        return None
    first_grid = grids[0]
    if first_grid is None or not all(
        _ast_equivalent(first_grid, grid)
        for grid in grids[1:]
        if grid is not None
    ):
        return None
    return first_grid


def _cds_exact_initial_survival_grid(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> ast.AST | None:
    """Return the grid for an exact first-live survival expression."""
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "float"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _cds_exact_initial_survival_grid(
            tree,
            expression.args[0],
            seen_names=seen_names,
        )
    if isinstance(expression, ast.IfExp):
        grid_expression = _cds_exact_initial_survival_grid(
            tree,
            expression.body,
            seen_names=seen_names,
        )
        if grid_expression is None:
            return None
        guard = expression.test
        fallback = expression.orelse
        if not (
            isinstance(guard, ast.Attribute)
            and guard.attr == "intervals"
            and _ast_equivalent(guard.value, grid_expression)
            and isinstance(fallback, ast.Constant)
            and isinstance(fallback.value, (int, float))
            and not isinstance(fallback.value, bool)
            and float(fallback.value) == 1.0
        ):
            return None
        return grid_expression
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return None
        assigned_values = _assigned_values_for_name(tree, expression.id)
        if not assigned_values:
            return None
        grids = tuple(
            _cds_exact_initial_survival_grid(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
        if any(grid is None for grid in grids):
            return None
        first_grid = grids[0]
        if first_grid is None or not all(
            _ast_equivalent(first_grid, grid)
            for grid in grids[1:]
            if grid is not None
        ):
            return None
        return first_grid
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "survival_probability")
        and isinstance(expression.func, ast.Attribute)
        and len(expression.args) == 1
        and not expression.keywords
        and _expression_resolves_to_credit_curve(tree, expression.func.value)
    ):
        return None
    first_interval_start = expression.args[0]
    if not (
        isinstance(first_interval_start, ast.Attribute)
        and first_interval_start.attr == "start_time"
        and isinstance(first_interval_start.value, ast.Subscript)
    ):
        return None
    first_interval = first_interval_start.value
    if not (
        isinstance(first_interval.value, ast.Attribute)
        and first_interval.value.attr == "intervals"
        and _subscript_uses_zero(first_interval)
    ):
        return None
    return first_interval.value.value


def _cds_initial_survival_expression_is_valid(
    tree: ast.AST,
    expression: ast.AST,
    *,
    conditional_probabilities: ast.AST,
) -> bool:
    """Recognize exact survival to the active first live event interval."""
    conditional_grid = _cds_conditional_event_grid(
        tree,
        conditional_probabilities,
    )
    survival_grid = _cds_exact_initial_survival_grid(tree, expression)
    return (
        conditional_grid is not None
        and survival_grid is not None
        and _ast_equivalent(conditional_grid, survival_grid)
    )


def _enumerated_collection(
    loop: ast.For | ast.AsyncFor,
    attribute: str,
) -> tuple[str, ast.AST] | None:
    """Return the index name and collection owner for an unsliced enumerate."""
    if not (
        isinstance(loop.target, ast.Tuple)
        and len(loop.target.elts) == 2
        and isinstance(loop.target.elts[0], ast.Name)
        and isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "enumerate"
        and len(loop.iter.args) == 1
    ):
        return None
    collection = loop.iter.args[0]
    if not isinstance(collection, ast.Attribute) or collection.attr != attribute:
        return None
    return loop.target.elts[0].id, collection.value


def _mapping_stop_assignment(
    period_loop: ast.For | ast.AsyncFor,
    *,
    grid_expression: ast.AST,
    period_index_name: str,
) -> str | None:
    """Return the stop variable bound to this period's interval-map entry."""
    for node in _direct_loop_body_nodes(period_loop):
        assignment = _simple_name_assignment(node)
        if assignment is None:
            continue
        assigned_name, value = assignment
        if not isinstance(value, ast.Subscript):
            continue
        mapping = value.value
        if (
            isinstance(mapping, ast.Attribute)
            and mapping.attr == "period_interval_stops"
            and _ast_equivalent(mapping.value, grid_expression)
            and _subscript_uses_name(value, period_index_name)
        ):
            return assigned_name
    return None


def _loop_updates_start_from_stop(
    period_loop: ast.For | ast.AsyncFor,
    *,
    start_name: str,
    stop_name: str,
    after_line: int,
) -> bool:
    """Require the next interval slice to begin at the prior mapped stop."""
    for node in _direct_loop_body_nodes(period_loop):
        assignment = _simple_name_assignment(node)
        if assignment is None:
            continue
        assigned_name, value = assignment
        if (
            assigned_name == start_name
            and isinstance(value, ast.Name)
            and value.id == stop_name
            and getattr(node, "lineno", 0) > after_line
        ):
            return True
    return False


def _tree_initializes_name_to_zero_before(
    tree: ast.AST,
    *,
    name: str,
    before_line: int,
) -> bool:
    """Require an interval cursor to start at the beginning of the event grid."""
    for node in ast.walk(tree):
        assignment = _simple_name_assignment(node)
        if assignment is None:
            continue
        assigned_name, value = assignment
        if (
            assigned_name == name
            and isinstance(value, ast.Constant)
            and isinstance(value.value, int)
            and not isinstance(value.value, bool)
            and value.value == 0
            and getattr(node, "lineno", 0) < before_line
        ):
            return True
    return False


def _name_has_only_expected_accumulator_mutations(
    tree: ast.AST,
    *,
    name: str,
    before_line: int,
    allowed_augments: tuple[ast.AugAssign, ...],
) -> bool:
    """Own every write to one zero initializer or a recognized loop update."""
    assignments = tuple(
        (node, assignment[1])
        for node in ast.walk(tree)
        for assignment in (_simple_name_assignment(node),)
        if assignment is not None and assignment[0] == name
    )
    if len(assignments) != 1:
        return False
    node, value = assignments[0]
    initializer_target = (
        node.targets[0] if isinstance(node, ast.Assign) else node.target
    )
    if not (
        len(allowed_augments) == 1
        and isinstance(initializer_target, ast.Name)
        and all(
            isinstance(augment.target, ast.Name) and augment.target.id == name
            for augment in allowed_augments
        )
        and getattr(node, "lineno", 0) < before_line
        and isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and float(value.value) == 0.0
    ):
        return False

    allowed_targets = {
        id(initializer_target),
        *(id(augment.target) for augment in allowed_augments),
    }
    actual_targets = {
        id(candidate)
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Name)
        and candidate.id == name
        and isinstance(candidate.ctx, (ast.Store, ast.Del))
    }
    return actual_targets == allowed_targets


def _loop_references_indexed_grid_intervals(
    loop: ast.For | ast.AsyncFor,
    *,
    grid_expression: ast.AST,
    interval_index_name: str,
) -> bool:
    """Require interval lookup by the active mapped range index."""
    return any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "intervals"
        and _ast_equivalent(node.value.value, grid_expression)
        and _subscript_uses_name(node, interval_index_name)
        for node in ast.walk(loop)
    )


def _period_loop_has_required_empty_interval_guard(
    period_loop: ast.For | ast.AsyncFor,
    *,
    start_name: str,
    stop_name: str,
) -> bool:
    """Require the bounded no-live-interval skip before scheduled-leg pricing."""
    premium_augments = _direct_loop_augments(
        period_loop,
        symbol="coupon_cashflow_pv",
    )
    if not premium_augments:
        return False
    first_premium_line = min(
        getattr(augment, "lineno", 0)
        for augment in premium_augments
    )
    stop_assignment_indexes = tuple(
        index
        for index, statement in enumerate(period_loop.body)
        for assignment in (_simple_name_assignment(statement),)
        if assignment is not None and assignment[0] == stop_name
    )
    guard_indexes = tuple(
        index
        for index, statement in enumerate(period_loop.body)
        if _is_supported_cds_empty_period_guard(
            period_loop,
            statement,
            start_name=start_name,
            stop_name=stop_name,
        )
    )
    return (
        len(stop_assignment_indexes) == 1
        and len(guard_indexes) == 1
        and guard_indexes[0] == stop_assignment_indexes[0] + 1
        and getattr(period_loop.body[guard_indexes[0]], "lineno", 0)
        < first_premium_line
    )


def _cds_full_event_grid_loops(
    tree: ast.AST,
) -> tuple[
    tuple[ast.For | ast.AsyncFor, ast.For | ast.AsyncFor, str],
    ...,
]:
    """Return complete period/interval loop pairs and their mapped stop name."""
    matches: list[
        tuple[ast.For | ast.AsyncFor, ast.For | ast.AsyncFor, str]
    ] = []
    direct_body = tree.body if isinstance(tree, ast.Module) else ()
    for period_loop in direct_body:
        if not isinstance(period_loop, (ast.For, ast.AsyncFor)):
            continue
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None:
            continue
        period_index_name, grid_expression = period_collection
        stop_name = _mapping_stop_assignment(
            period_loop,
            grid_expression=grid_expression,
            period_index_name=period_index_name,
        )
        if stop_name is None:
            continue
        if not _direct_loop_body_augments_with_call(
            period_loop,
            "coupon_cashflow_pv",
        ):
            continue
        for interval_loop in _direct_loop_body_nodes(period_loop):
            if not isinstance(interval_loop, (ast.For, ast.AsyncFor)):
                continue
            if not (
                isinstance(interval_loop.target, ast.Name)
                and isinstance(interval_loop.iter, ast.Call)
                and isinstance(interval_loop.iter.func, ast.Name)
                and interval_loop.iter.func.id == "range"
                and len(interval_loop.iter.args) == 2
                and isinstance(interval_loop.iter.args[0], ast.Name)
                and isinstance(interval_loop.iter.args[1], ast.Name)
                and interval_loop.iter.args[1].id == stop_name
            ):
                continue
            start_name = interval_loop.iter.args[0].id
            if not _tree_initializes_name_to_zero_before(
                tree,
                name=start_name,
                before_line=getattr(period_loop, "lineno", 0),
            ):
                continue
            if not _period_loop_has_required_empty_interval_guard(
                period_loop,
                start_name=start_name,
                stop_name=stop_name,
            ):
                continue
            if not _loop_references_indexed_grid_intervals(
                interval_loop,
                grid_expression=grid_expression,
                interval_index_name=interval_loop.target.id,
            ):
                continue
            if not _direct_loop_body_augments_with_call(
                interval_loop,
                "protection_payment_pv",
            ):
                continue
            if _loop_updates_start_from_stop(
                period_loop,
                start_name=start_name,
                stop_name=stop_name,
                after_line=getattr(interval_loop, "lineno", 0),
            ):
                matches.append((period_loop, interval_loop, stop_name))
    return tuple(matches)


def _cds_composes_full_event_grid(tree: ast.AST) -> bool:
    """Recognize nested period/interval aggregation across a default-event grid."""
    return bool(_cds_full_event_grid_loops(tree))


def _cds_selected_weight_symbol(plan: GenerationPlan) -> str:
    """Return the first-event primitive selected by the active generation plan."""
    weight_symbols = {
        primitive.symbol
        for primitive in (
            plan.primitive_plan.primitives
            if plan.primitive_plan is not None
            else ()
        )
        if primitive.required
        and not primitive.excluded
        and primitive.symbol
        in {
            "expected_first_event_weights",
            "sample_first_event_weights",
        }
    }
    if len(weight_symbols) == 1:
        return next(iter(weight_symbols))
    method = str(plan.method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if method in {"mc", "monte_carlo"}:
        return "sample_first_event_weights"
    return "expected_first_event_weights"


def _cds_uses_active_event_weights(
    tree: ast.AST,
    *,
    required_symbol: str,
) -> bool:
    """Bind CDS premium and event cashflows to their active grid positions."""
    for period_loop, interval_loop, stop_name in _cds_full_event_grid_loops(tree):
        premium_values = _direct_loop_augmented_values_with_call(
            period_loop,
            "coupon_cashflow_pv",
        )
        if not premium_values or not all(
            _cashflow_constructor_keyword_matches(
                tree,
                value,
                constructor_symbol="CouponAccrual",
                keyword_name="weight",
                predicate=lambda expression: _is_survival_weight_at_period_stop(
                    tree,
                    expression,
                    stop_name=stop_name,
                    required_symbol=required_symbol,
                ),
            )
            for value in premium_values
        ):
            continue
        if not isinstance(interval_loop.target, ast.Name):
            continue
        interval_index_name = interval_loop.target.id
        protection_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "protection_payment_pv",
        )
        event_accrual_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "coupon_cashflow_pv",
        )
        def event_weight_matches(expression: ast.AST) -> bool:
            return _is_weight_at_name(
                tree,
                expression,
                weight_attribute="event_weights",
                index_name=interval_index_name,
                required_symbol=required_symbol,
            )
        if (
            protection_values
            and event_accrual_values
            and all(
                _cashflow_constructor_keyword_matches(
                    tree,
                    value,
                    constructor_symbol="ProtectionPayment",
                    keyword_name="default_probability",
                    predicate=event_weight_matches,
                )
                for value in protection_values
            )
            and all(
                _cashflow_constructor_keyword_matches(
                    tree,
                    value,
                    constructor_symbol="CouponAccrual",
                    keyword_name="weight",
                    predicate=event_weight_matches,
                )
                for value in event_accrual_values
            )
        ):
            return True
    return False


def _cds_uses_active_discount_times(tree: ast.AST) -> bool:
    """Bind premium and event cashflows to their mapped discount coordinates."""
    for period_loop, interval_loop, _ in _cds_full_event_grid_loops(tree):
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None or not isinstance(interval_loop.target, ast.Name):
            continue
        period_index_name, grid_expression = period_collection
        interval_index_name = interval_loop.target.id
        interval_aliases = _interval_alias_names(
            interval_loop,
            grid_expression=grid_expression,
            interval_index_name=interval_index_name,
        )

        def scheduled_discount_matches(expression: ast.AST) -> bool:
            return _is_discount_call_at_period_payment(
                tree,
                expression,
                grid_expression=grid_expression,
                period_index_name=period_index_name,
            )

        def event_discount_matches(expression: ast.AST) -> bool:
            return _is_discount_call_at_interval_settlement(
                tree,
                expression,
                grid_expression=grid_expression,
                interval_index_name=interval_index_name,
                interval_aliases=interval_aliases,
            )

        premium_values = _direct_loop_augmented_values_with_call(
            period_loop,
            "coupon_cashflow_pv",
        )
        protection_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "protection_payment_pv",
        )
        event_accrual_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "coupon_cashflow_pv",
        )
        if (
            premium_values
            and protection_values
            and event_accrual_values
            and all(
                _cashflow_constructor_keyword_matches(
                    tree,
                    value,
                    constructor_symbol="CouponAccrual",
                    keyword_name="discount_factor",
                    predicate=scheduled_discount_matches,
                )
                for value in premium_values
            )
            and all(
                _cashflow_constructor_keyword_matches(
                    tree,
                    value,
                    constructor_symbol="ProtectionPayment",
                    keyword_name="discount_factor",
                    predicate=event_discount_matches,
                )
                for value in protection_values
            )
            and all(
                _cashflow_constructor_keyword_matches(
                    tree,
                    value,
                    constructor_symbol="CouponAccrual",
                    keyword_name="discount_factor",
                    predicate=event_discount_matches,
                )
                for value in event_accrual_values
            )
        ):
            return True
    return False


def _expression_resolves_to_active_spec(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize ``self._spec``/``self.spec`` and unambiguous aliases."""
    if (
        isinstance(expression, ast.Attribute)
        and expression.attr in {"_spec", "spec"}
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "self"
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    return bool(assigned_values) and all(
        _expression_resolves_to_active_spec(
            tree,
            assigned_value,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    )


def _is_one_basis_point_scale(expression: ast.AST) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, (int, float))
        and not isinstance(expression.value, bool)
        and float(expression.value) == 1e-4
    )


def _is_basis_point_guard(test: ast.AST, name: str) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Gt)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and isinstance(test.comparators[0].value, (int, float))
        and not isinstance(test.comparators[0].value, bool)
        and float(test.comparators[0].value) == 1.0
    )


def _is_supported_spread_normalization(
    tree: ast.AST,
    assignment: ast.AugAssign,
    *,
    name: str,
    before_line: int | None = None,
) -> bool:
    """Allow only the documented guarded basis-point conversion."""
    if not (
        isinstance(assignment.target, ast.Name)
        and assignment.target.id == name
        and isinstance(assignment.op, ast.Mult)
        and _is_one_basis_point_scale(assignment.value)
    ):
        return False
    return any(
        isinstance(tree, ast.Module)
        and isinstance(node, ast.If)
        and node in tree.body
        and _is_basis_point_guard(node.test, name)
        and assignment in node.body
        and (
            before_line is None
            or (
                getattr(node, "lineno", 0) < before_line
                and getattr(assignment, "lineno", 0) < before_line
            )
        )
        for node in ast.walk(tree)
    )


def _expression_is_direct_active_spec_field(
    tree: ast.AST,
    expression: ast.AST,
    *,
    field: str,
) -> bool:
    """Recognize a direct active-spec field, including one float wrapper."""
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "float"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _expression_is_direct_active_spec_field(
            tree,
            expression.args[0],
            field=field,
        )
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == field
        and _expression_resolves_to_active_spec(tree, expression.value)
    )


def _expression_resolves_to_normalized_spread(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Require one dominating guarded basis-point normalization before use."""
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "float"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _expression_resolves_to_normalized_spread(
            tree,
            expression.args[0],
            seen_names=seen_names,
        )
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assignments = tuple(
        (node, assignment[1])
        for node in ast.walk(tree)
        for assignment in (_simple_name_assignment(node),)
        if assignment is not None and assignment[0] == expression.id
    )
    use_line = getattr(expression, "lineno", 0)
    if len(assignments) != 1:
        return False
    assignment, value = assignments[0]
    if getattr(assignment, "lineno", 0) >= use_line:
        return False
    assignment_target = (
        assignment.targets[0]
        if isinstance(assignment, ast.Assign)
        else assignment.target
    )
    actual_targets = {
        id(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == expression.id
        and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    if _expression_is_direct_active_spec_field(tree, value, field="spread"):
        mutations = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == expression.id
        )
        if len(mutations) != 1:
            return False
        mutation = mutations[0]
        return (
            getattr(assignment, "lineno", 0)
            < getattr(mutation, "lineno", 0)
            and actual_targets == {id(assignment_target), id(mutation.target)}
            and _is_supported_spread_normalization(
                tree,
                mutation,
                name=expression.id,
                before_line=use_line,
            )
        )
    return (
        actual_targets == {id(assignment_target)}
        and _expression_resolves_to_normalized_spread(
            tree,
            value,
            seen_names=seen_names | {expression.id},
        )
    )


def _expression_resolves_to_active_spec_field(
    tree: ast.AST,
    expression: ast.AST,
    *,
    field: str,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize an exact active-spec field with bounded alias handling."""
    if field == "spread":
        return _expression_resolves_to_normalized_spread(
            tree,
            expression,
            seen_names=seen_names,
        )
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "float"
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _expression_resolves_to_active_spec_field(
            tree,
            expression.args[0],
            field=field,
            seen_names=seen_names,
        )
    if (
        isinstance(expression, ast.Attribute)
        and expression.attr == field
        and _expression_resolves_to_active_spec(tree, expression.value)
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    if not assigned_values or not all(
        _expression_resolves_to_active_spec_field(
            tree,
            assigned_value,
            field=field,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    ):
        return False
    mutations = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == expression.id
    )
    return not mutations


def _is_integer_constant(expression: ast.AST, value: int) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        and expression.value == value
    )


def _expression_resolves_to_exact_number(
    tree: ast.AST,
    expression: ast.AST,
    *,
    value: float,
    integer_only: bool = False,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Resolve one immutable local alias chain to an exact numeric constant."""
    if isinstance(expression, ast.Constant) and not isinstance(
        expression.value,
        bool,
    ):
        if integer_only and not isinstance(expression.value, int):
            return False
        return (
            isinstance(expression.value, (int, float))
            and expression.value == value
        )
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    bindings = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == expression.id
        and isinstance(node.ctx, (ast.Store, ast.Del))
    )
    return (
        len(assigned_values) == 1
        and len(bindings) == 1
        and _expression_resolves_to_exact_number(
            tree,
            assigned_values[0],
            value=value,
            integer_only=integer_only,
            seen_names=seen_names | {expression.id},
        )
    )


def _cds_path_count_is_active_spec_control(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize the active path-count control or its declared 250k fallback."""
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_path_count_is_active_spec_control(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.BoolOp)
        and isinstance(expression.op, ast.Or)
        and len(expression.values) == 2
        and _is_integer_constant(expression.values[1], 250000)
    ):
        return False
    configured_paths = expression.values[0]
    if _expression_resolves_to_active_spec_field(
        tree,
        configured_paths,
        field="n_paths",
    ):
        return True
    return (
        isinstance(configured_paths, ast.Call)
        and isinstance(configured_paths.func, ast.Name)
        and configured_paths.func.id == "getattr"
        and len(configured_paths.args) == 3
        and not configured_paths.keywords
        and _expression_resolves_to_active_spec(tree, configured_paths.args[0])
        and isinstance(configured_paths.args[1], ast.Constant)
        and configured_paths.args[1].value == "n_paths"
        and _is_integer_constant(configured_paths.args[2], 250000)
    )


def _cds_sample_path_count_is_valid(tree: ast.AST, call: ast.Call) -> bool:
    """Require one sampled-weight path count bound to the active spec control."""
    path_counts = tuple(
        keyword.value
        for keyword in call.keywords
        if keyword.arg == "n_paths"
    )
    return len(path_counts) == 1 and _cds_path_count_is_active_spec_control(
        tree,
        path_counts[0],
    )


def _cds_sample_seed_is_valid(tree: ast.AST, call: ast.Call) -> bool:
    """Require the canonical reproducible seed, including simple aliases."""
    if any(keyword.arg is None for keyword in call.keywords):
        return False
    seeds = tuple(
        keyword.value
        for keyword in call.keywords
        if keyword.arg == "seed"
    )
    return len(seeds) == 1 and _expression_resolves_to_exact_number(
        tree,
        seeds[0],
        value=42.0,
        integer_only=True,
    )


def _expression_resolves_to_market_settlement(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Recognize the active market-state settlement date and simple aliases."""
    if (
        isinstance(expression, ast.Attribute)
        and expression.attr == "settlement"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "market_state"
    ):
        return True
    if not isinstance(expression, ast.Name) or expression.id in seen_names:
        return False
    assigned_values = _assigned_values_for_name(tree, expression.id)
    return bool(assigned_values) and all(
        _expression_resolves_to_market_settlement(
            tree,
            assigned_value,
            seen_names=seen_names | {expression.id},
        )
        for assigned_value in assigned_values
    )


def _cds_time_origin_is_active_valuation_date(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Require market settlement or valuation-date-with-start fallback."""
    if _expression_resolves_to_market_settlement(tree, expression):
        return True
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_time_origin_is_active_valuation_date(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.BoolOp)
        and isinstance(expression.op, ast.Or)
        and len(expression.values) == 2
    ):
        return False
    valuation_date, fallback_date = expression.values
    direct_valuation_date = _expression_resolves_to_active_spec_field(
        tree,
        valuation_date,
        field="valuation_date",
    )
    optional_valuation_date = (
        isinstance(valuation_date, ast.Call)
        and isinstance(valuation_date.func, ast.Name)
        and valuation_date.func.id == "getattr"
        and len(valuation_date.args) == 3
        and not valuation_date.keywords
        and _expression_resolves_to_active_spec(tree, valuation_date.args[0])
        and _constant_string(valuation_date.args[1]) == "valuation_date"
        and isinstance(valuation_date.args[2], ast.Constant)
        and valuation_date.args[2].value is None
    )
    return (direct_valuation_date or optional_valuation_date) and (
        _expression_resolves_to_active_spec_field(
            tree,
            fallback_date,
            field="start_date",
        )
        or _expression_resolves_to_market_settlement(tree, fallback_date)
    )


def _module_evaluate_uses_unshadowed_builtin_name(
    tree: ast.AST,
    *,
    name: str,
) -> bool:
    """Require an unqualified call target to resolve to its builtin binding."""
    if not isinstance(tree, ast.Module):
        return False
    evaluate_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate"
    )
    if len(evaluate_functions) != 1:
        return False
    evaluate = evaluate_functions[0]
    if name in _function_argument_names(evaluate) or any(
        _node_binds_name_in_current_scope(statement, name=name)
        for statement in evaluate.body
    ):
        return False
    return not any(
        _node_binds_name_in_current_scope(statement, name=name)
        for statement in tree.body
    )


def _tree_uses_optional_active_valuation_date_getattr(tree: ast.AST) -> bool:
    """Detect the bounded optional valuation-date access admitted for CDS."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and not node.keywords
        and _constant_string(node.args[1]) == "valuation_date"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
        for node in ast.walk(tree)
    )


def _cds_schedule_uses_active_valuation_origin(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Resolve a period schedule and validate its declared time origin."""
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_schedule_uses_active_valuation_origin(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "build_period_schedule")
    ):
        return False
    time_origins = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "time_origin"
    )
    return len(time_origins) == 1 and _cds_time_origin_is_active_valuation_date(
        tree,
        time_origins[0],
    )


def _cds_schedule_uses_active_contract_fields(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Resolve a period schedule and bind its contract-defining spec fields."""
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_schedule_uses_active_contract_fields(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "build_period_schedule")
        and len(expression.args) == 3
    ):
        return False
    day_counts = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "day_count"
    )
    calendars = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "calendar"
    )
    business_day_adjustments = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "bda"
    )
    roll_conventions = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "roll_convention"
    )
    stubs = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "stub"
    )
    payment_lags = tuple(
        keyword.value
        for keyword in expression.keywords
        if keyword.arg == "payment_lag_days"
    )
    return (
        _expression_resolves_to_active_spec_field(
            tree,
            expression.args[0],
            field="start_date",
        )
        and _expression_resolves_to_active_spec_field(
            tree,
            expression.args[1],
            field="end_date",
        )
        and _expression_resolves_to_active_spec_field(
            tree,
            expression.args[2],
            field="frequency",
        )
        and len(day_counts) == 1
        and _expression_resolves_to_active_spec_field(
            tree,
            day_counts[0],
            field="day_count",
        )
        and len(calendars) == 1
        and isinstance(calendars[0], ast.Name)
        and calendars[0].id == "WEEKEND_ONLY"
        and len(business_day_adjustments) == 1
        and isinstance(business_day_adjustments[0], ast.Attribute)
        and isinstance(business_day_adjustments[0].value, ast.Name)
        and business_day_adjustments[0].value.id == "BusinessDayAdjustment"
        and business_day_adjustments[0].attr == "FOLLOWING"
        and len(roll_conventions) == 1
        and isinstance(roll_conventions[0], ast.Attribute)
        and isinstance(roll_conventions[0].value, ast.Name)
        and roll_conventions[0].value.id == "RollConvention"
        and roll_conventions[0].attr == "NONE"
        and len(stubs) == 1
        and isinstance(stubs[0], ast.Attribute)
        and isinstance(stubs[0].value, ast.Name)
        and stubs[0].value.id == "StubType"
        and stubs[0].attr == "SHORT_LAST"
        and len(payment_lags) == 1
        and _is_integer_constant(payment_lags[0], 0)
    )


def _cds_grid_uses_active_valuation_origin(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Resolve a default-event grid back to its valuation-origin schedule."""
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_grid_uses_active_valuation_origin(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "build_default_event_grid")
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return False
    return _cds_schedule_uses_active_valuation_origin(
        tree,
        expression.args[0],
    )


def _cds_grid_uses_active_contract_fields(
    tree: ast.AST,
    expression: ast.AST,
    *,
    seen_names: frozenset[str] = frozenset(),
) -> bool:
    """Resolve a default-event grid back to its active contract schedule."""
    if isinstance(expression, ast.Name):
        if expression.id in seen_names:
            return False
        assigned_values = _assigned_values_for_name(tree, expression.id)
        return bool(assigned_values) and all(
            _cds_grid_uses_active_contract_fields(
                tree,
                assigned_value,
                seen_names=seen_names | {expression.id},
            )
            for assigned_value in assigned_values
        )
    if not (
        isinstance(expression, ast.Call)
        and _call_matches_symbol(expression, "build_default_event_grid")
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return False
    return _cds_schedule_uses_active_contract_fields(tree, expression.args[0])


def _cds_uses_active_schedule_fields(tree: ast.AST) -> bool:
    """Bind every accepted cashflow grid to active CDS schedule fields."""
    loop_pairs = _cds_full_event_grid_loops(tree)
    if not loop_pairs:
        return False
    for period_loop, _, _ in loop_pairs:
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None:
            return False
        _, grid_expression = period_collection
        if not _cds_grid_uses_active_contract_fields(tree, grid_expression):
            return False
    return True


def _cds_uses_active_credit_curve(tree: ast.AST) -> bool:
    """Bind conditional probabilities and survival to the active credit curve."""
    conditional_calls = _find_calls_for_symbol(
        tree,
        "conditional_event_probabilities_from_curve",
    )
    survival_calls = _find_calls_for_symbol(tree, "survival_probability")
    return (
        bool(conditional_calls)
        and bool(survival_calls)
        and all(
            len(call.args) == 2
            and not call.keywords
            and _expression_resolves_to_credit_curve(tree, call.args[0])
            for call in conditional_calls
        )
        and all(
            isinstance(call.func, ast.Attribute)
            and _expression_resolves_to_credit_curve(tree, call.func.value)
            for call in survival_calls
        )
    )


def _cds_uses_valuation_origin_event_grid(tree: ast.AST) -> bool:
    """Bind every accepted cashflow loop to a valuation-origin event grid."""
    loop_pairs = _cds_full_event_grid_loops(tree)
    if not loop_pairs:
        return False
    grid_expressions: list[ast.AST] = []
    for period_loop, _, _ in loop_pairs:
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None:
            return False
        _, grid_expression = period_collection
        grid_expressions.append(grid_expression)
    if not all(
        _cds_grid_uses_active_valuation_origin(tree, grid_expression)
        for grid_expression in grid_expressions
    ):
        return False
    weighting_calls = tuple(
        call
        for symbol in (
            "expected_first_event_weights",
            "sample_first_event_weights",
        )
        for call in _find_calls_for_symbol(tree, symbol)
    )
    if not weighting_calls or any(not call.args for call in weighting_calls):
        return False
    conditional_grids = tuple(
        _cds_conditional_event_grid(tree, call.args[0])
        for call in weighting_calls
    )
    return all(
        conditional_grid is not None
        and any(
            _ast_equivalent(conditional_grid, grid_expression)
            for grid_expression in grid_expressions
        )
        for conditional_grid in conditional_grids
    )


def _multiplication_factors(expression: ast.AST) -> tuple[ast.AST, ...]:
    """Flatten an associative multiplication into its exact factors."""
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        return (
            _multiplication_factors(expression.left)
            + _multiplication_factors(expression.right)
        )
    return (expression,)


def _is_active_period_accrual(
    tree: ast.AST,
    expression: ast.AST,
    *,
    period_name: str,
) -> bool:
    return _expression_or_alias_matches(
        tree,
        expression,
        lambda candidate: (
            isinstance(candidate, ast.Attribute)
            and candidate.attr == "accrual_fraction"
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == period_name
        ),
    )


def _is_active_interval_elapsed_fraction(
    tree: ast.AST,
    expression: ast.AST,
    *,
    grid_expression: ast.AST,
    interval_index_name: str,
    interval_aliases: frozenset[str],
) -> bool:
    """Recognize the active interval's within-period elapsed fraction."""

    def matches(candidate: ast.AST) -> bool:
        if not (
            isinstance(candidate, ast.Attribute)
            and candidate.attr == "period_fraction_elapsed"
        ):
            return False
        interval = candidate.value
        if isinstance(interval, ast.Name):
            return interval.id in interval_aliases
        return (
            isinstance(interval, ast.Subscript)
            and isinstance(interval.value, ast.Attribute)
            and interval.value.attr == "intervals"
            and _ast_equivalent(interval.value.value, grid_expression)
            and _subscript_uses_name(interval, interval_index_name)
        )

    return _expression_or_alias_matches(tree, expression, matches)


def _cds_event_accrual_matches(
    tree: ast.AST,
    expression: ast.AST,
    *,
    period_name: str,
    grid_expression: ast.AST,
    interval_index_name: str,
    interval_aliases: frozenset[str],
) -> bool:
    """Require period accrual multiplied by the active interval elapsed fraction."""

    def matches(candidate: ast.AST) -> bool:
        factors = _multiplication_factors(candidate)
        return (
            len(factors) == 2
            and sum(
                _is_active_period_accrual(
                    tree,
                    factor,
                    period_name=period_name,
                )
                for factor in factors
            )
            == 1
            and sum(
                _is_active_interval_elapsed_fraction(
                    tree,
                    factor,
                    grid_expression=grid_expression,
                    interval_index_name=interval_index_name,
                    interval_aliases=interval_aliases,
                )
                for factor in factors
            )
            == 1
        )

    return _expression_or_alias_matches(tree, expression, matches)


def _constructor_keyword_matches(
    tree: ast.AST,
    expression: ast.AST,
    *,
    constructor: str,
    keyword_name: str,
    predicate: Callable[[ast.AST], bool],
) -> bool:
    """Require every matching constructor to bind one validated keyword."""
    constructors = tuple(
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Call) and _call_matches_symbol(node, constructor)
    )
    return bool(constructors) and all(
        sum(
            keyword.arg == keyword_name
            and _expression_or_alias_matches(tree, keyword.value, predicate)
            for keyword in call.keywords
        )
        == 1
        for call in constructors
    )


def _constructor_signs_are_positive(
    tree: ast.AST,
    expression: ast.AST,
    *,
    constructor: str,
) -> bool:
    """Require constructor signs to be absent or resolve exactly to positive one."""
    constructors = tuple(
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Call) and _call_matches_symbol(node, constructor)
    )
    if not constructors:
        return False
    for call in constructors:
        if any(keyword.arg is None for keyword in call.keywords):
            return False
        signs = tuple(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "sign"
        )
        if signs and not (
            len(signs) == 1
            and _expression_resolves_to_exact_number(
                tree,
                signs[0],
                value=1.0,
            )
        ):
            return False
    return True


def _cds_uses_active_coupon_accruals(tree: ast.AST) -> bool:
    """Bind scheduled and event coupon accruals to active grid coordinates."""
    for period_loop, interval_loop, _ in _cds_full_event_grid_loops(tree):
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None or not (
            isinstance(period_loop.target, ast.Tuple)
            and len(period_loop.target.elts) == 2
            and isinstance(period_loop.target.elts[1], ast.Name)
            and isinstance(interval_loop.target, ast.Name)
        ):
            continue
        _, grid_expression = period_collection
        period_name = period_loop.target.elts[1].id
        interval_index_name = interval_loop.target.id
        interval_aliases = _interval_alias_names(
            interval_loop,
            grid_expression=grid_expression,
            interval_index_name=interval_index_name,
        )
        premium_values = _direct_loop_augmented_values_with_call(
            period_loop,
            "coupon_cashflow_pv",
        )
        event_accrual_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "coupon_cashflow_pv",
        )
        if (
            premium_values
            and event_accrual_values
            and all(
                _constructor_keyword_matches(
                    tree,
                    value,
                    constructor="CouponAccrual",
                    keyword_name="accrual",
                    predicate=lambda expression: _is_active_period_accrual(
                        tree,
                        expression,
                        period_name=period_name,
                    ),
                )
                for value in premium_values
            )
            and all(
                _constructor_keyword_matches(
                    tree,
                    value,
                    constructor="CouponAccrual",
                    keyword_name="accrual",
                    predicate=lambda expression: _cds_event_accrual_matches(
                        tree,
                        expression,
                        period_name=period_name,
                        grid_expression=grid_expression,
                        interval_index_name=interval_index_name,
                        interval_aliases=interval_aliases,
                    ),
                )
                for value in event_accrual_values
            )
        ):
            return True
    return False


def _is_active_elapsed_period_fraction(
    expression: ast.AST,
    *,
    grid_expression: ast.AST,
    period_index_name: str,
) -> bool:
    return (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "elapsed_period_fractions"
        and _ast_equivalent(expression.value.value, grid_expression)
        and _subscript_uses_name(expression, period_index_name)
    )


def _cds_valuation_adjustment_matches(
    tree: ast.AST,
    expression: ast.AST,
    *,
    grid_expression: ast.AST,
    period_index_name: str,
    period_name: str,
) -> bool:
    """Require the exact notional/spread/accrual/elapsed adjustment product."""
    factors = _multiplication_factors(expression)
    if len(factors) != 4:
        return False
    predicates = (
        lambda factor: _expression_resolves_to_active_spec_field(
            tree,
            factor,
            field="notional",
        ),
        lambda factor: _expression_resolves_to_active_spec_field(
            tree,
            factor,
            field="spread",
        ),
        lambda factor: _is_active_period_accrual(
            tree,
            factor,
            period_name=period_name,
        ),
        lambda factor: _is_active_elapsed_period_fraction(
            factor,
            grid_expression=grid_expression,
            period_index_name=period_index_name,
        ),
    )
    return all(
        sum(predicate(factor) for factor in factors) == 1
        for predicate in predicates
    )


def _cds_binds_active_economic_terms(tree: ast.AST) -> bool:
    """Bind all CDS legs and valuation accrual to the active spec fields."""
    for period_loop, interval_loop, _ in _cds_full_event_grid_loops(tree):
        period_collection = _enumerated_collection(period_loop, "periods")
        if period_collection is None or not (
            isinstance(period_loop.target, ast.Tuple)
            and len(period_loop.target.elts) == 2
            and isinstance(period_loop.target.elts[1], ast.Name)
        ):
            continue
        period_index_name, grid_expression = period_collection
        period_name = period_loop.target.elts[1].id
        premium_values = _direct_loop_augmented_values_with_call(
            period_loop,
            "coupon_cashflow_pv",
        )
        protection_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "protection_payment_pv",
        )
        event_accrual_values = _direct_loop_augmented_values_with_call(
            interval_loop,
            "coupon_cashflow_pv",
        )
        premium_names = set(
            _direct_loop_augmented_target_names(
                period_loop,
                symbol="coupon_cashflow_pv",
            )
        )
        valuation_values = tuple(
            node.value
            for node in _direct_loop_body_nodes(period_loop)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Name)
            and node.target.id not in premium_names
        )

        def field_matches(
            value: ast.AST,
            constructor: str,
            keyword_name: str,
            field: str,
        ) -> bool:
            constructors = tuple(
                node
                for node in ast.walk(value)
                if isinstance(node, ast.Call)
                and _call_matches_symbol(node, constructor)
            )
            return bool(constructors) and all(
                any(
                    keyword.arg == keyword_name
                    and _expression_resolves_to_active_spec_field(
                        tree,
                        keyword.value,
                        field=field,
                    )
                    for keyword in call.keywords
                )
                for call in constructors
            )

        if (
            len(premium_values) == 1
            and len(protection_values) == 1
            and len(event_accrual_values) == 1
            and len(valuation_values) == 1
            and all(
                field_matches(value, "CouponAccrual", "notional", "notional")
                and field_matches(value, "CouponAccrual", "rate", "spread")
                and _constructor_signs_are_positive(
                    tree,
                    value,
                    constructor="CouponAccrual",
                )
                for value in premium_values + event_accrual_values
            )
            and all(
                field_matches(
                    value,
                    "ProtectionPayment",
                    "notional",
                    "notional",
                )
                and field_matches(
                    value,
                    "ProtectionPayment",
                    "recovery",
                    "recovery",
                )
                and _constructor_signs_are_positive(
                    tree,
                    value,
                    constructor="ProtectionPayment",
                )
                for value in protection_values
            )
            and all(
                _cds_valuation_adjustment_matches(
                    tree,
                    value,
                    grid_expression=grid_expression,
                    period_index_name=period_index_name,
                    period_name=period_name,
                )
                for value in valuation_values
            )
        ):
            return True
    return False


def _signed_name_terms(
    expression: ast.AST,
    *,
    sign: int = 1,
) -> tuple[tuple[str, int], ...] | None:
    """Flatten a signed additive expression into exact local-name terms."""
    if isinstance(expression, ast.Name):
        return ((expression.id, sign),)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.USub):
        return _signed_name_terms(expression.operand, sign=-sign)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _signed_name_terms(expression.left, sign=sign)
        right = _signed_name_terms(expression.right, sign=sign)
    elif isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Sub):
        left = _signed_name_terms(expression.left, sign=sign)
        right = _signed_name_terms(expression.right, sign=-sign)
    else:
        return None
    if left is None or right is None:
        return None
    return left + right


def _cds_preserves_sign_convention(tree: ast.AST) -> bool:
    """Require protection-buyer signs on the active four leg accumulators."""
    returned_terms: list[list[tuple[str, int]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        expression = node.value
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "float"
            and len(expression.args) == 1
            and not expression.keywords
        ):
            expression = expression.args[0]
        terms = _signed_name_terms(expression)
        if terms is not None:
            returned_terms.append(sorted(terms))
    for period_loop, interval_loop, _ in _cds_full_event_grid_loops(tree):
        premium_augments = _direct_loop_augments(
            period_loop,
            symbol="coupon_cashflow_pv",
        )
        protection_augments = _direct_loop_augments(
            interval_loop,
            symbol="protection_payment_pv",
        )
        event_accrual_augments = _direct_loop_augments(
            interval_loop,
            symbol="coupon_cashflow_pv",
        )
        premium_names = tuple(augment.target.id for augment in premium_augments)
        valuation_accrual_augments = tuple(
            augment
            for augment in _direct_loop_augments(period_loop)
            if augment.target.id not in premium_names
        )
        for protection_augment in protection_augments:
            for premium_augment in premium_augments:
                for event_accrual_augment in event_accrual_augments:
                    for valuation_accrual_augment in valuation_accrual_augments:
                        protection_name = protection_augment.target.id
                        premium_name = premium_augment.target.id
                        event_accrual_name = event_accrual_augment.target.id
                        valuation_accrual_name = valuation_accrual_augment.target.id
                        accumulator_names = (
                            protection_name,
                            premium_name,
                            event_accrual_name,
                            valuation_accrual_name,
                        )
                        expected = sorted(
                            (
                                (protection_name, 1),
                                (premium_name, -1),
                                (event_accrual_name, -1),
                                (valuation_accrual_name, 1),
                            )
                        )
                        if (
                            len(set(accumulator_names)) == 4
                            and all(
                                _name_has_only_expected_accumulator_mutations(
                                    tree,
                                    name=name,
                                    before_line=period_loop.lineno,
                                    allowed_augments=(augment,),
                                )
                                for name, augment in zip(
                                    accumulator_names,
                                    (
                                        protection_augment,
                                        premium_augment,
                                        event_accrual_augment,
                                        valuation_accrual_augment,
                                    ),
                                    strict=True,
                                )
                            )
                            and expected in returned_terms
                        ):
                            return True
    return False


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
        findings.extend(
            self._check_terminal_basket_boundary(
                source,
                route_spec,
                exact_surface_primitives,
            )
        )
        findings.extend(
            self._check_zcb_option_boundary(
                source,
                route_spec,
                exact_surface_primitives,
            )
        )
        findings.extend(
            self._check_credit_default_swap_boundary(
                source,
                plan,
                route_spec,
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
                or prim.owns_engine_family
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
        enforce_whole_route = (
            route_spec.id in _EXPLICIT_COMPOSITION_ROUTE_IDS
            or any(
                primitive.symbol == "resolve_terminal_basket_inputs"
                for primitive in exact_surface_primitives
            )
        )
        required_primitives = tuple(
            primitive
            for primitive in exact_surface_primitives
            if enforce_whole_route
            or primitive.owns_engine_family
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

    def _check_terminal_basket_boundary(
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
    ) -> list[SemanticFinding]:
        """Reject compatibility wrappers and ranked-basket substitution."""
        if not any(
            primitive.symbol == "resolve_terminal_basket_inputs"
            for primitive in exact_surface_primitives
        ):
            return []
        findings: list[SemanticFinding] = []
        for symbol in sorted(_TERMINAL_BASKET_FORBIDDEN_SYMBOLS):
            if not _calls_symbol(source, symbol):
                continue
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="terminal_basket_forbidden_helper",
                    message=(
                        f"Route '{route_spec.id}' is an ordinary terminal-basket "
                        f"composition and cannot call '{symbol}'. Use the declared "
                        "resolver, raw method kernel, process/engine, and payoff "
                        "primitives instead."
                    ),
                )
            )
        return findings

    def _check_zcb_option_boundary(
        self,
        source: str,
        route_spec: RouteSpec,
        exact_surface_primitives,
    ) -> list[SemanticFinding]:
        """Reject compatibility wrappers on the raw ZCB-option route."""
        if route_spec.id != "short_rate_bond_option":
            return []
        findings: list[SemanticFinding] = []
        for symbol in sorted(_ZCB_OPTION_FORBIDDEN_SYMBOLS):
            if not _calls_symbol(source, symbol):
                continue
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="zcb_option_forbidden_helper",
                    message=(
                        f"Route '{route_spec.id}' cannot call compatibility surface "
                        f"'{symbol}'. Compose the shared discount-bond claim resolver "
                        "with the raw Jamshidian kernel or generic calibrated-lattice "
                        "and partial-horizon rollback primitives."
                    ),
                )
            )
        return findings

    def _check_credit_default_swap_boundary(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec,
    ) -> list[SemanticFinding]:
        """Enforce the public CDS first-event composition boundary."""
        if route_spec.id != "credit_default_swap":
            return []
        findings: list[SemanticFinding] = []
        for symbol in sorted(_CREDIT_DEFAULT_SWAP_FORBIDDEN_SYMBOLS):
            if not _calls_symbol(source, symbol):
                continue
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="credit_default_swap_forbidden_helper",
                    message=(
                        f"Route '{route_spec.id}' cannot call compatibility surface "
                        f"'{symbol}'. Build the public period schedule, first-event "
                        "grid, survival-derived weights, and signed premium/protection "
                        "cashflows explicitly."
                    ),
                )
            )

        try:
            module_tree = ast.parse(source)
        except SyntaxError:
            return findings

        tree = _reachable_evaluate_tree(module_tree)
        if tree is None:
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="credit_default_swap_incomplete_event_grid",
                    message=(
                        f"Route '{route_spec.id}' must expose exactly one evaluate "
                        "body with no unsupported conditional return/raise exits "
                        "before its "
                        "direct final signed return. Composition in unused or nested "
                        "helpers does not satisfy the pricing contract."
                    ),
                )
            )
            return findings

        invalid_cashflow_imports = tuple(
            symbol
            for symbol in (
                "coupon_cashflow_pv",
                "protection_payment_pv",
            )
            if not _module_has_authoritative_cds_cashflow_import(
                module_tree,
                symbol=symbol,
            )
        )
        if invalid_cashflow_imports:
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="credit_default_swap_cashflow_primitive_binding",
                    message=(
                        f"Route '{route_spec.id}' must call the directly imported "
                        "public cashflow primitives from "
                        f"'{_CDS_CASHFLOW_PRIMITIVE_MODULE}' without attribute "
                        "dispatch, aliases, or local/module shadowing. Invalid "
                        f"bindings: {', '.join(invalid_cashflow_imports)}."
                    ),
                )
            )

        selected_weight_symbol = _cds_selected_weight_symbol(plan)

        for symbol in (
            "expected_first_event_weights",
            "sample_first_event_weights",
        ):
            for call in _find_calls_for_symbol(tree, symbol):
                initial_survival = next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == "initial_survival_weight"
                    ),
                    None,
                )
                if (
                    initial_survival is not None
                    and call.args
                    and _cds_initial_survival_expression_is_valid(
                        tree,
                        initial_survival,
                        conditional_probabilities=call.args[0],
                    )
                ):
                    continue
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_initial_survival_missing",
                        message=(
                            f"Route '{route_spec.id}' must call '{symbol}' with "
                            "initial_survival_weight equal to credit-curve survival "
                            "at the first live event-grid interval start. Conditional "
                            "weights alone overstate forward-start CDS cashflows."
                        ),
                    )
                )

        if selected_weight_symbol == "sample_first_event_weights":
            sampled_weight_calls = _find_calls_for_symbol(
                tree,
                "sample_first_event_weights",
            )
            invalid_path_count = any(
                not _cds_sample_path_count_is_valid(tree, call)
                for call in sampled_weight_calls
            )
            if invalid_path_count:
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_path_count_binding",
                        message=(
                            f"Route '{route_spec.id}' must bind sampled first-event "
                            "weights to the active spec's n_paths control, with only "
                            "the declared 250000-path fallback."
                        ),
                    )
                )
            invalid_seed = any(
                not _cds_sample_seed_is_valid(tree, call)
                for call in sampled_weight_calls
            )
            if invalid_seed:
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_seed_binding",
                        message=(
                            f"Route '{route_spec.id}' must bind sampled first-event "
                            "weights to the canonical reproducible seed 42; omitted, "
                            "None, opaque, or other seed values are not admitted."
                        ),
                    )
                )

        composes_full_grid = _cds_composes_full_event_grid(tree)
        if not composes_full_grid:
            findings.append(
                SemanticFinding(
                    validator="algorithm_contract",
                    severity="error",
                    category="credit_default_swap_incomplete_event_grid",
                    message=(
                        f"Route '{route_spec.id}' must aggregate scheduled premium "
                        "cashflows across every event-grid period and protection "
                        "cashflows across the nested period-to-interval mapping in "
                        "the reachable evaluate body. Pricing fixed index positions, "
                        "unreachable statements, unused helpers, or wrapped, negated, "
                        "or scaled cashflow calls does not cover the CDS horizon."
                    ),
                )
            )
        else:
            valuation_origin_is_valid = _cds_uses_valuation_origin_event_grid(tree)
            if (
                valuation_origin_is_valid
                and _tree_uses_optional_active_valuation_date_getattr(tree)
                and not _module_evaluate_uses_unshadowed_builtin_name(
                    module_tree,
                    name="getattr",
                )
            ):
                valuation_origin_is_valid = False
            if not valuation_origin_is_valid:
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_valuation_origin",
                        message=(
                            f"Route '{route_spec.id}' must build the active "
                            "cashflow event grid from a period schedule whose "
                            "time_origin is the valuation date (with the "
                            "declared start-date fallback). Using start_date "
                            "unconditionally prices forward-start CDS cashflows "
                            "conditional on survival to the contract start."
                        ),
                    )
                )
            if not _cds_uses_active_schedule_fields(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_schedule_binding",
                        message=(
                            f"Route '{route_spec.id}' must build the active "
                            "cashflow schedule from the spec's start date, end "
                            "date, frequency, and day-count convention, using "
                            "the bounded route's weekend calendar, following "
                            "adjustment, no roll, short-last stub, and zero "
                            "payment lag."
                        ),
                    )
                )
            if not _cds_uses_active_credit_curve(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_credit_curve_binding",
                        message=(
                            f"Route '{route_spec.id}' must derive conditional "
                            "event probabilities and initial survival from the "
                            "active market credit curve."
                        ),
                    )
                )
            if not _cds_uses_active_event_weights(
                tree,
                required_symbol=selected_weight_symbol,
            ):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_weight_mapping",
                        message=(
                            f"Route '{route_spec.id}' must use the mapped period-stop "
                            "survival weight for scheduled premium and the active "
                            "interval's event weight for protection and event accrual."
                        ),
                    )
                )
            if not _cds_uses_active_coupon_accruals(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_accrual_mapping",
                        message=(
                            f"Route '{route_spec.id}' must use the active period "
                            "accrual for scheduled premium and multiply it by "
                            "the active interval's elapsed fraction for event "
                            "accrual."
                        ),
                    )
                )
            if not _cds_uses_active_discount_times(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_discount_mapping",
                        message=(
                            f"Route '{route_spec.id}' must discount scheduled "
                            "premium at the mapped period payment time and "
                            "protection/event accrual at the active interval's "
                            "settlement time."
                        ),
                    )
                )
            if not _cds_binds_active_economic_terms(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_economic_binding",
                        message=(
                            f"Route '{route_spec.id}' must bind premium, "
                            "protection, event accrual, and valuation accrual "
                            "to the active spec's notional, normalized spread, "
                            "and recovery fields. The guarded spread normalization "
                            "must dominate every alias and cashflow use. Hard-coded "
                            "or raw economic terms can produce a valid-looking but "
                            "incorrect CDS PV."
                        ),
                    )
                )
            if not _cds_preserves_sign_convention(tree):
                findings.append(
                    SemanticFinding(
                        validator="algorithm_contract",
                        severity="error",
                        category="credit_default_swap_sign_convention",
                        message=(
                            f"Route '{route_spec.id}' must return the protection-buyer "
                            "value `protection_leg - premium_leg - accrued_on_event "
                            "+ accrued_to_valuation`."
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
