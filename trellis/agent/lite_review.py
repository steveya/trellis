"""Deterministic lite-reviewer for obviously wrong generated code.

This stage sits between semantic validation and the heavier runtime critic /
model-validator flow. It is intentionally conservative: it only flags
high-confidence anti-patterns that are cheap to detect and expensive to let
through to later LLM-backed review.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


_CAPABILITY_ATTR_PREFIXES = {
    "discount_curve": ("market_state.discount",),
    "forward_curve": ("market_state.forward_curve", "market_state.forecast_curves"),
    "black_vol_surface": ("market_state.vol_surface",),
    "fx_rates": ("market_state.fx_rates",),
    "spot": ("market_state.spot", "market_state.underlier_spots"),
    "local_vol_surface": ("market_state.local_vol_surface", "market_state.local_vol_surfaces"),
}

_SUSPICIOUS_LITERAL_NAMES = {
    "discount_curve": {"r", "rate", "risk_free_rate", "discount_rate"},
    "forward_curve": {"foreign_rate", "carry", "drift", "forward_curve"},
    "black_vol_surface": {"sigma", "vol", "volatility"},
    "fx_rates": {"fx", "fx_rate", "fx_spot", "exchange_rate"},
    "spot": {"spot", "s0", "underlier_spot", "underlying_spot"},
    "local_vol_surface": {"local_vol", "sigma_local", "local_vol_surface", "local_sigma"},
}

# Legacy _ROUTE_REQUIRED_ACCESSES and _ROUTE_ACCESS_ERROR_CODES removed.
# Market-data access requirements are now sourced from the route registry
# (routes.yaml market_data_access field) via _route_required_accesses_for().


@dataclass(frozen=True)
class LiteReviewIssue:
    """Single deterministic lite-review issue."""

    code: str
    message: str


@dataclass(frozen=True)
class LiteReviewSignals:
    """Static fingerprint of obviously suspicious generation patterns."""

    market_state_accesses: tuple[str, ...]
    call_names: tuple[str, ...]
    literal_assignments: tuple[str, ...]
    wall_clock_calls: tuple[str, ...]


@dataclass(frozen=True)
class LiteReviewReport:
    """Outcome of the deterministic lite review."""

    issues: tuple[LiteReviewIssue, ...]
    signals: LiteReviewSignals

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(f"{issue.code}: {issue.message}" for issue in self.issues)


class _LiteReviewVisitor(ast.NodeVisitor):
    """Extract cheap signals for obviously wrong generated pricing code."""

    def __init__(self) -> None:
        self.market_state_accesses: set[str] = set()
        self.call_names: set[str] = set()
        self.literal_assignments: list[str] = []
        self.wall_clock_calls: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _resolve_attribute_chain(node)
        if chain.startswith("market_state."):
            self.market_state_accesses.add(chain)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = _numeric_literal(node.value)
        if value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.literal_assignments.append(f"{target.id}={value}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = _numeric_literal(node.value)
        if value is not None and isinstance(node.target, ast.Name):
            self.literal_assignments.append(f"{node.target.id}={value}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _resolve_call_name(node.func)
        if call_name:
            self.call_names.add(call_name)
        if call_name in {"date.today", "datetime.today", "datetime.now"}:
            self.wall_clock_calls.add(call_name)
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            value = _numeric_literal(keyword.value)
            if value is not None:
                self.literal_assignments.append(f"{keyword.arg}={value}")
        self.generic_visit(node)


def review_generated_code(
    source: str,
    *,
    pricing_plan=None,
    product_ir=None,
    generation_plan=None,
) -> LiteReviewReport:
    """Run a cheap deterministic review for high-confidence anti-patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        issue = LiteReviewIssue(
            code="lite.parse_error",
            message=f"cannot run lite review because the generated code is not valid Python: {exc}",
        )
        return LiteReviewReport(
            issues=(issue,),
            signals=LiteReviewSignals(
                market_state_accesses=(),
                call_names=(),
                literal_assignments=(),
                wall_clock_calls=(),
            ),
        )

    visitor = _LiteReviewVisitor()
    visitor.visit(tree)
    signals = LiteReviewSignals(
        market_state_accesses=tuple(sorted(visitor.market_state_accesses)),
        call_names=tuple(sorted(visitor.call_names)),
        literal_assignments=tuple(visitor.literal_assignments),
        wall_clock_calls=tuple(sorted(visitor.wall_clock_calls)),
    )

    required_market_data = set()
    if pricing_plan is not None:
        required_market_data.update(getattr(pricing_plan, "required_market_data", ()) or ())
    if product_ir is not None:
        required_market_data.update(getattr(product_ir, "required_market_data", ()) or ())

    issues: list[LiteReviewIssue] = []
    for capability in sorted(required_market_data):
        issue = _capability_literal_issue(
            capability,
            signals,
            generation_plan=generation_plan,
        )
        if issue is not None:
            issues.append(issue)

    issues.extend(
        _route_specific_issues(
            signals,
            required_market_data=required_market_data,
            generation_plan=generation_plan,
        )
    )
    if signals.wall_clock_calls:
        issues.append(
            LiteReviewIssue(
                code="lite.wall_clock_valuation_date",
                message=(
                    "Generated code uses wall-clock time via "
                    + ", ".join(f"`{call}`" for call in signals.wall_clock_calls)
                    + ". Derive valuation time from `market_state` or shared resolver outputs instead."
                ),
            )
        )

    return LiteReviewReport(issues=tuple(issues), signals=signals)


def _capability_literal_issue(
    capability: str,
    signals: LiteReviewSignals,
    *,
    generation_plan=None,
) -> LiteReviewIssue | None:
    if _capability_literal_is_subsumed_by_helper(
        capability,
        signals,
        generation_plan=generation_plan,
    ):
        return None

    suspicious_names = _SUSPICIOUS_LITERAL_NAMES.get(capability)
    if not suspicious_names:
        return None

    matching_literals = []
    for assignment in signals.literal_assignments:
        name, _, value = assignment.partition("=")
        if name in suspicious_names:
            matching_literals.append(f"`{name} = {value}`")

    if not matching_literals:
        return None

    access_prefixes = _CAPABILITY_ATTR_PREFIXES.get(capability, ())
    if access_prefixes:
        for access in signals.market_state_accesses:
            if any(access.startswith(prefix) for prefix in access_prefixes):
                return None

    route = None
    if generation_plan is not None and getattr(generation_plan, "primitive_plan", None) is not None:
        route = generation_plan.primitive_plan.route

    capability_label = capability.replace("_", " ")
    route_text = f" for route `{route}`" if route else ""
    return LiteReviewIssue(
        code=f"lite.hardcoded_{capability}",
        message=(
            f"Generated code appears to hardcode {capability_label}{route_text} via "
            + ", ".join(matching_literals)
            + " instead of reading it from `market_state`."
        ),
    )


def _capability_literal_is_subsumed_by_helper(
    capability: str,
    signals: LiteReviewSignals,
    *,
    generation_plan=None,
) -> bool:
    """Return whether a helper-backed route legitimately owns the literal."""
    if capability != "black_vol_surface":
        return False
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is None:
        return False
    helper_by_route = {
        "analytical_black76": "price_swaption_black76",
        "rate_tree_backward_induction": "price_swaption_tree",
        "monte_carlo_paths": "price_swaption_monte_carlo",
    }
    helper_name = helper_by_route.get(primitive_plan.route)
    if helper_name is None or helper_name not in signals.call_names:
        return False
    return (
        any(item.startswith("mean_reversion=") for item in signals.literal_assignments)
        and any(item.startswith("sigma=") for item in signals.literal_assignments)
    )


def _resolve_attribute_chain(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _resolve_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _numeric_literal(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_literal(node.operand)
        if inner is not None:
            return f"-{inner}"
    return None


def _route_specific_issues(
    signals: LiteReviewSignals,
    *,
    required_market_data: set[str],
    generation_plan=None,
) -> tuple[LiteReviewIssue, ...]:
    issues: list[LiteReviewIssue] = []
    route = None
    adapters: tuple[str, ...] = ()
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    if primitive_plan is not None:
        route = primitive_plan.route
        adapters = primitive_plan.adapters

    if route == "analytical_black76" and "map_spot_discount_and_vol_to_forward_black76" not in adapters:
        return tuple(issues)
    if primitive_plan is not None:
        required_helper_symbols = tuple(
            primitive.symbol
            for primitive in primitive_plan.primitives
            if primitive.required and primitive.role == "route_helper"
        )
        if required_helper_symbols:
            if any(_call_names_include_symbol(signals.call_names, symbol) for symbol in required_helper_symbols):
                return tuple(issues)
            issues.append(
                LiteReviewIssue(
                    code=f"lite.{route}_route_helper_missing",
                    message=(
                        f"The `{route}` route must delegate through the required exact helper "
                        + ", ".join(f"`{symbol}`" for symbol in required_helper_symbols)
                        + " instead of rebuilding lower-level route logic inline."
                    ),
                )
            )

    required_accesses = _route_required_accesses_for(route)
    for capability, prefixes in required_accesses.items():
        if capability not in required_market_data:
            continue
        if any(_has_market_state_access(signals, prefix) for prefix in prefixes):
            continue
        issues.append(
            LiteReviewIssue(
                code=f"lite.{route}_{capability}_access_missing",
                message=_route_access_message(route, capability),
            )
        )

    return tuple(issues)


def _call_names_include_symbol(call_names: tuple[str, ...], symbol: str) -> bool:
    """Return whether one collected call name targets ``symbol``."""
    return any(call_name == symbol or call_name.endswith(f".{symbol}") for call_name in call_names)


def _route_required_accesses_for(route: str | None) -> dict[str, tuple[str, ...]]:
    """Return required market-state access patterns for a route.

    Reads from the route registry's ``market_data_access.required`` field.
    """
    if route is None:
        return {}
    try:
        from trellis.agent.route_registry import find_route_by_id
        spec = find_route_by_id(route)
        if spec is not None and spec.market_data_access.required:
            return dict(spec.market_data_access.required)
    except Exception:
        pass
    return {}


def _has_market_state_access(signals: LiteReviewSignals, prefix: str) -> bool:
    return any(access.startswith(prefix) for access in signals.market_state_accesses)


def _route_access_message(route: str, capability: str) -> str:
    capability_label = capability.replace("_", " ")
    if route == "analytical_black76" and capability == "discount_curve":
        return (
            "The `analytical_black76` vanilla route must derive present value from "
            "`market_state.discount`; no discount-curve access was detected."
        )
    if route == "analytical_black76" and capability == "black_vol_surface":
        return (
            "The `analytical_black76` vanilla route must source volatility from "
            "`market_state.vol_surface`; no vol-surface access was detected."
        )
    if route == "analytical_garman_kohlhagen" and capability == "discount_curve":
        return (
            "The `analytical_garman_kohlhagen` route must use `market_state.discount` "
            "for domestic discounting; no discount-curve access was detected."
        )
    if route == "analytical_garman_kohlhagen" and capability == "forward_curve":
        return (
            "The `analytical_garman_kohlhagen` route must source the foreign curve from "
            "`market_state.forward_curve` or `market_state.forecast_curves`; no foreign-curve access was detected."
        )
    if route == "analytical_garman_kohlhagen" and capability == "black_vol_surface":
        return (
            "The `analytical_garman_kohlhagen` route must source volatility from "
            "`market_state.vol_surface`; no vol-surface access was detected."
        )
    if route == "analytical_garman_kohlhagen" and capability == "spot":
        return (
            "The `analytical_garman_kohlhagen` route must source FX spot from "
            "`market_state.fx_rates`, `market_state.spot`, or `market_state.underlier_spots`; no spot access was detected."
        )
    if route in {"monte_carlo_paths", "exercise_monte_carlo"} and capability == "discount_curve":
        return (
            "The selected Monte Carlo route must discount from `market_state.discount`; "
            "no discount-curve access was detected."
        )
    if route in {"monte_carlo_paths", "exercise_monte_carlo"} and capability == "black_vol_surface":
        return (
            "The selected Monte Carlo route must source volatility from "
            "`market_state.vol_surface`; no vol-surface access was detected."
        )
    if route in {"rate_tree_backward_induction", "exercise_lattice"} and capability == "discount_curve":
        return (
            "The selected rate-tree route must calibrate or discount from "
            "`market_state.discount`; no discount-curve access was detected."
        )
    if route in {"rate_tree_backward_induction", "exercise_lattice"} and capability == "black_vol_surface":
        return (
            "The selected rate-tree route requires volatility input from "
            "`market_state.vol_surface`; no vol-surface access was detected."
        )
    return (
        f"The selected route `{route}` requires `{capability_label}` from "
        "`market_state`, but no matching access was detected."
    )
