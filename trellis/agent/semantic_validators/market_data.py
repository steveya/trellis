"""MarketDataValidator — verifies generated code accesses market data correctly.

Checks:
1. Required market-state attributes per route spec are accessed in the code
2. No hard-coded market data literals (rates, volatilities, discount factors)
3. Correct access patterns (callable, not raw attribute)
"""

from __future__ import annotations

import ast
import re

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec
from trellis.agent.semantic_validators.base import SemanticFinding


# Patterns that suggest hard-coded market data
_SUSPICIOUS_ASSIGNMENTS = re.compile(
    r"^\s*(?:r|rate|vol|sigma|discount_rate|risk_free)\s*=\s*(?:0\.\d+|\d+\.\d+)",
    re.MULTILINE,
)


class MarketDataValidator:
    """Validates market-state access in generated code."""

    def validate(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec | None,
    ) -> tuple[SemanticFinding, ...]:
        findings: list[SemanticFinding] = []

        # 1. Check required market-state accesses (only when route spec provided)
        if route_spec is not None:
            required = route_spec.market_data_access.required
            if required:
                findings.extend(self._check_required_accesses(source, required, route_spec.id))

        # 2. Check for hard-coded market data (always)
        findings.extend(self._check_hardcoded_market_data(source))
        findings.extend(self._check_fx_rate_scalar_extraction(source))
        findings.extend(self._check_heston_black_vol_surface_mismatch(source, plan, route_spec))

        return tuple(findings)

    def _check_required_accesses(
        self,
        source: str,
        required: dict[str, tuple[str, ...]],
        route_id: str,
    ) -> list[SemanticFinding]:
        """Verify that required market-state attributes are accessed."""
        findings = []
        for capability, access_patterns in required.items():
            found = any(pattern in source for pattern in access_patterns)
            if not found:
                findings.append(SemanticFinding(
                    validator="market_data",
                    severity="error",
                    category=f"missing_{capability}_access",
                    message=(
                        f"Route '{route_id}' requires {capability} access via "
                        f"{' or '.join(access_patterns)}, but none found in generated code."
                    ),
                ))
        return findings

    def _check_hardcoded_market_data(self, source: str) -> list[SemanticFinding]:
        """Flag suspicious hard-coded market data literals."""
        findings = []
        for match in _SUSPICIOUS_ASSIGNMENTS.finditer(source):
            line_text = match.group(0).strip()
            # Compute line number
            line_start = source[:match.start()].count("\n") + 1
            findings.append(SemanticFinding(
                validator="market_data",
                severity="warning",
                category="hardcoded_market_data",
                message=(
                    f"Possible hard-coded market data: '{line_text}'. "
                    "Market inputs should be read from market_state."
                ),
                line=line_start,
                evidence=line_text,
            ))
        return findings

    def _check_fx_rate_scalar_extraction(self, source: str) -> list[SemanticFinding]:
        """Require explicit scalar extraction before arithmetic on FXRate quotes."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        detector = _FXRateArithmeticDetector()
        detector.visit(tree)
        if not detector.offending_nodes:
            return []

        first = detector.offending_nodes[0]
        line = getattr(first, "lineno", None)
        evidence = ast.get_source_segment(source, first) or "market_state.fx_rates[...]"
        return [SemanticFinding(
            validator="market_data",
            severity="error",
            category="fx_rate_scalar_extraction_missing",
            message=(
                "`market_state.fx_rates[...]` returns an FXRate wrapper. Extract "
                "`.spot` before arithmetic or process seeding instead of multiplying "
                "or otherwise treating the wrapper as a scalar."
            ),
            line=line,
            evidence=evidence,
        )]

    def _check_heston_black_vol_surface_mismatch(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec | None,
    ) -> list[SemanticFinding]:
        """Reject Heston stochastic-volatility code that binds Black vol surfaces."""
        if not _is_heston_model_parameter_context(plan, route_spec):
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        detector = _MarketStateVolSurfaceDetector(source)
        detector.visit(tree)
        if detector.first_node is None:
            return []

        evidence = detector.first_evidence or "market_state.vol_surface"
        return [SemanticFinding(
            validator="market_data",
            severity="error",
            category="heston_black_vol_surface_mismatch",
            message=(
                "Heston stochastic-volatility routes must bind explicit "
                "model_parameters and must not price from market_state.vol_surface "
                "or a Black-vol adapter. Use the Heston model-parameter route, "
                "or fail closed if that route is unavailable."
            ),
            line=getattr(detector.first_node, "lineno", None),
            evidence=evidence,
        )]


class _FXRateArithmeticDetector(ast.NodeVisitor):
    """Detect arithmetic performed on raw FXRate wrappers."""

    def __init__(self) -> None:
        self.fx_quote_aliases: set[str] = set()
        self.offending_nodes: list[ast.AST] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_raw_fx_quote(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.fx_quote_aliases.add(target.id)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if _contains_raw_fx_quote(node.left, self.fx_quote_aliases) or _contains_raw_fx_quote(
            node.right, self.fx_quote_aliases,
        ):
            self.offending_nodes.append(node)
        self.generic_visit(node)


def _contains_raw_fx_quote(node: ast.AST, aliases: set[str]) -> bool:
    """Whether the AST node still represents a raw FXRate wrapper."""
    if _is_raw_fx_quote(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in aliases
    return False


def _is_raw_fx_quote(node: ast.AST) -> bool:
    """Whether the AST node directly references market_state.fx_rates[...] without .spot."""
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "fx_rates"
        and isinstance(value.value, ast.Name)
        and value.value.id == "market_state"
    )


class _MarketStateVolSurfaceDetector(ast.NodeVisitor):
    """Detect executable access to market_state.vol_surface."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.first_node: ast.AST | None = None
        self.first_evidence: str | None = None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.first_node is None and _is_market_state_vol_surface(node):
            self.first_node = node
            self.first_evidence = ast.get_source_segment(self.source, node)
        self.generic_visit(node)


def _is_market_state_vol_surface(node: ast.AST) -> bool:
    """Whether ``node`` is direct access to ``market_state.vol_surface``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "vol_surface"
        and isinstance(node.value, ast.Name)
        and node.value.id == "market_state"
    )


def _is_heston_model_parameter_context(
    plan: GenerationPlan,
    route_spec: RouteSpec | None,
) -> bool:
    """Return whether generated code is for a Heston explicit-parameter route."""
    plan_tokens = {
        str(getattr(plan, attr, "") or "").strip().lower()
        for attr in (
            "instrument_type",
            "semantic_requested_instrument_type",
            "semantic_instrument_class",
            "semantic_compatibility_bridge_status",
            "backend_binding_id",
            "validation_bundle_id",
        )
    }
    primitive_plan = getattr(plan, "primitive_plan", None)
    if primitive_plan is not None:
        plan_tokens.update(
            str(getattr(primitive_plan, attr, "") or "").strip().lower()
            for attr in ("route", "route_family", "engine_family")
        )
        for primitive in getattr(primitive_plan, "primitives", ()) or ():
            plan_tokens.add(str(getattr(primitive, "module", "") or "").strip().lower())
            plan_tokens.add(str(getattr(primitive, "symbol", "") or "").strip().lower())
    if route_spec is not None:
        plan_tokens.update(
            str(getattr(route_spec, attr, "") or "").strip().lower()
            for attr in ("id", "route_family", "engine_family")
        )
        for primitive in getattr(route_spec, "primitives", ()) or ():
            plan_tokens.add(str(getattr(primitive, "module", "") or "").strip().lower())
            plan_tokens.add(str(getattr(primitive, "symbol", "") or "").strip().lower())

    heston_context = any("heston" in token or "stochastic_vol" in token for token in plan_tokens)
    calibration_context = any("calibration" in token or "calibrate" in token for token in plan_tokens)
    return heston_context and not calibration_context
