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
