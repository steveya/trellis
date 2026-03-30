"""ParameterBindingValidator — verifies parameters are extracted from spec, not invented.

Checks:
1. Required parameters (maturity, strike, etc.) are read from the spec/contract object
2. Suspicious numeric literals that look like financial parameters
"""

from __future__ import annotations

import ast
import re

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec
from trellis.agent.semantic_validators.base import SemanticFinding

# Common parameter names and their spec-access patterns
_PARAM_ACCESS_PATTERNS = {
    "maturity": ("spec.maturity", "spec.expiry", "spec.T", "spec.time_to_maturity"),
    "strike": ("spec.strike", "spec.K", "spec.strike_price"),
    "coupon_rate": ("spec.coupon", "spec.coupon_rate", "spec.fixed_rate"),
    "notional": ("spec.notional", "spec.face_value", "spec.principal"),
}

# Suspicious literals: numbers that look like financial parameters
# but appear as bare assignments (not array sizes, loop indices, etc.)
_SUSPICIOUS_LITERAL_RE = re.compile(
    r"(?:maturity|strike|coupon|notional|face_value|principal)"
    r"\s*=\s*\d+(?:\.\d+)?(?!\s*[\*\+\-/].*(?:spec|market))",
    re.IGNORECASE,
)


class ParameterBindingValidator:
    """Validates that contract parameters are extracted from spec objects."""

    def validate(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec | None,
    ) -> tuple[SemanticFinding, ...]:
        findings: list[SemanticFinding] = []

        # 1. Check required parameter access (only when route spec provided)
        if route_spec is not None:
            required_params = route_spec.parameter_bindings.required
            if required_params:
                findings.extend(self._check_parameter_access(source, required_params))

        # 2. Check suspicious literals (always)
        findings.extend(self._check_suspicious_literals(source))

        return tuple(findings)

    def _check_parameter_access(
        self,
        source: str,
        required_params: tuple[str, ...],
    ) -> list[SemanticFinding]:
        """Verify required parameters are read from spec."""
        findings = []
        for param in required_params:
            patterns = _PARAM_ACCESS_PATTERNS.get(param, ())
            if not patterns:
                continue
            found = any(pattern in source for pattern in patterns)
            if not found:
                # Also check generic spec.{param} access
                if f"spec.{param}" not in source:
                    findings.append(SemanticFinding(
                        validator="parameter_binding",
                        severity="warning",
                        category=f"missing_{param}_from_spec",
                        message=(
                            f"Parameter '{param}' should be read from the spec object "
                            f"(e.g., {patterns[0] if patterns else f'spec.{param}'}), "
                            f"but no access found."
                        ),
                    ))
        return findings

    def _check_suspicious_literals(self, source: str) -> list[SemanticFinding]:
        """Flag numeric literals assigned to parameter-like names."""
        findings = []
        for match in _SUSPICIOUS_LITERAL_RE.finditer(source):
            line_text = match.group(0).strip()
            line_start = source[:match.start()].count("\n") + 1
            findings.append(SemanticFinding(
                validator="parameter_binding",
                severity="warning",
                category="hardcoded_parameter",
                message=(
                    f"Possible hard-coded parameter: '{line_text}'. "
                    "Contract parameters should be extracted from the spec object."
                ),
                line=line_start,
                evidence=line_text,
            ))
        return findings
