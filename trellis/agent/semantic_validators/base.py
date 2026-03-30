"""Base protocol and dataclasses for semantic validators.

Semantic validators perform deterministic AST analysis on generated code
to verify it satisfies the route contract — correct market data access,
parameter binding, and algorithm usage.  They sit between the existing
``validate_semantics()`` check and ``lite_review()`` in the executor
pipeline, sharing gate 3's retry slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.route_registry import RouteSpec


@dataclass(frozen=True)
class SemanticFinding:
    """A single finding from a semantic validator."""

    validator: str       # "market_data", "parameter_binding", "algorithm_contract"
    severity: str        # "error", "warning"
    category: str        # e.g., "missing_discount_access", "hardcoded_maturity"
    message: str
    line: int | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class SemanticValidationReport:
    """Aggregated findings from all semantic validators."""

    findings: tuple[SemanticFinding, ...]
    mode: str  # "warning" or "blocking"

    @property
    def ok(self) -> bool:
        """True if no blocking errors found (warnings always pass)."""
        if self.mode == "warning":
            return True
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self) -> tuple[SemanticFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[SemanticFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")


class SemanticValidator(Protocol):
    """Protocol for pluggable semantic validators."""

    def validate(
        self,
        source: str,
        plan: GenerationPlan,
        route_spec: RouteSpec | None,
    ) -> tuple[SemanticFinding, ...]:
        ...
