"""Validation findings and reports — structured output from model validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ValidationFinding:
    """A single issue found during model validation.

    Structured like a bank Model Risk Management (MRM) review: each finding
    has a severity level, category, supporting evidence, and a suggested fix.
    """

    id: str                         # "MV-001"
    severity: str                   # "critical", "high", "medium", "low"
    category: str                   # "conceptual", "implementation", "calibration",
                                    # "sensitivity", "benchmark", "limitation"
    description: str                # what's wrong
    evidence: str                   # how we know (test output, comparison, etc.)
    remediation: str                # what to fix
    test_code: str | None = None    # executable assertion to verify the fix


@dataclass
class ValidationReport:
    """Complete model validation report."""

    instrument: str
    method: str
    findings: list[ValidationFinding] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    approved: bool = False

    @property
    def critical_findings(self) -> list[ValidationFinding]:
        """Return the subset of findings marked ``critical``."""
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def high_findings(self) -> list[ValidationFinding]:
        """Return the subset of findings marked ``high``."""
        return [f for f in self.findings if f.severity == "high"]

    @property
    def has_blockers(self) -> bool:
        """True if there are critical or high findings."""
        return len(self.critical_findings) + len(self.high_findings) > 0

    def summary(self) -> str:
        """Return a compact human-readable status line summarizing severities."""
        counts = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        status = "BLOCKED" if self.has_blockers else "APPROVED"
        return f"[{status}] {', '.join(parts) or 'no findings'}"

    def format_for_builder(self) -> str:
        """Format findings as feedback for the builder agent."""
        if not self.has_blockers:
            return ""
        lines = ["## MODEL VALIDATION FINDINGS (must be remediated)\n"]
        for f in self.findings:
            if f.severity in ("critical", "high"):
                lines.append(f"### [{f.severity.upper()}] {f.id}: {f.description}")
                lines.append(f"Evidence: {f.evidence}")
                lines.append(f"Remediation: {f.remediation}")
                lines.append("")
        return "\n".join(lines)
