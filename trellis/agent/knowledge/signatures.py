"""Failure signature matching — YAML-driven error pattern recognition.

Replaces the hardcoded _DIAGNOSTIC_PATTERNS in test_resolution.py
with structured, extensible YAML-based pattern matching.
"""

from __future__ import annotations

from trellis.agent.knowledge.schema import FailureSignature


def match_failure(
    error_message: str,
    signatures: list[FailureSignature],
) -> list[FailureSignature]:
    """Match an error message against known failure signatures.

    Returns all matching signatures, ordered by specificity
    (longer patterns first).
    """
    import re

    matched: list[FailureSignature] = []
    for sig in signatures:
        try:
            if re.search(sig.pattern, error_message, re.IGNORECASE):
                matched.append(sig)
        except re.error:
            continue

    # Sort by pattern length (longer = more specific)
    matched.sort(key=lambda s: -len(s.pattern))
    return matched


def diagnose_from_signatures(
    failures: list[str],
    signatures: list[FailureSignature],
) -> str:
    """Diagnose a list of validation failures using known signatures.

    Returns formatted text for injection into the retry prompt.
    """
    if not failures or not signatures:
        return ""

    lines: list[str] = []
    seen_categories: set[str] = set()

    for failure_msg in failures:
        matches = match_failure(failure_msg, signatures)
        for sig in matches:
            if sig.category not in seen_categories:
                seen_categories.add(sig.category)
                lines.append(
                    f"\n**Known pattern ({sig.category}, {sig.magnitude}):** "
                    f"{sig.diagnostic_hint}"
                )
                if sig.probable_causes:
                    lines.append(
                        f"  Related lessons: {', '.join(sig.probable_causes)}"
                    )

    if not lines:
        return ""

    return "\n## Matched Failure Signatures\n" + "\n".join(lines)
