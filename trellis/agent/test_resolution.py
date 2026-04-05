"""Test resolution workflow — diagnose failures and record lessons learned.

This module is invoked whenever a test fails during the build/validate cycle.
It:
1. Diagnoses the failure (categorizes, identifies root cause)
2. Suggests a fix
3. After the fix, records the lesson in the canonical knowledge store so the
   agent can reuse it on future builds

The workflow:
    fail → diagnose → fix → verify → record canonical lesson → continue

Usage by agents:
    from trellis.agent.test_resolution import (
        diagnose_failure,
        record_lesson,
        format_diagnosis_prompt,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestFailure:
    """Structured representation of a test failure."""
    test_name: str
    test_file: str
    error_type: str       # e.g., "AssertionError", "ValueError", "RuntimeWarning"
    error_message: str
    expected: str | None   # what was expected
    actual: str | None     # what was produced
    traceback: str


@dataclass(frozen=True)
class Diagnosis:
    """Root cause analysis of a test failure."""
    category: str         # calibration, volatility, backward_induction, etc.
    root_cause: str       # what went wrong
    magnitude: str        # "catastrophic" (>10x), "significant" (>10%), "minor" (<10%)
    suggested_fix: str
    related_lessons: list[str]  # canonical lesson titles related to the failure


@dataclass(frozen=True)
class Lesson:
    """A lesson learned from resolving a test failure."""
    category: str
    title: str
    mistake: str
    why: str
    detect: str
    fix: str


# ---------------------------------------------------------------------------
# Diagnosis heuristics
# ---------------------------------------------------------------------------

# Pattern matchers: (condition on failure) → suspected category
_DIAGNOSTIC_PATTERNS = [
    # Catastrophic price errors
    {
        "condition": lambda f: _magnitude(f) == "catastrophic",
        "suspects": [
            ("volatility", "Vol unit mismatch — check if Black vol was passed where HW sigma expected"),
            ("calibration", "Tree not calibrated — check if discount_curve was passed"),
            ("monte_carlo", "CF convention mismatch — check if log(S_T) vs log(S_T/S0)"),
        ],
    },
    # Bond priced below terminal discounted value
    {
        "condition": lambda f: "bond" in f.test_name.lower() and _actual_float(f) and _actual_float(f) < 70,
        "suspects": [
            ("backward_induction", "Missing intermediate cashflows (coupons) in tree rollback"),
        ],
    },
    # Callable > straight
    {
        "condition": lambda f: "callable" in f.test_name.lower() and ">" in f.error_message,
        "suspects": [
            ("backward_induction", "Wrong exercise function — callable needs min, not max"),
        ],
    },
    # Gamma/second derivative way off
    {
        "condition": lambda f: "gamma" in f.test_name.lower(),
        "suspects": [
            ("finite_differences", "dS too small for gamma — use dS=2-5 and 500+ steps"),
            ("finite_differences", "Even/odd oscillation — average n and n+1 step trees"),
        ],
    },
    # NaN or Inf
    {
        "condition": lambda f: "nan" in f.error_message.lower() or "inf" in f.error_message.lower(),
        "suspects": [
            ("finite_differences", "Numerical overflow in PDE solver or Thomas algorithm"),
            ("monte_carlo", "COS truncation domain too wide or not centered"),
        ],
    },
    # MC price too high (high-vol bias)
    {
        "condition": lambda f: "lsm" in f.test_name.lower() or "longstaff" in f.test_name.lower(),
        "suspects": [
            ("monte_carlo", "LSM polynomial basis inadequate at high vol — try Laguerre"),
        ],
    },
]


def _magnitude(f: TestFailure) -> str:
    """Estimate the magnitude of the error."""
    actual = _actual_float(f)
    expected = _expected_float(f)
    if actual is None or expected is None or expected == 0:
        return "unknown"
    ratio = abs(actual / expected)
    if ratio > 10 or ratio < 0.1:
        return "catastrophic"
    elif abs(actual - expected) / abs(expected) > 0.10:
        return "significant"
    else:
        return "minor"


def _actual_float(f: TestFailure) -> float | None:
    """Extract a numeric actual value from a test failure when possible."""
    try:
        # Try to extract from "Obtained: 123.456"
        for part in f.error_message.split("\n"):
            if "Obtained:" in part:
                return float(part.split("Obtained:")[-1].strip())
        if f.actual:
            return float(f.actual)
    except (ValueError, TypeError):
        pass
    return None


def _expected_float(f: TestFailure) -> float | None:
    """Extract a numeric expected value from a test failure when possible."""
    try:
        for part in f.error_message.split("\n"):
            if "Expected:" in part:
                val = part.split("Expected:")[-1].strip().split("±")[0].strip()
                return float(val)
        if f.expected:
            return float(f.expected)
    except (ValueError, TypeError):
        pass
    return None


def _matched_failure_signatures(failure: TestFailure):
    """Return canonical failure signatures relevant to this failure."""
    try:
        from trellis.agent.knowledge import get_store
        from trellis.agent.knowledge.signatures import match_failure

        search_text = "\n".join(
            part
            for part in (
                failure.test_name,
                failure.error_message,
                failure.expected,
                failure.actual,
            )
            if part
        )
        return match_failure(search_text, get_store()._failure_signatures)
    except Exception:
        return []


def _related_lesson_titles(matched_signatures) -> list[str]:
    """Resolve canonical lesson titles from matched failure signatures."""
    try:
        from trellis.agent.knowledge import get_store

        store = get_store()
        titles: list[str] = []
        for sig in matched_signatures:
            for lesson_id in sig.probable_causes:
                lesson = store._load_lesson(lesson_id)
                if lesson is None or lesson.title in titles:
                    continue
                titles.append(lesson.title)
        return titles
    except Exception:
        return []


def diagnose_failure(failure: TestFailure) -> Diagnosis:
    """Run heuristic diagnosis on a test failure."""
    matched_signatures = _matched_failure_signatures(failure)
    related_lessons = _related_lesson_titles(matched_signatures)
    suspects = []
    for pattern in _DIAGNOSTIC_PATTERNS:
        try:
            if pattern["condition"](failure):
                suspects.extend(pattern["suspects"])
        except Exception:
            continue

    if suspects:
        category, root_cause = suspects[0]
        magnitude = _magnitude(failure)
    elif matched_signatures:
        sig = matched_signatures[0]
        category = sig.category
        root_cause = sig.diagnostic_hint or "Matched known failure signature."
        magnitude = sig.magnitude
    else:
        category = "unknown"
        root_cause = "No matching diagnostic pattern. Manual investigation needed."
        magnitude = _magnitude(failure)

    return Diagnosis(
        category=category,
        root_cause=root_cause,
        magnitude=magnitude,
        suggested_fix=root_cause,
        related_lessons=related_lessons,
    )


def record_lesson(
    lesson: Lesson,
    *,
    severity: str = "high",
    method: str | None = None,
    instrument: str | None = None,
    features: list[str] | None = None,
    error_signature: str | None = None,
    confidence: float = 0.5,
    validation: str = "",
    source_trace: str | None = None,
    version: str = "",
) -> str | None:
    """Capture a lesson in the canonical knowledge store."""
    from trellis.agent.knowledge.promotion import (
        build_lesson_payload,
        capture_lesson,
        validate_lesson_payload,
    )

    symptom = str(lesson.mistake).strip() or str(lesson.detect).strip()
    payload = build_lesson_payload(
        category=lesson.category,
        title=lesson.title,
        severity=severity,
        symptom=symptom,
        root_cause=lesson.why,
        fix=lesson.fix,
        validation=validation,
        method=method,
        instrument=instrument,
        features=features,
        error_signature=error_signature,
        confidence=confidence,
        source_trace=source_trace,
        version=version,
    )
    report = validate_lesson_payload(payload)
    if not report.valid:
        raise ValueError("Lesson payload failed validation: " + "; ".join(report.errors))

    return capture_lesson(
        category=lesson.category,
        title=lesson.title,
        severity=severity,
        symptom=symptom,
        root_cause=lesson.why,
        fix=lesson.fix,
        validation=validation,
        method=method,
        instrument=instrument,
        features=features,
        error_signature=error_signature,
        confidence=confidence,
        source_trace=source_trace,
        version=version,
    )


# ---------------------------------------------------------------------------
# Prompt formatting for LLM-based diagnosis
# ---------------------------------------------------------------------------

def format_diagnosis_prompt(failure: TestFailure) -> str:
    """Format a test failure for LLM-based diagnosis.

    Used when heuristic diagnosis is insufficient and we need the
    LLM validator to analyze the failure.
    """
    heuristic = diagnose_failure(failure)

    return f"""## Test Failure to Diagnose

**Test:** `{failure.test_name}` in `{failure.test_file}`
**Error type:** {failure.error_type}
**Error message:**
```
{failure.error_message}
```

**Heuristic diagnosis:**
- Category: {heuristic.category}
- Magnitude: {heuristic.magnitude}
- Suspected root cause: {heuristic.root_cause}
- Related lessons: {', '.join(heuristic.related_lessons) or 'None'}

**Traceback:**
```
{failure.traceback[:2000]}
```

## Your task
1. Confirm or revise the heuristic diagnosis
2. Identify the exact root cause
3. Suggest a specific fix (code change)
4. Write a one-paragraph lesson learned for the canonical knowledge store

Format your response as:
- **Root cause:** ...
- **Fix:** ...
- **Lesson:** (category, title, mistake, why, detect, fix)
"""
