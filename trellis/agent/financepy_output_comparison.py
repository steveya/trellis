"""Pure comparison helpers for FinancePy parity outputs (QUA-861).

Factored out of ``scripts/run_financepy_benchmark.py`` so tests and other
consumers can import the comparison logic without picking up the runner
script's import-time side effects (``sys.path`` mutation, ``load_env()``,
``os.environ.setdefault``).  No I/O, no network, no environment access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Explicit allowlist of output-name tokens that should be accounted as Greeks
# for coverage reporting.  `price` and payoff-specific outputs like
# `fair_strike_variance` intentionally stay out -- the earlier heuristic
# `is_greek = name != "price"` mis-labeled variance-swap outputs as Greeks.
# Add to this set when a binding introduces a new canonical Greek name.
# Refs: QUA-861 round-1 Codex review.
GREEK_OUTPUT_NAMES: frozenset[str] = frozenset(
    {
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "vanna",
        "volga",
        "charm",
        "credit_delta",
        "credit_gamma",
        "dv01",
        "duration",
        "convexity",
        "key_rate_durations",
    }
)


def is_greek_output(name: str) -> bool:
    """Return whether *name* is one of the canonical Greek output tokens."""
    return str(name or "").strip() in GREEK_OUTPUT_NAMES


def output_value(outputs: Mapping[str, Any], key: str) -> Any:
    """Look up ``key`` in an outputs dict with a fallback into ``greeks``."""
    if key in outputs:
        return outputs.get(key)
    greeks = outputs.get("greeks") or {}
    if isinstance(greeks, Mapping):
        return greeks.get(key)
    return None


def compare_benchmark_outputs(
    *,
    task: Mapping[str, Any],
    binding: Mapping[str, Any],
    trellis_outputs: Mapping[str, Any],
    financepy_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare Trellis vs FinancePy outputs, reporting Greek coverage honestly.

    Refs: QUA-861.  The older shape silently skipped declared outputs that
    one side didn't emit, so a task could "pass" on `price` alone while
    quietly missing every Greek.  This version records
    ``missing_trellis_outputs`` / ``missing_financepy_outputs`` and
    per-side Greek coverage + Greek-only parity so scorecards can surface
    the gap.
    """
    tolerance_pct = float(((task.get("cross_validate") or {}).get("tolerance_pct") or 5.0))
    trellis_outputs = dict(trellis_outputs or {})
    financepy_outputs = dict(financepy_outputs or {})
    overlapping_outputs = tuple(
        str(name).strip()
        for name in (binding.get("overlapping_outputs") or ())
        if str(name).strip()
    )
    compared_outputs: list[str] = []
    output_deltas: dict[str, float] = {}
    failures: list[str] = []
    missing_trellis: list[str] = []
    missing_financepy: list[str] = []
    greek_failures: list[str] = []
    trellis_greek_count = 0
    financepy_greek_count = 0
    compared_greek_count = 0
    compared_greeks = False
    for output_name in overlapping_outputs:
        is_greek = is_greek_output(output_name)
        trellis_value = output_value(trellis_outputs, output_name)
        financepy_value = output_value(financepy_outputs, output_name)
        if trellis_value is not None and is_greek:
            trellis_greek_count += 1
        if financepy_value is not None and is_greek:
            financepy_greek_count += 1
        if trellis_value is None:
            missing_trellis.append(output_name)
        if financepy_value is None:
            missing_financepy.append(output_name)
        if trellis_value is None or financepy_value is None:
            continue
        compared_outputs.append(output_name)
        if is_greek:
            compared_greek_count += 1
            compared_greeks = True
        denominator = max(abs(float(financepy_value)), 1e-12)
        deviation_pct = abs(float(trellis_value) - float(financepy_value)) / denominator * 100.0
        output_deltas[output_name] = round(deviation_pct, 6)
        if deviation_pct > tolerance_pct:
            failures.append(output_name)
            if is_greek:
                greek_failures.append(output_name)

    greek_coverage = {
        "trellis_greek_count": trellis_greek_count,
        "financepy_greek_count": financepy_greek_count,
        "compared_greek_count": compared_greek_count,
    }
    if not compared_greeks:
        greek_parity = "not_applicable"
    elif greek_failures:
        greek_parity = "failed"
    else:
        greek_parity = "passed"

    if not compared_outputs:
        return {
            "status": "insufficient_overlap",
            "tolerance_pct": tolerance_pct,
            "compared_outputs": (),
            "expected_overlapping_outputs": overlapping_outputs,
            "missing_trellis_outputs": tuple(missing_trellis),
            "missing_financepy_outputs": tuple(missing_financepy),
            "greek_coverage": greek_coverage,
            "greek_parity": greek_parity,
            "greek_failures": (),
            "trellis_outputs": trellis_outputs,
            "financepy_outputs": financepy_outputs,
        }
    return {
        "status": "passed" if not failures else "failed",
        "tolerance_pct": tolerance_pct,
        "compared_outputs": tuple(compared_outputs),
        "expected_overlapping_outputs": overlapping_outputs,
        "output_deviation_pct": output_deltas,
        "missing_trellis_outputs": tuple(missing_trellis),
        "missing_financepy_outputs": tuple(missing_financepy),
        "greek_coverage": greek_coverage,
        "greek_parity": greek_parity,
        "greek_failures": tuple(greek_failures),
        "trellis_outputs": trellis_outputs,
        "financepy_outputs": financepy_outputs,
    }


__all__ = (
    "GREEK_OUTPUT_NAMES",
    "compare_benchmark_outputs",
    "is_greek_output",
    "output_value",
)
