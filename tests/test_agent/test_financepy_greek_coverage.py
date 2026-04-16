"""Greek coverage reporting for the FinancePy parity comparison (QUA-861).

The old `_compare_outputs` collapsed silently to `price` when Trellis did
not emit the Greeks declared in a binding's `overlapping_outputs`.  This
hid missing Trellis Greek coverage inside a "passed" comparison.  The new
reporting shape distinguishes:

* `compared_outputs`  -- outputs where both sides emitted a value
* `missing_trellis_outputs`   -- declared-overlap outputs where Trellis was silent
* `missing_financepy_outputs` -- declared-overlap outputs where FinancePy was silent
* `greek_coverage` -- per-side count of Greek outputs emitted
* `greek_parity`   -- pass/fail across compared Greeks only (price excluded)
"""

from __future__ import annotations

from trellis.agent.financepy_output_comparison import (
    compare_benchmark_outputs as _compare_outputs,
)


def _task(tolerance_pct: float = 1.0):
    return {"cross_validate": {"tolerance_pct": tolerance_pct}}


def _binding(*outputs: str):
    return {"overlapping_outputs": list(outputs)}


def test_compare_outputs_records_missing_trellis_greeks_explicitly():
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "delta", "gamma", "vega", "theta"),
        trellis_outputs={"price": 10.45},
        financepy_outputs={
            "price": 10.46,
            "delta": 0.55,
            "gamma": 0.04,
            "vega": 0.42,
            "theta": -0.03,
        },
    )
    assert summary["status"] == "passed"
    assert summary["compared_outputs"] == ("price",)
    assert set(summary["missing_trellis_outputs"]) == {"delta", "gamma", "vega", "theta"}
    assert summary["missing_financepy_outputs"] == ()
    # Greek parity is "not_applicable" when no Greek was compared on both sides.
    assert summary["greek_parity"] == "not_applicable"
    assert summary["greek_coverage"] == {
        "trellis_greek_count": 0,
        "financepy_greek_count": 4,
        "compared_greek_count": 0,
    }


def test_compare_outputs_records_missing_financepy_greeks():
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "delta"),
        trellis_outputs={"price": 10.45, "delta": 0.55},
        financepy_outputs={"price": 10.46},
    )
    assert set(summary["compared_outputs"]) == {"price"}
    assert summary["missing_financepy_outputs"] == ("delta",)
    assert summary["missing_trellis_outputs"] == ()


def test_compare_outputs_computes_greek_parity_when_both_sides_emit_greek():
    summary = _compare_outputs(
        task=_task(tolerance_pct=2.0),
        binding=_binding("price", "delta", "vega"),
        trellis_outputs={"price": 10.45, "delta": 0.55, "vega": 0.40},
        financepy_outputs={"price": 10.46, "delta": 0.556, "vega": 0.42},
    )
    assert set(summary["compared_outputs"]) == {"price", "delta", "vega"}
    # delta deviation ~1.08%, vega deviation ~4.76%.  Under 2% tolerance vega fails.
    assert summary["greek_parity"] == "failed"
    assert "vega" in summary.get("greek_failures", ())
    assert summary["greek_coverage"] == {
        "trellis_greek_count": 2,
        "financepy_greek_count": 2,
        "compared_greek_count": 2,
    }


def test_compare_outputs_greek_parity_passes_when_all_compared_greeks_within_tolerance():
    summary = _compare_outputs(
        task=_task(tolerance_pct=5.0),
        binding=_binding("price", "delta"),
        trellis_outputs={"price": 10.45, "delta": 0.550},
        financepy_outputs={"price": 10.46, "delta": 0.556},
    )
    assert summary["greek_parity"] == "passed"
    assert summary["greek_coverage"]["compared_greek_count"] == 1


def test_compare_outputs_preserves_insufficient_overlap_shape():
    """When neither side emits anything in the declared overlap the status
    stays `insufficient_overlap` and missing-output lists describe what was
    expected but absent from each side."""
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "delta"),
        trellis_outputs={},
        financepy_outputs={},
    )
    assert summary["status"] == "insufficient_overlap"
    assert set(summary["missing_trellis_outputs"]) == {"price", "delta"}
    assert set(summary["missing_financepy_outputs"]) == {"price", "delta"}
    assert summary["greek_coverage"] == {
        "trellis_greek_count": 0,
        "financepy_greek_count": 0,
        "compared_greek_count": 0,
    }


def test_compare_outputs_treats_price_as_non_greek():
    """`price` is not a Greek; coverage counters must exclude it."""
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "delta"),
        trellis_outputs={"price": 10.45, "delta": 0.55},
        financepy_outputs={"price": 10.46, "delta": 0.556},
    )
    assert summary["greek_coverage"] == {
        "trellis_greek_count": 1,
        "financepy_greek_count": 1,
        "compared_greek_count": 1,
    }


def test_compare_outputs_does_not_count_payoff_specific_outputs_as_greeks():
    """`fair_strike_variance` is a variance-swap payoff output, not a Greek.

    The earlier heuristic `is_greek = output_name != "price"` mis-classified
    it as a Greek and falsely inflated Greek coverage counts.  The explicit
    allowlist in `GREEK_OUTPUT_NAMES` keeps accounting honest.
    (PR #593 round 1 Codex review.)
    """
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "fair_strike_variance"),
        trellis_outputs={"price": 12.3, "fair_strike_variance": 0.048},
        financepy_outputs={"price": 12.31, "fair_strike_variance": 0.048},
    )
    assert set(summary["compared_outputs"]) == {"price", "fair_strike_variance"}
    assert summary["greek_coverage"] == {
        "trellis_greek_count": 0,
        "financepy_greek_count": 0,
        "compared_greek_count": 0,
    }
    assert summary["greek_parity"] == "not_applicable"


def test_compare_outputs_insufficient_overlap_shape_includes_greek_failures_tuple():
    """Shape consistency: `greek_failures` is present in every return path
    so downstream consumers never branch on key presence.
    (PR #593 round 1 Copilot review.)
    """
    summary = _compare_outputs(
        task=_task(),
        binding=_binding("price", "delta"),
        trellis_outputs={},
        financepy_outputs={},
    )
    assert summary["status"] == "insufficient_overlap"
    assert "greek_failures" in summary
    assert summary["greek_failures"] == ()


def test_comparison_library_has_no_import_side_effects():
    """Importing `trellis.agent.financepy_output_comparison` must not mutate
    `sys.path`, invoke `load_env()`, or touch the environment.
    (PR #593 round 1 Copilot review.)
    """
    import importlib
    import sys

    original_path = list(sys.path)
    # Force a re-import to exercise the import-time codepath.
    module_name = "trellis.agent.financepy_output_comparison"
    if module_name in sys.modules:
        del sys.modules[module_name]
    importlib.import_module(module_name)
    assert sys.path == original_path


def test_compare_outputs_preserves_explicit_zero_tolerance():
    """An explicit `tolerance_pct: 0.0` on the task is strict-parity, not
    "fall back to the 5% default".  The `... or 5.0` idiom would silently
    upgrade 0 to 5.  (PR #593 round 3 Copilot review.)"""
    task = {"cross_validate": {"tolerance_pct": 0.0}}
    summary = _compare_outputs(
        task=task,
        binding=_binding("price"),
        trellis_outputs={"price": 10.0},
        financepy_outputs={"price": 10.01},
    )
    assert summary["tolerance_pct"] == 0.0
    # 0.1% deviation exceeds a 0% tolerance, so the comparison fails.
    assert summary["status"] == "failed"
