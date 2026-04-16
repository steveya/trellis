"""Tests for the benchmark Greek bump-and-reprice fallback (QUA-863)."""

from __future__ import annotations

import math

import pytest

from trellis.agent.benchmark_greek_fallback import (
    GreekFallbackReport,
    compute_bump_and_reprice_greeks,
)


class _StubDelta:
    def __init__(self, value: float = 0.55):
        self._value = value

    def compute(self, payoff, ms):
        return self._value


class _StubVega:
    def __init__(self, value: float = 0.40, **_kwargs):
        self._value = value

    def compute(self, payoff, ms):
        return self._value


class _RaisingGamma:
    def compute(self, payoff, ms):
        raise RuntimeError("no spot binding available")


def _binding(*outputs, policy_kind: str | None = "bump_and_reprice"):
    binding: dict = {"overlapping_outputs": list(outputs)}
    if policy_kind:
        binding["greek_fallback"] = {"kind": policy_kind}
    return binding


def test_fallback_returns_empty_when_policy_is_not_bump_and_reprice():
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "delta", policy_kind=None),
        measure_factories={"delta": _StubDelta},
    )
    assert report.greeks == {}
    assert report.skipped == {}
    assert report.policy == "none"


def test_fallback_fills_greek_declared_in_overlap_and_not_already_emitted():
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "delta"),
        already_emitted={"price": 10.45},
        measure_factories={"delta": _StubDelta},
    )
    assert report.greeks == {"delta": 0.55}
    assert report.skipped == {}
    assert report.policy == "bump_and_reprice"


def test_fallback_skips_greeks_already_emitted_natively():
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "delta"),
        already_emitted={"price": 10.45, "delta": 0.60},
        measure_factories={"delta": _StubDelta},
    )
    assert report.greeks == {}
    assert report.skipped == {}


def test_fallback_skips_greeks_without_a_registered_measure():
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "rho"),
        measure_factories={"delta": _StubDelta},  # no `rho`
    )
    assert report.greeks == {}
    assert "rho" in report.skipped
    assert "no bump-and-reprice measure" in report.skipped["rho"]


def test_fallback_records_measure_exception_without_raising():
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "gamma"),
        measure_factories={"gamma": _RaisingGamma},
    )
    assert report.greeks == {}
    assert "gamma" in report.skipped
    assert "no spot binding available" in report.skipped["gamma"]


def test_fallback_passes_binding_measure_overrides_to_constructor():
    captured: dict = {}

    class _CapturingVega(_StubVega):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    binding = {
        "overlapping_outputs": ["price", "vega"],
        "greek_fallback": {
            "kind": "bump_and_reprice",
            "measures": {"vega": {"bump_pct": 0.25}},
        },
    }
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=binding,
        measure_factories={"vega": _CapturingVega},
    )
    assert report.greeks == {"vega": 0.40}
    assert captured == {"bump_pct": 0.25}


def test_fallback_rejects_invalid_override_shape_without_blowing_up():
    class _BumpOnlyVega:
        def __init__(self, bump_pct: float = 1.0):
            self.bump_pct = bump_pct

        def compute(self, payoff, ms):
            return 0.41

    binding = {
        "overlapping_outputs": ["price", "vega"],
        "greek_fallback": {
            "kind": "bump_and_reprice",
            "measures": {"vega": {"unknown_kwarg": 42}},
        },
    }
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=binding,
        measure_factories={"vega": _BumpOnlyVega},
    )
    assert report.greeks == {}
    assert "vega" in report.skipped
    assert "measure constructor rejected" in report.skipped["vega"]


def test_fallback_treats_nested_native_greeks_as_already_emitted():
    """Native Greeks preserved under ``greeks[...]`` must skip the warm bump.

    ``extract_trellis_benchmark_outputs`` carries cold-run native Greeks
    inside a nested ``greeks`` mapping for sources that report them that
    way (``summary.greeks``, ``comparison.greeks``, ``result.greeks``).
    Treating those as missing would let the bump fallback overwrite
    native values when its output gets merged as a top-level key.  (PR
    #594 Codex P2 round 1.)
    """
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "delta", "vega"),
        already_emitted={
            "price": 10.45,
            "greeks": {"delta": 0.60, "vega": 0.41},
        },
        measure_factories={
            "delta": _StubDelta,
            "vega": _StubVega,
        },
    )
    assert report.greeks == {}
    assert report.skipped == {}


def test_fallback_nested_mapping_with_non_mapping_greeks_is_robust():
    """A non-Mapping value under ``greeks`` must not blow up the fallback.

    Defensive shape check so a malformed cold-run payload degrades to
    "top-level keys only" detection rather than raising, keeping the
    fallback available for any other correctly-typed declared Greeks.
    """
    report = compute_bump_and_reprice_greeks(
        payoff=object(),
        market_state=object(),
        binding=_binding("price", "delta"),
        already_emitted={
            "price": 10.45,
            "greeks": "not-a-mapping",
        },
        measure_factories={"delta": _StubDelta},
    )
    assert report.greeks == {"delta": 0.55}
    assert report.skipped == {}


def test_fallback_report_serializes_to_record():
    report = GreekFallbackReport(
        greeks={"delta": 0.55, "vega": 0.4},
        skipped={"gamma": "raised"},
        policy="bump_and_reprice",
    )
    record = report.as_record()
    assert record == {
        "greeks": {"delta": 0.55, "vega": 0.4},
        "skipped": {"gamma": "raised"},
        "policy": "bump_and_reprice",
    }
