"""Deterministic semantic track classification for the closure program."""

from __future__ import annotations

from dataclasses import dataclass


def _normalized(text: str | None) -> str:
    return str(text or "").strip().lower()


@dataclass(frozen=True)
class SemanticTrackClassification:
    track: str
    base_track: str = ""
    dynamic: bool = False
    reasons: tuple[str, ...] = ()


def classify_semantic_track(
    description: str,
    instrument_type: str | None = None,
) -> SemanticTrackClassification:
    """Classify a request into the current post-Phase-4 semantic tracks."""

    lower = _normalized(description)
    instrument = _normalized(instrument_type)
    reasons: list[str] = []

    if _looks_dynamic(lower, instrument):
        base_track = _dynamic_base_track(lower, instrument)
        reasons.append("dynamic_cues")
        if base_track:
            reasons.append(f"base:{base_track}")
        return SemanticTrackClassification(
            track="dynamic_wrapper",
            base_track=base_track,
            dynamic=True,
            reasons=tuple(reasons),
        )

    if _looks_quoted_observable(lower, instrument):
        return SemanticTrackClassification(
            track="quoted_observable",
            reasons=("quoted_snapshot_cues",),
        )

    if _looks_static_leg(lower, instrument):
        return SemanticTrackClassification(
            track="static_leg",
            reasons=("static_leg_cues",),
        )

    return SemanticTrackClassification(
        track="payoff_expression",
        reasons=("payoff_expression_default",),
    )


def _looks_dynamic(lower: str, instrument: str) -> bool:
    cues = (
        "autocall",
        "phoenix",
        "snowball",
        "tarn",
        "tarf",
        "target redemption",
        "range accrual",
        "swing option",
        "gmwb",
        "gmxb",
        "callable ",
        "issuer call",
        "puttable ",
    )
    if any(cue in lower for cue in cues):
        return True
    return instrument in {
        "callable_bond",
        "puttable_bond",
        "range_accrual",
    }


def _dynamic_base_track(lower: str, instrument: str) -> str:
    if any(cue in lower for cue in ("cms", "coupon", "bond", "swap", "basis")):
        return "static_leg"
    if any(cue in lower for cue in ("curve-spread", "curve spread", "vol-skew", "vol skew", "surface quote", "curve quote")):
        return "quoted_observable"
    if instrument in {"callable_bond", "puttable_bond", "range_accrual"}:
        return "static_leg"
    return "payoff_expression"


def _looks_quoted_observable(lower: str, instrument: str) -> bool:
    if instrument in {"quoted_observable", "curve_spread_payoff", "vol_skew_payoff"}:
        return True
    cues = (
        "curve-spread payoff",
        "curve spread payoff",
        "vol-skew payoff",
        "vol skew payoff",
        "terminal curve spread",
        "terminal vol skew",
    )
    return any(cue in lower for cue in cues)


def _looks_static_leg(lower: str, instrument: str) -> bool:
    if instrument in {"swap", "bond", "basis_swap", "interest_rate_swap"}:
        return True
    cues = (
        "interest rate swap",
        "basis swap",
        "coupon bond",
        "fixed coupon bond",
        "receive fixed",
        "pay fixed",
        "receive compounded sofr",
        "pay fed funds",
    )
    return any(cue in lower for cue in cues)


__all__ = [
    "SemanticTrackClassification",
    "classify_semantic_track",
]
