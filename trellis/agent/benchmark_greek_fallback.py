"""Bump-and-reprice Greek fallback for benchmark parity (QUA-863).

When a benchmark binding declares Greeks in ``overlapping_outputs`` but the
Trellis payoff doesn't expose them natively, this fallback invokes the
analytical measures in :mod:`trellis.analytics.measures` to compute them by
finite-difference repricing on a cloned market state.

The fallback is policy-driven from the binding metadata; no per-task or
per-instrument code lives here.  Each Greek measure declares its requirements
(e.g. spot bindings, vol surface) and any unsupported bump is skipped with an
explicit reason recorded in the returned record instead of raising.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GreekFallbackReport:
    """Result of one bump-and-reprice fallback pass.

    ``greeks`` maps Greek name → float for values we successfully computed.
    ``skipped`` maps Greek name → reason string for Greeks we declined to
    compute (e.g. requirement unmet, unsupported binding, exception raised).
    ``policy`` echoes the binding-level ``greek_fallback`` configuration so
    scorecards can record which policy generated which values.
    """

    greeks: dict[str, float]
    skipped: dict[str, str]
    policy: str

    def as_record(self) -> dict[str, Any]:
        return {
            "greeks": dict(self.greeks),
            "skipped": dict(self.skipped),
            "policy": self.policy,
        }


# Canonical Greek measure factories.  Each entry's value is a zero-arg
# callable returning a configured measure instance.  Kept here rather than in
# `measures.py` so adding a new fallback Greek doesn't require touching the
# analytics package.
def _default_measure_factories():
    from trellis.analytics.measures import Delta, Gamma, Theta, Vega

    return {
        "delta": Delta,
        "gamma": Gamma,
        "vega": Vega,
        "theta": Theta,
    }


def _binding_fallback_policy(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the binding's ``greek_fallback`` policy block, or an empty dict."""
    if not isinstance(binding, Mapping):
        return {}
    raw = binding.get("greek_fallback")
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def _already_emitted_names(already_emitted: Mapping[str, Any] | None) -> set[str]:
    """Collect native output names from top-level keys AND nested ``greeks``.

    ``extract_trellis_benchmark_outputs`` preserves cold-run native Greeks
    either at the top level (when the payoff's ``benchmark_outputs``
    returns ``{"delta": ..., "gamma": ...}``) or under a ``greeks`` key
    (when they come from ``summary.greeks`` / ``comparison.greeks`` /
    ``result.greeks``).  We treat both shapes as authoritative so the
    warm bump-and-reprice never silently overrides a native Greek just
    because it happened to live in the nested dict.  (PR #594 Codex P2
    round 1.)
    """
    if not already_emitted:
        return set()
    names: set[str] = set()
    for key, value in already_emitted.items():
        name = str(key).strip()
        if not name:
            continue
        if name == "greeks":
            if isinstance(value, Mapping):
                for nested_key in value.keys():
                    nested_name = str(nested_key).strip()
                    if nested_name:
                        names.add(nested_name)
            # Non-mapping ``greeks`` is treated as a malformed payload:
            # ignore silently and keep only the other top-level keys.
            continue
        names.add(name)
    return names


def _requested_greeks(
    binding: Mapping[str, Any],
    already_emitted: set[str],
) -> list[str]:
    """Return the declared Greeks the Trellis side did NOT already emit.

    The intersection of ``overlapping_outputs`` (minus ``price``) and the
    absent set is what the fallback should try to fill in.
    """
    overlapping = [
        str(name).strip()
        for name in (binding.get("overlapping_outputs") or ())
        if str(name).strip()
    ]
    return [
        name
        for name in overlapping
        if name != "price" and name not in already_emitted
    ]


def compute_bump_and_reprice_greeks(
    payoff,
    market_state,
    *,
    binding: Mapping[str, Any],
    already_emitted: Mapping[str, Any] | None = None,
    measure_factories: Mapping[str, Any] | None = None,
) -> GreekFallbackReport:
    """Invoke finite-difference Greek measures for any declared Greek the
    Trellis side didn't already emit.

    Parameters
    ----------
    payoff
        The built payoff with an ``evaluate(market_state)`` method.
    market_state
        The (benchmark-aligned) market state to price against.
    binding
        The binding metadata dict.  Controls which Greeks are declared
        (via ``overlapping_outputs``) and which policy applies
        (via ``greek_fallback``).
    already_emitted
        Outputs the payoff already produced natively.  Those Greeks are
        not recomputed; the fallback only fills gaps.
    measure_factories
        Optional override mapping Greek name → zero-arg factory.  Tests
        use this to inject stubs; production callers can leave it unset
        to pick up the canonical Delta/Gamma/Vega/Theta from
        ``trellis.analytics.measures``.
    """
    policy = _binding_fallback_policy(binding)
    policy_kind = str(policy.get("kind") or "").strip().lower()
    if policy_kind != "bump_and_reprice":
        return GreekFallbackReport(greeks={}, skipped={}, policy=policy_kind or "none")

    factories = dict(measure_factories or _default_measure_factories())
    already = _already_emitted_names(already_emitted)
    requested = _requested_greeks(binding, already)

    policy_overrides = policy.get("measures") or {}
    if not isinstance(policy_overrides, Mapping):
        policy_overrides = {}

    greeks: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in requested:
        factory = factories.get(name)
        if factory is None:
            skipped[name] = f"no bump-and-reprice measure registered for {name!r}"
            continue
        overrides = policy_overrides.get(name) or {}
        if not isinstance(overrides, Mapping):
            overrides = {}
        try:
            measure = factory(**overrides) if overrides else factory()
        except TypeError as exc:
            skipped[name] = f"measure constructor rejected policy overrides: {exc}"
            continue
        try:
            value = measure.compute(payoff, market_state)
        except Exception as exc:  # noqa: BLE001 -- want any bump failure surfaced
            skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            greeks[name] = float(value)
        except (TypeError, ValueError) as exc:
            skipped[name] = f"non-scalar measure result: {exc}"

    return GreekFallbackReport(greeks=greeks, skipped=skipped, policy=policy_kind)


__all__ = (
    "GreekFallbackReport",
    "compute_bump_and_reprice_greeks",
)
