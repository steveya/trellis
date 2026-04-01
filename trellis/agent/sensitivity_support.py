"""Deterministic sensitivity-support metadata for pricing methods.

This layer keeps route-selection decisions honest when requests ask for Greeks
or curve/vol sensitivities. It does not claim native Greeks unless the current
runtime actually exposes them through the supported analytics surface.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from trellis.agent.knowledge.methods import normalize_method


_SENSITIVITY_MEASURE_ALIASES = {
    "krd": "key_rate_durations",
}

_SENSITIVITY_MEASURES = {
    "dv01",
    "duration",
    "convexity",
    "key_rate_durations",
    "vega",
    "delta",
    "gamma",
    "theta",
    "rho",
}

_LEVEL_RANK = {
    "unsupported": 0,
    "experimental": 1,
    "bump_only": 2,
    "native": 3,
}

_STABILITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass(frozen=True)
class SensitivitySupport:
    """Method-level contract for supported sensitivity measures."""

    method: str
    level: str
    supported_measures: tuple[str, ...]
    stability: str = "medium"
    notes: tuple[str, ...] = ()

    def supports(self, measure: str) -> bool:
        """Whether the contract claims support for one normalized measure."""
        return measure in self.supported_measures

    def to_dict(self) -> dict[str, object]:
        """Serialize into a stable plain mapping for traces and summaries."""
        payload = asdict(self)
        payload["supported_measures"] = list(self.supported_measures)
        payload["notes"] = list(self.notes)
        return payload


_DEFAULT_NOTES = {
    "analytical": (
        "Current analytics exposure is mostly repricing-based; native closed-form Greeks are not yet a unified runtime contract.",
    ),
    "rate_tree": (
        "Sensitivities are available through deterministic repricing on calibrated trees, not native adjoints.",
    ),
    "monte_carlo": (
        "Greeks rely on repricing and can be noisy, especially around early-exercise boundaries.",
    ),
    "fft_pricing": (
        "Transform routes support repricing-based sensitivities; native differentiated transform Greeks are not yet exposed.",
    ),
    "pde_solver": (
        "Finite-difference routes can be repriced for sensitivities; native grid-adjoint Greeks are not yet standardized.",
    ),
    "copula": (
        "Credit/correlation routes currently expose sensitivities only via scenario or bump repricing.",
    ),
    "waterfall": (
        "Structured cashflow routes expose rate-style sensitivities only through repricing.",
    ),
}

_SENSITIVITY_SUPPORT_BY_METHOD = {
    "analytical": SensitivitySupport(
        method="analytical",
        level="bump_only",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="high",
        notes=_DEFAULT_NOTES["analytical"],
    ),
    "rate_tree": SensitivitySupport(
        method="rate_tree",
        level="bump_only",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="medium",
        notes=_DEFAULT_NOTES["rate_tree"],
    ),
    "pde_solver": SensitivitySupport(
        method="pde_solver",
        level="bump_only",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="medium",
        notes=_DEFAULT_NOTES["pde_solver"],
    ),
    "fft_pricing": SensitivitySupport(
        method="fft_pricing",
        level="bump_only",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="medium",
        notes=_DEFAULT_NOTES["fft_pricing"],
    ),
    "monte_carlo": SensitivitySupport(
        method="monte_carlo",
        level="experimental",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="low",
        notes=_DEFAULT_NOTES["monte_carlo"],
    ),
    "qmc": SensitivitySupport(
        method="qmc",
        level="experimental",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
            "vega",
        ),
        stability="low",
        notes=_DEFAULT_NOTES["monte_carlo"],
    ),
    "copula": SensitivitySupport(
        method="copula",
        level="experimental",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
        ),
        stability="low",
        notes=_DEFAULT_NOTES["copula"],
    ),
    "waterfall": SensitivitySupport(
        method="waterfall",
        level="bump_only",
        supported_measures=(
            "dv01",
            "duration",
            "convexity",
            "key_rate_durations",
        ),
        stability="medium",
        notes=_DEFAULT_NOTES["waterfall"],
    ),
}

_UNSUPPORTED = SensitivitySupport(
    method="unsupported",
    level="unsupported",
    supported_measures=(),
    stability="low",
    notes=("No standardized sensitivity contract is currently registered for this route.",),
)


def normalize_requested_outputs(
    outputs,
) -> tuple[str, ...]:
    """Return canonical output identifiers for request/compiler surfaces.

    Known DSL measures are normalized through ``normalize_dsl_measure``.
    Unknown outputs are preserved as lowercase snake_case strings so the
    request surface can remain forward-compatible while sensitivity routing
    still relies on the narrower ``normalize_requested_measures`` shim.
    """
    from trellis.core.types import normalize_dsl_measure

    if not outputs:
        return ()
    normalized: list[str] = []
    for output in outputs:
        if isinstance(output, str):
            raw = output
        elif isinstance(output, dict) and output:
            raw = str(next(iter(output)))
        else:
            raw = getattr(output, "value", getattr(output, "name", str(output)))
        text = str(raw).strip()
        if not text:
            continue
        try:
            canonical = normalize_dsl_measure(text).value
        except ValueError:
            canonical = text.lower().replace(" ", "_")
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def normalize_requested_measures(
    measures: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return the normalized sensitivity-only subset of requested measures.

    Returns ``DslMeasure`` enum members (which are ``str`` subclasses,
    so all existing comparisons continue to work).
    """
    from trellis.core.types import DslMeasure, normalize_dsl_measure

    if not measures:
        return ()
    normalized: list[str] = []
    for measure in measures:
        if not isinstance(measure, str):
            continue
        try:
            dsl = normalize_dsl_measure(measure)
        except ValueError:
            continue
        if dsl == DslMeasure.PRICE:
            continue
        if dsl.value in _SENSITIVITY_MEASURES and dsl not in normalized:
            normalized.append(dsl)
    return tuple(normalized)


def support_for_method(method: str) -> SensitivitySupport:
    """Return the registered sensitivity contract for one canonical method."""
    normalized = normalize_method(method)
    support = _SENSITIVITY_SUPPORT_BY_METHOD.get(normalized)
    if support is not None:
        return support
    return SensitivitySupport(
        method=normalized,
        level=_UNSUPPORTED.level,
        supported_measures=_UNSUPPORTED.supported_measures,
        stability=_UNSUPPORTED.stability,
        notes=_UNSUPPORTED.notes,
    )


def rank_sensitivity_support(
    support: SensitivitySupport,
    requested_measures: tuple[str, ...] | list[str] | None,
) -> tuple[int, int, int, str]:
    """Rank a route for requested sensitivities.

    Higher tuples are better. Missing coverage is penalized heavily so routes
    that cannot satisfy the requested sensitivity set lose to routes that can.
    """
    requested = normalize_requested_measures(requested_measures)
    if not requested:
        return (0, 0, 0, support.method)

    coverage = sum(1 for measure in requested if support.supports(measure))
    missing = len(requested) - coverage
    level_rank = _LEVEL_RANK.get(support.level, 0)
    stability_rank = _STABILITY_RANK.get(support.stability, 0)
    return (-missing, level_rank, stability_rank, support.method)
