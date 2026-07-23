"""Shared Hull-White model-parameter helpers."""

from __future__ import annotations

from typing import Mapping


def build_hull_white_parameter_payload(
    mean_reversion: float,
    sigma: float,
    *,
    parameter_set_name: str = "hull_white",
    source_kind: str = "calibrated",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a stable serializable Hull-White parameter payload."""
    payload: dict[str, object] = {
        "model_family": "hull_white",
        "mean_reversion": float(mean_reversion),
        "sigma": float(sigma),
        "parameter_set_name": parameter_set_name,
        "source_kind": source_kind,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def extract_hull_white_parameter_payload(market_state) -> dict[str, object] | None:
    """Return the first Hull-White parameter payload attached to ``market_state``."""
    candidates: list[tuple[str, Mapping[str, object], bool]] = []
    direct = getattr(market_state, "model_parameters", None)
    if isinstance(direct, Mapping):
        nested = direct.get("hull_white")
        if isinstance(nested, Mapping):
            candidates.append(("hull_white", nested, False))
        candidates.append(("", direct, True))
    for name, payload in dict(
        getattr(market_state, "model_parameter_sets", None) or {}
    ).items():
        if isinstance(payload, Mapping):
            candidates.append((str(name), payload, False))

    def _family(payload: Mapping[str, object]) -> str:
        return str(
            payload.get("model_family")
            or payload.get("model_name")
            or payload.get("family")
            or ""
        ).strip().lower().replace("-", "_").replace(" ", "_")

    for _, payload, _ in candidates:
        model_family = _family(payload)
        if model_family in {"hull_white", "hullwhite"} and (
            "mean_reversion" in payload or "sigma" in payload
        ):
            return dict(payload)

    for name, payload, _ in candidates:
        normalized_name = name.strip().lower().replace("-", "_").replace(" ", "_")
        if (
            not _family(payload)
            and "hull_white" in normalized_name
            and ("mean_reversion" in payload or "sigma" in payload)
        ):
            return dict(payload)

    # Preserve the legacy direct, untyped payload form without letting an
    # unrelated named parameter set with a generic ``sigma`` field win.
    for _, payload, is_direct in candidates:
        if not is_direct or _family(payload):
            continue
        if "mean_reversion" in payload or "sigma" in payload:
            return dict(payload)
    return None


def resolve_hull_white_mean_reversion(
    market_state,
    *,
    mean_reversion: float | None = None,
    default_mean_reversion: float = 0.1,
) -> float:
    """Resolve a Hull-White mean-reversion parameter from explicit or market inputs."""
    if mean_reversion is not None:
        return float(mean_reversion)
    payload = extract_hull_white_parameter_payload(market_state)
    if payload is not None and payload.get("mean_reversion") is not None:
        return float(payload["mean_reversion"])
    return float(default_mean_reversion)


def resolve_hull_white_parameters(
    market_state,
    *,
    mean_reversion: float | None = None,
    sigma: float | None = None,
    default_mean_reversion: float = 0.1,
    default_sigma: float | None = None,
) -> tuple[float, float]:
    """Resolve Hull-White mean reversion and sigma from explicit or market inputs."""
    resolved_mean_reversion = resolve_hull_white_mean_reversion(
        market_state,
        mean_reversion=mean_reversion,
        default_mean_reversion=default_mean_reversion,
    )
    if sigma is not None:
        return resolved_mean_reversion, float(sigma)

    payload = extract_hull_white_parameter_payload(market_state)
    if payload is not None and payload.get("sigma") is not None:
        return resolved_mean_reversion, float(payload["sigma"])

    if default_sigma is None:
        raise ValueError("Hull-White sigma must be provided explicitly or via market_state.model_parameters")
    return resolved_mean_reversion, float(default_sigma)


__all__ = [
    "build_hull_white_parameter_payload",
    "extract_hull_white_parameter_payload",
    "resolve_hull_white_mean_reversion",
    "resolve_hull_white_parameters",
]
