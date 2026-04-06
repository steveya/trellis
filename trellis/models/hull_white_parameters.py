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
    candidates = []
    direct = getattr(market_state, "model_parameters", None)
    if isinstance(direct, Mapping):
        candidates.append(direct)
    for payload in dict(getattr(market_state, "model_parameter_sets", None) or {}).values():
        if isinstance(payload, Mapping):
            candidates.append(payload)

    for payload in candidates:
        model_family = str(payload.get("model_family", "")).strip().lower()
        if model_family not in {"", "hull_white"}:
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
