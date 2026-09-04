"""Shared Hull-White model-parameter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


def _require_nonblank_text(value: object, *, field_name: str) -> str:
    """Return one stripped provenance value or reject the incomplete payload."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Hull-White {field_name} must be a nonblank string")
    return value.strip()


def validate_hull_white_parameter_set_name(value: object) -> str:
    """Return an exact parameter-set identity or reject coercion/whitespace."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "Hull-White parameter_set_name must be an exact nonblank string "
            "without surrounding whitespace"
        )
    return value


def _require_supported_source_kind(value: object) -> str:
    if not isinstance(value, str) or value not in {"observed", "calibrated"}:
        raise ValueError(
            "Hull-White source_kind must be exactly 'observed' or 'calibrated'"
        )
    return value


def _require_finite_nonnegative(value: object, *, field_name: str) -> float:
    """Return one finite non-negative model parameter."""
    if isinstance(value, bool):
        raise ValueError(f"Hull-White {field_name} must be a finite non-negative number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Hull-White {field_name} must be a finite non-negative number"
        ) from exc
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"Hull-White {field_name} must be a finite non-negative number")
    return resolved


@dataclass(frozen=True)
class ResolvedHullWhiteParameterSet:
    """Strictly resolved Hull-White parameters with their named provenance."""

    parameter_set_name: str
    mean_reversion: float
    sigma: float
    source_kind: str
    calibration_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_set_name",
            validate_hull_white_parameter_set_name(self.parameter_set_name),
        )
        object.__setattr__(
            self,
            "mean_reversion",
            _require_finite_nonnegative(
                self.mean_reversion,
                field_name="mean_reversion",
            ),
        )
        object.__setattr__(
            self,
            "sigma",
            _require_finite_nonnegative(self.sigma, field_name="sigma"),
        )
        object.__setattr__(
            self,
            "source_kind",
            _require_supported_source_kind(self.source_kind),
        )
        object.__setattr__(
            self,
            "calibration_source",
            _require_nonblank_text(
                self.calibration_source,
                field_name="calibration_source",
            ),
        )


def resolve_named_hull_white_parameter_set(
    market_state,
    *,
    parameter_set_name: str,
) -> ResolvedHullWhiteParameterSet:
    """Resolve exactly one explicit, provenance-complete Hull-White parameter set.

    This strict path deliberately ignores ``model_parameters``, volatility
    surfaces, and the permissive defaults used by legacy pricing routes.  It is
    intended for routes whose contract names a calibrated Hull-White parameter
    set and therefore cannot safely infer or synthesize one.
    """
    requested_name = validate_hull_white_parameter_set_name(parameter_set_name)

    parameter_sets = getattr(market_state, "model_parameter_sets", None)
    if not isinstance(parameter_sets, Mapping):
        parameter_sets = {}
    payload = parameter_sets.get(requested_name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Missing named Hull-White parameter set {requested_name!r}")

    payload_name = payload.get("parameter_set_name")
    if payload_name != requested_name:
        raise ValueError(
            "Hull-White parameter_set_name must exactly match the requested "
            f"name {requested_name!r}; received {payload_name!r}"
        )
    if payload.get("model_family") != "hull_white":
        raise ValueError(
            "Hull-White model_family must be exactly 'hull_white' for named parameter sets"
        )

    return ResolvedHullWhiteParameterSet(
        parameter_set_name=requested_name,
        mean_reversion=_require_finite_nonnegative(
            payload.get("mean_reversion"),
            field_name="mean_reversion",
        ),
        sigma=_require_finite_nonnegative(payload.get("sigma"), field_name="sigma"),
        source_kind=payload.get("source_kind"),
        calibration_source=_require_nonblank_text(
            payload.get("calibration_source"),
            field_name="calibration_source",
        ),
    )


def build_hull_white_parameter_payload(
    mean_reversion: float,
    sigma: float,
    *,
    calibration_source: str,
    parameter_set_name: str = "hull_white",
    source_kind: str = "calibrated",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a stable serializable Hull-White parameter payload."""
    resolved_parameter_set_name = validate_hull_white_parameter_set_name(
        parameter_set_name
    )
    resolved_source_kind = _require_supported_source_kind(source_kind)
    payload: dict[str, object] = {
        "model_family": "hull_white",
        "mean_reversion": float(mean_reversion),
        "sigma": float(sigma),
        "parameter_set_name": resolved_parameter_set_name,
        "source_kind": resolved_source_kind,
        "calibration_source": _require_nonblank_text(
            calibration_source,
            field_name="calibration_source",
        ),
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
    "ResolvedHullWhiteParameterSet",
    "build_hull_white_parameter_payload",
    "extract_hull_white_parameter_payload",
    "resolve_hull_white_mean_reversion",
    "resolve_hull_white_parameters",
    "resolve_named_hull_white_parameter_set",
    "validate_hull_white_parameter_set_name",
]
