"""Heston stochastic volatility model and runtime binding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from trellis.core.differentiable import get_numpy
from trellis.models.processes.base import StochasticProcess

np = get_numpy()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


def build_heston_parameter_payload(
    *,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    mu: float | None = None,
    parameter_set_name: str = "heston",
    source_kind: str = "explicit_input",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a serializable Heston model-parameter payload."""
    payload: dict[str, object] = {
        "model_family": "heston",
        "kappa": float(kappa),
        "theta": float(theta),
        "xi": float(xi),
        "rho": float(rho),
        "v0": float(v0),
        "parameter_set_name": parameter_set_name,
        "source_kind": source_kind,
    }
    if mu is not None:
        payload["mu"] = float(mu)
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def extract_heston_parameter_payload(market_state) -> dict[str, object] | None:
    """Return the first Heston parameter payload attached to ``market_state``."""
    candidates = []
    direct = getattr(market_state, "model_parameters", None)
    if isinstance(direct, Mapping):
        candidates.append(direct)
    for payload in dict(getattr(market_state, "model_parameter_sets", None) or {}).values():
        if isinstance(payload, Mapping):
            candidates.append(payload)

    required = {"kappa", "theta", "xi", "rho", "v0"}
    for payload in candidates:
        model_family = str(payload.get("model_family", "")).strip().lower()
        if model_family not in {"", "heston"}:
            continue
        if required.issubset(payload):
            return dict(payload)
    return None


@dataclass(frozen=True)
class HestonRuntimeBinding:
    """Resolved runtime-facing Heston process binding."""

    process: "Heston"
    parameter_set_name: str
    model_parameters: dict[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "parameter_set_name": self.parameter_set_name,
            "model_parameters": dict(self.model_parameters),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


class Heston(StochasticProcess):
    """Heston stochastic volatility model (two-factor)."""

    def __init__(self, mu: float, kappa: float, theta: float, xi: float, rho: float, v0: float):
        """Store the spot and variance dynamics parameters for the Heston model."""
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)

    @property
    def state_dim(self) -> int:
        """Return the two-dimensional Heston state size."""
        return 2

    @property
    def factor_dim(self) -> int:
        """Return the two Brownian factors used by the runtime process."""
        return 2

    def _state_components(self, x):
        """Return spot and variance slices from a runtime state tensor."""
        state = np.asarray(x, dtype=float)
        if state.shape[-1] != 2:
            raise ValueError("Heston runtime state must end with (spot, variance)")
        return state[..., 0], state[..., 1]

    def drift_s(self, s, v, t):
        """Return the spot drift ``mu * S_t``."""
        return self.mu * s

    def diffusion_s(self, s, v, t):
        """Return the spot diffusion loading ``sqrt(V_t) * S_t``."""
        return np.sqrt(np.maximum(v, 0.0)) * s

    def drift_v(self, s, v, t):
        """Return the variance drift ``kappa * (theta - V_t)``."""
        return self.kappa * (self.theta - v)

    def diffusion_v(self, s, v, t):
        """Return the variance diffusion loading ``xi * sqrt(max(V_t, 0))``."""
        return self.xi * np.sqrt(np.maximum(v, 0.0))

    def drift(self, x, t):
        """Return the Heston drift vector for spot and variance."""
        s, v = self._state_components(x)
        return np.stack((self.drift_s(s, v, t), self.drift_v(s, v, t)), axis=-1)

    def diffusion(self, x, t):
        """Return the Heston diffusion matrix with correlated variance shocks."""
        s, v = self._state_components(x)
        sqrt_v = np.sqrt(np.maximum(v, 0.0))
        spot_loading = sqrt_v * s
        variance_loading = self.xi * sqrt_v
        orthogonal_scale = np.sqrt(max(1.0 - self.rho ** 2, 0.0))
        first_row = np.stack((spot_loading, np.zeros_like(spot_loading)), axis=-1)
        second_row = np.stack((self.rho * variance_loading, orthogonal_scale * variance_loading), axis=-1)
        return np.stack((first_row, second_row), axis=-2)

    def characteristic_function(self, u, t, log_spot: float = 0.0):
        """Heston characteristic function phi(u, t) = E[exp(i*u*log(S_T))]."""
        kappa, theta, xi, rho = self.kappa, self.theta, self.xi, self.rho
        v0 = self.v0

        d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        C = (kappa * theta / xi ** 2) * (
            (kappa - rho * xi * 1j * u - d) * t
            - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )
        D = ((kappa - rho * xi * 1j * u - d) / xi ** 2) * (
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        )

        return np.exp(C + D * v0 + 1j * u * (log_spot + self.mu * t))


def resolve_heston_runtime_binding(
    market_state,
    *,
    mu: float | None = None,
    kappa: float | None = None,
    theta: float | None = None,
    xi: float | None = None,
    rho: float | None = None,
    v0: float | None = None,
    parameter_set_name: str = "heston",
) -> HestonRuntimeBinding:
    """Resolve a runtime-facing Heston process binding from explicit or market inputs."""
    payload = extract_heston_parameter_payload(market_state)
    warnings: list[str] = []
    assumptions: list[str] = []
    parameter_sources: dict[str, str] = {}

    def _resolve(name: str, explicit: float | None) -> float:
        if explicit is not None:
            parameter_sources[name] = "explicit"
            return float(explicit)
        if payload is not None and payload.get(name) is not None:
            parameter_sources[name] = "market_state"
            return float(payload[name])
        raise ValueError(f"Heston runtime binding requires {name!r}")

    if mu is not None:
        resolved_mu = float(mu)
        parameter_sources["mu"] = "explicit"
    elif payload is not None and payload.get("mu") is not None:
        resolved_mu = float(payload["mu"])
        parameter_sources["mu"] = "market_state"
    elif getattr(market_state, "discount", None) is not None:
        resolved_mu = float(market_state.discount.zero_rate(1.0))
        parameter_sources["mu"] = "discount_curve"
        warnings.append("mu defaulted from the discount curve 1Y zero rate; zero dividend yield was assumed.")
        assumptions.append("Heston runtime binding assumes zero dividend yield when mu is defaulted from the discount curve.")
    else:
        raise ValueError("Heston runtime binding requires mu explicitly, via model_parameters, or via market_state.discount")

    resolved_kappa = _resolve("kappa", kappa)
    resolved_theta = _resolve("theta", theta)
    resolved_xi = _resolve("xi", xi)
    resolved_rho = _resolve("rho", rho)
    resolved_v0 = _resolve("v0", v0)
    process = Heston(
        mu=resolved_mu,
        kappa=resolved_kappa,
        theta=resolved_theta,
        xi=resolved_xi,
        rho=resolved_rho,
        v0=resolved_v0,
    )
    model_parameters = build_heston_parameter_payload(
        mu=resolved_mu,
        kappa=resolved_kappa,
        theta=resolved_theta,
        xi=resolved_xi,
        rho=resolved_rho,
        v0=resolved_v0,
        parameter_set_name=parameter_set_name,
        source_kind="market_state" if payload is not None else "explicit_input",
        metadata={"parameter_sources": dict(parameter_sources)},
    )
    provenance = {
        "source_kind": "market_state" if payload is not None else "explicit_input",
        "source_ref": "resolve_heston_runtime_binding",
        "parameter_sources": dict(parameter_sources),
        "selected_curve_names": dict(getattr(market_state, "selected_curve_names", None) or {}),
    }
    return HestonRuntimeBinding(
        process=process,
        parameter_set_name=parameter_set_name,
        model_parameters=model_parameters,
        provenance=provenance,
        warnings=tuple(warnings),
        assumptions=tuple(assumptions),
    )


__all__ = [
    "Heston",
    "HestonRuntimeBinding",
    "build_heston_parameter_payload",
    "extract_heston_parameter_payload",
    "resolve_heston_runtime_binding",
]
