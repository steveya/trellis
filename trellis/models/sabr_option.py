"""SABR helpers for European forward-style vanilla options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.black import black76_call, black76_put
from trellis.models.processes.sabr import SABRProcess


@dataclass(frozen=True)
class ResolvedSabrForwardOptionInputs:
    """Resolved contract, market, and SABR parameters for a vanilla forward option."""

    notional: float
    forward: float
    strike: float
    maturity: float
    rate: float
    option_type: str
    alpha: float
    beta: float
    rho: float
    nu: float

    @property
    def discount_factor(self) -> float:
        """Return deterministic discount factor to expiry."""
        return float(raw_np.exp(-self.rate * self.maturity))


@dataclass(frozen=True)
class SabrForwardOptionMonteCarloResult:
    """Structured SABR Monte Carlo result for a vanilla forward option."""

    price: float
    standard_error: float
    n_paths: int
    n_steps: int
    seed: int | None


def resolve_sabr_forward_option_inputs(
    market_state,
    spec,
) -> ResolvedSabrForwardOptionInputs:
    """Resolve a European forward-style vanilla option under SABR dynamics."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for pricing")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    strike = float(getattr(spec, "strike"))
    forward = _resolve_forward(market_state, spec, maturity)
    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    sabr = _resolve_sabr_parameter_payload(market_state, spec)

    if market_state.discount is None:
        raise ValueError("SABR forward option pricing requires market_state.discount")
    rate = 0.0 if maturity <= 0.0 else float(market_state.discount.zero_rate(max(maturity, 1e-6)))

    alpha = _resolve_positive_float(sabr, ("alpha", "sigma0", "initial_vol"), default=None)
    beta = _resolve_float(sabr, ("beta",), default=None)
    rho = _resolve_float(sabr, ("rho", "correlation"), default=None)
    nu = _resolve_positive_float(sabr, ("nu", "vol_of_vol", "volvol"), default=None)
    missing = [
        name
        for name, value in (
            ("alpha", alpha),
            ("beta", beta),
            ("rho", rho),
            ("nu", nu),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"SABR parameter payload is missing {', '.join(missing)}")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"SABR beta must be in [0, 1], got {beta}")
    if not -1.0 < float(rho) < 1.0:
        raise ValueError(f"SABR rho must be strictly between -1 and 1, got {rho}")

    return ResolvedSabrForwardOptionInputs(
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
        forward=forward,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type=option_type,
        alpha=float(alpha),
        beta=float(beta),
        rho=float(rho),
        nu=float(nu),
    )


def price_sabr_forward_option_hagan(market_state, spec) -> float:
    """Price a vanilla forward option with Hagan's SABR implied-vol approximation."""
    resolved = resolve_sabr_forward_option_inputs(market_state, spec)
    if resolved.maturity <= 0.0:
        return float(
            resolved.notional
            * terminal_intrinsic(
                resolved.option_type,
                spot=resolved.forward,
                strike=resolved.strike,
            )
        )

    process = SABRProcess(resolved.alpha, resolved.beta, resolved.rho, resolved.nu)
    implied_vol = float(
        process.implied_vol(
            max(resolved.forward, 1e-12),
            max(resolved.strike, 1e-12),
            max(resolved.maturity, 1e-8),
        )
    )
    if resolved.option_type == "put":
        raw_price = black76_put(
            resolved.forward,
            resolved.strike,
            implied_vol,
            resolved.maturity,
        )
    else:
        raw_price = black76_call(
            resolved.forward,
            resolved.strike,
            implied_vol,
            resolved.maturity,
        )
    return float(resolved.notional * resolved.discount_factor * raw_price)


def price_sabr_forward_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> SabrForwardOptionMonteCarloResult:
    """Price a SABR forward option by Euler Monte Carlo with antithetic shocks."""
    resolved = resolve_sabr_forward_option_inputs(market_state, spec)
    resolved_n_paths = max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 120_000)), 2)
    resolved_n_steps = max(int(n_steps if n_steps is not None else getattr(spec, "n_steps", 96)), 1)
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    if resolved.maturity <= 0.0:
        intrinsic = terminal_intrinsic(
            resolved.option_type,
            spot=resolved.forward,
            strike=resolved.strike,
        )
        return SabrForwardOptionMonteCarloResult(
            price=float(resolved.notional * intrinsic),
            standard_error=0.0,
            n_paths=0,
            n_steps=resolved_n_steps,
            seed=resolved_seed,
        )

    rng = raw_np.random.default_rng(resolved_seed)
    half_paths = (resolved_n_paths + 1) // 2
    raw_forward_shocks = rng.standard_normal((half_paths, resolved_n_steps))
    raw_vol_shocks = rng.standard_normal((half_paths, resolved_n_steps))
    forward_shocks = raw_np.concatenate((raw_forward_shocks, -raw_forward_shocks), axis=0)[
        :resolved_n_paths
    ]
    independent_vol_shocks = raw_np.concatenate((raw_vol_shocks, -raw_vol_shocks), axis=0)[
        :resolved_n_paths
    ]
    vol_shocks = (
        resolved.rho * forward_shocks
        + raw_np.sqrt(max(1.0 - resolved.rho**2, 0.0)) * independent_vol_shocks
    )

    dt = resolved.maturity / resolved_n_steps
    sqrt_dt = raw_np.sqrt(dt)
    forward = raw_np.full(resolved_n_paths, resolved.forward, dtype=float)
    alpha = raw_np.full(resolved_n_paths, resolved.alpha, dtype=float)
    for step in range(resolved_n_steps):
        positive_forward = raw_np.maximum(forward, 0.0)
        forward = raw_np.maximum(
            forward + alpha * positive_forward**resolved.beta * sqrt_dt * forward_shocks[:, step],
            0.0,
        )
        alpha = raw_np.maximum(
            alpha
            * raw_np.exp(
                -0.5 * resolved.nu**2 * dt
                + resolved.nu * sqrt_dt * vol_shocks[:, step]
            ),
            0.0,
        )

    payoffs = resolved.notional * terminal_intrinsic(
        resolved.option_type,
        spot=forward,
        strike=resolved.strike,
    )
    discounted = resolved.discount_factor * payoffs
    return SabrForwardOptionMonteCarloResult(
        price=float(raw_np.mean(discounted)),
        standard_error=float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(resolved_n_paths)),
        n_paths=resolved_n_paths,
        n_steps=resolved_n_steps,
        seed=resolved_seed,
    )


def price_sabr_forward_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> float:
    """Return a scalar SABR Euler Monte Carlo forward-option price."""
    return float(
        price_sabr_forward_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        ).price
    )


def _resolve_forward(market_state, spec, maturity: float) -> float:
    for name in ("forward", "forward_rate", "swap_rate", "futures_price"):
        value = getattr(spec, name, None)
        if value is not None:
            return float(value)
    spot = getattr(spec, "spot", None)
    if spot is not None:
        return float(spot)
    market_spot = getattr(market_state, "spot", None)
    if market_spot is not None:
        return float(market_spot)
    if getattr(market_state, "forward_curve", None) is not None:
        return float(market_state.forward_curve.forward_rate(0.0, max(maturity, 1e-6)))
    raise ValueError("SABR forward option pricing requires spec.forward, spec.spot, or market_state.spot")


def _resolve_sabr_parameter_payload(market_state, spec) -> Mapping[str, object]:
    explicit = getattr(spec, "sabr", None)
    extracted = _extract_sabr_payload(explicit)
    if extracted is not None:
        return extracted

    set_name = (
        getattr(spec, "sabr_parameter_set", None)
        or getattr(spec, "model_parameter_set", None)
        or getattr(spec, "model_parameter_name", None)
        or getattr(spec, "model_parameter_id", None)
    )
    selected_curve_name = getattr(market_state, "selected_curve_name", None)
    if set_name is None and callable(selected_curve_name):
        set_name = selected_curve_name("model_parameters")
    sets = dict(getattr(market_state, "model_parameter_sets", None) or {})
    if set_name is not None:
        key = str(set_name)
        if key not in sets:
            raise ValueError(f"Unknown SABR model parameter set {key!r}")
        extracted = _extract_sabr_payload(sets[key])
        if extracted is None:
            raise ValueError(f"Model parameter set {key!r} is not a SABR payload")
        return extracted

    for candidate in _sabr_payload_candidates(market_state, sets):
        extracted = _extract_sabr_payload(candidate)
        if extracted is not None:
            return extracted
    raise ValueError("SABR forward option pricing requires SABR model_parameters")


def _sabr_payload_candidates(market_state, sets: Mapping[str, object]) -> tuple[object, ...]:
    provenance = getattr(market_state, "market_provenance", None)
    return (
        getattr(market_state, "model_parameters", None),
        sets.get("sabr"),
        sets.get("rate_vol_model"),
        sets.get("sabr_validation"),
        *_single_item_values(sets),
        _nested_mapping(
            provenance,
            (
                "prior_parameters",
                "synthetic_generation_contract",
                "model_packs",
                "rates",
                "rate_vol_model",
            ),
        ),
        _nested_mapping(
            provenance,
            ("synthetic_generation_contract", "model_packs", "rates", "rate_vol_model"),
        ),
    )


def _extract_sabr_payload(payload: object) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    if _has_sabr_keys(payload):
        return payload
    for key in ("sabr", "sabr_parameters", "rate_vol_model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and _has_sabr_keys(nested):
            return nested
    if str(payload.get("family") or "").strip().lower() == "sabr" and _has_sabr_keys(payload):
        return payload
    return None


def _has_sabr_keys(payload: Mapping[str, object]) -> bool:
    return all(key in payload and payload[key] is not None for key in ("alpha", "beta", "rho", "nu"))


def _single_item_values(mapping: Mapping[str, object]) -> tuple[object, ...]:
    if len(mapping) == 1:
        return (next(iter(mapping.values())),)
    return ()


def _nested_mapping(payload: object, path: tuple[str, ...]) -> object | None:
    current = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _resolve_float(
    payload: Mapping[str, object],
    names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    for name in names:
        if name in payload and payload[name] is not None:
            return float(payload[name])
    return default


def _resolve_positive_float(
    payload: Mapping[str, object],
    names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    value = _resolve_float(payload, names, default=default)
    if value is not None and value < 0.0:
        raise ValueError(f"SABR parameter {names[0]} must be non-negative")
    return value


__all__ = [
    "ResolvedSabrForwardOptionInputs",
    "SabrForwardOptionMonteCarloResult",
    "price_sabr_forward_option_hagan",
    "price_sabr_forward_option_monte_carlo",
    "price_sabr_forward_option_monte_carlo_result",
    "resolve_sabr_forward_option_inputs",
]
