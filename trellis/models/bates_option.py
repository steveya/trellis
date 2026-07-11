"""Bates stochastic-volatility jump helpers for European vanilla options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.schemes import Euler, HestonQuadraticExponential
from trellis.models.processes.heston import (
    HestonRuntimeBinding,
    normalize_heston_parameter_payload,
    resolve_heston_runtime_binding,
)
from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price


@dataclass(frozen=True)
class ResolvedBatesOptionInputs:
    """Resolved contract, Heston, and jump inputs for a Bates vanilla option."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    runtime_binding: HestonRuntimeBinding
    characteristic_family: str = "bates_log_spot"
    validation_bundle: str = "bates:affine_jump_stochastic_vol"

    @property
    def jump_compensator(self) -> float:
        """Return ``E[exp(Y)-1]`` for the lognormal jump size ``Y``."""
        return float(raw_np.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0)


@dataclass(frozen=True)
class BatesOptionTransformResult:
    """Structured transform result for a Bates vanilla option."""

    price: float
    method: str
    maturity: float
    model_parameters: dict[str, object]
    jump_parameters: dict[str, float]
    runtime_binding: dict[str, object]
    characteristic_family: str
    validation_bundle: str


@dataclass(frozen=True)
class BatesOptionMonteCarloResult:
    """Structured Monte Carlo result for a Bates vanilla option."""

    price: float
    standard_error: float
    n_paths: int
    n_steps: int
    seed: int | None
    variance_scheme: str
    model_parameters: dict[str, object]
    jump_parameters: dict[str, float]
    runtime_binding: dict[str, object]
    validation_bundle: str


def resolve_bates_option_inputs(
    market_state,
    spec,
    *,
    parameter_set_name: str | None = None,
) -> ResolvedBatesOptionInputs:
    """Resolve a European vanilla option under Bates dynamics."""
    settlement = getattr(market_state, "settlement", None) or getattr(
        market_state,
        "as_of",
        None,
    )
    if settlement is None:
        raise ValueError("Bates option pricing requires market_state.settlement or as_of")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = _resolve_maturity(settlement, spec, day_count)
    if getattr(market_state, "discount", None) is None and maturity > 0.0:
        raise ValueError("Bates option pricing requires market_state.discount")

    rate = 0.0 if maturity <= 0.0 else float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    dividend_yield = _resolve_dividend_yield(spec)
    jump_payload = _resolve_jump_parameter_payload(market_state, spec)
    jump_intensity = float(
        _resolve_positive_float(
            jump_payload,
            ("jump_intensity", "lambda", "lam", "intensity"),
            default=0.0,
        )
    )
    jump_mean = float(
        _resolve_float(
            jump_payload,
            ("jump_mean", "jump_mu", "mean_jump", "m"),
            default=0.0,
        )
    )
    jump_vol = _resolve_jump_vol(jump_payload)
    jump_compensator = float(raw_np.exp(jump_mean + 0.5 * jump_vol**2) - 1.0)

    heston_payload, selected_name = _resolve_heston_parameter_payload(
        market_state,
        spec,
        parameter_set_name=parameter_set_name,
    )
    runtime_binding = resolve_heston_runtime_binding(
        market_state,
        mu=rate - dividend_yield - jump_intensity * jump_compensator,
        kappa=float(heston_payload["kappa"]),
        theta=float(heston_payload["theta"]),
        xi=float(heston_payload["xi"]),
        rho=float(heston_payload["rho"]),
        v0=float(heston_payload["v0"]),
        parameter_set_name=selected_name,
    )

    spot = _resolve_spot(market_state, spec)
    return ResolvedBatesOptionInputs(
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
        spot=spot,
        strike=_resolve_strike(spec, spot=spot),
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=normalized_option_type(getattr(spec, "option_type", "call")),
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
        runtime_binding=runtime_binding,
    )


def bates_log_ratio_char_fn(resolved: ResolvedBatesOptionInputs):
    """Return ``E[exp(iu log(S_T/S_0))]`` under Bates dynamics."""
    process = resolved.runtime_binding.process

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        heston_cf = process.characteristic_function(
            u_arr,
            resolved.maturity,
            log_spot=0.0,
        )
        jump_cf = raw_np.exp(
            1j * u_arr * resolved.jump_mean
            - 0.5 * resolved.jump_vol**2 * u_arr**2
        )
        jump_part = raw_np.exp(resolved.jump_intensity * resolved.maturity * (jump_cf - 1.0))
        return heston_cf * jump_part

    return phi


def bates_log_spot_char_fn(resolved: ResolvedBatesOptionInputs):
    """Return ``E[exp(iu log(S_T))]`` under Bates dynamics."""
    log_spot = raw_np.log(resolved.spot)
    ratio_cf = bates_log_ratio_char_fn(resolved)

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        return raw_np.exp(1j * u_arr * log_spot) * ratio_cf(u_arr)

    return phi


def price_bates_option_transform_result(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    parameter_set_name: str | None = None,
) -> BatesOptionTransformResult:
    """Return a structured FFT/COS Bates vanilla option price."""
    resolved = resolve_bates_option_inputs(
        market_state,
        spec,
        parameter_set_name=parameter_set_name,
    )
    normalized_method = _normalized_transform_method(
        method if method is not None else getattr(spec, "transform_method", "fft")
    )

    if resolved.maturity <= 0.0:
        raw_price = terminal_intrinsic(
            resolved.option_type,
            spot=resolved.spot,
            strike=resolved.strike,
        )
    elif normalized_method == "fft":
        call_price = fft_price(
            bates_log_spot_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            alpha=float(fft_alpha if fft_alpha is not None else getattr(spec, "fft_alpha", 1.5)),
            N=max(int(fft_points if fft_points is not None else getattr(spec, "fft_points", 4096)), 32),
            eta=float(fft_eta if fft_eta is not None else getattr(spec, "fft_eta", 0.25)),
        )
        raw_price = call_price if resolved.option_type == "call" else _put_from_call_parity(call_price, resolved)
    else:
        raw_price = cos_price(
            bates_log_ratio_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=max(int(cos_points if cos_points is not None else getattr(spec, "cos_points", 1024)), 16),
            L=float(cos_truncation if cos_truncation is not None else getattr(spec, "cos_truncation", 14.0)),
            option_type=resolved.option_type,
        )

    return BatesOptionTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=normalized_method,
        maturity=resolved.maturity,
        model_parameters=dict(resolved.runtime_binding.model_parameters),
        jump_parameters=_jump_parameter_payload(resolved),
        runtime_binding=resolved.runtime_binding.to_payload(),
        characteristic_family=resolved.characteristic_family,
        validation_bundle=resolved.validation_bundle,
    )


def price_bates_option_transform(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
    parameter_set_name: str | None = None,
) -> float:
    """Return a scalar FFT/COS Bates vanilla option price."""
    return float(
        price_bates_option_transform_result(
            market_state,
            spec,
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
            parameter_set_name=parameter_set_name,
        ).price
    )


def price_bates_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    scheme: str | None = None,
    parameter_set_name: str | None = None,
) -> BatesOptionMonteCarloResult:
    """Price a Bates vanilla option by Heston paths plus compound-Poisson jumps."""
    resolved = resolve_bates_option_inputs(
        market_state,
        spec,
        parameter_set_name=parameter_set_name,
    )
    resolved_n_paths = max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 80_000)), 1)
    resolved_n_steps = max(int(n_steps if n_steps is not None else getattr(spec, "n_steps", 96)), 1)
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    variance_scheme = _normalized_mc_scheme(
        scheme if scheme is not None else getattr(spec, "variance_scheme", getattr(spec, "scheme", "heston_qe"))
    )

    if resolved.maturity <= 0.0:
        intrinsic = terminal_intrinsic(
            resolved.option_type,
            spot=resolved.spot,
            strike=resolved.strike,
        )
        return _mc_result(
            resolved,
            price=float(resolved.notional) * float(intrinsic),
            standard_error=0.0,
            n_paths=0,
            n_steps=resolved_n_steps,
            seed=resolved_seed,
            variance_scheme=variance_scheme,
        )

    rng = raw_np.random.default_rng(resolved_seed)
    engine = MonteCarloEngine(
        resolved.runtime_binding.process,
        n_paths=resolved_n_paths,
        n_steps=resolved_n_steps,
        seed=resolved_seed,
        scheme=_scheme_object(variance_scheme),
    )
    initial_state = raw_np.asarray(
        [resolved.spot, resolved.runtime_binding.process.v0],
        dtype=float,
    )
    paths = engine.simulate(initial_state, resolved.maturity)
    heston_terminal = raw_np.asarray(paths[:, -1, 0], dtype=float)
    arrivals = rng.poisson(resolved.jump_intensity * resolved.maturity, size=resolved_n_paths)
    jump_shock = rng.standard_normal(resolved_n_paths)
    jump_multiplier = raw_np.exp(
        arrivals * resolved.jump_mean
        + raw_np.sqrt(arrivals) * resolved.jump_vol * jump_shock
    )
    terminal = heston_terminal * jump_multiplier
    payoffs = resolved.notional * terminal_intrinsic(
        resolved.option_type,
        spot=terminal,
        strike=resolved.strike,
    )
    discounted = raw_np.exp(-resolved.rate * resolved.maturity) * payoffs
    return _mc_result(
        resolved,
        price=float(raw_np.mean(discounted)),
        standard_error=float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(resolved_n_paths)),
        n_paths=resolved_n_paths,
        n_steps=resolved_n_steps,
        seed=resolved_seed,
        variance_scheme=variance_scheme,
    )


def price_bates_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
    scheme: str | None = None,
    parameter_set_name: str | None = None,
) -> float:
    """Return a scalar Bates Monte Carlo price for a European vanilla option."""
    return float(
        price_bates_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            scheme=scheme,
            parameter_set_name=parameter_set_name,
        ).price
    )


def _mc_result(
    resolved: ResolvedBatesOptionInputs,
    *,
    price: float,
    standard_error: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    variance_scheme: str,
) -> BatesOptionMonteCarloResult:
    return BatesOptionMonteCarloResult(
        price=float(price),
        standard_error=float(standard_error),
        n_paths=int(n_paths),
        n_steps=int(n_steps),
        seed=seed,
        variance_scheme=variance_scheme,
        model_parameters=dict(resolved.runtime_binding.model_parameters),
        jump_parameters=_jump_parameter_payload(resolved),
        runtime_binding=resolved.runtime_binding.to_payload(),
        validation_bundle=resolved.validation_bundle,
    )


def _resolve_heston_parameter_payload(
    market_state,
    spec,
    *,
    parameter_set_name: str | None,
) -> tuple[Mapping[str, object], str]:
    for explicit in (
        getattr(spec, "model_parameters", None),
        getattr(spec, "heston_parameters", None),
        getattr(spec, "bates_parameters", None),
    ):
        extracted = _extract_heston_payload(explicit)
        if extracted is not None:
            return extracted, str(parameter_set_name or extracted.get("parameter_set_name") or "bates")

    set_name = (
        parameter_set_name
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
            raise ValueError(f"Unknown Bates model parameter set {key!r}")
        extracted = _extract_heston_payload(sets[key])
        if extracted is None:
            raise ValueError(f"Model parameter set {key!r} is not a Heston/Bates payload")
        return extracted, key

    candidates = (
        ("model_parameters", getattr(market_state, "model_parameters", None)),
        ("bates_equity", sets.get("bates_equity")),
        ("heston_equity", sets.get("heston_equity")),
        ("bates_validation", sets.get("bates_validation")),
        ("heston_validation", sets.get("heston_validation")),
        *_named_single_items(sets),
    )
    for name, candidate in candidates:
        extracted = _extract_heston_payload(candidate)
        if extracted is not None:
            return extracted, str(extracted.get("parameter_set_name") or name or "bates")
    raise ValueError("Bates option pricing requires Heston/Bates model_parameters")


def _extract_heston_payload(payload: object) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    normalized = normalize_heston_parameter_payload(payload)
    if _is_heston_payload(normalized):
        return normalized
    for key in ("bates", "heston", "heston_equity", "bates_equity"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            normalized_nested = normalize_heston_parameter_payload(nested)
            if _is_heston_payload(normalized_nested):
                return normalized_nested
    return None


def _is_heston_payload(payload: Mapping[str, object]) -> bool:
    family = str(payload.get("model_family", payload.get("family", "")) or "").strip().lower()
    required = {"kappa", "theta", "xi", "rho", "v0"}
    return family in {"", "heston", "bates"} and required.issubset(payload)


def _resolve_jump_parameter_payload(market_state, spec) -> Mapping[str, object]:
    explicit = getattr(spec, "jump_parameters", None)
    if isinstance(explicit, Mapping):
        return explicit
    set_name = (
        getattr(spec, "jump_parameter_set", None)
        or getattr(spec, "jump_parameter_name", None)
        or getattr(spec, "jump_parameter_id", None)
    )
    selected_curve_name = getattr(market_state, "selected_curve_name", None)
    if set_name is None and callable(selected_curve_name):
        set_name = selected_curve_name("jump_parameters")
    sets = dict(getattr(market_state, "jump_parameter_sets", None) or {})
    if set_name is not None:
        key = str(set_name)
        if key not in sets:
            raise ValueError(f"Unknown jump parameter set {key!r}")
        return sets[key]
    default = getattr(market_state, "jump_parameters", None)
    if isinstance(default, Mapping):
        return default
    for key in ("bates_equity", "merton_equity", "bates_validation", "merton_validation"):
        if isinstance(sets.get(key), Mapping):
            return sets[key]
    if len(sets) == 1:
        return next(iter(sets.values()))
    raise ValueError("Bates option pricing requires jump_parameters")


def _resolve_maturity(settlement: date, spec, day_count) -> float:
    for attr in (
        "expiry_date",
        "maturity_date",
        "expiration_date",
        "exercise_date",
        "expiry",
        "maturity",
        "expiry_years",
        "maturity_years",
        "time_to_maturity",
        "tenor_years",
    ):
        value = getattr(spec, attr, None)
        if value is None:
            continue
        if isinstance(value, date):
            return max(float(year_fraction(settlement, value, day_count)), 0.0)
        if isinstance(value, str):
            try:
                return max(float(year_fraction(settlement, date.fromisoformat(value.strip()), day_count)), 0.0)
            except ValueError:
                pass
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
    raise ValueError("Bates option pricing requires an expiry or numeric maturity")


def _resolve_spot(market_state, spec) -> float:
    for attr in ("spot", "underlier_spot", "s0"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    market_spot = getattr(market_state, "spot", None)
    if market_spot is not None:
        return float(market_spot)
    underlier_spots = getattr(market_state, "underlier_spots", None)
    if isinstance(underlier_spots, Mapping):
        underlier = str(getattr(spec, "underlier", "") or getattr(spec, "underlier_id", "") or "")
        if underlier and underlier in underlier_spots:
            return float(underlier_spots[underlier])
        if len(underlier_spots) == 1:
            return float(next(iter(underlier_spots.values())))
        if "SPX" in underlier_spots:
            return float(underlier_spots["SPX"])
    raise ValueError("Bates option pricing requires spec.spot or market_state.spot")


def _resolve_strike(spec, *, spot: float) -> float:
    for attr in ("strike", "strike_price", "k"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    return float(spot)


def _resolve_dividend_yield(spec) -> float:
    for attr in ("dividend_yield", "continuous_dividend_yield", "q"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _resolve_jump_vol(payload: Mapping[str, object]) -> float:
    jump_vol = _resolve_positive_float(
        payload,
        ("jump_vol", "jump_sigma", "jump_std", "delta"),
        default=None,
    )
    if jump_vol is not None:
        return float(jump_vol)
    jump_variance = _resolve_positive_float(
        payload,
        ("jump_variance", "jump_var", "jump_vol_squared"),
        default=0.0,
    )
    return float(raw_np.sqrt(float(jump_variance or 0.0)))


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
        raise ValueError(f"Jump parameter {names[0]} must be non-negative")
    return value


def _jump_parameter_payload(resolved: ResolvedBatesOptionInputs) -> dict[str, float]:
    return {
        "jump_intensity": float(resolved.jump_intensity),
        "jump_mean": float(resolved.jump_mean),
        "jump_vol": float(resolved.jump_vol),
    }


def _put_from_call_parity(
    call_price: float,
    resolved: ResolvedBatesOptionInputs,
) -> float:
    discounted_strike = resolved.strike * raw_np.exp(-resolved.rate * resolved.maturity)
    discounted_spot = resolved.spot * raw_np.exp(-resolved.dividend_yield * resolved.maturity)
    return float(call_price - discounted_spot + discounted_strike)


def _normalized_transform_method(value: object) -> str:
    method = str(value or "fft").strip().lower().replace("-", "_")
    aliases = {
        "bates_fft": "fft",
        "carr_madan": "fft",
        "fang_oosterlee": "cos",
        "fourier_cosine": "cos",
        "bates_cos": "cos",
    }
    method = aliases.get(method, method)
    if method not in {"fft", "cos"}:
        raise ValueError(f"Unsupported Bates transform method {value!r}")
    return method


def _normalized_mc_scheme(value: object) -> str:
    scheme = str(value or "heston_qe").strip().lower().replace("-", "_")
    aliases = {
        "qe": "heston_qe",
        "andersen_qe": "heston_qe",
        "quadratic_exponential": "heston_qe",
        "bates_qe": "heston_qe",
        "bates_mc": "heston_qe",
        "euler_heston": "euler",
        "bates_euler": "euler",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in {"euler", "heston_qe"}:
        raise ValueError(f"Unsupported Bates Monte Carlo scheme {value!r}")
    return scheme


def _scheme_object(variance_scheme: str):
    if variance_scheme == "heston_qe":
        return HestonQuadraticExponential()
    return Euler()


def _named_single_items(sets: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    if len(sets) != 1:
        return ()
    name, value = next(iter(sets.items()))
    return ((str(name), value),)


__all__ = [
    "BatesOptionMonteCarloResult",
    "BatesOptionTransformResult",
    "ResolvedBatesOptionInputs",
    "bates_log_ratio_char_fn",
    "bates_log_spot_char_fn",
    "price_bates_option_monte_carlo",
    "price_bates_option_monte_carlo_result",
    "price_bates_option_transform",
    "price_bates_option_transform_result",
    "resolve_bates_option_inputs",
]
