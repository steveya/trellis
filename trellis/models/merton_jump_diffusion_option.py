"""Merton jump-diffusion helpers for European vanilla equity options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.black import black76_call, black76_put
from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price


@dataclass(frozen=True)
class ResolvedMertonJumpDiffusionOptionInputs:
    """Resolved market, contract, and Merton jump parameters."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    sigma: float
    option_type: str
    jump_intensity: float
    jump_mean: float
    jump_vol: float

    @property
    def jump_compensator(self) -> float:
        """Return ``E[exp(Y)-1]`` for lognormal jump size ``Y``."""
        return float(raw_np.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0)


@dataclass(frozen=True)
class MertonJumpDiffusionOptionMonteCarloResult:
    """Structured terminal Monte Carlo result for a Merton vanilla option."""

    price: float
    standard_error: float
    n_paths: int
    seed: int | None


@dataclass(frozen=True)
class MertonJumpDiffusionOptionTransformResult:
    """Structured transform result for a Merton vanilla option."""

    price: float
    method: str
    maturity: float


def resolve_merton_jump_diffusion_option_inputs(
    market_state,
    spec,
) -> ResolvedMertonJumpDiffusionOptionInputs:
    """Resolve a European vanilla option under Merton jump diffusion."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for pricing")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    strike = float(getattr(spec, "strike"))
    spot = float(getattr(spec, "spot", None) or getattr(market_state, "spot", None))
    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    jump_params = _resolve_jump_parameter_payload(market_state, spec)

    if maturity <= 0.0:
        rate = 0.0
        sigma = 0.0
    else:
        if market_state.discount is None:
            raise ValueError("Merton jump-diffusion pricing requires market_state.discount")
        rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
        sigma = _resolve_positive_float(
            jump_params,
            ("sigma", "diffusion_sigma", "diffusion_vol", "volatility"),
            default=None,
        )
        if sigma is None:
            if market_state.vol_surface is None:
                raise ValueError(
                    "Merton jump-diffusion pricing requires diffusion sigma in jump "
                    "parameters or market_state.vol_surface"
                )
            sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
        if sigma < 0.0:
            raise ValueError(f"Invalid Merton diffusion sigma {sigma}")

    return ResolvedMertonJumpDiffusionOptionInputs(
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=float(getattr(spec, "dividend_yield", 0.0) or 0.0),
        sigma=float(sigma),
        option_type=option_type,
        jump_intensity=float(
            _resolve_positive_float(
                jump_params,
                ("jump_intensity", "lambda", "lam", "intensity"),
                default=0.0,
            )
        ),
        jump_mean=float(
            _resolve_float(
                jump_params,
                ("jump_mean", "jump_mu", "mean_jump", "m"),
                default=0.0,
            )
        ),
        jump_vol=float(
            _resolve_positive_float(
                jump_params,
                ("jump_vol", "jump_sigma", "jump_std", "delta"),
                default=0.0,
            )
        ),
    )


def merton_log_spot_char_fn(resolved: ResolvedMertonJumpDiffusionOptionInputs):
    """Return ``E[exp(iu log(S_T))]`` under risk-neutral Merton dynamics."""
    log_spot = raw_np.log(resolved.spot)
    log_ratio_cf = merton_log_ratio_char_fn(resolved)

    def phi(u):
        return raw_np.exp(1j * u * log_spot) * log_ratio_cf(u)

    return phi


def merton_log_ratio_char_fn(resolved: ResolvedMertonJumpDiffusionOptionInputs):
    """Return ``E[exp(iu log(S_T / S_0))]`` under Merton dynamics."""
    maturity = resolved.maturity
    jump_compensator = resolved.jump_compensator
    drift = (
        resolved.rate
        - resolved.dividend_yield
        - resolved.jump_intensity * jump_compensator
        - 0.5 * resolved.sigma**2
    ) * maturity
    diffusion_variance = resolved.sigma**2 * maturity
    jump_arrival = resolved.jump_intensity * maturity

    def phi(u):
        jump_cf = raw_np.exp(1j * u * resolved.jump_mean - 0.5 * resolved.jump_vol**2 * u**2)
        return raw_np.exp(
            1j * u * drift
            - 0.5 * diffusion_variance * u**2
            + jump_arrival * (jump_cf - 1.0)
        )

    return phi


def price_merton_jump_diffusion_option_poisson_series(
    market_state,
    spec,
    *,
    max_terms: int | None = None,
    tail_tolerance: float = 1e-12,
) -> float:
    """Price a Merton vanilla option by summing conditional Black76 prices."""
    resolved = resolve_merton_jump_diffusion_option_inputs(market_state, spec)
    if resolved.maturity <= 0.0:
        return float(
            resolved.notional
            * terminal_intrinsic(
                resolved.option_type,
                spot=resolved.spot,
                strike=resolved.strike,
            )
        )

    arrival = resolved.jump_intensity * resolved.maturity
    max_n = int(max_terms if max_terms is not None else max(60, arrival + 12.0 * raw_np.sqrt(arrival + 1.0)))
    df = float(raw_np.exp(-resolved.rate * resolved.maturity))
    jump_compensator = resolved.jump_compensator

    total = 0.0
    weight = float(raw_np.exp(-arrival))
    cumulative = 0.0
    for n in range(max_n + 1):
        variance = resolved.sigma**2 * resolved.maturity + n * resolved.jump_vol**2
        volatility = float(raw_np.sqrt(max(variance, 0.0) / resolved.maturity))
        conditional_forward = resolved.spot * raw_np.exp(
            (resolved.rate - resolved.dividend_yield - resolved.jump_intensity * jump_compensator)
            * resolved.maturity
            + n * resolved.jump_mean
            + 0.5 * n * resolved.jump_vol**2
        )
        if resolved.option_type == "put":
            conditional = black76_put(
                conditional_forward,
                resolved.strike,
                volatility,
                resolved.maturity,
            )
        else:
            conditional = black76_call(
                conditional_forward,
                resolved.strike,
                volatility,
                resolved.maturity,
            )
        total += weight * float(conditional)
        cumulative += weight
        if n >= arrival and 1.0 - cumulative < tail_tolerance:
            break
        weight *= arrival / float(n + 1) if arrival > 0.0 else 0.0

    return float(resolved.notional * df * total)


def price_merton_jump_diffusion_option_transform_result(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> MertonJumpDiffusionOptionTransformResult:
    """Return a structured FFT/COS Merton vanilla option price."""
    resolved = resolve_merton_jump_diffusion_option_inputs(market_state, spec)
    normalized_method = _normalized_transform_method(
        method if method is not None else getattr(spec, "transform_method", "fft")
    )
    if resolved.maturity <= 0.0:
        price = float(
            resolved.notional
            * terminal_intrinsic(
                resolved.option_type,
                spot=resolved.spot,
                strike=resolved.strike,
            )
        )
        return MertonJumpDiffusionOptionTransformResult(
            price=price,
            method=normalized_method,
            maturity=resolved.maturity,
        )

    if normalized_method == "fft":
        call_price = fft_price(
            merton_log_spot_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            alpha=float(fft_alpha if fft_alpha is not None else getattr(spec, "fft_alpha", 1.5)),
            N=max(int(fft_points if fft_points is not None else getattr(spec, "fft_points", 4096)), 32),
            eta=float(fft_eta if fft_eta is not None else getattr(spec, "fft_eta", 0.25)),
        )
        if resolved.option_type == "put":
            discounted_strike = resolved.strike * raw_np.exp(-resolved.rate * resolved.maturity)
            discounted_spot = resolved.spot * raw_np.exp(-resolved.dividend_yield * resolved.maturity)
            raw_price = call_price - discounted_spot + discounted_strike
        else:
            raw_price = call_price
    else:
        raw_price = cos_price(
            merton_log_ratio_char_fn(resolved),
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=max(int(cos_points if cos_points is not None else getattr(spec, "cos_points", 1024)), 16),
            L=float(cos_truncation if cos_truncation is not None else getattr(spec, "cos_truncation", 14.0)),
            option_type=resolved.option_type,
        )

    return MertonJumpDiffusionOptionTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=normalized_method,
        maturity=resolved.maturity,
    )


def price_merton_jump_diffusion_option_transform(
    market_state,
    spec,
    *,
    method: str | None = None,
    fft_alpha: float | None = None,
    fft_points: int | None = None,
    fft_eta: float | None = None,
    cos_points: int | None = None,
    cos_truncation: float | None = None,
) -> float:
    """Return a scalar FFT/COS Merton vanilla option price."""
    return float(
        price_merton_jump_diffusion_option_transform_result(
            market_state,
            spec,
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


def price_merton_jump_diffusion_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> MertonJumpDiffusionOptionMonteCarloResult:
    """Price a Merton vanilla option by direct terminal distribution sampling."""
    resolved = resolve_merton_jump_diffusion_option_inputs(market_state, spec)
    resolved_n_paths = max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 100_000)), 1)
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    if resolved.maturity <= 0.0:
        intrinsic = terminal_intrinsic(
            resolved.option_type,
            spot=resolved.spot,
            strike=resolved.strike,
        )
        return MertonJumpDiffusionOptionMonteCarloResult(
            price=float(resolved.notional) * float(intrinsic),
            standard_error=0.0,
            n_paths=0,
            seed=resolved_seed,
        )

    rng = raw_np.random.default_rng(resolved_seed)
    arrivals = rng.poisson(resolved.jump_intensity * resolved.maturity, size=resolved_n_paths)
    diffusion_shock = rng.standard_normal(resolved_n_paths)
    jump_shock = rng.standard_normal(resolved_n_paths)
    log_terminal = (
        raw_np.log(resolved.spot)
        + (
            resolved.rate
            - resolved.dividend_yield
            - resolved.jump_intensity * resolved.jump_compensator
            - 0.5 * resolved.sigma**2
        )
        * resolved.maturity
        + resolved.sigma * raw_np.sqrt(resolved.maturity) * diffusion_shock
        + arrivals * resolved.jump_mean
        + raw_np.sqrt(arrivals) * resolved.jump_vol * jump_shock
    )
    terminal = raw_np.exp(log_terminal)
    payoffs = resolved.notional * terminal_intrinsic(
        resolved.option_type,
        spot=terminal,
        strike=resolved.strike,
    )
    discounted = raw_np.exp(-resolved.rate * resolved.maturity) * payoffs
    return MertonJumpDiffusionOptionMonteCarloResult(
        price=float(raw_np.mean(discounted)),
        standard_error=float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(resolved_n_paths)),
        n_paths=resolved_n_paths,
        seed=resolved_seed,
    )


def price_merton_jump_diffusion_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """Return a scalar terminal Monte Carlo Merton vanilla option price."""
    return float(
        price_merton_jump_diffusion_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            seed=seed,
        ).price
    )


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
    if "merton_equity" in sets:
        return sets["merton_equity"]
    if len(sets) == 1:
        return next(iter(sets.values()))
    raise ValueError("Merton jump-diffusion pricing requires jump_parameters")


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


def _normalized_transform_method(value: object) -> str:
    method = str(value or "fft").strip().lower().replace("-", "_")
    aliases = {
        "carr_madan": "fft",
        "fang_oosterlee": "cos",
        "fourier_cosine": "cos",
    }
    method = aliases.get(method, method)
    if method not in {"fft", "cos"}:
        raise ValueError(f"Unsupported Merton transform method {value!r}")
    return method


__all__ = [
    "MertonJumpDiffusionOptionMonteCarloResult",
    "MertonJumpDiffusionOptionTransformResult",
    "ResolvedMertonJumpDiffusionOptionInputs",
    "merton_log_ratio_char_fn",
    "merton_log_spot_char_fn",
    "price_merton_jump_diffusion_option_monte_carlo",
    "price_merton_jump_diffusion_option_monte_carlo_result",
    "price_merton_jump_diffusion_option_poisson_series",
    "price_merton_jump_diffusion_option_transform",
    "price_merton_jump_diffusion_option_transform_result",
    "resolve_merton_jump_diffusion_option_inputs",
]
