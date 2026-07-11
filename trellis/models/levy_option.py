"""Levy-process helpers for European vanilla equity options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as raw_np
from scipy.special import gamma as gamma_function

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price


@dataclass(frozen=True)
class ResolvedLevyOptionInputs:
    """Resolved contract, market, and Levy model parameters."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    model_family: str
    parameters: Mapping[str, float]


@dataclass(frozen=True)
class LevyOptionTransformResult:
    """Structured transform result for a Levy vanilla option."""

    price: float
    method: str
    model_family: str
    maturity: float


@dataclass(frozen=True)
class LevyOptionMonteCarloResult:
    """Structured Monte Carlo result for a Levy vanilla option."""

    price: float
    standard_error: float
    n_paths: int
    seed: int | None
    model_family: str
    sampler: str


def resolve_levy_option_inputs(
    market_state,
    spec,
    *,
    model_family: str | None = None,
) -> ResolvedLevyOptionInputs:
    """Resolve a European vanilla option under a supported Levy model."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("Levy option pricing requires market_state.settlement or as_of")
    if market_state.discount is None:
        raise ValueError("Levy option pricing requires market_state.discount")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    requested_family = _normalize_family(model_family or getattr(spec, "model_family", None))
    payload = _resolve_model_parameter_payload(market_state, spec, requested_family)
    resolved_family = _infer_family_from_payload(payload, requested_family)
    parameters = _canonical_parameters(payload, resolved_family)

    return ResolvedLevyOptionInputs(
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
        spot=_resolve_spot(market_state, spec),
        strike=float(getattr(spec, "strike")),
        maturity=maturity,
        rate=0.0 if maturity <= 0.0 else float(market_state.discount.zero_rate(max(maturity, 1e-6))),
        dividend_yield=float(getattr(spec, "dividend_yield", 0.0) or 0.0),
        option_type=normalized_option_type(getattr(spec, "option_type", "call")),
        model_family=resolved_family,
        parameters=parameters,
    )


def price_variance_gamma_option_transform(
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
    """Return a scalar FFT/COS Variance Gamma vanilla option price."""
    return float(
        _price_levy_option_transform_result(
            market_state,
            spec,
            model_family="variance_gamma",
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


def price_variance_gamma_option_reference(market_state, spec) -> float:
    """Return a high-resolution COS reference for a Variance Gamma option."""
    return price_variance_gamma_option_transform(
        market_state,
        spec,
        method="cos",
        cos_points=4096,
        cos_truncation=18.0,
    )


def price_variance_gamma_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> LevyOptionMonteCarloResult:
    """Price a Variance Gamma option by direct gamma-subordination sampling."""
    resolved = resolve_levy_option_inputs(
        market_state,
        spec,
        model_family="variance_gamma",
    )
    resolved_n_paths = max(
        int(n_paths if n_paths is not None else getattr(spec, "n_paths", 120_000)),
        1,
    )
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    if resolved.maturity <= 0.0:
        return _intrinsic_mc_result(resolved, resolved_seed, "variance_gamma_gamma_subordination")

    sigma = resolved.parameters["sigma"]
    theta = resolved.parameters["theta"]
    nu = resolved.parameters["nu"]
    omega = _variance_gamma_martingale_adjustment(resolved)
    rng = raw_np.random.default_rng(resolved_seed)
    gamma_time = rng.gamma(
        shape=max(resolved.maturity / nu, 1e-12),
        scale=nu,
        size=resolved_n_paths,
    )
    shocks = rng.standard_normal(resolved_n_paths)
    log_terminal = (
        raw_np.log(resolved.spot)
        + (resolved.rate - resolved.dividend_yield + omega) * resolved.maturity
        + theta * gamma_time
        + sigma * raw_np.sqrt(gamma_time) * shocks
    )
    return _discounted_terminal_mc_result(
        resolved,
        raw_np.exp(log_terminal),
        resolved_n_paths,
        resolved_seed,
        "variance_gamma_gamma_subordination",
    )


def price_variance_gamma_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """Return a scalar Variance Gamma terminal Monte Carlo price."""
    return float(
        price_variance_gamma_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            seed=seed,
        ).price
    )


def price_cgmy_option_transform(
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
    """Return a scalar FFT/COS CGMY vanilla option price."""
    return float(
        _price_levy_option_transform_result(
            market_state,
            spec,
            model_family="cgmy",
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


def price_cgmy_option_reference(market_state, spec) -> float:
    """Return a high-resolution COS reference for a CGMY option."""
    return price_cgmy_option_transform(
        market_state,
        spec,
        method="cos",
        cos_points=4096,
        cos_truncation=18.0,
    )


def price_cgmy_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> LevyOptionMonteCarloResult:
    """Price a CGMY option by terminal-distribution Monte Carlo from its CF."""
    resolved = resolve_levy_option_inputs(market_state, spec, model_family="cgmy")
    resolved_n_paths = max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 120_000)), 1)
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    if resolved.maturity <= 0.0:
        return _intrinsic_mc_result(resolved, resolved_seed, "cgmy_cf_inversion_terminal")

    rng = raw_np.random.default_rng(resolved_seed)
    returns_grid, cdf = _terminal_return_cdf_from_cf(
        _cgmy_log_ratio_char_fn(resolved),
        grid_size=max(int(getattr(spec, "cdf_grid_size", 2048)), 512),
        integration_points=max(int(getattr(spec, "cdf_integration_points", 2048)), 512),
        width=float(getattr(spec, "cdf_width", 3.0)),
        frequency_cutoff=float(getattr(spec, "cdf_frequency_cutoff", 160.0)),
    )
    samples = raw_np.interp(rng.random(resolved_n_paths), cdf, returns_grid)
    terminal = resolved.spot * raw_np.exp(samples)
    return _discounted_terminal_mc_result(
        resolved,
        terminal,
        resolved_n_paths,
        resolved_seed,
        "cgmy_cf_inversion_terminal",
    )


def price_cgmy_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """Return a scalar CGMY terminal-distribution Monte Carlo price."""
    return float(
        price_cgmy_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            seed=seed,
        ).price
    )


def price_kou_option_transform(
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
    """Return a scalar FFT/COS Kou double-exponential vanilla option price."""
    return float(
        _price_levy_option_transform_result(
            market_state,
            spec,
            model_family="kou",
            method=method,
            fft_alpha=fft_alpha,
            fft_points=fft_points,
            fft_eta=fft_eta,
            cos_points=cos_points,
            cos_truncation=cos_truncation,
        ).price
    )


def price_kou_option_reference(market_state, spec) -> float:
    """Return a high-resolution COS reference for a Kou vanilla option."""
    return price_kou_option_transform(
        market_state,
        spec,
        method="cos",
        cos_points=4096,
        cos_truncation=18.0,
    )


def price_kou_option_monte_carlo_result(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> LevyOptionMonteCarloResult:
    """Price a Kou option by direct terminal double-exponential sampling."""
    resolved = resolve_levy_option_inputs(market_state, spec, model_family="kou")
    resolved_n_paths = max(int(n_paths if n_paths is not None else getattr(spec, "n_paths", 120_000)), 1)
    resolved_seed = seed if seed is not None else getattr(spec, "seed", 42)
    if resolved.maturity <= 0.0:
        return _intrinsic_mc_result(
            resolved,
            resolved_seed,
            "kou_double_exponential_terminal",
        )

    sigma = resolved.parameters["sigma"]
    jump_intensity = resolved.parameters["jump_intensity"]
    up_probability = resolved.parameters["up_probability"]
    eta_up = resolved.parameters["eta_up"]
    eta_down = resolved.parameters["eta_down"]
    jump_compensator = _kou_jump_compensator(resolved)

    rng = raw_np.random.default_rng(resolved_seed)
    arrivals = rng.poisson(jump_intensity * resolved.maturity, size=resolved_n_paths)
    up_counts = rng.binomial(arrivals, up_probability)
    down_counts = arrivals - up_counts

    up_jump_sum = raw_np.zeros(resolved_n_paths)
    up_mask = up_counts > 0
    if raw_np.any(up_mask):
        up_jump_sum[up_mask] = rng.gamma(
            shape=up_counts[up_mask].astype(float),
            scale=1.0 / eta_up,
        )

    down_jump_sum = raw_np.zeros(resolved_n_paths)
    down_mask = down_counts > 0
    if raw_np.any(down_mask):
        down_jump_sum[down_mask] = rng.gamma(
            shape=down_counts[down_mask].astype(float),
            scale=1.0 / eta_down,
        )

    diffusion_shock = rng.standard_normal(resolved_n_paths)
    log_terminal = (
        raw_np.log(resolved.spot)
        + (
            resolved.rate
            - resolved.dividend_yield
            - jump_intensity * jump_compensator
            - 0.5 * sigma**2
        )
        * resolved.maturity
        + sigma * raw_np.sqrt(resolved.maturity) * diffusion_shock
        + up_jump_sum
        - down_jump_sum
    )
    return _discounted_terminal_mc_result(
        resolved,
        raw_np.exp(log_terminal),
        resolved_n_paths,
        resolved_seed,
        "kou_double_exponential_terminal",
    )


def price_kou_option_monte_carlo(
    market_state,
    spec,
    *,
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """Return a scalar terminal Monte Carlo Kou vanilla option price."""
    return float(
        price_kou_option_monte_carlo_result(
            market_state,
            spec,
            n_paths=n_paths,
            seed=seed,
        ).price
    )


def variance_gamma_log_ratio_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T/S_0))]`` for Variance Gamma."""
    sigma = resolved.parameters["sigma"]
    theta = resolved.parameters["theta"]
    nu = resolved.parameters["nu"]
    omega = _variance_gamma_martingale_adjustment(resolved)
    drift = (resolved.rate - resolved.dividend_yield + omega) * resolved.maturity

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        vg_part = (1.0 - 1j * theta * nu * u_arr + 0.5 * sigma**2 * nu * u_arr**2) ** (
            -resolved.maturity / nu
        )
        return raw_np.exp(1j * u_arr * drift) * vg_part

    return phi


def variance_gamma_log_spot_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T))]`` for Variance Gamma."""
    log_spot = raw_np.log(resolved.spot)
    ratio_cf = variance_gamma_log_ratio_char_fn(resolved)

    def phi(u):
        return raw_np.exp(1j * raw_np.asarray(u, dtype=complex) * log_spot) * ratio_cf(u)

    return phi


def cgmy_log_ratio_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T/S_0))]`` for CGMY."""
    return _cgmy_log_ratio_char_fn(resolved)


def cgmy_log_spot_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T))]`` for CGMY."""
    log_spot = raw_np.log(resolved.spot)
    ratio_cf = _cgmy_log_ratio_char_fn(resolved)

    def phi(u):
        return raw_np.exp(1j * raw_np.asarray(u, dtype=complex) * log_spot) * ratio_cf(u)

    return phi


def kou_log_ratio_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T/S_0))]`` for Kou double-exponential jumps."""
    sigma = resolved.parameters["sigma"]
    jump_intensity = resolved.parameters["jump_intensity"]
    up_probability = resolved.parameters["up_probability"]
    eta_up = resolved.parameters["eta_up"]
    eta_down = resolved.parameters["eta_down"]
    jump_compensator = _kou_jump_compensator(resolved)
    drift = (
        resolved.rate
        - resolved.dividend_yield
        - jump_intensity * jump_compensator
        - 0.5 * sigma**2
    ) * resolved.maturity
    diffusion_variance = sigma**2 * resolved.maturity
    jump_arrival = jump_intensity * resolved.maturity

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        jump_cf = (
            up_probability * eta_up / (eta_up - 1j * u_arr)
            + (1.0 - up_probability) * eta_down / (eta_down + 1j * u_arr)
        )
        return raw_np.exp(
            1j * u_arr * drift
            - 0.5 * diffusion_variance * u_arr**2
            + jump_arrival * (jump_cf - 1.0)
        )

    return phi


def kou_log_spot_char_fn(resolved: ResolvedLevyOptionInputs):
    """Return ``E[exp(iu log(S_T))]`` for Kou double-exponential jumps."""
    log_spot = raw_np.log(resolved.spot)
    ratio_cf = kou_log_ratio_char_fn(resolved)

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        return raw_np.exp(1j * u_arr * log_spot) * ratio_cf(u_arr)

    return phi


def _price_levy_option_transform_result(
    market_state,
    spec,
    *,
    model_family: str,
    method: str | None,
    fft_alpha: float | None,
    fft_points: int | None,
    fft_eta: float | None,
    cos_points: int | None,
    cos_truncation: float | None,
) -> LevyOptionTransformResult:
    resolved = resolve_levy_option_inputs(
        market_state,
        spec,
        model_family=model_family,
    )
    normalized_method = _normalized_transform_method(
        method if method is not None else getattr(spec, "transform_method", "cos")
    )
    if resolved.maturity <= 0.0:
        return LevyOptionTransformResult(
            price=float(
                resolved.notional
                * terminal_intrinsic(
                    resolved.option_type,
                    spot=resolved.spot,
                    strike=resolved.strike,
                )
            ),
            method=normalized_method,
            model_family=resolved.model_family,
            maturity=resolved.maturity,
        )

    if resolved.model_family == "variance_gamma":
        ratio_cf = variance_gamma_log_ratio_char_fn(resolved)
        spot_cf = variance_gamma_log_spot_char_fn(resolved)
    elif resolved.model_family == "cgmy":
        ratio_cf = _cgmy_log_ratio_char_fn(resolved)
        spot_cf = cgmy_log_spot_char_fn(resolved)
    elif resolved.model_family == "kou":
        ratio_cf = kou_log_ratio_char_fn(resolved)
        spot_cf = kou_log_spot_char_fn(resolved)
    else:
        raise ValueError(f"Unsupported Levy model family {resolved.model_family!r}")

    if normalized_method == "fft":
        call_price = fft_price(
            spot_cf,
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
            discounted_spot = resolved.spot * raw_np.exp(
                -resolved.dividend_yield * resolved.maturity
            )
            raw_price = call_price - discounted_spot + discounted_strike
        else:
            raw_price = call_price
    else:
        raw_price = cos_price(
            ratio_cf,
            resolved.spot,
            resolved.strike,
            resolved.maturity,
            resolved.rate,
            N=max(int(cos_points if cos_points is not None else getattr(spec, "cos_points", 1024)), 16),
            L=float(cos_truncation if cos_truncation is not None else getattr(spec, "cos_truncation", 14.0)),
            option_type=resolved.option_type,
        )

    return LevyOptionTransformResult(
        price=float(resolved.notional) * float(raw_price),
        method=normalized_method,
        model_family=resolved.model_family,
        maturity=resolved.maturity,
    )


def _resolve_model_parameter_payload(
    market_state,
    spec,
    requested_family: str,
) -> Mapping[str, object]:
    for explicit in (
        getattr(spec, "model_parameters", None),
        getattr(spec, "levy_parameters", None),
        getattr(spec, "variance_gamma", None),
        getattr(spec, "vg", None),
        getattr(spec, "cgmy", None),
        getattr(spec, "kou", None),
        getattr(spec, "double_exponential_jump", None),
    ):
        extracted = _extract_family_payload(explicit, requested_family)
        if extracted is not None:
            return extracted

    set_name = (
        getattr(spec, "model_parameter_set", None)
        or getattr(spec, "model_parameter_name", None)
        or getattr(spec, "model_parameter_id", None)
        or getattr(spec, "levy_parameter_set", None)
    )
    selected_curve_name = getattr(market_state, "selected_curve_name", None)
    if set_name is None and callable(selected_curve_name):
        set_name = selected_curve_name("model_parameters")
    sets = dict(getattr(market_state, "model_parameter_sets", None) or {})
    if set_name is not None:
        key = str(set_name)
        if key not in sets:
            raise ValueError(f"Unknown Levy model parameter set {key!r}")
        extracted = _extract_family_payload(sets[key], requested_family)
        if extracted is None:
            raise ValueError(f"Model parameter set {key!r} is not a {requested_family} payload")
        return extracted

    for candidate in _levy_payload_candidates(market_state, sets, requested_family):
        extracted = _extract_family_payload(candidate, requested_family)
        if extracted is not None:
            return extracted
    raise ValueError(f"Levy option pricing requires {requested_family} model_parameters")


def _levy_payload_candidates(
    market_state,
    sets: Mapping[str, object],
    requested_family: str,
) -> tuple[object, ...]:
    family_aliases = _family_aliases(requested_family)
    named_sets = tuple(sets.get(alias) for alias in family_aliases)
    validation_sets = tuple(sets.get(f"{alias}_validation") for alias in family_aliases)
    return (
        getattr(market_state, "model_parameters", None),
        *named_sets,
        *validation_sets,
        *_single_item_values(sets),
    )


def _extract_family_payload(
    payload: object,
    requested_family: str,
) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    family = _normalize_family(payload.get("family"))
    if family and requested_family and family != requested_family:
        return None
    if _has_family_keys(payload, requested_family):
        return payload
    for key in _family_aliases(requested_family):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and _has_family_keys(nested, requested_family):
            return nested
    return None


def _infer_family_from_payload(payload: Mapping[str, object], requested_family: str) -> str:
    family = _normalize_family(payload.get("family"))
    if requested_family:
        if family and family != requested_family:
            raise ValueError(f"Expected {requested_family} parameters, got {family}")
        return requested_family
    if family:
        return family
    if _has_family_keys(payload, "variance_gamma"):
        return "variance_gamma"
    if _has_family_keys(payload, "cgmy"):
        return "cgmy"
    if _has_family_keys(payload, "kou"):
        return "kou"
    raise ValueError("Cannot infer Levy model family from model parameter payload")


def _canonical_parameters(payload: Mapping[str, object], family: str) -> dict[str, float]:
    if family == "variance_gamma":
        sigma = _resolve_positive_float(payload, ("sigma", "volatility", "vol"), default=None)
        theta = _resolve_float(payload, ("theta", "drift", "skew"), default=None)
        nu = _resolve_positive_float(payload, ("nu", "variance_rate", "variance_time"), default=None)
        missing = [
            name
            for name, value in (("sigma", sigma), ("theta", theta), ("nu", nu))
            if value is None
        ]
        if missing:
            raise ValueError(f"Variance Gamma payload is missing {', '.join(missing)}")
        if 1.0 - float(theta) * float(nu) - 0.5 * float(sigma) ** 2 * float(nu) <= 0.0:
            raise ValueError("Variance Gamma martingale adjustment is undefined")
        return {"sigma": float(sigma), "theta": float(theta), "nu": float(nu)}
    if family == "cgmy":
        c_value = _resolve_positive_float(payload, ("C", "c"), default=None)
        g_value = _resolve_positive_float(payload, ("G", "g"), default=None)
        m_value = _resolve_positive_float(payload, ("M", "m"), default=None)
        y_value = _resolve_positive_float(payload, ("Y", "y"), default=None)
        missing = [
            name
            for name, value in (("C", c_value), ("G", g_value), ("M", m_value), ("Y", y_value))
            if value is None
        ]
        if missing:
            raise ValueError(f"CGMY payload is missing {', '.join(missing)}")
        if not 0.0 < float(y_value) < 2.0:
            raise ValueError(f"CGMY Y must be in (0, 2), got {y_value}")
        if float(m_value) <= 1.0:
            raise ValueError("CGMY M must exceed 1.0 for the stock-price martingale")
        return {
            "C": float(c_value),
            "G": float(g_value),
            "M": float(m_value),
            "Y": float(y_value),
        }
    if family == "kou":
        sigma = _resolve_positive_float(
            payload,
            ("sigma", "diffusion_sigma", "volatility", "vol"),
            default=None,
        )
        jump_intensity = _resolve_nonnegative_float(
            payload,
            ("jump_intensity", "lambda", "lam", "intensity"),
            default=None,
        )
        up_probability = _resolve_float(
            payload,
            ("up_probability", "p_up", "prob_up", "p"),
            default=None,
        )
        eta_up = _resolve_positive_float(
            payload,
            ("eta_up", "eta1", "eta_plus", "up_eta"),
            default=None,
        )
        eta_down = _resolve_positive_float(
            payload,
            ("eta_down", "eta2", "eta_minus", "down_eta"),
            default=None,
        )
        missing = [
            name
            for name, value in (
                ("sigma", sigma),
                ("jump_intensity", jump_intensity),
                ("up_probability", up_probability),
                ("eta_up", eta_up),
                ("eta_down", eta_down),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Kou payload is missing {', '.join(missing)}")
        if not 0.0 <= float(up_probability) <= 1.0:
            raise ValueError(
                f"Kou up_probability must be in [0, 1], got {up_probability}"
            )
        if float(eta_up) <= 1.0:
            raise ValueError("Kou eta_up must exceed 1.0 for the stock-price martingale")
        return {
            "sigma": float(sigma),
            "jump_intensity": float(jump_intensity),
            "up_probability": float(up_probability),
            "eta_up": float(eta_up),
            "eta_down": float(eta_down),
        }
    raise ValueError(f"Unsupported Levy model family {family!r}")


def _has_family_keys(payload: Mapping[str, object], family: str) -> bool:
    if family == "variance_gamma":
        return (
            _has_any_key(payload, ("sigma", "volatility", "vol"))
            and _has_any_key(payload, ("theta", "drift", "skew"))
            and _has_any_key(payload, ("nu", "variance_rate", "variance_time"))
        )
    if family == "cgmy":
        return (
            _has_any_key(payload, ("C", "c"))
            and _has_any_key(payload, ("G", "g"))
            and _has_any_key(payload, ("M", "m"))
            and _has_any_key(payload, ("Y", "y"))
        )
    if family == "kou":
        return (
            _has_any_key(payload, ("sigma", "diffusion_sigma", "volatility", "vol"))
            and _has_any_key(payload, ("jump_intensity", "lambda", "lam", "intensity"))
            and _has_any_key(payload, ("up_probability", "p_up", "prob_up", "p"))
            and _has_any_key(payload, ("eta_up", "eta1", "eta_plus", "up_eta"))
            and _has_any_key(payload, ("eta_down", "eta2", "eta_minus", "down_eta"))
        )
    return False


def _has_any_key(payload: Mapping[str, object], keys: tuple[str, ...]) -> bool:
    return any(key in payload and payload[key] is not None for key in keys)


def _resolve_spot(market_state, spec) -> float:
    spot = getattr(spec, "spot", None)
    if spot is not None:
        return float(spot)
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
    raise ValueError("Levy option pricing requires spec.spot or market_state.spot")


def _variance_gamma_martingale_adjustment(resolved: ResolvedLevyOptionInputs) -> float:
    sigma = resolved.parameters["sigma"]
    theta = resolved.parameters["theta"]
    nu = resolved.parameters["nu"]
    return float(raw_np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu) / nu)


def _kou_jump_compensator(resolved: ResolvedLevyOptionInputs) -> float:
    up_probability = resolved.parameters["up_probability"]
    eta_up = resolved.parameters["eta_up"]
    eta_down = resolved.parameters["eta_down"]
    return float(
        up_probability * eta_up / (eta_up - 1.0)
        + (1.0 - up_probability) * eta_down / (eta_down + 1.0)
        - 1.0
    )


def _cgmy_log_ratio_char_fn(resolved: ResolvedLevyOptionInputs):
    c_value = resolved.parameters["C"]
    g_value = resolved.parameters["G"]
    m_value = resolved.parameters["M"]
    y_value = resolved.parameters["Y"]

    def psi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        return c_value * gamma_function(-y_value) * (
            (m_value - 1j * u_arr) ** y_value
            - m_value**y_value
            + (g_value + 1j * u_arr) ** y_value
            - g_value**y_value
        )

    omega = -float(raw_np.real(psi(-1j)))
    drift = (resolved.rate - resolved.dividend_yield + omega) * resolved.maturity

    def phi(u):
        u_arr = raw_np.asarray(u, dtype=complex)
        return raw_np.exp(1j * u_arr * drift + resolved.maturity * psi(u_arr))

    return phi


def _terminal_return_cdf_from_cf(
    char_fn,
    *,
    grid_size: int,
    integration_points: int,
    width: float,
    frequency_cutoff: float,
) -> tuple[raw_np.ndarray, raw_np.ndarray]:
    x_grid = raw_np.linspace(-abs(width), abs(width), grid_size)
    u_grid = raw_np.linspace(0.0, abs(frequency_cutoff), integration_points)
    weights = raw_np.ones_like(u_grid)
    weights[0] = 0.5
    weights[-1] = 0.5
    cf_vals = char_fn(u_grid) * weights
    pdf = raw_np.empty_like(x_grid)
    chunk_size = 256
    for start in range(0, len(x_grid), chunk_size):
        stop = min(start + chunk_size, len(x_grid))
        kernel = raw_np.exp(-1j * x_grid[start:stop, None] * u_grid[None, :])
        pdf[start:stop] = raw_np.real(kernel @ cf_vals)
    pdf *= (u_grid[1] - u_grid[0]) / raw_np.pi
    pdf = raw_np.maximum(pdf, 0.0)
    area = float(raw_np.trapezoid(pdf, x_grid))
    if area <= 0.0 or not raw_np.isfinite(area):
        raise ValueError("CGMY characteristic-function inversion produced no valid mass")
    pdf /= area
    increments = 0.5 * (pdf[1:] + pdf[:-1]) * raw_np.diff(x_grid)
    cdf = raw_np.concatenate(([0.0], raw_np.cumsum(increments)))
    cdf /= cdf[-1]
    cdf = raw_np.maximum.accumulate(cdf)
    cdf[-1] = 1.0
    return x_grid, cdf


def _discounted_terminal_mc_result(
    resolved: ResolvedLevyOptionInputs,
    terminal,
    n_paths: int,
    seed: int | None,
    sampler: str,
) -> LevyOptionMonteCarloResult:
    payoffs = resolved.notional * terminal_intrinsic(
        resolved.option_type,
        spot=terminal,
        strike=resolved.strike,
    )
    discounted = raw_np.exp(-resolved.rate * resolved.maturity) * payoffs
    return LevyOptionMonteCarloResult(
        price=float(raw_np.mean(discounted)),
        standard_error=float(raw_np.std(discounted, ddof=1) / raw_np.sqrt(n_paths)),
        n_paths=n_paths,
        seed=seed,
        model_family=resolved.model_family,
        sampler=sampler,
    )


def _intrinsic_mc_result(
    resolved: ResolvedLevyOptionInputs,
    seed: int | None,
    sampler: str,
) -> LevyOptionMonteCarloResult:
    intrinsic = terminal_intrinsic(
        resolved.option_type,
        spot=resolved.spot,
        strike=resolved.strike,
    )
    return LevyOptionMonteCarloResult(
        price=float(resolved.notional * intrinsic),
        standard_error=0.0,
        n_paths=0,
        seed=seed,
        model_family=resolved.model_family,
        sampler=sampler,
    )


def _normalized_transform_method(value: object) -> str:
    method = str(value or "cos").strip().lower().replace("-", "_")
    aliases = {
        "carr_madan": "fft",
        "madan_carr_chang": "fft",
        "fang_oosterlee": "cos",
        "fourier_cosine": "cos",
    }
    method = aliases.get(method, method)
    if method not in {"fft", "cos"}:
        raise ValueError(f"Unsupported Levy transform method {value!r}")
    return method


def _normalize_family(value: object) -> str:
    family = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "vg": "variance_gamma",
        "variancegamma": "variance_gamma",
        "tempered_stable": "cgmy",
        "double_exponential": "kou",
        "double_exponential_jump": "kou",
        "double_exponential_jump_diffusion": "kou",
        "kou_jump": "kou",
        "kou_process": "kou",
    }
    return aliases.get(family, family)


def _family_aliases(family: str) -> tuple[str, ...]:
    if family == "variance_gamma":
        return ("variance_gamma", "vg", "variance_gamma_equity")
    if family == "cgmy":
        return ("cgmy", "tempered_stable", "cgmy_equity")
    if family == "kou":
        return (
            "kou",
            "kou_equity",
            "double_exponential",
            "double_exponential_jump",
            "double_exponential_jump_diffusion",
        )
    return (family,)


def _single_item_values(mapping: Mapping[str, object]) -> tuple[object, ...]:
    if len(mapping) == 1:
        return (next(iter(mapping.values())),)
    return ()


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
    if value is not None and value <= 0.0:
        raise ValueError(f"Levy parameter {names[0]} must be positive")
    return value


def _resolve_nonnegative_float(
    payload: Mapping[str, object],
    names: tuple[str, ...],
    *,
    default: float | None,
) -> float | None:
    value = _resolve_float(payload, names, default=default)
    if value is not None and value < 0.0:
        raise ValueError(f"Levy parameter {names[0]} must be non-negative")
    return value


__all__ = [
    "LevyOptionMonteCarloResult",
    "LevyOptionTransformResult",
    "ResolvedLevyOptionInputs",
    "cgmy_log_ratio_char_fn",
    "cgmy_log_spot_char_fn",
    "kou_log_ratio_char_fn",
    "kou_log_spot_char_fn",
    "price_cgmy_option_monte_carlo",
    "price_cgmy_option_monte_carlo_result",
    "price_cgmy_option_reference",
    "price_cgmy_option_transform",
    "price_kou_option_monte_carlo",
    "price_kou_option_monte_carlo_result",
    "price_kou_option_reference",
    "price_kou_option_transform",
    "price_variance_gamma_option_monte_carlo",
    "price_variance_gamma_option_monte_carlo_result",
    "price_variance_gamma_option_reference",
    "price_variance_gamma_option_transform",
    "resolve_levy_option_inputs",
    "variance_gamma_log_ratio_char_fn",
    "variance_gamma_log_spot_char_fn",
]
