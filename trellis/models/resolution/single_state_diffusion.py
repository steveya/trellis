"""Shared resolver helpers for bounded single-state diffusion products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic

np = get_numpy()


class DiscountCurveLike(Protocol):
    """Discount interface required by the single-state diffusion resolver."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...

    def zero_rate(self, t: float) -> float:
        """Return a zero rate to time ``t``."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the single-state diffusion resolver."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class SingleStateDiffusionMarketStateLike(Protocol):
    """Minimal market-state interface required by the shared resolver."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None
    spot: float | None
    model_parameters: dict[str, object] | None


class ScalarDiffusionMarketSpecLike(Protocol):
    """Minimal product-neutral scalar diffusion market coordinates."""

    spot: float
    expiry_date: date


class SingleStateDiffusionSpecLike(Protocol):
    """Minimal semantic spec surface consumed by the shared resolver."""

    notional: float
    spot: float
    strike: float
    expiry_date: date


@dataclass(frozen=True)
class ResolvedScalarDiffusionMarketInputs:
    """Resolved scalar market inputs independent of derivative settlement."""

    spot: float
    maturity: float
    rate: float
    dividend_yield: float
    sigma: float
    discount_factor: float
    volatility_coordinate: float


@dataclass(frozen=True)
class ResolvedSingleStateDiffusionInputs:
    """Resolved scalar inputs shared by bounded single-state diffusion families."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    sigma: float
    option_type: str


def _resolve_scalar_diffusion_carry(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: ScalarDiffusionMarketSpecLike,
) -> float:
    for field_name in ("dividend_yield", "dividend_rate"):
        value = getattr(spec, field_name, None)
        if value is not None:
            resolved = float(value)
            if not isfinite(resolved):
                raise ValueError(f"{field_name} must be finite")
            return resolved

    parameters = dict(getattr(market_state, "model_parameters", None) or {})
    carry_rates = dict(parameters.get("underlier_carry_rates") or {})
    if not carry_rates:
        return 0.0

    underlier = next(
        (
            str(value).strip()
            for value in (
                getattr(spec, "underlier", None),
                getattr(spec, "underlier_id", None),
                getattr(spec, "ticker", None),
            )
            if value is not None and str(value).strip()
        ),
        "",
    )
    if underlier:
        if underlier not in carry_rates:
            raise ValueError(
                f"underlier carry rate {underlier!r} is not available"
            )
        resolved = float(carry_rates[underlier])
    elif len(carry_rates) == 1:
        resolved = float(next(iter(carry_rates.values())))
    else:
        raise ValueError(
            "multiple underlier carry rates are available; specify an underlier"
        )
    if not isfinite(resolved):
        raise ValueError("resolved underlier carry rate must be finite")
    return resolved


def resolve_scalar_diffusion_market_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: ScalarDiffusionMarketSpecLike,
    *,
    volatility_coordinate: float | None = None,
) -> ResolvedScalarDiffusionMarketInputs:
    """Resolve product-neutral scalar diffusion market coordinates.

    ``volatility_coordinate`` identifies the exact scalar surface coordinate
    selected by caller-owned product semantics. It defaults to spot and is
    recorded in the result rather than inferred again downstream.
    """
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for pricing")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    spot_value = getattr(spec, "spot", None)
    if spot_value is None:
        spot_value = getattr(market_state, "spot", None)
    if spot_value is None:
        raise ValueError("scalar diffusion pricing requires an exact spot binding")
    spot = float(spot_value)
    if not isfinite(spot):
        raise ValueError("scalar diffusion spot must be finite")

    coordinate = spot if volatility_coordinate is None else float(volatility_coordinate)
    if not isfinite(coordinate):
        raise ValueError("volatility_coordinate must be finite")
    dividend_yield = _resolve_scalar_diffusion_carry(market_state, spec)

    if maturity <= 0.0:
        rate = 0.0
        sigma = 0.0
        discount_factor = 1.0
    else:
        if market_state.discount is None:
            raise ValueError("scalar diffusion pricing requires market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("scalar diffusion pricing requires market_state.vol_surface")
        rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
        discount_factor = float(market_state.discount.discount(maturity))
        sigma = float(
            market_state.vol_surface.black_vol(
                max(maturity, 1e-6),
                coordinate,
            )
        )
        if not isfinite(rate) or not isfinite(discount_factor):
            raise ValueError("resolved discount inputs must be finite")
        if discount_factor <= 0.0:
            raise ValueError("resolved discount factor must be positive")
        if not isfinite(sigma) or sigma < 0.0:
            raise ValueError(
                f"Invalid Black volatility at T={maturity}, coordinate={coordinate}: {sigma}"
            )

    return ResolvedScalarDiffusionMarketInputs(
        spot=spot,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        sigma=sigma,
        discount_factor=discount_factor,
        volatility_coordinate=coordinate,
    )


def resolve_single_state_diffusion_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
) -> ResolvedSingleStateDiffusionInputs:
    """Resolve spot/rate/dividend/vol inputs for a single-state diffusion product."""
    strike = float(spec.strike)
    option_type = normalized_option_type(getattr(spec, "option_type", "call"))
    market_inputs = resolve_scalar_diffusion_market_inputs(
        market_state,
        spec,
        volatility_coordinate=strike,
    )

    return ResolvedSingleStateDiffusionInputs(
        notional=float(spec.notional),
        spot=market_inputs.spot,
        strike=strike,
        maturity=market_inputs.maturity,
        rate=market_inputs.rate,
        dividend_yield=market_inputs.dividend_yield,
        sigma=market_inputs.sigma,
        option_type=option_type,
    )


def gbm_log_spot_char_fn(resolved: ResolvedSingleStateDiffusionInputs):
    """Return the GBM characteristic function for ``log(S_T)``."""
    drift = np.log(resolved.spot) + (
        resolved.rate - resolved.dividend_yield - 0.5 * resolved.sigma**2
    ) * resolved.maturity
    variance = resolved.sigma**2 * resolved.maturity

    def phi(u):
        return np.exp(1j * u * drift - 0.5 * variance * u**2)

    return phi


def gbm_log_ratio_char_fn(resolved: ResolvedSingleStateDiffusionInputs):
    """Return the GBM characteristic function for ``log(S_T / S_0)``."""
    drift = (
        resolved.rate - resolved.dividend_yield - 0.5 * resolved.sigma**2
    ) * resolved.maturity
    variance = resolved.sigma**2 * resolved.maturity

    def phi(u):
        return np.exp(1j * u * drift - 0.5 * variance * u**2)

    return phi


def put_from_call_parity(
    call_price: float,
    resolved: ResolvedSingleStateDiffusionInputs,
) -> float:
    """Return the put value implied by the call price under put-call parity."""
    discounted_strike = resolved.strike * np.exp(-resolved.rate * resolved.maturity)
    discounted_spot = resolved.spot * np.exp(-resolved.dividend_yield * resolved.maturity)
    return float(call_price - discounted_spot + discounted_strike)


def terminal_intrinsic_from_resolved(
    spot: float,
    resolved: ResolvedSingleStateDiffusionInputs,
):
    """Return terminal intrinsic value using the resolved option semantics."""
    return terminal_intrinsic(
        resolved.option_type,
        spot=spot,
        strike=resolved.strike,
    )


__all__ = [
    "DiscountCurveLike",
    "ResolvedScalarDiffusionMarketInputs",
    "ResolvedSingleStateDiffusionInputs",
    "ScalarDiffusionMarketSpecLike",
    "SingleStateDiffusionMarketStateLike",
    "SingleStateDiffusionSpecLike",
    "VolSurfaceLike",
    "gbm_log_ratio_char_fn",
    "gbm_log_spot_char_fn",
    "put_from_call_parity",
    "resolve_scalar_diffusion_market_inputs",
    "resolve_single_state_diffusion_inputs",
    "terminal_intrinsic_from_resolved",
]
