"""Shared resolver helpers for bounded single-state diffusion products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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


class SingleStateDiffusionSpecLike(Protocol):
    """Minimal semantic spec surface consumed by the shared resolver."""

    notional: float
    spot: float
    strike: float
    expiry_date: date


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


def resolve_single_state_diffusion_inputs(
    market_state: SingleStateDiffusionMarketStateLike,
    spec: SingleStateDiffusionSpecLike,
) -> ResolvedSingleStateDiffusionInputs:
    """Resolve spot/rate/dividend/vol inputs for a single-state diffusion product."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for pricing")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = max(float(year_fraction(settlement, spec.expiry_date, day_count)), 0.0)
    strike = float(spec.strike)
    option_type = normalized_option_type(getattr(spec, "option_type", "call"))

    if maturity <= 0.0:
        rate = 0.0
        sigma = 0.0
    else:
        if market_state.discount is None:
            raise ValueError("single-state diffusion pricing requires market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("single-state diffusion pricing requires market_state.vol_surface")
        rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
        sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
        if sigma < 0.0:
            raise ValueError(f"Invalid Black volatility at T={maturity}, K={strike}: {sigma}")

    return ResolvedSingleStateDiffusionInputs(
        notional=float(spec.notional),
        spot=float(spec.spot),
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=float(getattr(spec, "dividend_yield", 0.0) or 0.0),
        sigma=sigma,
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
    "ResolvedSingleStateDiffusionInputs",
    "SingleStateDiffusionMarketStateLike",
    "SingleStateDiffusionSpecLike",
    "VolSurfaceLike",
    "gbm_log_ratio_char_fn",
    "gbm_log_spot_char_fn",
    "put_from_call_parity",
    "resolve_single_state_diffusion_inputs",
    "terminal_intrinsic_from_resolved",
]
