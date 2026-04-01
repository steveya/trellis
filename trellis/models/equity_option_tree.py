"""Stable vanilla-equity tree helpers on the generalized lattice substrate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.trees.control import resolve_lattice_exercise_policy
from trellis.models.trees.lattice import build_spot_lattice, lattice_backward_induction


class DiscountCurveLike(Protocol):
    """Discount interface required by the equity tree helpers."""

    def zero_rate(self, t: float) -> float:
        """Return a zero rate."""
        ...


class VolSurfaceLike(Protocol):
    """Volatility interface required by the equity tree helpers."""

    def black_vol(self, t: float, strike: float) -> float:
        """Return a Black-style volatility quote."""
        ...


class EquityTreeMarketStateLike(Protocol):
    """Market-state interface required by the equity tree helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    vol_surface: VolSurfaceLike | None


class VanillaEquityOptionSpecLike(Protocol):
    """Spec fields consumed by the equity tree helpers."""

    spot: float
    strike: float
    expiry_date: date


@dataclass(frozen=True)
class ResolvedEquityTreeInputs:
    """Resolved market inputs for a vanilla equity tree."""

    spot: float
    strike: float
    maturity: float
    rate: float
    sigma: float
    option_type: str
    exercise_style: str


def build_vanilla_equity_lattice(
    *,
    spot: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_steps: int = 500,
    model: str = "crr",
):
    """Build a one-factor equity lattice on the shared lattice substrate."""
    return build_spot_lattice(
        spot,
        rate,
        sigma,
        maturity,
        n_steps,
        model=model,
    )


def price_vanilla_equity_option_on_lattice(
    lattice,
    *,
    strike: float,
    option_type: str = "call",
    exercise_style: str = "european",
):
    """Price a vanilla equity option on a pre-built lattice."""
    option_kind = str(option_type).strip().lower()
    exercise_kind = str(exercise_style).strip().lower()
    if option_kind not in {"call", "put"}:
        raise ValueError(f"Unsupported option_type {option_type!r}")
    if exercise_kind not in {"european", "american", "bermudan"}:
        raise ValueError(f"Unsupported exercise_style {exercise_style!r}")

    def payoff(step, node, lat):
        spot = float(lat.get_state(step, node))
        if option_kind == "call":
            return max(spot - float(strike), 0.0)
        return max(float(strike) - spot, 0.0)

    if exercise_kind == "european":
        return float(lattice_backward_induction(lattice, payoff))

    exercise_policy = resolve_lattice_exercise_policy(exercise_kind)
    return float(
        lattice_backward_induction(
            lattice,
            payoff,
            exercise_value=payoff,
            exercise_policy=exercise_policy,
        )
    )


def price_vanilla_equity_option_tree(
    market_state: EquityTreeMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
    *,
    model: str = "crr",
    n_steps: int = 500,
) -> float:
    """Price a vanilla equity option via the shared lattice helper surface."""
    resolved = resolve_vanilla_equity_tree_inputs(market_state, spec)
    if resolved.maturity <= 0.0:
        return 0.0

    lattice = build_vanilla_equity_lattice(
        spot=resolved.spot,
        rate=resolved.rate,
        sigma=resolved.sigma,
        maturity=resolved.maturity,
        n_steps=n_steps,
        model=model,
    )
    return price_vanilla_equity_option_on_lattice(
        lattice,
        strike=resolved.strike,
        option_type=resolved.option_type,
        exercise_style=resolved.exercise_style,
    )


def resolve_vanilla_equity_tree_inputs(
    market_state: EquityTreeMarketStateLike,
    spec: VanillaEquityOptionSpecLike,
) -> ResolvedEquityTreeInputs:
    """Resolve settlement, maturity, rate, and vol for the tree helper."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for tree pricing")
    if market_state.discount is None:
        raise ValueError("equity tree pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("equity tree pricing requires market_state.vol_surface")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = float(year_fraction(settlement, spec.expiry_date, day_count))
    strike = float(spec.strike)
    return ResolvedEquityTreeInputs(
        spot=float(spec.spot),
        strike=strike,
        maturity=maturity,
        rate=float(market_state.discount.zero_rate(max(maturity, 1e-6))),
        sigma=float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike)),
        option_type=str(getattr(spec, "option_type", "call")).strip().lower(),
        exercise_style=str(getattr(spec, "exercise_style", "european")).strip().lower(),
    )


__all__ = [
    "ResolvedEquityTreeInputs",
    "build_vanilla_equity_lattice",
    "price_vanilla_equity_option_on_lattice",
    "price_vanilla_equity_option_tree",
    "resolve_vanilla_equity_tree_inputs",
]
