"""Stable vanilla-equity tree helpers on the generalized lattice substrate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import trellis.models.trees.algebra as lattice_algebra

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention


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
    return lattice_algebra.build_lattice(
        lattice_algebra.BINOMIAL_1F_TOPOLOGY,
        lattice_algebra.LOG_SPOT_MESH,
        lattice_algebra.LATTICE_MODEL_REGISTRY[str(model).strip().lower()],
        calibration_target=lattice_algebra.NO_CALIBRATION_TARGET(),
        spot=spot,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_steps=n_steps,
    )


def compile_vanilla_equity_contract_spec(
    *,
    strike: float,
    option_type: str = "call",
    exercise_style: str = "european",
) -> lattice_algebra.LatticeContractSpec:
    """Compile a vanilla equity option into the generalized lattice contract surface."""
    recipe = lattice_algebra.equity_tree(
        model_family="crr",
        strike=float(strike),
        option_type=str(option_type).strip().lower(),
    )
    normalized_exercise = str(exercise_style).strip().lower()
    if normalized_exercise in {"american", "bermudan"}:
        recipe = lattice_algebra.with_control(recipe, normalized_exercise)
    _, _, _, contract = lattice_algebra.compile_lattice_recipe(recipe)
    return contract


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
    return float(
        lattice_algebra.price_on_lattice(
            lattice,
            compile_vanilla_equity_contract_spec(
                strike=float(strike),
                option_type=option_kind,
                exercise_style=exercise_kind,
            ),
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
    "compile_vanilla_equity_contract_spec",
    "price_vanilla_equity_option_on_lattice",
    "price_vanilla_equity_option_tree",
    "resolve_vanilla_equity_tree_inputs",
]
