"""Reusable barrier support primitives for analytical, PDE, and MC routes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.monte_carlo.path_state import (
    BarrierMonitor,
    MonteCarloPathRequirement,
    StateAwarePayoff,
)


@dataclass(frozen=True)
class DoubleBarrierSpec:
    """Runtime contract for a zero-rebate double-barrier vanilla payoff."""

    notional: float = 1.0
    spot: float = 100.0
    strike: float = 100.0
    lower_barrier: float = 70.0
    upper_barrier: float = 140.0
    maturity: float = 1.0
    rate: float = 0.0
    sigma: float = 0.2
    option_type: str = "call"
    knock: str = "out"

    def __post_init__(self) -> None:
        if self.lower_barrier <= 0.0:
            raise ValueError("lower_barrier must be positive")
        if self.upper_barrier <= self.lower_barrier:
            raise ValueError("upper_barrier must exceed lower_barrier")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        option_type = normalized_option_type(self.option_type)
        object.__setattr__(self, "option_type", option_type)
        knock = str(self.knock or "out").strip().lower().replace("-", "_")
        if knock not in {"out", "in", "knock_out", "knock_in"}:
            raise ValueError("knock must be 'out' or 'in'")
        object.__setattr__(
            self,
            "knock",
            {"knock_out": "out", "knock_in": "in"}.get(knock, knock),
        )

    @classmethod
    def from_spec(cls, spec: Any, **overrides: Any) -> "DoubleBarrierSpec":
        """Build a barrier spec from common generated-adapter field aliases."""
        spot = float(_coalesce_attr(spec, ("spot", "underlier_spot", "s0"), 100.0))
        legacy_barrier = getattr(spec, "barrier", None)
        default_lower = 70.0
        default_upper = 140.0
        if legacy_barrier is not None:
            barrier_value = float(legacy_barrier)
            if barrier_value < spot:
                default_lower = barrier_value
            elif barrier_value > spot:
                default_upper = barrier_value

        values = {
            "notional": _coalesce_attr(spec, ("notional",), 1.0),
            "spot": spot,
            "strike": _coalesce_attr(spec, ("strike", "strike_price", "k"), 100.0),
            "lower_barrier": _coalesce_attr(
                spec,
                ("lower_barrier", "down_barrier", "lower_bound", "barrier_low"),
                default_lower,
            ),
            "upper_barrier": _coalesce_attr(
                spec,
                ("upper_barrier", "up_barrier", "upper_bound", "barrier_high"),
                default_upper,
            ),
            "maturity": _coalesce_attr(
                spec,
                ("maturity", "expiry_years", "time_to_maturity", "tenor_years"),
                1.0,
            ),
            "rate": _coalesce_attr(spec, ("rate", "risk_free_rate", "r"), 0.0),
            "sigma": _coalesce_attr(spec, ("sigma", "vol", "volatility"), 0.2),
            "option_type": _coalesce_attr(spec, ("option_type", "payoff_type"), "call"),
            "knock": _coalesce_attr(spec, ("knock", "knock_type", "barrier_style"), "out"),
        }
        values.update(overrides)
        return cls(**values)


def terminal_double_barrier_payoff(
    terminal_spot,
    spec: DoubleBarrierSpec,
) -> raw_np.ndarray:
    """Return the vanilla terminal payoff before barrier activation."""
    return raw_np.asarray(
        float(spec.notional)
        * terminal_intrinsic(spec.option_type, spot=terminal_spot, strike=spec.strike),
        dtype=float,
    )


def double_barrier_hit_mask(paths, spec: DoubleBarrierSpec) -> raw_np.ndarray:
    """Return a pathwise indicator for lower-or-upper barrier breach."""
    path_array = raw_np.asarray(paths, dtype=float)
    return raw_np.any(
        (path_array <= spec.lower_barrier) | (path_array >= spec.upper_barrier),
        axis=1,
    )


def double_barrier_path_payoff(paths, spec: DoubleBarrierSpec) -> raw_np.ndarray:
    """Return pathwise knock-in or knock-out double-barrier payoffs."""
    path_array = raw_np.asarray(paths, dtype=float)
    vanilla = terminal_double_barrier_payoff(path_array[:, -1], spec)
    hit = double_barrier_hit_mask(path_array, spec)
    if spec.knock == "out":
        return raw_np.where(hit, 0.0, vanilla)
    return raw_np.where(hit, vanilla, 0.0)


def double_barrier_state_payoff(spec: DoubleBarrierSpec) -> StateAwarePayoff:
    """Build a reduced-storage state-aware double-barrier payoff."""
    lower = BarrierMonitor("lower_barrier", spec.lower_barrier, "down")
    upper = BarrierMonitor("upper_barrier", spec.upper_barrier, "up")

    def evaluate_paths(paths):
        return double_barrier_path_payoff(paths, spec)

    def evaluate_state(state):
        vanilla = terminal_double_barrier_payoff(state.terminal_values, spec)
        hit = state.barrier_hit("lower_barrier") | state.barrier_hit("upper_barrier")
        if spec.knock == "out":
            return raw_np.where(hit, 0.0, vanilla)
        return raw_np.where(hit, vanilla, 0.0)

    return StateAwarePayoff(
        path_requirement=MonteCarloPathRequirement(barrier_monitors=(lower, upper)),
        evaluate_paths_fn=evaluate_paths,
        evaluate_state_fn=evaluate_state,
        name="double_barrier_payoff",
        derivative_metadata={
            "discontinuous_features": ("barrier_monitor",),
            "unsupported_reason": "double_barrier_monitor_discontinuity",
        },
    )


def resolve_double_barrier_inputs(market_state, spec) -> DoubleBarrierSpec:
    """Resolve market state, dates, vol, and aliases into a double-barrier spec."""
    base = spec if isinstance(spec, DoubleBarrierSpec) else DoubleBarrierSpec.from_spec(spec)
    maturity = _resolve_maturity(market_state, spec, default=base.maturity)
    spot = _resolve_spot(market_state, spec, default=base.spot)
    strike = float(_coalesce_attr(spec, ("strike", "strike_price", "k"), base.strike))
    rate = _resolve_rate(market_state, maturity, default=base.rate)
    sigma = _resolve_sigma(market_state, maturity, strike, default=base.sigma)
    return replace(
        base,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        sigma=sigma,
    )


def _resolve_maturity(market_state, spec, *, default: float) -> float:
    for attr in ("maturity", "expiry_years", "time_to_maturity", "tenor_years"):
        value = getattr(spec, attr, None)
        if value is not None:
            return max(float(value), 0.0)
    expiry = getattr(spec, "expiry_date", None) or getattr(spec, "maturity_date", None)
    if expiry is not None:
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry)
        settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
        if settlement is not None:
            day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
            return max(float(year_fraction(settlement, expiry, day_count)), 0.0)
    return max(float(default), 0.0)


def _resolve_spot(market_state, spec, *, default: float) -> float:
    for attr in ("spot", "underlier_spot", "s0"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    value = getattr(market_state, "spot", None)
    if value is not None:
        return float(value)
    return float(default)


def _resolve_rate(market_state, maturity: float, *, default: float) -> float:
    discount = getattr(market_state, "discount", None)
    if discount is None or maturity <= 0.0:
        return float(default)
    return float(discount.zero_rate(max(maturity, 1e-8)))


def _resolve_sigma(market_state, maturity: float, strike: float, *, default: float) -> float:
    vol_surface = getattr(market_state, "vol_surface", None)
    if vol_surface is None or maturity <= 0.0:
        return float(default)
    return float(vol_surface.black_vol(max(maturity, 1e-8), strike))


def _coalesce_attr(spec, names: tuple[str, ...], default):
    for name in names:
        value = getattr(spec, name, None)
        if value is not None:
            return value
    return default


__all__ = [
    "DoubleBarrierSpec",
    "double_barrier_hit_mask",
    "double_barrier_path_payoff",
    "double_barrier_state_payoff",
    "resolve_double_barrier_inputs",
    "terminal_double_barrier_payoff",
]
