"""Bounded local-volatility vanilla option helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as raw_np

from trellis.models.monte_carlo.local_vol import local_vol_european_vanilla_price
from trellis.models.pde.event_aware import (
    EventAwarePDEBoundarySpec,
    EventAwarePDEGridSpec,
    EventAwarePDEOperatorSpec,
    EventAwarePDEProblemSpec,
    build_event_aware_pde_problem,
    interpolate_pde_values,
    solve_event_aware_pde,
)


@dataclass(frozen=True)
class LocalVolVanillaOptionSpec:
    """Contract inputs for a bounded European vanilla local-vol option."""

    notional: float = 1.0
    spot: float = 100.0
    strike: float = 100.0
    maturity_years: float = 1.0
    discount_rate: float = 0.04
    dividend_yield: float = 0.0
    local_vol_level: float = 0.20
    option_type: str = "call"
    local_vol_surface_name: str | None = "spx_local_vol"
    local_vol_surface: Callable | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        fields = {
            "notional": float(self.notional),
            "spot": float(self.spot),
            "strike": float(self.strike),
            "maturity_years": float(self.maturity_years),
            "discount_rate": float(self.discount_rate),
            "dividend_yield": float(self.dividend_yield),
            "local_vol_level": float(self.local_vol_level),
            "option_type": str(self.option_type or "").strip().lower(),
            "local_vol_surface_name": (
                None
                if self.local_vol_surface_name is None
                else str(self.local_vol_surface_name).strip()
            ),
        }
        for name in ("notional", "strike", "maturity_years", "local_vol_level"):
            if fields[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if fields["spot"] <= 0.0:
            raise ValueError("spot must be positive")
        if fields["option_type"] not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        for name, value in fields.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LocalVolPDEResult:
    """Structured local-vol PDE result for diagnostics."""

    price: float
    maturity: float
    rate: float
    grid_points: int
    time_steps: int
    validation_bundle: str = "local_vol:pde"


def price_local_vol_option_pde_result(
    market_state,
    spec: LocalVolVanillaOptionSpec,
    *,
    theta: float = 0.5,
    n_x: int = 161,
    n_t: int = 180,
    s_max_multiplier: float = 4.0,
) -> LocalVolPDEResult:
    """Price a European vanilla option with the local-vol one-dimensional PDE."""
    if abs(float(spec.dividend_yield)) > 1e-14:
        raise ValueError("local-vol PDE helper currently supports zero dividend_yield only")

    maturity = max(float(spec.maturity_years), 0.0)
    if maturity <= 0.0:
        return LocalVolPDEResult(
            price=float(spec.notional) * _intrinsic(spec.option_type, spec.spot, spec.strike),
            maturity=0.0,
            rate=0.0,
            grid_points=max(int(n_x), 5),
            time_steps=max(int(n_t), 1),
        )

    discount_curve = _resolve_discount_curve(market_state, spec)
    rate = _zero_rate(discount_curve, maturity)
    local_vol_surface = _resolve_local_vol_surface(market_state, spec)
    s_max = max(
        float(s_max_multiplier) * max(float(spec.spot), 1e-12),
        2.0 * max(float(spec.strike), 1e-12),
        1e-6,
    )
    lower_bc, upper_bc = _boundary_conditions(
        option_type=spec.option_type,
        strike=float(spec.strike),
        rate=rate,
        maturity=maturity,
        s_max=s_max,
    )
    problem = build_event_aware_pde_problem(
        EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=s_max,
                n_x=max(int(n_x), 5),
                maturity=maturity,
                n_t=max(int(n_t), 1),
            ),
            operator_spec=EventAwarePDEOperatorSpec(
                family="local_vol_1d",
                sigma_fn=local_vol_surface,
                r=rate,
            ),
            terminal_condition=lambda spots: _terminal_payoff(
                spots,
                spec.option_type,
                float(spec.strike),
            ),
            boundary_spec=EventAwarePDEBoundarySpec(lower=lower_bc, upper=upper_bc),
            theta=float(theta),
            rannacher_timesteps=2,
        )
    )
    values = solve_event_aware_pde(problem)
    unit_price = interpolate_pde_values(values, problem.grid.x, float(spec.spot))
    return LocalVolPDEResult(
        price=float(spec.notional) * float(unit_price),
        maturity=maturity,
        rate=rate,
        grid_points=int(problem.grid.n_x),
        time_steps=int(problem.grid.n_t),
    )


def price_local_vol_option_pde(
    market_state,
    spec: LocalVolVanillaOptionSpec,
    **kwargs,
) -> float:
    """Return the scalar local-vol PDE present value."""
    return float(price_local_vol_option_pde_result(market_state, spec, **kwargs).price)


def price_local_vol_option_monte_carlo(
    market_state,
    spec: LocalVolVanillaOptionSpec,
    *,
    n_paths: int = 80_000,
    n_steps: int = 120,
    seed: int | None = 59,
) -> float:
    """Return the scalar local-vol Monte Carlo present value."""
    discount_curve = _resolve_discount_curve(market_state, spec)
    local_vol_surface = _resolve_local_vol_surface(market_state, spec)
    unit_price = local_vol_european_vanilla_price(
        spot=float(spec.spot),
        strike=float(spec.strike),
        maturity=max(float(spec.maturity_years), 0.0),
        discount_curve=discount_curve,
        local_vol_surface=local_vol_surface,
        option_type=str(spec.option_type),
        dividend_yield=float(spec.dividend_yield),
        n_paths=max(int(n_paths), 1),
        n_steps=max(int(n_steps), 1),
        seed=seed,
    )
    return float(float(spec.notional) * float(unit_price))


def _resolve_discount_curve(market_state, spec: LocalVolVanillaOptionSpec):
    discount = getattr(market_state, "discount", None) if market_state is not None else None
    if discount is None:
        return _FlatDiscountCurve(float(spec.discount_rate))
    if hasattr(discount, "zero_rate"):
        return discount
    if hasattr(discount, "discount"):
        return _DiscountCurveAdapter(discount)
    raise ValueError("local-vol option pricing requires a discount curve")


def _resolve_local_vol_surface(market_state, spec: LocalVolVanillaOptionSpec):
    surface = spec.local_vol_surface
    if surface is None and market_state is not None:
        surface = getattr(market_state, "local_vol_surface", None)
    if surface is None and market_state is not None:
        surfaces = getattr(market_state, "local_vol_surfaces", None)
        if isinstance(surfaces, dict) and surfaces:
            name = spec.local_vol_surface_name
            if name and name in surfaces:
                surface = surfaces[name]
            else:
                surface = next(iter(surfaces.values()))
    if surface is None:
        level = float(spec.local_vol_level)

        def surface(spot, _time, _level=level):
            spot_array = raw_np.asarray(spot, dtype=float)
            if spot_array.ndim == 0:
                return _level
            return raw_np.full(spot_array.shape, _level, dtype=float)

    return _checked_local_vol_surface(surface)


def _checked_local_vol_surface(surface):
    def checked(spot, time):
        values = raw_np.asarray(surface(spot, time), dtype=float)
        if raw_np.any(values < 0.0):
            raise ValueError("local_vol_surface returned negative volatility")
        if values.ndim == 0:
            return float(values)
        return values

    return checked


class _FlatDiscountCurve:
    def __init__(self, rate: float):
        self._rate = float(rate)

    def zero_rate(self, _t: float) -> float:
        return self._rate

    def discount(self, t: float) -> float:
        return float(raw_np.exp(-self._rate * float(t)))


class _DiscountCurveAdapter:
    def __init__(self, curve):
        self._curve = curve

    def zero_rate(self, t: float) -> float:
        tenor = max(float(t), 0.0)
        if tenor <= 0.0:
            return 0.0
        df = float(self._curve.discount(tenor))
        if df <= 0.0:
            raise ValueError(f"Invalid discount factor at T={tenor}: {df}")
        return float(-raw_np.log(df) / tenor)

    def discount(self, t: float) -> float:
        return float(self._curve.discount(float(t)))


def _boundary_conditions(*, option_type: str, strike: float, rate: float, maturity: float, s_max: float):
    if option_type == "put":
        lower_bc = lambda t: float(strike * raw_np.exp(-rate * (maturity - t)))
        upper_bc = lambda _t: 0.0
    else:
        lower_bc = lambda _t: 0.0
        upper_bc = lambda t: float(s_max - strike * raw_np.exp(-rate * (maturity - t)))
    return lower_bc, upper_bc


def _terminal_payoff(spots: raw_np.ndarray, option_type: str, strike: float) -> raw_np.ndarray:
    if option_type == "put":
        return raw_np.maximum(float(strike) - spots, 0.0)
    return raw_np.maximum(spots - float(strike), 0.0)


def _intrinsic(option_type: str, spot: float, strike: float) -> float:
    if option_type == "put":
        return max(float(strike) - float(spot), 0.0)
    return max(float(spot) - float(strike), 0.0)


def _zero_rate(discount_curve, maturity: float) -> float:
    return float(discount_curve.zero_rate(max(float(maturity), 1e-12)))


__all__ = [
    "LocalVolPDEResult",
    "LocalVolVanillaOptionSpec",
    "price_local_vol_option_monte_carlo",
    "price_local_vol_option_pde",
    "price_local_vol_option_pde_result",
]
