"""Single-barrier option pricing adapters over shared PDE and MC primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import math
from typing import Any

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import StateAwarePayoff, barrier_payoff
from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class SingleBarrierSpec:
    """Runtime contract for a zero-rebate single-barrier vanilla payoff."""

    notional: float = 1.0
    spot: float = 100.0
    strike: float = 100.0
    barrier: float = 80.0
    maturity: float = 1.0
    rate: float = 0.0
    sigma: float = 0.2
    option_type: str = "call"
    barrier_type: str = "down_and_out"
    rebate: float = 0.0
    observations_per_year: int | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.barrier <= 0.0:
            raise ValueError("barrier must be positive")
        object.__setattr__(self, "option_type", normalized_option_type(self.option_type))
        barrier_type = str(self.barrier_type or "").strip().lower().replace("-", "_")
        valid_barriers = {"down_and_out", "down_and_in", "up_and_out", "up_and_in"}
        if barrier_type not in valid_barriers:
            raise ValueError(f"Unsupported barrier_type {barrier_type!r}")
        object.__setattr__(self, "barrier_type", barrier_type)
        object.__setattr__(self, "maturity", max(float(self.maturity), 0.0))
        if self.observations_per_year is not None:
            observations = int(self.observations_per_year)
            if observations <= 0:
                raise ValueError("observations_per_year must be positive when provided")
            object.__setattr__(self, "observations_per_year", observations)

    @classmethod
    def from_spec(cls, spec: Any, **overrides: Any) -> "SingleBarrierSpec":
        """Build a single-barrier spec from common generated-adapter aliases."""
        values = {
            "notional": _coalesce_attr(spec, ("notional",), 1.0),
            "spot": _coalesce_attr(spec, ("spot", "underlier_spot", "s0"), 100.0),
            "strike": _coalesce_attr(spec, ("strike", "strike_price", "k"), 100.0),
            "barrier": _coalesce_attr(spec, ("barrier", "barrier_level"), 80.0),
            "maturity": _coalesce_attr(
                spec,
                ("maturity", "expiry_years", "time_to_maturity", "tenor_years"),
                1.0,
            ),
            "rate": _coalesce_attr(spec, ("rate", "risk_free_rate", "r"), 0.0),
            "sigma": _coalesce_attr(spec, ("sigma", "vol", "volatility"), 0.2),
            "option_type": _coalesce_attr(spec, ("option_type", "payoff_type"), "call"),
            "barrier_type": _coalesce_attr(
                spec,
                ("barrier_type", "barrier_style", "knock_type"),
                "down_and_out",
            ),
            "rebate": _coalesce_attr(spec, ("rebate",), 0.0),
            "observations_per_year": getattr(spec, "observations_per_year", None),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class SingleBarrierPDEConfig:
    """Numerical controls for the bounded one-dimensional PDE route."""

    spot_steps: int = 241
    time_steps: int = 420
    theta: float = 0.5
    rannacher_timesteps: int = 2
    far_spot_multiplier: float = 4.0


@dataclass(frozen=True)
class SingleBarrierMonteCarloConfig:
    """Simulation controls for the single-barrier Monte Carlo route."""

    n_paths: int = 50_000
    n_steps: int = 252
    seed: int | None = 42
    method: str = "exact"


@dataclass(frozen=True)
class SingleBarrierPDEResult:
    """Structured result and contract evidence for single-barrier PDE pricing."""

    price: float
    knock_out_price: float
    vanilla_price: float
    resolved_spec: SingleBarrierSpec
    grid_bounds: tuple[float, float]
    grid_shape: tuple[int, int]
    boundary_conditions: str
    operator_signature: str
    validation_bundle: str = "single_barrier:pde_theta_1d"


@dataclass(frozen=True)
class SingleBarrierMonteCarloResult:
    """Structured result and contract evidence for single-barrier MC pricing."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    resolved_spec: SingleBarrierSpec
    path_contract: tuple[str, ...]
    derivative_metadata: dict[str, object]
    validation_bundle: str = "single_barrier:monte_carlo_gbm"


def resolve_single_barrier_inputs(market_state, spec) -> SingleBarrierSpec:
    """Resolve market state, dates, volatility, and aliases into a barrier spec."""
    base = spec if isinstance(spec, SingleBarrierSpec) else SingleBarrierSpec.from_spec(spec)
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


def single_barrier_state_payoff(spec) -> StateAwarePayoff:
    """Build a reduced-storage state-aware single-barrier payoff."""
    resolved = spec if isinstance(spec, SingleBarrierSpec) else SingleBarrierSpec.from_spec(spec)
    direction = "down" if resolved.barrier_type.startswith("down") else "up"
    knock = "out" if resolved.barrier_type.endswith("_out") else "in"

    def terminal_payoff(terminal):
        return float(resolved.notional) * terminal_intrinsic(
            resolved.option_type,
            spot=terminal,
            strike=resolved.strike,
        )

    return barrier_payoff(
        barrier=resolved.barrier,
        direction=direction,
        knock=knock,
        terminal_payoff_fn=terminal_payoff,
        name=f"{resolved.barrier_type}_{resolved.option_type}_barrier_payoff",
    )


def price_single_barrier_option_pde_result(
    market_state,
    spec,
    *,
    config: SingleBarrierPDEConfig | None = None,
) -> SingleBarrierPDEResult:
    """Return a bounded-grid PDE price for a zero-rebate single-barrier option."""
    resolved = resolve_single_barrier_inputs(market_state, spec)
    if abs(float(resolved.rebate)) > 1e-14:
        raise ValueError("single-barrier PDE helper currently supports zero rebate only")
    cfg = config or SingleBarrierPDEConfig()
    vanilla_price = _black_scholes_vanilla_price(resolved)
    knock_out_price, grid_bounds, grid_shape = _price_knock_out_pde(resolved, cfg)
    if resolved.barrier_type.endswith("_out"):
        price = knock_out_price
    else:
        price = max(vanilla_price - knock_out_price, 0.0)

    return SingleBarrierPDEResult(
        price=float(price),
        knock_out_price=float(knock_out_price),
        vanilla_price=float(vanilla_price),
        resolved_spec=resolved,
        grid_bounds=grid_bounds,
        grid_shape=grid_shape,
        boundary_conditions="absorbing_at_barrier",
        operator_signature="BlackScholesOperator(sigma_fn, r_fn)",
    )


def price_single_barrier_option_pde(
    market_state,
    spec,
    *,
    config: SingleBarrierPDEConfig | None = None,
) -> float:
    """Return the scalar single-barrier PDE price."""
    return float(price_single_barrier_option_pde_result(market_state, spec, config=config).price)


def price_single_barrier_option_monte_carlo_result(
    market_state,
    spec,
    *,
    config: SingleBarrierMonteCarloConfig | None = None,
) -> SingleBarrierMonteCarloResult:
    """Return a GBM Monte Carlo price using one explicit barrier monitor."""
    resolved = resolve_single_barrier_inputs(market_state, spec)
    if abs(float(resolved.rebate)) > 1e-14:
        raise ValueError("single-barrier Monte Carlo helper currently supports zero rebate only")
    cfg = config or SingleBarrierMonteCarloConfig()
    n_steps = _resolve_mc_steps(resolved, cfg)
    payoff = single_barrier_state_payoff(resolved)
    process = GBM(mu=float(resolved.rate), sigma=float(resolved.sigma))
    engine = MonteCarloEngine(
        process,
        n_paths=max(int(cfg.n_paths), 1),
        n_steps=n_steps,
        seed=cfg.seed,
        method=str(cfg.method or "exact"),
    )
    result = engine.price(
        float(resolved.spot),
        float(resolved.maturity),
        payoff,
        discount_rate=float(resolved.rate),
        storage_policy=payoff.path_requirement,
        return_paths=False,
    )
    monitors = payoff.path_requirement.barrier_monitors
    return SingleBarrierMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
        n_steps=engine.n_steps,
        resolved_spec=resolved,
        path_contract=tuple(f"{monitor.name}:{monitor.direction}" for monitor in monitors),
        derivative_metadata=dict(result.get("derivative_metadata") or {}),
    )


def price_single_barrier_option_monte_carlo(
    market_state,
    spec,
    *,
    config: SingleBarrierMonteCarloConfig | None = None,
) -> float:
    """Return the scalar single-barrier Monte Carlo price."""
    return float(
        price_single_barrier_option_monte_carlo_result(
            market_state,
            spec,
            config=config,
        ).price
    )


def _price_knock_out_pde(
    spec: SingleBarrierSpec,
    config: SingleBarrierPDEConfig,
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    """Price the knock-out leg with an absorbing barrier boundary."""
    n_x = max(int(config.spot_steps), 5)
    n_t = max(int(config.time_steps), 1)
    if spec.maturity <= 0.0:
        price = _terminal_if_not_breached(spec)
        return price, _domain_bounds(spec, config), (n_x, n_t)
    if _barrier_breached_at_spot(spec):
        return 0.0, _domain_bounds(spec, config), (n_x, n_t)

    x_min, x_max = _domain_bounds(spec, config)
    grid = Grid(x_min, x_max, n_x, float(spec.maturity), n_t, log_spacing=False)
    terminal = raw_np.asarray(
        float(spec.notional)
        * terminal_intrinsic(spec.option_type, spot=grid.x, strike=spec.strike),
        dtype=float,
    )
    if spec.barrier_type.startswith("down"):
        terminal[0] = 0.0
        lower_bc_fn = lambda _time: 0.0
        upper_bc_fn = lambda time: _vanilla_boundary_value(spec, x_max, time)
    else:
        terminal[-1] = 0.0
        lower_bc_fn = lambda time: _vanilla_boundary_value(spec, x_min, time)
        upper_bc_fn = lambda _time: 0.0

    operator = BlackScholesOperator(
        sigma_fn=lambda _spot, _time: float(spec.sigma),
        r_fn=lambda _time: float(spec.rate),
    )
    values = theta_method_1d(
        grid,
        operator,
        terminal,
        theta=min(max(float(config.theta), 0.0), 1.0),
        lower_bc_fn=lower_bc_fn,
        upper_bc_fn=upper_bc_fn,
        rannacher_timesteps=max(int(config.rannacher_timesteps), 0),
    )
    price = float(raw_np.interp(float(spec.spot), grid.x, values))
    return max(price, 0.0), (float(x_min), float(x_max)), (n_x, n_t)


def _resolve_mc_steps(spec: SingleBarrierSpec, config: SingleBarrierMonteCarloConfig) -> int:
    if spec.observations_per_year is not None and spec.maturity > 0.0:
        return max(int(round(float(spec.observations_per_year) * float(spec.maturity))), 1)
    return max(int(config.n_steps), 1)


def _domain_bounds(
    spec: SingleBarrierSpec,
    config: SingleBarrierPDEConfig,
) -> tuple[float, float]:
    if spec.barrier_type.startswith("down"):
        far = max(
            float(config.far_spot_multiplier) * max(float(spec.spot), 1e-12),
            2.0 * max(float(spec.strike), 1e-12),
            1.25 * float(spec.barrier),
        )
        return float(spec.barrier), far
    return 0.0, float(spec.barrier)


def _barrier_breached_at_spot(spec: SingleBarrierSpec) -> bool:
    if spec.barrier_type.startswith("down"):
        return float(spec.spot) <= float(spec.barrier)
    return float(spec.spot) >= float(spec.barrier)


def _terminal_if_not_breached(spec: SingleBarrierSpec) -> float:
    if _barrier_breached_at_spot(spec):
        return 0.0
    return float(
        float(spec.notional)
        * terminal_intrinsic(spec.option_type, spot=raw_np.asarray([spec.spot]), strike=spec.strike)[0]
    )


def _vanilla_boundary_value(spec: SingleBarrierSpec, spot: float, time: float) -> float:
    tau = max(float(spec.maturity) - float(time), 0.0)
    if spec.option_type == "call":
        return float(spec.notional) * max(
            float(spot) - float(spec.strike) * math.exp(-float(spec.rate) * tau),
            0.0,
        )
    return float(spec.notional) * max(
        float(spec.strike) * math.exp(-float(spec.rate) * tau) - float(spot),
        0.0,
    )


def _black_scholes_vanilla_price(spec: SingleBarrierSpec) -> float:
    """Return the zero-dividend Black-Scholes vanilla price for parity checks."""
    if spec.maturity <= 0.0:
        return float(
            float(spec.notional)
            * terminal_intrinsic(
                spec.option_type,
                spot=raw_np.asarray([spec.spot]),
                strike=spec.strike,
            )[0]
        )
    sqrt_t = math.sqrt(float(spec.maturity))
    sigma_sqrt_t = float(spec.sigma) * sqrt_t
    df = math.exp(-float(spec.rate) * float(spec.maturity))
    if sigma_sqrt_t <= 0.0:
        forward = float(spec.spot) / max(df, 1e-12)
        intrinsic = (
            max(forward - float(spec.strike), 0.0)
            if spec.option_type == "call"
            else max(float(spec.strike) - forward, 0.0)
        )
        return float(spec.notional) * df * intrinsic

    d1 = (
        math.log(float(spec.spot) / float(spec.strike))
        + (float(spec.rate) + 0.5 * float(spec.sigma) ** 2) * float(spec.maturity)
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    if spec.option_type == "call":
        unit = float(spec.spot) * _normal_cdf(d1) - float(spec.strike) * df * _normal_cdf(d2)
    else:
        unit = float(spec.strike) * df * _normal_cdf(-d2) - float(spec.spot) * _normal_cdf(-d1)
    return float(spec.notional) * unit


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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


__all__ = [
    "SingleBarrierMonteCarloConfig",
    "SingleBarrierMonteCarloResult",
    "SingleBarrierPDEConfig",
    "SingleBarrierPDEResult",
    "SingleBarrierSpec",
    "price_single_barrier_option_monte_carlo",
    "price_single_barrier_option_monte_carlo_result",
    "price_single_barrier_option_pde",
    "price_single_barrier_option_pde_result",
    "resolve_single_barrier_inputs",
    "single_barrier_state_payoff",
]
