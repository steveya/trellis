"""Double-barrier option pricing adapters over shared PDE and MC primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as raw_np

from trellis.models.analytical.support.barriers import (
    DoubleBarrierSpec,
    double_barrier_state_payoff,
    resolve_double_barrier_inputs,
    terminal_double_barrier_payoff,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class DoubleBarrierPDEConfig:
    """Numerical controls for the bounded one-dimensional PDE route."""

    spot_steps: int = 141
    time_steps: int = 260
    theta: float = 0.5
    rannacher_timesteps: int = 2


@dataclass(frozen=True)
class DoubleBarrierMonteCarloConfig:
    """Simulation controls for the double-barrier Monte Carlo route."""

    n_paths: int = 50_000
    n_steps: int = 180
    seed: int | None = 12345
    method: str = "exact"


@dataclass(frozen=True)
class DoubleBarrierPDEResult:
    """Structured result and contract evidence for double-barrier PDE pricing."""

    price: float
    knock_out_price: float
    vanilla_price: float
    resolved_spec: DoubleBarrierSpec
    grid_bounds: tuple[float, float]
    grid_shape: tuple[int, int]
    boundary_conditions: str
    operator_signature: str
    validation_bundle: str = "double_barrier:pde_theta_1d"


@dataclass(frozen=True)
class DoubleBarrierMonteCarloResult:
    """Structured result and contract evidence for double-barrier MC pricing."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    resolved_spec: DoubleBarrierSpec
    path_contract: tuple[str, ...]
    derivative_metadata: dict[str, object]
    validation_bundle: str = "double_barrier:monte_carlo_gbm"


def price_double_barrier_option_pde_result(
    market_state,
    spec,
    *,
    config: DoubleBarrierPDEConfig | None = None,
) -> DoubleBarrierPDEResult:
    """Return a bounded-grid PDE price for a zero-rebate double-barrier option."""
    resolved = resolve_double_barrier_inputs(market_state, spec)
    cfg = config or DoubleBarrierPDEConfig()
    vanilla_price = _black_scholes_vanilla_price(resolved)
    knock_out_price = _price_knock_out_pde(resolved, cfg)
    if resolved.knock == "out":
        price = knock_out_price
    else:
        price = max(vanilla_price - knock_out_price, 0.0)

    return DoubleBarrierPDEResult(
        price=float(price),
        knock_out_price=float(knock_out_price),
        vanilla_price=float(vanilla_price),
        resolved_spec=resolved,
        grid_bounds=(float(resolved.lower_barrier), float(resolved.upper_barrier)),
        grid_shape=(max(int(cfg.spot_steps), 5), max(int(cfg.time_steps), 1)),
        boundary_conditions="absorbing",
        operator_signature="BlackScholesOperator(sigma_fn, r_fn)",
    )


def price_double_barrier_option_pde(
    market_state,
    spec,
    *,
    config: DoubleBarrierPDEConfig | None = None,
) -> float:
    """Return the scalar double-barrier PDE price."""
    return float(price_double_barrier_option_pde_result(market_state, spec, config=config).price)


def price_double_barrier_option_monte_carlo_result(
    market_state,
    spec,
    *,
    config: DoubleBarrierMonteCarloConfig | None = None,
) -> DoubleBarrierMonteCarloResult:
    """Return a GBM Monte Carlo price using explicit lower and upper monitors."""
    resolved = resolve_double_barrier_inputs(market_state, spec)
    cfg = config or DoubleBarrierMonteCarloConfig()
    payoff = double_barrier_state_payoff(resolved)
    process = GBM(mu=float(resolved.rate), sigma=float(resolved.sigma))
    engine = MonteCarloEngine(
        process,
        n_paths=max(int(cfg.n_paths), 1),
        n_steps=max(int(cfg.n_steps), 1),
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
    return DoubleBarrierMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
        n_steps=engine.n_steps,
        resolved_spec=resolved,
        path_contract=tuple(f"{monitor.name}:{monitor.direction}" for monitor in monitors),
        derivative_metadata=dict(result.get("derivative_metadata") or {}),
    )


def price_double_barrier_option_monte_carlo(
    market_state,
    spec,
    *,
    config: DoubleBarrierMonteCarloConfig | None = None,
) -> float:
    """Return the scalar double-barrier Monte Carlo price."""
    return float(
        price_double_barrier_option_monte_carlo_result(
            market_state,
            spec,
            config=config,
        ).price
    )


def _price_knock_out_pde(
    spec: DoubleBarrierSpec,
    config: DoubleBarrierPDEConfig,
) -> float:
    """Price the knock-out leg on a bounded Black-Scholes grid."""
    if spec.maturity <= 0.0:
        if spec.lower_barrier < spec.spot < spec.upper_barrier:
            return float(terminal_double_barrier_payoff(raw_np.asarray([spec.spot]), spec)[0])
        return 0.0
    if spec.spot <= spec.lower_barrier or spec.spot >= spec.upper_barrier:
        return 0.0

    n_x = max(int(config.spot_steps), 5)
    n_t = max(int(config.time_steps), 1)
    grid = Grid(
        float(spec.lower_barrier),
        float(spec.upper_barrier),
        n_x,
        float(spec.maturity),
        n_t,
        log_spacing=False,
    )
    terminal = terminal_double_barrier_payoff(grid.x, spec)
    terminal[0] = 0.0
    terminal[-1] = 0.0
    operator = BlackScholesOperator(
        sigma_fn=lambda _spot, _time: float(spec.sigma),
        r_fn=lambda _time: float(spec.rate),
    )
    values = theta_method_1d(
        grid,
        operator,
        terminal,
        theta=min(max(float(config.theta), 0.0), 1.0),
        lower_bc_fn=lambda _time: 0.0,
        upper_bc_fn=lambda _time: 0.0,
        rannacher_timesteps=max(int(config.rannacher_timesteps), 0),
    )
    return float(raw_np.interp(float(spec.spot), grid.x, values))


def _black_scholes_vanilla_price(spec: DoubleBarrierSpec) -> float:
    """Return the zero-dividend Black-Scholes vanilla price for parity checks."""
    if spec.maturity <= 0.0:
        return float(terminal_double_barrier_payoff(raw_np.asarray([spec.spot]), spec)[0])
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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


__all__ = [
    "DoubleBarrierMonteCarloConfig",
    "DoubleBarrierMonteCarloResult",
    "DoubleBarrierPDEConfig",
    "DoubleBarrierPDEResult",
    "price_double_barrier_option_monte_carlo",
    "price_double_barrier_option_monte_carlo_result",
    "price_double_barrier_option_pde",
    "price_double_barrier_option_pde_result",
]
