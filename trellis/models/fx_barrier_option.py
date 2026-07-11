"""FX single-barrier option helpers with domestic/foreign discounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.barrier import barrier_option_price
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class FXBarrierOptionSpec:
    """Runtime contract for a zero-rebate single-barrier FX option."""

    notional: float = 1.0
    strike: float = 1.0
    barrier: float = 0.9
    expiry_date: date | None = None
    expiry_years: float | None = None
    fx_pair: str = "EURUSD"
    foreign_discount_key: str = "EUR-DISC"
    option_type: str = "call"
    barrier_type: str = "down_and_in"
    rebate: float = 0.0
    observations_per_year: int | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    spot: float | None = None
    n_paths: int = 120_000
    n_steps: int = 252
    seed: int | None = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", normalized_option_type(self.option_type))
        barrier_type = str(self.barrier_type or "").strip().lower().replace("-", "_")
        valid_barriers = {"down_and_out", "down_and_in", "up_and_out", "up_and_in"}
        if barrier_type not in valid_barriers:
            raise ValueError(f"Unsupported barrier_type {barrier_type!r}")
        object.__setattr__(self, "barrier_type", barrier_type)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be positive")
        if float(self.barrier) <= 0.0:
            raise ValueError("barrier must be positive")
        if self.spot is not None and float(self.spot) <= 0.0:
            raise ValueError("spot must be positive when provided")
        if self.observations_per_year is not None and int(self.observations_per_year) <= 0:
            raise ValueError("observations_per_year must be positive when provided")

    @classmethod
    def from_spec(cls, spec: Any, **overrides: Any) -> "FXBarrierOptionSpec":
        """Build an FX barrier spec from common generated-adapter aliases."""
        values = {
            "notional": _coalesce_attr(spec, ("notional",), 1.0),
            "strike": _coalesce_attr(spec, ("strike", "strike_price", "k"), 1.0),
            "barrier": _coalesce_attr(spec, ("barrier", "barrier_level"), 0.9),
            "expiry_date": _coalesce_attr(spec, ("expiry_date", "maturity_date"), None),
            "expiry_years": _coalesce_attr(
                spec,
                ("expiry_years", "maturity", "time_to_maturity", "tenor_years"),
                None,
            ),
            "fx_pair": _coalesce_attr(spec, ("fx_pair", "currency_pair"), "EURUSD"),
            "foreign_discount_key": _coalesce_attr(
                spec,
                ("foreign_discount_key", "foreign_curve_key", "foreign_curve"),
                "EUR-DISC",
            ),
            "option_type": _coalesce_attr(spec, ("option_type", "payoff_type"), "call"),
            "barrier_type": _coalesce_attr(
                spec,
                ("barrier_type", "barrier_style", "knock_type"),
                "down_and_in",
            ),
            "rebate": _coalesce_attr(spec, ("rebate",), 0.0),
            "observations_per_year": getattr(spec, "observations_per_year", None),
            "day_count": getattr(spec, "day_count", DayCountConvention.ACT_365),
            "spot": _coalesce_attr(spec, ("spot", "underlier_spot", "s0"), None),
            "n_paths": _coalesce_attr(spec, ("n_paths",), 120_000),
            "n_steps": _coalesce_attr(spec, ("n_steps",), 252),
            "seed": _coalesce_attr(spec, ("seed",), 42),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class ResolvedFXBarrierInputs:
    """Resolved market and numerical inputs for one FX barrier route."""

    notional: float
    spot: float
    strike: float
    barrier: float
    maturity: float
    domestic_rate: float
    foreign_rate: float
    sigma: float
    option_type: str
    barrier_type: str
    rebate: float
    observations_per_year: int | None
    n_paths: int
    n_steps: int
    seed: int | None


@dataclass(frozen=True)
class FXBarrierMonteCarloResult:
    """Structured result and contract evidence for FX barrier Monte Carlo."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    resolved: ResolvedFXBarrierInputs
    validation_bundle: str = "fx_barrier:monte_carlo_gbm"


def resolve_fx_barrier_inputs(
    market_state: MarketState,
    spec,
) -> ResolvedFXBarrierInputs:
    """Resolve FX spot, curves, vol, and barrier fields for one FX barrier option."""
    base = spec if isinstance(spec, FXBarrierOptionSpec) else FXBarrierOptionSpec.from_spec(spec)
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for FX barrier pricing")

    foreign_key = _resolve_foreign_discount_key(market_state, base.foreign_discount_key)
    if not market_state.forecast_curves or foreign_key not in market_state.forecast_curves:
        raise ValueError(
            f"market_state.forecast_curves must contain foreign discount key {foreign_key!r}"
        )

    pair = _resolve_fx_pair(market_state, base.fx_pair)
    spot = _resolve_spot(market_state, base, pair=pair)
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for FX barrier pricing")
    maturity = _resolve_maturity(settlement, base)

    strike = float(base.strike)
    domestic_rate = (
        float(market_state.discount.zero_rate(max(maturity, 1e-8)))
        if maturity > 0.0
        else 0.0
    )
    foreign_curve = market_state.forecast_curves[foreign_key]
    foreign_rate = (
        float(foreign_curve.zero_rate(max(maturity, 1e-8)))
        if maturity > 0.0
        else 0.0
    )
    if market_state.vol_surface is None and maturity > 0.0:
        raise ValueError("market_state.vol_surface is required for FX barrier pricing")
    sigma = (
        float(market_state.vol_surface.black_vol(max(maturity, 1e-8), strike))
        if maturity > 0.0 and market_state.vol_surface is not None
        else 0.0
    )
    return ResolvedFXBarrierInputs(
        notional=float(base.notional),
        spot=spot,
        strike=strike,
        barrier=float(base.barrier),
        maturity=maturity,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        sigma=sigma,
        option_type=base.option_type,
        barrier_type=base.barrier_type,
        rebate=float(base.rebate),
        observations_per_year=_resolve_observations_per_year(base, maturity),
        n_paths=max(int(base.n_paths), 1),
        n_steps=max(int(base.n_steps), 1),
        seed=base.seed,
    )


def price_fx_barrier_option_analytical(
    market_state: MarketState,
    spec,
) -> float:
    """Price a single-barrier FX option with domestic/foreign discounting."""
    resolved = resolve_fx_barrier_inputs(market_state, spec)
    unit_price = barrier_option_price(
        resolved.spot,
        resolved.strike,
        resolved.barrier,
        resolved.domestic_rate,
        resolved.sigma,
        resolved.maturity,
        barrier_type=resolved.barrier_type,
        option_type=resolved.option_type,
        rebate=resolved.rebate,
        q=resolved.foreign_rate,
        observations_per_year=resolved.observations_per_year,
    )
    return float(resolved.notional) * float(unit_price)


def price_fx_barrier_option_monte_carlo_result(
    market_state: MarketState,
    spec,
) -> FXBarrierMonteCarloResult:
    """Return a Monte Carlo estimate for a single-barrier FX option."""
    resolved = resolve_fx_barrier_inputs(market_state, spec)
    process = GBM(
        mu=float(resolved.domestic_rate - resolved.foreign_rate),
        sigma=float(resolved.sigma),
    )
    engine = MonteCarloEngine(
        process,
        n_paths=resolved.n_paths,
        n_steps=resolved.n_steps,
        seed=resolved.seed,
        method="exact",
    )

    def payoff_fn(paths):
        path_array = raw_np.asarray(paths, dtype=float)
        terminal = path_array[:, -1]
        if resolved.barrier_type.startswith("down"):
            touched = raw_np.min(path_array, axis=1) <= resolved.barrier
        else:
            touched = raw_np.max(path_array, axis=1) >= resolved.barrier
        active = touched if resolved.barrier_type.endswith("_in") else ~touched
        intrinsic = terminal_intrinsic(
            resolved.option_type,
            spot=terminal,
            strike=resolved.strike,
        )
        payoff = raw_np.where(active, intrinsic, resolved.rebate)
        return resolved.notional * payoff

    result = engine.price(
        resolved.spot,
        resolved.maturity,
        payoff_fn,
        discount_rate=resolved.domestic_rate,
        return_paths=False,
    )
    return FXBarrierMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
        n_steps=int(engine.n_steps),
        resolved=resolved,
    )


def price_fx_barrier_option_monte_carlo(
    market_state: MarketState,
    spec,
) -> float:
    """Return the scalar Monte Carlo FX barrier price."""
    return float(price_fx_barrier_option_monte_carlo_result(market_state, spec).price)


def _resolve_maturity(settlement: date, spec: FXBarrierOptionSpec) -> float:
    if spec.expiry_years is not None:
        return max(float(spec.expiry_years), 0.0)
    if spec.expiry_date is None:
        return 1.0
    expiry = spec.expiry_date
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    return max(float(year_fraction(settlement, expiry, spec.day_count)), 0.0)


def _resolve_observations_per_year(
    spec: FXBarrierOptionSpec,
    maturity: float,
) -> int | None:
    if spec.observations_per_year is not None:
        return int(spec.observations_per_year)
    if maturity <= 0.0:
        return None
    return max(int(round(float(spec.n_steps) / maturity)), 1)


def _resolve_foreign_discount_key(market_state: MarketState, requested: str) -> str:
    key = str(requested or "").strip()
    if key:
        return key
    forecast_curves = getattr(market_state, "forecast_curves", None) or {}
    if len(forecast_curves) == 1:
        return str(next(iter(forecast_curves)))
    return "EUR-DISC"


def _resolve_fx_pair(market_state: MarketState, requested: str) -> str:
    pair = str(requested or "").strip().upper()
    if pair:
        return pair
    fx_rates = getattr(market_state, "fx_rates", None) or {}
    if len(fx_rates) == 1:
        return str(next(iter(fx_rates)))
    return "EURUSD"


def _resolve_spot(market_state: MarketState, spec: FXBarrierOptionSpec, *, pair: str) -> float:
    if spec.spot is not None:
        return float(spec.spot)
    fx_quote = (market_state.fx_rates or {}).get(pair)
    if fx_quote is not None:
        return float(fx_quote.spot)
    if market_state.spot is not None:
        return float(market_state.spot)
    if market_state.underlier_spots and pair in market_state.underlier_spots:
        return float(market_state.underlier_spots[pair])
    raise ValueError(f"FX spot for pair {pair!r} is not available in market_state")


def _coalesce_attr(spec, names: tuple[str, ...], default):
    for name in names:
        value = getattr(spec, name, None)
        if value is not None:
            return value
    return default


__all__ = [
    "FXBarrierMonteCarloResult",
    "FXBarrierOptionSpec",
    "ResolvedFXBarrierInputs",
    "price_fx_barrier_option_analytical",
    "price_fx_barrier_option_monte_carlo",
    "price_fx_barrier_option_monte_carlo_result",
    "resolve_fx_barrier_inputs",
]
