"""Agent-generated payoff: Build a pricer for: European equity call under local vol: PDE vs MC

Implementation target: local_vol_mc
Preferred method family: monte_carlo

Implementation target: local_vol_mc."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.core.differentiable import get_numpy



@dataclass(frozen=True)
class EuropeanLocalVolOptionSpec:
    """Specification for Build a pricer for: European equity call under local vol: PDE vs MC

Implementation target: local_vol_mc
Preferred method family: monte_carlo

Implementation target: local_vol_mc."""
    notional: float
    strike: float
    expiry_date: date
    option_type: str = 'call'
    day_count: DayCountConvention = DayCountConvention.ACT_365
    n_paths: int = 50000
    n_steps: int = 252


class EuropeanLocalVolMonteCarloPayoff:
    """Build a pricer for: European equity call under local vol: PDE vs MC

Implementation target: local_vol_mc
Preferred method family: monte_carlo

Implementation target: local_vol_mc."""

    def __init__(self, spec: EuropeanLocalVolOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> EuropeanLocalVolOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount", "local_vol", "spot"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.monte_carlo.local_vol import local_vol_european_vanilla_price
        from trellis.models.processes.local_vol import LocalVol

        np = get_numpy()
        spec = self._spec

        if spec.option_type.lower() not in {"call", "put"}:
            raise ValueError(f"Unsupported option_type: {spec.option_type!r}")

        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        if T <= 0.0:
            intrinsic = max(spec.strike - market_state.spot, 0.0) if spec.option_type.lower() == "put" else max(market_state.spot - spec.strike, 0.0)
            return float(spec.notional * intrinsic)

        spot = float(market_state.spot)
        discount_curve = market_state.discount

        try:
            local_vol_surface = market_state.local_vol_surface
        except AttributeError:
            local_vol_surface = market_state.local_vol_surfaces["spot"]

        if hasattr(discount_curve, "discount"):
            discount_factor = float(discount_curve.discount(T))
        else:
            discount_factor = float(np.exp(-float(discount_curve.zero_rate(T)) * T))

        lv_process = LocalVol(local_vol_surface)

        n_steps = max(int(spec.n_steps), int(np.ceil(T * 252)))
        n_paths = max(int(spec.n_paths), 10_000)

        engine = MonteCarloEngine(
            lv_process,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=42,
            method="euler",
        )

        def payoff_fn(paths):
            st = paths[:, -1]
            if spec.option_type.lower() == "call":
                return np.maximum(st - spec.strike, 0.0) * spec.notional
            return np.maximum(spec.strike - st, 0.0) * spec.notional

        try:
            result = local_vol_european_vanilla_price(
                spot=spot,
                strike=float(spec.strike),
                expiry=T,
                option_type=spec.option_type.lower(),
                notional=float(spec.notional),
                local_vol_surface=local_vol_surface,
                discount_curve=discount_curve,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=42,
            )
            price = float(result["price"] if isinstance(result, dict) and "price" in result else result)
        except Exception:
            simulated = engine.price(spot, T, payoff_fn, discount_rate=0.0)
            if isinstance(simulated, dict):
                raw_price = float(simulated["price"])
                se = float(simulated.get("stderr", simulated.get("se", 0.0)))
            else:
                raw_price = float(simulated)
                se = 0.0
            price = discount_factor * raw_price
            if price != 0.0 and se > 0.0 and (discount_factor * se) / abs(price) >= 0.01:
                raise RuntimeError("Monte Carlo standard error exceeds 1% of price")

        return float(price)
