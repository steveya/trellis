"""Checked analytical adapter for the FinancePy variance-swap parity task."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.interpolation import linear_interp
from trellis.models.analytical.support import discount_factor_from_zero_rate


QuoteGrid = str | tuple[float, ...] | None


def _parse_quote_grid(value: QuoteGrid, *, name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    raw_values = value.split(",") if isinstance(value, str) else value
    try:
        return tuple(
            float(item.strip()) if isinstance(item, str) else float(item)
            for item in raw_values
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


@dataclass(frozen=True)
class VarianceSwapSpec:
    """Specification for one equity variance swap."""

    notional: float
    spot: float
    strike_variance: float
    expiry_date: date
    realized_variance: float = 0.0
    replication_strikes: QuoteGrid = None
    replication_volatilities: QuoteGrid = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class VarianceSwapPayoff:
    """Compose the admitted smile-slope approximation from public primitives."""

    def __init__(self, spec: VarianceSwapSpec):
        self._spec = spec

    @property
    def spec(self) -> VarianceSwapSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"discount_curve", "black_vol_surface"}

    def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:
        return self._outputs(market_state)

    def evaluate(self, market_state: MarketState) -> float:
        return self._outputs(market_state)["price"]

    def _outputs(self, market_state: MarketState) -> dict[str, float]:
        spec = self._spec
        if market_state.discount is None:
            raise ValueError(
                "variance swap analytical pricing requires a discount curve"
            )
        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError(
                "variance swap analytical pricing requires settlement or as_of"
            )

        contract_spot = float(spec.spot)
        spot = float(
            contract_spot if market_state.spot is None else market_state.spot
        )
        notional = float(spec.notional)
        strike_variance = float(spec.strike_variance)
        if not all(
            isfinite(value)
            for value in (contract_spot, spot, notional, strike_variance)
        ):
            raise ValueError("variance swap scalar contract values must be finite")
        if contract_spot <= 0.0 or spot <= 0.0:
            raise ValueError("variance swap spot must be positive")

        maturity = float(year_fraction(settlement, spec.expiry_date, spec.day_count))
        if maturity < 0.0:
            raise ValueError("variance swap expiry must not precede settlement")
        if maturity == 0.0:
            return {
                "price": 0.0,
                "fair_strike_variance": strike_variance,
            }

        strike_grid = _parse_quote_grid(
            spec.replication_strikes,
            name="replication_strikes",
        )
        if not strike_grid:
            strike_grid = tuple(
                weight * spot for weight in (0.6, 0.8, 1.0, 1.2, 1.4)
            )

        volatility_grid = _parse_quote_grid(
            spec.replication_volatilities,
            name="replication_volatilities",
        )
        if not volatility_grid:
            if market_state.vol_surface is None:
                raise ValueError(
                    "variance swap analytical pricing requires a vol surface "
                    "or explicit replication_volatilities"
                )
            volatility_grid = tuple(
                float(
                    market_state.vol_surface.black_vol(
                        max(maturity, 1e-6),
                        strike,
                    )
                )
                for strike in strike_grid
            )

        if len(strike_grid) != len(volatility_grid):
            raise ValueError(
                "replication_strikes and replication_volatilities must have "
                "the same length"
            )
        if len(strike_grid) < 2:
            raise ValueError("variance swap quote grid requires at least two points")
        if not all(isfinite(strike) for strike in strike_grid):
            raise ValueError("replication_strikes must be finite")
        if not all(strike > 0.0 for strike in strike_grid):
            raise ValueError("replication_strikes must be positive")
        if any(
            right <= left for left, right in zip(strike_grid, strike_grid[1:])
        ):
            raise ValueError("replication_strikes must be strictly increasing")
        if not all(isfinite(volatility) for volatility in volatility_grid):
            raise ValueError("replication_volatilities must be finite")
        if not all(volatility > 0.0 for volatility in volatility_grid):
            raise ValueError("replication_volatilities must be positive")

        atm_volatility = float(linear_interp(spot, strike_grid, volatility_grid))
        strike_span = strike_grid[-1] - strike_grid[0]
        smile_slope = spot * (
            volatility_grid[-1] - volatility_grid[0]
        ) / strike_span
        fair_strike_variance = float(
            atm_volatility**2
            * sqrt(1.0 + 3.0 * maturity * smile_slope * smile_slope)
        )
        zero_rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
        discount_factor = float(
            discount_factor_from_zero_rate(zero_rate, maturity)
        )
        price = float(
            notional
            * discount_factor
            * (fair_strike_variance - strike_variance)
        )
        if not isfinite(fair_strike_variance) or not isfinite(price):
            raise ValueError(
                "variance swap smile-slope approximation returned non-finite outputs"
            )
        return {
            "price": price,
            "fair_strike_variance": fair_strike_variance,
        }
