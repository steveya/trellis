"""Checked cliquet adapter assembled from public observation-return primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, sqrt

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support.expectations import (
    gauss_hermite_product_expectation,
)
from trellis.models.black import black76_call, black76_put
from trellis.models.observation_returns import (
    ObservationReturnContract,
    bounded_observation_return_sum,
)


@dataclass(frozen=True)
class CliquetOptionSpec:
    """Specification for one reset-style cliquet option."""

    notional: float
    spot: float
    expiry_date: date
    observation_dates: tuple[date, ...]
    option_type: str = "call"
    local_cap: float | None = None
    local_floor: float | None = None
    global_cap: float | None = None
    global_floor: float | None = None
    time_day_count: DayCountConvention = DayCountConvention.ACT_365
    quadrature_order: int = 21
    max_quadrature_nodes: int = 2_000_000
    n_paths: int = 120_000
    seed: int | None = 42
    day_count: DayCountConvention = DayCountConvention.THIRTY_E_360


class CliquetOptionPayoff:
    """Reset-style payoff with explicit primitive-level analytical assembly."""

    def __init__(self, spec: CliquetOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> CliquetOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("scheduled observation returns require settlement or as_of")
        if market_state.discount is None:
            raise ValueError("scheduled observation returns require market_state.discount")
        if market_state.vol_surface is None:
            raise ValueError("scheduled observation returns require market_state.vol_surface")

        observation_dates = tuple(sorted(spec.observation_dates))
        observation_times = tuple(
            float(year_fraction(settlement, observation_date, spec.time_day_count))
            for observation_date in observation_dates
        )
        if not observation_times:
            raise ValueError("scheduled observation returns require observation dates")

        option_type = spec.option_type.strip().lower()
        if option_type not in {"call", "put"}:
            raise ValueError(f"unsupported observation-return direction {option_type!r}")
        contract = ObservationReturnContract(
            observation_times=observation_times,
            direction="up" if option_type == "call" else "down",
            local_floor=0.0 if spec.local_floor is None else float(spec.local_floor),
            local_cap=float("inf") if spec.local_cap is None else float(spec.local_cap),
            global_floor=(
                float("-inf")
                if spec.global_floor is None
                else float(spec.global_floor)
            ),
            global_cap=(
                float("inf") if spec.global_cap is None else float(spec.global_cap)
            ),
        )
        params = dict(market_state.model_parameters or {})
        carry_rates = dict(params.get("underlier_carry_rates") or {})
        carry = float(next(iter(carry_rates.values()), 0.0))
        notional = float(spec.notional)
        spot = float(spec.spot)

        has_explicit_bounds = any(
            value is not None
            for value in (
                spec.local_floor,
                spec.local_cap,
                spec.global_floor,
                spec.global_cap,
            )
        )
        if not has_explicit_bounds:
            total = 0.0
            previous_time = 0.0
            for observation_time in observation_times:
                tau = observation_time - previous_time
                rate = float(
                    market_state.discount.zero_rate(max(observation_time, 1e-6))
                )
                sigma = float(
                    market_state.vol_surface.black_vol(max(tau, 1e-6), spot)
                )
                forward_ratio = exp((rate - carry) * tau)
                if option_type == "call":
                    optionlet = black76_call(forward_ratio, 1.0, sigma, tau)
                else:
                    optionlet = black76_put(forward_ratio, 1.0, sigma, tau)
                total += (
                    spot
                    * exp(-carry * previous_time)
                    * exp(-rate * tau)
                    * optionlet
                )
                previous_time = observation_time
            return float(notional * total)

        period_inputs: list[tuple[float, float, float]] = []
        previous_time = 0.0
        for observation_time in observation_times:
            tau = observation_time - previous_time
            rate = float(
                market_state.discount.zero_rate(max(observation_time, 1e-6))
            )
            sigma = float(market_state.vol_surface.black_vol(max(tau, 1e-6), spot))
            period_inputs.append((tau, rate, sigma))
            previous_time = observation_time

        def interval_payoff(normals):
            gross_returns = [
                exp(
                    (rate - carry - 0.5 * sigma * sigma) * tau
                    + sigma * sqrt(tau) * float(normal)
                )
                for normal, (tau, rate, sigma) in zip(normals, period_inputs)
            ]
            return bounded_observation_return_sum(gross_returns, contract)

        expected_return = gauss_hermite_product_expectation(
            interval_payoff,
            dimension=len(period_inputs),
            order=max(int(spec.quadrature_order), 3),
            max_nodes=max(int(spec.max_quadrature_nodes), 1),
        )
        discount_factor = float(
            market_state.discount.discount(observation_times[-1])
        )
        return float(notional * spot * discount_factor * expected_return)
