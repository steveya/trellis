"""Agent-generated payoff: American put option on equity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention


@dataclass(frozen=True)
class AmericanPutEquitySpec:
    """Specification for the generated American equity put payoff."""
    spot: float
    strike: float
    expiry_date: date
    notional: float = 1.0
    option_type: str = "put"
    exercise_style: str = "american"
    exercise_dates: tuple[date, ...] | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


class AmericanOptionPayoff:
    """Generated American put pricer using Longstaff-Schwartz on GBM paths."""
    def __init__(self, spec: AmericanPutEquitySpec):
        """Store the generated American-option specification."""
        self._spec = spec

    @property
    def spec(self) -> AmericanPutEquitySpec:
        """Return the immutable generated American-option specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Declare that valuation needs discounting and Black volatility."""
        return {"discount_curve", "black_vol_surface"}

    def evaluate(self, market_state: MarketState) -> float:
        """Price the generated American put with Longstaff-Schwartz regression."""
        from trellis.models.monte_carlo.engine import MonteCarloEngine
        from trellis.models.monte_carlo.event_state import event_step_indices
        from trellis.models.monte_carlo.lsm import longstaff_schwartz
        from trellis.models.monte_carlo.schemes import LaguerreBasis
        from trellis.models.monte_carlo.single_state_diffusion import (
            resolve_single_state_monte_carlo_inputs,
        )
        from trellis.models.processes.gbm import GBM
        from trellis.models.resolution.single_state_diffusion import (
            terminal_intrinsic_from_resolved,
        )

        spec = self._spec
        resolved = resolve_single_state_monte_carlo_inputs(
            market_state,
            spec,
            scheme="exact",
            variance_reduction="none",
            n_paths=4096,
            n_steps=64,
            seed=42,
        )
        if resolved.maturity <= 0.0:
            return float(
                resolved.notional
                * terminal_intrinsic_from_resolved(resolved.spot, resolved)
            )

        process = GBM(
            mu=resolved.rate - resolved.dividend_yield,
            sigma=resolved.sigma,
        )
        engine = MonteCarloEngine(
            process,
            n_paths=resolved.n_paths,
            n_steps=resolved.n_steps,
            seed=resolved.seed,
            method="exact",
        )
        paths = engine.simulate(resolved.spot, resolved.maturity)
        dt = resolved.maturity / engine.n_steps
        exercise_style = str(spec.exercise_style).strip().lower()
        if exercise_style == "american":
            exercise_steps = list(range(1, engine.n_steps + 1))
        elif exercise_style == "bermudan":
            if not spec.exercise_dates:
                raise ValueError("Bermudan LSM requires spec.exercise_dates")
            settlement = market_state.settlement or market_state.as_of
            if settlement is None:
                raise ValueError("Bermudan LSM requires market settlement or as_of")
            event_times = []
            for exercise_date in spec.exercise_dates:
                exercise_time = float(
                    year_fraction(settlement, exercise_date, spec.day_count)
                )
                if 0.0 <= exercise_time <= resolved.maturity:
                    event_times.append(exercise_time)
            if not event_times:
                raise ValueError("Bermudan LSM resolved no valid exercise dates")
            exercise_steps = list(
                event_step_indices(
                    tuple(event_times), resolved.maturity, resolved.n_steps
                )
            )
            exercise_steps = sorted(
                {max(int(step), 1) for step in exercise_steps}
            )
        elif exercise_style == "european":
            exercise_steps = [resolved.n_steps]
        else:
            raise ValueError(f"Unsupported exercise_style {exercise_style!r}")
        basis = LaguerreBasis(degree=2)

        def payoff_fn(spots):
            """Return intrinsic put values at the exercise spots used by LSM."""
            return terminal_intrinsic_from_resolved(spots, resolved)

        return float(resolved.notional) * float(
            longstaff_schwartz(
                paths,
                exercise_steps,
                payoff_fn,
                discount_rate=resolved.rate,
                dt=dt,
                basis_fn=lambda spots: basis(
                    spots / max(resolved.strike, 1e-12)
                ),
            )
        )
