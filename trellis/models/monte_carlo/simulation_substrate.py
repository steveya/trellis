"""Reusable factor-state simulation substrate and future-value cube helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

from trellis.book import Book, FutureValueCube, FutureValueCubeMetadata
from trellis.core.date_utils import build_payment_timeline, year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.types import DayCountConvention, Frequency
from trellis.instruments.swap import SwapPayoff, SwapSpec
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.event_aware import (
    build_event_aware_monte_carlo_process,
    resolve_hull_white_monte_carlo_process_inputs,
)
from trellis.models.monte_carlo.path_state import MonteCarloPathRequirement

np = get_numpy()


@dataclass(frozen=True)
class FactorStateSimulationResult:
    """Observed factor-state cross-sections on a deterministic simulation grid."""

    observation_times: tuple[float, ...]
    factor_paths: Any
    path_state: object
    measure: str = "risk_neutral"
    numeraire: str = "discount_curve"
    state_names: tuple[str, ...] = ()
    process_family: str = ""
    observation_dates: tuple[date, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_times",
            tuple(float(time) for time in self.observation_times),
        )
        factor_paths = np.asarray(self.factor_paths, dtype=float)
        if factor_paths.ndim not in {2, 3}:
            raise ValueError(
                "FactorStateSimulationResult factor_paths must have shape "
                "(observation_time, path) or (observation_time, path, state_dim)"
            )
        object.__setattr__(self, "factor_paths", factor_paths.copy())
        object.__setattr__(
            self,
            "observation_dates",
            tuple(self.observation_dates or ()),
        )
        if self.observation_dates and len(self.observation_dates) != len(self.observation_times):
            raise ValueError(
                "FactorStateSimulationResult observation_dates must match observation_times"
            )
        object.__setattr__(
            self,
            "state_names",
            tuple(str(name) for name in (self.state_names or ())),
        )
        object.__setattr__(self, "measure", str(self.measure or "risk_neutral"))
        object.__setattr__(self, "numeraire", str(self.numeraire or "discount_curve"))
        object.__setattr__(self, "process_family", str(self.process_family or ""))

    @property
    def n_paths(self) -> int:
        return int(self.factor_paths.shape[1])

    @property
    def state_dimension(self) -> int:
        if self.factor_paths.ndim == 2:
            return 1
        return int(self.factor_paths.shape[2])

    def cross_section(self, index: int):
        """Return one observed factor-state cross-section."""
        return np.asarray(self.factor_paths[int(index)], dtype=float).copy()


@runtime_checkable
class SimulatedMarketProjection(Protocol):
    """Project latent factor state onto a valuation-facing market view."""

    def project(
        self,
        *,
        observation_time: float,
        factor_state,
        observation_date: date | None = None,
    ):
        ...


@runtime_checkable
class ObservationProgram(Protocol):
    """Advance contract state on the shared observation grid before valuation."""

    def initialize(self, *, n_paths: int):
        ...

    def advance(
        self,
        *,
        observation_time: float,
        observation_date: date | None,
        factor_state,
        projected_market,
        state,
    ):
        ...


@runtime_checkable
class ConditionalValuationModel(Protocol):
    """Map `(t_i, X_{t_i}, Y_i)` into pathwise clean future values."""

    def value(
        self,
        *,
        observation_time: float,
        observation_date: date | None,
        factor_state,
        projected_market,
        state,
    ):
        ...


@dataclass(frozen=True)
class _ResolvedSwapPeriod:
    """Schedule period with one unified simulation time coordinate."""

    start_date: date
    end_date: date
    payment_date: date
    accrual_fraction: float
    t_start: float
    t_end: float
    t_payment: float


@dataclass(frozen=True)
class _ResolvedSwapPortfolioPosition:
    """One normalized swap position with a shared-cube multiplier."""

    name: str
    spec: SwapSpec
    fixed_periods: tuple[_ResolvedSwapPeriod, ...]
    float_periods: tuple[_ResolvedSwapPeriod, ...]
    position_multiplier: float = 1.0


@dataclass(frozen=True)
class HullWhiteRateMarketView:
    """Projected rate-facing market view under one-factor Hull-White state."""

    observation_time: float
    observation_date: date | None
    short_rate: Any
    discount_curve: object
    forward_curve: object
    mean_reversion: float

    def __post_init__(self) -> None:
        short_rate = np.asarray(self.short_rate, dtype=float)
        if short_rate.ndim == 2 and short_rate.shape[1] == 1:
            short_rate = short_rate[:, 0]
        if short_rate.ndim != 1:
            raise ValueError("HullWhiteRateMarketView short_rate must be pathwise scalar state")
        object.__setattr__(self, "short_rate", short_rate.copy())
        object.__setattr__(self, "observation_time", float(self.observation_time))
        object.__setattr__(self, "mean_reversion", float(self.mean_reversion))

    @property
    def n_paths(self) -> int:
        return int(self.short_rate.shape[0])

    def bond_price(self, maturity_time: float):
        """Return `P(t, T)` for each simulated path."""
        maturity = float(maturity_time)
        if maturity <= self.observation_time + 1e-12:
            return np.ones(self.n_paths, dtype=float)

        if self.observation_time <= 1e-12:
            return np.full(
                self.n_paths,
                float(self.discount_curve.discount(maturity)),
                dtype=float,
            )

        anchor_discount = max(float(self.discount_curve.discount(self.observation_time)), 1e-12)
        maturity_discount = max(float(self.discount_curve.discount(maturity)), 1e-12)
        tau = max(maturity - self.observation_time, 0.0)
        if abs(self.mean_reversion) < 1e-12:
            B = tau
        else:
            B = (1.0 - np.exp(-self.mean_reversion * tau)) / self.mean_reversion
        anchor_short_rate = -np.log(anchor_discount) / max(self.observation_time, 1e-12)
        anchored_ratio = maturity_discount / anchor_discount
        return anchored_ratio * np.exp(-B * (self.short_rate - anchor_short_rate))

    def forward_rate(self, start_time: float, end_time: float):
        """Return a projected forecast forward with the initial basis preserved."""
        start = max(float(start_time), 1e-6)
        end = float(end_time)
        if end <= start:
            raise ValueError("HullWhiteRateMarketView forward_rate requires end_time > start_time")
        tau = max(end - start, 1e-12)

        discount_forward = (self.bond_price(start) / self.bond_price(end) - 1.0) / tau

        discount_base_curve = self.discount_curve
        if hasattr(self.forward_curve, "_curve"):
            forecast_base_curve = self.forward_curve._curve
        else:
            forecast_base_curve = self.forward_curve

        base_discount_forward = (
            max(float(discount_base_curve.discount(start)), 1e-12)
            / max(float(discount_base_curve.discount(end)), 1e-12)
            - 1.0
        ) / tau
        if hasattr(self.forward_curve, "forward_rate"):
            base_forecast_forward = float(self.forward_curve.forward_rate(start, end))
        else:
            base_forecast_forward = (
                max(float(forecast_base_curve.discount(start)), 1e-12)
                / max(float(forecast_base_curve.discount(end)), 1e-12)
                - 1.0
            ) / tau

        return discount_forward + (base_forecast_forward - base_discount_forward)


@dataclass(frozen=True)
class HullWhiteRateProjection:
    """Explicit `Phi_t` from short-rate factor state onto a rate-facing market view."""

    discount_curve: object
    forward_curve: object
    mean_reversion: float

    def project(
        self,
        *,
        observation_time: float,
        factor_state,
        observation_date: date | None = None,
    ) -> HullWhiteRateMarketView:
        factor_array = np.asarray(factor_state, dtype=float)
        if factor_array.ndim == 2 and factor_array.shape[1] == 1:
            factor_array = factor_array[:, 0]
        if factor_array.ndim != 1:
            raise ValueError(
                "HullWhiteRateProjection requires scalar short-rate cross-sections"
            )
        return HullWhiteRateMarketView(
            observation_time=float(observation_time),
            observation_date=observation_date,
            short_rate=factor_array,
            discount_curve=self.discount_curve,
            forward_curve=self.forward_curve,
            mean_reversion=float(self.mean_reversion),
        )


@dataclass(frozen=True)
class SwapContractState:
    """Current floating reset state carried on the observation grid."""

    current_float_start_date: date | None = None
    current_float_end_date: date | None = None
    current_float_start: float | None = None
    current_float_end: float | None = None
    current_float_rate: Any | None = None


@dataclass(frozen=True)
class SwapFloatResetProgram:
    """Contract-state program for vanilla floating reset events."""

    float_periods: tuple[_ResolvedSwapPeriod, ...]

    def initialize(self, *, n_paths: int) -> SwapContractState:
        del n_paths
        return SwapContractState()

    def advance(
        self,
        *,
        observation_time: float,
        observation_date: date | None,
        factor_state,
        projected_market: HullWhiteRateMarketView,
        state: SwapContractState | None,
    ) -> SwapContractState:
        del factor_state
        current = state or SwapContractState()
        if observation_date is None:
            return current

        next_state = current
        if (
            next_state.current_float_end_date is not None
            and observation_date >= next_state.current_float_end_date
        ):
            next_state = SwapContractState()

        for period in self.float_periods:
            if period.start_date == observation_date and period.end_date > observation_date:
                next_state = SwapContractState(
                    current_float_start_date=period.start_date,
                    current_float_end_date=period.end_date,
                    current_float_start=period.t_start,
                    current_float_end=period.t_end,
                    current_float_rate=np.asarray(
                        projected_market.forward_rate(period.t_start, period.t_end),
                        dtype=float,
                    ).copy(),
                )
                break

        return next_state


@dataclass(frozen=True)
class HullWhiteSwapFutureValueModel:
    """Pathwise clean swap future value on an observation grid."""

    spec: SwapSpec
    fixed_periods: tuple[_ResolvedSwapPeriod, ...]
    float_periods: tuple[_ResolvedSwapPeriod, ...]

    def value(
        self,
        *,
        observation_time: float,
        observation_date: date | None,
        factor_state,
        projected_market: HullWhiteRateMarketView,
        state: SwapContractState | None,
    ):
        del factor_state
        if observation_date is None:
            raise ValueError("HullWhiteSwapFutureValueModel requires observation_date")
        if observation_date >= self.spec.end_date:
            return np.zeros(projected_market.n_paths, dtype=float)

        fixed_leg = self._fixed_leg_clean_value(
            observation_time=observation_time,
            observation_date=observation_date,
            projected_market=projected_market,
        )
        float_leg = self._float_leg_clean_value(
            observation_time=observation_time,
            observation_date=observation_date,
            projected_market=projected_market,
            state=state or SwapContractState(),
        )
        sign = 1.0 if self.spec.is_payer else -1.0
        return sign * (float_leg - fixed_leg)

    def _fixed_leg_clean_value(
        self,
        *,
        observation_time: float,
        observation_date: date,
        projected_market: HullWhiteRateMarketView,
    ):
        value = np.zeros(projected_market.n_paths, dtype=float)
        for period in self.fixed_periods:
            if period.end_date <= observation_date:
                continue
            bond = projected_market.bond_price(period.t_payment)
            value = value + (
                float(self.spec.notional)
                * float(self.spec.fixed_rate)
                * float(period.accrual_fraction)
                * bond
            )

        active_period = _active_period(self.fixed_periods, observation_date)
        if active_period is not None and active_period.start_date < observation_date < active_period.end_date:
            accrued = year_fraction(
                active_period.start_date,
                observation_date,
                self.spec.fixed_day_count,
            )
            value = value - (
                float(self.spec.notional)
                * float(self.spec.fixed_rate)
                * float(accrued)
            )
        return value

    def _float_leg_clean_value(
        self,
        *,
        observation_time: float,
        observation_date: date,
        projected_market: HullWhiteRateMarketView,
        state: SwapContractState,
    ):
        value = np.zeros(projected_market.n_paths, dtype=float)

        if (
            state.current_float_rate is not None
            and state.current_float_start_date is not None
            and state.current_float_end_date is not None
            and state.current_float_start is not None
            and state.current_float_end is not None
            and state.current_float_start_date < observation_date < state.current_float_end_date
        ):
            current_rate = np.asarray(state.current_float_rate, dtype=float)
            current_period = _period_by_dates(
                self.float_periods,
                state.current_float_start_date,
                state.current_float_end_date,
            )
            if current_period is not None:
                value = value + (
                    float(self.spec.notional)
                    * current_rate
                    * float(current_period.accrual_fraction)
                    * projected_market.bond_price(current_period.t_payment)
                )
                accrued = year_fraction(
                    current_period.start_date,
                    observation_date,
                    self.spec.float_day_count,
                )
                value = value - float(self.spec.notional) * current_rate * float(accrued)

        for period in self.float_periods:
            if period.end_date <= observation_date:
                continue
            if period.start_date < observation_date:
                continue
            forward_rate = np.asarray(
                projected_market.forward_rate(period.t_start, period.t_end),
                dtype=float,
            )
            value = value + (
                float(self.spec.notional)
                * forward_rate
                * float(period.accrual_fraction)
                * projected_market.bond_price(period.t_payment)
            )

        return value


def simulate_factor_state_observations(
    engine: MonteCarloEngine,
    initial_state,
    maturity: float,
    *,
    observation_times: tuple[float, ...] | list[float],
    observation_dates: tuple[date, ...] | list[date] | None = None,
    state_names: tuple[str, ...] | list[str] = (),
    process_family: str = "",
    measure: str = "risk_neutral",
    numeraire: str = "discount_curve",
) -> FactorStateSimulationResult:
    """Simulate and materialize one factor-state cross-section per observation time."""
    resolved_times = tuple(float(time) for time in observation_times)
    if not resolved_times:
        raise ValueError("simulate_factor_state_observations requires observation_times")
    if any(time < 0.0 for time in resolved_times):
        raise ValueError("simulate_factor_state_observations requires non-negative times")
    if any(later < earlier for earlier, later in zip(resolved_times, resolved_times[1:])):
        raise ValueError("simulate_factor_state_observations requires sorted observation_times")

    steps = tuple(
        _step_index(time, maturity=float(maturity), n_steps=engine.n_steps)
        for time in resolved_times
    )
    requirement = MonteCarloPathRequirement(
        snapshot_steps=tuple(sorted({step for step in steps if 0 < step < engine.n_steps}))
    )
    path_state = engine.simulate_state(initial_state, float(maturity), requirement)
    factor_paths = tuple(
        np.asarray(path_state.snapshot(step), dtype=float)
        for step in steps
    )

    return FactorStateSimulationResult(
        observation_times=resolved_times,
        factor_paths=np.asarray(factor_paths, dtype=float),
        path_state=path_state,
        measure=measure,
        numeraire=numeraire,
        state_names=tuple(state_names),
        process_family=process_family,
        observation_dates=tuple(observation_dates or ()),
    )


def evaluate_conditional_valuation_paths(
    simulation: FactorStateSimulationResult,
    *,
    projection: SimulatedMarketProjection,
    valuation_model: ConditionalValuationModel,
    observation_program: ObservationProgram | None = None,
):
    """Evaluate `V(t_i, X_{t_i}, Y_i)` over one observed factor-state grid."""
    values = np.zeros(
        (len(simulation.observation_times), simulation.n_paths),
        dtype=float,
    )
    state = (
        observation_program.initialize(n_paths=simulation.n_paths)
        if observation_program is not None
        else None
    )
    observation_dates = simulation.observation_dates or (None,) * len(simulation.observation_times)

    for index, observation_time in enumerate(simulation.observation_times):
        factor_state = simulation.cross_section(index)
        observation_date = observation_dates[index]
        projected_market = projection.project(
            observation_time=observation_time,
            factor_state=factor_state,
            observation_date=observation_date,
        )
        if observation_program is not None:
            state = observation_program.advance(
                observation_time=observation_time,
                observation_date=observation_date,
                factor_state=factor_state,
                projected_market=projected_market,
                state=state,
            )
        values[index] = _coerce_path_vector(
            valuation_model.value(
                observation_time=observation_time,
                observation_date=observation_date,
                factor_state=factor_state,
                projected_market=projected_market,
                state=state,
            ),
            n_paths=simulation.n_paths,
        )

    return values


def build_future_value_cube(
    *,
    position_values: dict[str, Any],
    simulation: FactorStateSimulationResult,
    metadata: FutureValueCubeMetadata | dict[str, Any] | None = None,
) -> FutureValueCube:
    """Emit one stable trade/date/path cube from named pathwise valuations."""
    names = tuple(position_values)
    tensor = np.asarray(
        [
            _coerce_position_matrix(
                position_values[name],
                n_times=len(simulation.observation_times),
                n_paths=simulation.n_paths,
            )
            for name in names
        ],
        dtype=float,
    )
    return FutureValueCube(
        values=tensor,
        position_names=names,
        observation_times=simulation.observation_times,
        observation_dates=simulation.observation_dates,
        metadata=metadata,
    )


def price_interest_rate_swap_portfolio_future_value_cube(
    *,
    positions: Mapping[str, SwapSpec | SwapPayoff] | Book,
    market_state,
    n_paths: int = 10_000,
    n_steps: int = 120,
    seed: int | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> FutureValueCube:
    """Price a shared-path swap portfolio future-value cube on the reusable substrate."""
    if getattr(market_state, "discount", None) is None:
        raise ValueError("Swap future-value cube requires market_state.discount")

    resolved_positions = _resolve_swap_portfolio_positions(
        positions,
        settlement=market_state.settlement,
    )
    observation_dates, observation_times = _swap_portfolio_observation_grid(
        resolved_positions,
        settlement=market_state.settlement,
    )
    horizon = max(float(observation_times[-1]), 1e-12)
    reference_fixed_rate = _portfolio_reference_fixed_rate(resolved_positions)

    process_spec, initial_short_rate = resolve_hull_white_monte_carlo_process_inputs(
        market_state,
        option_horizon=horizon,
        strike=reference_fixed_rate,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    process = build_event_aware_monte_carlo_process(process_spec)
    engine = MonteCarloEngine(
        process,
        n_paths=max(int(n_paths), 1),
        n_steps=max(int(n_steps), 1),
        seed=seed,
        method="exact",
    )
    simulation = simulate_factor_state_observations(
        engine,
        float(initial_short_rate),
        horizon,
        observation_times=observation_times,
        observation_dates=observation_dates,
        state_names=("short_rate",),
        process_family="hull_white_1f",
    )

    position_values: dict[str, Any] = {}
    position_provenance: dict[str, dict[str, Any]] = {}
    for position in resolved_positions:
        projection = HullWhiteRateProjection(
            discount_curve=market_state.discount,
            forward_curve=market_state.forecast_forward_curve(position.spec.rate_index),
            mean_reversion=float(process_spec.mean_reversion or 0.0),
        )
        observation_program = SwapFloatResetProgram(float_periods=position.float_periods)
        valuation_model = HullWhiteSwapFutureValueModel(
            spec=position.spec,
            fixed_periods=position.fixed_periods,
            float_periods=position.float_periods,
        )
        values = evaluate_conditional_valuation_paths(
            simulation,
            projection=projection,
            valuation_model=valuation_model,
            observation_program=observation_program,
        )
        if abs(position.position_multiplier - 1.0) > 0.0:
            values = np.asarray(values, dtype=float) * float(position.position_multiplier)
        position_values[position.name] = values

        provenance = {
            "instrument_type": "interest_rate_swap",
            "rate_index": position.spec.rate_index,
            "is_payer": bool(position.spec.is_payer),
            "contract_notional": float(position.spec.notional),
        }
        if abs(position.position_multiplier - 1.0) > 0.0:
            provenance["book_notional_multiplier"] = float(position.position_multiplier)
        position_provenance[position.name] = provenance

    return build_future_value_cube(
        position_values=position_values,
        simulation=simulation,
        metadata=FutureValueCubeMetadata(
            measure=simulation.measure,
            numeraire=simulation.numeraire,
            value_semantics="clean_future_value",
            phase_semantics="post_event",
            state_names=simulation.state_names,
            process_family=simulation.process_family,
            compute_plan={
                "engine_family": "simulation_substrate",
                "process_family": "hull_white_1f",
                "projection_family": "hull_white_1f_rate_projection",
                "conditional_valuation_family": "pathwise_clean_swap_value",
                "observation_program": "swap_float_reset_state",
                "observation_grid": (
                    "portfolio_float_boundary_union"
                    if len(resolved_positions) > 1
                    else "float_boundary_dates"
                ),
                "portfolio_size": len(resolved_positions),
                "position_input_type": "book" if isinstance(positions, Book) else "mapping",
                "reference_fixed_rate": float(reference_fixed_rate),
                "n_paths": int(n_paths),
                "n_steps": int(n_steps),
            },
            position_provenance=position_provenance,
        ),
    )


def price_interest_rate_swap_future_value_cube(
    *,
    name: str,
    spec: SwapSpec,
    market_state,
    n_paths: int = 10_000,
    n_steps: int = 120,
    seed: int | None = None,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> FutureValueCube:
    """Price one vanilla IRS future-value cube through the portfolio substrate."""
    return price_interest_rate_swap_portfolio_future_value_cube(
        positions={str(name): spec},
        market_state=market_state,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )


def _resolve_swap_periods(
    start_date: date,
    end_date: date,
    frequency: Frequency,
    *,
    leg_day_count: DayCountConvention,
    time_origin: date,
) -> tuple[_ResolvedSwapPeriod, ...]:
    timeline = build_payment_timeline(
        start_date,
        end_date,
        frequency,
        day_count=leg_day_count,
        time_origin=time_origin,
    )
    resolved: list[_ResolvedSwapPeriod] = []
    for period in timeline:
        resolved.append(
            _ResolvedSwapPeriod(
                start_date=period.start_date,
                end_date=period.end_date,
                payment_date=period.payment_date,
                accrual_fraction=float(
                    period.accrual_fraction
                    if period.accrual_fraction is not None
                    else year_fraction(period.start_date, period.end_date, leg_day_count)
                ),
                t_start=float(
                    period.t_start
                    if period.t_start is not None
                    else year_fraction(time_origin, period.start_date, leg_day_count)
                ),
                t_end=float(
                    period.t_end
                    if period.t_end is not None
                    else year_fraction(time_origin, period.end_date, leg_day_count)
                ),
                t_payment=float(
                    period.t_payment
                    if period.t_payment is not None
                    else year_fraction(time_origin, period.payment_date, leg_day_count)
                ),
            )
        )
    return tuple(resolved)


def _swap_observation_grid(
    float_periods: tuple[_ResolvedSwapPeriod, ...],
    *,
    settlement: date,
) -> tuple[tuple[date, ...], tuple[float, ...]]:
    date_to_time: dict[date, float] = {settlement: 0.0}
    for period in float_periods:
        if period.start_date >= settlement:
            date_to_time.setdefault(period.start_date, float(period.t_start))
        if period.end_date >= settlement:
            date_to_time.setdefault(period.end_date, float(period.t_end))
    ordered_dates = tuple(sorted(date_to_time))
    ordered_times = tuple(
        float(date_to_time[observation_date])
        for observation_date in ordered_dates
    )
    return ordered_dates, ordered_times


def _swap_portfolio_observation_grid(
    positions: tuple[_ResolvedSwapPortfolioPosition, ...],
    *,
    settlement: date,
) -> tuple[tuple[date, ...], tuple[float, ...]]:
    date_to_time: dict[date, float] = {settlement: 0.0}
    for position in positions:
        for period in position.float_periods:
            if period.start_date >= settlement:
                date_to_time.setdefault(period.start_date, float(period.t_start))
            if period.end_date >= settlement:
                date_to_time.setdefault(period.end_date, float(period.t_end))
    ordered_dates = tuple(sorted(date_to_time))
    ordered_times = tuple(float(date_to_time[observation_date]) for observation_date in ordered_dates)
    return ordered_dates, ordered_times


def _step_index(time: float, *, maturity: float, n_steps: int) -> int:
    if maturity <= 0.0 or n_steps <= 0:
        return 0
    return int(np.clip(np.rint((float(time) / float(maturity)) * int(n_steps)), 0, int(n_steps)))


def _active_period(
    periods: tuple[_ResolvedSwapPeriod, ...],
    observation_date: date,
) -> _ResolvedSwapPeriod | None:
    for period in periods:
        if period.start_date < observation_date < period.end_date:
            return period
    return None


def _period_by_dates(
    periods: tuple[_ResolvedSwapPeriod, ...],
    start_date: date,
    end_date: date,
) -> _ResolvedSwapPeriod | None:
    for period in periods:
        if period.start_date == start_date and period.end_date == end_date:
            return period
    return None


def _resolve_swap_portfolio_positions(
    positions: Mapping[str, SwapSpec | SwapPayoff] | Book,
    *,
    settlement: date,
) -> tuple[_ResolvedSwapPortfolioPosition, ...]:
    if isinstance(positions, Book):
        position_items = tuple((name, positions[name], positions.notional(name)) for name in positions)
    elif isinstance(positions, Mapping):
        position_items = tuple((str(name), instrument, 1.0) for name, instrument in positions.items())
    else:
        raise TypeError("Swap portfolio positions must be provided as a mapping or Book")

    resolved_positions: list[_ResolvedSwapPortfolioPosition] = []
    for name, instrument, position_multiplier in position_items:
        spec = _coerce_swap_spec(instrument)
        fixed_periods = _resolve_swap_periods(
            spec.start_date,
            spec.end_date,
            spec.fixed_frequency,
            leg_day_count=spec.fixed_day_count,
            time_origin=settlement,
        )
        float_periods = _resolve_swap_periods(
            spec.start_date,
            spec.end_date,
            spec.float_frequency,
            leg_day_count=spec.float_day_count,
            time_origin=settlement,
        )
        resolved_positions.append(
            _ResolvedSwapPortfolioPosition(
                name=str(name),
                spec=spec,
                fixed_periods=fixed_periods,
                float_periods=float_periods,
                position_multiplier=float(position_multiplier),
            )
        )

    if not resolved_positions:
        raise ValueError("Swap portfolio future-value cube requires at least one position")
    return tuple(resolved_positions)


def _coerce_swap_spec(instrument: SwapSpec | SwapPayoff) -> SwapSpec:
    if isinstance(instrument, SwapSpec):
        return instrument
    if isinstance(instrument, SwapPayoff):
        return instrument.spec
    raise TypeError(
        "Swap portfolio positions must be SwapSpec or SwapPayoff instances"
    )


def _portfolio_reference_fixed_rate(
    positions: tuple[_ResolvedSwapPortfolioPosition, ...],
) -> float:
    weighted_rate_sum = 0.0
    weight_sum = 0.0
    for position in positions:
        weight = abs(float(position.spec.notional) * float(position.position_multiplier))
        if weight <= 0.0:
            continue
        weighted_rate_sum += weight * float(position.spec.fixed_rate)
        weight_sum += weight
    if weight_sum > 0.0:
        return weighted_rate_sum / weight_sum
    return float(np.mean([float(position.spec.fixed_rate) for position in positions]))


def _coerce_path_vector(values, *, n_paths: int):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(n_paths, float(array), dtype=float)
    if array.ndim == 1 and array.shape[0] == n_paths:
        return array.copy()
    raise ValueError(
        f"Expected one pathwise value vector of shape ({n_paths},); received {array.shape}"
    )


def _coerce_position_matrix(values, *, n_times: int, n_paths: int):
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n_times, n_paths):
        raise ValueError(
            "Future-value position matrices must have shape "
            f"({n_times}, {n_paths}); received {matrix.shape}"
        )
    return matrix.copy()


__all__ = [
    "ConditionalValuationModel",
    "FactorStateSimulationResult",
    "HullWhiteRateProjection",
    "ObservationProgram",
    "SimulatedMarketProjection",
    "SwapFloatResetProgram",
    "HullWhiteSwapFutureValueModel",
    "build_future_value_cube",
    "evaluate_conditional_valuation_paths",
    "price_interest_rate_swap_future_value_cube",
    "simulate_factor_state_observations",
]
