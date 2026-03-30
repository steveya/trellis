"""Payoff protocol and base classes for pricing instruments.

Every priceable instrument in Trellis implements the Payoff protocol.
A payoff takes market data (via MarketState) and returns a present value.
This module also provides base classes for two common pricing patterns:

- ResolvedInputPayoff: for analytical or tree-based pricing where market
  data is extracted once, then fed into a pricing formula.
- MonteCarloPathPayoff: for simulation-based pricing where paths are
  generated and payoffs are computed per path then averaged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Generic, Protocol, TypeVar, runtime_checkable

from trellis.core.date_utils import year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Instrument

np = get_numpy()

SpecT = TypeVar("SpecT")
ResolvedT = TypeVar("ResolvedT")


@runtime_checkable
class Payoff(Protocol):
    """Protocol for anything that can be priced from a MarketState.

    ``evaluate()`` returns the present value as a float.
    Each payoff handles its own discounting internally.
    """

    @property
    def requirements(self) -> set[str]:
        """Capability names this payoff needs from MarketState."""
        ...

    def evaluate(self, market_state: MarketState) -> float:
        """Compute the present value given market data.

        The payoff is responsible for all discounting. The returned
        float is the final PV — ``price_payoff()`` returns it directly.
        """
        ...


class ResolvedInputPayoff(Generic[SpecT, ResolvedT], ABC):
    """Base class for payoffs that first extract market data, then price.

    Subclasses implement two steps:
    1. resolve_inputs() — pull the needed numbers from MarketState
       (e.g. spot price, volatility, discount factor) into a typed dataclass.
    2. evaluate_from_resolved() — compute the present value from those numbers.

    This separation lets the pricing formula stay independent of MarketState,
    making it easier to test and to support automatic differentiation.
    """

    def __init__(self, spec: SpecT):
        self._spec = spec

    @property
    def spec(self) -> SpecT:
        """Return the immutable contract specification."""
        return self._spec

    @property
    def requirements(self) -> set[str]:
        """Capability names this payoff needs from MarketState."""
        return set()

    @abstractmethod
    def resolve_inputs(self, market_state: MarketState) -> ResolvedT:
        """Resolve and normalize market inputs before pricing."""

    @abstractmethod
    def evaluate_from_resolved(self, resolved: ResolvedT) -> float:
        """Compute the present value from normalized resolved inputs."""

    def evaluate_raw(self, resolved: ResolvedT) -> float:
        """Compute the present value from resolved inputs (differentiation-safe).

        This method exists so that automatic differentiation (autograd) can
        trace through the pricing calculation. Override it when your pricing
        formula needs special handling for differentiability. By default it
        just calls ``evaluate_from_resolved()``.
        """
        return self.evaluate_from_resolved(resolved)

    def evaluate_at_expiry(self, resolved: ResolvedT) -> float:
        """Handle the deterministic expiry case for resolved inputs."""
        return float(self.evaluate_raw(resolved))

    def resolve_time_to_expiry(self, resolved: ResolvedT) -> float | None:
        """Read time-to-expiry from common resolved-input field names."""
        for field_name in ("T", "time_to_expiry"):
            value = getattr(resolved, field_name, None)
            if value is not None:
                return float(value)
        return None

    def evaluate(self, market_state: MarketState) -> float:
        """Resolve inputs once, route expiry cleanly, then delegate pricing."""
        resolved = self.resolve_inputs(market_state)
        time_to_expiry = self.resolve_time_to_expiry(resolved)
        if time_to_expiry is not None and time_to_expiry <= 0.0:
            return float(self.evaluate_at_expiry(resolved))
        return float(self.evaluate_raw(resolved))


class MonteCarloPathPayoff(ResolvedInputPayoff[SpecT, ResolvedT], ABC):
    """Base class for payoffs priced by Monte Carlo simulation.

    Subclasses define:
    - build_process() — the random model driving the simulation (e.g. GBM).
    - build_initial_state() — the starting value (e.g. spot price).
    - pathwise_payoff() — how to compute the payoff from each simulated path.

    The base class handles engine setup, path normalization, discounting,
    and averaging across paths.
    """

    @abstractmethod
    def build_process(self, resolved: ResolvedT):
        """Create the random process to simulate (e.g. GBM, Heston)."""

    @abstractmethod
    def build_initial_state(self, resolved: ResolvedT):
        """Return the starting value for the simulation (e.g. spot price)."""

    @abstractmethod
    def pathwise_payoff(self, paths, resolved: ResolvedT):
        """Compute the payoff for each simulated path, before discounting.

        Args:
            paths: Array of shape (n_paths, n_steps, n_assets). Each row is
                one simulated price trajectory.
            resolved: The pre-extracted market data.

        Returns:
            Array of per-path payoff values (not yet discounted to present).
        """

    def engine_kwargs(self, resolved: ResolvedT) -> dict[str, object]:
        """Return simulation settings (path count, step count, seed, method).

        Reads overrides from self.spec if present, otherwise uses defaults.
        """
        return {
            "n_paths": max(int(getattr(self.spec, "n_paths", 10000)), 4096),
            "n_steps": max(int(getattr(self.spec, "n_steps", 100)), 64),
            "seed": getattr(self.spec, "seed", 42),
            "method": str(getattr(self.spec, "mc_method", "exact")),
        }

    def build_engine(self, process, resolved: ResolvedT):
        """Create a MonteCarloEngine configured with the given process."""
        from trellis.models.monte_carlo.engine import MonteCarloEngine

        return MonteCarloEngine(process, **self.engine_kwargs(resolved))

    def time_horizon(self, resolved: ResolvedT) -> float:
        """Read the simulation horizon from resolved inputs."""
        time_to_expiry = self.resolve_time_to_expiry(resolved)
        if time_to_expiry is None:
            raise ValueError(
                "MonteCarloPathPayoff requires resolved inputs with `T` or `time_to_expiry`."
            )
        return float(time_to_expiry)

    def normalize_paths(self, paths):
        """Reshape simulation output to a consistent 3-D array.

        The engine may return 2-D (n_paths, n_steps) for single-asset models.
        This adds a trailing dimension so all downstream code can assume
        shape (n_paths, n_steps, n_assets).
        """
        normalized = np.asarray(paths, dtype=float)
        if normalized.ndim == 2:
            normalized = normalized[:, :, np.newaxis]
        if normalized.ndim != 3:
            raise ValueError(
                f"Expected Monte Carlo paths with rank 2 or 3; received shape {normalized.shape}."
            )
        return normalized

    def discount_factor(self, resolved: ResolvedT) -> float:
        """Look up the discount factor from resolved inputs.

        Checks several common field names (domestic_df, discount_factor, etc.).
        Returns 1.0 if no discount factor is found (i.e. no discounting).
        """
        for field_name in (
            "domestic_df",
            "discount_factor",
            "domestic_discount_factor",
        ):
            value = getattr(resolved, field_name, None)
            if value is not None:
                return float(value)
        return 1.0

    def payoff_scale(self, resolved: ResolvedT) -> float:
        """Return the notional amount used to scale payoff values.

        Reads from self.spec.notional; defaults to 1.0 if not set.
        """
        return float(getattr(self.spec, "notional", 1.0))

    def aggregate_pathwise_payoff(self, payoff_samples, resolved: ResolvedT) -> float:
        """Average per-path payoffs and apply discounting and notional scaling.

        Final price = notional * discount_factor * mean(payoff_samples).
        """
        samples = np.asarray(payoff_samples, dtype=float)
        return (
            self.payoff_scale(resolved)
            * self.discount_factor(resolved)
            * float(np.mean(samples))
        )

    def evaluate_from_resolved(self, resolved: ResolvedT) -> float:
        """Simulate, normalize, aggregate, and discount pathwise payoffs."""
        process = self.build_process(resolved)
        engine = self.build_engine(process, resolved)
        paths = self.normalize_paths(
            engine.simulate(self.build_initial_state(resolved), self.time_horizon(resolved))
        )
        return float(
            self.aggregate_pathwise_payoff(self.pathwise_payoff(paths, resolved), resolved)
        )


class DeterministicCashflowPayoff:
    r"""Adapter: wraps any Instrument (e.g. Bond) into the Payoff protocol.

    Discounts each cashflow using ``market_state.discount``. This is the
    bridge between instrument objects that expose dated cashflows and the
    payoff protocol that expects a single present-value evaluation.

    Mathematically, this computes

    .. math::

       PV = \sum_i CF_i \cdot D(t_i)

    where ``CF_i`` are the future cashflows and ``D(t_i)`` are discount
    factors from the session or market state.
    """

    def __init__(self, instrument: Instrument,
                 day_count: DayCountConvention = DayCountConvention.ACT_365):
        """Wrap an instrument plus the day-count convention for discount timing."""
        self._instrument = instrument
        self._day_count = day_count

    @property
    def instrument(self) -> Instrument:
        """Return the wrapped instrument object."""
        return self._instrument

    @property
    def requirements(self) -> set[str]:
        """Declare that deterministic cashflow pricing needs a discount curve."""
        return {"discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        """Discount each future cashflow date and sum the resulting PV."""
        schedule = self._instrument.cashflows(market_state.settlement)
        pv = 0.0
        for d, amt in zip(schedule.dates, schedule.amounts):
            t = year_fraction(market_state.settlement, d, self._day_count)
            pv += amt * market_state.discount.discount(t)
        return pv


# Backward-compat aliases (deprecated)
class Cashflows:
    """Deprecated. evaluate() now returns float directly."""
    def __init__(self, flows):
        """Store legacy raw cashflow tuples for backward compatibility only."""
        self.flows = flows

class PresentValue:
    """Deprecated. evaluate() now returns float directly."""
    def __init__(self, pv):
        """Store a legacy pre-discounted scalar PV for backward compatibility."""
        self.pv = pv
