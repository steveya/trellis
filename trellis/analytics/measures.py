"""Composable, parameterized analytics measures.

Each measure knows:
- What it computes (name, description)
- What it needs (market data, parameters)
- How to compute it (given a payoff + market state)

Measures are the building blocks of analyze(). Users specify them as:
- Strings: "price", "dv01" → instantiated with defaults
- Dicts: {"oas": {"market_price": 95.0}} → instantiated with params
- Objects: OAS(market_price=95.0) → used directly

The registry maps names to measure classes for string/dict dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from trellis.core.market_state import MarketState


class Measure(Protocol):
    """Protocol for analytics measures."""

    @property
    def name(self) -> str:
        """Human-readable measure identifier used in request specifications."""
        ...

    @property
    def requires(self) -> set[str]:
        """Market data capabilities needed (e.g., 'discount', 'black_vol')."""
        ...

    def compute(self, payoff, market_state: MarketState, **context) -> Any:
        """Compute the measure.

        Parameters
        ----------
        payoff : Payoff
            The instrument to analyze.
        market_state : MarketState
            Current market snapshot.
        context : dict
            Shared computation context — measures can store intermediate
            results here to avoid redundant repricing.

        Returns
        -------
        Any
            The measure value (float, dict, list, etc.).
        """
        ...


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    """Present value of the instrument."""
    name: str = "price"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Return the base present value, caching the first evaluation."""
        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        return ctx["base_price"]


@dataclass(frozen=True)
class DV01:
    """Dollar value of a 1bp parallel shift.

    DV01 = -(V(+1bp) - V(-1bp)) / 2
    """
    bump_bps: float = 1.0
    name: str = "dv01"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Approximate DV01 with a central finite-difference bump."""
        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]
        return -(v_up - v_down) / 2


@dataclass(frozen=True)
class Duration:
    """Modified duration: -(1/V) * dV/dy, approximated by parallel bump.

    Returns duration in years.
    """
    bump_bps: float = 1.0
    name: str = "duration"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Approximate modified duration from the parallel bump repricing."""
        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]
        if base == 0:
            return 0.0

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]

        dy = self.bump_bps / 10_000  # convert bps to decimal
        return -(v_up - v_down) / (2 * dy * base)


@dataclass(frozen=True)
class Convexity:
    """Convexity: (1/V) * d²V/dy², approximated by parallel bump.

    Returns convexity in years².
    """
    bump_bps: float = 10.0
    name: str = "convexity"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Approximate convexity from a second-order parallel bump stencil."""
        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]
        if base == 0:
            return 0.0

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]

        dy = self.bump_bps / 10_000
        return (v_up - 2 * base + v_down) / (dy**2 * base)


@dataclass(frozen=True)
class Vega:
    """Sensitivity to a 1% absolute bump in volatility.

    vega = (V(σ+bump) - V(σ-bump)) / 2
    """
    bump_pct: float = 1.0
    name: str = "vega"
    requires: set = field(default_factory=lambda: {"discount", "black_vol"})

    def compute(self, payoff, ms, **ctx):
        """Approximate vega with a symmetric absolute volatility bump."""
        from trellis.models.vol_surface import FlatVol

        vol = ms.vol_surface
        if vol is None:
            return 0.0

        # Bump vol surface up and down
        base_vol = vol.black_vol(1.0, 0.05)  # representative vol level
        bump = self.bump_pct / 100

        ms_up = MarketState(
            as_of=ms.as_of, settlement=ms.settlement,
            discount=ms.discount,
            vol_surface=FlatVol(base_vol + bump),
            credit_curve=ms.credit_curve,
            forecast_curves=ms.forecast_curves,
            fx_rates=ms.fx_rates,
        )
        ms_down = MarketState(
            as_of=ms.as_of, settlement=ms.settlement,
            discount=ms.discount,
            vol_surface=FlatVol(base_vol - bump),
            credit_curve=ms.credit_curve,
            forecast_curves=ms.forecast_curves,
            fx_rates=ms.fx_rates,
        )
        v_up = payoff.evaluate(ms_up)
        v_down = payoff.evaluate(ms_down)
        return (v_up - v_down) / 2


@dataclass(frozen=True)
class KeyRateDurations:
    """Per-tenor rate sensitivity (key rate durations).

    Bumps each tenor individually and measures the price change.
    Returns {tenor: krd} dict.
    """
    tenors: tuple = (1, 2, 3, 5, 7, 10, 20, 30)
    bump_bps: float = 25.0
    name: str = "key_rate_durations"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Compute per-tenor durations by bumping one tenor at a time."""
        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]
        if base == 0:
            return {}

        dy = self.bump_bps / 10_000
        result = {}
        for tenor in self.tenors:
            ms_up = _tenor_bumped_ms(ms, tenor, +self.bump_bps)
            ms_down = _tenor_bumped_ms(ms, tenor, -self.bump_bps)
            v_up = payoff.evaluate(ms_up)
            v_down = payoff.evaluate(ms_down)
            result[tenor] = -(v_up - v_down) / (2 * dy * base)

        return result


@dataclass(frozen=True)
class OAS:
    """Option-Adjusted Spread — the spread over the curve that matches market price.

    Returns OAS in basis points.
    """
    market_price: float = 100.0
    vol_fixed: bool = True
    name: str = "oas"
    requires: set = field(default_factory=lambda: {"discount", "black_vol"})

    def compute(self, payoff, ms, **ctx):
        """Solve for the option-adjusted spread matching ``market_price``."""
        from trellis.analytics.oas import compute_oas
        return compute_oas(
            payoff, self.market_price, ms.discount, ms.settlement,
            vol_surface=ms.vol_surface,
        )


@dataclass(frozen=True)
class ZSpread:
    """Z-Spread — parallel shift to match market price (ignoring optionality).

    For option-free bonds, Z-spread ≈ OAS. For callable/puttable bonds,
    Z-spread ≠ OAS — the option value is not accounted for.
    Returns z-spread in basis points.
    """
    market_price: float = 100.0
    name: str = "z_spread"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Solve for the parallel curve shift matching ``market_price``."""
        from scipy.optimize import brentq

        def objective(bps):
            """Return the repricing error after a parallel shift of ``bps`` basis points."""
            shifted = ms.discount.shift(bps)
            ms_shifted = MarketState(
                as_of=ms.as_of, settlement=ms.settlement,
                discount=shifted, vol_surface=ms.vol_surface,
                credit_curve=ms.credit_curve,
                forecast_curves=ms.forecast_curves,
                fx_rates=ms.fx_rates,
            )
            return payoff.evaluate(ms_shifted) - self.market_price

        return brentq(objective, -500, 500, xtol=0.01)


@dataclass(frozen=True)
class ScenarioPnL:
    """P&L under parallel rate shifts.

    Returns {shift_bps: pnl} dict where pnl = V(shifted) - V(base).
    """
    shifts_bps: tuple = (-100, -50, +50, +100, +200)
    name: str = "scenario_pnl"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Return P&L relative to base value under configured rate shocks."""
        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]

        result = {}
        for shift in self.shifts_bps:
            shifted = ms.discount.shift(shift)
            ms_shifted = MarketState(
                as_of=ms.as_of, settlement=ms.settlement,
                discount=shifted, vol_surface=ms.vol_surface,
                credit_curve=ms.credit_curve,
                forecast_curves=ms.forecast_curves,
                fx_rates=ms.fx_rates,
            )
            v = payoff.evaluate(ms_shifted)
            result[shift] = v - base
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parallel_bump(payoff, ms, bump_bps):
    """Reprice under a symmetric parallel bump and return ``(v_up, v_down)``."""
    shifted_up = ms.discount.shift(+bump_bps)
    shifted_down = ms.discount.shift(-bump_bps)

    ms_up = MarketState(
        as_of=ms.as_of, settlement=ms.settlement,
        discount=shifted_up, vol_surface=ms.vol_surface,
        credit_curve=ms.credit_curve,
        forecast_curves=ms.forecast_curves,
        fx_rates=ms.fx_rates,
    )
    ms_down = MarketState(
        as_of=ms.as_of, settlement=ms.settlement,
        discount=shifted_down, vol_surface=ms.vol_surface,
        credit_curve=ms.credit_curve,
        forecast_curves=ms.forecast_curves,
        fx_rates=ms.fx_rates,
    )
    return payoff.evaluate(ms_up), payoff.evaluate(ms_down)


def _tenor_bumped_ms(ms, tenor, bump_bps):
    """Create a market state with a single discount-curve tenor bumped."""
    bumped = ms.discount.bump({tenor: bump_bps})
    return MarketState(
        as_of=ms.as_of, settlement=ms.settlement,
        discount=bumped, vol_surface=ms.vol_surface,
        credit_curve=ms.credit_curve,
        forecast_curves=ms.forecast_curves,
        fx_rates=ms.fx_rates,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MEASURE_REGISTRY: dict[str, type] = {
    "price": Price,
    "dv01": DV01,
    "duration": Duration,
    "convexity": Convexity,
    "vega": Vega,
    "key_rate_durations": KeyRateDurations,
    "krd": KeyRateDurations,
    "oas": OAS,
    "z_spread": ZSpread,
    "scenario_pnl": ScenarioPnL,
}


# Convenient short aliases
KRD = KeyRateDurations


def resolve_measures(specs) -> list:
    """Normalize measure specs into Measure objects.

    Accepts:
    - str: "dv01" → DV01()
    - dict: {"oas": {"market_price": 95.0}} → OAS(market_price=95.0)
    - Measure object: OAS(market_price=95.0) → pass through
    """
    measures = []
    for spec in specs:
        if isinstance(spec, str):
            cls = MEASURE_REGISTRY.get(spec)
            if cls is None:
                raise ValueError(f"Unknown measure: {spec!r}. "
                                 f"Available: {sorted(MEASURE_REGISTRY.keys())}")
            measures.append(cls())
        elif isinstance(spec, dict):
            for name, params in spec.items():
                cls = MEASURE_REGISTRY.get(name)
                if cls is None:
                    raise ValueError(f"Unknown measure: {name!r}")
                measures.append(cls(**params))
        else:
            # Assume it's already a Measure object
            measures.append(spec)
    return measures
