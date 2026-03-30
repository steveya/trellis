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

from trellis.core.differentiable import get_numpy, gradient
from trellis.core.market_state import MarketState
from trellis.curves.interpolation import linear_interp
from trellis.models.vol_surface import FlatVol

np = get_numpy()


class Measure(Protocol):
    """Interface for a single analytics calculation (e.g. price, DV01, vega).

    Each measure knows what market data it needs and how to compute itself
    from a payoff and a market state. Measures share a context dict to
    cache intermediate results and avoid repricing the same instrument twice.
    """

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
            Shared cache across measures. Measures can store/read
            intermediate results here (e.g. base price, rate gradient)
            to avoid computing the same thing twice.

        Returns
        -------
        Any
            The measure value (float, dict, list, etc.).
        """
        ...


@dataclass(frozen=True)
class _AutodiffDiscountCurve:
    """Minimal differentiable discount curve used for sensitivity extraction."""

    tenors: Any
    rates: Any

    def zero_rate(self, t: float) -> float:
        """Return the interpolated zero rate at time *t*."""
        return linear_interp(t, self.tenors, self.rates)

    def discount(self, t: float) -> float:
        """Return the discount factor implied by the traced rate vector."""
        return np.exp(-self.zero_rate(t) * t)


def _clone_market_state(ms: MarketState, **overrides) -> MarketState:
    """Clone a market state while preserving all optional market components."""
    data = {
        "as_of": ms.as_of,
        "settlement": ms.settlement,
        "discount": ms.discount,
        "forward_curve": ms.forward_curve,
        "vol_surface": ms.vol_surface,
        "state_space": ms.state_space,
        "credit_curve": ms.credit_curve,
        "forecast_curves": ms.forecast_curves,
        "fx_rates": ms.fx_rates,
        "spot": ms.spot,
        "underlier_spots": ms.underlier_spots,
        "local_vol_surface": ms.local_vol_surface,
        "local_vol_surfaces": ms.local_vol_surfaces,
        "jump_parameters": ms.jump_parameters,
        "jump_parameter_sets": ms.jump_parameter_sets,
        "model_parameters": ms.model_parameters,
        "model_parameter_sets": ms.model_parameter_sets,
    }
    data.update(overrides)
    return MarketState(**data)


def _autodiff_rate_bundle(
    payoff,
    ms: MarketState,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all rate sensitivities in one autodiff pass (DV01, duration, convexity, KRDs).

    Only works when the discount curve has ``tenors`` and ``rates`` attributes
    (i.e. a YieldCurve). Raises TypeError otherwise, in which case callers
    should fall back to finite-difference bumping.
    """

    discount = ms.discount
    tenors = getattr(discount, "tenors", None)
    rates = getattr(discount, "rates", None)
    if tenors is None or rates is None:
        raise TypeError("Autodiff curve sensitivities require a tenor/rate discount curve.")

    tenors_arr = np.asarray(tenors, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)

    def price_from_rates(rates_vec):
        traced_ms = _clone_market_state(
            ms,
            discount=_AutodiffDiscountCurve(tenors_arr, rates_vec),
            forward_curve=None,
        )
        return payoff.evaluate(traced_ms)

    cache = ctx.get("_cache", {}) if ctx is not None else {}
    if "base_price" in cache:
        price = float(cache["base_price"])
    else:
        price = float(price_from_rates(rates_arr))
    if price == 0.0:
        grad = np.zeros_like(rates_arr, dtype=float)
    else:
        grad = gradient(price_from_rates, 0)(rates_arr)

    def shifted_price(shift):
        return price_from_rates(rates_arr + shift)

    d2p_dy2 = float(gradient(gradient(shifted_price, 0), 0)(0.0))

    key_rate_durations: dict[Any, float] = {}
    for idx, tenor in enumerate(tenors_arr):
        key_rate_durations[tenor] = 0.0 if price == 0.0 else -float(grad[idx]) / price

    duration = 0.0 if price == 0.0 else -float(np.sum(grad)) / price
    return {
        "price": price,
        "gradient": grad,
        "dv01": -float(np.sum(grad)) * 0.0001,
        "duration": duration,
        "convexity": 0.0 if price == 0.0 else d2p_dy2 / price,
        "key_rate_durations": key_rate_durations,
    }


def _cached_rate_bundle(payoff, ms: MarketState, ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Return the shared autodiff bundle if it can be built for this payoff."""
    cache = ctx.setdefault("_cache", {})
    if "autodiff_rate_bundle" in cache:
        return cache["autodiff_rate_bundle"]
    try:
        cache["autodiff_rate_bundle"] = _autodiff_rate_bundle(payoff, ms, ctx)
    except Exception:
        cache["autodiff_rate_bundle"] = None
    return cache["autodiff_rate_bundle"]


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    """Present value of the instrument (what it is worth today)."""
    name: str = "price"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Return the base present value, caching the first evaluation."""
        cache = ctx.setdefault("_cache", {})
        if "base_price" in cache:
            ctx["base_price"] = cache["base_price"]
            return cache["base_price"]
        bundle = cache.get("autodiff_rate_bundle")
        if bundle is not None and "price" in bundle:
            cache["base_price"] = bundle["price"]
            ctx["base_price"] = bundle["price"]
            return bundle["price"]
        if "base_price" not in cache:
            cache["base_price"] = payoff.evaluate(ms)
        ctx["base_price"] = cache["base_price"]
        return cache["base_price"]


@dataclass(frozen=True)
class DV01:
    """Dollar value of a 1 basis point (0.01%) parallel rate shift.

    Measures how much the price changes when all interest rates move up by
    1bp. A DV01 of $45 means the price drops ~$45 if rates rise by 1bp.
    """
    bump_bps: float = 1.0
    name: str = "dv01"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Compute DV01 with autodiff when available, otherwise bump."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            return bundle["dv01"]

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]
        return -(v_up - v_down) / 2


@dataclass(frozen=True)
class Duration:
    """Modified duration: percentage price change per 1% rate move, in years.

    A duration of 5.0 means the price drops ~5% if rates rise by 1%.
    Computed as -(1/V) * dV/dy via autodiff or finite-difference bump.
    """
    bump_bps: float = 1.0
    name: str = "duration"
    requires: set = field(default_factory=lambda: {"discount"})

    def compute(self, payoff, ms, **ctx):
        """Compute modified duration with autodiff when available."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            return bundle["duration"]

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
        """Compute convexity with autodiff when available."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            return bundle["convexity"]

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
    """Price sensitivity to a 1 percentage point increase in volatility.

    For example, if vol moves from 20% to 21%, vega is the resulting
    price change. Uses autodiff when the vol surface is flat, otherwise
    falls back to finite-difference bumping.
    """
    bump_pct: float = 1.0
    name: str = "vega"
    requires: set = field(default_factory=lambda: {"discount", "black_vol"})

    def compute(self, payoff, ms, **ctx):
        """Compute vega with autodiff when the surface is flat."""

        vol = ms.vol_surface
        if vol is None:
            return 0.0

        bump = self.bump_pct / 100
        cache = ctx.setdefault("_cache", {})

        if isinstance(vol, FlatVol):
            base_vol = vol.vol
            vol_value = np.asarray(base_vol, dtype=float)

            def price_from_vol(vol_input):
                traced_ms = _clone_market_state(ms, vol_surface=FlatVol(vol_input))
                return payoff.evaluate(traced_ms)

            if "base_price" in cache:
                ctx["base_price"] = cache["base_price"]
            else:
                cache["base_price"] = float(price_from_vol(vol_value))
                ctx["base_price"] = cache["base_price"]

            return float(gradient(price_from_vol, 0)(vol_value) * bump)

        # Bump vol surface up and down
        base_vol = vol.black_vol(1.0, 0.05)  # representative vol level

        ms_up = _clone_market_state(ms, vol_surface=FlatVol(base_vol + bump))
        ms_down = _clone_market_state(ms, vol_surface=FlatVol(base_vol - bump))
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
        """Compute per-tenor durations with autodiff when available."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            return {
                tenor: bundle["key_rate_durations"].get(float(tenor), 0.0)
                for tenor in self.tenors
            }

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

    ms_up = _clone_market_state(ms, discount=shifted_up, forward_curve=None)
    ms_down = _clone_market_state(ms, discount=shifted_down, forward_curve=None)
    return payoff.evaluate(ms_up), payoff.evaluate(ms_down)


def _tenor_bumped_ms(ms, tenor, bump_bps):
    """Create a market state with a single discount-curve tenor bumped."""
    bumped = ms.discount.bump({tenor: bump_bps})
    return _clone_market_state(ms, discount=bumped, forward_curve=None)


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
