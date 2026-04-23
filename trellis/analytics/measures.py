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
from datetime import timedelta
from typing import Any, Protocol

from trellis.analytics.derivative_methods import derivative_method_payload
from trellis.analytics.result import RiskMeasureOutput, ScalarRiskMeasureOutput
from trellis.core.differentiable import get_numpy, gradient
from trellis.core.market_state import MarketState
from trellis.curves.bootstrap import (
    bootstrap_curve_input_bundle_from_payload,
    bootstrap_yield_curve,
    build_bootstrap_quote_buckets,
    bump_bootstrap_quote_buckets,
)
from trellis.curves.scenario_packs import build_rate_curve_scenario_pack
from trellis.curves.shocks import build_curve_shock_surface
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.models.vol_surface_shocks import build_vol_surface_shock_surface

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
        """Market data capabilities needed (e.g., 'discount_curve', 'black_vol_surface')."""
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
        "fixing_histories": ms.fixing_histories,
        "forecast_curves": ms.forecast_curves,
        "selected_curve_names": ms.selected_curve_names,
        "market_provenance": ms.market_provenance,
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


def _warning_payload_from_parts(
    code: str,
    message: str,
    **details,
) -> dict[str, Any]:
    payload = {
        "code": str(code),
        "message": str(message),
    }
    payload.update({key: value for key, value in details.items() if value is not None})
    return payload


def _declared_risk_support(component, capability: str) -> dict[str, Any] | None:
    support = getattr(component, "risk_derivative_support", None)
    if not isinstance(support, dict):
        return None
    resolved = support.get(capability)
    if not isinstance(resolved, dict):
        return None
    return dict(resolved)


def _parallel_rate_bundle_fallback_reason(discount) -> dict[str, Any]:
    curve_type = None if discount is None else type(discount).__name__
    support = _declared_risk_support(discount, "parallel_rate_bundle")
    if discount is None:
        return _warning_payload_from_parts(
            "autodiff_public_curve_unavailable",
            "No discount curve is available for autodiff rate sensitivities.",
        )
    if support is None:
        return _warning_payload_from_parts(
            "autodiff_public_curve_unavailable",
            f"{curve_type} does not declare public autodiff support for rate sensitivities.",
            curve_type=curve_type,
        )
    if str(support.get("method") or "") != "autodiff_public_curve":
        return _warning_payload_from_parts(
            "autodiff_public_curve_unavailable",
            f"{curve_type} does not expose the autodiff_public_curve rate-derivative contract.",
            curve_type=curve_type,
            declared_method=support.get("method"),
        )
    if getattr(discount, "tenors", None) is None or getattr(discount, "rates", None) is None:
        return _warning_payload_from_parts(
            "autodiff_public_curve_unavailable",
            f"{curve_type} declares autodiff_public_curve support but does not expose tenor/rate nodes.",
            curve_type=curve_type,
        )
    return _warning_payload_from_parts(
        "autodiff_public_curve_unavailable",
        f"{curve_type} could not be reconstructed through the declared public autodiff curve contract.",
        curve_type=curve_type,
    )


def _autodiff_rate_bundle_failure_reason(
    ms: MarketState,
    exc: Exception,
) -> dict[str, Any]:
    discount = ms.discount
    support = _declared_risk_support(discount, "parallel_rate_bundle")
    if (
        discount is None
        or support is None
        or str(support.get("method") or "") != "autodiff_public_curve"
        or getattr(discount, "tenors", None) is None
        or getattr(discount, "rates", None) is None
    ):
        return _parallel_rate_bundle_fallback_reason(discount)
    return _warning_payload_from_parts(
        "autodiff_price_trace_failed",
        "Autodiff rate sensitivities could not trace the pricing function through the declared public curve contract.",
        curve_type=type(discount).__name__,
        error_type=type(exc).__name__,
    )


def _rate_measure_metadata(
    ms: MarketState,
    *,
    resolved_derivative_method: str,
    parameterization: str | None = None,
    bump_bps: float | None = None,
    fallback_reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return derivative_method_payload(
        resolved_derivative_method,
        selected_curve_name=ms.selected_curve_name("discount_curve"),
        resolved_curve_type=None if ms.discount is None else type(ms.discount).__name__,
        parameterization=None if parameterization is None else str(parameterization),
        bump_bps=None if bump_bps is None else float(bump_bps),
        fallback_reason=fallback_reason,
    )


def _scalar_risk_output(value, *, metadata: dict[str, Any]) -> ScalarRiskMeasureOutput:
    return ScalarRiskMeasureOutput(value, metadata=metadata)


def _autodiff_rate_bundle(
    payoff,
    ms: MarketState,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all rate sensitivities in one autodiff pass (DV01, duration, convexity, KRDs).

    Only works when the public discount curve declares the
    ``autodiff_public_curve`` support contract and exposes tenor/rate nodes.
    Raises ``TypeError`` otherwise, in which case callers should fall back
    to finite-difference bumping.
    """

    discount = ms.discount
    support = _declared_risk_support(discount, "parallel_rate_bundle")
    if support is None or str(support.get("method") or "") != "autodiff_public_curve":
        raise TypeError("Autodiff curve sensitivities require a declared public autodiff curve.")
    tenors = getattr(discount, "tenors", None)
    rates = getattr(discount, "rates", None)
    if tenors is None or rates is None:
        raise TypeError("Autodiff curve sensitivities require exposed tenor/rate nodes.")

    tenors_arr = np.asarray(tenors, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    curve_cls = type(discount)

    def price_from_rates(rates_vec):
        traced_ms = _clone_market_state(
            ms,
            discount=curve_cls(tenors_arr, rates_vec),
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
    metadata = _rate_measure_metadata(
        ms,
        resolved_derivative_method="autodiff_public_curve",
        parameterization=support.get("parameterization"),
        fallback_reason=None,
    )
    return {
        "price": price,
        "gradient": grad,
        "dv01": -float(np.sum(grad)) * 0.0001,
        "duration": duration,
        "convexity": 0.0 if price == 0.0 else d2p_dy2 / price,
        "key_rate_durations": key_rate_durations,
        "metadata": metadata,
    }


def _cached_rate_bundle(payoff, ms: MarketState, ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Return the shared autodiff bundle if it can be built for this payoff."""
    cache = ctx.setdefault("_cache", {})
    if "autodiff_rate_bundle" in cache:
        return cache["autodiff_rate_bundle"]
    try:
        cache["autodiff_rate_bundle"] = _autodiff_rate_bundle(payoff, ms, ctx)
        cache["autodiff_rate_bundle_failure"] = None
    except Exception as exc:
        cache["autodiff_rate_bundle"] = None
        cache["autodiff_rate_bundle_failure"] = _autodiff_rate_bundle_failure_reason(ms, exc)
    return cache["autodiff_rate_bundle"]


def _cached_rate_bundle_fallback_reason(
    ctx: dict[str, Any],
    ms: MarketState,
) -> dict[str, Any]:
    cache = ctx.setdefault("_cache", {})
    cached_reason = cache.get("autodiff_rate_bundle_failure")
    if cached_reason is not None:
        return dict(cached_reason)
    return _parallel_rate_bundle_fallback_reason(ms.discount)


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    """Present value of the instrument (what it is worth today)."""
    name: str = "price"
    requires: set = field(default_factory=lambda: {"discount_curve"})

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
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Compute DV01 with autodiff when available, otherwise bump."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            metadata = dict(bundle.get("metadata", {}))
            metadata["reporting_bump_bps"] = float(self.bump_bps)
            return _scalar_risk_output(bundle["dv01"], metadata=metadata)

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]
        return _scalar_risk_output(
            -(v_up - v_down) / 2,
            metadata=_rate_measure_metadata(
                ms,
                resolved_derivative_method="parallel_curve_bump",
                bump_bps=float(self.bump_bps),
                fallback_reason=_cached_rate_bundle_fallback_reason(ctx, ms),
            ),
        )


@dataclass(frozen=True)
class Duration:
    """Modified duration: percentage price change per 1% rate move, in years.

    A duration of 5.0 means the price drops ~5% if rates rise by 1%.
    Computed as -(1/V) * dV/dy via autodiff or finite-difference bump.
    """
    bump_bps: float = 1.0
    name: str = "duration"
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Compute modified duration with autodiff when available."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            metadata = dict(bundle.get("metadata", {}))
            metadata["reporting_bump_bps"] = float(self.bump_bps)
            return _scalar_risk_output(bundle["duration"], metadata=metadata)

        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]
        if base == 0:
            return _scalar_risk_output(
                0.0,
                metadata=_rate_measure_metadata(
                    ms,
                    resolved_derivative_method="parallel_curve_bump",
                    bump_bps=float(self.bump_bps),
                    fallback_reason=_cached_rate_bundle_fallback_reason(ctx, ms),
                ),
            )

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]

        dy = self.bump_bps / 10_000  # convert bps to decimal
        return _scalar_risk_output(
            -(v_up - v_down) / (2 * dy * base),
            metadata=_rate_measure_metadata(
                ms,
                resolved_derivative_method="parallel_curve_bump",
                bump_bps=float(self.bump_bps),
                fallback_reason=_cached_rate_bundle_fallback_reason(ctx, ms),
            ),
        )


@dataclass(frozen=True)
class Convexity:
    """Convexity: (1/V) * d²V/dy², approximated by parallel bump.

    Returns convexity in years².
    """
    bump_bps: float = 10.0
    name: str = "convexity"
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Compute convexity with autodiff when available."""
        bundle = _cached_rate_bundle(payoff, ms, ctx)
        if bundle is not None:
            metadata = dict(bundle.get("metadata", {}))
            metadata["reporting_bump_bps"] = float(self.bump_bps)
            return _scalar_risk_output(bundle["convexity"], metadata=metadata)

        if "base_price" not in ctx:
            ctx["base_price"] = payoff.evaluate(ms)
        base = ctx["base_price"]
        if base == 0:
            return _scalar_risk_output(
                0.0,
                metadata=_rate_measure_metadata(
                    ms,
                    resolved_derivative_method="parallel_curve_bump",
                    bump_bps=float(self.bump_bps),
                    fallback_reason=_cached_rate_bundle_fallback_reason(ctx, ms),
                ),
            )

        key = f"_bump_{self.bump_bps}bp"
        if key not in ctx:
            ctx[key] = _parallel_bump(payoff, ms, self.bump_bps)
        v_up, v_down = ctx[key]

        dy = self.bump_bps / 10_000
        return _scalar_risk_output(
            (v_up - 2 * base + v_down) / (dy**2 * base),
            metadata=_rate_measure_metadata(
                ms,
                resolved_derivative_method="parallel_curve_bump",
                bump_bps=float(self.bump_bps),
                fallback_reason=_cached_rate_bundle_fallback_reason(ctx, ms),
            ),
        )


@dataclass(frozen=True)
class Vega:
    """Price sensitivity to a 1 percentage point increase in volatility.

    For example, if vol moves from 20% to 21%, vega is the resulting
    price change. Uses autodiff when the vol surface is flat, otherwise
    falls back to finite-difference bumping.
    """
    bump_pct: float = 1.0
    expiries: tuple | None = None
    strikes: tuple | None = None
    name: str = "vega"
    requires: set = field(default_factory=lambda: {"discount_curve", "black_vol_surface"})

    def compute(self, payoff, ms, **ctx):
        """Compute vega with autodiff when the surface is flat."""

        vol = ms.vol_surface
        if vol is None:
            requested_buckets = _requested_vega_bucket_axes(self.expiries, self.strikes)
            if requested_buckets is None:
                return _scalar_risk_output(
                    0.0,
                    metadata=derivative_method_payload(
                        "vol_surface_unavailable",
                        resolved_surface_type=None,
                        bump_pct=float(self.bump_pct),
                        bump_vol_bps=float(self.bump_pct) * 100.0,
                    ),
                )
            expiries, strikes = requested_buckets
            return _risk_output(
                {
                    expiry: {strike: 0.0 for strike in strikes}
                    for expiry in expiries
                },
                metadata=derivative_method_payload(
                    "vol_surface_unavailable",
                    bucket_convention="expiry_strike",
                    bucket_expiries=[float(expiry) for expiry in expiries],
                    bucket_strikes=[float(strike) for strike in strikes],
                    bump_pct=float(self.bump_pct),
                    bump_vol_bps=float(self.bump_pct) * 100.0,
                    resolved_surface_type=None,
                ),
            )

        bump = self.bump_pct / 100
        cache = ctx.setdefault("_cache", {})
        requested_buckets = _requested_vega_bucket_axes(self.expiries, self.strikes)

        if requested_buckets is not None:
            return _bucketed_vega_surface(
                payoff,
                ms,
                expiries=requested_buckets[0],
                strikes=requested_buckets[1],
                bump_pct=float(self.bump_pct),
            )

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

            support = _declared_risk_support(vol, "scalar_vega") or {}
            method_id = str(support.get("method") or "autodiff_flat_vol")
            return _scalar_risk_output(
                float(gradient(price_from_vol, 0)(vol_value) * bump),
                metadata=derivative_method_payload(
                    method_id,
                    parameterization=str(support.get("parameterization") or "scalar_flat_vol"),
                    resolved_surface_type=_vol_surface_type_label(vol),
                    bump_pct=float(self.bump_pct),
                    bump_vol_bps=float(self.bump_pct) * 100.0,
                ),
            )

        if isinstance(vol, GridVolSurface):
            support = _declared_risk_support(vol, "scalar_vega") or {}
            method_id = str(support.get("method") or "surface_parallel_bucket_bump")
            return _scalar_risk_output(
                _parallel_surface_vega(
                    payoff,
                    ms,
                    expiries=tuple(float(expiry) for expiry in vol.expiries),
                    strikes=tuple(float(strike) for strike in vol.strikes),
                    bump_pct=float(self.bump_pct),
                ),
                metadata=derivative_method_payload(
                    method_id,
                    parameterization=str(support.get("parameterization") or "grid_node_vols"),
                    resolved_surface_type=_vol_surface_type_label(vol),
                    bump_pct=float(self.bump_pct),
                    bump_vol_bps=float(self.bump_pct) * 100.0,
                ),
            )

        # Bump vol surface up and down
        base_vol = float(vol.black_vol(1.0, 0.05))  # representative vol level
        fallback_reason = _warning_payload_from_parts(
            "representative_surface_reduction",
            "Scalar vega reduced the active surface to one representative flat volatility because no explicit surface derivative contract was available.",
            resolved_surface_type=_vol_surface_type_label(vol),
        )

        ms_up = _clone_market_state(ms, vol_surface=FlatVol(base_vol + bump))
        ms_down = _clone_market_state(ms, vol_surface=FlatVol(base_vol - bump))
        v_up = payoff.evaluate(ms_up)
        v_down = payoff.evaluate(ms_down)
        return _scalar_risk_output(
            (v_up - v_down) / 2,
            metadata=derivative_method_payload(
                "representative_flat_vol_bump",
                resolved_surface_type=_vol_surface_type_label(vol),
                parameterization="representative_flat_vol",
                representative_flat_vol=float(base_vol),
                bump_pct=float(self.bump_pct),
                bump_vol_bps=float(self.bump_pct) * 100.0,
                fallback_reason=fallback_reason,
            ),
        )


@dataclass(frozen=True)
class Delta:
    """Spot delta via finite-difference repricing on one selected spot binding."""

    bump_pct: float = 1.0
    underlier: str | None = None
    name: str = "delta"
    requires: set = field(default_factory=lambda: {"spot"})

    def compute(self, payoff, ms, **ctx):
        """Compute delta on the selected runtime spot binding."""
        base_spot, bind_spot, resolved_binding = _resolve_spot_bump_binding(ms, underlier=self.underlier)
        bump_size = _spot_bump_size(base_spot, self.bump_pct)
        ms_up = bind_spot(base_spot + bump_size)
        ms_down = bind_spot(base_spot - bump_size)
        return _scalar_risk_output(
            (float(payoff.evaluate(ms_up)) - float(payoff.evaluate(ms_down))) / (2.0 * bump_size),
            metadata=derivative_method_payload(
                "spot_central_bump",
                resolved_spot_binding=resolved_binding,
                underlier=self.underlier,
                bump_pct=float(self.bump_pct),
                bump_size=float(bump_size),
            ),
        )


@dataclass(frozen=True)
class Gamma:
    """Spot gamma via second-order finite-difference repricing."""

    bump_pct: float = 1.0
    underlier: str | None = None
    name: str = "gamma"
    requires: set = field(default_factory=lambda: {"spot"})

    def compute(self, payoff, ms, **ctx):
        """Compute gamma on the selected runtime spot binding."""
        base_spot, bind_spot, resolved_binding = _resolve_spot_bump_binding(ms, underlier=self.underlier)
        bump_size = _spot_bump_size(base_spot, self.bump_pct)
        base = _base_price(payoff, ms, ctx)
        ms_up = bind_spot(base_spot + bump_size)
        ms_down = bind_spot(base_spot - bump_size)
        v_up = float(payoff.evaluate(ms_up))
        v_down = float(payoff.evaluate(ms_down))
        return _scalar_risk_output(
            (v_up - 2.0 * base + v_down) / (bump_size**2),
            metadata=derivative_method_payload(
                "spot_central_bump",
                resolved_spot_binding=resolved_binding,
                underlier=self.underlier,
                bump_pct=float(self.bump_pct),
                bump_size=float(bump_size),
            ),
        )


@dataclass(frozen=True)
class Theta:
    """One-step roll-down theta using a forward settlement-date shift."""

    day_step: int = 1
    name: str = "theta"
    requires: set = field(default_factory=set)

    def compute(self, payoff, ms, **ctx):
        """Compute theta as ``V(t + dt) - V(t)`` for a calendar-day step."""
        if int(self.day_step) <= 0:
            raise ValueError("Theta requires day_step >= 1.")
        base = _base_price(payoff, ms, ctx)
        rolled_ms = _clone_market_state(
            ms,
            as_of=ms.as_of + timedelta(days=int(self.day_step)),
            settlement=ms.settlement + timedelta(days=int(self.day_step)),
        )
        return _scalar_risk_output(
            float(payoff.evaluate(rolled_ms)) - base,
            metadata=derivative_method_payload(
                "calendar_roll_down_bump",
                day_step=int(self.day_step),
            ),
        )


@dataclass(frozen=True)
class KeyRateDurations:
    """Per-tenor rate sensitivity (key rate durations).

    Bumps each tenor individually and measures the price change.
    Returns {tenor: krd} dict.
    """
    tenors: tuple | None = None
    bump_bps: float = 25.0
    methodology: str = "zero_curve"
    allow_fallback: bool = True
    name: str = "key_rate_durations"
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Compute per-tenor durations on the requested bucket grid."""
        requested_methodology = _normalize_risk_methodology(self.methodology)
        if requested_methodology == "curve_rebuild":
            rebuilt = _rebuild_quote_space_key_rate_durations(
                payoff,
                ms,
                requested_tenors=None if self.tenors is None else tuple(float(tenor) for tenor in self.tenors),
                bump_bps=float(self.bump_bps),
                ctx=ctx,
            )
            if rebuilt is not None:
                return rebuilt
            if not self.allow_fallback:
                raise ValueError("curve_rebuild KRD requested but no supported bootstrap-backed discount curve is available")

        zero_curve = _zero_curve_key_rate_durations(
            payoff,
            ms,
            tenors=_krd_tenors_for_zero_curve(ms, self.tenors),
            bump_bps=float(self.bump_bps),
            ctx=ctx,
        )
        metadata = dict(getattr(zero_curve, "metadata", {}))
        metadata["requested_methodology"] = requested_methodology
        if requested_methodology == "curve_rebuild":
            metadata["fallback_reason"] = {
                "code": "bootstrap_discount_curve_unavailable",
                "message": "No supported bootstrap-backed discount curve was available; used zero-curve bucket shocks instead.",
            }
        zero_curve.metadata = metadata
        return zero_curve


@dataclass(frozen=True)
class OAS:
    """Option-Adjusted Spread — the spread over the curve that matches market price.

    Returns OAS in basis points.
    """
    market_price: float = 100.0
    vol_fixed: bool = True
    name: str = "oas"
    requires: set = field(default_factory=lambda: {"discount_curve", "black_vol_surface"})

    def compute(self, payoff, ms, **ctx):
        """Solve for the option-adjusted spread matching ``market_price``."""
        from trellis.analytics.oas import compute_oas
        return compute_oas(
            payoff, self.market_price, ms.discount, ms.settlement,
            vol_surface=ms.vol_surface,
        )


@dataclass(frozen=True)
class OASDuration:
    """Callable-bond effective duration with optional OAS anchoring."""

    market_price: float | None = None
    bump_bps: float = 25.0
    name: str = "oas_duration"
    requires: set = field(default_factory=lambda: {"discount_curve", "black_vol_surface"})

    def compute(self, payoff, ms, **ctx):
        """Compute callable OAS duration around the current callable-tree price."""
        _resolve_callable_bond_analytics_spec(payoff)
        if ms.discount is None or not hasattr(ms.discount, "shift"):
            raise ValueError("OAS duration requires a shiftable discount curve.")

        base_price = _base_price(payoff, ms, ctx)
        target_price = float(self.market_price) if self.market_price is not None else base_price
        if target_price == 0.0:
            return 0.0

        oas_bps = 0.0
        if self.market_price is not None:
            oas_bps = _callable_market_price_oas_bps(
                payoff,
                ms,
                market_price=float(self.market_price),
            )

        curve_up = ms.discount.shift(oas_bps + float(self.bump_bps))
        curve_down = ms.discount.shift(oas_bps - float(self.bump_bps))
        ms_up = _clone_market_state(ms, discount=curve_up, forward_curve=None)
        ms_down = _clone_market_state(ms, discount=curve_down, forward_curve=None)
        v_up = float(payoff.evaluate(ms_up))
        v_down = float(payoff.evaluate(ms_down))
        dy = float(self.bump_bps) / 10_000.0
        return -(v_up - v_down) / (2.0 * dy * target_price)


@dataclass(frozen=True)
class ZSpread:
    """Z-Spread — parallel shift to match market price (ignoring optionality).

    For option-free bonds, Z-spread ≈ OAS. For callable/puttable bonds,
    Z-spread ≠ OAS — the option value is not accounted for.
    Returns z-spread in basis points.
    """
    market_price: float = 100.0
    name: str = "z_spread"
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Solve for the parallel curve shift matching ``market_price``."""
        from scipy.optimize import brentq

        def objective(bps):
            """Return the repricing error after a parallel shift of ``bps`` basis points."""
            shifted = ms.discount.shift(bps)
            ms_shifted = _clone_market_state(ms, discount=shifted, forward_curve=None)
            return payoff.evaluate(ms_shifted) - self.market_price

        return brentq(objective, -500, 500, xtol=0.01)


@dataclass(frozen=True)
class CallableScenarioExplain:
    """Callable-bond scenario explain showing optionality under rate shocks."""

    shifts_bps: tuple = (-100.0, -50.0, 50.0, 100.0)
    name: str = "callable_scenario_explain"
    requires: set = field(default_factory=lambda: {"discount_curve", "black_vol_surface"})

    def compute(self, payoff, ms, **ctx):
        """Return callable scenario explain output keyed by parallel rate shock."""
        spec = _resolve_callable_bond_analytics_spec(payoff)
        if ms.discount is None or not hasattr(ms.discount, "shift"):
            raise ValueError("Callable scenario explain requires a shiftable discount curve.")

        base_price = _base_price(payoff, ms, ctx)
        base_straight_price = _callable_straight_bond_price(ms, spec)
        base_option_value = max(base_straight_price - base_price, 0.0)
        scenario_values: dict[float, dict[str, Any]] = {}

        for shift in tuple(float(value) for value in self.shifts_bps):
            shifted_ms = _clone_market_state(
                ms,
                discount=ms.discount.shift(shift),
                forward_curve=None,
            )
            scenario_price = float(payoff.evaluate(shifted_ms))
            scenario_straight = _callable_straight_bond_price(shifted_ms, spec)
            scenario_option_value = max(scenario_straight - scenario_price, 0.0)
            option_delta = scenario_option_value - base_option_value
            scenario_values[float(shift)] = {
                "price": scenario_price,
                "straight_bond_price": scenario_straight,
                "pnl": scenario_price - base_price,
                "call_option_value": scenario_option_value,
                "call_option_value_change": option_delta,
                "exercise_incentive": (
                    "higher"
                    if option_delta > 1e-12
                    else "lower"
                    if option_delta < -1e-12
                    else "unchanged"
                ),
            }

        return _risk_output(
            scenario_values,
            metadata={
                "controller_role": "issuer",
                "schedule_role": "decision_dates",
                "exercise_dates": [exercise_date.isoformat() for exercise_date in spec.call_dates],
                "base_price": base_price,
                "base_straight_bond_price": base_straight_price,
                "base_call_option_value": base_option_value,
                "assumptions": [
                    "Callable scenario explain uses parallel curve shifts on the current callable-tree setup.",
                    "Exercise incentive is inferred from the callable discount to straight-bond value, not from a separate exercise-probability model.",
                ],
                "warnings": [],
            },
        )


@dataclass(frozen=True)
class ScenarioPnL:
    """P&L under parallel rate shifts.

    Returns {shift_bps: pnl} dict where pnl = V(shifted) - V(base).
    """
    shifts_bps: tuple = (-100, -50, +50, +100, +200)
    scenario_packs: tuple = ()
    bucket_tenors: tuple | None = None
    pack_amplitude_bps: float = 25.0
    methodology: str = "zero_curve"
    allow_fallback: bool = True
    include_parallel_shifts: bool | None = None
    name: str = "scenario_pnl"
    requires: set = field(default_factory=lambda: {"discount_curve"})

    def compute(self, payoff, ms, **ctx):
        """Return P&L relative to base value under configured rate shocks."""
        requested_methodology = _normalize_risk_methodology(self.methodology)
        include_parallel_shifts = self.include_parallel_shifts
        if include_parallel_shifts is None:
            include_parallel_shifts = not bool(self.scenario_packs)

        if requested_methodology == "curve_rebuild":
            rebuilt = _rebuild_quote_space_scenario_pnl(
                payoff,
                ms,
                shifts_bps=tuple(float(shift) for shift in self.shifts_bps),
                scenario_packs=tuple(str(pack) for pack in self.scenario_packs),
                bucket_tenors=None if self.bucket_tenors is None else tuple(float(tenor) for tenor in self.bucket_tenors),
                pack_amplitude_bps=float(self.pack_amplitude_bps),
                include_parallel_shifts=bool(include_parallel_shifts),
                ctx=ctx,
            )
            if rebuilt is not None:
                return rebuilt
            if not self.allow_fallback:
                raise ValueError("curve_rebuild scenario_pnl requested but no supported bootstrap-backed discount curve is available")

        zero_curve = _zero_curve_scenario_pnl(
            payoff,
            ms,
            shifts_bps=tuple(float(shift) for shift in self.shifts_bps),
            scenario_packs=tuple(str(pack) for pack in self.scenario_packs),
            bucket_tenors=_scenario_bucket_tenors(self.bucket_tenors),
            pack_amplitude_bps=float(self.pack_amplitude_bps),
            include_parallel_shifts=bool(include_parallel_shifts),
            ctx=ctx,
        )
        metadata = dict(getattr(zero_curve, "metadata", {}))
        metadata["requested_methodology"] = requested_methodology
        if requested_methodology == "curve_rebuild":
            metadata["fallback_reason"] = {
                "code": "bootstrap_discount_curve_unavailable",
                "message": "No supported bootstrap-backed discount curve was available; used zero-curve scenarios instead.",
            }
        zero_curve.metadata = metadata
        return zero_curve


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


def _normalize_risk_methodology(methodology: str) -> str:
    key = str(methodology or "zero_curve").strip().lower().replace("-", "_")
    aliases = {
        "zero_curve": "zero_curve",
        "curve_rebuild": "curve_rebuild",
        "rebuild": "curve_rebuild",
        "quote_rebuild": "curve_rebuild",
        "bootstrap_quote": "curve_rebuild",
    }
    if key not in aliases:
        raise ValueError(f"Unknown risk methodology: {methodology!r}")
    return aliases[key]


def _krd_tenors_for_zero_curve(ms: MarketState, tenors: tuple | None) -> tuple[float, ...]:
    if tenors is not None:
        return tuple(float(tenor) for tenor in tenors)
    discount = getattr(ms, "discount", None)
    if discount is not None and hasattr(discount, "tenors"):
        return tuple(float(tenor) for tenor in np.asarray(discount.tenors, dtype=float))
    return (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def _scenario_bucket_tenors(bucket_tenors: tuple | None) -> tuple[float, ...]:
    if bucket_tenors is None:
        return (2.0, 5.0, 10.0, 30.0)
    return tuple(float(tenor) for tenor in bucket_tenors)


def _requested_vega_bucket_axes(
    expiries: tuple | None,
    strikes: tuple | None,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if expiries is None and strikes is None:
        return None
    if expiries is None or strikes is None:
        raise ValueError("Bucketed vega requires both expiries and strikes.")
    return (
        tuple(float(expiry) for expiry in expiries),
        tuple(float(strike) for strike in strikes),
    )


def _resolve_callable_bond_analytics_spec(payoff):
    spec = getattr(payoff, "spec", None)
    if spec is None or not hasattr(spec, "call_dates"):
        raise ValueError("callable analytics currently support callable bond payoffs only.")
    call_dates = tuple(getattr(spec, "call_dates") or ())
    if not call_dates:
        raise ValueError("callable analytics require callable bonds with explicit call_dates.")
    return spec


def _callable_straight_bond_price(ms: MarketState, spec) -> float:
    from trellis.models.callable_bond_tree import straight_bond_present_value

    return float(
        straight_bond_present_value(
            ms,
            spec,
            settlement=ms.settlement,
        )
    )


def _callable_market_price_oas_bps(payoff, ms: MarketState, *, market_price: float) -> float:
    from scipy.optimize import brentq

    if ms.discount is None or not hasattr(ms.discount, "shift"):
        raise ValueError("Callable OAS analytics require a shiftable discount curve.")

    def objective(bps: float) -> float:
        shifted_ms = _clone_market_state(
            ms,
            discount=ms.discount.shift(float(bps)),
            forward_curve=None,
        )
        return float(payoff.evaluate(shifted_ms)) - float(market_price)

    return float(brentq(objective, -500.0, 500.0, xtol=0.01))


def _spot_bump_size(base_spot: float, bump_pct: float) -> float:
    base_spot = float(base_spot)
    bump_size = abs(base_spot) * float(bump_pct) / 100.0
    if not np.isfinite(base_spot) or base_spot == 0.0:
        raise ValueError("Spot delta/gamma requires a finite non-zero spot binding.")
    if bump_size <= 0.0:
        raise ValueError("Spot delta/gamma requires bump_pct > 0.")
    return float(bump_size)


def _resolve_spot_bump_binding(
    ms: MarketState,
    *,
    underlier: str | None = None,
):
    underlier_spots = dict(ms.underlier_spots or {})

    if underlier is not None:
        if underlier not in underlier_spots:
            raise ValueError(f"Unknown underlier spot {underlier!r} for delta/gamma request.")
        base_spot = float(underlier_spots[underlier])

        def _bind(value: float):
            updated_underlier_spots = dict(underlier_spots)
            updated_underlier_spots[underlier] = float(value)
            updated_spot = ms.spot
            if ms.spot is None and len(updated_underlier_spots) == 1:
                updated_spot = float(value)
            elif ms.spot is not None and np.isclose(float(ms.spot), base_spot):
                updated_spot = float(value)
            return _clone_market_state(
                ms,
                spot=updated_spot,
                underlier_spots=updated_underlier_spots,
            )

        return base_spot, _bind, f"underlier_spots[{underlier}]"

    if ms.spot is not None:
        base_spot = float(ms.spot)

        def _bind(value: float):
            updated_underlier_spots = underlier_spots or None
            if updated_underlier_spots is not None and len(updated_underlier_spots) == 1:
                only_name = next(iter(updated_underlier_spots))
                updated_underlier_spots = dict(updated_underlier_spots)
                updated_underlier_spots[only_name] = float(value)
            return _clone_market_state(
                ms,
                spot=float(value),
                underlier_spots=updated_underlier_spots,
            )

        return base_spot, _bind, "spot"

    if len(underlier_spots) == 1:
        only_name, only_spot = next(iter(underlier_spots.items()))
        base_spot = float(only_spot)

        def _bind(value: float):
            updated_underlier_spots = {only_name: float(value)}
            return _clone_market_state(
                ms,
                spot=float(value),
                underlier_spots=updated_underlier_spots,
            )

        return base_spot, _bind, f"underlier_spots[{only_name}]"

    raise ValueError(
        "Delta/gamma requires market_state.spot, one selected underlier spot, "
        "or an explicit underlier=... selection."
    )


def _risk_output(values, *, metadata: dict[str, Any]) -> RiskMeasureOutput:
    return RiskMeasureOutput(values, metadata=metadata)


def _base_price(payoff, ms: MarketState, ctx: dict[str, Any]) -> float:
    cache = ctx.setdefault("_cache", {})
    if "base_price" not in cache:
        cache["base_price"] = float(payoff.evaluate(ms))
    ctx["base_price"] = cache["base_price"]
    return float(cache["base_price"])


def _parallel_surface_vega(
    payoff,
    ms: MarketState,
    *,
    expiries: tuple[float, ...],
    strikes: tuple[float, ...],
    bump_pct: float,
) -> float:
    shock_surface = build_vol_surface_shock_surface(
        ms.vol_surface,
        expiries=expiries,
        strikes=strikes,
    )
    bump_vol_bps = float(bump_pct) * 100.0
    all_bucket_bumps = {
        (float(bucket.expiry), float(bucket.strike)): bump_vol_bps
        for bucket in shock_surface.buckets
    }

    ms_up = _clone_market_state(ms, vol_surface=shock_surface.apply_bumps(all_bucket_bumps))
    ms_down = _clone_market_state(
        ms,
        vol_surface=shock_surface.apply_bumps(
            {
                key: -float(value)
                for key, value in all_bucket_bumps.items()
            }
        ),
    )
    return (float(payoff.evaluate(ms_up)) - float(payoff.evaluate(ms_down))) / 2.0


def _bucketed_vega_surface(
    payoff,
    ms: MarketState,
    *,
    expiries: tuple[float, ...],
    strikes: tuple[float, ...],
    bump_pct: float,
) -> RiskMeasureOutput:
    shock_surface = build_vol_surface_shock_surface(
        ms.vol_surface,
        expiries=expiries,
        strikes=strikes,
    )
    bump_vol_bps = float(bump_pct) * 100.0
    values: dict[float, dict[float, float]] = {}
    for expiry in shock_surface.requested_expiries:
        row: dict[float, float] = {}
        for strike in shock_surface.requested_strikes:
            bucket_bump = {(float(expiry), float(strike)): bump_vol_bps}
            ms_up = _clone_market_state(ms, vol_surface=shock_surface.apply_bumps(bucket_bump))
            ms_down = _clone_market_state(
                ms,
                vol_surface=shock_surface.apply_bumps(
                    {(float(expiry), float(strike)): -bump_vol_bps}
                ),
            )
            row[float(strike)] = (float(payoff.evaluate(ms_up)) - float(payoff.evaluate(ms_down))) / 2.0
        values[float(expiry)] = row

    support = _declared_risk_support(ms.vol_surface, "bucketed_vega") or {}
    metadata = derivative_method_payload(
        str(support.get("method") or "surface_bucket_bump"),
        bucket_convention="expiry_strike",
        bucket_expiries=[float(expiry) for expiry in shock_surface.requested_expiries],
        bucket_strikes=[float(strike) for strike in shock_surface.requested_strikes],
        bump_pct=float(bump_pct),
        bump_vol_bps=bump_vol_bps,
        parameterization=support.get("parameterization"),
        resolved_surface_type=_vol_surface_type_label(ms.vol_surface),
        buckets=[_vol_surface_bucket_payload(bucket) for bucket in shock_surface.buckets],
        warnings=_vol_surface_warning_payloads(shock_surface),
    )
    return _risk_output(values, metadata=metadata)


def _vol_surface_type_label(surface) -> str:
    if isinstance(surface, GridVolSurface):
        return "grid"
    if isinstance(surface, FlatVol):
        return "flat"
    return type(surface).__name__


def _vol_surface_bucket_payload(bucket) -> dict[str, Any]:
    return {
        "expiry": float(bucket.expiry),
        "strike": float(bucket.strike),
        "base_vol": float(bucket.base_vol),
        "is_exact_surface_node": bool(bucket.is_exact_surface_node),
        "expiry_support": _support_payload(bucket.expiry_support),
        "strike_support": _support_payload(bucket.strike_support),
        "warnings": [_warning_payload(warning) for warning in bucket.warnings],
    }


def _vol_surface_warning_payloads(shock_surface) -> list[dict[str, Any]]:
    warnings = [_warning_payload(warning) for warning in shock_surface.warnings]
    for bucket in shock_surface.buckets:
        if bucket.is_exact_surface_node:
            continue
        warnings.append(
            {
                "code": "interpolated_surface_bucket",
                "message": "Bucketed vega uses an interpolated expiry/strike surface node.",
                "expiry": float(bucket.expiry),
                "strike": float(bucket.strike),
                "expiry_support": _support_payload(bucket.expiry_support),
                "strike_support": _support_payload(bucket.strike_support),
            }
        )
    return warnings


def _warning_payload(warning) -> dict[str, Any]:
    payload = {
        "code": str(warning.code),
        "message": str(warning.message),
    }
    if getattr(warning, "expiry", None) is not None:
        payload["expiry"] = float(warning.expiry)
    if getattr(warning, "strike", None) is not None:
        payload["strike"] = float(warning.strike)
    return payload


def _support_payload(support: tuple[float | None, float | None]) -> list[float | None]:
    return [
        None if support[0] is None else float(support[0]),
        None if support[1] is None else float(support[1]),
    ]


def _zero_curve_key_rate_durations(
    payoff,
    ms: MarketState,
    *,
    tenors: tuple[float, ...],
    bump_bps: float,
    ctx: dict[str, Any],
) -> RiskMeasureOutput:
    discount = getattr(ms, "discount", None)
    bundle = _cached_rate_bundle(payoff, ms, ctx)
    if bundle is not None:
        bundle_result: dict[float, float] = {}
        exact_bucket_match = True
        for tenor in tenors:
            resolved_value = None
            for bundle_tenor, bundle_krd in bundle["key_rate_durations"].items():
                if np.isclose(float(bundle_tenor), float(tenor)):
                    resolved_value = float(bundle_krd)
                    break
            if resolved_value is None:
                exact_bucket_match = False
                break
            bundle_result[float(tenor)] = resolved_value
        if exact_bucket_match:
            metadata = {
                "resolved_methodology": "zero_curve",
                "bucket_convention": "curve_tenor",
                "bucket_tenors": [float(tenor) for tenor in tenors],
                "bump_bps": float(bump_bps),
                **dict(bundle.get("metadata", {})),
            }
            return _risk_output(bundle_result, metadata=metadata)

    if discount is not None and hasattr(discount, "tenors") and hasattr(discount, "rates"):
        return _interpolation_aware_key_rate_durations(
            payoff,
            ms,
            tenors,
            bump_bps,
            ctx,
        )

    fallback_reason = _cached_rate_bundle_fallback_reason(ctx, ms)

    base = _base_price(payoff, ms, ctx)
    if base == 0.0:
        return _risk_output(
            {float(tenor): 0.0 for tenor in tenors},
            metadata=derivative_method_payload(
                "curve_bucket_bump",
                resolved_methodology="zero_curve",
                bucket_convention="curve_tenor",
                bucket_tenors=[float(tenor) for tenor in tenors],
                bump_bps=float(bump_bps),
                selected_curve_name=ms.selected_curve_name("discount_curve"),
                fallback_reason=dict(fallback_reason),
            ),
        )

    dy = bump_bps / 10_000.0
    result = {}
    for tenor in tenors:
        ms_up = _tenor_bumped_ms(ms, tenor, +bump_bps)
        ms_down = _tenor_bumped_ms(ms, tenor, -bump_bps)
        v_up = payoff.evaluate(ms_up)
        v_down = payoff.evaluate(ms_down)
        result[float(tenor)] = -(v_up - v_down) / (2 * dy * base)
    return _risk_output(
        result,
        metadata=derivative_method_payload(
            "curve_bucket_bump",
            resolved_methodology="zero_curve",
            bucket_convention="curve_tenor",
            bucket_tenors=[float(tenor) for tenor in tenors],
            bump_bps=float(bump_bps),
            selected_curve_name=ms.selected_curve_name("discount_curve"),
            fallback_reason=dict(fallback_reason),
        ),
    )


def _bootstrap_discount_curve_bundle(ms: MarketState):
    provenance = dict(ms.market_provenance or {})
    selected_curve_name = ms.selected_curve_name("discount_curve")
    discount_runs = dict((provenance.get("bootstrap_runs") or {}).get("discount_curves") or {})
    discount_inputs = dict((provenance.get("bootstrap_inputs") or {}).get("discount_curves") or {})

    bundle_payload = None
    resolved_curve_name = selected_curve_name
    if resolved_curve_name and resolved_curve_name in discount_runs:
        bundle_payload = dict(discount_runs[resolved_curve_name]).get("input_bundle")
    elif resolved_curve_name and resolved_curve_name in discount_inputs:
        bundle_payload = discount_inputs[resolved_curve_name]
    elif len(discount_runs) == 1:
        resolved_curve_name, run_payload = next(iter(discount_runs.items()))
        bundle_payload = dict(run_payload).get("input_bundle")
    elif len(discount_inputs) == 1:
        resolved_curve_name, bundle_payload = next(iter(discount_inputs.items()))

    if bundle_payload is None:
        return None, None
    return str(resolved_curve_name or "").strip() or None, bootstrap_curve_input_bundle_from_payload(bundle_payload)


def _select_quote_buckets(bundle, requested_tenors: tuple[float, ...] | None):
    buckets = build_bootstrap_quote_buckets(bundle)
    if requested_tenors is None:
        return buckets

    selected = []
    for tenor in requested_tenors:
        matches = [bucket for bucket in buckets if np.isclose(bucket.tenor, float(tenor))]
        if len(matches) != 1:
            return None
        selected.append(matches[0])
    return tuple(selected)


def _zero_curve_scenario_template_spec(
    scenario,
    *,
    selected_curve_name: str | None,
) -> dict[str, object]:
    spec = dict(scenario.to_pipeline_spec())
    spec["methodology"] = "zero_curve"
    spec["bucket_convention"] = "curve_tenor"
    spec["selected_curve_name"] = selected_curve_name
    return spec


def _quote_space_scenario_template_spec(
    scenario,
    *,
    selected_curve_name: str | None,
    tenor_to_bucket: dict[float, object],
) -> dict[str, object]:
    quote_bucket_bumps = {
        tenor_to_bucket[float(tenor)].bucket_id: float(bump)
        for tenor, bump in scenario.tenor_bumps.items()
    }
    spec = dict(scenario.to_pipeline_spec())
    spec["methodology"] = "curve_rebuild"
    spec["bucket_convention"] = "bootstrap_quote"
    spec["selected_curve_name"] = selected_curve_name
    spec["quote_bucket_bumps"] = quote_bucket_bumps
    return spec


def _rebuild_quote_space_key_rate_durations(
    payoff,
    ms: MarketState,
    *,
    requested_tenors: tuple[float, ...] | None,
    bump_bps: float,
    ctx: dict[str, Any],
) -> RiskMeasureOutput | None:
    selected_curve_name, bundle = _bootstrap_discount_curve_bundle(ms)
    if bundle is None:
        return None
    buckets = _select_quote_buckets(bundle, requested_tenors)
    if not buckets:
        return None

    base = _base_price(payoff, ms, ctx)
    dy = bump_bps / 10_000.0
    result = {}
    try:
        for bucket in buckets:
            up_bundle = bump_bootstrap_quote_buckets(bundle, {bucket.bucket_id: +bump_bps})
            down_bundle = bump_bootstrap_quote_buckets(bundle, {bucket.bucket_id: -bump_bps})
            up_curve = bootstrap_yield_curve(up_bundle)
            down_curve = bootstrap_yield_curve(down_bundle)
            v_up = float(payoff.evaluate(_clone_market_state(ms, discount=up_curve, forward_curve=None)))
            v_down = float(payoff.evaluate(_clone_market_state(ms, discount=down_curve, forward_curve=None)))
            result[bucket.bucket_id] = 0.0 if base == 0.0 else -(v_up - v_down) / (2 * dy * base)
    except Exception:
        return None

    metadata = derivative_method_payload(
        "bootstrap_quote_bump_rebuild",
        resolved_methodology="curve_rebuild",
        bucket_convention="bootstrap_quote",
        selected_curve_name=selected_curve_name,
        bucket_ids=[bucket.bucket_id for bucket in buckets],
        bucket_tenors=[float(bucket.tenor) for bucket in buckets],
        bucket_definitions=[bucket.to_payload() for bucket in buckets],
        bump_bps=float(bump_bps),
    )
    ctx["key_rate_durations_metadata"] = metadata
    return _risk_output(result, metadata=metadata)


def _zero_curve_scenario_pnl(
    payoff,
    ms: MarketState,
    *,
    shifts_bps: tuple[float, ...],
    scenario_packs: tuple[str, ...],
    bucket_tenors: tuple[float, ...],
    pack_amplitude_bps: float,
    include_parallel_shifts: bool,
    ctx: dict[str, Any],
) -> RiskMeasureOutput:
    base = _base_price(payoff, ms, ctx)
    result = {}
    selected_curve_name = ms.selected_curve_name("discount_curve")
    if include_parallel_shifts:
        for shift in shifts_bps:
            shifted = ms.discount.shift(shift)
            ms_shifted = _clone_market_state(ms, discount=shifted, forward_curve=None)
            result[shift] = payoff.evaluate(ms_shifted) - base

    scenarios = []
    scenario_templates = []
    if scenario_packs:
        for pack in scenario_packs:
            scenarios.extend(
                build_rate_curve_scenario_pack(
                    ms.discount,
                    pack=str(pack),
                    bucket_tenors=bucket_tenors,
                    amplitude_bps=float(pack_amplitude_bps),
                )
            )
        scenario_templates = [
            _zero_curve_scenario_template_spec(
                scenario,
                selected_curve_name=selected_curve_name,
            )
            for scenario in scenarios
        ]
        ctx["scenario_pnl_templates"] = tuple(scenario_templates)
        for scenario in scenarios:
            shifted = ms.discount.bump(scenario.tenor_bumps)
            ms_shifted = _clone_market_state(ms, discount=shifted, forward_curve=None)
            result[scenario.name] = payoff.evaluate(ms_shifted) - base

    metadata = {
        "resolved_methodology": "zero_curve",
        "bucket_convention": "curve_tenor",
        "selected_curve_name": selected_curve_name,
        "bucket_tenors": [float(tenor) for tenor in bucket_tenors],
        "parallel_shifts_bps": [float(shift) for shift in shifts_bps] if include_parallel_shifts else [],
        "include_parallel_shifts": bool(include_parallel_shifts),
        "scenario_packs": list(scenario_packs),
        "scenario_templates": scenario_templates,
        "pack_amplitude_bps": float(pack_amplitude_bps),
        "warnings": [],
        "fallback_reason": None,
    }
    ctx["scenario_pnl_metadata"] = metadata
    return _risk_output(result, metadata=metadata)


def _rebuild_quote_space_scenario_pnl(
    payoff,
    ms: MarketState,
    *,
    shifts_bps: tuple[float, ...],
    scenario_packs: tuple[str, ...],
    bucket_tenors: tuple[float, ...] | None,
    pack_amplitude_bps: float,
    include_parallel_shifts: bool,
    ctx: dict[str, Any],
) -> RiskMeasureOutput | None:
    selected_curve_name, bundle = _bootstrap_discount_curve_bundle(ms)
    if bundle is None:
        return None
    buckets = _select_quote_buckets(bundle, bucket_tenors)
    if not buckets:
        return None

    tenor_to_bucket = {}
    for bucket in buckets:
        if float(bucket.tenor) in tenor_to_bucket:
            return None
        tenor_to_bucket[float(bucket.tenor)] = bucket

    from trellis.curves.yield_curve import YieldCurve

    quote_curve = YieldCurve(
        [bucket.tenor for bucket in buckets],
        [bucket.rate_like_quote for bucket in buckets],
    )
    base = _base_price(payoff, ms, ctx)
    result = {}
    scenarios = []
    scenario_templates = []

    try:
        if include_parallel_shifts:
            for shift in shifts_bps:
                bumped_bundle = bump_bootstrap_quote_buckets(
                    bundle,
                    {bucket.bucket_id: float(shift) for bucket in buckets},
                )
                shifted_curve = bootstrap_yield_curve(bumped_bundle)
                result[shift] = payoff.evaluate(
                    _clone_market_state(ms, discount=shifted_curve, forward_curve=None)
                ) - base

        for pack in scenario_packs:
            scenarios.extend(
                build_rate_curve_scenario_pack(
                    quote_curve,
                    pack=str(pack),
                    bucket_tenors=tuple(float(bucket.tenor) for bucket in buckets),
                    amplitude_bps=float(pack_amplitude_bps),
                )
            )
        scenario_templates = [
            _quote_space_scenario_template_spec(
                scenario,
                selected_curve_name=selected_curve_name,
                tenor_to_bucket=tenor_to_bucket,
            )
            for scenario in scenarios
        ]
        ctx["scenario_pnl_templates"] = tuple(scenario_templates)
        for scenario, template_spec in zip(scenarios, scenario_templates):
            quote_bumps = dict(template_spec["quote_bucket_bumps"])
            shifted_bundle = bump_bootstrap_quote_buckets(bundle, quote_bumps)
            shifted_curve = bootstrap_yield_curve(shifted_bundle)
            result[scenario.name] = payoff.evaluate(
                _clone_market_state(ms, discount=shifted_curve, forward_curve=None)
            ) - base
    except Exception:
        return None

    metadata = {
        "resolved_methodology": "curve_rebuild",
        "bucket_convention": "bootstrap_quote",
        "selected_curve_name": selected_curve_name,
        "bucket_ids": [bucket.bucket_id for bucket in buckets],
        "bucket_tenors": [float(bucket.tenor) for bucket in buckets],
        "bucket_definitions": [bucket.to_payload() for bucket in buckets],
        "parallel_shifts_bps": [float(shift) for shift in shifts_bps] if include_parallel_shifts else [],
        "include_parallel_shifts": bool(include_parallel_shifts),
        "scenario_packs": list(scenario_packs),
        "scenario_templates": scenario_templates,
        "pack_amplitude_bps": float(pack_amplitude_bps),
        "warnings": [],
        "fallback_reason": None,
    }
    ctx["scenario_pnl_metadata"] = metadata
    return _risk_output(result, metadata=metadata)


def _interpolation_aware_key_rate_durations(
    payoff,
    ms: MarketState,
    tenors: tuple[float, ...],
    bump_bps: float,
    ctx: dict[str, Any],
) -> RiskMeasureOutput:
    """Compute KRDs on the requested bucket grid using the shared shock substrate."""
    cache = ctx.setdefault("_cache", {})
    cache_key = ("key_rate_durations", tuple(float(tenor) for tenor in tenors), float(bump_bps))
    if cache_key in cache:
        return cache[cache_key]

    surface = build_curve_shock_surface(ms.discount, tenors)
    bucket_curve = surface.bucketed_curve()
    cache["key_rate_duration_surface"] = surface
    ctx["key_rate_duration_surface"] = surface
    ctx["key_rate_duration_warnings"] = tuple(
        {
            "code": warning.code,
            "message": warning.message,
            "tenor": warning.tenor,
        }
        for warning in surface.warnings
    )

    base = _base_price(payoff, ms, ctx)
    if base == 0.0:
        fallback_reason = ctx.setdefault("_cache", {}).get("autodiff_rate_bundle_failure")
        metadata = derivative_method_payload(
            "curve_bucket_bump",
            resolved_methodology="zero_curve",
            bucket_convention="curve_tenor",
            selected_curve_name=ms.selected_curve_name("discount_curve"),
            bucket_tenors=[float(bucket.tenor) for bucket in surface.buckets],
            bucket_definitions=[
                {
                    "tenor": float(bucket.tenor),
                    "base_zero_rate": float(bucket.base_zero_rate),
                    "is_exact_curve_tenor": bool(bucket.is_exact_curve_tenor),
                    "left_support_tenor": bucket.left_support_tenor,
                    "right_support_tenor": bucket.right_support_tenor,
                    "support_width": bucket.support_width,
                }
                for bucket in surface.buckets
            ],
            bump_bps=float(bump_bps),
            warnings=list(ctx["key_rate_duration_warnings"]),
            fallback_reason=None if fallback_reason is None else dict(fallback_reason),
        )
        cache[cache_key] = _risk_output(
            {float(bucket.tenor): 0.0 for bucket in surface.buckets},
            metadata=metadata,
        )
        return cache[cache_key]

    dy = bump_bps / 10_000.0
    result: dict[float, float] = {}
    for bucket in surface.buckets:
        ms_up = _clone_market_state(
            ms,
            discount=bucket_curve.bump({bucket.tenor: +bump_bps}),
            forward_curve=None,
        )
        ms_down = _clone_market_state(
            ms,
            discount=bucket_curve.bump({bucket.tenor: -bump_bps}),
            forward_curve=None,
        )
        v_up = float(payoff.evaluate(ms_up))
        v_down = float(payoff.evaluate(ms_down))
        result[float(bucket.tenor)] = -(v_up - v_down) / (2 * dy * base)

    fallback_reason = ctx.setdefault("_cache", {}).get("autodiff_rate_bundle_failure")
    metadata = derivative_method_payload(
        "curve_bucket_bump",
        resolved_methodology="zero_curve",
        bucket_convention="curve_tenor",
        selected_curve_name=ms.selected_curve_name("discount_curve"),
        bucket_tenors=[float(bucket.tenor) for bucket in surface.buckets],
        bucket_definitions=[
            {
                "tenor": float(bucket.tenor),
                "base_zero_rate": float(bucket.base_zero_rate),
                "is_exact_curve_tenor": bool(bucket.is_exact_curve_tenor),
                "left_support_tenor": bucket.left_support_tenor,
                "right_support_tenor": bucket.right_support_tenor,
                "support_width": bucket.support_width,
            }
            for bucket in surface.buckets
        ],
        bump_bps=float(bump_bps),
        warnings=list(ctx["key_rate_duration_warnings"]),
        fallback_reason=None if fallback_reason is None else dict(fallback_reason),
    )
    cache[cache_key] = _risk_output(result, metadata=metadata)
    ctx["key_rate_durations_metadata"] = metadata
    return cache[cache_key]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MEASURE_REGISTRY: dict[str, type] = {
    "price": Price,
    "dv01": DV01,
    "duration": Duration,
    "convexity": Convexity,
    "vega": Vega,
    "delta": Delta,
    "gamma": Gamma,
    "theta": Theta,
    "key_rate_durations": KeyRateDurations,
    "krd": KeyRateDurations,
    "oas": OAS,
    "oas_duration": OASDuration,
    "z_spread": ZSpread,
    "scenario_pnl": ScenarioPnL,
    "callable_scenario_explain": CallableScenarioExplain,
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


def dsl_measure_to_runtime(measure_name: str) -> type | None:
    """Map a DSL measure name to the runtime Measure class, if available.

    Returns ``None`` when the DSL measure has no runtime implementation.
    """
    return MEASURE_REGISTRY.get(measure_name)
