"""Rates-vol market-object reconstruction and staged model-compression helpers."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import isfinite, sqrt
from types import MappingProxyType
from typing import Mapping, Sequence

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.black import black76_call, black76_put
from trellis.models.calibration.materialization import materialize_black_vol_surface
from trellis.models.calibration.rates import (
    _cap_floor_terms,
    _implied_flat_vol,
    calibrate_swaption_black_vol,
    swaption_terms,
)
from trellis.models.calibration.sabr_fit import SABRSmileCalibrationResult, build_sabr_smile_surface, fit_sabr_smile_surface
from trellis.models.processes.sabr import SABRProcess
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.vol_surface import GridVolSurface


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


def _round_axis(value: float, *, digits: int = 12) -> float:
    """Return a stable rounded axis coordinate."""
    return round(float(value), digits)


def _sorted_unique(values: Sequence[float]) -> tuple[float, ...]:
    """Return a strictly increasing tuple of unique floats."""
    unique = sorted({_round_axis(value) for value in values})
    return tuple(float(value) for value in unique)


def _lerp(left: float, right: float, weight: float) -> float:
    """Return linear interpolation between ``left`` and ``right``."""
    return (1.0 - float(weight)) * float(left) + float(weight) * float(right)


def _bracket_and_weight(value: float, grid: tuple[float, ...]) -> tuple[int, int, float]:
    """Locate the bracketing nodes and interpolation weight on a sorted grid."""
    if value <= grid[0]:
        return 0, 0, 0.0
    if value >= grid[-1]:
        last = len(grid) - 1
        return last, last, 0.0

    upper = bisect_right(grid, value)
    lower = upper - 1
    left = grid[lower]
    right = grid[upper]
    if right == left:
        return lower, upper, 0.0
    return lower, upper, (float(value) - float(left)) / (float(right) - float(left))


def _market_curve_roles(market_state: MarketState) -> dict[str, str]:
    """Return the selected curve-role bindings preserved from ``MarketState``."""
    return {
        str(key): str(value)
        for key, value in dict(market_state.selected_curve_names or {}).items()
        if str(key).strip() and str(value).strip()
    }


def _market_provenance(market_state: MarketState) -> dict[str, object]:
    """Return a shallow JSON-friendly market provenance snapshot."""
    return dict(getattr(market_state, "market_provenance", None) or {})


def _strike_group_key(spec: CapFloorSpec) -> tuple[float, str, str, str, str, str]:
    """Return the grouping key for one cap/floor strip line."""
    return (
        float(spec.strike),
        str(spec.rate_index or ""),
        str(spec.frequency.name),
        str(getattr(spec.day_count, "name", spec.day_count)),
        spec.start_date.isoformat(),
        str(spec.calendar_name or ""),
    )


def _swaption_underlier_tenor_years(spec) -> float:
    """Return the underlier tenor in years for one swaption-like spec."""
    whole_months = (spec.swap_end.year - spec.swap_start.year) * 12 + (spec.swap_end.month - spec.swap_start.month)
    if spec.swap_end.day == spec.swap_start.day:
        return float(max(whole_months / 12.0, 1e-8))
    return float(max(year_fraction(spec.swap_start, spec.swap_end, DayCountConvention.ACT_365), 1e-8))


@dataclass(frozen=True)
class CapletStripQuote:
    """One cap/floor quote used for bounded caplet stripping."""

    spec: CapFloorSpec
    quote: float
    quote_kind: str = "price"
    kind: str = "cap"
    label: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.quote_kind not in {"price", "black_vol"}:
            raise ValueError("quote_kind must be 'price' or 'black_vol'")
        if self.kind not in {"cap", "floor"}:
            raise ValueError("kind must be 'cap' or 'floor'")
        if not isfinite(float(self.quote)) or float(self.quote) < 0.0:
            raise ValueError("quote must be finite and non-negative")
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "weight", float(self.weight))

    def resolved_label(self, index: int) -> str:
        """Return a stable label for one stripping quote."""
        if self.label:
            return self.label
        return (
            f"{self.kind}_{self.spec.start_date.isoformat()}_{self.spec.end_date.isoformat()}"
            f"_{float(self.spec.strike):g}_{index}"
        )

    def to_payload(self, index: int) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "label": self.resolved_label(index),
            "quote": float(self.quote),
            "quote_kind": self.quote_kind,
            "kind": self.kind,
            "weight": float(self.weight),
            "spec": {
                "notional": float(self.spec.notional),
                "strike": float(self.spec.strike),
                "start_date": self.spec.start_date.isoformat(),
                "end_date": self.spec.end_date.isoformat(),
                "frequency": self.spec.frequency.name,
                "day_count": getattr(self.spec.day_count, "name", str(self.spec.day_count)),
                "rate_index": self.spec.rate_index,
                "calendar_name": self.spec.calendar_name,
                "business_day_adjustment": self.spec.business_day_adjustment,
            },
        }


@dataclass(frozen=True)
class SwaptionCubeQuote:
    """One swaption quote used for bounded cube assembly."""

    spec: object
    quote: float
    quote_kind: str = "black_vol"
    label: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.quote_kind not in {"price", "black_vol"}:
            raise ValueError("quote_kind must be 'price' or 'black_vol'")
        if not isfinite(float(self.quote)) or float(self.quote) < 0.0:
            raise ValueError("quote must be finite and non-negative")
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "weight", float(self.weight))

    def resolved_label(self, index: int) -> str:
        """Return a stable label for one cube quote."""
        if self.label:
            return self.label
        return (
            f"swaption_{self.spec.expiry_date.isoformat()}_{self.spec.swap_end.isoformat()}"
            f"_{float(self.spec.strike):g}_{index}"
        )

    def to_payload(self, index: int) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "label": self.resolved_label(index),
            "quote": float(self.quote),
            "quote_kind": self.quote_kind,
            "weight": float(self.weight),
            "spec": {
                "notional": float(self.spec.notional),
                "strike": float(self.spec.strike),
                "expiry_date": self.spec.expiry_date.isoformat(),
                "swap_start": self.spec.swap_start.isoformat(),
                "swap_end": self.spec.swap_end.isoformat(),
                "swap_frequency": self.spec.swap_frequency.name,
                "day_count": getattr(self.spec.day_count, "name", str(self.spec.day_count)),
                "rate_index": self.spec.rate_index,
                "is_payer": bool(self.spec.is_payer),
            },
        }


@dataclass(frozen=True)
class CapletVolStripDiagnostics:
    """Diagnostics for one bounded caplet-stripping result."""

    quote_count: int
    strike_count: int
    expiry_count: int
    max_abs_repricing_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_count", int(self.quote_count))
        object.__setattr__(self, "strike_count", int(self.strike_count))
        object.__setattr__(self, "expiry_count", int(self.expiry_count))
        object.__setattr__(self, "max_abs_repricing_error", float(self.max_abs_repricing_error))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics payload."""
        return {
            "quote_count": int(self.quote_count),
            "strike_count": int(self.strike_count),
            "expiry_count": int(self.expiry_count),
            "max_abs_repricing_error": float(self.max_abs_repricing_error),
        }


@dataclass(frozen=True)
class SwaptionVolCubeDiagnostics:
    """Diagnostics for one bounded swaption-cube result."""

    quote_count: int
    slice_count: int
    expiry_count: int
    tenor_count: int
    strike_count: int
    max_abs_quote_residual: float
    max_abs_price_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_count", int(self.quote_count))
        object.__setattr__(self, "slice_count", int(self.slice_count))
        object.__setattr__(self, "expiry_count", int(self.expiry_count))
        object.__setattr__(self, "tenor_count", int(self.tenor_count))
        object.__setattr__(self, "strike_count", int(self.strike_count))
        object.__setattr__(self, "max_abs_quote_residual", float(self.max_abs_quote_residual))
        object.__setattr__(self, "max_abs_price_residual", float(self.max_abs_price_residual))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics payload."""
        return {
            "quote_count": int(self.quote_count),
            "slice_count": int(self.slice_count),
            "expiry_count": int(self.expiry_count),
            "tenor_count": int(self.tenor_count),
            "strike_count": int(self.strike_count),
            "max_abs_quote_residual": float(self.max_abs_quote_residual),
            "max_abs_price_residual": float(self.max_abs_price_residual),
        }


@dataclass(frozen=True)
class SwaptionVolCube:
    """Tenor-aware swaption Black-vol cube with trilinear interpolation."""

    expiries: tuple[float, ...]
    tenors: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[tuple[float, ...], ...], ...]
    default_tenor: float | None = None

    def __post_init__(self) -> None:
        if not self.expiries or not self.tenors or not self.strikes:
            raise ValueError("swaption cubes require non-empty expiry, tenor, and strike grids")
        if tuple(sorted(self.expiries)) != tuple(self.expiries):
            raise ValueError("expiries must be sorted ascending")
        if tuple(sorted(self.tenors)) != tuple(self.tenors):
            raise ValueError("tenors must be sorted ascending")
        if tuple(sorted(self.strikes)) != tuple(self.strikes):
            raise ValueError("strikes must be sorted ascending")
        if len(self.vols) != len(self.expiries):
            raise ValueError("vol cube expiry dimension must match expiries")
        for expiry_slice in self.vols:
            if len(expiry_slice) != len(self.tenors):
                raise ValueError("vol cube tenor dimension must match tenors")
            for tenor_slice in expiry_slice:
                if len(tenor_slice) != len(self.strikes):
                    raise ValueError("vol cube strike dimension must match strikes")
                if any(float(vol) < 0.0 for vol in tenor_slice):
                    raise ValueError("vol cube nodes must be non-negative")
        resolved_default_tenor = self.default_tenor
        if resolved_default_tenor is None:
            resolved_default_tenor = float(self.tenors[len(self.tenors) // 2])
        object.__setattr__(self, "default_tenor", float(resolved_default_tenor))

    def swaption_black_vol(self, expiry: float, strike: float, tenor: float) -> float:
        """Return a tenor-aware swaption Black vol."""
        i0, i1, w_expiry = _bracket_and_weight(float(expiry), self.expiries)
        j0, j1, w_tenor = _bracket_and_weight(float(tenor), self.tenors)
        k0, k1, w_strike = _bracket_and_weight(float(strike), self.strikes)

        def _slice_value(expiry_index: int, tenor_index: int) -> float:
            left = self.vols[expiry_index][tenor_index][k0]
            right = self.vols[expiry_index][tenor_index][k1]
            return float(_lerp(left, right, w_strike))

        lower = _lerp(_slice_value(i0, j0), _slice_value(i0, j1), w_tenor)
        upper = _lerp(_slice_value(i1, j0), _slice_value(i1, j1), w_tenor)
        return float(_lerp(lower, upper, w_expiry))

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return a Black vol using the default tenor slice."""
        return float(self.swaption_black_vol(expiry, strike, float(self.default_tenor)))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly cube payload."""
        return {
            "expiries": list(self.expiries),
            "tenors": list(self.tenors),
            "strikes": list(self.strikes),
            "vols": [[list(tenor_slice) for tenor_slice in expiry_slice] for expiry_slice in self.vols],
            "default_tenor": float(self.default_tenor),
        }


@dataclass(frozen=True)
class CapletVolStripAuthorityResult:
    """Bounded caplet-strip authority reconstructed from cap/floor quotes."""

    quotes: tuple[CapletStripQuote, ...]
    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    stripped_vols: tuple[tuple[float, ...], ...]
    vol_surface: GridVolSurface
    repriced_quotes: tuple[float, ...]
    quote_residuals: tuple[float, ...]
    diagnostics: CapletVolStripDiagnostics
    surface_name: str = "caplet_vol_strip"
    provenance: Mapping[str, object] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "expiries", tuple(float(value) for value in self.expiries))
        object.__setattr__(self, "strikes", tuple(float(value) for value in self.strikes))
        object.__setattr__(
            self,
            "stripped_vols",
            tuple(tuple(float(value) for value in row) for row in self.stripped_vols),
        )
        object.__setattr__(self, "repriced_quotes", tuple(float(value) for value in self.repriced_quotes))
        object.__setattr__(self, "quote_residuals", tuple(float(value) for value in self.quote_residuals))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` with the stripped caplet surface materialized."""
        return materialize_black_vol_surface(
            market_state,
            surface_name=self.surface_name,
            vol_surface=self.vol_surface,
            source_kind="calibrated",
            source_ref="calibrate_caplet_vol_strip_workflow",
            selected_curve_roles=_market_curve_roles(market_state),
            metadata={
                "instrument_family": "rates",
                "surface_model_family": "caplet_vol_strip",
                "surface_name": self.surface_name,
                "strike_count": len(self.strikes),
                "expiry_count": len(self.expiries),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly result payload."""
        return {
            "surface_name": self.surface_name,
            "expiries": list(self.expiries),
            "strikes": list(self.strikes),
            "stripped_vols": [list(row) for row in self.stripped_vols],
            "repriced_quotes": list(self.repriced_quotes),
            "quote_residuals": list(self.quote_residuals),
            "diagnostics": self.diagnostics.to_payload(),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class SwaptionVolCubeAuthorityResult:
    """Bounded swaption cube authority reconstructed from normalized quotes."""

    quotes: tuple[SwaptionCubeQuote, ...]
    expiries: tuple[float, ...]
    tenors: tuple[float, ...]
    strikes: tuple[float, ...]
    forwards: tuple[tuple[float, ...], ...]
    market_vols: tuple[tuple[tuple[float, ...], ...], ...]
    vol_cube: SwaptionVolCube
    repriced_vols: tuple[float, ...]
    repriced_prices: tuple[float, ...]
    quote_residuals: tuple[float, ...]
    price_residuals: tuple[float, ...]
    diagnostics: SwaptionVolCubeDiagnostics
    surface_name: str = "swaption_vol_cube"
    provenance: Mapping[str, object] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "expiries", tuple(float(value) for value in self.expiries))
        object.__setattr__(self, "tenors", tuple(float(value) for value in self.tenors))
        object.__setattr__(self, "strikes", tuple(float(value) for value in self.strikes))
        object.__setattr__(
            self,
            "forwards",
            tuple(tuple(float(value) for value in row) for row in self.forwards),
        )
        object.__setattr__(
            self,
            "market_vols",
            tuple(tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row) for expiry_row in self.market_vols),
        )
        object.__setattr__(self, "repriced_vols", tuple(float(value) for value in self.repriced_vols))
        object.__setattr__(self, "repriced_prices", tuple(float(value) for value in self.repriced_prices))
        object.__setattr__(self, "quote_residuals", tuple(float(value) for value in self.quote_residuals))
        object.__setattr__(self, "price_residuals", tuple(float(value) for value in self.price_residuals))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` with the swaption cube materialized."""
        return materialize_black_vol_surface(
            market_state,
            surface_name=self.surface_name,
            vol_surface=self.vol_cube,
            source_kind="calibrated",
            source_ref="calibrate_swaption_vol_cube_workflow",
            selected_curve_roles=_market_curve_roles(market_state),
            metadata={
                "instrument_family": "rates",
                "surface_model_family": "swaption_vol_cube",
                "surface_name": self.surface_name,
                "expiry_count": len(self.expiries),
                "tenor_count": len(self.tenors),
                "strike_count": len(self.strikes),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly result payload."""
        return {
            "surface_name": self.surface_name,
            "expiries": list(self.expiries),
            "tenors": list(self.tenors),
            "strikes": list(self.strikes),
            "forwards": [list(row) for row in self.forwards],
            "market_vols": [[list(tenor_row) for tenor_row in expiry_row] for expiry_row in self.market_vols],
            "repriced_vols": list(self.repriced_vols),
            "repriced_prices": list(self.repriced_prices),
            "quote_residuals": list(self.quote_residuals),
            "price_residuals": list(self.price_residuals),
            "diagnostics": self.diagnostics.to_payload(),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class SwaptionCubeStageComparisonResult:
    """Comparison between the market swaption cube and per-slice SABR compression."""

    authority_result: SwaptionVolCubeAuthorityResult
    sabr_slice_results: tuple[SABRSmileCalibrationResult, ...]
    model_vols: tuple[tuple[tuple[float, ...], ...], ...]
    residuals: tuple[tuple[tuple[float, ...], ...], ...]
    max_abs_vol_error: float
    rms_vol_error: float
    summary: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sabr_slice_results", tuple(self.sabr_slice_results))
        object.__setattr__(
            self,
            "model_vols",
            tuple(tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row) for expiry_row in self.model_vols),
        )
        object.__setattr__(
            self,
            "residuals",
            tuple(tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row) for expiry_row in self.residuals),
        )
        object.__setattr__(self, "max_abs_vol_error", float(self.max_abs_vol_error))
        object.__setattr__(self, "rms_vol_error", float(self.rms_vol_error))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly comparison payload."""
        return {
            "model_vols": [[list(tenor_row) for tenor_row in expiry_row] for expiry_row in self.model_vols],
            "residuals": [[list(tenor_row) for tenor_row in expiry_row] for expiry_row in self.residuals],
            "max_abs_vol_error": float(self.max_abs_vol_error),
            "rms_vol_error": float(self.rms_vol_error),
            "summary": dict(self.summary),
            "provenance": dict(self.provenance),
        }


def _quote_price_from_caplet_strip_quote(
    quote: CapletStripQuote,
    market_state: MarketState,
) -> float:
    """Normalize one cap/floor quote onto price space."""
    if quote.quote_kind == "price":
        return float(quote.quote)
    quote_state = MarketState(
        as_of=market_state.as_of,
        settlement=market_state.settlement,
        discount=market_state.discount,
        forward_curve=market_state.forward_curve,
        forecast_curves=market_state.forecast_curves,
        vol_surface=GridVolSurface(
            expiries=(1e-8, 100.0),
            strikes=(max(float(quote.spec.strike), 1e-8), max(float(quote.spec.strike), 1e-8) + 1e-8),
            vols=((float(quote.quote), float(quote.quote)), (float(quote.quote), float(quote.quote))),
        ),
        selected_curve_names=dict(market_state.selected_curve_names or {}),
        market_provenance=dict(getattr(market_state, "market_provenance", None) or {}),
    )
    payoff = CapPayoff(quote.spec) if quote.kind == "cap" else FloorPayoff(quote.spec)
    return float(payoff.evaluate(quote_state))


def _caplet_incremental_price(
    quote: CapletStripQuote,
    market_state: MarketState,
) -> tuple[float, float]:
    """Return the normalized cap/floor price plus the last fixing expiry."""
    normalized_price = _quote_price_from_caplet_strip_quote(quote, market_state)
    terms = _cap_floor_terms(quote.spec, market_state)
    if not terms:
        raise ValueError("Caplet stripping requires at least one surviving caplet period")
    return float(normalized_price), float(terms[-1].fixing_years)


def _caplet_price_function(
    quote: CapletStripQuote,
    market_state: MarketState,
):
    """Return the scalar caplet Black-76 pricing function for the last strip node."""
    terms = _cap_floor_terms(quote.spec, market_state)
    if not terms:
        raise ValueError("Caplet stripping requires at least one surviving caplet period")
    last = terms[-1]
    option_kernel = black76_call if quote.kind == "cap" else black76_put

    def _price(vol: float) -> float:
        return float(
            quote.spec.notional
            * last.accrual_fraction
            * last.discount_factor
            * option_kernel(last.forward_rate, float(quote.spec.strike), float(vol), max(float(last.fixing_years), 1e-8))
        )

    return _price


def calibrate_caplet_vol_strip_workflow(
    quotes: Sequence[CapletStripQuote],
    market_state: MarketState,
    *,
    surface_name: str = "caplet_vol_strip",
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
    tol: float = 1e-10,
) -> CapletVolStripAuthorityResult:
    """Bootstrap a bounded caplet-vol surface from sequential cap/floor quotes."""
    if market_state.discount is None:
        raise ValueError("Caplet stripping requires market_state.discount")
    resolved_quotes = tuple(quotes)
    if not resolved_quotes:
        raise ValueError("Caplet stripping requires at least one quote")

    payoff_kind_set = {quote.kind for quote in resolved_quotes}
    if len(payoff_kind_set) != 1:
        raise ValueError("Bounded caplet stripping requires all quotes to share one payoff kind")

    by_strike: dict[tuple[float, str, str, str, str, str], list[CapletStripQuote]] = {}
    for quote in resolved_quotes:
        key = _strike_group_key(quote.spec)
        by_strike.setdefault(key, []).append(quote)

    strike_lines: dict[float, list[tuple[float, float]]] = {}
    node_solver_payloads: list[dict[str, object]] = []
    for key, strike_quotes in sorted(by_strike.items(), key=lambda item: item[0][0]):
        ordered = sorted(strike_quotes, key=lambda quote: quote.spec.end_date)
        previous_price = 0.0
        previous_count = 0
        line: list[tuple[float, float]] = []
        for index, quote in enumerate(ordered):
            normalized_price, expiry_years = _caplet_incremental_price(quote, market_state)
            terms = _cap_floor_terms(quote.spec, market_state)
            current_count = len(terms)
            if index > 0 and current_count != previous_count + 1:
                raise ValueError(
                    "Bounded caplet stripping requires each quote ladder step to add exactly one caplet period"
                )
            incremental_price = float(normalized_price - previous_price)
            if incremental_price < -max(1e-10, abs(normalized_price) * 1e-10):
                raise ValueError("Caplet stripping produced a negative incremental caplet price")
            solve_request_price_fn = _caplet_price_function(quote, market_state)
            stripped_vol, solve_request, solve_result = _implied_flat_vol(
                solve_request_price_fn,
                max(float(incremental_price), 0.0),
                lower=vol_lower,
                upper=vol_upper,
                tol=tol,
            )
            line.append((_round_axis(expiry_years), float(stripped_vol)))
            node_solver_payloads.append(
                {
                    "label": quote.resolved_label(index),
                    "strike": float(quote.spec.strike),
                    "expiry_years": float(expiry_years),
                    "solve_request": solve_request.to_payload(),
                    "solve_result": solve_result.to_payload(),
                }
            )
            previous_price = float(normalized_price)
            previous_count = current_count
        strike_lines[float(key[0])] = line

    strikes = tuple(sorted(strike_lines))
    if not strikes:
        raise ValueError("Caplet stripping did not produce any strike lines")
    expiries = tuple(expiry for expiry, _vol in strike_lines[strikes[0]])
    for strike in strikes[1:]:
        candidate_expiries = tuple(expiry for expiry, _vol in strike_lines[strike])
        if candidate_expiries != expiries:
            raise ValueError("Bounded caplet stripping requires a rectangular expiry grid across strikes")

    stripped_vols = tuple(
        tuple(float(dict(strike_lines[strike])[expiry]) for strike in strikes)
        for expiry in expiries
    )
    vol_surface = GridVolSurface(expiries=expiries, strikes=strikes, vols=stripped_vols)
    enriched_state = replace_market_state_vol_surface(market_state, vol_surface)

    repriced_quotes: list[float] = []
    quote_residuals: list[float] = []
    for quote in resolved_quotes:
        payoff = CapPayoff(quote.spec) if quote.kind == "cap" else FloorPayoff(quote.spec)
        repriced = float(payoff.evaluate(enriched_state))
        target_price = _quote_price_from_caplet_strip_quote(quote, market_state)
        repriced_quotes.append(repriced)
        quote_residuals.append(repriced - target_price)

    diagnostics = CapletVolStripDiagnostics(
        quote_count=len(resolved_quotes),
        strike_count=len(strikes),
        expiry_count=len(expiries),
        max_abs_repricing_error=max(abs(value) for value in quote_residuals) if quote_residuals else 0.0,
    )
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "calibrate_caplet_vol_strip_workflow",
        "surface_name": surface_name,
        "selected_curve_names": _market_curve_roles(market_state),
        "market_provenance": _market_provenance(market_state),
        "quotes": [quote.to_payload(index) for index, quote in enumerate(resolved_quotes)],
        "node_solvers": node_solver_payloads,
    }
    summary = {
        "quote_count": len(resolved_quotes),
        "strike_count": len(strikes),
        "expiry_count": len(expiries),
        "surface_name": surface_name,
        "surface_model_family": "caplet_vol_strip",
        "payoff_kind": next(iter(payoff_kind_set)),
        "input_quote_kinds": sorted({quote.quote_kind for quote in resolved_quotes}),
        "max_abs_repricing_error": diagnostics.max_abs_repricing_error,
    }
    return CapletVolStripAuthorityResult(
        quotes=resolved_quotes,
        expiries=expiries,
        strikes=strikes,
        stripped_vols=stripped_vols,
        vol_surface=vol_surface,
        repriced_quotes=tuple(repriced_quotes),
        quote_residuals=tuple(quote_residuals),
        diagnostics=diagnostics,
        surface_name=surface_name,
        provenance=provenance,
        summary=summary,
    )


def _normalize_swaption_quote_vol(
    quote: SwaptionCubeQuote,
    market_state: MarketState,
) -> float:
    """Return one swaption quote normalized onto Black-vol space."""
    if quote.quote_kind == "black_vol":
        return float(quote.quote)
    result = calibrate_swaption_black_vol(quote.spec, market_state, float(quote.quote))
    return float(result.calibrated_vol)


def replace_market_state_vol_surface(market_state: MarketState, vol_surface: object) -> MarketState:
    """Return ``market_state`` with the supplied vol surface attached."""
    return MarketState(
        as_of=market_state.as_of,
        settlement=market_state.settlement,
        discount=market_state.discount,
        forward_curve=market_state.forward_curve,
        vol_surface=vol_surface,
        state_space=market_state.state_space,
        credit_curve=market_state.credit_curve,
        fixing_histories=market_state.fixing_histories,
        forecast_curves=market_state.forecast_curves,
        fx_rates=market_state.fx_rates,
        spot=market_state.spot,
        underlier_spots=market_state.underlier_spots,
        local_vol_surface=market_state.local_vol_surface,
        local_vol_surfaces=market_state.local_vol_surfaces,
        jump_parameters=market_state.jump_parameters,
        jump_parameter_sets=market_state.jump_parameter_sets,
        model_parameters=market_state.model_parameters,
        model_parameter_sets=market_state.model_parameter_sets,
        selected_curve_names=market_state.selected_curve_names,
        market_provenance=market_state.market_provenance,
    )


def calibrate_swaption_vol_cube_workflow(
    quotes: Sequence[SwaptionCubeQuote],
    market_state: MarketState,
    *,
    surface_name: str = "swaption_vol_cube",
) -> SwaptionVolCubeAuthorityResult:
    """Assemble a bounded tenor-aware swaption cube from normalized quote slices."""
    if market_state.discount is None:
        raise ValueError("Swaption cube calibration requires market_state.discount")
    resolved_quotes = tuple(quotes)
    if not resolved_quotes:
        raise ValueError("Swaption cube calibration requires at least one quote")

    normalized_payloads: list[dict[str, object]] = []
    cube_nodes: dict[tuple[float, float, float], float] = {}
    forward_nodes: dict[tuple[float, float], float] = {}

    for index, quote in enumerate(resolved_quotes):
        expiry_years, _annuity, forward_swap_rate, _payment_count = swaption_terms(quote.spec, market_state)
        tenor_years = _swaption_underlier_tenor_years(quote.spec)
        strike = float(quote.spec.strike)
        normalized_vol = _normalize_swaption_quote_vol(quote, market_state)
        expiry_key = _round_axis(expiry_years)
        tenor_key = _round_axis(tenor_years)
        strike_key = _round_axis(strike)
        cube_nodes[(expiry_key, tenor_key, strike_key)] = float(normalized_vol)
        forward_nodes[(expiry_key, tenor_key)] = float(forward_swap_rate)
        payload = quote.to_payload(index)
        payload["normalized_vol"] = float(normalized_vol)
        payload["expiry_years"] = float(expiry_key)
        payload["tenor_years"] = float(tenor_key)
        payload["forward_swap_rate"] = float(forward_swap_rate)
        normalized_payloads.append(payload)

    expiries = _sorted_unique(node[0] for node in cube_nodes)
    tenors = _sorted_unique(node[1] for node in cube_nodes)
    strikes = _sorted_unique(node[2] for node in cube_nodes)
    if len(strikes) < 3:
        raise ValueError("Swaption cube assembly requires at least three strike points per slice")

    market_vol_rows: list[list[list[float]]] = []
    forward_rows: list[list[float]] = []
    for expiry in expiries:
        expiry_vol_rows: list[list[float]] = []
        expiry_forwards: list[float] = []
        for tenor in tenors:
            slice_forwards = forward_nodes.get((_round_axis(expiry), _round_axis(tenor)))
            if slice_forwards is None:
                raise ValueError("Swaption cube assembly requires a rectangular expiry/tenor slice grid")
            expiry_forwards.append(float(slice_forwards))
            strike_row: list[float] = []
            for strike in strikes:
                key = (_round_axis(expiry), _round_axis(tenor), _round_axis(strike))
                if key not in cube_nodes:
                    raise ValueError("Swaption cube assembly requires a rectangular strike grid across slices")
                strike_row.append(float(cube_nodes[key]))
            expiry_vol_rows.append(strike_row)
        market_vol_rows.append(expiry_vol_rows)
        forward_rows.append(expiry_forwards)

    market_vols = tuple(
        tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row)
        for expiry_row in market_vol_rows
    )
    forwards = tuple(tuple(float(value) for value in row) for row in forward_rows)
    vol_cube = SwaptionVolCube(expiries=expiries, tenors=tenors, strikes=strikes, vols=market_vols)
    enriched_state = replace_market_state_vol_surface(market_state, vol_cube)

    repriced_vols: list[float] = []
    repriced_prices: list[float] = []
    quote_residuals: list[float] = []
    price_residuals: list[float] = []
    for quote, payload in zip(resolved_quotes, normalized_payloads):
        repriced_vol = float(
            vol_cube.swaption_black_vol(
                float(payload["expiry_years"]),
                float(quote.spec.strike),
                float(payload["tenor_years"]),
            )
        )
        repriced_price = float(price_swaption_black76(enriched_state, quote.spec))
        repriced_vols.append(repriced_vol)
        repriced_prices.append(repriced_price)
        quote_residuals.append(repriced_vol - float(payload["normalized_vol"]))
        target_price = float(quote.quote) if quote.quote_kind == "price" else float(
            price_swaption_black76(replace_market_state_vol_surface(market_state, GridVolSurface(
                expiries=(1e-8, 100.0),
                strikes=(max(float(quote.spec.strike), 1e-8), max(float(quote.spec.strike), 1e-8) + 1e-8),
                vols=((float(payload["normalized_vol"]), float(payload["normalized_vol"])), (float(payload["normalized_vol"]), float(payload["normalized_vol"]))),
            )), quote.spec)
        )
        price_residuals.append(repriced_price - target_price)

    diagnostics = SwaptionVolCubeDiagnostics(
        quote_count=len(resolved_quotes),
        slice_count=len(expiries) * len(tenors),
        expiry_count=len(expiries),
        tenor_count=len(tenors),
        strike_count=len(strikes),
        max_abs_quote_residual=max(abs(value) for value in quote_residuals) if quote_residuals else 0.0,
        max_abs_price_residual=max(abs(value) for value in price_residuals) if price_residuals else 0.0,
    )
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "calibrate_swaption_vol_cube_workflow",
        "surface_name": surface_name,
        "selected_curve_names": _market_curve_roles(market_state),
        "market_provenance": _market_provenance(market_state),
        "quotes": normalized_payloads,
        "vol_cube": vol_cube.to_payload(),
    }
    summary = {
        "quote_count": len(resolved_quotes),
        "slice_count": len(expiries) * len(tenors),
        "expiry_count": len(expiries),
        "tenor_count": len(tenors),
        "strike_count": len(strikes),
        "surface_name": surface_name,
        "surface_model_family": "swaption_vol_cube",
        "max_abs_quote_residual": diagnostics.max_abs_quote_residual,
        "max_abs_price_residual": diagnostics.max_abs_price_residual,
    }
    return SwaptionVolCubeAuthorityResult(
        quotes=resolved_quotes,
        expiries=expiries,
        tenors=tenors,
        strikes=strikes,
        forwards=forwards,
        market_vols=market_vols,
        vol_cube=vol_cube,
        repriced_vols=tuple(repriced_vols),
        repriced_prices=tuple(repriced_prices),
        quote_residuals=tuple(quote_residuals),
        price_residuals=tuple(price_residuals),
        diagnostics=diagnostics,
        surface_name=surface_name,
        provenance=provenance,
        summary=summary,
    )


def compare_sabr_to_swaption_cube_workflow(
    authority_result: SwaptionVolCubeAuthorityResult,
    *,
    beta: float = 0.5,
) -> SwaptionCubeStageComparisonResult:
    """Fit one SABR smile per expiry/tenor slice and compare it back to the cube."""
    sabr_slice_results: list[SABRSmileCalibrationResult] = []
    model_rows: list[list[list[float]]] = []
    residual_rows: list[list[list[float]]] = []
    squared_errors: list[float] = []
    absolute_errors: list[float] = []

    for expiry_index, expiry in enumerate(authority_result.expiries):
        expiry_model_rows: list[list[float]] = []
        expiry_residual_rows: list[list[float]] = []
        for tenor_index, tenor in enumerate(authority_result.tenors):
            forward = float(authority_result.forwards[expiry_index][tenor_index])
            market_slice = tuple(float(value) for value in authority_result.market_vols[expiry_index][tenor_index])
            sabr_result = fit_sabr_smile_surface(
                build_sabr_smile_surface(
                    forward,
                    expiry,
                    authority_result.strikes,
                    market_slice,
                    beta=beta,
                    surface_name=f"{authority_result.surface_name}_{expiry:g}_{tenor:g}",
                    source_ref="compare_sabr_to_swaption_cube_workflow",
                    metadata={
                        "surface_name": authority_result.surface_name,
                        "expiry_years": float(expiry),
                        "tenor_years": float(tenor),
                    },
                )
            )
            sabr_slice_results.append(sabr_result)
            model_slice = [
                float(sabr_result.sabr.implied_vol(forward, strike, expiry))
                for strike in authority_result.strikes
            ]
            residual_slice = [model - market for model, market in zip(model_slice, market_slice)]
            expiry_model_rows.append(model_slice)
            expiry_residual_rows.append(residual_slice)
            squared_errors.extend(value * value for value in residual_slice)
            absolute_errors.extend(abs(value) for value in residual_slice)
        model_rows.append(expiry_model_rows)
        residual_rows.append(expiry_residual_rows)

    max_abs_vol_error = max(absolute_errors) if absolute_errors else 0.0
    rms_vol_error = sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0.0
    summary = {
        "surface_name": authority_result.surface_name,
        "slice_count": len(authority_result.expiries) * len(authority_result.tenors),
        "point_count": len(squared_errors),
        "beta": float(beta),
    }
    provenance = {
        "source_kind": "model_compression",
        "source_ref": "compare_sabr_to_swaption_cube_workflow",
        "surface_name": authority_result.surface_name,
        "authority_summary": dict(authority_result.summary),
    }
    return SwaptionCubeStageComparisonResult(
        authority_result=authority_result,
        sabr_slice_results=tuple(sabr_slice_results),
        model_vols=tuple(
            tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row)
            for expiry_row in model_rows
        ),
        residuals=tuple(
            tuple(tuple(float(value) for value in tenor_row) for tenor_row in expiry_row)
            for expiry_row in residual_rows
        ),
        max_abs_vol_error=float(max_abs_vol_error),
        rms_vol_error=float(rms_vol_error),
        summary=summary,
        provenance=provenance,
    )


__all__ = [
    "CapletStripQuote",
    "CapletVolStripDiagnostics",
    "CapletVolStripAuthorityResult",
    "SwaptionCubeQuote",
    "SwaptionVolCube",
    "SwaptionVolCubeDiagnostics",
    "SwaptionVolCubeAuthorityResult",
    "SwaptionCubeStageComparisonResult",
    "calibrate_caplet_vol_strip_workflow",
    "calibrate_swaption_vol_cube_workflow",
    "compare_sabr_to_swaption_cube_workflow",
]
