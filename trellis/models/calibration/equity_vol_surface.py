"""Bounded equity-vol surface authority and staged-fit helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np

from trellis.core.market_state import MarketState
from trellis.models.calibration.heston_fit import (
    HestonSmileCalibrationResult,
    HestonSurfaceCalibrationResult,
    calibrate_heston_smile_workflow,
    calibrate_heston_surface_from_equity_vol_surface_workflow,
)
from trellis.models.calibration.implied_vol import _bs_price
from trellis.models.calibration.local_vol import (
    LocalVolCalibrationResult,
    calibrate_local_vol_surface_workflow,
)
from trellis.models.calibration.materialization import materialize_black_vol_surface
from trellis.models.calibration.quote_maps import (
    QuoteAxisSpec,
    QuoteMapSpec,
    QuoteSemanticsSpec,
    QuoteUnitSpec,
)
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveBounds,
    SolveProvenance,
    SolveReplayArtifact,
    SolveRequest,
    SolveResult,
    build_solve_provenance,
    build_solve_replay_artifact,
    execute_solve_request,
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


def _default_implied_vol_quote_map_spec() -> QuoteMapSpec:
    """Return the shared implied-vol quote-map spec used by equity-vol workflows."""
    return QuoteMapSpec(
        quote_family="implied_vol",
        convention="black",
        semantics=QuoteSemanticsSpec(
            quote_family="implied_vol",
            convention="black",
            quote_subject="equity_option",
            axes=(
                QuoteAxisSpec("expiry", axis_kind="time_to_expiry", unit="years"),
                QuoteAxisSpec("strike", axis_kind="option_strike", unit="price"),
            ),
            unit=QuoteUnitSpec(
                unit_name="decimal_volatility",
                value_domain="volatility",
                scaling="absolute",
            ),
        ),
    )


def _representative_total_variance(expiry_years: float, vol: float) -> float:
    """Return total variance for one expiry/vol pair."""
    return float(expiry_years) * float(vol) * float(vol)


def _curve_forward(spot: float, rate: float, dividend_yield: float, expiry_years: float) -> float:
    """Return the Black forward used for log-moneyness conversion."""
    return float(spot) * float(raw_np.exp((float(rate) - float(dividend_yield)) * float(expiry_years)))


def _call_prices_from_smile(
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    expiry_years: float,
    strikes: raw_np.ndarray,
    vols: raw_np.ndarray,
) -> raw_np.ndarray:
    """Return Black-style call prices for one smile row."""
    return raw_np.asarray(
        [
            _bs_price(
                float(spot),
                float(strike),
                float(expiry_years),
                float(rate),
                float(vol),
                "call",
                dividend_yield=float(dividend_yield),
            )
            for strike, vol in zip(strikes, vols)
        ],
        dtype=float,
    )


def _smile_no_arb_violation_counts(
    *,
    strikes: raw_np.ndarray,
    call_prices: raw_np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[int, int]:
    """Return discrete monotonicity and convexity violation counts."""
    monotonicity_violations = 0
    for left, right in zip(call_prices, call_prices[1:]):
        if float(right) > float(left) + float(tolerance):
            monotonicity_violations += 1

    convexity_violations = 0
    if strikes.size >= 3:
        slopes = raw_np.diff(call_prices) / raw_np.diff(strikes)
        for left, right in zip(slopes, slopes[1:]):
            if float(right) < float(left) - float(tolerance):
                convexity_violations += 1
    return int(monotonicity_violations), int(convexity_violations)


def _calendar_total_variance_violations(
    *,
    expiries: raw_np.ndarray,
    strikes: raw_np.ndarray,
    vol_grid: raw_np.ndarray,
    tolerance: float = 1e-10,
) -> int:
    """Return the number of discrete calendar monotonicity violations."""
    if expiries.size < 2:
        return 0
    violation_count = 0
    total_variance = (vol_grid ** 2) * expiries[:, None]
    for strike_index in range(strikes.size):
        column = total_variance[:, strike_index]
        for earlier, later in zip(column, column[1:]):
            if float(later) < float(earlier) - float(tolerance):
                violation_count += 1
    return int(violation_count)


def _svi_total_variance(
    log_moneyness: raw_np.ndarray | float,
    *,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> raw_np.ndarray:
    """Return raw-SVI total variance."""
    k = raw_np.asarray(log_moneyness, dtype=float)
    shifted = k - float(m)
    return float(a) + float(b) * (
        float(rho) * shifted + raw_np.sqrt(shifted * shifted + float(sigma) * float(sigma))
    )


def _normalize_expiry_index(expiries: Sequence[float], expiry_years: float) -> int:
    """Return the exact expiry index for staged comparisons."""
    for index, value in enumerate(expiries):
        if abs(float(value) - float(expiry_years)) <= 1e-12:
            return int(index)
    raise ValueError(f"expiry {float(expiry_years):g} is not present in the fitted surface grid")


@dataclass(frozen=True)
class EquityVolSurfaceInput:
    """Normalized multi-expiry equity-vol input surface."""

    spot: float
    rate: float
    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    market_vols: tuple[tuple[float, ...], ...]
    dividend_yield: float = 0.0
    surface_name: str = ""
    warnings: tuple[str, ...] = ()
    quote_map_spec: QuoteMapSpec = field(default_factory=_default_implied_vol_quote_map_spec)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not raw_np.isfinite(self.spot) or float(self.spot) <= 0.0:
            raise ValueError("spot must be finite and positive")
        if not raw_np.isfinite(self.rate):
            raise ValueError("rate must be finite")
        if not raw_np.isfinite(self.dividend_yield):
            raise ValueError("dividend_yield must be finite")
        expiries = tuple(float(value) for value in self.expiries)
        strikes = tuple(float(value) for value in self.strikes)
        if len(expiries) < 2:
            raise ValueError("equity-vol surface inputs require at least two expiries")
        if len(strikes) < 5:
            raise ValueError("equity-vol surface inputs require at least five strikes")
        if tuple(sorted(expiries)) != expiries or len(set(expiries)) != len(expiries):
            raise ValueError("expiries must be strictly increasing")
        if tuple(sorted(strikes)) != strikes or len(set(strikes)) != len(strikes):
            raise ValueError("strikes must be strictly increasing")
        if any(value <= 0.0 or not raw_np.isfinite(value) for value in expiries):
            raise ValueError("expiries must be finite and positive")
        if any(value <= 0.0 or not raw_np.isfinite(value) for value in strikes):
            raise ValueError("strikes must be finite and positive")
        vol_grid = tuple(tuple(float(vol) for vol in row) for row in self.market_vols)
        if len(vol_grid) != len(expiries):
            raise ValueError("market_vol rows must align with expiries")
        if any(len(row) != len(strikes) for row in vol_grid):
            raise ValueError("each market_vol row must align with strikes")
        if any(vol <= 0.0 or not raw_np.isfinite(vol) for row in vol_grid for vol in row):
            raise ValueError("market vols must be finite and positive")
        if self.quote_map_spec.quote_family != "implied_vol":
            raise ValueError("equity-vol surface inputs require an implied-vol quote map")
        object.__setattr__(self, "spot", float(self.spot))
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "dividend_yield", float(self.dividend_yield))
        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "market_vols", vol_grid)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def point_count(self) -> int:
        return int(len(self.expiries) * len(self.strikes))

    def forward(self, expiry_years: float) -> float:
        """Return the forward level for one expiry."""
        return _curve_forward(self.spot, self.rate, self.dividend_yield, float(expiry_years))

    def expiry_row(self, expiry_years: float) -> tuple[float, ...]:
        """Return one market-vol row for an expiry present in the grid."""
        return self.market_vols[_normalize_expiry_index(self.expiries, expiry_years)]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "source_kind": "option_surface",
            "surface_name": self.surface_name,
            "spot": float(self.spot),
            "rate": float(self.rate),
            "dividend_yield": float(self.dividend_yield),
            "expiries": list(self.expiries),
            "strikes": list(self.strikes),
            "market_vols": [list(row) for row in self.market_vols],
            "point_count": self.point_count,
            "warnings": list(self.warnings),
            "quote_map": self.quote_map_spec.to_payload(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EquityVolQuoteAdjustment:
    """One quote-cleaning adjustment on the observed vol grid."""

    expiry_years: float
    strike: float
    raw_vol: float
    cleaned_vol: float
    adjustment: float
    relative_adjustment: float
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "strike", float(self.strike))
        object.__setattr__(self, "raw_vol", float(self.raw_vol))
        object.__setattr__(self, "cleaned_vol", float(self.cleaned_vol))
        object.__setattr__(self, "adjustment", float(self.adjustment))
        object.__setattr__(self, "relative_adjustment", float(self.relative_adjustment))
        object.__setattr__(self, "flags", tuple(str(flag) for flag in self.flags))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly adjustment payload."""
        return {
            "expiry_years": float(self.expiry_years),
            "strike": float(self.strike),
            "raw_vol": float(self.raw_vol),
            "cleaned_vol": float(self.cleaned_vol),
            "adjustment": float(self.adjustment),
            "relative_adjustment": float(self.relative_adjustment),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class EquityVolQuoteCleaningDiagnostics:
    """Diagnostics for the quote-governance stage ahead of surface repair."""

    adjusted_point_count: int
    max_abs_adjustment: float
    rms_adjustment: float
    raw_smile_violation_count: int
    cleaned_smile_violation_count: int
    raw_calendar_violation_count: int
    cleaned_calendar_violation_count: int
    point_count: int
    flagged_nodes: tuple[EquityVolQuoteAdjustment, ...] = ()
    warning_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjusted_point_count", int(self.adjusted_point_count))
        object.__setattr__(self, "max_abs_adjustment", float(self.max_abs_adjustment))
        object.__setattr__(self, "rms_adjustment", float(self.rms_adjustment))
        object.__setattr__(self, "raw_smile_violation_count", int(self.raw_smile_violation_count))
        object.__setattr__(self, "cleaned_smile_violation_count", int(self.cleaned_smile_violation_count))
        object.__setattr__(self, "raw_calendar_violation_count", int(self.raw_calendar_violation_count))
        object.__setattr__(self, "cleaned_calendar_violation_count", int(self.cleaned_calendar_violation_count))
        object.__setattr__(self, "point_count", int(self.point_count))
        object.__setattr__(self, "flagged_nodes", tuple(self.flagged_nodes))
        object.__setattr__(self, "warning_count", int(self.warning_count))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics payload."""
        return {
            "adjusted_point_count": int(self.adjusted_point_count),
            "max_abs_adjustment": float(self.max_abs_adjustment),
            "rms_adjustment": float(self.rms_adjustment),
            "raw_smile_violation_count": int(self.raw_smile_violation_count),
            "cleaned_smile_violation_count": int(self.cleaned_smile_violation_count),
            "raw_calendar_violation_count": int(self.raw_calendar_violation_count),
            "cleaned_calendar_violation_count": int(self.cleaned_calendar_violation_count),
            "point_count": int(self.point_count),
            "flagged_nodes": [node.to_payload() for node in self.flagged_nodes],
            "warning_count": int(self.warning_count),
        }


@dataclass(frozen=True)
class EquityVolQuoteCleaningResult:
    """Structured result for the equity-vol quote-governance stage."""

    raw_surface: EquityVolSurfaceInput
    cleaned_surface: EquityVolSurfaceInput
    diagnostics: EquityVolQuoteCleaningDiagnostics
    warnings: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly quote-cleaning payload."""
        return {
            "raw_surface": self.raw_surface.to_payload(),
            "cleaned_surface": self.cleaned_surface.to_payload(),
            "fit_diagnostics": self.diagnostics.to_payload(),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class SVISmileParameters:
    """Raw-SVI parameter set for one expiry smile."""

    expiry_years: float
    forward: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "forward", float(self.forward))
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "b", float(self.b))
        object.__setattr__(self, "rho", float(self.rho))
        object.__setattr__(self, "m", float(self.m))
        object.__setattr__(self, "sigma", float(self.sigma))

    def total_variance(self, log_moneyness: raw_np.ndarray | float) -> raw_np.ndarray:
        """Return raw-SVI total variance at one or more log-moneyness points."""
        return _svi_total_variance(
            log_moneyness,
            a=self.a,
            b=self.b,
            rho=self.rho,
            m=self.m,
            sigma=self.sigma,
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "expiry_years": float(self.expiry_years),
            "forward": float(self.forward),
            "a": float(self.a),
            "b": float(self.b),
            "rho": float(self.rho),
            "m": float(self.m),
            "sigma": float(self.sigma),
        }


@dataclass(frozen=True)
class SVISmileFitDiagnostics:
    """Diagnostics for one fitted SVI smile."""

    expiry_years: float
    market_vols: tuple[float, ...]
    repaired_vols: tuple[float, ...]
    residuals: tuple[float, ...]
    max_abs_vol_error: float
    rms_vol_error: float
    raw_monotonicity_violation_count: int
    raw_convexity_violation_count: int
    repaired_monotonicity_violation_count: int
    repaired_convexity_violation_count: int
    warning_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "market_vols", tuple(float(value) for value in self.market_vols))
        object.__setattr__(self, "repaired_vols", tuple(float(value) for value in self.repaired_vols))
        object.__setattr__(self, "residuals", tuple(float(value) for value in self.residuals))
        object.__setattr__(self, "max_abs_vol_error", float(self.max_abs_vol_error))
        object.__setattr__(self, "rms_vol_error", float(self.rms_vol_error))
        object.__setattr__(self, "raw_monotonicity_violation_count", int(self.raw_monotonicity_violation_count))
        object.__setattr__(self, "raw_convexity_violation_count", int(self.raw_convexity_violation_count))
        object.__setattr__(
            self,
            "repaired_monotonicity_violation_count",
            int(self.repaired_monotonicity_violation_count),
        )
        object.__setattr__(
            self,
            "repaired_convexity_violation_count",
            int(self.repaired_convexity_violation_count),
        )
        object.__setattr__(self, "warning_count", int(self.warning_count))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "expiry_years": float(self.expiry_years),
            "market_vols": list(self.market_vols),
            "repaired_vols": list(self.repaired_vols),
            "residuals": list(self.residuals),
            "max_abs_vol_error": float(self.max_abs_vol_error),
            "rms_vol_error": float(self.rms_vol_error),
            "raw_monotonicity_violation_count": int(self.raw_monotonicity_violation_count),
            "raw_convexity_violation_count": int(self.raw_convexity_violation_count),
            "repaired_monotonicity_violation_count": int(self.repaired_monotonicity_violation_count),
            "repaired_convexity_violation_count": int(self.repaired_convexity_violation_count),
            "warning_count": int(self.warning_count),
        }


@dataclass(frozen=True)
class SVISmileFitResult:
    """Structured record for one fitted SVI smile."""

    expiry_years: float
    parameters: SVISmileParameters
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    diagnostics: SVISmileFitDiagnostics
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "expiry_years": float(self.expiry_years),
            "parameters": self.parameters.to_payload(),
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "fit_diagnostics": self.diagnostics.to_payload(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SVIVolSurface:
    """Parameterized repaired equity-vol surface assembled from per-expiry SVI smiles."""

    spot: float
    rate: float
    dividend_yield: float
    smiles: tuple[SVISmileParameters, ...]
    repair_calendar_monotonicity: bool = True

    def __post_init__(self) -> None:
        expiries = tuple(float(smile.expiry_years) for smile in self.smiles)
        if len(expiries) < 2:
            raise ValueError("SVI surface requires at least two fitted smiles")
        if tuple(sorted(expiries)) != expiries or len(set(expiries)) != len(expiries):
            raise ValueError("SVI smile expiries must be strictly increasing")
        object.__setattr__(self, "spot", float(self.spot))
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "dividend_yield", float(self.dividend_yield))
        object.__setattr__(self, "smiles", tuple(self.smiles))
        object.__setattr__(self, "repair_calendar_monotonicity", bool(self.repair_calendar_monotonicity))

    @property
    def expiries(self) -> tuple[float, ...]:
        return tuple(float(smile.expiry_years) for smile in self.smiles)

    def _node_total_variances(self, strike: float) -> raw_np.ndarray:
        """Return node total variances for one strike, optionally calendar-repaired."""
        if not raw_np.isfinite(strike) or float(strike) <= 0.0:
            raise ValueError("strike must be finite and positive")
        total_variances = raw_np.asarray(
            [
                max(
                    float(
                        smile.total_variance(
                            raw_np.log(float(strike) / max(float(smile.forward), 1e-12))
                        )
                    ),
                    1e-12,
                )
                for smile in self.smiles
            ],
            dtype=float,
        )
        if self.repair_calendar_monotonicity:
            total_variances = raw_np.maximum.accumulate(total_variances)
        return total_variances

    def black_vol(self, expiry: float, strike: float) -> float:
        """Return a repaired Black implied volatility for one expiry/strike pair."""
        expiry_years = max(float(expiry), 1e-8)
        expiries = raw_np.asarray(self.expiries, dtype=float)
        node_total_variances = self._node_total_variances(float(strike))
        if expiry_years <= float(expiries[0]):
            total_variance = float(node_total_variances[0]) * (expiry_years / float(expiries[0]))
        elif expiry_years >= float(expiries[-1]):
            total_variance = float(node_total_variances[-1]) * (expiry_years / float(expiries[-1]))
        else:
            upper = int(raw_np.searchsorted(expiries, expiry_years))
            lower = upper - 1
            left_expiry = float(expiries[lower])
            right_expiry = float(expiries[upper])
            weight = (expiry_years - left_expiry) / (right_expiry - left_expiry)
            total_variance = float(
                (1.0 - weight) * float(node_total_variances[lower])
                + weight * float(node_total_variances[upper])
            )
        return float(raw_np.sqrt(max(total_variance / expiry_years, 1e-12)))

    def sample_grid(
        self,
        *,
        expiries: Sequence[float],
        strikes: Sequence[float],
    ) -> tuple[tuple[float, ...], ...]:
        """Return the implied-vol grid sampled from the repaired surface."""
        return tuple(
            tuple(self.black_vol(float(expiry), float(strike)) for strike in strikes)
            for expiry in expiries
        )


@dataclass(frozen=True)
class EquityVolSurfaceFitDiagnostics:
    """Surface-level diagnostics for the repaired equity-vol authority."""

    smile_fits: tuple[SVISmileFitDiagnostics, ...]
    raw_calendar_violation_count: int
    repaired_calendar_violation_count: int
    raw_smile_violation_count: int
    repaired_smile_violation_count: int
    max_abs_vol_error: float
    rms_vol_error: float
    point_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "smile_fits", tuple(self.smile_fits))
        object.__setattr__(self, "raw_calendar_violation_count", int(self.raw_calendar_violation_count))
        object.__setattr__(self, "repaired_calendar_violation_count", int(self.repaired_calendar_violation_count))
        object.__setattr__(self, "raw_smile_violation_count", int(self.raw_smile_violation_count))
        object.__setattr__(self, "repaired_smile_violation_count", int(self.repaired_smile_violation_count))
        object.__setattr__(self, "max_abs_vol_error", float(self.max_abs_vol_error))
        object.__setattr__(self, "rms_vol_error", float(self.rms_vol_error))
        object.__setattr__(self, "point_count", int(self.point_count))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "smile_fits": [smile.to_payload() for smile in self.smile_fits],
            "raw_calendar_violation_count": int(self.raw_calendar_violation_count),
            "repaired_calendar_violation_count": int(self.repaired_calendar_violation_count),
            "raw_smile_violation_count": int(self.raw_smile_violation_count),
            "repaired_smile_violation_count": int(self.repaired_smile_violation_count),
            "max_abs_vol_error": float(self.max_abs_vol_error),
            "rms_vol_error": float(self.rms_vol_error),
            "point_count": int(self.point_count),
        }


@dataclass(frozen=True)
class EquityVolSurfaceAuthorityResult:
    """Structured result for the bounded repaired equity-vol surface workflow."""

    input_surface: EquityVolSurfaceInput
    vol_surface: SVIVolSurface
    repaired_vols: tuple[tuple[float, ...], ...]
    smile_fits: tuple[SVISmileFitResult, ...]
    diagnostics: EquityVolSurfaceFitDiagnostics
    surface_name: str
    raw_input_surface: EquityVolSurfaceInput | None = None
    quote_cleaning: EquityVolQuoteCleaningResult | None = None
    warnings: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "smile_fits", tuple(self.smile_fits))
        object.__setattr__(self, "repaired_vols", tuple(tuple(float(vol) for vol in row) for row in self.repaired_vols))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the repaired vol surface."""
        return materialize_black_vol_surface(
            market_state,
            surface_name=self.surface_name,
            vol_surface=self.vol_surface,
            source_kind=str(self.provenance.get("source_kind", "calibrated_surface")),
            source_ref=str(
                self.provenance.get("source_ref", "calibrate_equity_vol_surface_workflow")
            ),
            metadata={
                "instrument_family": "equity_vol",
                "surface_name": self.surface_name,
                "surface_model_family": "raw_svi_surface",
                "point_count": int(self.diagnostics.point_count),
                "quote_cleaning_adjusted_point_count": int(
                    self.quote_cleaning.diagnostics.adjusted_point_count
                )
                if self.quote_cleaning is not None
                else 0,
                "raw_calendar_violation_count": int(self.diagnostics.raw_calendar_violation_count),
                "repaired_calendar_violation_count": int(self.diagnostics.repaired_calendar_violation_count),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "input_surface": self.input_surface.to_payload(),
            "raw_input_surface": None if self.raw_input_surface is None else self.raw_input_surface.to_payload(),
            "repaired_vols": [list(row) for row in self.repaired_vols],
            "smile_fits": [smile.to_payload() for smile in self.smile_fits],
            "fit_diagnostics": self.diagnostics.to_payload(),
            "quote_cleaning": None if self.quote_cleaning is None else self.quote_cleaning.to_payload(),
            "surface_name": self.surface_name,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class EquityVolStageComparisonResult:
    """Comparison between the repaired surface stage and a Heston model-compression stage."""

    expiry_years: float
    strikes: tuple[float, ...]
    market_vols: tuple[float, ...]
    surface_vols: tuple[float, ...]
    model_vols: tuple[float, ...]
    surface_residuals: tuple[float, ...]
    model_residuals: tuple[float, ...]
    surface_max_abs_vol_error: float
    model_max_abs_vol_error: float
    preferred_stage: str
    heston_result: HestonSmileCalibrationResult
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiry_years", float(self.expiry_years))
        object.__setattr__(self, "strikes", tuple(float(value) for value in self.strikes))
        object.__setattr__(self, "market_vols", tuple(float(value) for value in self.market_vols))
        object.__setattr__(self, "surface_vols", tuple(float(value) for value in self.surface_vols))
        object.__setattr__(self, "model_vols", tuple(float(value) for value in self.model_vols))
        object.__setattr__(self, "surface_residuals", tuple(float(value) for value in self.surface_residuals))
        object.__setattr__(self, "model_residuals", tuple(float(value) for value in self.model_residuals))
        object.__setattr__(self, "surface_max_abs_vol_error", float(self.surface_max_abs_vol_error))
        object.__setattr__(self, "model_max_abs_vol_error", float(self.model_max_abs_vol_error))
        object.__setattr__(self, "preferred_stage", str(self.preferred_stage))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "expiry_years": float(self.expiry_years),
            "strikes": list(self.strikes),
            "market_vols": list(self.market_vols),
            "surface_vols": list(self.surface_vols),
            "model_vols": list(self.model_vols),
            "surface_residuals": list(self.surface_residuals),
            "model_residuals": list(self.model_residuals),
            "surface_max_abs_vol_error": float(self.surface_max_abs_vol_error),
            "model_max_abs_vol_error": float(self.model_max_abs_vol_error),
            "preferred_stage": self.preferred_stage,
            "heston_result": self.heston_result.to_payload(),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class EquityVolSurfaceStageComparisonResult:
    """Full-surface comparison between repaired surface and Heston compression stages."""

    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    market_vols: tuple[tuple[float, ...], ...]
    surface_vols: tuple[tuple[float, ...], ...]
    model_vols: tuple[tuple[float, ...], ...]
    surface_max_abs_vol_error: float
    model_max_abs_vol_error: float
    surface_rms_vol_error: float
    model_rms_vol_error: float
    preferred_stage: str
    heston_surface_result: HestonSurfaceCalibrationResult
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiries", tuple(float(value) for value in self.expiries))
        object.__setattr__(self, "strikes", tuple(float(value) for value in self.strikes))
        object.__setattr__(self, "market_vols", tuple(tuple(float(value) for value in row) for row in self.market_vols))
        object.__setattr__(self, "surface_vols", tuple(tuple(float(value) for value in row) for row in self.surface_vols))
        object.__setattr__(self, "model_vols", tuple(tuple(float(value) for value in row) for row in self.model_vols))
        object.__setattr__(self, "surface_max_abs_vol_error", float(self.surface_max_abs_vol_error))
        object.__setattr__(self, "model_max_abs_vol_error", float(self.model_max_abs_vol_error))
        object.__setattr__(self, "surface_rms_vol_error", float(self.surface_rms_vol_error))
        object.__setattr__(self, "model_rms_vol_error", float(self.model_rms_vol_error))
        object.__setattr__(self, "preferred_stage", str(self.preferred_stage))

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-friendly payload."""
        return {
            "expiries": list(self.expiries),
            "strikes": list(self.strikes),
            "market_vols": [list(row) for row in self.market_vols],
            "surface_vols": [list(row) for row in self.surface_vols],
            "model_vols": [list(row) for row in self.model_vols],
            "surface_max_abs_vol_error": float(self.surface_max_abs_vol_error),
            "model_max_abs_vol_error": float(self.model_max_abs_vol_error),
            "surface_rms_vol_error": float(self.surface_rms_vol_error),
            "model_rms_vol_error": float(self.model_rms_vol_error),
            "preferred_stage": self.preferred_stage,
            "heston_surface_result": self.heston_surface_result.to_payload(),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
        }


def build_equity_vol_surface_input(
    spot: float,
    expiries: Sequence[float],
    strikes: Sequence[float],
    market_vols: Sequence[Sequence[float]],
    *,
    rate: float,
    dividend_yield: float = 0.0,
    surface_name: str = "",
    metadata: Mapping[str, object] | None = None,
) -> EquityVolSurfaceInput:
    """Normalize one multi-expiry equity-vol surface input."""
    expiry_array = raw_np.asarray(expiries, dtype=float)
    strike_array = raw_np.asarray(strikes, dtype=float)
    vol_array = raw_np.asarray(market_vols, dtype=float)
    if expiry_array.ndim != 1 or strike_array.ndim != 1:
        raise ValueError("expiries and strikes must be one-dimensional sequences")
    if vol_array.ndim != 2:
        raise ValueError("market_vols must be a two-dimensional surface")
    if vol_array.shape != (expiry_array.size, strike_array.size):
        raise ValueError("market_vols must have shape (len(expiries), len(strikes))")

    warnings: list[str] = []
    for expiry in expiry_array:
        forward = _curve_forward(float(spot), float(rate), float(dividend_yield), float(expiry))
        below = raw_np.any(strike_array < forward)
        above = raw_np.any(strike_array > forward)
        if not (below and above):
            warnings.append(
                f"Equity-vol expiry {float(expiry):g} does not bracket the forward with strikes on both sides."
            )

    return EquityVolSurfaceInput(
        spot=float(spot),
        rate=float(rate),
        expiries=tuple(float(value) for value in expiry_array),
        strikes=tuple(float(value) for value in strike_array),
        market_vols=tuple(
            tuple(float(vol) for vol in row)
            for row in vol_array
        ),
        dividend_yield=float(dividend_yield),
        surface_name=surface_name,
        warnings=tuple(warnings),
        metadata=metadata or {},
    )


def _surface_smile_violation_count(
    surface: EquityVolSurfaceInput,
    vol_grid: raw_np.ndarray,
) -> int:
    """Return the total discrete smile violation count across expiries."""
    strikes = raw_np.asarray(surface.strikes, dtype=float)
    total = 0
    for expiry_index, expiry_years in enumerate(surface.expiries):
        call_prices = _call_prices_from_smile(
            spot=surface.spot,
            rate=surface.rate,
            dividend_yield=surface.dividend_yield,
            expiry_years=float(expiry_years),
            strikes=strikes,
            vols=raw_np.asarray(vol_grid[expiry_index], dtype=float),
        )
        monotonicity_violations, convexity_violations = _smile_no_arb_violation_counts(
            strikes=strikes,
            call_prices=call_prices,
        )
        total += monotonicity_violations + convexity_violations
    return int(total)


def _govern_equity_vol_surface_input(
    surface: EquityVolSurfaceInput,
    *,
    absolute_outlier_threshold: float = 0.04,
    relative_outlier_threshold: float = 0.15,
    mad_multiplier: float = 4.0,
    min_vol: float = 0.01,
    max_vol: float = 3.0,
) -> EquityVolQuoteCleaningResult:
    """Return a cleaned quote surface and explicit cleaning diagnostics."""
    raw_vols = raw_np.asarray(surface.market_vols, dtype=float)
    cleaned_vols = raw_vols.copy()
    adjustments: list[EquityVolQuoteAdjustment] = []

    for expiry_index in range(raw_vols.shape[0]):
        for strike_index in range(raw_vols.shape[1]):
            raw_vol = float(raw_vols[expiry_index, strike_index])
            cleaned_vol = float(raw_vol)
            flags: list[str] = []

            if cleaned_vol < float(min_vol):
                cleaned_vol = float(min_vol)
                flags.append("vol_floor")
            elif cleaned_vol > float(max_vol):
                cleaned_vol = float(max_vol)
                flags.append("vol_cap")

            row_start = max(0, expiry_index - 1)
            row_stop = min(raw_vols.shape[0], expiry_index + 2)
            col_start = max(0, strike_index - 1)
            col_stop = min(raw_vols.shape[1], strike_index + 2)
            neighborhood = raw_vols[row_start:row_stop, col_start:col_stop].reshape(-1)
            if neighborhood.size > 1:
                center = (expiry_index - row_start) * (col_stop - col_start) + (strike_index - col_start)
                peers = raw_np.delete(neighborhood, center)
            else:
                peers = raw_np.asarray((), dtype=float)

            if peers.size >= 3:
                local_median = float(raw_np.median(peers))
                local_mad = float(raw_np.median(raw_np.abs(peers - local_median)))
                threshold = max(
                    float(absolute_outlier_threshold),
                    abs(local_median) * float(relative_outlier_threshold),
                    float(mad_multiplier) * local_mad,
                )
                if abs(raw_vol - local_median) > threshold:
                    cleaned_vol = min(max(local_median, float(min_vol)), float(max_vol))
                    flags.append("local_outlier")

            cleaned_vols[expiry_index, strike_index] = float(cleaned_vol)
            adjustment = float(cleaned_vol - raw_vol)
            if flags or abs(adjustment) > 1e-12:
                adjustments.append(
                    EquityVolQuoteAdjustment(
                        expiry_years=float(surface.expiries[expiry_index]),
                        strike=float(surface.strikes[strike_index]),
                        raw_vol=float(raw_vol),
                        cleaned_vol=float(cleaned_vol),
                        adjustment=float(adjustment),
                        relative_adjustment=float(adjustment / raw_vol) if abs(raw_vol) > 1e-12 else 0.0,
                        flags=tuple(flags or ("manual_adjustment",)),
                    )
                )

    raw_calendar_violation_count = _calendar_total_variance_violations(
        expiries=raw_np.asarray(surface.expiries, dtype=float),
        strikes=raw_np.asarray(surface.strikes, dtype=float),
        vol_grid=raw_vols,
    )
    cleaned_calendar_violation_count = _calendar_total_variance_violations(
        expiries=raw_np.asarray(surface.expiries, dtype=float),
        strikes=raw_np.asarray(surface.strikes, dtype=float),
        vol_grid=cleaned_vols,
    )
    raw_smile_violation_count = _surface_smile_violation_count(surface, raw_vols)
    cleaned_smile_violation_count = _surface_smile_violation_count(surface, cleaned_vols)
    adjustment_grid = cleaned_vols - raw_vols

    warnings = list(surface.warnings)
    if adjustments:
        warnings.append(
            f"Quote cleaning adjusted {len(adjustments)} observed equity-vol nodes before surface repair."
        )

    cleaned_surface = replace(
        surface,
        market_vols=tuple(tuple(float(vol) for vol in row) for row in cleaned_vols),
        warnings=tuple(warnings),
        metadata={
            **dict(surface.metadata),
            "quote_cleaning_adjusted_point_count": len(adjustments),
        },
    )
    diagnostics = EquityVolQuoteCleaningDiagnostics(
        adjusted_point_count=len(adjustments),
        max_abs_adjustment=float(raw_np.max(raw_np.abs(adjustment_grid))) if adjustment_grid.size else 0.0,
        rms_adjustment=float(raw_np.sqrt(raw_np.mean(adjustment_grid ** 2))) if adjustment_grid.size else 0.0,
        raw_smile_violation_count=raw_smile_violation_count,
        cleaned_smile_violation_count=cleaned_smile_violation_count,
        raw_calendar_violation_count=raw_calendar_violation_count,
        cleaned_calendar_violation_count=cleaned_calendar_violation_count,
        point_count=surface.point_count,
        flagged_nodes=tuple(adjustments),
        warning_count=len(warnings),
    )
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "clean_equity_vol_surface_quotes",
        "raw_surface": surface.to_payload(),
        "fit_diagnostics": diagnostics.to_payload(),
        "warnings": list(warnings),
    }
    summary = {
        "surface_name": surface.surface_name,
        "cleaning_stage": "quote_governance",
        "point_count": surface.point_count,
        "adjusted_point_count": len(adjustments),
        "raw_smile_violation_count": raw_smile_violation_count,
        "cleaned_smile_violation_count": cleaned_smile_violation_count,
        "raw_calendar_violation_count": raw_calendar_violation_count,
        "cleaned_calendar_violation_count": cleaned_calendar_violation_count,
    }
    return EquityVolQuoteCleaningResult(
        raw_surface=surface,
        cleaned_surface=cleaned_surface,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        provenance=provenance,
        summary=summary,
    )


def clean_equity_vol_surface_quotes(
    spot: float,
    expiries: Sequence[float],
    strikes: Sequence[float],
    market_vols: Sequence[Sequence[float]],
    *,
    rate: float,
    dividend_yield: float = 0.0,
    surface_name: str = "equity_vol_surface",
    metadata: Mapping[str, object] | None = None,
) -> EquityVolQuoteCleaningResult:
    """Run the quote-governance stage on a raw equity-vol quote grid."""
    surface = build_equity_vol_surface_input(
        spot,
        expiries,
        strikes,
        market_vols,
        rate=rate,
        dividend_yield=dividend_yield,
        surface_name=surface_name,
        metadata=metadata,
    )
    return _govern_equity_vol_surface_input(surface)


def _default_svi_initial_guess(
    *,
    log_moneyness: raw_np.ndarray,
    target_total_variance: raw_np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Return a stable raw-SVI starting point."""
    minimum = float(raw_np.min(target_total_variance))
    maximum = float(raw_np.max(target_total_variance))
    atm_index = int(raw_np.argmin(raw_np.abs(log_moneyness)))
    wing_span = max(float(raw_np.max(log_moneyness) - raw_np.min(log_moneyness)), 0.25)
    skew = float(target_total_variance[-1] - target_total_variance[0])
    rho_guess = -0.3 if skew < -1e-8 else 0.3 if skew > 1e-8 else 0.0
    sigma_guess = max(0.15, min(0.75, 0.25 * wing_span))
    b_guess = max((maximum - minimum) / max(wing_span + sigma_guess, 1e-6), 0.05)
    a_guess = max(minimum - b_guess * sigma_guess, 1e-6)
    m_guess = float(log_moneyness[atm_index])
    return (a_guess, b_guess, rho_guess, m_guess, sigma_guess)


def _svi_penalty_values(
    *,
    parameters: Sequence[float],
    spot: float,
    rate: float,
    dividend_yield: float,
    expiry_years: float,
    forward: float,
    diagnostic_strikes: raw_np.ndarray,
) -> tuple[raw_np.ndarray, list[str]]:
    """Return positive no-arbitrage penalty values and their labels."""
    a, b, rho, m, sigma = (float(value) for value in parameters)
    diagnostic_log_moneyness = raw_np.log(diagnostic_strikes / max(float(forward), 1e-12))
    total_variance = raw_np.asarray(
        _svi_total_variance(
            diagnostic_log_moneyness,
            a=a,
            b=b,
            rho=rho,
            m=m,
            sigma=sigma,
        ),
        dtype=float,
    )
    positivity_penalties = raw_np.maximum(1e-8 - total_variance, 0.0)
    vols = raw_np.sqrt(raw_np.maximum(total_variance / float(expiry_years), 1e-12))
    call_prices = _call_prices_from_smile(
        spot=float(spot),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        expiry_years=float(expiry_years),
        strikes=diagnostic_strikes,
        vols=vols,
    )
    monotonicity_penalties = raw_np.maximum(call_prices[1:] - call_prices[:-1], 0.0)
    if diagnostic_strikes.size >= 3:
        slopes = raw_np.diff(call_prices) / raw_np.diff(diagnostic_strikes)
        convexity_penalties = raw_np.maximum(slopes[:-1] - slopes[1:], 0.0)
    else:
        convexity_penalties = raw_np.asarray((), dtype=float)

    labels = [f"positivity_{index}" for index in range(positivity_penalties.size)]
    labels.extend(f"monotonicity_{index}" for index in range(monotonicity_penalties.size))
    labels.extend(f"convexity_{index}" for index in range(convexity_penalties.size))
    penalties = raw_np.concatenate(
        (
            positivity_penalties,
            monotonicity_penalties,
            convexity_penalties,
        ),
        dtype=float,
    )
    return penalties, labels


def _fit_one_svi_smile(
    surface: EquityVolSurfaceInput,
    *,
    expiry_index: int,
    penalty_weight: float,
) -> SVISmileFitResult:
    """Fit one bounded raw-SVI smile with no-arbitrage penalty residuals."""
    expiry_years = float(surface.expiries[expiry_index])
    strikes = raw_np.asarray(surface.strikes, dtype=float)
    market_vols = raw_np.asarray(surface.market_vols[expiry_index], dtype=float)
    forward = surface.forward(expiry_years)
    log_moneyness = raw_np.log(strikes / max(float(forward), 1e-12))
    target_total_variance = raw_np.asarray(
        [_representative_total_variance(expiry_years, vol) for vol in market_vols],
        dtype=float,
    )
    raw_prices = _call_prices_from_smile(
        spot=surface.spot,
        rate=surface.rate,
        dividend_yield=surface.dividend_yield,
        expiry_years=expiry_years,
        strikes=strikes,
        vols=market_vols,
    )
    raw_monotonicity_violations, raw_convexity_violations = _smile_no_arb_violation_counts(
        strikes=strikes,
        call_prices=raw_prices,
    )

    strike_padding = max(float(strikes[-1] - strikes[0]) * 0.10, 1e-3)
    diagnostic_strikes = raw_np.linspace(
        max(float(strikes[0]) - strike_padding, 1e-6),
        float(strikes[-1]) + strike_padding,
        max(int(strikes.size) * 3, 15),
    )
    initial_guess = _default_svi_initial_guess(
        log_moneyness=log_moneyness,
        target_total_variance=target_total_variance,
    )
    penalty_sample, penalty_labels = _svi_penalty_values(
        parameters=initial_guess,
        spot=surface.spot,
        rate=surface.rate,
        dividend_yield=surface.dividend_yield,
        expiry_years=expiry_years,
        forward=forward,
        diagnostic_strikes=diagnostic_strikes,
    )
    labels = tuple(
        [f"total_variance_strike_{float(strike):g}" for strike in strikes]
        + penalty_labels
    )
    target_values = tuple(
        list(float(value) for value in target_total_variance)
        + [0.0 for _ in range(int(penalty_sample.size))]
    )
    weights = tuple(
        [1.0 for _ in range(len(strikes))]
        + [float(penalty_weight) for _ in range(int(penalty_sample.size))]
    )

    def vector_objective_fn(params: raw_np.ndarray) -> raw_np.ndarray:
        observed_total_variance = raw_np.asarray(
            _svi_total_variance(
                log_moneyness,
                a=float(params[0]),
                b=float(params[1]),
                rho=float(params[2]),
                m=float(params[3]),
                sigma=float(params[4]),
            ),
            dtype=float,
        )
        penalties, _ = _svi_penalty_values(
            parameters=params,
            spot=surface.spot,
            rate=surface.rate,
            dividend_yield=surface.dividend_yield,
            expiry_years=expiry_years,
            forward=forward,
            diagnostic_strikes=diagnostic_strikes,
        )
        return raw_np.concatenate((observed_total_variance, penalties), dtype=float)

    request = SolveRequest(
        request_id=f"svi_smile_{str(expiry_years).replace('.', '_')}y_least_squares",
        problem_kind="least_squares",
        parameter_names=("a", "b", "rho", "m", "sigma"),
        initial_guess=tuple(float(value) for value in initial_guess),
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=labels,
            target_values=target_values,
            weights=weights,
            vector_objective_fn=vector_objective_fn,
            metadata={
                "model_family": "raw_svi",
                "fit_space": "total_variance",
                "expiry_years": float(expiry_years),
                "surface_name": surface.surface_name,
                "quote_map": surface.quote_map_spec.to_payload(),
                "penalty_kind": "discrete_no_arb",
            },
        ),
        bounds=SolveBounds(
            lower=(-0.5, 1e-4, -0.999, -3.0, 1e-4),
            upper=(5.0, 10.0, 0.999, 3.0, 3.0),
        ),
        solver_hint="trf",
        metadata={
            "surface_name": surface.surface_name,
            "expiry_years": float(expiry_years),
            "model_family": "raw_svi",
            "admissibility_policy": (
                "Bounded raw-SVI parameter fit with explicit positivity, monotonicity, "
                "and convexity penalty residuals on a diagnostic strike grid."
            ),
            "input_surface": surface.to_payload(),
        },
        options={"ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10, "maxiter": 200},
    )
    solve_result = execute_solve_request(request)
    solver_provenance = build_solve_provenance(request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(request, solve_result)

    parameters = SVISmileParameters(
        expiry_years=expiry_years,
        forward=forward,
        a=float(solve_result.solution[0]),
        b=float(solve_result.solution[1]),
        rho=float(solve_result.solution[2]),
        m=float(solve_result.solution[3]),
        sigma=float(solve_result.solution[4]),
    )
    repaired_total_variance = raw_np.asarray(parameters.total_variance(log_moneyness), dtype=float)
    repaired_vols = raw_np.sqrt(raw_np.maximum(repaired_total_variance / expiry_years, 1e-12))
    repaired_prices = _call_prices_from_smile(
        spot=surface.spot,
        rate=surface.rate,
        dividend_yield=surface.dividend_yield,
        expiry_years=expiry_years,
        strikes=strikes,
        vols=repaired_vols,
    )
    repaired_monotonicity_violations, repaired_convexity_violations = _smile_no_arb_violation_counts(
        strikes=strikes,
        call_prices=repaired_prices,
    )
    residuals = repaired_vols - market_vols
    warnings: list[str] = []
    if raw_monotonicity_violations or raw_convexity_violations:
        warnings.append(
            "Raw market smile showed discrete no-arbitrage violations before the repaired surface fit."
        )
    if repaired_monotonicity_violations or repaired_convexity_violations:
        warnings.append(
            "Repaired SVI smile still shows discrete no-arbitrage violations on the observed strike grid."
        )
    diagnostics = SVISmileFitDiagnostics(
        expiry_years=expiry_years,
        market_vols=tuple(float(value) for value in market_vols),
        repaired_vols=tuple(float(value) for value in repaired_vols),
        residuals=tuple(float(value) for value in residuals),
        max_abs_vol_error=float(raw_np.max(raw_np.abs(residuals))),
        rms_vol_error=float(raw_np.sqrt(raw_np.mean(residuals * residuals))),
        raw_monotonicity_violation_count=raw_monotonicity_violations,
        raw_convexity_violation_count=raw_convexity_violations,
        repaired_monotonicity_violation_count=repaired_monotonicity_violations,
        repaired_convexity_violation_count=repaired_convexity_violations,
        warning_count=len(warnings),
    )
    return SVISmileFitResult(
        expiry_years=expiry_years,
        parameters=parameters,
        solve_request=request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )


def fit_equity_vol_surface(
    surface: EquityVolSurfaceInput,
    *,
    penalty_weight: float = 1_000.0,
    surface_name: str | None = None,
) -> EquityVolSurfaceAuthorityResult:
    """Fit a bounded repaired equity-vol authority surface from multi-expiry quotes."""
    if not raw_np.isfinite(penalty_weight) or float(penalty_weight) <= 0.0:
        raise ValueError("penalty_weight must be finite and positive")
    smile_fits = tuple(
        _fit_one_svi_smile(
            surface,
            expiry_index=index,
            penalty_weight=float(penalty_weight),
        )
        for index in range(len(surface.expiries))
    )
    vol_surface = SVIVolSurface(
        spot=surface.spot,
        rate=surface.rate,
        dividend_yield=surface.dividend_yield,
        smiles=tuple(smile.parameters for smile in smile_fits),
        repair_calendar_monotonicity=True,
    )
    repaired_vols = vol_surface.sample_grid(expiries=surface.expiries, strikes=surface.strikes)
    repaired_vol_array = raw_np.asarray(repaired_vols, dtype=float)
    market_vol_array = raw_np.asarray(surface.market_vols, dtype=float)
    residuals = repaired_vol_array - market_vol_array
    raw_calendar_violation_count = _calendar_total_variance_violations(
        expiries=raw_np.asarray(surface.expiries, dtype=float),
        strikes=raw_np.asarray(surface.strikes, dtype=float),
        vol_grid=market_vol_array,
    )
    repaired_calendar_violation_count = _calendar_total_variance_violations(
        expiries=raw_np.asarray(surface.expiries, dtype=float),
        strikes=raw_np.asarray(surface.strikes, dtype=float),
        vol_grid=repaired_vol_array,
    )
    diagnostics = EquityVolSurfaceFitDiagnostics(
        smile_fits=tuple(smile.diagnostics for smile in smile_fits),
        raw_calendar_violation_count=raw_calendar_violation_count,
        repaired_calendar_violation_count=repaired_calendar_violation_count,
        raw_smile_violation_count=sum(
            smile.diagnostics.raw_monotonicity_violation_count
            + smile.diagnostics.raw_convexity_violation_count
            for smile in smile_fits
        ),
        repaired_smile_violation_count=sum(
            smile.diagnostics.repaired_monotonicity_violation_count
            + smile.diagnostics.repaired_convexity_violation_count
            for smile in smile_fits
        ),
        max_abs_vol_error=float(raw_np.max(raw_np.abs(residuals))),
        rms_vol_error=float(raw_np.sqrt(raw_np.mean(residuals * residuals))),
        point_count=surface.point_count,
    )

    warnings = list(surface.warnings)
    if diagnostics.raw_smile_violation_count or diagnostics.raw_calendar_violation_count:
        warnings.append(
            "Raw equity-vol quotes showed discrete smile or calendar no-arbitrage violations before surface repair."
        )
    if diagnostics.repaired_smile_violation_count or diagnostics.repaired_calendar_violation_count:
        warnings.append(
            "The repaired equity-vol surface still shows residual discrete no-arbitrage violations on the observed grid."
        )
    surface_name = str(surface_name or surface.surface_name or "equity_vol_surface")
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "fit_equity_vol_surface",
        "calibration_target": surface.to_payload(),
        "smile_fits": [smile.to_payload() for smile in smile_fits],
        "fit_diagnostics": diagnostics.to_payload(),
        "warnings": list(warnings),
    }
    summary = {
        "surface_name": surface_name,
        "spot": float(surface.spot),
        "rate": float(surface.rate),
        "dividend_yield": float(surface.dividend_yield),
        "expiry_count": len(surface.expiries),
        "strike_count": len(surface.strikes),
        "point_count": surface.point_count,
        "surface_model_family": "raw_svi_surface",
        "max_abs_vol_error": float(diagnostics.max_abs_vol_error),
        "rms_vol_error": float(diagnostics.rms_vol_error),
        "raw_calendar_violation_count": int(diagnostics.raw_calendar_violation_count),
        "repaired_calendar_violation_count": int(diagnostics.repaired_calendar_violation_count),
        "raw_smile_violation_count": int(diagnostics.raw_smile_violation_count),
        "repaired_smile_violation_count": int(diagnostics.repaired_smile_violation_count),
    }
    return EquityVolSurfaceAuthorityResult(
        input_surface=surface,
        vol_surface=vol_surface,
        repaired_vols=repaired_vols,
        smile_fits=smile_fits,
        diagnostics=diagnostics,
        surface_name=surface_name,
        raw_input_surface=surface,
        warnings=tuple(warnings),
        provenance=provenance,
        summary=summary,
    )


def calibrate_equity_vol_surface_workflow(
    spot: float,
    expiries: Sequence[float],
    strikes: Sequence[float],
    market_vols: Sequence[Sequence[float]],
    *,
    rate: float,
    dividend_yield: float = 0.0,
    surface_name: str = "equity_vol_surface",
    metadata: Mapping[str, object] | None = None,
) -> EquityVolSurfaceAuthorityResult:
    """Run the bounded repaired equity-vol surface workflow from raw grid inputs."""
    raw_surface = build_equity_vol_surface_input(
        spot,
        expiries,
        strikes,
        market_vols,
        rate=rate,
        dividend_yield=dividend_yield,
        surface_name=surface_name,
        metadata=metadata,
    )
    quote_cleaning = _govern_equity_vol_surface_input(raw_surface)
    result = fit_equity_vol_surface(quote_cleaning.cleaned_surface, surface_name=surface_name)
    provenance = dict(result.provenance)
    provenance["source_ref"] = "calibrate_equity_vol_surface_workflow"
    provenance["raw_input_surface"] = raw_surface.to_payload()
    provenance["quote_cleaning"] = quote_cleaning.to_payload()
    summary = dict(result.summary)
    summary["raw_smile_violation_count"] = int(quote_cleaning.diagnostics.raw_smile_violation_count)
    summary["cleaned_smile_violation_count"] = int(quote_cleaning.diagnostics.cleaned_smile_violation_count)
    summary["raw_calendar_violation_count"] = int(quote_cleaning.diagnostics.raw_calendar_violation_count)
    summary["cleaned_calendar_violation_count"] = int(quote_cleaning.diagnostics.cleaned_calendar_violation_count)
    summary["quote_cleaning_adjusted_point_count"] = int(quote_cleaning.diagnostics.adjusted_point_count)
    return replace(
        result,
        raw_input_surface=raw_surface,
        quote_cleaning=quote_cleaning,
        provenance=provenance,
        summary=summary,
    )


def calibrate_local_vol_surface_from_equity_vol_surface_workflow(
    authority_result: EquityVolSurfaceAuthorityResult,
    *,
    strikes: Sequence[float] | None = None,
    expiries: Sequence[float] | None = None,
    surface_name: str = "local_vol",
    metadata: Mapping[str, object] | None = None,
) -> LocalVolCalibrationResult:
    """Extract local vol from the repaired equity-vol surface rather than from raw quotes."""
    sampled_strikes = tuple(float(value) for value in (strikes or authority_result.input_surface.strikes))
    sampled_expiries = tuple(float(value) for value in (expiries or authority_result.input_surface.expiries))
    repaired_vol_grid = authority_result.vol_surface.sample_grid(
        expiries=sampled_expiries,
        strikes=sampled_strikes,
    )
    local_vol_result = calibrate_local_vol_surface_workflow(
        raw_np.asarray(sampled_strikes, dtype=float),
        raw_np.asarray(sampled_expiries, dtype=float),
        raw_np.asarray(repaired_vol_grid, dtype=float),
        authority_result.input_surface.spot,
        authority_result.input_surface.rate,
        dividend_yield=authority_result.input_surface.dividend_yield,
        surface_name=surface_name,
        metadata={
            "source_surface_name": authority_result.surface_name,
            "source_surface_kind": "repaired_equity_vol_surface",
            **dict(metadata or {}),
        },
    )
    calibration_target = dict(local_vol_result.calibration_target)
    calibration_target["source_surface_name"] = authority_result.surface_name
    calibration_target["source_surface_kind"] = "repaired_equity_vol_surface"
    provenance = dict(local_vol_result.provenance)
    provenance["source_ref"] = "calibrate_local_vol_surface_from_equity_vol_surface_workflow"
    provenance["source_surface"] = {
        "surface_name": authority_result.surface_name,
        "source_ref": str(authority_result.provenance.get("source_ref", "")),
        "fit_diagnostics": authority_result.diagnostics.to_payload(),
    }
    summary = dict(local_vol_result.summary)
    summary["source_surface_name"] = authority_result.surface_name
    summary["source_surface_kind"] = "repaired_equity_vol_surface"
    return replace(
        local_vol_result,
        calibration_target=calibration_target,
        provenance=provenance,
        summary=summary,
    )


def compare_heston_to_equity_vol_surface_workflow(
    authority_result: EquityVolSurfaceAuthorityResult,
    *,
    expiry_years: float,
    parameter_set_name: str = "heston",
    initial_guess: Sequence[float] | None = None,
    warm_start: Sequence[float] | None = None,
) -> EquityVolStageComparisonResult:
    """Compare the repaired surface stage against a Heston smile fit on one expiry slice."""
    reference_surface = authority_result.raw_input_surface or authority_result.input_surface
    expiry_index = _normalize_expiry_index(authority_result.input_surface.expiries, float(expiry_years))
    strikes = tuple(float(value) for value in authority_result.input_surface.strikes)
    market_vols = tuple(float(value) for value in reference_surface.market_vols[expiry_index])
    surface_vols = tuple(
        authority_result.vol_surface.black_vol(float(expiry_years), float(strike))
        for strike in strikes
    )
    heston_result = calibrate_heston_smile_workflow(
        authority_result.input_surface.spot,
        float(expiry_years),
        strikes,
        market_vols,
        rate=authority_result.input_surface.rate,
        dividend_yield=authority_result.input_surface.dividend_yield,
        surface_name=f"{authority_result.surface_name}_{str(expiry_years).replace('.', '_')}y_heston_stage",
        parameter_set_name=parameter_set_name,
        initial_guess=initial_guess,
        warm_start=warm_start,
        metadata={
            "source_surface_name": authority_result.surface_name,
            "stage_kind": "model_compression",
        },
    )
    model_vols = tuple(float(value) for value in heston_result.diagnostics.model_vols)
    surface_residuals = tuple(float(surface_vol - market_vol) for surface_vol, market_vol in zip(surface_vols, market_vols))
    model_residuals = tuple(float(model_vol - market_vol) for model_vol, market_vol in zip(model_vols, market_vols))
    surface_max_abs_vol_error = max((abs(value) for value in surface_residuals), default=0.0)
    model_max_abs_vol_error = max((abs(value) for value in model_residuals), default=0.0)
    preferred_stage = "surface_authority" if surface_max_abs_vol_error <= model_max_abs_vol_error else "model_fit"
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "compare_heston_to_equity_vol_surface_workflow",
        "surface_stage": authority_result.to_payload(),
        "model_stage": heston_result.to_payload(),
    }
    summary = {
        "expiry_years": float(expiry_years),
        "point_count": len(strikes),
        "surface_max_abs_vol_error": float(surface_max_abs_vol_error),
        "model_max_abs_vol_error": float(model_max_abs_vol_error),
        "preferred_stage": preferred_stage,
    }
    return EquityVolStageComparisonResult(
        expiry_years=float(expiry_years),
        strikes=strikes,
        market_vols=market_vols,
        surface_vols=surface_vols,
        model_vols=model_vols,
        surface_residuals=surface_residuals,
        model_residuals=model_residuals,
        surface_max_abs_vol_error=surface_max_abs_vol_error,
        model_max_abs_vol_error=model_max_abs_vol_error,
        preferred_stage=preferred_stage,
        heston_result=heston_result,
        provenance=provenance,
        summary=summary,
    )


def compare_heston_surface_to_equity_vol_surface_workflow(
    authority_result: EquityVolSurfaceAuthorityResult,
    *,
    parameter_set_name: str = "heston",
    initial_guess: Sequence[float] | None = None,
    warm_start: Sequence[float] | None = None,
) -> EquityVolSurfaceStageComparisonResult:
    """Compare the repaired surface stage against Heston compression on the full surface."""
    reference_surface = authority_result.raw_input_surface or authority_result.input_surface
    surface_vols = tuple(tuple(float(value) for value in row) for row in authority_result.repaired_vols)
    heston_result = calibrate_heston_surface_from_equity_vol_surface_workflow(
        authority_result,
        parameter_set_name=parameter_set_name,
        initial_guess=initial_guess,
        warm_start=warm_start,
    )
    model_vols = tuple(tuple(float(value) for value in row) for row in heston_result.model_vols)
    market_vols = tuple(tuple(float(value) for value in row) for row in reference_surface.market_vols)
    surface_residuals = raw_np.asarray(surface_vols, dtype=float) - raw_np.asarray(market_vols, dtype=float)
    model_residuals = raw_np.asarray(model_vols, dtype=float) - raw_np.asarray(market_vols, dtype=float)
    surface_max_abs_vol_error = float(raw_np.max(raw_np.abs(surface_residuals)))
    model_max_abs_vol_error = float(raw_np.max(raw_np.abs(model_residuals)))
    surface_rms_vol_error = float(raw_np.sqrt(raw_np.mean(surface_residuals ** 2)))
    model_rms_vol_error = float(raw_np.sqrt(raw_np.mean(model_residuals ** 2)))
    preferred_stage = "surface_authority" if surface_max_abs_vol_error <= model_max_abs_vol_error else "model_fit"
    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "compare_heston_surface_to_equity_vol_surface_workflow",
        "surface_stage": authority_result.to_payload(),
        "model_stage": heston_result.to_payload(),
    }
    summary = {
        "point_count": len(authority_result.input_surface.expiries) * len(authority_result.input_surface.strikes),
        "surface_max_abs_vol_error": surface_max_abs_vol_error,
        "model_max_abs_vol_error": model_max_abs_vol_error,
        "surface_rms_vol_error": surface_rms_vol_error,
        "model_rms_vol_error": model_rms_vol_error,
        "preferred_stage": preferred_stage,
    }
    return EquityVolSurfaceStageComparisonResult(
        expiries=authority_result.input_surface.expiries,
        strikes=authority_result.input_surface.strikes,
        market_vols=market_vols,
        surface_vols=surface_vols,
        model_vols=model_vols,
        surface_max_abs_vol_error=surface_max_abs_vol_error,
        model_max_abs_vol_error=model_max_abs_vol_error,
        surface_rms_vol_error=surface_rms_vol_error,
        model_rms_vol_error=model_rms_vol_error,
        preferred_stage=preferred_stage,
        heston_surface_result=heston_result,
        provenance=provenance,
        summary=summary,
    )


__all__ = [
    "EquityVolSurfaceInput",
    "EquityVolQuoteAdjustment",
    "EquityVolQuoteCleaningDiagnostics",
    "EquityVolQuoteCleaningResult",
    "SVISmileParameters",
    "SVISmileFitDiagnostics",
    "SVISmileFitResult",
    "SVIVolSurface",
    "EquityVolSurfaceFitDiagnostics",
    "EquityVolSurfaceAuthorityResult",
    "EquityVolStageComparisonResult",
    "EquityVolSurfaceStageComparisonResult",
    "build_equity_vol_surface_input",
    "clean_equity_vol_surface_quotes",
    "fit_equity_vol_surface",
    "calibrate_equity_vol_surface_workflow",
    "calibrate_local_vol_surface_from_equity_vol_surface_workflow",
    "compare_heston_to_equity_vol_surface_workflow",
    "compare_heston_surface_to_equity_vol_surface_workflow",
]
