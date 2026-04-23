"""Differentiable curve bootstrapping with typed market-input and solve surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as raw_np

from trellis.conventions.calendar import BusinessDayAdjustment, Calendar
from trellis.conventions.day_count import DayCountConvention
from trellis.conventions.schedule import RollConvention, StubType
from trellis.core.date_utils import build_period_schedule, year_fraction
from trellis.core.differentiable import get_numpy, jacobian
from trellis.core.market_state import MarketState
from trellis.core.types import Frequency
from trellis.curves.interpolation import linear_interp
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveProvenance,
    SolveReplayArtifact,
    SolveRequest,
    SolveResult,
    WarmStart,
    build_solve_provenance,
    build_solve_replay_artifact,
    execute_solve_request,
)

np = get_numpy()

BootstrapInstrumentType = Literal["deposit", "future", "swap"]
BootstrapCurveRole = Literal["discount_curve", "forecast_curve", "basis_curve"]


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy for metadata payloads."""
    return MappingProxyType(dict(mapping or {}))


def _enum_name(value: object) -> object:
    """Return ``Enum.name`` when available, otherwise the original value."""
    return getattr(value, "name", value)


def _normalize_float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize numeric sequences onto immutable float tuples."""
    return tuple(float(value) for value in values)


def _normalize_matrix(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Normalize nested numeric sequences onto immutable tuples."""
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _frequency_years(frequency: Frequency) -> float:
    """Return the simple year fraction implied by one coupon/reset period."""
    return 1.0 / float(frequency.value)


def _discount_factor(t: float, rates: object, tenors: object) -> object:
    """Return the discount factor implied by ``rates`` at time ``t``."""
    if t <= 0.0:
        return np.array(1.0)
    r = linear_interp(t, tenors, rates)
    return np.exp(-r * t)


def _schedule_times(start: float, end: float, step: float) -> tuple[float, ...]:
    """Return payment/reset times from ``start`` to ``end`` using ``step``."""
    if step <= 0.0:
        raise ValueError("schedule step must be positive")
    if end <= start:
        raise ValueError("schedule end must be greater than start")

    times: list[float] = []
    current = float(start)
    target = float(end)
    while current + step < target - 1e-12:
        current += step
        times.append(float(current))
    times.append(target)
    return tuple(times)


@dataclass(frozen=True)
class BootstrapInstrument:
    """One market calibration instrument quote for a curve bootstrap."""

    tenor: float
    quote: float
    instrument_type: BootstrapInstrumentType = "deposit"
    start_tenor: float = 0.0
    accrual_years: float | None = None
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        instrument_types = {"deposit", "future", "swap"}
        if self.instrument_type not in instrument_types:
            raise ValueError(
                f"instrument_type must be one of {sorted(instrument_types)}, got {self.instrument_type!r}"
            )
        tenor = float(self.tenor)
        start_tenor = float(self.start_tenor)
        if tenor <= 0.0:
            raise ValueError("BootstrapInstrument tenor must be positive")
        if start_tenor < 0.0:
            raise ValueError("BootstrapInstrument start_tenor must be non-negative")
        if tenor <= start_tenor:
            raise ValueError("BootstrapInstrument tenor must be greater than start_tenor")
        accrual_years = None if self.accrual_years is None else float(self.accrual_years)
        if accrual_years is not None and accrual_years <= 0.0:
            raise ValueError("BootstrapInstrument accrual_years must be positive when provided")
        object.__setattr__(self, "tenor", tenor)
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "start_tenor", start_tenor)
        object.__setattr__(self, "accrual_years", accrual_years)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def natural_accrual_years(self) -> float:
        """Return the start/end tenor gap implied by the instrument times."""
        return float(self.tenor) - float(self.start_tenor)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "tenor": float(self.tenor),
            "quote": float(self.quote),
            "instrument_type": self.instrument_type,
            "start_tenor": float(self.start_tenor),
            "accrual_years": None if self.accrual_years is None else float(self.accrual_years),
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BootstrapConventionBundle:
    """Explicit bootstrap conventions for rates calibration inputs."""

    deposit_day_count: DayCountConvention = DayCountConvention.ACT_360
    deposit_compounding: Literal["simple"] = "simple"
    future_contract_frequency: Frequency = Frequency.QUARTERLY
    future_day_count: DayCountConvention = DayCountConvention.ACT_360
    future_quote_style: Literal["price", "rate"] = "price"
    swap_fixed_frequency: Frequency = Frequency.SEMI_ANNUAL
    swap_fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360_US
    swap_float_frequency: Frequency = Frequency.QUARTERLY
    swap_float_day_count: DayCountConvention = DayCountConvention.ACT_360
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deposit_compounding != "simple":
            raise ValueError("Only simple deposit compounding is currently supported")
        if self.future_quote_style not in {"price", "rate"}:
            raise ValueError("future_quote_style must be 'price' or 'rate'")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def future_contract_years(self) -> float:
        """Return the default accrual length for futures under this bundle."""
        return _frequency_years(self.future_contract_frequency)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "deposit_day_count": _enum_name(self.deposit_day_count),
            "deposit_compounding": self.deposit_compounding,
            "future_contract_frequency": _enum_name(self.future_contract_frequency),
            "future_day_count": _enum_name(self.future_day_count),
            "future_quote_style": self.future_quote_style,
            "swap_fixed_frequency": _enum_name(self.swap_fixed_frequency),
            "swap_fixed_day_count": _enum_name(self.swap_fixed_day_count),
            "swap_float_frequency": _enum_name(self.swap_float_frequency),
            "swap_float_day_count": _enum_name(self.swap_float_day_count),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BootstrapCurveInputBundle:
    """Reusable market-instrument and convention bundle for one curve."""

    instruments: tuple[BootstrapInstrument, ...]
    curve_name: str = ""
    currency: str = ""
    rate_index: str | None = None
    conventions: BootstrapConventionBundle = field(default_factory=BootstrapConventionBundle)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        instruments = tuple(self.instruments)
        if not instruments:
            raise ValueError("BootstrapCurveInputBundle requires at least one instrument")
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def sorted_instruments(self) -> tuple[BootstrapInstrument, ...]:
        """Return instruments in deterministic bootstrap order."""
        return tuple(
            sorted(
                self.instruments,
                key=lambda inst: (
                    float(inst.tenor),
                    float(inst.start_tenor),
                    inst.instrument_type,
                    float(inst.quote),
                    inst.label,
                ),
            )
        )

    def with_curve_name(self, curve_name: str) -> BootstrapCurveInputBundle:
        """Return ``self`` with ``curve_name`` filled when currently blank."""
        if not curve_name or self.curve_name == curve_name:
            return self
        if self.curve_name:
            return self
        return replace(self, curve_name=curve_name)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "curve_name": self.curve_name,
            "currency": self.currency,
            "rate_index": self.rate_index,
            "conventions": self.conventions.to_payload(),
            "instruments": [inst.to_payload() for inst in self.sorted_instruments()],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatedBootstrapInstrument:
    """One dated market calibration instrument quote for a bootstrap."""

    start_date: date
    end_date: date
    quote: float
    instrument_type: BootstrapInstrumentType = "deposit"
    label: str = ""
    day_count: DayCountConvention | None = None
    fixed_frequency: Frequency | None = None
    fixed_day_count: DayCountConvention | None = None
    float_frequency: Frequency | None = None
    float_day_count: DayCountConvention | None = None
    stub_type: StubType = StubType.SHORT_LAST
    roll_convention: RollConvention = RollConvention.NONE
    calendar: Calendar | None = None
    business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.UNADJUSTED
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        instrument_types = {"deposit", "future", "swap"}
        if self.instrument_type not in instrument_types:
            raise ValueError(
                f"instrument_type must be one of {sorted(instrument_types)}, got {self.instrument_type!r}"
            )
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "quote": float(self.quote),
            "instrument_type": self.instrument_type,
            "label": self.label,
            "day_count": None if self.day_count is None else _enum_name(self.day_count),
            "fixed_frequency": None if self.fixed_frequency is None else _enum_name(self.fixed_frequency),
            "fixed_day_count": None if self.fixed_day_count is None else _enum_name(self.fixed_day_count),
            "float_frequency": None if self.float_frequency is None else _enum_name(self.float_frequency),
            "float_day_count": None if self.float_day_count is None else _enum_name(self.float_day_count),
            "stub_type": _enum_name(self.stub_type),
            "roll_convention": _enum_name(self.roll_convention),
            "calendar": None if self.calendar is None else self.calendar.name,
            "business_day_adjustment": _enum_name(self.business_day_adjustment),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatedBootstrapCurveInputBundle:
    """Reusable dated market-instrument and convention bundle for one curve."""

    instruments: tuple[DatedBootstrapInstrument, ...]
    curve_name: str
    currency: str = ""
    rate_index: str | None = None
    curve_role: BootstrapCurveRole = "discount_curve"
    dependency_names: Mapping[str, str] = field(default_factory=dict)
    conventions: BootstrapConventionBundle = field(default_factory=BootstrapConventionBundle)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        instruments = tuple(self.instruments)
        if not instruments:
            raise ValueError("DatedBootstrapCurveInputBundle requires at least one instrument")
        if not str(self.curve_name or "").strip():
            raise ValueError("DatedBootstrapCurveInputBundle requires curve_name")
        if self.curve_role not in {"discount_curve", "forecast_curve", "basis_curve"}:
            raise ValueError("curve_role must be one of discount_curve, forecast_curve, basis_curve")
        object.__setattr__(self, "instruments", instruments)
        object.__setattr__(
            self,
            "dependency_names",
            MappingProxyType(
                {
                    str(key).strip(): str(value).strip()
                    for key, value in dict(self.dependency_names or {}).items()
                    if str(key).strip() and str(value).strip()
                }
            ),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def sorted_instruments(self) -> tuple[DatedBootstrapInstrument, ...]:
        """Return instruments in deterministic bootstrap order."""
        return tuple(
            sorted(
                self.instruments,
                key=lambda inst: (
                    inst.end_date,
                    inst.start_date,
                    inst.instrument_type,
                    float(inst.quote),
                    inst.label,
                ),
            )
        )

    def with_curve_name(self, curve_name: str) -> DatedBootstrapCurveInputBundle:
        """Return ``self`` with ``curve_name`` filled when currently blank."""
        if not curve_name or self.curve_name == curve_name:
            return self
        return replace(self, curve_name=curve_name)

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "curve_name": self.curve_name,
            "currency": self.currency,
            "rate_index": self.rate_index,
            "curve_role": self.curve_role,
            "dependency_names": dict(self.dependency_names),
            "conventions": self.conventions.to_payload(),
            "instruments": [inst.to_payload() for inst in self.sorted_instruments()],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiCurveBootstrapProgram:
    """Explicit dependency-aware program for chained multi-curve bootstraps."""

    settlement_date: date
    curve_inputs: tuple[DatedBootstrapCurveInputBundle, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bundles = tuple(self.curve_inputs)
        if not bundles:
            raise ValueError("MultiCurveBootstrapProgram requires at least one curve bundle")
        curve_names = [bundle.curve_name for bundle in bundles]
        if len(set(curve_names)) != len(curve_names):
            raise ValueError("MultiCurveBootstrapProgram curve names must be unique")
        object.__setattr__(self, "curve_inputs", bundles)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "settlement_date": self.settlement_date.isoformat(),
            "curve_inputs": [bundle.to_payload() for bundle in self.curve_inputs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MultiCurveBootstrapResult:
    """Structured result for a chained multi-curve bootstrap program."""

    program: MultiCurveBootstrapProgram
    node_results: Mapping[str, BootstrapCalibrationResult]
    dependency_order: tuple[str, ...]
    dependency_graph: Mapping[str, Mapping[str, str]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_results", MappingProxyType(dict(self.node_results)))
        object.__setattr__(self, "dependency_order", tuple(str(name) for name in self.dependency_order))
        object.__setattr__(
            self,
            "dependency_graph",
            MappingProxyType(
                {
                    str(name): MappingProxyType(
                        {
                            str(key): str(value)
                            for key, value in dict(edges).items()
                            if str(key).strip() and str(value).strip()
                        }
                    )
                    for name, edges in dict(self.dependency_graph).items()
                }
            ),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def apply_to_market_state(
        self,
        market_state: MarketState,
        *,
        discount_curve_name: str,
        forecast_curve_name: str | None = None,
    ) -> MarketState:
        """Return ``market_state`` enriched with the chained bootstrap outputs."""
        if discount_curve_name not in self.node_results:
            raise ValueError(f"Unknown discount_curve_name {discount_curve_name!r}")
        discount_curve = self.node_results[discount_curve_name].curve
        forecast_curves = dict(market_state.forecast_curves or {})
        if forecast_curve_name is not None:
            if forecast_curve_name not in self.node_results:
                raise ValueError(f"Unknown forecast_curve_name {forecast_curve_name!r}")
            forecast_curves[forecast_curve_name] = self.node_results[forecast_curve_name].curve
        selected_curve_names = dict(market_state.selected_curve_names or {})
        selected_curve_names["discount_curve"] = discount_curve_name
        if forecast_curve_name is not None:
            selected_curve_names["forecast_curve"] = forecast_curve_name
        market_provenance = dict(market_state.market_provenance or {})
        bootstrap_runs = dict(market_provenance.get("bootstrap_runs") or {})
        bootstrap_runs.setdefault("multi_curve_program", {})
        bootstrap_runs["multi_curve_program"] = {
            name: result.to_payload()
            for name, result in self.node_results.items()
        }
        market_provenance["bootstrap_runs"] = bootstrap_runs
        market_provenance["bootstrap_dependency_order"] = list(self.dependency_order)
        market_provenance["bootstrap_dependency_graph"] = {
            name: dict(edges)
            for name, edges in self.dependency_graph.items()
        }
        return replace(
            market_state,
            discount=discount_curve,
            forecast_curves=forecast_curves or None,
            selected_curve_names=selected_curve_names,
            market_provenance=market_provenance,
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "program": self.program.to_payload(),
            "node_results": {
                name: result.to_payload()
                for name, result in self.node_results.items()
            },
            "dependency_order": list(self.dependency_order),
            "dependency_graph": {
                name: dict(edges)
                for name, edges in self.dependency_graph.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BootstrapCalibrationDiagnostics:
    """Residual and Jacobian diagnostics for one bootstrap solve path."""

    model_values: tuple[float, ...]
    market_quotes: tuple[float, ...]
    residual_vector: tuple[float, ...]
    jacobian_matrix: tuple[tuple[float, ...], ...]
    max_abs_residual: float
    l2_norm: float
    jacobian_condition_number: float | None = None
    jacobian_rank: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_values", _normalize_float_tuple(self.model_values))
        object.__setattr__(self, "market_quotes", _normalize_float_tuple(self.market_quotes))
        object.__setattr__(self, "residual_vector", _normalize_float_tuple(self.residual_vector))
        object.__setattr__(self, "jacobian_matrix", _normalize_matrix(self.jacobian_matrix))
        object.__setattr__(self, "max_abs_residual", float(self.max_abs_residual))
        object.__setattr__(self, "l2_norm", float(self.l2_norm))
        if self.jacobian_condition_number is not None:
            object.__setattr__(self, "jacobian_condition_number", float(self.jacobian_condition_number))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "model_values": list(self.model_values),
            "market_quotes": list(self.market_quotes),
            "residual_vector": list(self.residual_vector),
            "jacobian_matrix": [list(row) for row in self.jacobian_matrix],
            "max_abs_residual": self.max_abs_residual,
            "l2_norm": self.l2_norm,
            "jacobian_condition_number": self.jacobian_condition_number,
            "jacobian_rank": self.jacobian_rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BootstrapCalibrationResult:
    """Structured bootstrap result for replayable rates calibration workflows."""

    input_bundle: BootstrapCurveInputBundle
    tenors: tuple[float, ...]
    zero_rates: tuple[float, ...]
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    diagnostics: BootstrapCalibrationDiagnostics
    curve: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenors", _normalize_float_tuple(self.tenors))
        object.__setattr__(self, "zero_rates", _normalize_float_tuple(self.zero_rates))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "input_bundle": self.input_bundle.to_payload(),
            "tenors": list(self.tenors),
            "zero_rates": list(self.zero_rates),
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "diagnostics": self.diagnostics.to_payload(),
        }


@dataclass(frozen=True)
class BootstrapQuoteBucket:
    """One quote-space risk bucket derived from a bootstrap input bundle."""

    bucket_id: str
    label: str
    tenor: float
    start_tenor: float
    instrument_type: BootstrapInstrumentType
    quote: float
    rate_like_quote: float
    quote_bump_per_bp: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenor", float(self.tenor))
        object.__setattr__(self, "start_tenor", float(self.start_tenor))
        object.__setattr__(self, "quote", float(self.quote))
        object.__setattr__(self, "rate_like_quote", float(self.rate_like_quote))
        object.__setattr__(self, "quote_bump_per_bp", float(self.quote_bump_per_bp))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "bucket_id": self.bucket_id,
            "label": self.label,
            "tenor": self.tenor,
            "start_tenor": self.start_tenor,
            "instrument_type": self.instrument_type,
            "quote": self.quote,
            "rate_like_quote": self.rate_like_quote,
            "quote_bump_per_bp": self.quote_bump_per_bp,
            "metadata": dict(self.metadata),
        }


def _enum_member(enum_cls, raw_value):
    if isinstance(raw_value, enum_cls):
        return raw_value
    key = str(raw_value or "").strip()
    if not key:
        raise ValueError(f"Cannot resolve empty enum value for {enum_cls.__name__}")
    try:
        return enum_cls[key]
    except KeyError:
        for member in enum_cls:
            if member.value == key:
                return member
        raise ValueError(f"Unknown {enum_cls.__name__} value: {raw_value!r}") from None


def bootstrap_curve_input_bundle_from_payload(payload: Mapping[str, object]) -> BootstrapCurveInputBundle:
    """Reconstruct a typed bootstrap bundle from a serialized payload."""
    payload = dict(payload or {})
    conventions_payload = dict(payload.get("conventions") or {})
    conventions = BootstrapConventionBundle(
        deposit_day_count=_enum_member(
            DayCountConvention,
            conventions_payload.get("deposit_day_count", DayCountConvention.ACT_360.name),
        ),
        deposit_compounding=str(
            conventions_payload.get("deposit_compounding", "simple")
        ).strip()
        or "simple",
        future_contract_frequency=_enum_member(
            Frequency,
            conventions_payload.get(
                "future_contract_frequency",
                Frequency.QUARTERLY.name,
            ),
        ),
        future_day_count=_enum_member(
            DayCountConvention,
            conventions_payload.get("future_day_count", DayCountConvention.ACT_360.name),
        ),
        future_quote_style=str(
            conventions_payload.get("future_quote_style", "price")
        ).strip()
        or "price",
        swap_fixed_frequency=_enum_member(
            Frequency,
            conventions_payload.get(
                "swap_fixed_frequency",
                Frequency.SEMI_ANNUAL.name,
            ),
        ),
        swap_fixed_day_count=_enum_member(
            DayCountConvention,
            conventions_payload.get(
                "swap_fixed_day_count",
                DayCountConvention.THIRTY_360.name,
            ),
        ),
        swap_float_frequency=_enum_member(
            Frequency,
            conventions_payload.get(
                "swap_float_frequency",
                Frequency.QUARTERLY.name,
            ),
        ),
        swap_float_day_count=_enum_member(
            DayCountConvention,
            conventions_payload.get(
                "swap_float_day_count",
                DayCountConvention.ACT_360.name,
            ),
        ),
        metadata=dict(conventions_payload.get("metadata") or {}),
    )
    instruments = tuple(
        BootstrapInstrument(
            tenor=float(inst_payload["tenor"]),
            quote=float(inst_payload["quote"]),
            instrument_type=str(inst_payload.get("instrument_type", "deposit")).strip() or "deposit",
            start_tenor=float(inst_payload.get("start_tenor", 0.0) or 0.0),
            accrual_years=inst_payload.get("accrual_years"),
            label=str(inst_payload.get("label", "") or ""),
            metadata=dict(inst_payload.get("metadata") or {}),
        )
        for inst_payload in payload.get("instruments") or ()
    )
    return BootstrapCurveInputBundle(
        instruments=instruments,
        curve_name=str(payload.get("curve_name", "") or ""),
        currency=str(payload.get("currency", "") or ""),
        rate_index=payload.get("rate_index"),
        conventions=conventions,
        metadata=dict(payload.get("metadata") or {}),
    )


def build_bootstrap_quote_buckets(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
) -> tuple[BootstrapQuoteBucket, ...]:
    """Return one stable quote-bucket surface for a bootstrap bundle."""
    bundle = _ordered_bundle(curve_inputs)
    buckets: list[BootstrapQuoteBucket] = []
    seen_ids: set[str] = set()

    for index, inst in enumerate(bundle.instruments):
        base_id = str(inst.label or f"{inst.instrument_type}_{float(inst.tenor):g}y")
        bucket_id = base_id
        suffix = 2
        while bucket_id in seen_ids:
            bucket_id = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(bucket_id)
        buckets.append(
            BootstrapQuoteBucket(
                bucket_id=bucket_id,
                label=inst.label or bucket_id,
                tenor=inst.tenor,
                start_tenor=inst.start_tenor,
                instrument_type=inst.instrument_type,
                quote=inst.quote,
                rate_like_quote=_rate_like_quote(inst, bundle.conventions),
                quote_bump_per_bp=_quote_bump_per_bp(inst, bundle.conventions),
                metadata={
                    "curve_name": bundle.curve_name,
                    "currency": bundle.currency,
                    "rate_index": bundle.rate_index,
                },
            )
        )
    return tuple(buckets)


def bump_bootstrap_quote_buckets(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    quote_bumps_bps: Mapping[str, float],
) -> BootstrapCurveInputBundle:
    """Return a bundle with the requested quote buckets bumped in rate-bps terms."""
    bundle = _ordered_bundle(curve_inputs)
    buckets = build_bootstrap_quote_buckets(bundle)
    instruments = []
    for bucket, inst in zip(buckets, bundle.instruments):
        bump_bps = float(quote_bumps_bps.get(bucket.bucket_id, 0.0))
        quote_shift = bump_bps * bucket.quote_bump_per_bp
        instruments.append(
            replace(
                inst,
                quote=float(inst.quote) + float(quote_shift),
            )
        )
    return replace(bundle, instruments=tuple(instruments))


def _rate_like_quote(inst: BootstrapInstrument, conventions: BootstrapConventionBundle) -> float:
    """Return a rate-like quote representation suitable for scenario shaping."""
    if inst.instrument_type != "future":
        return float(inst.quote)
    if conventions.future_quote_style == "price":
        return (100.0 - float(inst.quote)) / 100.0
    return float(inst.quote)


def _quote_bump_per_bp(inst: BootstrapInstrument, conventions: BootstrapConventionBundle) -> float:
    """Return quote units corresponding to a +1bp rate-like bump."""
    if inst.instrument_type == "future" and conventions.future_quote_style == "price":
        return -0.01
    return 0.0001


def _coerce_curve_input_bundle(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    *,
    curve_name: str = "",
) -> BootstrapCurveInputBundle:
    """Normalize legacy instrument lists onto the typed bundle surface."""
    if isinstance(curve_inputs, BootstrapCurveInputBundle):
        return curve_inputs.with_curve_name(curve_name)
    return BootstrapCurveInputBundle(
        instruments=tuple(curve_inputs),
        curve_name=curve_name,
    )


def _ordered_bundle(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    *,
    curve_name: str = "",
) -> BootstrapCurveInputBundle:
    """Return one normalized bundle with deterministic instrument order."""
    bundle = _coerce_curve_input_bundle(curve_inputs, curve_name=curve_name)
    return replace(bundle, instruments=bundle.sorted_instruments())


def _ordered_tenors(bundle: BootstrapCurveInputBundle) -> tuple[float, ...]:
    """Return the ordered bootstrap tenor grid."""
    return tuple(float(inst.tenor) for inst in bundle.instruments)


def _ordered_quotes(bundle: BootstrapCurveInputBundle) -> tuple[float, ...]:
    """Return the ordered market quotes."""
    return tuple(float(inst.quote) for inst in bundle.instruments)


def _initial_guess(bundle: BootstrapCurveInputBundle) -> tuple[float, ...]:
    """Return the initial zero-rate guess for the solve path."""
    guesses = []
    for inst in bundle.instruments:
        guesses.append(float(inst.quote) if inst.instrument_type == "deposit" else 0.05)
    return tuple(guesses)


def _instrument_labels(bundle: BootstrapCurveInputBundle) -> tuple[str, ...]:
    """Return deterministic solve labels for the ordered instrument set."""
    labels: list[str] = []
    for index, inst in enumerate(bundle.instruments):
        if inst.label:
            labels.append(inst.label)
            continue
        labels.append(f"{inst.instrument_type}_{index}_{float(inst.tenor):g}")
    return tuple(labels)


def _reprice(
    rates: object,
    tenors: object,
    bundle: BootstrapCurveInputBundle,
) -> object:
    """Reprice all calibration instruments given a zero-rate vector."""
    model_values = []
    conventions = bundle.conventions

    for inst in bundle.instruments:
        t_end = float(inst.tenor)

        if inst.instrument_type == "deposit":
            t_start = float(inst.start_tenor)
            accrual = inst.accrual_years or inst.natural_accrual_years()
            df_start = _discount_factor(t_start, rates, tenors)
            df_end = _discount_factor(t_end, rates, tenors)
            model_values.append((df_start / df_end - 1.0) / accrual)
            continue

        if inst.instrument_type == "future":
            if inst.accrual_years is not None:
                accrual = float(inst.accrual_years)
            elif inst.start_tenor > 0.0:
                accrual = inst.natural_accrual_years()
            else:
                accrual = conventions.future_contract_years
            t_start = float(inst.start_tenor) if inst.start_tenor > 0.0 else max(t_end - accrual, 0.0)
            df_start = _discount_factor(max(t_start, 0.001), rates, tenors)
            df_end = _discount_factor(t_end, rates, tenors)
            fwd = (df_start / df_end - 1.0) / accrual
            if conventions.future_quote_style == "price":
                model_values.append(100.0 - fwd * 100.0)
            else:
                model_values.append(fwd)
            continue

        if inst.instrument_type == "swap":
            t_start = float(inst.start_tenor)
            fixed_step = _frequency_years(conventions.swap_fixed_frequency)
            float_step = _frequency_years(conventions.swap_float_frequency)

            float_pv = np.array(0.0)
            prev_time = t_start
            for pay_time in _schedule_times(t_start, t_end, float_step):
                accrual = pay_time - prev_time
                df_start = _discount_factor(max(prev_time, 0.001), rates, tenors)
                df_end = _discount_factor(pay_time, rates, tenors)
                fwd = (df_start / df_end - 1.0) / accrual
                float_pv = float_pv + fwd * accrual * df_end
                prev_time = pay_time

            annuity = np.array(0.0)
            prev_time = t_start
            for pay_time in _schedule_times(t_start, t_end, fixed_step):
                accrual = pay_time - prev_time
                annuity = annuity + accrual * _discount_factor(pay_time, rates, tenors)
                prev_time = pay_time

            par_rate = float_pv / annuity
            model_values.append(par_rate)
            continue

        raise ValueError(f"Unknown instrument type: {inst.instrument_type!r}")

    return np.array(model_values)


def build_bootstrap_solve_request(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> SolveRequest:
    """Build the explicit solve request for one rates bootstrap lane."""
    bundle = _ordered_bundle(curve_inputs)
    tenors = raw_np.asarray(_ordered_tenors(bundle), dtype=float)
    quotes = raw_np.asarray(_ordered_quotes(bundle), dtype=float)
    initial_guess = _initial_guess(bundle)

    def vector_objective(params):
        return _reprice(params, tenors, bundle)

    def scalar_objective(params):
        residual = vector_objective(params) - quotes
        return np.sum(residual ** 2)

    objective_jacobian = jacobian(vector_objective)

    return SolveRequest(
        request_id="rates_bootstrap_least_squares",
        problem_kind="least_squares",
        parameter_names=tuple(f"zero_rate_{tenor:g}y" for tenor in tenors),
        initial_guess=initial_guess,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=_instrument_labels(bundle),
            target_values=tuple(float(quote) for quote in quotes),
            vector_objective_fn=vector_objective,
            scalar_objective_fn=scalar_objective,
            jacobian_fn=objective_jacobian,
            metadata={
                "curve_name": bundle.curve_name,
                "currency": bundle.currency,
                "rate_index": bundle.rate_index,
                "instrument_count": len(bundle.instruments),
                "derivative_method": "autodiff_vector_jacobian",
            },
        ),
        solver_hint="trf",
        warm_start=WarmStart(parameter_values=initial_guess, source="quote_seed"),
        metadata={
            "problem_family": "rates_bootstrap",
            "curve_name": bundle.curve_name,
            "currency": bundle.currency,
            "rate_index": bundle.rate_index,
            "solver_family": "scipy",
        },
        options={"maxiter": int(max_iter), "ftol": float(tol), "xtol": float(tol), "gtol": float(tol)},
    )


def _bootstrap_jacobian_diagnostics(
    bundle: BootstrapCurveInputBundle,
    tenors: raw_np.ndarray,
    zero_rates: raw_np.ndarray,
) -> tuple[tuple[tuple[float, ...], ...], float | None, int | None]:
    """Return the repricing Jacobian matrix plus basic conditioning stats."""

    def vector_objective(params):
        return _reprice(params, tenors, bundle)

    objective_jacobian = jacobian(vector_objective)
    jacobian_matrix = raw_np.asarray(objective_jacobian(zero_rates), dtype=float)
    condition_number: float | None
    rank: int | None
    try:
        condition_number = float(raw_np.linalg.cond(jacobian_matrix))
    except raw_np.linalg.LinAlgError:
        condition_number = None
    try:
        rank = int(raw_np.linalg.matrix_rank(jacobian_matrix))
    except raw_np.linalg.LinAlgError:
        rank = None
    return _normalize_matrix(jacobian_matrix), condition_number, rank


def _bootstrap_diagnostics(
    bundle: BootstrapCurveInputBundle,
    tenors: raw_np.ndarray,
    zero_rates: raw_np.ndarray,
    solve_result: SolveResult,
) -> BootstrapCalibrationDiagnostics:
    """Return residual and Jacobian diagnostics for one solved bootstrap lane."""
    quotes = raw_np.asarray(_ordered_quotes(bundle), dtype=float)
    model_values = raw_np.asarray(_reprice(zero_rates, tenors, bundle), dtype=float)
    residuals = raw_np.asarray(solve_result.residual_vector, dtype=float)
    jacobian_matrix, condition_number, rank = _bootstrap_jacobian_diagnostics(bundle, tenors, zero_rates)
    return BootstrapCalibrationDiagnostics(
        model_values=tuple(float(value) for value in model_values),
        market_quotes=tuple(float(value) for value in quotes),
        residual_vector=tuple(float(value) for value in residuals),
        jacobian_matrix=jacobian_matrix,
        max_abs_residual=float(raw_np.max(raw_np.abs(residuals))) if residuals.size else 0.0,
        l2_norm=float(raw_np.linalg.norm(residuals)) if residuals.size else 0.0,
        jacobian_condition_number=condition_number,
        jacobian_rank=rank,
        metadata={"instrument_count": len(bundle.instruments)},
    )


def bootstrap_curve_result(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> BootstrapCalibrationResult:
    """Solve one typed rates bootstrap and return full diagnostics."""
    from trellis.curves.yield_curve import YieldCurve

    bundle = _ordered_bundle(curve_inputs)
    tenors = raw_np.asarray(_ordered_tenors(bundle), dtype=float)
    solve_request = build_bootstrap_solve_request(bundle, max_iter=max_iter, tol=tol)
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(
            f"Rates bootstrap failed: {solve_result.metadata.get('message', 'unknown failure')}"
        )

    zero_rates = raw_np.asarray(solve_result.solution, dtype=float)
    diagnostics = _bootstrap_diagnostics(bundle, tenors, zero_rates, solve_result)
    solver_provenance = build_solve_provenance(solve_request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(solve_request, solve_result)
    curve = YieldCurve(tenors, zero_rates)
    return BootstrapCalibrationResult(
        input_bundle=bundle,
        tenors=tuple(float(value) for value in tenors),
        zero_rates=tuple(float(value) for value in zero_rates),
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        diagnostics=diagnostics,
        curve=curve,
    )


def bootstrap_named_curve_results(
    curve_sets: Mapping[str, Sequence[BootstrapInstrument] | BootstrapCurveInputBundle],
    **kwargs,
) -> dict[str, BootstrapCalibrationResult]:
    """Solve multiple named bootstraps and return structured results."""
    return {
        name: bootstrap_curve_result(_ordered_bundle(curve_inputs, curve_name=name), **kwargs)
        for name, curve_inputs in curve_sets.items()
    }


def bootstrap(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[object, object]:
    """Bootstrap zero rates from market instruments through the typed solve lane."""
    result = bootstrap_curve_result(curve_inputs, max_iter=max_iter, tol=tol)
    return np.array(result.tenors), np.array(result.zero_rates)


def bootstrap_yield_curve(
    curve_inputs: Sequence[BootstrapInstrument] | BootstrapCurveInputBundle,
    **kwargs,
):
    """Bootstrap and return a YieldCurve."""
    return bootstrap_curve_result(curve_inputs, **kwargs).curve


def bootstrap_named_yield_curves(
    curve_sets: Mapping[str, Sequence[BootstrapInstrument] | BootstrapCurveInputBundle],
    **kwargs,
) -> dict[str, object]:
    """Bootstrap multiple named yield curves from typed input bundles."""
    results = bootstrap_named_curve_results(curve_sets, **kwargs)
    return {name: result.curve for name, result in results.items()}


def _ordered_dated_bundle(bundle: DatedBootstrapCurveInputBundle) -> DatedBootstrapCurveInputBundle:
    """Return one dated bundle with deterministic instrument order."""
    return replace(bundle, instruments=bundle.sorted_instruments())


def _dated_years(
    settlement_date: date,
    target_date: date,
    day_count: DayCountConvention,
) -> float:
    """Return model time in years from settlement to a target date."""
    return float(year_fraction(settlement_date, target_date, day_count))


def _dated_bundle_tenors(
    bundle: DatedBootstrapCurveInputBundle,
    settlement_date: date,
) -> tuple[float, ...]:
    """Return the dated bootstrap knot grid implied by instrument end dates."""
    default_day_count = bundle.conventions.future_day_count
    return tuple(
        _dated_years(
            settlement_date,
            inst.end_date,
            inst.day_count or default_day_count,
        )
        for inst in bundle.instruments
    )


def _dated_initial_guess(bundle: DatedBootstrapCurveInputBundle) -> tuple[float, ...]:
    """Return the initial zero-rate guess for a dated bootstrap solve."""
    guesses = []
    for inst in bundle.instruments:
        if inst.instrument_type == "deposit":
            guesses.append(float(inst.quote))
        elif inst.instrument_type == "future":
            if bundle.conventions.future_quote_style == "price":
                guesses.append((100.0 - float(inst.quote)) / 100.0)
            else:
                guesses.append(float(inst.quote))
        else:
            guesses.append(0.05)
    return tuple(guesses)


def _dated_instrument_labels(bundle: DatedBootstrapCurveInputBundle) -> tuple[str, ...]:
    """Return deterministic solve labels for the dated instrument set."""
    labels: list[str] = []
    for index, inst in enumerate(bundle.instruments):
        if inst.label:
            labels.append(inst.label)
            continue
        labels.append(f"{inst.instrument_type}_{index}_{inst.end_date.isoformat()}")
    return tuple(labels)


def _discount_value(
    t: float,
    rates: object,
    tenors: object,
    *,
    external_discount_curve=None,
) -> object:
    """Return a discount factor either from the solved curve or an external dependency."""
    if external_discount_curve is not None:
        return external_discount_curve.discount(float(max(t, 0.0)))
    return _discount_factor(float(max(t, 0.0)), rates, tenors)


def _reprice_dated(
    rates: object,
    tenors: object,
    bundle: DatedBootstrapCurveInputBundle,
    *,
    settlement_date: date,
    external_discount_curve=None,
) -> object:
    """Reprice all dated calibration instruments given a zero-rate vector."""
    model_values = []
    conventions = bundle.conventions

    for inst in bundle.instruments:
        instrument_day_count = inst.day_count or conventions.future_day_count

        if inst.instrument_type == "deposit":
            start_years = _dated_years(settlement_date, inst.start_date, instrument_day_count)
            end_years = _dated_years(settlement_date, inst.end_date, instrument_day_count)
            accrual = float(year_fraction(inst.start_date, inst.end_date, instrument_day_count))
            df_start = _discount_factor(start_years, rates, tenors)
            df_end = _discount_factor(end_years, rates, tenors)
            model_values.append((df_start / df_end - 1.0) / accrual)
            continue

        if inst.instrument_type == "future":
            start_years = _dated_years(settlement_date, inst.start_date, instrument_day_count)
            end_years = _dated_years(settlement_date, inst.end_date, instrument_day_count)
            accrual = float(year_fraction(inst.start_date, inst.end_date, instrument_day_count))
            df_start = _discount_factor(start_years, rates, tenors)
            df_end = _discount_factor(end_years, rates, tenors)
            fwd = (df_start / df_end - 1.0) / accrual
            if conventions.future_quote_style == "price":
                model_values.append(100.0 - fwd * 100.0)
            else:
                model_values.append(fwd)
            continue

        if inst.instrument_type == "swap":
            fixed_frequency = inst.fixed_frequency or conventions.swap_fixed_frequency
            fixed_day_count = inst.fixed_day_count or conventions.swap_fixed_day_count
            float_frequency = inst.float_frequency or conventions.swap_float_frequency
            float_day_count = inst.float_day_count or conventions.swap_float_day_count

            fixed_schedule = build_period_schedule(
                inst.start_date,
                inst.end_date,
                fixed_frequency,
                calendar=inst.calendar,
                bda=inst.business_day_adjustment,
                stub=inst.stub_type,
                roll_convention=inst.roll_convention,
                day_count=fixed_day_count,
                time_origin=settlement_date,
            )
            float_schedule = build_period_schedule(
                inst.start_date,
                inst.end_date,
                float_frequency,
                calendar=inst.calendar,
                bda=inst.business_day_adjustment,
                stub=inst.stub_type,
                roll_convention=inst.roll_convention,
                day_count=float_day_count,
                time_origin=settlement_date,
            )

            float_pv = np.array(0.0)
            for period in float_schedule.periods:
                accrual = float(period.accrual_fraction or 0.0)
                start_years = max(float(period.t_start or 0.0), 0.0)
                end_years = float(period.t_end or 0.0)
                df_start = _discount_factor(start_years, rates, tenors)
                df_end = _discount_factor(end_years, rates, tenors)
                fwd = (df_start / df_end - 1.0) / accrual
                discount = _discount_value(
                    float(period.t_payment or 0.0),
                    rates,
                    tenors,
                    external_discount_curve=external_discount_curve,
                )
                float_pv = float_pv + fwd * accrual * discount

            annuity = np.array(0.0)
            for period in fixed_schedule.periods:
                accrual = float(period.accrual_fraction or 0.0)
                discount = _discount_value(
                    float(period.t_payment or 0.0),
                    rates,
                    tenors,
                    external_discount_curve=external_discount_curve,
                )
                annuity = annuity + accrual * discount

            model_values.append(float_pv / annuity)
            continue

        raise ValueError(f"Unknown instrument type: {inst.instrument_type!r}")

    return np.array(model_values)


def build_dated_bootstrap_solve_request(
    bundle: DatedBootstrapCurveInputBundle,
    *,
    settlement_date: date,
    external_discount_curve=None,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> SolveRequest:
    """Build the explicit solve request for one dated rates bootstrap lane."""
    bundle = _ordered_dated_bundle(bundle)
    tenors = raw_np.asarray(_dated_bundle_tenors(bundle, settlement_date), dtype=float)
    quotes = raw_np.asarray(tuple(float(inst.quote) for inst in bundle.instruments), dtype=float)
    initial_guess = _dated_initial_guess(bundle)

    def vector_objective(params):
        return _reprice_dated(
            params,
            tenors,
            bundle,
            settlement_date=settlement_date,
            external_discount_curve=external_discount_curve,
        )

    def scalar_objective(params):
        residual = vector_objective(params) - quotes
        return np.sum(residual ** 2)

    objective_jacobian = jacobian(vector_objective)
    metadata = {
        "curve_name": bundle.curve_name,
        "currency": bundle.currency,
        "rate_index": bundle.rate_index,
        "curve_role": bundle.curve_role,
        "dependency_names": dict(bundle.dependency_names),
        "instrument_count": len(bundle.instruments),
        "settlement_date": settlement_date.isoformat(),
    }
    if external_discount_curve is not None:
        metadata["external_discount_curve"] = "provided"

    return SolveRequest(
        request_id="rates_dated_bootstrap_least_squares",
        problem_kind="least_squares",
        parameter_names=tuple(f"zero_rate_{tenor:g}y" for tenor in tenors),
        initial_guess=initial_guess,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=_dated_instrument_labels(bundle),
            target_values=tuple(float(quote) for quote in quotes),
            vector_objective_fn=vector_objective,
            scalar_objective_fn=scalar_objective,
            jacobian_fn=objective_jacobian,
            metadata=metadata,
        ),
        solver_hint="trf",
        warm_start=WarmStart(parameter_values=initial_guess, source="dated_quote_seed"),
        metadata={**metadata, "problem_family": "rates_dated_bootstrap", "solver_family": "scipy"},
        options={"maxiter": int(max_iter), "ftol": float(tol), "xtol": float(tol), "gtol": float(tol)},
    )


def _dated_bootstrap_jacobian_diagnostics(
    bundle: DatedBootstrapCurveInputBundle,
    tenors: raw_np.ndarray,
    zero_rates: raw_np.ndarray,
    *,
    settlement_date: date,
    external_discount_curve=None,
) -> tuple[tuple[tuple[float, ...], ...], float | None, int | None]:
    """Return the repricing Jacobian matrix plus basic conditioning stats."""

    def vector_objective(params):
        return _reprice_dated(
            params,
            tenors,
            bundle,
            settlement_date=settlement_date,
            external_discount_curve=external_discount_curve,
        )

    objective_jacobian = jacobian(vector_objective)
    jacobian_matrix = raw_np.asarray(objective_jacobian(zero_rates), dtype=float)
    condition_number: float | None
    rank: int | None
    try:
        condition_number = float(raw_np.linalg.cond(jacobian_matrix))
    except raw_np.linalg.LinAlgError:
        condition_number = None
    try:
        rank = int(raw_np.linalg.matrix_rank(jacobian_matrix))
    except raw_np.linalg.LinAlgError:
        rank = None
    return _normalize_matrix(jacobian_matrix), condition_number, rank


def _dated_bootstrap_diagnostics(
    bundle: DatedBootstrapCurveInputBundle,
    tenors: raw_np.ndarray,
    zero_rates: raw_np.ndarray,
    solve_result: SolveResult,
    *,
    settlement_date: date,
    external_discount_curve=None,
) -> BootstrapCalibrationDiagnostics:
    """Return residual and Jacobian diagnostics for one solved dated bootstrap lane."""
    quotes = raw_np.asarray(tuple(float(inst.quote) for inst in bundle.instruments), dtype=float)
    model_values = raw_np.asarray(
        _reprice_dated(
            zero_rates,
            tenors,
            bundle,
            settlement_date=settlement_date,
            external_discount_curve=external_discount_curve,
        ),
        dtype=float,
    )
    residuals = raw_np.asarray(solve_result.residual_vector, dtype=float)
    jacobian_matrix, condition_number, rank = _dated_bootstrap_jacobian_diagnostics(
        bundle,
        tenors,
        zero_rates,
        settlement_date=settlement_date,
        external_discount_curve=external_discount_curve,
    )
    return BootstrapCalibrationDiagnostics(
        model_values=tuple(float(value) for value in model_values),
        market_quotes=tuple(float(value) for value in quotes),
        residual_vector=tuple(float(value) for value in residuals),
        jacobian_matrix=jacobian_matrix,
        max_abs_residual=float(raw_np.max(raw_np.abs(residuals))) if residuals.size else 0.0,
        l2_norm=float(raw_np.linalg.norm(residuals)) if residuals.size else 0.0,
        jacobian_condition_number=condition_number,
        jacobian_rank=rank,
        metadata={
            "instrument_count": len(bundle.instruments),
            "curve_role": bundle.curve_role,
            "dependency_names": dict(bundle.dependency_names),
        },
    )


def bootstrap_dated_curve_result(
    bundle: DatedBootstrapCurveInputBundle,
    *,
    settlement_date: date,
    external_discount_curve=None,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> BootstrapCalibrationResult:
    """Solve one typed dated rates bootstrap and return full diagnostics."""
    from trellis.curves.yield_curve import YieldCurve

    bundle = _ordered_dated_bundle(bundle)
    tenors = raw_np.asarray(_dated_bundle_tenors(bundle, settlement_date), dtype=float)
    solve_request = build_dated_bootstrap_solve_request(
        bundle,
        settlement_date=settlement_date,
        external_discount_curve=external_discount_curve,
        max_iter=max_iter,
        tol=tol,
    )
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(
            f"Rates dated bootstrap failed: {solve_result.metadata.get('message', 'unknown failure')}"
        )

    zero_rates = raw_np.asarray(solve_result.solution, dtype=float)
    diagnostics = _dated_bootstrap_diagnostics(
        bundle,
        tenors,
        zero_rates,
        solve_result,
        settlement_date=settlement_date,
        external_discount_curve=external_discount_curve,
    )
    solver_provenance = build_solve_provenance(solve_request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(solve_request, solve_result)
    curve = YieldCurve(tenors, zero_rates)
    return BootstrapCalibrationResult(
        input_bundle=bundle,  # type: ignore[arg-type]
        tenors=tuple(float(value) for value in tenors),
        zero_rates=tuple(float(value) for value in zero_rates),
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        diagnostics=diagnostics,
        curve=curve,
    )


def bootstrap_multi_curve_program(
    program: MultiCurveBootstrapProgram,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> MultiCurveBootstrapResult:
    """Solve a dependency-aware multi-curve bootstrap program."""
    pending = {
        bundle.curve_name: _ordered_dated_bundle(bundle)
        for bundle in program.curve_inputs
    }
    node_results: dict[str, BootstrapCalibrationResult] = {}
    dependency_graph: dict[str, dict[str, str]] = {}
    dependency_order: list[str] = []

    while pending:
        progress = False
        for curve_name, bundle in tuple(pending.items()):
            dependencies = dict(bundle.dependency_names)
            unresolved = [
                dependency_name
                for dependency_name in dependencies.values()
                if dependency_name not in node_results
            ]
            unknown = [dependency_name for dependency_name in unresolved if dependency_name not in pending]
            if unknown:
                raise ValueError(
                    f"Unknown multi-curve dependency for {curve_name!r}: {sorted(unknown)}"
                )
            if unresolved:
                continue

            external_discount_curve = None
            discount_dependency_name = dependencies.get("discount_curve")
            if discount_dependency_name is not None:
                external_discount_curve = node_results[discount_dependency_name].curve

            node_results[curve_name] = bootstrap_dated_curve_result(
                bundle,
                settlement_date=program.settlement_date,
                external_discount_curve=external_discount_curve,
                max_iter=max_iter,
                tol=tol,
            )
            dependency_graph[curve_name] = dependencies
            dependency_order.append(curve_name)
            del pending[curve_name]
            progress = True
        if not progress:
            raise ValueError("Multi-curve bootstrap dependency graph contains a cycle")

    return MultiCurveBootstrapResult(
        program=program,
        node_results=node_results,
        dependency_order=tuple(dependency_order),
        dependency_graph=dependency_graph,
        metadata={"curve_count": len(node_results)},
    )


__all__ = [
    "BootstrapCalibrationDiagnostics",
    "BootstrapCalibrationResult",
    "BootstrapConventionBundle",
    "BootstrapCurveInputBundle",
    "BootstrapInstrument",
    "BootstrapCurveRole",
    "BootstrapQuoteBucket",
    "DatedBootstrapCurveInputBundle",
    "DatedBootstrapInstrument",
    "MultiCurveBootstrapProgram",
    "MultiCurveBootstrapResult",
    "bootstrap",
    "bootstrap_dated_curve_result",
    "bootstrap_curve_input_bundle_from_payload",
    "bootstrap_curve_result",
    "bootstrap_multi_curve_program",
    "bootstrap_named_curve_results",
    "bootstrap_named_yield_curves",
    "bootstrap_yield_curve",
    "build_bootstrap_quote_buckets",
    "build_dated_bootstrap_solve_request",
    "bump_bootstrap_quote_buckets",
    "build_bootstrap_solve_request",
]
