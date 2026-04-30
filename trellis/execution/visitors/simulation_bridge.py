"""Monte Carlo visitors for route-free execution IR artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.execution.admission import admit_execution_capabilities
from trellis.execution.ir import ContractExecutionIR
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.event_state import event_step_indices
from trellis.models.monte_carlo.lsm import longstaff_schwartz_multistate_result
from trellis.models.processes.correlated_gbm import CorrelatedGBM


@dataclass(frozen=True)
class BermudanBestOfBasketMCInputs:
    """Explicit market inputs consumed by the Bermudan basket MC visitor."""

    valuation_date: date
    spot_values: Mapping[str, float] | tuple[tuple[str, float], ...]
    volatilities: Mapping[str, float] | tuple[tuple[str, float], ...]
    correlation_matrix: Sequence[Sequence[float]]
    risk_free_rate: float
    carry_rates: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    day_count: DayCountConvention = DayCountConvention.ACT_365

    def __post_init__(self) -> None:
        if not isinstance(self.valuation_date, date):
            raise TypeError("valuation_date must be a datetime.date")
        object.__setattr__(self, "spot_values", _float_mapping(self.spot_values))
        object.__setattr__(self, "volatilities", _float_mapping(self.volatilities))
        object.__setattr__(self, "carry_rates", _float_mapping(self.carry_rates))
        object.__setattr__(
            self,
            "correlation_matrix",
            _float_matrix(self.correlation_matrix),
        )
        object.__setattr__(self, "risk_free_rate", float(self.risk_free_rate))


@dataclass(frozen=True)
class BermudanBestOfBasketMCControls:
    """Simulation controls for the route-free Bermudan basket visitor."""

    n_paths: int = 50_000
    n_steps: int = 252
    seed: int | None = 42
    method: str = "exact"

    def __post_init__(self) -> None:
        n_paths = int(self.n_paths)
        n_steps = int(self.n_steps)
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        method = str(self.method or "exact").strip().lower()
        if method not in {"exact", "euler"}:
            raise ValueError("method must be 'exact' or 'euler'")
        object.__setattr__(self, "n_paths", n_paths)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class BermudanBestOfBasketMCResult:
    """Audited MC visitor result for a Bermudan best-of basket execution IR."""

    price: float
    lower_bound: float
    currency: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "currency", str(self.currency or "").strip().upper())
        object.__setattr__(self, "diagnostics", _mapping_proxy(self.diagnostics))
        object.__setattr__(self, "provenance", _mapping_proxy(self.provenance))

    def with_provenance(
        self,
        values: Mapping[str, object],
    ) -> "BermudanBestOfBasketMCResult":
        """Return a copy with additional provenance fields."""
        provenance = dict(self.provenance)
        provenance.update(dict(values))
        return replace(self, provenance=provenance)


def price_bermudan_best_of_basket_monte_carlo(
    ir: ContractExecutionIR,
    inputs: BermudanBestOfBasketMCInputs,
    *,
    controls: BermudanBestOfBasketMCControls | None = None,
) -> BermudanBestOfBasketMCResult:
    """Price a Bermudan best-of basket by visiting route-free execution IR."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    controls = controls or BermudanBestOfBasketMCControls()
    admission = admit_execution_capabilities(ir, method="monte_carlo")
    if not admission.admitted:
        blocker_ids = tuple(blocker.blocker_id for blocker in admission.blockers)
        raise ValueError(
            "execution IR is not admitted for Monte Carlo; "
            f"blockers={blocker_ids!r}"
        )

    source_metadata = _metadata_dict(ir.source_track.source_metadata)
    underliers = _ordered_underliers(ir)
    strike_metadata = source_metadata.get("strike")
    if strike_metadata in {None, ""}:
        strike_metadata = _strike_from_settlement(ir)
    strike = float(strike_metadata)
    notional = float(source_metadata.get("notional", 1.0))
    currency = str(source_metadata.get("currency", _currency_from_settlement(ir)) or "USD")
    expiry_date = _expiry_date(ir)
    exercise_dates = _exercise_dates(ir) or (expiry_date,)
    observation_dates = _observation_dates(ir)

    spot_vector = _aligned_vector(inputs.spot_values, underliers, "spot_values")
    vol_vector = _aligned_vector(inputs.volatilities, underliers, "volatilities")
    carry_vector = tuple(
        float(inputs.carry_rates.get(underlier, 0.0))
        for underlier in underliers
    )
    correlation_matrix = _validated_correlation_matrix(
        inputs.correlation_matrix,
        len(underliers),
    )
    maturity = year_fraction(inputs.valuation_date, expiry_date, inputs.day_count)
    if maturity <= 0.0:
        intrinsic = notional * max(max(spot_vector) - strike, 0.0)
        return BermudanBestOfBasketMCResult(
            price=intrinsic,
            lower_bound=intrinsic,
            currency=currency,
            diagnostics={
                "policy_class": "intrinsic",
                "exercise_dates_count": 0,
                "exercised_paths_fraction": 0.0,
                "regression_failures": 0,
            },
            provenance=_provenance(
                ir,
                controls,
                underliers=underliers,
                observation_dates=observation_dates,
                exercise_dates=exercise_dates,
                expiry_date=expiry_date,
                maturity=maturity,
                admission=admission,
                policy_class="intrinsic",
            ),
        )

    process = CorrelatedGBM(
        mu=[inputs.risk_free_rate for _ in underliers],
        sigma=vol_vector,
        corr=correlation_matrix,
        dividend_yield=carry_vector,
    )
    engine = MonteCarloEngine(
        process,
        n_paths=controls.n_paths,
        n_steps=controls.n_steps,
        seed=controls.seed,
        method=controls.method,
    )
    paths = engine.simulate(spot_vector, maturity)
    exercise_times = tuple(
        year_fraction(inputs.valuation_date, exercise_date, inputs.day_count)
        for exercise_date in exercise_dates
        if exercise_date > inputs.valuation_date
    )
    exercise_steps = event_step_indices(exercise_times, maturity, controls.n_steps)

    def payoff(state):
        values = raw_np.asarray(state, dtype=float)
        if values.ndim == 1:
            best = values
        else:
            best = raw_np.max(values, axis=1)
        return raw_np.maximum(best - strike, 0.0)

    policy_result = longstaff_schwartz_multistate_result(
        paths,
        list(exercise_steps),
        payoff,
        inputs.risk_free_rate,
        maturity / controls.n_steps,
    )
    policy_diagnostics = _policy_diagnostics(policy_result.diagnostics)
    price = notional * float(policy_result.price_lower)
    policy_class = str(policy_diagnostics.get("policy_class") or policy_result.policy_class)
    return BermudanBestOfBasketMCResult(
        price=price,
        lower_bound=price,
        currency=currency,
        diagnostics=policy_diagnostics,
        provenance=_provenance(
            ir,
            controls,
            underliers=underliers,
            observation_dates=observation_dates,
            exercise_dates=exercise_dates,
            expiry_date=expiry_date,
            maturity=maturity,
            admission=admission,
            policy_class=policy_class,
        ),
    )


def _float_mapping(values: object) -> Mapping[str, float]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values or ()
    normalized = {
        str(key).strip(): float(value)
        for key, value in items
        if str(key).strip()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _float_matrix(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(cell) for cell in row) for row in values)


def _mapping_proxy(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(sorted(dict(values or {}).items())))


def _metadata_dict(items: object) -> dict[str, object]:
    try:
        return dict(items or ())
    except (TypeError, ValueError):
        return {}


def _ordered_underliers(ir: ContractExecutionIR) -> tuple[str, ...]:
    candidates: list[tuple[int, str]] = []
    for observable in ir.observables:
        if observable.observable_kind != "spot":
            continue
        metadata = _metadata_dict(observable.metadata)
        name = str(
            metadata.get("underlier")
            or _suffix(observable.source_ref)
            or _suffix(observable.observable_id)
        ).strip()
        if not name:
            continue
        index = int(metadata.get("vector_index", len(candidates)))
        candidates.append((index, name))
    underliers = tuple(name for _, name in sorted(candidates))
    if len(underliers) < 2:
        raise ValueError(
            "Bermudan best-of basket MC visitor requires at least two spot underliers"
        )
    if len(set(underliers)) != len(underliers):
        raise ValueError("Bermudan best-of basket MC visitor requires unique underlier names")
    return underliers


def _suffix(value: object) -> str:
    text = str(value or "").strip()
    return text.rsplit(":", 1)[-1] if text else ""


def _aligned_vector(
    values: Mapping[str, float],
    underliers: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    missing = tuple(underlier for underlier in underliers if underlier not in values)
    if missing:
        raise ValueError(f"{label} missing values for {missing!r}")
    return tuple(float(values[underlier]) for underlier in underliers)


def _validated_correlation_matrix(
    matrix: tuple[tuple[float, ...], ...],
    n_assets: int,
) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != n_assets or any(len(row) != n_assets for row in matrix):
        raise ValueError(
            f"correlation_matrix must have shape ({n_assets}, {n_assets})"
        )
    arr = raw_np.asarray(matrix, dtype=float)
    if not raw_np.allclose(arr, arr.T, atol=1e-12, rtol=0.0):
        raise ValueError("correlation_matrix must be symmetric")
    if not raw_np.allclose(raw_np.diag(arr), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("correlation_matrix diagonal entries must equal 1")
    return matrix


def _coerce_date(value: object, label: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return date.fromisoformat(text)


def _event_dates(ir: ContractExecutionIR, event_kind: str) -> tuple[date, ...]:
    return tuple(
        _coerce_date(event.event_date, event_kind)
        for event in ir.event_plan.events
        if event.event_kind == event_kind
    )


def _expiry_date(ir: ContractExecutionIR) -> date:
    settlement_dates = _event_dates(ir, "settlement")
    if settlement_dates:
        return settlement_dates[-1]
    raise ValueError("Bermudan best-of basket MC visitor requires a settlement event")


def _exercise_dates(ir: ContractExecutionIR) -> tuple[date, ...]:
    return _event_dates(ir, "decision")


def _observation_dates(ir: ContractExecutionIR) -> tuple[date, ...]:
    return _event_dates(ir, "observation")


def _currency_from_settlement(ir: ContractExecutionIR) -> str:
    for step in ir.settlement_program.steps:
        if step.currency:
            return step.currency
    return "USD"


def _strike_from_settlement(ir: ContractExecutionIR) -> float:
    for step in ir.settlement_program.steps:
        text = step.expression
        if " - " in text:
            try:
                return float(text.split(" - ", 1)[1].split(",", 1)[0])
            except (ValueError, IndexError):
                continue
    raise ValueError("Bermudan best-of basket MC visitor requires strike metadata")


def _policy_diagnostics(diagnostics: object) -> Mapping[str, object]:
    if diagnostics is None:
        return {}
    return {
        "policy_class": getattr(diagnostics, "policy_class", ""),
        "exercise_dates_count": int(getattr(diagnostics, "exercise_dates_count", 0)),
        "exercised_paths_fraction": float(
            getattr(diagnostics, "exercised_paths_fraction", 0.0)
        ),
        "regression_failures": int(getattr(diagnostics, "regression_failures", 0)),
        "estimator_name": getattr(diagnostics, "estimator_name", None),
    }


def _provenance(
    ir: ContractExecutionIR,
    controls: BermudanBestOfBasketMCControls,
    *,
    underliers: tuple[str, ...],
    observation_dates: tuple[date, ...],
    exercise_dates: tuple[date, ...],
    expiry_date: date,
    maturity: float,
    admission: object,
    policy_class: str,
) -> Mapping[str, object]:
    return {
        "source_semantic_id": ir.source_track.semantic_id,
        "source_ref": ir.source_track.source_ref,
        "product_family": ir.source_track.product_family,
        "method": "monte_carlo",
        "engine_family": "monte_carlo",
        "process_family": "correlated_gbm",
        "policy_class": policy_class,
        "pricing_authority": "execution_ir_visitor",
        "underliers": underliers,
        "observation_dates": observation_dates,
        "exercise_dates": exercise_dates,
        "expiry_date": expiry_date,
        "maturity": float(maturity),
        "n_paths": controls.n_paths,
        "n_steps": controls.n_steps,
        "seed": controls.seed,
        "simulation_method": controls.method,
        "route_ids": (),
        "required_capabilities": tuple(getattr(admission, "required_capabilities", ())),
        "matched_capabilities": tuple(getattr(admission, "matched_capabilities", ())),
    }


__all__ = [
    "BermudanBestOfBasketMCControls",
    "BermudanBestOfBasketMCInputs",
    "BermudanBestOfBasketMCResult",
    "price_bermudan_best_of_basket_monte_carlo",
]
