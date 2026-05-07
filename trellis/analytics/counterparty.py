"""Institutional counterparty semantics and exposure workflow helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import MappingProxyType
from typing import Any

from trellis.book import FutureValueCube
from trellis.core.differentiable import get_numpy

np = get_numpy()


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    """Return stable, unique, non-empty string tokens."""
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _normalize_token(value: str | None, *, fallback: str = "") -> str:
    """Normalize one semantic token without inventing content."""
    text = str(value or "").strip()
    return text or fallback


def _non_negative_float(value: float | int, *, field_name: str) -> float:
    """Normalize and validate a non-negative numeric field."""
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def _non_negative_int(value: int, *, field_name: str) -> int:
    """Normalize and validate a non-negative integer field."""
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


@dataclass(frozen=True)
class CollateralAgreement:
    """Operational collateral agreement semantics used by counterparty workflows."""

    agreement_id: str
    collateral_currency: str = ""
    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    independent_amount: float = 0.0
    margin_period_of_risk_days: int = 0
    call_frequency_days: int = 1
    valuation_lag_days: int = 0
    remuneration_rate: float = 0.0
    eligible_collateral: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agreement_id", _normalize_token(self.agreement_id))
        if not self.agreement_id:
            raise ValueError("agreement_id is required")
        object.__setattr__(self, "collateral_currency", _normalize_token(self.collateral_currency))
        object.__setattr__(
            self,
            "threshold",
            _non_negative_float(self.threshold, field_name="threshold"),
        )
        object.__setattr__(
            self,
            "minimum_transfer_amount",
            _non_negative_float(
                self.minimum_transfer_amount,
                field_name="minimum_transfer_amount",
            ),
        )
        object.__setattr__(
            self,
            "independent_amount",
            _non_negative_float(self.independent_amount, field_name="independent_amount"),
        )
        object.__setattr__(
            self,
            "margin_period_of_risk_days",
            _non_negative_int(
                self.margin_period_of_risk_days,
                field_name="margin_period_of_risk_days",
            ),
        )
        object.__setattr__(
            self,
            "call_frequency_days",
            max(_non_negative_int(self.call_frequency_days, field_name="call_frequency_days"), 1),
        )
        object.__setattr__(
            self,
            "valuation_lag_days",
            _non_negative_int(self.valuation_lag_days, field_name="valuation_lag_days"),
        )
        object.__setattr__(self, "remuneration_rate", float(self.remuneration_rate))
        object.__setattr__(self, "eligible_collateral", _string_tuple(self.eligible_collateral))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly semantic payload."""
        return {
            "agreement_id": self.agreement_id,
            "collateral_currency": self.collateral_currency,
            "threshold": self.threshold,
            "minimum_transfer_amount": self.minimum_transfer_amount,
            "independent_amount": self.independent_amount,
            "margin_period_of_risk_days": self.margin_period_of_risk_days,
            "call_frequency_days": self.call_frequency_days,
            "valuation_lag_days": self.valuation_lag_days,
            "remuneration_rate": self.remuneration_rate,
            "eligible_collateral": list(self.eligible_collateral),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NettingSet:
    """Position membership and closeout semantics for one counterparty netting set."""

    netting_set_id: str
    counterparty_id: str
    position_names: tuple[str, ...] = ()
    collateral_agreement_id: str = ""
    exposure_currency: str = ""
    closeout_convention: str = "replacement_cost"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "netting_set_id", _normalize_token(self.netting_set_id))
        object.__setattr__(self, "counterparty_id", _normalize_token(self.counterparty_id))
        if not self.netting_set_id:
            raise ValueError("netting_set_id is required")
        if not self.counterparty_id:
            raise ValueError("counterparty_id is required")
        object.__setattr__(self, "position_names", _string_tuple(self.position_names))
        object.__setattr__(
            self,
            "collateral_agreement_id",
            _normalize_token(self.collateral_agreement_id),
        )
        object.__setattr__(self, "exposure_currency", _normalize_token(self.exposure_currency))
        object.__setattr__(
            self,
            "closeout_convention",
            _normalize_token(self.closeout_convention, fallback="replacement_cost"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly semantic payload."""
        return {
            "netting_set_id": self.netting_set_id,
            "counterparty_id": self.counterparty_id,
            "position_names": list(self.position_names),
            "collateral_agreement_id": self.collateral_agreement_id,
            "exposure_currency": self.exposure_currency,
            "closeout_convention": self.closeout_convention,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CounterpartySemanticValidationReport:
    """Validation result for collateral and netting-set semantic contracts."""

    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether the contract is complete enough for downstream workflows."""
        return not self.missing_fields

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly validation payload."""
        return {
            "ok": self.ok,
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CounterpartySemanticContract:
    """Collateral and netting-set semantics for institutional valuation workflows."""

    contract_id: str
    netting_sets: tuple[NettingSet, ...]
    collateral_agreements: tuple[CollateralAgreement, ...] = ()
    covered_position_names: tuple[str, ...] = ()
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract_id = _normalize_token(self.contract_id)
        if not contract_id:
            raise ValueError("contract_id is required")
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "netting_sets", tuple(self.netting_sets or ()))
        object.__setattr__(self, "collateral_agreements", tuple(self.collateral_agreements or ()))
        object.__setattr__(self, "covered_position_names", _string_tuple(self.covered_position_names))
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def collateral_agreement_map(self) -> dict[str, CollateralAgreement]:
        """Return collateral agreements keyed by id."""
        return {
            agreement.agreement_id: agreement
            for agreement in self.collateral_agreements
        }

    @property
    def runtime_semantics(self) -> dict[str, object]:
        """Return the bounded runtime semantics consumed by later workflows."""
        conventions = tuple(
            dict.fromkeys(netting_set.closeout_convention for netting_set in self.netting_sets)
        )
        return {
            "semantic_id": "counterparty_collateral_netting",
            "netting_level": "netting_set",
            "collateral_level": "agreement",
            "closeout_convention": conventions[0] if len(conventions) == 1 else "mixed",
            "position_axis": "position_names",
            "exposure_axis": "observation_date_path",
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly semantic payload."""
        return {
            "contract_id": self.contract_id,
            "description": self.description,
            "covered_position_names": list(self.covered_position_names),
            "netting_sets": [netting_set.to_dict() for netting_set in self.netting_sets],
            "collateral_agreements": [
                agreement.to_dict()
                for agreement in self.collateral_agreements
            ],
            "runtime_semantics": deepcopy(self.runtime_semantics),
            "validation": validate_counterparty_semantic_contract(self).to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CollateralStateProjection:
    """Pathwise projected collateral state for one netting set."""

    netting_set_id: str
    agreement_id: str
    observation_times: tuple[float, ...]
    observation_dates: tuple[date, ...]
    netted_values: Any
    closeout_values: Any
    collateral_balance: Any
    collateralized_exposure: Any
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "netting_set_id", _normalize_token(self.netting_set_id))
        object.__setattr__(self, "agreement_id", _normalize_token(self.agreement_id))
        object.__setattr__(
            self,
            "observation_times",
            tuple(float(value) for value in self.observation_times),
        )
        object.__setattr__(self, "observation_dates", tuple(self.observation_dates or ()))
        netted = _coerce_path_matrix(self.netted_values, field_name="netted_values")
        closeout = _coerce_path_matrix(self.closeout_values, field_name="closeout_values")
        collateral = _coerce_path_matrix(
            self.collateral_balance,
            field_name="collateral_balance",
        )
        collateralized = _coerce_path_matrix(
            self.collateralized_exposure,
            field_name="collateralized_exposure",
        )
        expected_shape = netted.shape
        for name, matrix in (
            ("closeout_values", closeout),
            ("collateral_balance", collateral),
            ("collateralized_exposure", collateralized),
        ):
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"{name} must match netted_values shape {expected_shape}; got {matrix.shape}"
                )
        if len(self.observation_times) != expected_shape[0]:
            raise ValueError("observation_times must match collateral projection time axis")
        if self.observation_dates and len(self.observation_dates) != expected_shape[0]:
            raise ValueError("observation_dates must match collateral projection time axis")
        object.__setattr__(self, "netted_values", netted.copy())
        object.__setattr__(self, "closeout_values", closeout.copy())
        object.__setattr__(self, "collateral_balance", collateral.copy())
        object.__setattr__(self, "collateralized_exposure", collateralized.copy())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def n_observation_times(self) -> int:
        """Return the number of projected future dates."""
        return int(self.netted_values.shape[0])

    @property
    def n_paths(self) -> int:
        """Return the number of projected paths."""
        return int(self.netted_values.shape[1])

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly projection payload."""
        return {
            "netting_set_id": self.netting_set_id,
            "agreement_id": self.agreement_id,
            "observation_times": list(self.observation_times),
            "observation_dates": [obs.isoformat() for obs in self.observation_dates],
            "netted_values": np.asarray(self.netted_values, dtype=float).tolist(),
            "closeout_values": np.asarray(self.closeout_values, dtype=float).tolist(),
            "collateral_balance": np.asarray(self.collateral_balance, dtype=float).tolist(),
            "collateralized_exposure": np.asarray(
                self.collateralized_exposure,
                dtype=float,
            ).tolist(),
            "metadata": dict(self.metadata),
        }


def validate_counterparty_semantic_contract(
    contract: CounterpartySemanticContract,
) -> CounterpartySemanticValidationReport:
    """Validate the bounded collateral/netting representation."""
    missing: list[str] = []
    warnings: list[str] = []

    if not contract.netting_sets:
        missing.append("netting_sets")
    if not contract.collateral_agreements:
        missing.append("collateral_agreements")

    agreement_ids = set(contract.collateral_agreement_map)
    assigned_positions: set[str] = set()
    duplicate_positions: set[str] = set()

    for netting_set in contract.netting_sets:
        prefix = f"netting_sets.{netting_set.netting_set_id}"
        if not netting_set.position_names:
            missing.append(f"{prefix}.position_names")
        if not netting_set.exposure_currency:
            missing.append(f"{prefix}.exposure_currency")
        if not netting_set.collateral_agreement_id:
            warnings.append(f"{prefix}.collateral_agreement_id")
        elif netting_set.collateral_agreement_id not in agreement_ids:
            missing.append(f"{prefix}.collateral_agreement_id")
        for position_name in netting_set.position_names:
            if position_name in assigned_positions:
                duplicate_positions.add(position_name)
            assigned_positions.add(position_name)

    for agreement in contract.collateral_agreements:
        prefix = f"collateral_agreements.{agreement.agreement_id}"
        if not agreement.collateral_currency:
            missing.append(f"{prefix}.collateral_currency")

    for position_name in contract.covered_position_names:
        if position_name not in assigned_positions:
            missing.append(f"covered_position_names.{position_name}")

    for position_name in sorted(duplicate_positions):
        warnings.append(f"position_names.{position_name}.assigned_to_multiple_netting_sets")

    return CounterpartySemanticValidationReport(
        missing_fields=tuple(dict.fromkeys(missing)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def project_collateral_state(
    future_value_cube: FutureValueCube,
    *,
    netting_set: NettingSet,
    collateral_agreement: CollateralAgreement,
) -> CollateralStateProjection:
    """Project pathwise collateral state for one netting set.

    ``collateral_balance`` is based on the valuation-lagged netted value.
    ``closeout_values`` are read from the first observation date on or after the
    margin-period horizon. The resulting exposure is floored after collateral.
    """
    if netting_set.collateral_agreement_id and (
        netting_set.collateral_agreement_id != collateral_agreement.agreement_id
    ):
        raise ValueError(
            "netting_set.collateral_agreement_id does not match collateral_agreement.agreement_id"
        )

    netted_values = _netted_values_for_set(future_value_cube, netting_set)
    observation_dates = future_value_cube.observation_dates
    has_calendar_offset = (
        collateral_agreement.margin_period_of_risk_days
        or collateral_agreement.valuation_lag_days
    )
    if has_calendar_offset and not observation_dates:
        raise ValueError(
            "Collateral projection with margin-period or valuation-lag days requires observation_dates"
        )

    if observation_dates:
        margin_indices = _forward_date_indices(
            observation_dates,
            collateral_agreement.margin_period_of_risk_days,
        )
        lag_indices = _backward_date_indices(
            observation_dates,
            collateral_agreement.valuation_lag_days,
        )
    else:
        identity_indices = tuple(range(netted_values.shape[0]))
        margin_indices = identity_indices
        lag_indices = identity_indices
    closeout_values = np.asarray(
        [netted_values[index] for index in margin_indices],
        dtype=float,
    )
    lagged_values = np.asarray(
        [netted_values[index] for index in lag_indices],
        dtype=float,
    )
    collateral_balance = _collateral_balance_from_values(
        lagged_values,
        collateral_agreement=collateral_agreement,
    )
    collateralized_exposure = np.maximum(closeout_values - collateral_balance, 0.0)

    return CollateralStateProjection(
        netting_set_id=netting_set.netting_set_id,
        agreement_id=collateral_agreement.agreement_id,
        observation_times=future_value_cube.observation_times,
        observation_dates=observation_dates,
        netted_values=netted_values,
        closeout_values=closeout_values,
        collateral_balance=collateral_balance,
        collateralized_exposure=collateralized_exposure,
        metadata={
            "workflow": "project_collateral_state",
            "margin_period_of_risk_days": collateral_agreement.margin_period_of_risk_days,
            "valuation_lag_days": collateral_agreement.valuation_lag_days,
            "threshold": collateral_agreement.threshold,
            "minimum_transfer_amount": collateral_agreement.minimum_transfer_amount,
            "independent_amount": collateral_agreement.independent_amount,
            "margin_period_indices": tuple(int(index) for index in margin_indices),
            "valuation_lag_indices": tuple(int(index) for index in lag_indices),
            "exposure_currency": netting_set.exposure_currency,
            "collateral_currency": collateral_agreement.collateral_currency,
        },
    )


def _coerce_path_matrix(values, *, field_name: str):
    """Return a defensive two-dimensional path matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{field_name} must have shape (observation_time, path)")
    return matrix


def _netted_values_for_set(
    future_value_cube: FutureValueCube,
    netting_set: NettingSet,
):
    """Aggregate the future-value cube over one netting set's position names."""
    if not netting_set.position_names:
        raise ValueError("NettingSet position_names must not be empty")
    matrices = [
        future_value_cube.values_for_position(position_name)
        for position_name in netting_set.position_names
    ]
    return np.sum(np.asarray(matrices, dtype=float), axis=0)


def _forward_date_indices(observation_dates: tuple[date, ...], days: int) -> tuple[int, ...]:
    """Return first indices on or after each date plus the supplied day offset."""
    if not observation_dates:
        return ()
    offset = timedelta(days=int(days))
    indices: list[int] = []
    for obs_date in observation_dates:
        target = obs_date + offset
        selected = len(observation_dates) - 1
        for index, candidate in enumerate(observation_dates):
            if candidate >= target:
                selected = index
                break
        indices.append(selected)
    return tuple(indices)


def _backward_date_indices(observation_dates: tuple[date, ...], days: int) -> tuple[int, ...]:
    """Return latest indices on or before each date minus the supplied day offset."""
    if not observation_dates:
        return ()
    offset = timedelta(days=int(days))
    indices: list[int] = []
    for current_index, obs_date in enumerate(observation_dates):
        target = obs_date - offset
        selected = 0
        for index, candidate in enumerate(observation_dates):
            if candidate <= target:
                selected = index
            else:
                break
        if target < observation_dates[0]:
            selected = current_index if days <= 0 else 0
        indices.append(selected)
    return tuple(indices)


def _collateral_balance_from_values(
    netted_values,
    *,
    collateral_agreement: CollateralAgreement,
):
    """Return held collateral from lagged netted values and agreement terms."""
    positive_excess = np.maximum(
        np.asarray(netted_values, dtype=float) - float(collateral_agreement.threshold),
        0.0,
    )
    variation_margin = np.where(
        positive_excess >= float(collateral_agreement.minimum_transfer_amount),
        positive_excess,
        0.0,
    )
    return variation_margin + float(collateral_agreement.independent_amount)


__all__ = [
    "CollateralAgreement",
    "CollateralStateProjection",
    "CounterpartySemanticContract",
    "CounterpartySemanticValidationReport",
    "NettingSet",
    "project_collateral_state",
    "validate_counterparty_semantic_contract",
]
