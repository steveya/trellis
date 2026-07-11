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


@dataclass(frozen=True)
class NettingSetExposureCube:
    """Netting-set/date/path exposure tensor with closeout-ready inputs."""

    netting_set_ids: tuple[str, ...]
    observation_times: tuple[float, ...]
    observation_dates: tuple[date, ...]
    netted_values: Any
    closeout_values: Any
    collateral_balance: Any
    exposure_values: Any
    netting_set_metadata: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = _string_tuple(self.netting_set_ids)
        if not ids:
            raise ValueError("NettingSetExposureCube requires at least one netting set")
        netted = _coerce_netting_tensor(self.netted_values, field_name="netted_values")
        closeout = _coerce_netting_tensor(self.closeout_values, field_name="closeout_values")
        collateral = _coerce_netting_tensor(
            self.collateral_balance,
            field_name="collateral_balance",
        )
        exposure = _coerce_netting_tensor(self.exposure_values, field_name="exposure_values")
        expected_shape = netted.shape
        for name, tensor in (
            ("closeout_values", closeout),
            ("collateral_balance", collateral),
            ("exposure_values", exposure),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must match netted_values shape {expected_shape}; got {tensor.shape}"
                )
        if expected_shape[0] != len(ids):
            raise ValueError("netting_set_ids must match the netting-set axis")
        times = tuple(float(value) for value in self.observation_times)
        if expected_shape[1] != len(times):
            raise ValueError("observation_times must match the observation axis")
        dates = tuple(self.observation_dates or ())
        if dates and expected_shape[1] != len(dates):
            raise ValueError("observation_dates must match the observation axis")
        object.__setattr__(self, "netting_set_ids", ids)
        object.__setattr__(self, "observation_times", times)
        object.__setattr__(self, "observation_dates", dates)
        object.__setattr__(self, "netted_values", netted.copy())
        object.__setattr__(self, "closeout_values", closeout.copy())
        object.__setattr__(self, "collateral_balance", collateral.copy())
        object.__setattr__(self, "exposure_values", exposure.copy())
        object.__setattr__(
            self,
            "netting_set_metadata",
            MappingProxyType(
                {
                    str(key): MappingProxyType(dict(value))
                    for key, value in (self.netting_set_metadata or {}).items()
                }
            ),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def n_netting_sets(self) -> int:
        """Return the number of netting sets."""
        return int(self.netted_values.shape[0])

    @property
    def n_observation_times(self) -> int:
        """Return the number of observation times."""
        return int(self.netted_values.shape[1])

    @property
    def n_paths(self) -> int:
        """Return the number of paths."""
        return int(self.netted_values.shape[2])

    def netting_set_index(self, netting_set_id: str) -> int:
        """Return the tensor index for one netting set id."""
        token = str(netting_set_id)
        try:
            return self.netting_set_ids.index(token)
        except ValueError as exc:
            raise KeyError(f"Unknown netting set {netting_set_id!r}") from exc

    def netted_values_for_set(self, netting_set_id: str):
        """Return netted clean values for one netting set."""
        return np.asarray(
            self.netted_values[self.netting_set_index(netting_set_id)],
            dtype=float,
        ).copy()

    def exposure_values_for_set(self, netting_set_id: str):
        """Return post-closeout exposure values for one netting set."""
        return np.asarray(
            self.exposure_values[self.netting_set_index(netting_set_id)],
            dtype=float,
        ).copy()

    def closeout_input_for_set(self, netting_set_id: str) -> dict[str, object]:
        """Return the closeout-ready exposure input packet for one netting set."""
        index = self.netting_set_index(netting_set_id)
        resolved_id = self.netting_set_ids[index]
        meta = dict(self.netting_set_metadata.get(resolved_id, {}))
        return {
            "netting_set_id": resolved_id,
            "counterparty_id": meta.get("counterparty_id", ""),
            "position_names": list(meta.get("position_names", ())),
            "exposure_currency": meta.get("exposure_currency", ""),
            "collateralized": bool(meta.get("collateralized", False)),
            "observation_times": self.observation_times,
            "observation_dates": self.observation_dates,
            "netted_values": np.asarray(self.netted_values[index], dtype=float).copy(),
            "closeout_values": np.asarray(self.closeout_values[index], dtype=float).copy(),
            "collateral_balance": np.asarray(self.collateral_balance[index], dtype=float).copy(),
            "exposure_values": np.asarray(self.exposure_values[index], dtype=float).copy(),
            "metadata": meta,
        }

    def portfolio_exposure_values(self):
        """Return aggregate exposure across netting sets."""
        return np.sum(np.asarray(self.exposure_values, dtype=float), axis=0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly netting exposure payload."""
        return {
            "netting_set_ids": list(self.netting_set_ids),
            "observation_times": list(self.observation_times),
            "observation_dates": [obs.isoformat() for obs in self.observation_dates],
            "netted_values": np.asarray(self.netted_values, dtype=float).tolist(),
            "closeout_values": np.asarray(self.closeout_values, dtype=float).tolist(),
            "collateral_balance": np.asarray(self.collateral_balance, dtype=float).tolist(),
            "exposure_values": np.asarray(self.exposure_values, dtype=float).tolist(),
            "netting_set_metadata": {
                key: dict(value)
                for key, value in self.netting_set_metadata.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExposureMetricResult:
    """Stable EE/EPE/PFE exposure metric output."""

    aggregation_level: str
    observation_times: tuple[float, ...]
    observation_dates: tuple[date, ...]
    expected_exposure: tuple[float, ...]
    epe: float
    pfe: Mapping[float, tuple[float, ...]]
    netting_set_metrics: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aggregation_level",
            _normalize_token(self.aggregation_level, fallback="portfolio"),
        )
        object.__setattr__(
            self,
            "observation_times",
            tuple(float(value) for value in self.observation_times),
        )
        object.__setattr__(self, "observation_dates", tuple(self.observation_dates or ()))
        object.__setattr__(
            self,
            "expected_exposure",
            tuple(float(value) for value in self.expected_exposure),
        )
        object.__setattr__(self, "epe", float(self.epe))
        object.__setattr__(
            self,
            "pfe",
            MappingProxyType(
                {
                    float(level): tuple(float(value) for value in values)
                    for level, values in (self.pfe or {}).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "netting_set_metrics",
            MappingProxyType(
                {
                    str(key): MappingProxyType(dict(value))
                    for key, value in (self.netting_set_metrics or {}).items()
                }
            ),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def expected_positive_exposure(self) -> tuple[float, ...]:
        """Compatibility alias for EE as expected positive exposure."""
        return self.expected_exposure

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly metric payload."""
        return {
            "aggregation_level": self.aggregation_level,
            "observation_times": list(self.observation_times),
            "observation_dates": [obs.isoformat() for obs in self.observation_dates],
            "expected_exposure": list(self.expected_exposure),
            "expected_positive_exposure": list(self.expected_positive_exposure),
            "epe": self.epe,
            "pfe": {
                str(level): list(values)
                for level, values in self.pfe.items()
            },
            "netting_set_metrics": {
                key: {
                    metric_key: (
                        list(metric_value)
                        if isinstance(metric_value, tuple)
                        else {
                            str(level): list(values)
                            for level, values in metric_value.items()
                        }
                        if isinstance(metric_value, Mapping)
                        else metric_value
                    )
                    for metric_key, metric_value in value.items()
                }
                for key, value in self.netting_set_metrics.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class XVAAssumptionSet:
    """Scalar assumptions for the bounded counterparty xVA workflow."""

    counterparty_hazard_rate: float
    counterparty_recovery_rate: float
    own_hazard_rate: float = 0.0
    own_recovery_rate: float = 0.0
    funding_spread: float = 0.0
    discount_rate: float = 0.0
    currency: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterparty_hazard_rate",
            _non_negative_float(
                self.counterparty_hazard_rate,
                field_name="counterparty_hazard_rate",
            ),
        )
        object.__setattr__(
            self,
            "own_hazard_rate",
            _non_negative_float(self.own_hazard_rate, field_name="own_hazard_rate"),
        )
        object.__setattr__(
            self,
            "funding_spread",
            _non_negative_float(self.funding_spread, field_name="funding_spread"),
        )
        object.__setattr__(self, "discount_rate", float(self.discount_rate))
        for field_name in ("counterparty_recovery_rate", "own_recovery_rate"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must satisfy 0 <= value <= 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "currency", _normalize_token(self.currency))

    @property
    def counterparty_loss_given_default(self) -> float:
        """Return counterparty LGD."""
        return 1.0 - self.counterparty_recovery_rate

    @property
    def own_loss_given_default(self) -> float:
        """Return own LGD."""
        return 1.0 - self.own_recovery_rate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly assumption payload."""
        return {
            "counterparty_hazard_rate": self.counterparty_hazard_rate,
            "counterparty_recovery_rate": self.counterparty_recovery_rate,
            "own_hazard_rate": self.own_hazard_rate,
            "own_recovery_rate": self.own_recovery_rate,
            "funding_spread": self.funding_spread,
            "discount_rate": self.discount_rate,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class XVAResult:
    """Bounded CVA/DVA/FVA result over the institutional exposure stack."""

    exposure_metrics: ExposureMetricResult
    assumptions: XVAAssumptionSet
    expected_negative_exposure: tuple[float, ...]
    negative_epe: float
    cva: float
    dva: float
    fva: float
    total_xva: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_negative_exposure",
            tuple(float(value) for value in self.expected_negative_exposure),
        )
        object.__setattr__(self, "negative_epe", float(self.negative_epe))
        object.__setattr__(self, "cva", float(self.cva))
        object.__setattr__(self, "dva", float(self.dva))
        object.__setattr__(self, "fva", float(self.fva))
        object.__setattr__(self, "total_xva", float(self.total_xva))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly xVA payload."""
        return {
            "cva": self.cva,
            "dva": self.dva,
            "fva": self.fva,
            "total_xva": self.total_xva,
            "expected_negative_exposure": list(self.expected_negative_exposure),
            "negative_epe": self.negative_epe,
            "assumptions": self.assumptions.to_dict(),
            "exposure_metrics": self.exposure_metrics.to_dict(),
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


def aggregate_netting_set_exposures(
    future_value_cube: FutureValueCube,
    *,
    netting_sets: tuple[NettingSet, ...] | list[NettingSet],
    collateral_projections: Mapping[str, CollateralStateProjection] | None = None,
) -> NettingSetExposureCube:
    """Aggregate trade future values into closeout-ready netting-set exposures."""
    resolved_sets = tuple(netting_sets or ())
    if not resolved_sets:
        raise ValueError("aggregate_netting_set_exposures requires at least one netting set")

    projections = dict(collateral_projections or {})
    ids: list[str] = []
    netted_tensors = []
    closeout_tensors = []
    collateral_tensors = []
    exposure_tensors = []
    set_metadata: dict[str, dict[str, object]] = {}

    for netting_set in resolved_sets:
        netted = _netted_values_for_set(future_value_cube, netting_set)
        projection = projections.get(netting_set.netting_set_id)
        if projection is None:
            closeout = netted
            collateral = np.zeros_like(netted)
            exposure = np.maximum(closeout, 0.0)
            collateralized = False
            projection_metadata: dict[str, object] = {}
        else:
            _validate_projection_alignment(
                projection,
                future_value_cube=future_value_cube,
                netting_set=netting_set,
                netted_values=netted,
            )
            closeout = np.asarray(projection.closeout_values, dtype=float)
            collateral = np.asarray(projection.collateral_balance, dtype=float)
            exposure = np.asarray(projection.collateralized_exposure, dtype=float)
            collateralized = True
            projection_metadata = dict(projection.metadata)

        ids.append(netting_set.netting_set_id)
        netted_tensors.append(netted)
        closeout_tensors.append(closeout)
        collateral_tensors.append(collateral)
        exposure_tensors.append(exposure)
        set_metadata[netting_set.netting_set_id] = {
            "counterparty_id": netting_set.counterparty_id,
            "position_names": netting_set.position_names,
            "exposure_currency": netting_set.exposure_currency,
            "closeout_convention": netting_set.closeout_convention,
            "collateralized": collateralized,
            "collateral_agreement_id": netting_set.collateral_agreement_id,
            "collateral_projection": projection_metadata,
        }

    return NettingSetExposureCube(
        netting_set_ids=tuple(ids),
        observation_times=future_value_cube.observation_times,
        observation_dates=future_value_cube.observation_dates,
        netted_values=np.asarray(netted_tensors, dtype=float),
        closeout_values=np.asarray(closeout_tensors, dtype=float),
        collateral_balance=np.asarray(collateral_tensors, dtype=float),
        exposure_values=np.asarray(exposure_tensors, dtype=float),
        netting_set_metadata=set_metadata,
        metadata={
            "workflow": "aggregate_netting_set_exposures",
            "source_value_semantics": future_value_cube.value_semantics,
            "source_phase_semantics": future_value_cube.phase_semantics,
            "netting_set_count": len(ids),
            "position_names": future_value_cube.position_names,
        },
    )


def compute_exposure_metrics(
    exposure_cube: NettingSetExposureCube,
    *,
    pfe_levels: tuple[float, ...] | list[float] = (0.95,),
) -> ExposureMetricResult:
    """Compute deterministic EE, EPE, and PFE outputs from exposure values."""
    levels = _normalize_pfe_levels(pfe_levels)
    portfolio_metrics = _exposure_metrics_for_matrix(
        exposure_cube.portfolio_exposure_values(),
        observation_times=exposure_cube.observation_times,
        pfe_levels=levels,
    )
    per_set: dict[str, dict[str, object]] = {}
    for netting_set_id in exposure_cube.netting_set_ids:
        set_metrics = _exposure_metrics_for_matrix(
            exposure_cube.exposure_values_for_set(netting_set_id),
            observation_times=exposure_cube.observation_times,
            pfe_levels=levels,
        )
        per_set[netting_set_id] = {
            "expected_exposure": set_metrics["expected_exposure"],
            "epe": set_metrics["epe"],
            "pfe": set_metrics["pfe"],
        }

    return ExposureMetricResult(
        aggregation_level="portfolio",
        observation_times=exposure_cube.observation_times,
        observation_dates=exposure_cube.observation_dates,
        expected_exposure=portfolio_metrics["expected_exposure"],
        epe=portfolio_metrics["epe"],
        pfe=portfolio_metrics["pfe"],
        netting_set_metrics=per_set,
        metadata={
            "workflow": "compute_exposure_metrics",
            "source": "NettingSetExposureCube",
            "pfe_levels": levels,
            "netting_set_ids": exposure_cube.netting_set_ids,
            "time_weighting": "trapezoidal_observation_time",
        },
    )


def compute_xva_from_exposure_cube(
    exposure_cube: NettingSetExposureCube,
    *,
    assumptions: XVAAssumptionSet,
    pfe_levels: tuple[float, ...] | list[float] = (0.95,),
    metadata: Mapping[str, object] | None = None,
) -> XVAResult:
    """Compute bounded CVA/DVA/FVA from a netting-set exposure cube."""
    metrics = compute_exposure_metrics(exposure_cube, pfe_levels=pfe_levels)
    residual_values = (
        np.asarray(exposure_cube.closeout_values, dtype=float)
        - np.asarray(exposure_cube.collateral_balance, dtype=float)
    )
    negative_values = np.maximum(-residual_values, 0.0)
    portfolio_negative_values = np.sum(negative_values, axis=0)
    expected_negative = tuple(
        float(value)
        for value in np.mean(portfolio_negative_values, axis=1)
    )
    negative_epe = _time_weighted_average(
        expected_negative,
        observation_times=exposure_cube.observation_times,
    )
    positive_integral = _discounted_time_integral(
        metrics.expected_exposure,
        observation_times=exposure_cube.observation_times,
        discount_rate=assumptions.discount_rate,
    )
    negative_integral = _discounted_time_integral(
        expected_negative,
        observation_times=exposure_cube.observation_times,
        discount_rate=assumptions.discount_rate,
    )
    cva = (
        assumptions.counterparty_loss_given_default
        * assumptions.counterparty_hazard_rate
        * positive_integral
    )
    dva = assumptions.own_loss_given_default * assumptions.own_hazard_rate * negative_integral
    fva = assumptions.funding_spread * positive_integral
    return XVAResult(
        exposure_metrics=metrics,
        assumptions=assumptions,
        expected_negative_exposure=expected_negative,
        negative_epe=negative_epe,
        cva=cva,
        dva=dva,
        fva=fva,
        total_xva=cva - dva + fva,
        metadata={
            "workflow": "compute_xva_from_exposure_cube",
            "xva_components": ("cva", "dva", "fva"),
            "discounting": "flat_continuous_rate",
            **dict(metadata or {}),
        },
    )


def price_counterparty_xva(
    future_value_cube: FutureValueCube,
    *,
    counterparty_contract: CounterpartySemanticContract,
    assumptions: XVAAssumptionSet,
    pfe_levels: tuple[float, ...] | list[float] = (0.95,),
) -> XVAResult:
    """Run the bounded counterparty xVA workflow from semantic inputs."""
    validation = validate_counterparty_semantic_contract(counterparty_contract)
    if not validation.ok:
        missing = ", ".join(validation.missing_fields)
        raise ValueError(f"Counterparty semantic contract is incomplete: {missing}")

    agreements = counterparty_contract.collateral_agreement_map
    projections: dict[str, CollateralStateProjection] = {}
    for netting_set in counterparty_contract.netting_sets:
        if not netting_set.collateral_agreement_id:
            continue
        agreement = agreements[netting_set.collateral_agreement_id]
        projections[netting_set.netting_set_id] = project_collateral_state(
            future_value_cube,
            netting_set=netting_set,
            collateral_agreement=agreement,
        )
    exposure_cube = aggregate_netting_set_exposures(
        future_value_cube,
        netting_sets=counterparty_contract.netting_sets,
        collateral_projections=projections,
    )
    return compute_xva_from_exposure_cube(
        exposure_cube,
        assumptions=assumptions,
        pfe_levels=pfe_levels,
        metadata={
            "workflow": "price_counterparty_xva",
            "counterparty_contract_id": counterparty_contract.contract_id,
            "netting_set_ids": tuple(
                netting_set.netting_set_id
                for netting_set in counterparty_contract.netting_sets
            ),
            "validation_warnings": validation.warnings,
        },
    )


def price_interest_rate_swap_cva_monte_carlo(
    market_state,
    *,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.045,
    maturity_years: int = 3,
    rate_index: str = "USD-SOFR-3M",
    counterparty_hazard_rate: float = 0.02,
    counterparty_recovery_rate: float = 0.40,
    n_paths: int = 1024,
    n_steps: int = 72,
    seed: int | None = 52,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Price bounded IRS CVA from a simulated swap future-value cube."""
    exposure_cube, assumptions = _interest_rate_swap_cva_exposure_stack(
        market_state,
        notional=notional,
        fixed_rate=fixed_rate,
        maturity_years=maturity_years,
        rate_index=rate_index,
        counterparty_hazard_rate=counterparty_hazard_rate,
        counterparty_recovery_rate=counterparty_recovery_rate,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    return float(compute_xva_from_exposure_cube(exposure_cube, assumptions=assumptions).cva)


def price_interest_rate_swap_cva_analytical_approx(
    market_state,
    *,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.045,
    maturity_years: int = 3,
    rate_index: str = "USD-SOFR-3M",
    counterparty_hazard_rate: float = 0.02,
    counterparty_recovery_rate: float = 0.40,
    n_paths: int = 1024,
    n_steps: int = 72,
    seed: int | None = 52,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Approximate IRS CVA by integrating expected exposure under flat hazard."""
    return price_interest_rate_swap_cva_monte_carlo(
        market_state,
        notional=notional,
        fixed_rate=fixed_rate,
        maturity_years=maturity_years,
        rate_index=rate_index,
        counterparty_hazard_rate=counterparty_hazard_rate,
        counterparty_recovery_rate=counterparty_recovery_rate,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )


def price_interest_rate_swap_independent_cva(
    market_state,
    *,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.045,
    maturity_years: int = 3,
    rate_index: str = "USD-SOFR-3M",
    counterparty_hazard_rate: float = 0.02,
    counterparty_recovery_rate: float = 0.40,
    n_paths: int = 1024,
    n_steps: int = 72,
    seed: int | None = 54,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Price independent-default IRS CVA on the bounded task fixture."""
    return price_interest_rate_swap_cva_monte_carlo(
        market_state,
        notional=notional,
        fixed_rate=fixed_rate,
        maturity_years=maturity_years,
        rate_index=rate_index,
        counterparty_hazard_rate=counterparty_hazard_rate,
        counterparty_recovery_rate=counterparty_recovery_rate,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )


def price_interest_rate_swap_wrong_way_cva(
    market_state,
    *,
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.045,
    maturity_years: int = 3,
    rate_index: str = "USD-SOFR-3M",
    counterparty_hazard_rate: float = 0.02,
    counterparty_recovery_rate: float = 0.40,
    default_exposure_correlation: float = 0.35,
    n_paths: int = 1024,
    n_steps: int = 72,
    seed: int | None = 54,
    mean_reversion: float | None = None,
    sigma: float | None = None,
) -> float:
    """Price bounded IRS CVA with pathwise default intensity tilted by exposure."""
    if not -1.0 <= float(default_exposure_correlation) <= 1.0:
        raise ValueError("default_exposure_correlation must satisfy -1 <= rho <= 1")
    exposure_cube, assumptions = _interest_rate_swap_cva_exposure_stack(
        market_state,
        notional=notional,
        fixed_rate=fixed_rate,
        maturity_years=maturity_years,
        rate_index=rate_index,
        counterparty_hazard_rate=counterparty_hazard_rate,
        counterparty_recovery_rate=counterparty_recovery_rate,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    weighted_expected_exposure = _wrong_way_expected_exposure(
        exposure_cube.portfolio_exposure_values(),
        default_exposure_correlation=float(default_exposure_correlation),
    )
    positive_integral = _discounted_time_integral(
        weighted_expected_exposure,
        observation_times=exposure_cube.observation_times,
        discount_rate=assumptions.discount_rate,
    )
    return float(
        assumptions.counterparty_loss_given_default
        * assumptions.counterparty_hazard_rate
        * positive_integral
    )


def _interest_rate_swap_cva_exposure_stack(
    market_state,
    *,
    notional: float,
    fixed_rate: float,
    maturity_years: int,
    rate_index: str,
    counterparty_hazard_rate: float,
    counterparty_recovery_rate: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    mean_reversion: float | None,
    sigma: float | None,
) -> tuple[NettingSetExposureCube, XVAAssumptionSet]:
    """Build the supported IRS exposure stack used by task-facing CVA helpers."""
    from trellis.instruments.swap import SwapSpec
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_future_value_cube,
    )
    from trellis.core.types import DayCountConvention, Frequency

    settlement = market_state.settlement
    end_date = _same_month_day_year_offset(settlement, int(maturity_years))
    swap_spec = SwapSpec(
        notional=float(notional),
        fixed_rate=float(fixed_rate),
        start_date=settlement,
        end_date=end_date,
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.QUARTERLY,
        fixed_day_count=DayCountConvention.THIRTY_360,
        float_day_count=DayCountConvention.ACT_360,
        rate_index=str(rate_index or "USD-SOFR-3M"),
        is_payer=True,
    )
    future_value_cube = price_interest_rate_swap_future_value_cube(
        name="payer_swap",
        spec=swap_spec,
        market_state=market_state,
        n_paths=max(int(n_paths), 1),
        n_steps=max(int(n_steps), 1),
        seed=seed,
        mean_reversion=mean_reversion,
        sigma=sigma,
    )
    netting_set = NettingSet(
        netting_set_id="ns_irs_cva",
        counterparty_id="counterparty_alpha",
        position_names=("payer_swap",),
        exposure_currency="USD",
    )
    exposure_cube = aggregate_netting_set_exposures(
        future_value_cube,
        netting_sets=(netting_set,),
    )
    assumptions = XVAAssumptionSet(
        counterparty_hazard_rate=float(counterparty_hazard_rate),
        counterparty_recovery_rate=float(counterparty_recovery_rate),
        discount_rate=_discount_rate_from_market_state(market_state),
        currency="USD",
    )
    return exposure_cube, assumptions


def _wrong_way_expected_exposure(
    exposure_values,
    *,
    default_exposure_correlation: float,
) -> tuple[float, ...]:
    """Return exposure expectations tilted by a bounded default-intensity proxy."""
    matrix = _coerce_path_matrix(exposure_values, field_name="exposure_values")
    rho = float(default_exposure_correlation)
    weighted: list[float] = []
    for row in matrix:
        row = np.asarray(row, dtype=float)
        row_mean = float(np.mean(row))
        row_std = float(np.std(row))
        if row_std <= 1e-12 or abs(rho) <= 1e-12:
            weighted.append(row_mean)
            continue
        z_score = (row - row_mean) / row_std
        intensity_multiplier = np.maximum(0.05, 1.0 + rho * z_score)
        intensity_multiplier = intensity_multiplier / float(np.mean(intensity_multiplier))
        weighted.append(float(np.mean(row * intensity_multiplier)))
    return tuple(weighted)


def _discount_rate_from_market_state(market_state) -> float:
    """Return a stable short discount rate from a market state."""
    discount = getattr(market_state, "discount", None)
    if discount is None or not hasattr(discount, "zero_rate"):
        return 0.0
    try:
        return float(discount.zero_rate(1.0))
    except Exception:
        return 0.0


def _same_month_day_year_offset(start: date, years: int) -> date:
    """Return a same-month/day date, falling back for leap-day style dates."""
    target_year = int(start.year) + max(int(years), 1)
    try:
        return date(target_year, start.month, start.day)
    except ValueError:
        return date(target_year, start.month, 28)


def _coerce_path_matrix(values, *, field_name: str):
    """Return a defensive two-dimensional path matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{field_name} must have shape (observation_time, path)")
    return matrix


def _coerce_netting_tensor(values, *, field_name: str):
    """Return a defensive three-dimensional netting-set tensor."""
    tensor = np.asarray(values, dtype=float)
    if tensor.ndim != 3:
        raise ValueError(
            f"{field_name} must have shape (netting_set, observation_time, path)"
        )
    return tensor


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


def _normalize_pfe_levels(levels) -> tuple[float, ...]:
    """Normalize PFE quantile levels."""
    if not levels:
        raise ValueError("pfe_levels must not be empty")
    resolved: list[float] = []
    for level in levels:
        value = float(level)
        if not 0.0 <= value <= 1.0:
            raise ValueError("PFE levels must satisfy 0 <= level <= 1")
        if value not in resolved:
            resolved.append(value)
    return tuple(resolved)


def _exposure_metrics_for_matrix(
    exposure_values,
    *,
    observation_times: tuple[float, ...],
    pfe_levels: tuple[float, ...],
) -> dict[str, object]:
    """Return EE/EPE/PFE metrics for one date/path exposure matrix."""
    matrix = _coerce_path_matrix(exposure_values, field_name="exposure_values")
    expected = tuple(float(value) for value in np.mean(matrix, axis=1))
    pfe = {
        level: tuple(float(value) for value in np.quantile(matrix, level, axis=1))
        for level in pfe_levels
    }
    return {
        "expected_exposure": expected,
        "epe": _time_weighted_average(expected, observation_times=observation_times),
        "pfe": pfe,
    }


def _time_weighted_average(
    values: tuple[float, ...],
    *,
    observation_times: tuple[float, ...],
) -> float:
    """Return trapezoidal time-weighted average exposure."""
    if not values:
        return 0.0
    times = tuple(float(time) for time in observation_times)
    if len(values) != len(times):
        raise ValueError("exposure metric values must match observation_times")
    if len(values) == 1:
        return float(values[0])
    horizon = float(times[-1] - times[0])
    if horizon <= 0.0:
        return float(np.mean(np.asarray(values, dtype=float)))
    integral = 0.0
    for left_value, right_value, left_time, right_time in zip(
        values,
        values[1:],
        times,
        times[1:],
    ):
        integral += 0.5 * (float(left_value) + float(right_value)) * float(
            right_time - left_time
        )
    return float(integral / horizon)


def _discounted_time_integral(
    values: tuple[float, ...],
    *,
    observation_times: tuple[float, ...],
    discount_rate: float,
) -> float:
    """Return the trapezoidal discounted exposure integral."""
    if not values:
        return 0.0
    times = tuple(float(time) for time in observation_times)
    if len(values) != len(times):
        raise ValueError("discounted exposure values must match observation_times")
    if len(values) == 1:
        return 0.0
    integral = 0.0
    for left_value, right_value, left_time, right_time in zip(
        values,
        values[1:],
        times,
        times[1:],
    ):
        left_discount = float(np.exp(-float(discount_rate) * float(left_time)))
        right_discount = float(np.exp(-float(discount_rate) * float(right_time)))
        integral += 0.5 * (
            float(left_value) * left_discount
            + float(right_value) * right_discount
        ) * float(right_time - left_time)
    return float(integral)


def _validate_projection_alignment(
    projection: CollateralStateProjection,
    *,
    future_value_cube: FutureValueCube,
    netting_set: NettingSet,
    netted_values,
) -> None:
    """Validate that a collateral projection belongs to the netting exposure input."""
    if projection.netting_set_id != netting_set.netting_set_id:
        raise ValueError("Collateral projection netting_set_id does not match netting set")
    if projection.observation_times != future_value_cube.observation_times:
        raise ValueError("Collateral projection observation_times do not match cube")
    if projection.observation_dates != future_value_cube.observation_dates:
        raise ValueError("Collateral projection observation_dates do not match cube")
    netted_difference = np.abs(
        np.asarray(projection.netted_values, dtype=float) - netted_values
    )
    if bool(np.any(netted_difference > 1e-8)):
        raise ValueError("Collateral projection netted_values do not match cube aggregation")


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
    "aggregate_netting_set_exposures",
    "CollateralAgreement",
    "CollateralStateProjection",
    "CounterpartySemanticContract",
    "CounterpartySemanticValidationReport",
    "compute_exposure_metrics",
    "compute_xva_from_exposure_cube",
    "ExposureMetricResult",
    "NettingSet",
    "NettingSetExposureCube",
    "project_collateral_state",
    "price_counterparty_xva",
    "price_interest_rate_swap_cva_analytical_approx",
    "price_interest_rate_swap_cva_monte_carlo",
    "price_interest_rate_swap_independent_cva",
    "price_interest_rate_swap_wrong_way_cva",
    "validate_counterparty_semantic_contract",
    "XVAAssumptionSet",
    "XVAResult",
]
