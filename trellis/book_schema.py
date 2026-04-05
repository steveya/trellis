"""Stable position-import contracts for mixed-book ingestion workflows."""

from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",")]
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize_instrument_type(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class PositionImportContract:
    """Normalized one-position import contract aligned with structured trade entry."""

    position_id: str
    instrument_type: str
    quantity: float = 1.0
    structured_trade: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position_id = str(self.position_id or "").strip()
        instrument_type = _normalize_instrument_type(self.instrument_type)
        if not position_id:
            raise ValueError("Position import requires `position_id`.")
        if not instrument_type:
            raise ValueError("Position import requires `instrument_type`.")

        structured_trade = dict(self.structured_trade or {})
        embedded_type = _normalize_instrument_type(structured_trade.get("instrument_type"))
        if embedded_type and embedded_type != instrument_type:
            raise ValueError(
                "Position import instrument_type must match structured_trade.instrument_type."
            )
        structured_trade["instrument_type"] = instrument_type

        object.__setattr__(self, "position_id", position_id)
        object.__setattr__(self, "instrument_type", instrument_type)
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "structured_trade", _freeze_mapping(structured_trade))
        object.__setattr__(self, "tags", _string_tuple(self.tags))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "instrument_type": self.instrument_type,
            "quantity": self.quantity,
            "structured_trade": dict(self.structured_trade),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PositionImportResult:
    """Stable parsing result for one imported position payload."""

    parse_status: str
    position_id: str = ""
    instrument_type: str = ""
    asset_class: str = ""
    quantity: float = 1.0
    position_contract: Mapping[str, object] = field(default_factory=dict)
    trade_summary: Mapping[str, object] = field(default_factory=dict)
    field_map: Mapping[str, object] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    normalization_profile: str = "canonical"
    position_contract_object: PositionImportContract | None = field(default=None, repr=False, compare=False)
    trade_parse_result: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parse_status", str(self.parse_status or "").strip())
        object.__setattr__(self, "position_id", str(self.position_id or "").strip())
        object.__setattr__(self, "instrument_type", _normalize_instrument_type(self.instrument_type))
        object.__setattr__(self, "asset_class", str(self.asset_class or "").strip())
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "position_contract", _freeze_mapping(self.position_contract))
        object.__setattr__(self, "trade_summary", _freeze_mapping(self.trade_summary))
        object.__setattr__(self, "field_map", _freeze_mapping(self.field_map))
        object.__setattr__(self, "missing_fields", _string_tuple(self.missing_fields))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))
        object.__setattr__(
            self,
            "normalization_profile",
            str(self.normalization_profile or "canonical").strip() or "canonical",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "parse_status": self.parse_status,
            "position_id": self.position_id,
            "instrument_type": self.instrument_type,
            "asset_class": self.asset_class,
            "quantity": self.quantity,
            "position_contract": dict(self.position_contract),
            "trade_summary": dict(self.trade_summary),
            "field_map": dict(self.field_map),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "normalization_profile": self.normalization_profile,
        }


class ImportedBook(MappingABC[str, PositionImportContract]):
    """Mapping-compatible collection of imported position contracts."""

    def __init__(self, positions: Mapping[str, PositionImportContract]):
        self._positions = dict(positions)

    def __getitem__(self, key: str) -> PositionImportContract:
        return self._positions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def names(self) -> list[str]:
        return list(self._positions.keys())

    @property
    def positions(self) -> dict[str, PositionImportContract]:
        return dict(self._positions)

    def to_dict(self) -> dict[str, object]:
        return {
            "positions": {
                name: contract.to_dict()
                for name, contract in self._positions.items()
            }
        }


@dataclass(frozen=True)
class ImportedBookLoadResult:
    """Stable mixed-book loader result with per-row validation summaries."""

    load_status: str
    position_book: Mapping[str, object] = field(default_factory=dict)
    row_results: tuple[Mapping[str, object], ...] = ()
    parsed_count: int = 0
    incomplete_count: int = 0
    invalid_count: int = 0
    warnings: tuple[str, ...] = ()
    position_book_object: ImportedBook | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "load_status", str(self.load_status or "").strip())
        object.__setattr__(self, "position_book", _freeze_mapping(self.position_book))
        object.__setattr__(
            self,
            "row_results",
            tuple(_freeze_mapping(result) for result in self.row_results),
        )
        object.__setattr__(self, "parsed_count", int(self.parsed_count))
        object.__setattr__(self, "incomplete_count", int(self.incomplete_count))
        object.__setattr__(self, "invalid_count", int(self.invalid_count))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "load_status": self.load_status,
            "position_book": dict(self.position_book),
            "row_results": [dict(result) for result in self.row_results],
            "parsed_count": self.parsed_count,
            "incomplete_count": self.incomplete_count,
            "invalid_count": self.invalid_count,
            "warnings": list(self.warnings),
        }


__all__ = [
    "ImportedBook",
    "ImportedBookLoadResult",
    "PositionImportContract",
    "PositionImportResult",
]
