"""Transport-neutral typed trade parsing for governed workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from trellis.agent.knowledge.schema import ProductIR
from trellis.book_schema import (
    ImportedBook,
    ImportedBookLoadResult,
    PositionImportContract,
    PositionImportResult,
)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable ordered tuple of unique strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class TradeParseResult:
    """Stable machine-readable parsing result for governed trade workflows."""

    parse_status: str
    semantic_id: str = ""
    semantic_version: str = ""
    trade_type: str = ""
    asset_class: str = ""
    parsed_contract: Mapping[str, object] = field(default_factory=dict)
    contract_summary: Mapping[str, object] = field(default_factory=dict)
    product_ir: Mapping[str, object] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    candidate_methods: tuple[str, ...] = ()
    required_market_data: tuple[str, ...] = ()
    normalization_profile: str = "canonical"
    compatibility_surface: str = ""
    semantic_contract: object | None = field(default=None, repr=False, compare=False)
    product_ir_object: ProductIR | None = field(default=None, repr=False, compare=False)
    semantic_blueprint: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "parse_status", str(self.parse_status or "").strip())
        object.__setattr__(self, "semantic_id", str(self.semantic_id or "").strip())
        object.__setattr__(self, "semantic_version", str(self.semantic_version or "").strip())
        object.__setattr__(self, "trade_type", str(self.trade_type or "").strip())
        object.__setattr__(self, "asset_class", str(self.asset_class or "").strip())
        object.__setattr__(self, "parsed_contract", _freeze_mapping(self.parsed_contract))
        object.__setattr__(self, "contract_summary", _freeze_mapping(self.contract_summary))
        object.__setattr__(self, "product_ir", _freeze_mapping(self.product_ir))
        object.__setattr__(self, "missing_fields", _string_tuple(self.missing_fields))
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))
        object.__setattr__(self, "candidate_methods", _string_tuple(self.candidate_methods))
        object.__setattr__(self, "required_market_data", _string_tuple(self.required_market_data))
        object.__setattr__(self, "normalization_profile", str(self.normalization_profile or "canonical").strip() or "canonical")
        object.__setattr__(self, "compatibility_surface", str(self.compatibility_surface or "").strip())

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable payload."""
        return {
            "parse_status": self.parse_status,
            "semantic_id": self.semantic_id,
            "semantic_version": self.semantic_version,
            "trade_type": self.trade_type,
            "asset_class": self.asset_class,
            "parsed_contract": dict(self.parsed_contract),
            "contract_summary": dict(self.contract_summary),
            "product_ir": dict(self.product_ir),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "candidate_methods": list(self.candidate_methods),
            "required_market_data": list(self.required_market_data),
            "normalization_profile": self.normalization_profile,
            "compatibility_surface": self.compatibility_surface,
        }


class TradeService:
    """Normalize natural-language or structured trades into typed contract surfaces."""

    POSITION_FIELD_ALIASES = {
        "position_id": ("position_id", "id", "name", "position_name"),
        "instrument_type": ("instrument_type", "trade_type"),
        "quantity": ("quantity", "qty", "units", "position_scale"),
        "structured_trade": ("structured_trade", "trade", "terms"),
        "tags": ("tags", "labels"),
        "metadata": ("metadata",),
    }
    POSITION_SEQUENCE_FIELDS = frozenset(
        {
            "tags",
            "labels",
            "observation_schedule",
            "fixing_schedule",
            "fixing_dates",
            "exercise_schedule",
            "call_dates",
            "call_schedule",
            "underliers",
            "constituents",
            "reference_entities",
        }
    )
    POSITION_MAPPING_FIELDS = frozenset({"trade", "structured_trade", "terms", "metadata"})

    def parse_trade(
        self,
        *,
        description: str | None = None,
        instrument_type: str | None = None,
        structured_trade: Mapping[str, object] | None = None,
        normalization_profile: str = "canonical",
    ) -> TradeParseResult:
        """Parse one trade request into a typed semantic contract surface."""
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract
        from trellis.agent.semantic_contract_validation import (
            classify_semantic_gap,
            semantic_gap_summary,
            validate_semantic_contract,
        )
        from trellis.agent.semantic_contracts import draft_semantic_contract

        warnings: list[str] = []
        profile = str(normalization_profile or "canonical").strip() or "canonical"
        if profile not in {"canonical", "strict"}:
            warnings.append(
                f"Normalization profile {profile!r} is unsupported and was treated as canonical."
            )
            profile = "canonical"

        if structured_trade is not None:
            try:
                maybe_contract = self._structured_semantic_contract(
                    structured_trade,
                    description=description,
                    instrument_type=instrument_type,
                )
            except ValueError as exc:
                warning_text = str(exc).strip()
                normalized_type = self._instrument_type(
                    structured_trade.get("instrument_type") or instrument_type
                )
                warnings.append(warning_text)
                return TradeParseResult(
                    parse_status="invalid",
                    trade_type=normalized_type,
                    asset_class=self._asset_class_for_instrument(normalized_type),
                    contract_summary={"summary": warning_text},
                    warnings=warnings,
                    normalization_profile=profile,
                    compatibility_surface="structured_trade",
                )
            if maybe_contract is None:
                missing = self._structured_missing_fields(
                    structured_trade,
                    instrument_type=instrument_type,
                )
                warning_text = "Structured trade input is missing required fields for typed normalization."
                warnings.append(warning_text)
                return TradeParseResult(
                    parse_status="incomplete",
                    trade_type=self._instrument_type(structured_trade.get("instrument_type") or instrument_type),
                    asset_class=self._asset_class_for_instrument(structured_trade.get("instrument_type") or instrument_type),
                    contract_summary={
                        "missing_fields": list(missing),
                        "summary": warning_text,
                    },
                    missing_fields=missing,
                    warnings=warnings,
                    normalization_profile=profile,
                    compatibility_surface="structured_trade",
                )
            validation = validate_semantic_contract(maybe_contract)
            warnings.extend(validation.warnings)
            return self._successful_parse_result(
                contract=validation.normalized_contract,
                semantic_blueprint=compile_semantic_contract(validation.normalized_contract),
                validation_warnings=warnings,
                normalization_profile=profile,
                compatibility_surface="structured_trade",
            )

        text = str(description or "").strip()
        normalized_instrument_type = self._instrument_type(instrument_type)
        try:
            semantic_contract = draft_semantic_contract(
                text,
                instrument_type=normalized_instrument_type or None,
            )
        except ValueError as exc:
            warning_text = str(exc).strip()
            missing = self._missing_fields_from_error(
                warning_text,
                instrument_type=normalized_instrument_type,
            )
            if missing:
                warnings.append(warning_text)
                return TradeParseResult(
                    parse_status="incomplete",
                    trade_type=normalized_instrument_type,
                    asset_class=self._asset_class_for_instrument(normalized_instrument_type),
                    contract_summary={
                        "missing_fields": list(missing),
                        "summary": warning_text,
                    },
                    missing_fields=missing,
                    warnings=warnings,
                    normalization_profile=profile,
                    compatibility_surface="semantic_contract",
                )
            semantic_contract = None
        if semantic_contract is not None:
            validation = validate_semantic_contract(semantic_contract)
            warnings.extend(validation.warnings)
            return self._successful_parse_result(
                contract=validation.normalized_contract,
                semantic_blueprint=compile_semantic_contract(validation.normalized_contract),
                validation_warnings=warnings,
                normalization_profile=profile,
                compatibility_surface="semantic_contract",
            )

        gap = classify_semantic_gap(
            text,
            instrument_type=normalized_instrument_type or None,
        )
        gap_summary = semantic_gap_summary(gap)
        warnings.extend((str(gap.summary).strip(),))
        return TradeParseResult(
            parse_status="incomplete",
            trade_type=normalized_instrument_type,
            asset_class=self._asset_class_for_instrument(normalized_instrument_type),
            contract_summary=gap_summary,
            missing_fields=gap.missing_contract_fields,
            warnings=warnings,
            normalization_profile=profile,
            compatibility_surface="gap_report",
        )

    def parse_position(
        self,
        *,
        structured_position: Mapping[str, object],
        normalization_profile: str = "canonical",
    ) -> PositionImportResult:
        """Normalize one imported position row onto the generic position schema."""
        payload = dict(structured_position or {})
        profile = str(normalization_profile or "canonical").strip() or "canonical"
        field_map: dict[str, object] = {}
        warnings: list[str] = []
        missing_fields: list[str] = []

        position_id, position_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["position_id"],
        )
        if position_key is not None:
            field_map["position_id"] = position_key
        if str(position_id or "").strip() == "":
            missing_fields.append("position_id")

        instrument_value, instrument_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["instrument_type"],
        )
        if instrument_key is not None:
            field_map["instrument_type"] = instrument_key

        quantity_value, quantity_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["quantity"],
        )
        if quantity_key is not None:
            field_map["quantity"] = quantity_key
        quantity = 1.0
        if quantity_key is not None:
            try:
                quantity = float(quantity_value)
            except (TypeError, ValueError):
                warning_text = "Position import requires `quantity` to be numeric when provided."
                warnings.append(warning_text)
                return PositionImportResult(
                    parse_status="invalid",
                    position_id=str(position_id or "").strip(),
                    instrument_type=str(instrument_value or "").strip(),
                    quantity=1.0,
                    field_map=field_map,
                    warnings=warnings,
                    normalization_profile=profile,
                )

        tags_value, tags_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["tags"],
        )
        if tags_key is not None:
            field_map["tags"] = tags_key
        metadata_value, metadata_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["metadata"],
        )
        if metadata_key is not None:
            field_map["metadata"] = metadata_key
        metadata = dict(metadata_value or {}) if isinstance(metadata_value, Mapping) else {}

        trade_payload, trade_field_source, trade_warning = self._position_trade_payload(payload)
        if trade_warning:
            warnings.append(trade_warning)
            return PositionImportResult(
                parse_status="invalid",
                position_id=str(position_id or "").strip(),
                instrument_type=str(instrument_value or "").strip(),
                quantity=quantity,
                field_map=field_map,
                warnings=warnings,
                normalization_profile=profile,
            )
        field_map["structured_trade"] = trade_field_source

        embedded_instrument = self._instrument_type(trade_payload.get("instrument_type"))
        normalized_instrument = self._instrument_type(instrument_value or embedded_instrument)
        if not normalized_instrument:
            missing_fields.append("instrument_type")
        elif embedded_instrument and embedded_instrument != normalized_instrument:
            warning_text = (
                "Position import instrument_type must match structured_trade.instrument_type."
            )
            warnings.append(warning_text)
            return PositionImportResult(
                parse_status="invalid",
                position_id=str(position_id or "").strip(),
                instrument_type=normalized_instrument,
                quantity=quantity,
                field_map=field_map,
                warnings=warnings,
                normalization_profile=profile,
            )

        if missing_fields:
            return PositionImportResult(
                parse_status="incomplete",
                position_id=str(position_id or "").strip(),
                instrument_type=normalized_instrument,
                quantity=quantity,
                field_map=field_map,
                missing_fields=tuple(missing_fields),
                warnings=warnings,
                normalization_profile=profile,
            )

        contract = PositionImportContract(
            position_id=str(position_id).strip(),
            instrument_type=normalized_instrument,
            quantity=quantity,
            structured_trade=trade_payload,
            tags=tags_value or (),
            metadata=metadata,
        )
        trade_result = self.parse_trade(
            structured_trade=contract.structured_trade,
            normalization_profile=profile,
        )
        warnings.extend(trade_result.warnings)
        return PositionImportResult(
            parse_status=trade_result.parse_status,
            position_id=contract.position_id,
            instrument_type=contract.instrument_type,
            asset_class=trade_result.asset_class,
            quantity=contract.quantity,
            position_contract=contract.to_dict(),
            trade_summary=trade_result.to_dict(),
            field_map=field_map,
            missing_fields=trade_result.missing_fields,
            warnings=warnings,
            normalization_profile=profile,
            position_contract_object=contract,
            trade_parse_result=trade_result,
        )

    def load_positions(
        self,
        *,
        structured_positions,
        normalization_profile: str = "canonical",
    ) -> ImportedBookLoadResult:
        """Load one mixed supported book from normalized position rows."""
        profile = str(normalization_profile or "canonical").strip() or "canonical"
        row_results: list[dict[str, object]] = []
        positions: dict[str, PositionImportContract] = {}
        warnings: list[str] = []
        parsed_count = 0
        incomplete_count = 0
        invalid_count = 0

        for row_index, raw_row in enumerate(structured_positions or (), start=1):
            if not isinstance(raw_row, Mapping):
                invalid_count += 1
                row_results.append(
                    {
                        "row_index": row_index,
                        "parse_status": "invalid",
                        "warnings": ["Position import row must be a mapping."],
                    }
                )
                continue

            result = self.parse_position(
                structured_position=raw_row,
                normalization_profile=profile,
            )
            row_payload = {"row_index": row_index, **result.to_dict()}
            if result.parse_status == "parsed":
                contract = result.position_contract_object
                if contract is None:
                    invalid_count += 1
                    row_payload["parse_status"] = "invalid"
                    row_payload["warnings"] = ["Parsed position is missing its typed import contract."]
                elif contract.position_id in positions:
                    invalid_count += 1
                    warning_text = (
                        f"Duplicate position_id {contract.position_id!r} is not allowed in imported books."
                    )
                    row_payload["parse_status"] = "invalid"
                    row_payload["warnings"] = list(
                        _string_tuple([*row_payload.get("warnings", ()), warning_text])
                    )
                    warnings.append(warning_text)
                else:
                    parsed_count += 1
                    positions[contract.position_id] = contract
            elif result.parse_status == "incomplete":
                incomplete_count += 1
            else:
                invalid_count += 1
            row_results.append(row_payload)

        load_status = self._load_status(
            parsed_count=parsed_count,
            incomplete_count=incomplete_count,
            invalid_count=invalid_count,
        )
        if not positions and row_results:
            warnings.append("Imported book did not contain any fully parsed positions.")

        imported_book = ImportedBook(positions)
        return ImportedBookLoadResult(
            load_status=load_status,
            position_book={
                name: contract.to_dict()
                for name, contract in imported_book.items()
            },
            row_results=tuple(row_results),
            parsed_count=parsed_count,
            incomplete_count=incomplete_count,
            invalid_count=invalid_count,
            warnings=warnings,
            position_book_object=imported_book,
        )

    def load_positions_csv(
        self,
        path,
        *,
        normalization_profile: str = "canonical",
    ) -> ImportedBookLoadResult:
        """Load one mixed supported book from a CSV file."""
        source_path = Path(path).expanduser()
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Position import CSV must include a header row: {source_path}")
            rows = [self._normalize_loader_row(row) for row in reader]
        return self.load_positions(
            structured_positions=rows,
            normalization_profile=normalization_profile,
        )

    def load_positions_json(
        self,
        path,
        *,
        normalization_profile: str = "canonical",
    ) -> ImportedBookLoadResult:
        """Load one mixed supported book from a JSON row list or positions payload."""
        source_path = Path(path).expanduser()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = payload.get("positions")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("Position import JSON must be a row list or a mapping with `positions`.")
        return self.load_positions(
            structured_positions=[self._normalize_loader_row(row) for row in rows],
            normalization_profile=normalization_profile,
        )

    def _successful_parse_result(
        self,
        *,
        contract,
        semantic_blueprint,
        validation_warnings,
        normalization_profile: str,
        compatibility_surface: str,
    ) -> TradeParseResult:
        from trellis.agent.semantic_contracts import semantic_contract_summary

        summary = semantic_contract_summary(contract)
        return TradeParseResult(
            parse_status="parsed",
            semantic_id=contract.product.semantic_id,
            semantic_version=contract.product.semantic_version,
            trade_type=self._trade_type_for_contract(contract),
            asset_class=self._asset_class_for_contract(contract),
            parsed_contract=summary,
            contract_summary=summary,
            product_ir=self._product_ir_summary(semantic_blueprint.product_ir),
            warnings=validation_warnings,
            candidate_methods=getattr(contract.methods, "candidate_methods", ()),
            required_market_data=tuple(sorted(getattr(semantic_blueprint, "required_market_data", ()) or ())),
            normalization_profile=normalization_profile,
            compatibility_surface=compatibility_surface,
            semantic_contract=contract,
            product_ir_object=semantic_blueprint.product_ir,
            semantic_blueprint=semantic_blueprint,
        )

    @staticmethod
    def _missing_fields_from_error(
        error_text: str,
        *,
        instrument_type: str,
    ) -> tuple[str, ...]:
        """Extract stable missing-field ids from a deterministic parser error."""
        text = str(error_text or "").strip()
        if not text:
            return ()
        if "requires " not in text:
            return ()
        missing_text = text.split("requires ", 1)[1].rstrip(".")
        parts = [part.strip() for part in missing_text.split(",") if part.strip()]
        if parts:
            return tuple(parts)
        if instrument_type in {"range_accrual", "range_accrual_note", "range_note"}:
            if "schedule" in text.lower():
                return ("observation_schedule",)
        return ()

    @staticmethod
    def _structured_coupon_definition(structured_trade: Mapping[str, object]) -> Mapping[str, object]:
        """Return the structured coupon-definition payload for range accruals."""
        coupon_definition = dict(structured_trade.get("coupon_definition") or {})
        if "coupon_rate" not in coupon_definition and structured_trade.get("coupon_rate") is not None:
            coupon_definition["coupon_rate"] = structured_trade.get("coupon_rate")
        if "coupon_style" not in coupon_definition and structured_trade.get("coupon_style") is not None:
            coupon_definition["coupon_style"] = structured_trade.get("coupon_style")
        return coupon_definition

    @staticmethod
    def _structured_range_condition(structured_trade: Mapping[str, object]) -> Mapping[str, object]:
        """Return the structured range-condition payload for range accruals."""
        range_condition = dict(structured_trade.get("range_condition") or {})
        if "lower_bound" not in range_condition and structured_trade.get("lower_bound") is not None:
            range_condition["lower_bound"] = structured_trade.get("lower_bound")
        if "upper_bound" not in range_condition and structured_trade.get("upper_bound") is not None:
            range_condition["upper_bound"] = structured_trade.get("upper_bound")
        if "inclusive_lower" not in range_condition and structured_trade.get("inclusive_lower") is not None:
            range_condition["inclusive_lower"] = structured_trade.get("inclusive_lower")
        if "inclusive_upper" not in range_condition and structured_trade.get("inclusive_upper") is not None:
            range_condition["inclusive_upper"] = structured_trade.get("inclusive_upper")
        return range_condition

    @staticmethod
    def _structured_callability(structured_trade: Mapping[str, object]) -> Mapping[str, object]:
        """Return optional callability hooks for the range-accrual trade-entry slice."""
        callability = dict(structured_trade.get("callability") or {})
        call_schedule = (
            callability.get("call_schedule")
            or structured_trade.get("call_schedule")
            or structured_trade.get("call_dates")
            or ()
        )
        if isinstance(call_schedule, str):
            call_schedule = (call_schedule,)
        call_schedule = tuple(str(item).strip() for item in call_schedule if str(item).strip())
        if not call_schedule:
            return {}
        if "call_schedule" not in callability:
            callability["call_schedule"] = call_schedule
        if "call_style" not in callability:
            callability["call_style"] = structured_trade.get("call_style") or "issuer_callable"
        return callability

    @staticmethod
    def _callable_bond_missing_fields(structured_trade: Mapping[str, object]) -> tuple[str, ...]:
        missing: list[str] = []
        if structured_trade.get("notional") in {None, ""}:
            missing.append("notional")
        if structured_trade.get("coupon") in {None, ""}:
            missing.append("coupon")
        if structured_trade.get("start_date") in {None, ""}:
            missing.append("start_date")
        if structured_trade.get("end_date") in {None, ""}:
            missing.append("end_date")
        return tuple(missing)

    @staticmethod
    def _bermudan_swaption_missing_fields(structured_trade: Mapping[str, object]) -> tuple[str, ...]:
        missing: list[str] = []
        if structured_trade.get("notional") in {None, ""}:
            missing.append("notional")
        if structured_trade.get("strike") in {None, ""}:
            missing.append("strike")
        if structured_trade.get("swap_end") in {None, ""}:
            missing.append("swap_end")
        return tuple(missing)

    @staticmethod
    def _normalized_frequency_name(value, *, field_name: str, default: str) -> str:
        if value in {None, ""}:
            return str(default).strip().upper()
        raw_value = getattr(value, "name", value)
        text = str(raw_value).strip().lower()
        normalized = text.replace("-", "_").replace(" ", "_")
        if normalized.startswith("frequency."):
            normalized = normalized.split(".", 1)[1]
        aliases = {
            "annual": "ANNUAL",
            "annually": "ANNUAL",
            "semi_annual": "SEMI_ANNUAL",
            "semiannual": "SEMI_ANNUAL",
            "quarterly": "QUARTERLY",
            "monthly": "MONTHLY",
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise ValueError(
                f"Structured trade field `{field_name}` has unsupported frequency {value!r}."
            )
        return resolved

    @staticmethod
    def _normalized_day_count_name(value, *, field_name: str, default: str) -> str:
        if value in {None, ""}:
            return str(default).strip().upper()
        raw_value = getattr(value, "name", value)
        text = str(raw_value).strip().lower()
        normalized = text.replace("-", "_").replace("/", "_").replace(" ", "_")
        if normalized.startswith("daycountconvention."):
            normalized = normalized.split(".", 1)[1]
        aliases = {
            "act_360": "ACT_360",
            "act360": "ACT_360",
            "actual_360": "ACT_360",
            "act_365": "ACT_365",
            "act365": "ACT_365",
            "actual_365": "ACT_365",
            "act_365_fixed": "ACT_365",
            "act_act": "ACT_ACT",
            "act_act_isda": "ACT_ACT",
            "actual_actual": "ACT_ACT",
            "30_360": "THIRTY_360",
            "thirty_360": "THIRTY_360",
            "30_360_us": "THIRTY_360",
            "thirty_360_us": "THIRTY_360",
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise ValueError(
                f"Structured trade field `{field_name}` has unsupported day_count {value!r}."
            )
        return resolved

    def _structured_callable_bond_term_fields(
        self,
        structured_trade: Mapping[str, object],
        *,
        schedule: tuple[str, ...],
    ) -> Mapping[str, object] | None:
        if self._callable_bond_missing_fields(structured_trade):
            return None
        return _freeze_mapping(
            {
                "notional": float(structured_trade.get("notional")),
                "coupon": float(structured_trade.get("coupon")),
                "start_date": str(structured_trade.get("start_date")).strip(),
                "end_date": str(structured_trade.get("end_date")).strip(),
                "call_price": float(structured_trade.get("call_price", 100.0) or 100.0),
                "frequency": self._normalized_frequency_name(
                    structured_trade.get("frequency"),
                    field_name="frequency",
                    default="SEMI_ANNUAL",
                ),
                "day_count": self._normalized_day_count_name(
                    structured_trade.get("day_count"),
                    field_name="day_count",
                    default="ACT_365",
                ),
                "exercise_schedule": list(schedule),
            }
        )

    def _structured_bermudan_swaption_term_fields(
        self,
        structured_trade: Mapping[str, object],
        *,
        schedule: tuple[str, ...],
    ) -> Mapping[str, object] | None:
        if self._bermudan_swaption_missing_fields(structured_trade):
            return None
        is_payer = structured_trade.get("is_payer", True)
        if isinstance(is_payer, str):
            normalized = is_payer.strip().lower()
            is_payer = normalized not in {"false", "0", "no", "receiver"}
        return _freeze_mapping(
            {
                "notional": float(structured_trade.get("notional")),
                "strike": float(structured_trade.get("strike")),
                "swap_end": str(structured_trade.get("swap_end")).strip(),
                "swap_frequency": self._normalized_frequency_name(
                    structured_trade.get("swap_frequency"),
                    field_name="swap_frequency",
                    default="SEMI_ANNUAL",
                ),
                "day_count": self._normalized_day_count_name(
                    structured_trade.get("day_count"),
                    field_name="day_count",
                    default="ACT_360",
                ),
                "is_payer": bool(is_payer),
                "exercise_schedule": list(schedule),
            }
        )

    def _structured_semantic_contract(
        self,
        structured_trade: Mapping[str, object],
        *,
        description: str | None,
        instrument_type: str | None,
    ):
        from dataclasses import replace

        from trellis.agent.semantic_contracts import (
            make_callable_bond_contract,
            make_credit_default_swap_contract,
            make_nth_to_default_contract,
            make_quanto_option_contract,
            make_range_accrual_contract,
            make_ranked_observation_basket_contract,
            make_rate_style_swaption_contract,
            make_vanilla_option_contract,
            parse_semantic_contract,
        )

        if any(
            key in structured_trade
            for key in ("product", "market_data", "methods", "blueprint")
        ):
            try:
                return parse_semantic_contract(structured_trade)
            except Exception:
                return None

        normalized_type = self._instrument_type(
            structured_trade.get("instrument_type") or instrument_type
        )
        payload_description = (
            str(structured_trade.get("description") or "").strip()
            or str(description or "").strip()
            or normalized_type
        )
        schedule = self._structured_schedule(
            structured_trade,
            instrument_type=normalized_type,
        )
        term_fields: Mapping[str, object] = {}
        if normalized_type in {"european_option", "vanilla_option", "option"}:
            underliers = structured_trade.get("underliers") or structured_trade.get("constituents") or ()
            try:
                contract = make_vanilla_option_contract(
                    description=payload_description,
                    underliers=underliers,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "analytical")).strip() or "analytical",
                )
            except ValueError:
                return None
        elif normalized_type in {"quanto_option"}:
            underliers = structured_trade.get("underliers") or structured_trade.get("constituents") or ()
            try:
                contract = make_quanto_option_contract(
                    description=payload_description,
                    underliers=underliers,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "analytical")).strip() or "analytical",
                )
            except ValueError:
                return None
        elif normalized_type in {"swaption", "bermudan_swaption", "rate_style_swaption"}:
            exercise_style = "bermudan" if normalized_type == "bermudan_swaption" else str(
                structured_trade.get("exercise_style", "european")
            ).strip() or "european"
            if exercise_style == "bermudan":
                term_fields = self._structured_bermudan_swaption_term_fields(
                    structured_trade,
                    schedule=schedule,
                )
                if term_fields is None:
                    return None
            try:
                contract = make_rate_style_swaption_contract(
                    description=payload_description,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "analytical")).strip() or "analytical",
                    exercise_style=exercise_style,
                )
            except ValueError:
                return None
        elif normalized_type in {"callable_bond", "callable_debt"}:
            term_fields = self._structured_callable_bond_term_fields(
                structured_trade,
                schedule=schedule,
            )
            if term_fields is None:
                return None
            try:
                contract = make_callable_bond_contract(
                    description=payload_description,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "rate_tree")).strip() or "rate_tree",
                )
            except ValueError:
                return None
        elif normalized_type in {
            "range_accrual",
            "range_accrual_note",
            "range_note",
            "callable_range_note",
            "callable_range_accrual",
        }:
            try:
                contract = make_range_accrual_contract(
                    description=payload_description,
                    reference_index=str(
                        structured_trade.get("reference_index")
                        or structured_trade.get("underlier_index")
                        or structured_trade.get("underlier")
                        or ""
                    ).strip(),
                    observation_schedule=schedule,
                    coupon_definition=self._structured_coupon_definition(structured_trade),
                    range_condition=self._structured_range_condition(structured_trade),
                    settlement_profile=structured_trade.get("settlement_profile"),
                    callability=self._structured_callability(structured_trade),
                    preferred_method=str(structured_trade.get("preferred_method", "analytical")).strip() or "analytical",
                )
            except ValueError:
                return None
        elif normalized_type in {"cds", "credit_default_swap"}:
            try:
                contract = make_credit_default_swap_contract(
                    description=payload_description,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "analytical")).strip() or "analytical",
                    reference_entities=structured_trade.get("reference_entities") or (),
                )
            except ValueError:
                return None
        elif normalized_type in {"nth_to_default", "basket_cds"}:
            try:
                contract = make_nth_to_default_contract(
                    description=payload_description,
                    observation_schedule=schedule,
                    reference_entities=structured_trade.get("reference_entities") or (),
                    trigger_rank=int(structured_trade.get("trigger_rank", 1) or 1),
                    preferred_method=str(structured_trade.get("preferred_method", "copula")).strip() or "copula",
                )
            except ValueError:
                return None
        elif normalized_type in {"basket_option", "ranked_observation_basket", "basket_path_payoff"}:
            try:
                contract = make_ranked_observation_basket_contract(
                    description=payload_description,
                    constituents=structured_trade.get("constituents") or structured_trade.get("underliers") or (),
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "monte_carlo")).strip() or "monte_carlo",
                )
            except ValueError:
                return None
        else:
            return None

        payment_currency = str(structured_trade.get("payout_currency", "")).strip()
        reporting_currency = str(structured_trade.get("reporting_currency", "")).strip()
        if payment_currency or reporting_currency:
            product = replace(
                contract.product,
                conventions=replace(
                    contract.product.conventions,
                    payment_currency=payment_currency,
                    reporting_currency=reporting_currency,
                ),
            )
            contract = replace(contract, product=product)
        if term_fields:
            contract = replace(
                contract,
                product=replace(
                    contract.product,
                    term_fields=term_fields,
                ),
            )
        return contract

    def _structured_missing_fields(
        self,
        structured_trade: Mapping[str, object],
        *,
        instrument_type: str | None,
    ) -> tuple[str, ...]:
        normalized_type = self._instrument_type(
            structured_trade.get("instrument_type") or instrument_type
        )
        schedule = self._structured_schedule(
            structured_trade,
            instrument_type=normalized_type,
        )
        missing: list[str] = []
        if normalized_type in {"european_option", "vanilla_option", "option", "quanto_option"}:
            if not (structured_trade.get("underliers") or structured_trade.get("constituents")):
                missing.append("underliers")
            if not schedule:
                missing.append("observation_schedule")
        elif normalized_type in {"swaption", "rate_style_swaption", "cds", "credit_default_swap"}:
            if not schedule:
                missing.append("observation_schedule")
        elif normalized_type == "bermudan_swaption":
            if not schedule:
                missing.append("observation_schedule")
            missing.extend(self._bermudan_swaption_missing_fields(structured_trade))
        elif normalized_type in {"callable_bond", "callable_debt"}:
            if not schedule:
                missing.append("observation_schedule")
            missing.extend(self._callable_bond_missing_fields(structured_trade))
        elif normalized_type in {
            "range_accrual",
            "range_accrual_note",
            "range_note",
            "callable_range_note",
            "callable_range_accrual",
        }:
            if not (structured_trade.get("reference_index") or structured_trade.get("underlier_index") or structured_trade.get("underlier")):
                missing.append("reference_index")
            if not self._structured_coupon_definition(structured_trade):
                missing.append("coupon_definition")
            if not self._structured_range_condition(structured_trade):
                missing.append("range_condition")
            if not schedule:
                missing.append("observation_schedule")
        elif normalized_type in {"nth_to_default", "basket_cds"}:
            if not schedule:
                missing.append("observation_schedule")
            if not structured_trade.get("reference_entities"):
                missing.append("reference_entities")
        elif normalized_type in {"basket_option", "ranked_observation_basket", "basket_path_payoff"}:
            if not (structured_trade.get("constituents") or structured_trade.get("underliers")):
                missing.append("constituents")
            if not schedule:
                missing.append("observation_schedule")
        else:
            missing.append("instrument_type")
        return tuple(missing)

    @staticmethod
    def _structured_schedule(
        structured_trade: Mapping[str, object],
        *,
        instrument_type: str | None = None,
    ) -> tuple[str, ...]:
        normalized_type = str(
            instrument_type
            or structured_trade.get("instrument_type")
            or ""
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_type in {"callable_bond", "callable_debt"}:
            keys = (
                "call_schedule",
                "call_dates",
                "issuer_call_dates",
                "observation_schedule",
                "observation_dates",
            )
        elif normalized_type in {
            "range_accrual",
            "range_accrual_note",
            "range_note",
            "callable_range_note",
            "callable_range_accrual",
        }:
            keys = (
                "observation_schedule",
                "observation_dates",
                "fixing_schedule",
                "fixing_dates",
            )
        elif normalized_type in {"swaption", "bermudan_swaption", "rate_style_swaption"}:
            keys = (
                "exercise_schedule",
                "exercise_dates",
                "exercise_date",
                "observation_schedule",
                "observation_dates",
                "expiry_date",
                "expiry",
            )
        elif normalized_type in {"european_option", "vanilla_option", "option", "quanto_option"}:
            keys = (
                "observation_schedule",
                "observation_dates",
                "expiry_date",
                "expiry",
                "exercise_date",
            )
        else:
            keys = (
                "observation_schedule",
                "observation_dates",
                "fixing_schedule",
                "fixing_dates",
                "exercise_schedule",
                "exercise_dates",
                "exercise_date",
                "expiry_date",
                "expiry",
            )

        schedule = ()
        for key in keys:
            value = structured_trade.get(key)
            if value is not None and value != "":
                schedule = value
                break
        if isinstance(schedule, str):
            return (schedule,)
        if schedule and not isinstance(schedule, (list, tuple, set, frozenset)):
            return (str(schedule).strip(),)
        return tuple(str(item).strip() for item in schedule if str(item).strip())

    @staticmethod
    def _instrument_type(value) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _asset_class_for_instrument(instrument_type: str | None) -> str:
        normalized = str(instrument_type or "").strip().lower()
        if normalized in {
            "swaption",
            "bermudan_swaption",
            "rate_style_swaption",
            "callable_bond",
            "range_accrual",
            "range_accrual_note",
            "range_note",
            "callable_range_note",
            "callable_range_accrual",
        }:
            return "rates"
        if normalized in {"cds", "credit_default_swap", "nth_to_default", "basket_cds"}:
            return "credit"
        if normalized in {"quanto_option"}:
            return "fx"
        if normalized:
            return "equity"
        return ""

    def _asset_class_for_contract(self, contract) -> str:
        model_family = str(getattr(contract.product, "model_family", "")).strip().lower()
        underlier_structure = str(getattr(contract.product, "underlier_structure", "")).strip().lower()
        if "interest_rate" in model_family or "single_curve_rate_style" == underlier_structure:
            return "rates"
        if model_family.startswith("credit"):
            return "credit"
        if "cross_currency" in underlier_structure:
            return "fx"
        return "equity"

    @staticmethod
    def _trade_type_for_contract(contract) -> str:
        instrument_class = str(getattr(contract.product, "instrument_class", "")).strip()
        exercise_style = str(getattr(contract.product, "exercise_style", "")).strip().lower()
        if instrument_class == "swaption" and exercise_style == "bermudan":
            return "bermudan_swaption"
        return instrument_class

    @staticmethod
    def _product_ir_summary(product_ir: ProductIR) -> dict[str, object]:
        return {
            "instrument": product_ir.instrument,
            "payoff_family": product_ir.payoff_family,
            "payoff_traits": list(product_ir.payoff_traits),
            "exercise_style": product_ir.exercise_style,
            "state_dependence": product_ir.state_dependence,
            "schedule_dependence": product_ir.schedule_dependence,
            "model_family": product_ir.model_family,
            "candidate_engine_families": list(product_ir.candidate_engine_families),
            "route_families": list(product_ir.route_families),
            "required_market_data": sorted(product_ir.required_market_data),
            "reusable_primitives": list(product_ir.reusable_primitives),
            "unresolved_primitives": list(product_ir.unresolved_primitives),
            "supported": product_ir.supported,
        }

    @staticmethod
    def _position_field_value(
        payload: Mapping[str, object],
        aliases: tuple[str, ...],
    ) -> tuple[object | None, str | None]:
        for alias in aliases:
            if alias in payload:
                return payload.get(alias), alias
        return None, None

    def _position_trade_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, object], str, str | None]:
        nested_value, nested_key = self._position_field_value(
            payload,
            self.POSITION_FIELD_ALIASES["structured_trade"],
        )
        if nested_key is not None:
            if not isinstance(nested_value, Mapping):
                return {}, nested_key, "Position import requires structured_trade/trade/terms to be a mapping."
            return dict(nested_value), nested_key, None

        excluded_keys = set().union(*self.POSITION_FIELD_ALIASES.values())
        trade_payload = {
            str(key): value
            for key, value in payload.items()
            if key not in excluded_keys
        }
        return trade_payload, "top_level_trade_fields", None

    def _normalize_loader_row(self, row) -> dict[str, object]:
        if not isinstance(row, Mapping):
            return row
        normalized: dict[str, object] = {}
        for key, value in row.items():
            field = str(key or "").strip()
            if not field:
                continue
            normalized_value = self._normalize_loader_value(value, field=field)
            if normalized_value is None:
                continue
            normalized[field] = normalized_value
        return normalized

    def _normalize_loader_value(self, value, *, field: str):
        if isinstance(value, Mapping):
            return {
                str(key).strip(): normalized
                for key, item in value.items()
                if str(key).strip()
                for normalized in [self._normalize_loader_value(item, field=str(key))]
                if normalized is not None
            }
        if isinstance(value, (list, tuple)):
            return [
                normalized
                for item in value
                for normalized in [self._normalize_loader_value(item, field=field)]
                if normalized is not None
            ]
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return None
        if field in self.POSITION_MAPPING_FIELDS and text[0] in "[{":
            try:
                return self._normalize_loader_value(json.loads(text), field=field)
            except json.JSONDecodeError:
                return text
        if field in self.POSITION_SEQUENCE_FIELDS:
            if text[0] in "[{":
                try:
                    return self._normalize_loader_value(json.loads(text), field=field)
                except json.JSONDecodeError:
                    pass
            delimiter = "|" if "|" in text else None
            if delimiter is None and "," in text and field in {"tags", "labels"}:
                delimiter = ","
            if delimiter is not None:
                return [
                    normalized
                    for part in text.split(delimiter)
                    for normalized in [self._normalize_loader_value(part, field=field)]
                    if normalized is not None
                ]
        if text[0] in "[{":
            try:
                return self._normalize_loader_value(json.loads(text), field=field)
            except json.JSONDecodeError:
                pass
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if self._looks_numeric(text):
            if any(char in text for char in (".", "e", "E")):
                return float(text)
            return int(text)
        return text

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text.count("-") > 1 or (text.startswith("-") is False and "-" in text):
            return False
        if any(char.isalpha() for char in text.replace("e", "").replace("E", "")):
            return False
        try:
            float(text)
        except ValueError:
            return False
        return True

    @staticmethod
    def _load_status(
        *,
        parsed_count: int,
        incomplete_count: int,
        invalid_count: int,
    ) -> str:
        if parsed_count and not incomplete_count and not invalid_count:
            return "parsed"
        if parsed_count:
            return "partial"
        if invalid_count and not incomplete_count:
            return "invalid"
        if incomplete_count and not invalid_count:
            return "incomplete"
        if incomplete_count or invalid_count:
            return "invalid"
        return "empty"


__all__ = [
    "TradeParseResult",
    "TradeService",
]
