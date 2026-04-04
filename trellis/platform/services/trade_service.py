"""Transport-neutral typed trade parsing for governed workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from trellis.agent.knowledge.schema import ProductIR


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
        from trellis.agent.semantic_contracts import draft_semantic_contract, semantic_contract_summary

        warnings: list[str] = []
        profile = str(normalization_profile or "canonical").strip() or "canonical"
        if profile not in {"canonical", "strict"}:
            warnings.append(
                f"Normalization profile {profile!r} is unsupported and was treated as canonical."
            )
            profile = "canonical"

        if structured_trade is not None:
            maybe_contract = self._structured_semantic_contract(
                structured_trade,
                description=description,
                instrument_type=instrument_type,
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
        semantic_contract = draft_semantic_contract(
            text,
            instrument_type=normalized_instrument_type or None,
        )
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
            trade_type=contract.product.instrument_class,
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
        schedule = self._structured_schedule(structured_trade)
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
            try:
                contract = make_callable_bond_contract(
                    description=payload_description,
                    observation_schedule=schedule,
                    preferred_method=str(structured_trade.get("preferred_method", "rate_tree")).strip() or "rate_tree",
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
        schedule = self._structured_schedule(structured_trade)
        missing: list[str] = []
        if normalized_type in {"european_option", "vanilla_option", "option", "quanto_option"}:
            if not (structured_trade.get("underliers") or structured_trade.get("constituents")):
                missing.append("underliers")
            if not schedule:
                missing.append("observation_schedule")
        elif normalized_type in {"swaption", "bermudan_swaption", "rate_style_swaption", "callable_bond", "callable_debt", "cds", "credit_default_swap"}:
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
    def _structured_schedule(structured_trade: Mapping[str, object]) -> tuple[str, ...]:
        schedule = (
            structured_trade.get("observation_schedule")
            or structured_trade.get("exercise_schedule")
            or structured_trade.get("call_dates")
            or structured_trade.get("expiry")
            or ()
        )
        if isinstance(schedule, str):
            return (schedule,)
        return tuple(str(item).strip() for item in schedule if str(item).strip())

    @staticmethod
    def _instrument_type(value) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _asset_class_for_instrument(instrument_type: str | None) -> str:
        normalized = str(instrument_type or "").strip().lower()
        if normalized in {"swaption", "bermudan_swaption", "rate_style_swaption", "callable_bond"}:
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


__all__ = [
    "TradeParseResult",
    "TradeService",
]
