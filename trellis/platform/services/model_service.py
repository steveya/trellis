"""Deterministic governed model matching over the canonical registry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from trellis.platform.models import (
    ModelLifecycleStatus,
    ModelLineage,
    ModelRecord,
    ModelRegistryStore,
    ModelVersionRecord,
)
from trellis.platform.services.trade_service import TradeParseResult


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


def _status_rank(status: str) -> int:
    """Return a deterministic lifecycle preference rank."""
    return {
        "approved": 3,
        "validated": 2,
        "draft": 1,
        "deprecated": 0,
    }.get(str(status or "").strip().lower(), -1)


@dataclass(frozen=True)
class ModelMatchResult:
    """Stable deterministic model-match payload."""

    match_type: str
    match_basis: Mapping[str, object] = field(default_factory=dict)
    selected_candidate: Mapping[str, object] = field(default_factory=dict)
    candidates: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "match_type", str(self.match_type or "").strip())
        object.__setattr__(self, "match_basis", _freeze_mapping(self.match_basis))
        object.__setattr__(self, "selected_candidate", _freeze_mapping(self.selected_candidate))
        object.__setattr__(
            self,
            "candidates",
            tuple(_freeze_mapping(candidate) for candidate in (self.candidates or ())),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe machine-readable payload."""
        return {
            "match_type": self.match_type,
            "match_basis": dict(self.match_basis),
            "selected_candidate": dict(self.selected_candidate),
            "candidates": [dict(candidate) for candidate in self.candidates],
        }


class ModelService:
    """Deterministic structural matching against the governed model registry."""

    def __init__(self, *, registry: ModelRegistryStore):
        self.registry = registry

    def match_trade(self, parsed_trade: TradeParseResult) -> ModelMatchResult:
        """Match one parsed trade against the deterministic governed registry."""
        match_basis = self._match_basis(parsed_trade)
        if parsed_trade.parse_status != "parsed":
            return ModelMatchResult(
                match_type="no_match",
                match_basis=match_basis,
                selected_candidate={},
                candidates=(),
            )
        candidates = tuple(self._evaluate_candidate(record, match_basis) for record in self.registry.list_models())
        compatible = [
            candidate
            for candidate in candidates
            if candidate["match_status"] == "compatible"
        ]
        compatible.sort(
            key=lambda item: (
                _status_rank(item["status"]),
                len(item["matched_fields"]),
                item["model_id"],
            ),
            reverse=True,
        )
        selected = compatible[0] if compatible else {}
        if not selected:
            match_type = "no_match"
        else:
            status = str(selected.get("status", "")).strip()
            if status == ModelLifecycleStatus.APPROVED.value:
                match_type = "exact_approved_match"
            elif status == ModelLifecycleStatus.VALIDATED.value:
                match_type = "exact_validated_match"
            else:
                match_type = "structurally_compatible_not_execution_eligible"
        return ModelMatchResult(
            match_type=match_type,
            match_basis=match_basis,
            selected_candidate=selected,
            candidates=tuple(candidates),
        )

    def explain_match(self, parsed_trade: TradeParseResult) -> ModelMatchResult:
        """Explain the same deterministic checks used by the matcher."""
        return self.match_trade(parsed_trade)

    def generate_candidate(
        self,
        parsed_trade: TradeParseResult,
        *,
        model_id: str | None = None,
        version: str | None = None,
        method_family: str | None = None,
        engine_binding: Mapping[str, object] | None = None,
        assumptions=(),
        tags=(),
        implementation_source: str | None = None,
        module_path: str | None = None,
        validation_plan: Mapping[str, object] | None = None,
        actor: str = "mcp",
        reason: str = "generate_candidate",
        notes: str = "",
        lineage: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Persist one draft governed candidate model version from a typed trade."""
        self._require_parsed_trade(parsed_trade)
        match_basis = self._match_basis(parsed_trade)
        resolved_model_id = str(model_id or self._default_model_id(parsed_trade, method_family)).strip()
        resolved_version = str(version or self._next_version(resolved_model_id)).strip()
        resolved_method_family = str(
            method_family or match_basis.get("method_family") or "unspecified"
        ).strip()
        stored_model = self._ensure_model_record(
            parsed_trade,
            model_id=resolved_model_id,
            method_family=resolved_method_family,
            tags=tags,
            metadata=metadata,
        )
        lineage_record = (
            lineage if isinstance(lineage, ModelLineage) else ModelLineage.from_dict(lineage or {})
        )
        methodology_summary = {
            "method_family": resolved_method_family,
            "candidate_source": "mcp",
        }
        version_record = ModelVersionRecord(
            model_id=stored_model.model_id,
            version=resolved_version,
            status=ModelLifecycleStatus.DRAFT,
            contract_summary=dict(parsed_trade.contract_summary),
            methodology_summary=methodology_summary,
            assumptions=assumptions,
            engine_binding=dict(engine_binding or {}),
            lineage=lineage_record,
            artifacts={},
            metadata={
                "candidate_source": "mcp",
                "reason": reason,
                **dict(metadata or {}),
            },
        )
        stored_version = self.registry.create_version(
            version_record,
            actor=actor,
            reason=reason,
            notes=notes,
            metadata=metadata,
        )
        artifact_uris = self._persist_candidate_artifacts(
            stored_version=stored_version,
            parsed_trade=parsed_trade,
            methodology_summary=methodology_summary,
            implementation_source=implementation_source,
            module_path=module_path,
            validation_plan=validation_plan,
        )
        stored_version = self.registry.save_version(
            replace(
                stored_version,
                artifacts={**dict(stored_version.artifacts), **artifact_uris, "module_path": str(module_path or "").strip()},
            )
        )
        stored_model = self.registry.get_model(stored_model.model_id) or stored_model
        return {
            "model": stored_model.to_dict(),
            "version": stored_version.to_dict(),
            "artifact_uris": {
                "contract": artifact_uris.get("contract_uri", ""),
                "code": artifact_uris.get("code_uri", ""),
                "methodology": artifact_uris.get("methodology_uri", ""),
                "validation_plan": artifact_uris.get("validation_plan_uri", ""),
                "lineage": artifact_uris.get("lineage_uri", ""),
            },
        }

    def promote_version(
        self,
        *,
        model_id: str,
        version: str,
        to_status: str,
        actor: str,
        reason: str,
        notes: str = "",
        metadata: Mapping[str, object] | None = None,
        validation_store=None,
    ) -> dict[str, object]:
        """Apply one explicit governed lifecycle transition to a stored version."""
        from trellis.mcp.errors import TrellisMcpError

        current = self.registry.get_version(model_id, version)
        if current is None:
            raise TrellisMcpError(
                code="unknown_model_version",
                message=f"Unknown model version: {model_id}:{version}",
                details={"model_id": model_id, "version": version},
            )
        normalized_status = ModelLifecycleStatus.normalize(to_status)
        if normalized_status in {ModelLifecycleStatus.VALIDATED, ModelLifecycleStatus.APPROVED}:
            if validation_store is not None:
                latest_validation = validation_store.latest_validation(
                    model_id=model_id,
                    version=version,
                )
                latest_status = "" if latest_validation is None else latest_validation.status
                if latest_validation is None or latest_status != "passed":
                    raise TrellisMcpError(
                        code="validation_required",
                        message=(
                            f"Lifecycle transition to {normalized_status.value!r} requires the latest deterministic validation "
                            "record for this exact version to be passed."
                        ),
                        details={
                            "model_id": model_id,
                            "version": version,
                            "to_status": normalized_status.value,
                            "latest_validation_status": latest_status,
                        },
                    )
            elif not current.validation_summary.get("all_checks_passed"):
                raise TrellisMcpError(
                    code="validation_required",
                    message=(
                        f"Lifecycle transition to {normalized_status.value!r} requires a passed deterministic validation report."
                    ),
                    details={"model_id": model_id, "version": version, "to_status": normalized_status.value},
                )

        try:
            updated = self.registry.transition_version(
                model_id,
                version,
                normalized_status,
                actor=actor,
                reason=reason,
                notes=notes,
                metadata=metadata,
            )
        except Exception as exc:
            raise TrellisMcpError(
                code="invalid_lifecycle_transition",
                message=str(exc),
                details={"model_id": model_id, "version": version, "to_status": normalized_status.value},
            ) from exc
        model = self.registry.get_model(model_id)
        return {
            "model": {} if model is None else model.to_dict(),
            "version": updated.to_dict(),
        }

    def persist_version(
        self,
        *,
        model_id: str,
        base_version: str | None = None,
        new_version: str | None = None,
        actor: str,
        reason: str,
        notes: str = "",
        contract_summary: Mapping[str, object] | None = None,
        methodology_summary: Mapping[str, object] | None = None,
        assumptions=(),
        engine_binding: Mapping[str, object] | None = None,
        validation_summary: Mapping[str, object] | None = None,
        validation_refs=(),
        artifacts: Mapping[str, object] | None = None,
        implementation_source: str | None = None,
        module_path: str | None = None,
        validation_plan: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Persist one explicit governed model version, optionally derived from another."""
        from trellis.mcp.errors import TrellisMcpError

        model = self.registry.get_model(model_id)
        if model is None:
            raise TrellisMcpError(
                code="unknown_model",
                message=f"Unknown model id: {model_id!r}",
                details={"model_id": model_id},
            )
        base = None if not base_version else self.registry.get_version(model_id, base_version)
        if base_version and base is None:
            raise TrellisMcpError(
                code="unknown_model_version",
                message=f"Unknown model version: {model_id}:{base_version}",
                details={"model_id": model_id, "version": base_version},
            )
        version = str(new_version or self._next_version(model_id)).strip()
        lineage_record = self._lineage_for_persisted_version(base, lineage)
        resolved_contract_summary = dict(contract_summary or getattr(base, "contract_summary", {}) or {})
        resolved_methodology_summary = dict(
            methodology_summary or getattr(base, "methodology_summary", {}) or {}
        )
        resolved_validation_summary = dict(validation_summary or {})
        resolved_validation_refs = tuple(validation_refs or ())
        stored = self.registry.create_version(
            ModelVersionRecord(
                model_id=model_id,
                version=version,
                status=ModelLifecycleStatus.DRAFT,
                contract_summary=resolved_contract_summary,
                methodology_summary=resolved_methodology_summary,
                assumptions=assumptions or getattr(base, "assumptions", ()),
                engine_binding=dict(engine_binding or getattr(base, "engine_binding", {}) or {}),
                validation_summary=resolved_validation_summary,
                validation_refs=resolved_validation_refs,
                lineage=lineage_record,
                artifacts=dict(artifacts or {}),
                metadata={"persist_reason": reason, **dict(metadata or {})},
            ),
            actor=actor,
            reason=reason,
            notes=notes,
            metadata=metadata,
        )
        artifact_uris = self._persist_version_artifacts(
            stored_version=stored,
            contract_summary=resolved_contract_summary,
            methodology_summary=resolved_methodology_summary,
            implementation_source=implementation_source,
            validation_plan=validation_plan,
            lineage_record=lineage_record,
            base_version=base,
            module_path=module_path,
        )
        stored = self.registry.save_version(
            replace(
                stored,
                artifacts={**dict(stored.artifacts), **artifact_uris},
            )
        )
        model = self.registry.get_model(model_id) or model
        return {
            "model": model.to_dict(),
            "version": stored.to_dict(),
            "artifact_uris": {
                "contract": artifact_uris.get("contract_uri", ""),
                "code": artifact_uris.get("code_uri", ""),
                "methodology": artifact_uris.get("methodology_uri", ""),
                "validation_plan": artifact_uris.get("validation_plan_uri", ""),
                "lineage": artifact_uris.get("lineage_uri", ""),
            },
        }

    def list_version_history(self, *, model_id: str) -> dict[str, object]:
        """Return the governed version history for one stored model."""
        from trellis.mcp.errors import TrellisMcpError

        model = self.registry.get_model(model_id)
        if model is None:
            raise TrellisMcpError(
                code="unknown_model",
                message=f"Unknown model id: {model_id!r}",
                details={"model_id": model_id},
            )
        return {
            "model": model.to_dict(),
            "versions": [record.to_dict() for record in self.registry.list_versions(model_id)],
        }

    def diff_versions(self, *, model_id: str, left_version: str, right_version: str) -> dict[str, object]:
        """Return the governed comparison surface for two persisted model versions."""
        from trellis.mcp.errors import TrellisMcpError

        try:
            diff = self.registry.diff_versions(model_id, left_version, right_version)
        except FileNotFoundError as exc:
            raise TrellisMcpError(
                code="unknown_model_version",
                message=str(exc),
                details={"model_id": model_id, "left_version": left_version, "right_version": right_version},
            ) from exc
        return {"diff": diff}

    def _evaluate_candidate(self, record, match_basis: Mapping[str, object]) -> dict[str, object]:
        selected_version = self._selected_version(record)
        matched_fields: list[str] = []
        rejections: list[str] = []

        self._compare_field(
            rejections,
            matched_fields,
            field_name="semantic_id",
            expected=str(match_basis.get("semantic_id", "")),
            actual=record.semantic_id,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="semantic_version",
            expected=str(match_basis.get("semantic_version", "")),
            actual=record.semantic_version,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="product_family",
            expected=str(match_basis.get("product_family", "")),
            actual=record.product_family,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="instrument_class",
            expected=str(match_basis.get("instrument_class", "")),
            actual=record.instrument_class,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="payoff_family",
            expected=str(match_basis.get("payoff_family", "")),
            actual=record.payoff_family,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="exercise_style",
            expected=str(match_basis.get("exercise_style", "")),
            actual=record.exercise_style,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="underlier_structure",
            expected=str(match_basis.get("underlier_structure", "")),
            actual=record.underlier_structure,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="payout_currency",
            expected=str(match_basis.get("payout_currency", "")),
            actual=record.payout_currency,
        )
        self._compare_field(
            rejections,
            matched_fields,
            field_name="reporting_currency",
            expected=str(match_basis.get("reporting_currency", "")),
            actual=record.reporting_currency,
        )
        if any(
            code in rejections
            for code in ("payout_currency_mismatch", "reporting_currency_mismatch")
        ) and "currency_mismatch" not in rejections:
            rejections.append("currency_mismatch")
        required_market_data = tuple(match_basis.get("required_market_data") or ())
        if required_market_data:
            if not set(required_market_data).issubset(set(record.required_market_data)):
                rejections.append("required_market_data_mismatch")
            else:
                matched_fields.append("required_market_data")
        method_family = str(match_basis.get("method_family", "")).strip()
        if method_family:
            if record.supported_method_families and method_family not in record.supported_method_families:
                rejections.append("method_family_mismatch")
            else:
                matched_fields.append("method_family")

        status = str(selected_version.get("status", record.status.value)).strip()
        return {
            "model_id": record.model_id,
            "version": str(selected_version.get("version", "")).strip(),
            "status": status,
            "match_status": "compatible" if not rejections else "rejected",
            "execution_eligible": status in {"approved", "validated"},
            "matched_fields": tuple(matched_fields),
            "rejections": tuple(rejections),
            "engine_binding": dict(selected_version.get("engine_binding", {})),
            "methodology_summary": dict(selected_version.get("methodology_summary", {})),
            "match_basis": record.match_basis(),
        }

    @staticmethod
    def _require_parsed_trade(parsed_trade: TradeParseResult) -> None:
        from trellis.mcp.errors import TrellisMcpError

        if parsed_trade.parse_status == "parsed":
            return
        raise TrellisMcpError(
            code="trade_parse_incomplete",
            message="Candidate generation requires one fully parsed trade contract.",
            details={"missing_fields": list(parsed_trade.missing_fields)},
        )

    def _ensure_model_record(
        self,
        parsed_trade: TradeParseResult,
        *,
        model_id: str,
        method_family: str,
        tags=(),
        metadata: Mapping[str, object] | None = None,
    ) -> ModelRecord:
        match_basis = self._match_basis(parsed_trade)
        existing = self.registry.get_model(model_id)
        if existing is None:
            return self.registry.create_model(
                ModelRecord(
                    model_id=model_id,
                    semantic_id=str(match_basis.get("semantic_id", "")).strip(),
                    semantic_version=str(match_basis.get("semantic_version", "")).strip(),
                    product_family=str(match_basis.get("product_family", "")).strip(),
                    instrument_class=str(match_basis.get("instrument_class", "")).strip(),
                    payoff_family=str(match_basis.get("payoff_family", "")).strip(),
                    exercise_style=str(match_basis.get("exercise_style", "")).strip(),
                    underlier_structure=str(match_basis.get("underlier_structure", "")).strip(),
                    payout_currency=str(match_basis.get("payout_currency", "")).strip(),
                    reporting_currency=str(match_basis.get("reporting_currency", "")).strip(),
                    required_market_data=tuple(match_basis.get("required_market_data") or ()),
                    supported_method_families=(method_family,),
                    tags=tuple(tags or ()),
                    metadata=dict(metadata or {}),
                )
            )
        if existing.semantic_id != str(match_basis.get("semantic_id", "")).strip():
            from trellis.mcp.errors import TrellisMcpError

            raise TrellisMcpError(
                code="model_identity_conflict",
                message=(
                    f"Model id {model_id!r} already exists with semantic id {existing.semantic_id!r}."
                ),
                details={"model_id": model_id},
            )
        supported = tuple(
            sorted(
                {
                    *existing.supported_method_families,
                    *(item for item in (method_family,) if item),
                }
            )
        )
        merged_tags = tuple(sorted({*existing.tags, *(str(tag).strip() for tag in tags or () if str(tag).strip())}))
        updated = replace(
            existing,
            supported_method_families=supported,
            tags=merged_tags,
            metadata={**dict(existing.metadata), **dict(metadata or {})},
        )
        return self.registry.save_model(updated)

    def _persist_candidate_artifacts(
        self,
        *,
        stored_version: ModelVersionRecord,
        parsed_trade: TradeParseResult,
        methodology_summary: Mapping[str, object],
        implementation_source: str | None,
        module_path: str | None,
        validation_plan: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return self._persist_version_artifacts(
            stored_version=stored_version,
            contract_summary=dict(parsed_trade.contract_summary),
            methodology_summary=dict(methodology_summary),
            implementation_source=implementation_source,
            validation_plan=validation_plan,
            lineage_record=stored_version.lineage,
            base_version=None,
            module_path=module_path,
        )

    def _persist_version_artifacts(
        self,
        *,
        stored_version: ModelVersionRecord,
        contract_summary: Mapping[str, object],
        methodology_summary: Mapping[str, object],
        implementation_source: str | None,
        validation_plan: Mapping[str, object] | None,
        lineage_record: ModelLineage,
        base_version: ModelVersionRecord | None,
        module_path: str | None,
    ) -> dict[str, object]:
        model_id = stored_version.model_id
        version = stored_version.version
        resolved_validation_plan = validation_plan
        if resolved_validation_plan is None and base_version is not None:
            resolved_validation_plan = self.registry.load_version_artifact(
                base_version.model_id,
                base_version.version,
                "validation-plan",
            )
        artifacts = {
            "contract_uri": self.registry.write_version_artifact(
                model_id, version, "contract", dict(contract_summary)
            ),
            "methodology_uri": self.registry.write_version_artifact(
                model_id, version, "methodology", dict(methodology_summary)
            ),
            "validation_plan_uri": self.registry.write_version_artifact(
                model_id,
                version,
                "validation-plan",
                resolved_validation_plan
                or {
                    "bundle": "deterministic_manifest_v1",
                    "checks": ("contract_summary", "methodology_summary", "engine_binding"),
                },
            ),
            "lineage_uri": self.registry.write_version_artifact(
                model_id, version, "lineage", lineage_record.to_dict()
            ),
        }
        resolved_implementation_source = implementation_source
        if resolved_implementation_source is None and base_version is not None:
            resolved_implementation_source = self.registry.load_version_artifact(
                base_version.model_id,
                base_version.version,
                "code",
            )
        if resolved_implementation_source is not None:
            artifacts["code_uri"] = self.registry.write_version_artifact(
                model_id, version, "code", resolved_implementation_source
            )
        resolved_module_path = str(module_path or "").strip()
        if not resolved_module_path and base_version is not None:
            resolved_module_path = str(base_version.artifacts.get("module_path", "")).strip()
        if resolved_module_path:
            artifacts["module_path"] = resolved_module_path
        return artifacts

    def _default_model_id(self, parsed_trade: TradeParseResult, method_family: str | None) -> str:
        method_token = str(
            method_family or self._match_basis(parsed_trade).get("method_family") or "candidate"
        ).strip()
        trade_token = str(parsed_trade.semantic_id or parsed_trade.trade_type or "candidate").strip()
        return f"{trade_token}_{method_token}_candidate".lower()

    def _next_version(self, model_id: str) -> str:
        existing = self.registry.list_versions(model_id)
        if not existing:
            return "v1"
        return f"v{len(existing) + 1}"

    @staticmethod
    def _lineage_for_persisted_version(base, lineage: Mapping[str, object] | None):
        if base is None:
            return ModelLineage.from_dict(lineage or {})
        lineage_payload = {
            **base.lineage.to_dict(),
            **dict(lineage or {}),
            "parent_model_id": base.model_id,
            "parent_version": base.version,
        }
        return ModelLineage.from_dict(lineage_payload)

    @staticmethod
    def _compare_field(
        rejections: list[str],
        matched_fields: list[str],
        *,
        field_name: str,
        expected: str,
        actual: str,
    ) -> None:
        expected = str(expected or "").strip()
        actual = str(actual or "").strip()
        if expected and actual and expected != actual:
            rejections.append(f"{field_name}_mismatch")
        elif expected or actual:
            matched_fields.append(field_name)

    def _selected_version(self, record) -> Mapping[str, object]:
        version = ""
        if record.latest_approved_version:
            version = record.latest_approved_version
        elif record.latest_validated_version:
            version = record.latest_validated_version
        else:
            version = record.latest_version
        if not version:
            return {
                "version": "",
                "status": record.status.value,
                "engine_binding": {},
                "methodology_summary": {},
            }
        loaded = self.registry.get_version(record.model_id, version)
        if loaded is None:
            return {
                "version": version,
                "status": record.status.value,
                "engine_binding": {},
                "methodology_summary": {},
            }
        return {
            "version": loaded.version,
            "status": loaded.status.value,
            "engine_binding": dict(loaded.engine_binding),
            "methodology_summary": dict(loaded.methodology_summary),
        }

    @staticmethod
    def _match_basis(parsed_trade: TradeParseResult) -> dict[str, object]:
        contract = parsed_trade.semantic_contract
        if contract is None:
            return {
                "parse_status": parsed_trade.parse_status,
            }
        product = contract.product
        conventions = getattr(product, "conventions", None)
        method_family = str(getattr(parsed_trade.semantic_blueprint, "preferred_method", "")).strip()
        return {
            "semantic_id": product.semantic_id,
            "semantic_version": product.semantic_version,
            "product_family": ModelService._product_family(product.semantic_id, product.underlier_structure),
            "instrument_class": product.instrument_class,
            "payoff_family": product.payoff_family,
            "exercise_style": product.exercise_style,
            "underlier_structure": product.underlier_structure,
            "required_market_data": tuple(sorted(parsed_trade.required_market_data)),
            "payout_currency": str(getattr(conventions, "payment_currency", "") or "").strip(),
            "reporting_currency": str(getattr(conventions, "reporting_currency", "") or "").strip(),
            "method_family": method_family,
            "lifecycle_status": "",
        }

    @staticmethod
    def _product_family(semantic_id: str, underlier_structure: str) -> str:
        if semantic_id in {"vanilla_option"}:
            return "equity_option"
        if semantic_id in {"quanto_option"} or "cross_currency" in underlier_structure:
            return "fx_option"
        if semantic_id in {"rate_style_swaption"}:
            return "rates_option"
        if semantic_id in {"callable_bond"}:
            return "callable_bond"
        if semantic_id in {"range_accrual"}:
            return "rates_exotic"
        if semantic_id in {"credit_default_swap"}:
            return "credit_default_swap"
        if semantic_id in {"nth_to_default"}:
            return "credit_basket"
        if semantic_id in {"ranked_observation_basket"}:
            return "equity_option"
        return semantic_id


__all__ = [
    "ModelMatchResult",
    "ModelService",
]
