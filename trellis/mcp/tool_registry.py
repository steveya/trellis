"""Thin MCP tool registry over the transport-neutral platform services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping

from trellis.data.file_snapshot import FILE_IMPORT_PROVIDER_ID
from trellis.mcp.errors import TrellisMcpError
from trellis.mcp.schemas import ToolDefinition


ToolHandler = Callable[[Mapping[str, object]], Mapping[str, object]]


class ToolRegistry:
    """Small in-process registry for Trellis MCP tool handlers."""

    def __init__(self):
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Mapping[str, object] | None = None,
    ) -> None:
        self._definitions[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema or {},
        )
        self._handlers[name] = handler

    def list_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Return one registered tool definition if present."""
        return self._definitions.get(str(name or "").strip())

    def iter_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the registered tool definitions in stable name order."""
        return tuple(self._definitions[name] for name in self.list_tools())

    def call_tool(self, name: str, arguments: Mapping[str, object] | None = None) -> Mapping[str, object]:
        try:
            handler = self._handlers[name]
        except KeyError as exc:
            raise TrellisMcpError(
                code="unknown_tool",
                message=f"Unknown Trellis MCP tool: {name!r}",
                details={"tool_name": name},
            ) from exc
        return handler(dict(arguments or {}))


def build_tool_registry(services) -> ToolRegistry:
    """Register the currently supported Trellis MCP tools."""
    registry = ToolRegistry()
    registry.register(
        name="trellis.session.get_context",
        description="Return the governed session context and active policy bundle.",
        handler=lambda arguments: services.session_service.get_context(arguments.get("session_id")),
    )
    registry.register(
        name="trellis.run_mode.set",
        description="Persist the explicit governed run mode for one session.",
        handler=lambda arguments: services.session_service.set_run_mode(
            session_id=arguments.get("session_id"),
            run_mode=str(arguments.get("run_mode", "")).strip(),
        ),
    )
    registry.register(
        name="trellis.providers.list",
        description="List visible providers and their current binding slots for one session.",
        handler=lambda arguments: services.provider_service.list_providers(
            session_id=arguments.get("session_id"),
            kind=arguments.get("kind"),
        ),
    )
    registry.register(
        name="trellis.providers.configure",
        description="Persist explicit provider bindings for one governed session.",
        handler=lambda arguments: services.provider_service.configure(
            session_id=arguments.get("session_id"),
            provider_bindings=arguments.get("provider_bindings") or {},
        ),
    )
    registry.register(
        name="trellis.run.get",
        description="Return the canonical governed run record for one persisted run id.",
        handler=lambda arguments: services.audit_service.get_run(
            run_id=str(arguments.get("run_id", "")).strip(),
        ),
    )
    registry.register(
        name="trellis.run.get_audit",
        description="Return the canonical governed audit bundle for one persisted run id.",
        handler=lambda arguments: services.audit_service.get_audit(
            run_id=str(arguments.get("run_id", "")).strip(),
        ),
    )
    registry.register(
        name="trellis.price.trade",
        description="Execute the narrow approved-model governed MCP pricing path and persist canonical run and audit records.",
        handler=lambda arguments: services.pricing_service.price_trade(
            session_id=arguments.get("session_id"),
            description=arguments.get("description"),
            instrument_type=arguments.get("instrument_type"),
            structured_trade=arguments.get("structured_trade"),
            normalization_profile=str(arguments.get("normalization_profile", "canonical")).strip() or "canonical",
            output_mode=arguments.get("output_mode"),
            valuation_date=arguments.get("valuation_date"),
        ),
    )
    registry.register(
        name="trellis.trade.parse",
        description="Normalize one natural-language or structured trade into a typed semantic contract surface.",
        handler=lambda arguments: services.trade_service.parse_trade(
            description=arguments.get("description"),
            instrument_type=arguments.get("instrument_type"),
            structured_trade=arguments.get("structured_trade"),
            normalization_profile=str(arguments.get("normalization_profile", "canonical")).strip() or "canonical",
        ).to_dict(),
    )
    registry.register(
        name="trellis.model.match",
        description="Deterministically match a parsed trade against governed model records.",
        handler=lambda arguments: services.model_service.match_trade(
            services.trade_service.parse_trade(
                description=arguments.get("description"),
                instrument_type=arguments.get("instrument_type"),
                structured_trade=arguments.get("structured_trade"),
                normalization_profile=str(arguments.get("normalization_profile", "canonical")).strip() or "canonical",
            )
        ).to_dict(),
    )
    registry.register(
        name="trellis.model.explain_match",
        description="Explain why governed model candidates were selected or rejected for a trade.",
        handler=lambda arguments: services.model_service.explain_match(
            services.trade_service.parse_trade(
                description=arguments.get("description"),
                instrument_type=arguments.get("instrument_type"),
                structured_trade=arguments.get("structured_trade"),
                normalization_profile=str(arguments.get("normalization_profile", "canonical")).strip() or "canonical",
            )
        ).to_dict(),
    )
    registry.register(
        name="trellis.model.generate_candidate",
        description="Generate one draft governed model candidate from a typed trade and persist its initial artifacts.",
        handler=lambda arguments: services.model_service.generate_candidate(
            services.trade_service.parse_trade(
                description=arguments.get("description"),
                instrument_type=arguments.get("instrument_type"),
                structured_trade=arguments.get("structured_trade"),
                normalization_profile=str(arguments.get("normalization_profile", "canonical")).strip() or "canonical",
            ),
            model_id=arguments.get("model_id"),
            version=arguments.get("version"),
            method_family=arguments.get("method_family"),
            engine_binding=arguments.get("engine_binding"),
            assumptions=arguments.get("assumptions") or (),
            tags=arguments.get("tags") or (),
            implementation_source=arguments.get("implementation_source"),
            module_path=arguments.get("module_path"),
            validation_plan=arguments.get("validation_plan"),
            actor=str(arguments.get("actor", "mcp")).strip() or "mcp",
            reason=str(arguments.get("reason", "generate_candidate")).strip() or "generate_candidate",
            notes=str(arguments.get("notes", "")).strip(),
            lineage=arguments.get("lineage"),
            metadata=arguments.get("metadata"),
        ),
    )
    registry.register(
        name="trellis.model.validate",
        description="Run the deterministic governed validation surface for one persisted model version.",
        handler=lambda arguments: services.validation_service.validate_model(
            model_id=str(arguments.get("model_id", "")).strip(),
            version=str(arguments.get("version", "")).strip(),
            actor=str(arguments.get("actor", "validator")).strip() or "validator",
            reason=str(arguments.get("reason", "validate")).strip() or "validate",
            notes=str(arguments.get("notes", "")).strip(),
            refs=arguments.get("refs") or (),
            policy_outcome=arguments.get("policy_outcome"),
            metadata=arguments.get("metadata"),
        ),
    )
    registry.register(
        name="trellis.model.promote",
        description="Apply one explicit governed lifecycle transition to a persisted model version.",
        handler=lambda arguments: services.model_service.promote_version(
            model_id=str(arguments.get("model_id", "")).strip(),
            version=str(arguments.get("version", "")).strip(),
            to_status=str(arguments.get("to_status", "")).strip(),
            actor=str(arguments.get("actor", "reviewer")).strip() or "reviewer",
            reason=str(arguments.get("reason", "manual_transition")).strip() or "manual_transition",
            notes=str(arguments.get("notes", "")).strip(),
            metadata=arguments.get("metadata"),
            validation_store=services.validation_store,
        ),
    )
    registry.register(
        name="trellis.model.persist",
        description="Persist one governed model version with explicit lineage and durable artifact sidecars.",
        handler=lambda arguments: services.model_service.persist_version(
            model_id=str(arguments.get("model_id", "")).strip(),
            base_version=arguments.get("base_version"),
            new_version=arguments.get("new_version"),
            actor=str(arguments.get("actor", "mcp")).strip() or "mcp",
            reason=str(arguments.get("reason", "persist")).strip() or "persist",
            notes=str(arguments.get("notes", "")).strip(),
            contract_summary=arguments.get("contract_summary"),
            methodology_summary=arguments.get("methodology_summary"),
            assumptions=arguments.get("assumptions") or (),
            engine_binding=arguments.get("engine_binding"),
            validation_summary=arguments.get("validation_summary"),
            validation_refs=arguments.get("validation_refs") or (),
            artifacts=arguments.get("artifacts"),
            implementation_source=arguments.get("implementation_source"),
            module_path=arguments.get("module_path"),
            validation_plan=arguments.get("validation_plan"),
            lineage=arguments.get("lineage"),
            metadata=arguments.get("metadata"),
        ),
    )
    registry.register(
        name="trellis.model.versions.list",
        description="List the governed version history for one persisted model id.",
        handler=lambda arguments: services.model_service.list_version_history(
            model_id=str(arguments.get("model_id", "")).strip(),
        ),
    )
    registry.register(
        name="trellis.model.diff",
        description="Diff two governed model versions across contract, code, methodology, validation, and lineage surfaces.",
        handler=lambda arguments: services.model_service.diff_versions(
            model_id=str(arguments.get("model_id", "")).strip(),
            left_version=str(arguments.get("left_version", "")).strip(),
            right_version=str(arguments.get("right_version", "")).strip(),
        ),
    )
    registry.register(
        name="trellis.snapshot.persist_run",
        description="Persist one governed reproducibility bundle for an existing run.",
        handler=lambda arguments: services.snapshot_service.persist_run(
            run_id=str(arguments.get("run_id", "")).strip(),
            tolerances=arguments.get("tolerances"),
            random_seed=arguments.get("random_seed"),
            calendars=arguments.get("calendars") or (),
        ),
    )
    registry.register(
        name="trellis.snapshot.import_files",
        description="Import one explicit market snapshot manifest from local files, persist it, and optionally activate it for a session.",
        handler=lambda arguments: _import_market_snapshot(services, arguments),
    )
    return registry


def _import_market_snapshot(services, arguments: Mapping[str, object]) -> Mapping[str, object]:
    payload = services.snapshot_service.import_files(
        manifest_path=arguments.get("manifest_path"),
        reference_date=arguments.get("reference_date"),
    )
    if arguments.get("activate_session"):
        session_payload = services.session_service.activate_market_snapshot(
            session_id=arguments.get("session_id"),
            snapshot_id=payload["snapshot"]["snapshot_id"],
            provider_id=FILE_IMPORT_PROVIDER_ID,
        )
        return {
            **payload,
            "session": session_payload["session"],
            "policy_bundle": session_payload["policy_bundle"],
        }
    return payload


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
]
