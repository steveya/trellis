"""Canonical audit-bundle assembly on top of governed run records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from trellis.agent.model_audit import load_model_audit_record
from trellis.agent.platform_traces import load_platform_trace_payload
from trellis.agent.task_run_store import load_task_run_record
from trellis.platform.runs import RunRecord


_AUDIT_BUNDLE_SCHEMA_VERSION = 1


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable shallow copy of one mapping."""
    return MappingProxyType(dict(mapping or {}))


def _to_mapping(value: Any) -> dict[str, object]:
    """Convert a mapping-like value into a plain dict."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_list(values) -> list[str]:
    """Return a stable ordered list of unique strings."""
    if not values:
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


@dataclass(frozen=True)
class RunAuditBundle:
    """Deterministic governed audit package for one run."""

    run: Mapping[str, object]
    inputs: Mapping[str, object]
    execution: Mapping[str, object]
    outputs: Mapping[str, object]
    diagnostics: Mapping[str, object]
    artifacts: Mapping[str, object]
    schema_version: int = _AUDIT_BUNDLE_SCHEMA_VERSION

    def __post_init__(self):
        """Normalize top-level bundle sections into immutable mappings."""
        object.__setattr__(self, "run", _freeze_mapping(self.run))
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs))
        object.__setattr__(self, "execution", _freeze_mapping(self.execution))
        object.__setattr__(self, "outputs", _freeze_mapping(self.outputs))
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))
        object.__setattr__(self, "artifacts", _freeze_mapping(self.artifacts))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe payload."""
        return {
            "schema_version": self.schema_version,
            "run": dict(self.run),
            "inputs": dict(self.inputs),
            "execution": dict(self.execution),
            "outputs": dict(self.outputs),
            "diagnostics": dict(self.diagnostics),
            "artifacts": dict(self.artifacts),
        }


def build_run_audit_bundle(record: RunRecord) -> RunAuditBundle:
    """Assemble the canonical governed audit package for one persisted run."""
    loaded_artifacts = _load_artifacts(record)
    trace_payload = _payload_for(loaded_artifacts, "platform_trace")
    trace_summary = _summary_for(loaded_artifacts, "platform_trace")
    model_audit = _summary_for(loaded_artifacts, "model_audit")
    task_run = _summary_for(loaded_artifacts, "task_run_history", "task_run_latest")
    diagnosis = _summary_for(loaded_artifacts, "diagnosis_packet")

    semantic_checkpoint = _to_mapping(trace_payload.get("semantic_checkpoint"))
    generation_boundary = _to_mapping(trace_payload.get("generation_boundary"))
    validation_contract = _to_mapping(trace_payload.get("validation_contract"))
    request_metadata = _to_mapping(trace_payload.get("request_metadata"))
    lowering = _to_mapping(generation_boundary.get("lowering"))

    run_section = {
        "run_id": record.run_id,
        "request_id": record.request_id,
        "status": record.status,
        "action": record.action,
        "run_mode": record.run_mode,
        "session_id": record.session_id,
        "policy_id": record.policy_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }

    inputs_section = {
        "request": {
            "trade_identity": dict(record.trade_identity),
            "request_type": str(trace_payload.get("request_type", "")).strip(),
            "entry_point": str(trace_payload.get("entry_point", "")).strip(),
            "instrument_type": (
                trace_payload.get("instrument_type")
                or record.trade_identity.get("instrument_type")
                or record.trade_identity.get("product")
            ),
            "product_instrument": trace_payload.get("product_instrument"),
            "measures": list(trace_payload.get("measures") or []),
            "request_metadata": request_metadata,
        },
        "parsed_contract": {
            "semantic_checkpoint": semantic_checkpoint,
            "generation_boundary": generation_boundary,
            "validation_contract": validation_contract,
        },
        "provider_bindings": dict(record.provider_bindings),
        "market_snapshot_id": record.market_snapshot_id,
        "valuation_timestamp": record.valuation_timestamp,
    }

    execution_section = {
        "selected_model": dict(record.selected_model),
        "selected_engine": dict(record.selected_engine),
        "route_method": (
            str(trace_payload.get("route_method", "")).strip()
            or str(generation_boundary.get("method", "")).strip()
            or str(record.provenance.get("route_method", "")).strip()
        ),
        "route_family": (
            str(record.provenance.get("route_family", "")).strip()
            or str(lowering.get("route_family", "")).strip()
        ),
        "method_family": (
            str(record.provenance.get("method_family", "")).strip()
            or str(generation_boundary.get("method", "")).strip()
            or str(trace_payload.get("route_method", "")).strip()
        ),
        "provenance": dict(record.provenance),
        "validation_summary": dict(record.validation_summary),
        "policy_outcome": dict(record.policy_outcome),
    }

    diagnostics_section = {
        "warnings": list(record.warnings),
        "blocked": _is_blocked(record, trace_payload),
        "blocker_codes": _blocker_codes(record, trace_payload),
        "trace_status": str(trace_payload.get("status", "")).strip(),
        "trace_outcome": str(trace_payload.get("outcome", "")).strip(),
        "trace_events": list(trace_payload.get("events") or []),
        "trace_details": _to_mapping(trace_payload.get("details")),
        "diagnosis": diagnosis,
        "failure_context": _failure_context(
            record=record,
            trace_payload=trace_payload,
            diagnosis=diagnosis,
            task_run=task_run,
        ),
    }

    artifact_entries = [
        {
            "artifact_id": entry["artifact_id"],
            "artifact_kind": entry["artifact_kind"],
            "uri": entry["uri"],
            "role": entry["role"],
            "metadata": dict(entry["metadata"]),
            "available": entry["available"],
            "error": entry["error"],
            "summary": dict(entry["summary"]),
        }
        for entry in loaded_artifacts
    ]
    artifacts_section = {
        "refs": [item.to_dict() for item in record.artifacts],
        "loaded": artifact_entries,
        "platform_trace": trace_summary,
        "model_audit": model_audit,
        "task_run": task_run,
        "diagnosis_packet": diagnosis,
    }

    return RunAuditBundle(
        run=run_section,
        inputs=inputs_section,
        execution=execution_section,
        outputs=dict(record.result_summary),
        diagnostics=diagnostics_section,
        artifacts=artifacts_section,
    )


def _load_artifacts(record: RunRecord) -> list[dict[str, object]]:
    """Load and summarize known artifact kinds attached to one run."""
    loaded: list[dict[str, object]] = []
    for ref in record.artifacts:
        path = Path(ref.uri)
        entry = {
            "artifact_id": ref.artifact_id,
            "artifact_kind": ref.artifact_kind,
            "uri": ref.uri,
            "role": ref.role,
            "metadata": dict(ref.metadata),
            "available": False,
            "error": "",
            "summary": {},
            "payload": {},
        }
        if not ref.uri:
            entry["error"] = "missing_uri"
            loaded.append(entry)
            continue
        if not path.exists():
            entry["error"] = "missing_artifact"
            loaded.append(entry)
            continue
        try:
            payload, summary = _load_known_artifact(ref.artifact_kind, path)
            entry["available"] = True
            entry["payload"] = payload
            entry["summary"] = summary
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        loaded.append(entry)
    return loaded


def _load_known_artifact(artifact_kind: str, path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Load one known artifact kind and return raw payload plus compact summary."""
    if artifact_kind == "platform_trace":
        payload = load_platform_trace_payload(path)
        return payload, _summarize_platform_trace(payload)
    if artifact_kind == "model_audit":
        payload = load_model_audit_record(path)
        return payload, _summarize_model_audit(payload)
    if artifact_kind in {"task_run_history", "task_run_latest"}:
        payload = load_task_run_record(path)
        return payload, _summarize_task_run(payload)
    if artifact_kind == "diagnosis_packet":
        payload = json.loads(path.read_text())
        return payload, _summarize_diagnosis_packet(payload)
    if artifact_kind == "diagnosis_dossier":
        text = path.read_text()
        payload = {"markdown": text}
        return payload, _summarize_dossier(text)
    payload = json.loads(path.read_text())
    return _to_mapping(payload), {"keys": sorted(_to_mapping(payload).keys())}


def _payload_for(loaded_artifacts: list[dict[str, object]], *artifact_kinds: str) -> dict[str, object]:
    """Return the first loaded raw payload matching one of the given kinds."""
    for kind in artifact_kinds:
        for entry in loaded_artifacts:
            if entry["artifact_kind"] != kind or not entry["available"]:
                continue
            return _to_mapping(entry.get("payload"))
    return {}


def _summary_for(loaded_artifacts: list[dict[str, object]], *artifact_kinds: str) -> dict[str, object]:
    """Return the first loaded summary matching one of the given kinds."""
    for kind in artifact_kinds:
        for entry in loaded_artifacts:
            if entry["artifact_kind"] != kind or not entry["available"]:
                continue
            return _to_mapping(entry.get("summary"))
    return {}


def _summarize_platform_trace(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical audit-facing summary for one platform trace."""
    return {
        "request_id": str(payload.get("request_id", "")).strip(),
        "request_type": str(payload.get("request_type", "")).strip(),
        "entry_point": str(payload.get("entry_point", "")).strip(),
        "action": str(payload.get("action", "")).strip(),
        "status": str(payload.get("status", "")).strip(),
        "outcome": str(payload.get("outcome", "")).strip(),
        "instrument_type": payload.get("instrument_type"),
        "product_instrument": payload.get("product_instrument"),
        "route_method": str(payload.get("route_method", "")).strip(),
        "measures": list(payload.get("measures") or []),
        "blocker_codes": _string_list(payload.get("blocker_codes") or []),
        "details": _to_mapping(payload.get("details")),
        "events": list(payload.get("events") or []),
        "request_metadata": _to_mapping(payload.get("request_metadata")),
        "semantic_checkpoint": _to_mapping(payload.get("semantic_checkpoint")),
        "generation_boundary": _to_mapping(payload.get("generation_boundary")),
        "validation_contract": _to_mapping(payload.get("validation_contract")),
    }


def _summarize_model_audit(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the audit-facing summary for one model-audit artifact."""
    return {
        "audit_id": str(payload.get("audit_id", "")).strip(),
        "task_id": str(payload.get("task_id", "")).strip(),
        "run_id": str(payload.get("run_id", "")).strip(),
        "method": str(payload.get("method", "")).strip(),
        "instrument_type": str(payload.get("instrument_type", "")).strip(),
        "timestamp": str(payload.get("timestamp", "")).strip(),
        "class_name": str(payload.get("class_name", "")).strip(),
        "module_path": str(payload.get("module_path", "")).strip(),
        "source_code_hash": str(payload.get("source_code_hash", "")).strip(),
        "all_gates_passed": bool(payload.get("all_gates_passed")),
        "approval_status": str(payload.get("approval_status", "")).strip(),
        "validation_gates": list(payload.get("validation_gates") or []),
        "build_metrics": _to_mapping(payload.get("build_metrics")),
        "benchmark": _to_mapping(payload.get("benchmark")),
        "approval": _to_mapping(payload.get("approval")),
        "has_prompt_log": bool(payload.get("has_prompt_log")),
        "prompt_log_path": str(payload.get("prompt_log_path", "")).strip(),
    }


def _summarize_task_run(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the audit-facing summary for one task-run artifact."""
    return {
        "task_id": str(payload.get("task_id", "")).strip(),
        "task_kind": str(payload.get("task_kind", "")).strip(),
        "run_id": str(payload.get("run_id", "")).strip(),
        "persisted_at": str(payload.get("persisted_at", "")).strip(),
        "summary": _to_mapping(payload.get("summary")),
        "workflow": _to_mapping(payload.get("workflow")),
        "comparison": _to_mapping(payload.get("comparison")),
        "market": _to_mapping(payload.get("market")),
        "issue_refs": _to_mapping(payload.get("issue_refs")),
        "trace_summaries": list(payload.get("trace_summaries") or []),
    }


def _summarize_diagnosis_packet(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the audit-facing summary for one diagnosis packet."""
    task = _to_mapping(payload.get("task"))
    outcome = _to_mapping(payload.get("outcome"))
    primary_failure = _to_mapping(payload.get("primary_failure"))
    return {
        "task_id": str(task.get("id", "")).strip(),
        "task_title": str(task.get("title", "")).strip(),
        "status": str(outcome.get("status", "")).strip(),
        "success": bool(outcome.get("success")),
        "failure_bucket": str(outcome.get("failure_bucket", "")).strip(),
        "headline": str(outcome.get("headline", "")).strip(),
        "next_action": str(outcome.get("next_action", "")).strip(),
        "primary_failure": primary_failure,
    }


def _summarize_dossier(text: str) -> dict[str, object]:
    """Return a compact summary for one Markdown dossier artifact."""
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "preview": "\n".join(lines[:8]).strip(),
    }


def _blocker_codes(record: RunRecord, trace_payload: Mapping[str, object]) -> list[str]:
    """Return the stable union of run-ledger and trace blocker codes."""
    blocker_codes = set(_string_list(record.policy_outcome.get("blocker_codes") or []))
    blocker_codes.update(_string_list(trace_payload.get("blocker_codes") or []))
    return sorted(blocker_codes)


def _is_blocked(record: RunRecord, trace_payload: Mapping[str, object]) -> bool:
    """Return whether the run represents a blocked governed outcome."""
    if str(record.status or "").strip().lower() == "blocked":
        return True
    if record.policy_outcome and record.policy_outcome.get("allowed") is False:
        return True
    trace_status = str(trace_payload.get("status", "")).strip().lower()
    trace_outcome = str(trace_payload.get("outcome", "")).strip().lower()
    return trace_status == "blocked" or trace_outcome == "request_blocked"


def _failure_context(
    *,
    record: RunRecord,
    trace_payload: Mapping[str, object],
    diagnosis: Mapping[str, object],
    task_run: Mapping[str, object],
) -> dict[str, object]:
    """Assemble operator-facing failure context from the canonical run plus artifacts."""
    context: dict[str, object] = {}
    error = record.result_summary.get("error")
    if error:
        context["error"] = error
    trace_details = _to_mapping(trace_payload.get("details"))
    if trace_details.get("reason"):
        context["trace_reason"] = trace_details.get("reason")
    if trace_details:
        context["trace_details"] = trace_details
    if diagnosis:
        for key in ("headline", "failure_bucket", "next_action"):
            value = diagnosis.get(key)
            if value:
                context[key] = value
        primary_failure = _to_mapping(diagnosis.get("primary_failure"))
        if primary_failure:
            context["primary_failure"] = primary_failure
    if task_run:
        summary = _to_mapping(task_run.get("summary"))
        workflow = _to_mapping(task_run.get("workflow"))
        if summary:
            context["task_run_summary"] = summary
        if workflow:
            context["task_run_workflow"] = workflow
    policy_blockers = list(record.policy_outcome.get("blockers") or [])
    if policy_blockers:
        context["policy_blockers"] = policy_blockers
    return context
