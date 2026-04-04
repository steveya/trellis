"""Model audit trail: immutable records for AI-generated pricing models.

Each successful build produces a ModelAuditRecord capturing the generated
source code, environment fingerprint, market state snapshot, validation
gate results, and build metrics.  Records are stored under
``task_runs/audits/{task_id}/`` and linked from the diagnostic packet.

Post-build data (cross-validation results, approval status) is stored as
sidecar files so the core record remains immutable.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema version — bump when the audit record format changes
# ---------------------------------------------------------------------------
AUDIT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Storage root (relative to repo root)
# ---------------------------------------------------------------------------
_AUDIT_DIR = Path("task_runs") / "audits"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Immutable snapshot of the build environment at model-generation time."""

    repo_revision: str
    llm_model_id: str
    knowledge_hash: str
    python_version: str = field(default_factory=lambda: sys.version.split()[0])


@dataclass(frozen=True)
class ValidationGateResult:
    """Outcome of a single validation gate (import, semantic, lite_review, etc.)."""

    gate: str
    passed: bool
    issues: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildMetrics:
    """Quantitative build-process metrics."""

    attempt_number: int
    total_attempts: int
    token_usage: dict[str, Any] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class ModelAuditRecord:
    """Self-contained audit record for one AI-generated pricing model."""

    # Identity
    schema_version: int
    audit_id: str
    task_id: str
    run_id: str
    method: str
    instrument_type: str
    timestamp: str

    # The model itself
    source_code: str
    source_code_hash: str
    spec_schema: dict[str, Any]
    class_name: str
    module_path: str

    # Provenance
    environment: EnvironmentFingerprint
    market_state_summary: dict[str, Any]
    pricing_plan_summary: dict[str, Any]

    # Validation
    validation_gates: tuple[ValidationGateResult, ...]
    all_gates_passed: bool

    # Build process
    build_metrics: BuildMetrics

    # Governance
    approval_status: str = "pending_review"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_audit_record(
    *,
    task_id: str,
    run_id: str,
    method: str,
    instrument_type: str,
    source_code: str,
    spec_schema_dict: dict[str, Any],
    class_name: str,
    module_path: str,
    repo_revision: str,
    llm_model_id: str,
    knowledge_hash: str,
    market_state_summary: dict[str, Any],
    pricing_plan_summary: dict[str, Any],
    validation_gates: list[ValidationGateResult] | tuple[ValidationGateResult, ...],
    attempt_number: int,
    total_attempts: int,
    token_usage: dict[str, Any] | None = None,
    wall_clock_seconds: float = 0.0,
) -> ModelAuditRecord:
    """Construct a complete ModelAuditRecord from build outputs."""
    audit_id = f"{run_id}_{method}"
    code_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]
    gates = tuple(validation_gates)
    all_passed = all(g.passed for g in gates)

    return ModelAuditRecord(
        schema_version=AUDIT_SCHEMA_VERSION,
        audit_id=audit_id,
        task_id=task_id,
        run_id=run_id,
        method=method,
        instrument_type=instrument_type or "",
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_code=source_code,
        source_code_hash=code_hash,
        spec_schema=spec_schema_dict,
        class_name=class_name,
        module_path=module_path,
        environment=EnvironmentFingerprint(
            repo_revision=repo_revision,
            llm_model_id=llm_model_id,
            knowledge_hash=knowledge_hash,
        ),
        market_state_summary=market_state_summary,
        pricing_plan_summary=pricing_plan_summary,
        validation_gates=gates,
        all_gates_passed=all_passed,
        build_metrics=BuildMetrics(
            attempt_number=attempt_number,
            total_attempts=total_attempts,
            token_usage=dict(token_usage or {}),
            wall_clock_seconds=wall_clock_seconds,
        ),
        approval_status="pending_review",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass instances and tuples for JSON."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    return obj


def write_model_audit_record(record: ModelAuditRecord, base_dir: Path | None = None) -> Path:
    """Write an immutable audit record to disk.

    Returns the path to the written JSON file.
    """
    root = base_dir or _AUDIT_DIR
    task_dir = root / record.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{record.audit_id}.json"
    path = task_dir / filename
    path.write_text(json.dumps(_serialize(record), indent=2, default=str))
    return path


def load_model_audit_record(path: Path) -> dict[str, Any]:
    """Load an audit record, merging any sidecar files.

    Transparently merges ``.benchmark.json`` and ``.approval.json`` sidecars
    into the returned dict so callers see the complete picture.
    """
    record = json.loads(path.read_text())

    benchmark = path.with_suffix(".benchmark.json")
    if benchmark.exists():
        record["benchmark"] = json.loads(benchmark.read_text())

    approval = path.with_suffix(".approval.json")
    if approval.exists():
        record["approval"] = json.loads(approval.read_text())

    prompts = path.with_suffix(".prompts.jsonl")
    if prompts.exists():
        record["has_prompt_log"] = True
        record["prompt_log_path"] = str(prompts)

    return record


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------

def write_benchmark_sidecar(
    audit_path: Path,
    *,
    comparison_status: str,
    prices: dict[str, float],
    deviations_pct: dict[str, float],
    reference_targets: list[str] | None = None,
) -> Path:
    """Write cross-validation benchmark results as a sidecar."""
    sidecar = audit_path.with_suffix(".benchmark.json")
    data = {
        "comparison_status": comparison_status,
        "prices": prices,
        "deviations_pct": deviations_pct,
        "reference_targets": reference_targets or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(data, indent=2, default=str))
    return sidecar


def approve_model(
    audit_path: Path,
    *,
    reviewer: str,
    status: str = "approved",
    notes: str = "",
) -> Path:
    """Write approval sidecar for an audit record.

    Parameters
    ----------
    audit_path
        Path to the core ``.json`` audit record.
    reviewer
        Identifier for the approver (e.g. name, email, or system ID).
    status
        One of ``approved``, ``rejected``, ``conditionally_approved``.
    notes
        Free-text justification or conditions.
    """
    sidecar = audit_path.with_suffix(".approval.json")
    data = {
        "status": status,
        "reviewer": reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "audit_record_hash": hashlib.sha256(audit_path.read_bytes()).hexdigest()[:16],
    }
    sidecar.write_text(json.dumps(data, indent=2))
    return sidecar


def write_prompt_log(
    audit_path: Path,
    usage_records: list[dict[str, Any]],
) -> Path | None:
    """Write LLM prompt/response log as a JSONL sidecar (opt-in).

    Only writes records that contain ``prompt_text``.
    Returns the path if any records were written, else None.
    """
    records_with_prompts = [r for r in usage_records if "prompt_text" in r]
    if not records_with_prompts:
        return None
    log_path = audit_path.with_suffix(".prompts.jsonl")
    with open(log_path, "w") as f:
        for rec in records_with_prompts:
            json.dump(rec, f, default=str)
            f.write("\n")
    return log_path
