"""Structured decision checkpoints for agent pipeline runs (QUA-425).

Captures the decision made at each agent stage in a diffable format,
enabling drift detection across releases and precise failure localization.

Usage::

    # After a build_with_knowledge() or build_payoff() run:
    checkpoint = capture_checkpoint(
        task_id="T38",
        instrument_type="callable_bond",
        build_meta=build_meta,
        pricing_plan=pricing_plan,
        token_summary=token_summary,
        outcome="pass",
        final_price=98.234,
    )
    save_checkpoint(checkpoint)

    # Diffing two checkpoints:
    divergences = diff_checkpoints(old_checkpoint, new_checkpoint)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).parent / "knowledge" / "traces" / "checkpoints"
DEFAULT_RETENTION = 10  # keep last N checkpoints per task


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDecision:
    """One agent stage's decision in a pipeline run."""

    agent: str                          # "quant", "planner", "builder", "validator", etc.
    decision: str                       # Primary decision value (e.g. "rate_tree", "compiled")
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: int = 0
    input_hash: str = ""                # SHA-256 of input for change detection
    output_hash: str = ""               # SHA-256 of output for change detection


@dataclass(frozen=True)
class DecisionCheckpoint:
    """Complete decision trace for one pipeline run."""

    task_id: str
    instrument_type: str
    timestamp: str
    stages: tuple[StageDecision, ...]
    outcome: str                        # "pass", "fail_build", "fail_validate", "reused"
    total_tokens: int = 0
    final_price: float | None = None
    tolerance: float | None = None
    attempts: int = 1
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class StageDivergence:
    """One stage where two checkpoints differ."""

    agent: str
    old_decision: str
    new_decision: str
    old_metadata: dict[str, Any] = field(default_factory=dict)
    new_metadata: dict[str, Any] = field(default_factory=dict)
    severity: str = "decision"          # "decision", "metadata", "price"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_value(value: Any) -> str:
    """SHA-256 hex of a stable JSON-safe representation (first 16 chars)."""
    normalized = _normalize_for_hash(value)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_imports_from_code(code: str) -> list[str]:
    """Extract import names from Python source code."""
    imports = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            imports.append(stripped)
    return imports


def _normalize_for_hash(value: Any) -> Any:
    """Normalize nested values into a stable JSON-safe shape for hashing."""
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_hash(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_for_hash(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def _nonempty_mapping(*, source: Mapping[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    """Project selected keys from a mapping while dropping empty values."""
    if source is None:
        return {}
    selected: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value not in (None, "", [], {}, ()):
            selected[key] = value
    return selected


# ---------------------------------------------------------------------------
# Checkpoint capture
# ---------------------------------------------------------------------------

def capture_checkpoint(
    *,
    task_id: str,
    instrument_type: str,
    build_meta: dict[str, Any] | None = None,
    pricing_plan: Any = None,
    spec_schema: Any = None,
    code: str | None = None,
    token_summary: dict[str, Any] | None = None,
    semantic_checkpoint: Mapping[str, Any] | None = None,
    generation_boundary: Mapping[str, Any] | None = None,
    validation_contract: Mapping[str, Any] | None = None,
    outcome: str = "unknown",
    final_price: float | None = None,
    tolerance: float | None = None,
    attempts: int = 1,
    provider: str = "",
    model: str = "",
) -> DecisionCheckpoint:
    """Build a DecisionCheckpoint from pipeline artifacts.

    This extracts decision boundaries from the artifacts that already exist
    in the pipeline (pricing_plan, spec_schema, generated code, build_meta)
    without requiring changes to executor.py internals. Semantic-route callers
    can additionally pass compact semantic, generation, and validation
    summaries to make route drift diffable across reruns.
    """
    stages: list[StageDecision] = []
    semantic_checkpoint_data = dict(semantic_checkpoint or {})
    generation_boundary_data = dict(generation_boundary or {})
    validation_contract_data = dict(validation_contract or {})

    # Stage 1: Quant agent decision
    if pricing_plan is not None:
        method = getattr(pricing_plan, "method", "unknown")
        required_data = sorted(getattr(pricing_plan, "required_market_data", []) or [])
        method_modules = list(getattr(pricing_plan, "method_modules", []) or [])
        reason = getattr(pricing_plan, "selection_reason", None)
        quant_metadata = {
            "required_market_data": required_data,
            "method_modules": method_modules,
            "selection_reason": reason or "",
        }
        stages.append(StageDecision(
            agent="quant",
            decision=method,
            metadata=quant_metadata,
            input_hash=_hash_value(instrument_type),
            output_hash=_hash_value({
                "decision": method,
                "metadata": quant_metadata,
            }),
        ))

    # Stage 2: Semantic identity and compatibility bridge summary.
    if semantic_checkpoint_data:
        semantic_decision = str(semantic_checkpoint_data.get("semantic_id") or "unknown")
        stages.append(StageDecision(
            agent="semantic",
            decision=semantic_decision,
            metadata=semantic_checkpoint_data,
            input_hash=_hash_value(instrument_type),
            output_hash=_hash_value({
                "decision": semantic_decision,
                "metadata": semantic_checkpoint_data,
            }),
        ))

    # Stage 3: Route / lowering boundary.
    if generation_boundary_data:
        route_metadata = _nonempty_mapping(
            source=generation_boundary_data,
            keys=(
                "method",
                "valuation_context",
                "required_data_spec",
                "market_binding_spec",
                "lowering",
                "route_binding_authority",
                "primitive_plan",
            ),
        )
        lowering = route_metadata.get("lowering") or {}
        route_decision = str(
            lowering.get("route_id")
            or generation_boundary_data.get("method")
            or "unknown"
        )
        stages.append(StageDecision(
            agent="route",
            decision=route_decision,
            metadata=route_metadata,
            output_hash=_hash_value({
                "decision": route_decision,
                "metadata": route_metadata,
            }),
        ))

    # Stage 4: Planner decision
    if spec_schema is not None:
        spec_name = getattr(spec_schema, "spec_name", "unknown")
        class_name = getattr(spec_schema, "class_name", "unknown")
        fields = []
        raw_fields = getattr(spec_schema, "fields", [])
        if raw_fields:
            fields = [getattr(f, "name", str(f)) for f in raw_fields]
        stages.append(StageDecision(
            agent="planner",
            decision=spec_name,
            metadata={
                "class_name": class_name,
                "field_count": len(fields),
                "fields": sorted(fields),
            },
            output_hash=_hash_value(f"{spec_name}:{class_name}:{sorted(fields)}"),
        ))

    # Stage 5: Builder (code generation) decision
    builder_boundary = _nonempty_mapping(
        source=generation_boundary_data,
        keys=("approved_modules", "inspected_modules", "symbols_to_reuse"),
    )
    builder_code = code
    if builder_code is None and build_meta is not None:
        builder_code = build_meta.get("code")
    if builder_code:
        code_lines = len(builder_code.splitlines())
        imports = _extract_imports_from_code(builder_code)
        builder_metadata = dict(builder_boundary)
        builder_metadata.update({
            "code_lines": code_lines,
            "import_count": len(imports),
            "imports": imports[:20],  # cap to avoid huge metadata
        })
        stages.append(StageDecision(
            agent="builder",
            decision="compiled",
            metadata=builder_metadata,
            output_hash=_hash_value({
                "code": builder_code,
                "boundary": builder_boundary,
            }),
        ))
    elif builder_boundary:
        stages.append(StageDecision(
            agent="builder",
            decision="planned",
            metadata=builder_boundary,
            output_hash=_hash_value(builder_boundary),
        ))

    # Stage 6: Validation boundary + outcome.
    if outcome in ("pass", "fail_validate", "fail_price") or validation_contract_data:
        validation_meta = _nonempty_mapping(
            source=validation_contract_data,
            keys=(
                "contract_id",
                "bundle_id",
                "route_id",
                "route_family",
                "required_market_data",
                "deterministic_checks",
                "comparison_relations",
                "lowering_errors",
                "admissibility_failures",
                "residual_risks",
            ),
        )
        if final_price is not None:
            validation_meta["final_price"] = final_price
        if tolerance is not None:
            validation_meta["tolerance"] = tolerance
        if build_meta is not None:
            failures = build_meta.get("failures", [])
            if failures:
                validation_meta["failure_count"] = len(failures)
                validation_meta["first_failure"] = str(failures[0])[:200]
        validation_decision = "planned"
        if outcome == "pass":
            validation_decision = "pass"
        elif outcome in ("fail_validate", "fail_price"):
            validation_decision = "fail"
        validation_hash_meta = _nonempty_mapping(
            source=validation_meta,
            keys=(
                "contract_id",
                "bundle_id",
                "route_id",
                "route_family",
                "required_market_data",
                "deterministic_checks",
                "comparison_relations",
                "lowering_errors",
                "admissibility_failures",
                "residual_risks",
            ),
        )
        stages.append(StageDecision(
            agent="validator",
            decision=validation_decision,
            metadata=validation_meta,
            output_hash=_hash_value({
                "decision": validation_decision,
                "metadata": validation_hash_meta,
            }),
        ))

    # Extract total tokens
    total_tokens = 0
    if token_summary is not None:
        total_tokens = token_summary.get("total_tokens", 0) or 0

    return DecisionCheckpoint(
        task_id=task_id,
        instrument_type=instrument_type or "unknown",
        timestamp=datetime.now(timezone.utc).isoformat(),
        stages=tuple(stages),
        outcome=outcome,
        total_tokens=total_tokens,
        final_price=final_price,
        tolerance=tolerance,
        attempts=attempts,
        provider=provider,
        model=model,
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff_checkpoints(
    old: DecisionCheckpoint,
    new: DecisionCheckpoint,
) -> list[StageDivergence]:
    """Compare two checkpoints stage-by-stage.  Returns divergences found.

    Compares by agent name matching (not positional), so stage order
    differences don't produce false positives.
    """
    divergences: list[StageDivergence] = []

    old_stages = {s.agent: s for s in old.stages}
    new_stages = {s.agent: s for s in new.stages}

    all_agents = sorted(set(old_stages) | set(new_stages))
    for agent in all_agents:
        old_s = old_stages.get(agent)
        new_s = new_stages.get(agent)

        if old_s is None or new_s is None:
            # Stage added or removed — structural divergence
            divergences.append(StageDivergence(
                agent=agent,
                old_decision=old_s.decision if old_s else "(absent)",
                new_decision=new_s.decision if new_s else "(absent)",
                old_metadata=old_s.metadata if old_s else {},
                new_metadata=new_s.metadata if new_s else {},
                severity="decision",
            ))
            continue

        # Decision-level divergence (e.g., method changed)
        if old_s.decision != new_s.decision:
            divergences.append(StageDivergence(
                agent=agent,
                old_decision=old_s.decision,
                new_decision=new_s.decision,
                old_metadata=old_s.metadata,
                new_metadata=new_s.metadata,
                severity="decision",
            ))
            continue

        # Metadata-level divergence (same decision, different details)
        if old_s.output_hash and new_s.output_hash and old_s.output_hash != new_s.output_hash:
            divergences.append(StageDivergence(
                agent=agent,
                old_decision=old_s.decision,
                new_decision=new_s.decision,
                old_metadata=old_s.metadata,
                new_metadata=new_s.metadata,
                severity="metadata",
            ))

    # Price drift (validator stage has same decision but price shifted)
    old_val = old_stages.get("validator")
    new_val = new_stages.get("validator")
    if (
        old_val is not None
        and new_val is not None
        and old_val.decision == new_val.decision == "pass"
    ):
        old_price = old_val.metadata.get("final_price")
        new_price = new_val.metadata.get("final_price")
        old_tol = old_val.metadata.get("tolerance")
        if (
            old_price is not None
            and new_price is not None
            and old_tol is not None
            and old_tol > 0
        ):
            drift_ratio = abs(new_price - old_price) / old_tol
            if drift_ratio > 0.5:
                # Price consumed >50% of tolerance — warn
                divergences.append(StageDivergence(
                    agent="validator",
                    old_decision=f"price={old_price}",
                    new_decision=f"price={new_price}",
                    old_metadata={"tolerance": old_tol, "drift_ratio": round(drift_ratio, 3)},
                    new_metadata={"tolerance": old_tol, "drift_ratio": round(drift_ratio, 3)},
                    severity="price",
                ))

    return divergences


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _checkpoint_to_dict(cp: DecisionCheckpoint) -> dict[str, Any]:
    """Convert a checkpoint to a plain dict for YAML serialization."""
    return {
        "task_id": cp.task_id,
        "instrument_type": cp.instrument_type,
        "timestamp": cp.timestamp,
        "provider": cp.provider,
        "model": cp.model,
        "outcome": cp.outcome,
        "attempts": cp.attempts,
        "total_tokens": cp.total_tokens,
        "final_price": cp.final_price,
        "tolerance": cp.tolerance,
        "stages": [
            {
                "agent": s.agent,
                "decision": s.decision,
                "metadata": s.metadata,
                "tokens_used": s.tokens_used,
                "latency_ms": s.latency_ms,
                "input_hash": s.input_hash,
                "output_hash": s.output_hash,
            }
            for s in cp.stages
        ],
    }


def _dict_to_checkpoint(d: dict[str, Any]) -> DecisionCheckpoint:
    """Reconstruct a checkpoint from a plain dict."""
    stages = tuple(
        StageDecision(
            agent=s["agent"],
            decision=s["decision"],
            metadata=s.get("metadata", {}),
            tokens_used=s.get("tokens_used", 0),
            latency_ms=s.get("latency_ms", 0),
            input_hash=s.get("input_hash", ""),
            output_hash=s.get("output_hash", ""),
        )
        for s in d.get("stages", [])
    )
    return DecisionCheckpoint(
        task_id=d["task_id"],
        instrument_type=d.get("instrument_type", "unknown"),
        timestamp=d.get("timestamp", ""),
        stages=stages,
        outcome=d.get("outcome", "unknown"),
        total_tokens=d.get("total_tokens", 0),
        final_price=d.get("final_price"),
        tolerance=d.get("tolerance"),
        attempts=d.get("attempts", 1),
        provider=d.get("provider", ""),
        model=d.get("model", ""),
    )


def save_checkpoint(
    checkpoint: DecisionCheckpoint,
    *,
    directory: Path | None = None,
    retention: int = DEFAULT_RETENTION,
) -> Path:
    """Write a checkpoint to YAML and enforce retention policy.

    Returns the path of the written file.
    """
    base_dir = directory or CHECKPOINT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    # Filename: {task_id}_{timestamp_compact}.yaml
    ts_compact = re.sub(r"[^0-9]", "", checkpoint.timestamp[:19])
    filename = f"{checkpoint.task_id}_{ts_compact}.yaml"
    path = base_dir / filename

    data = _checkpoint_to_dict(checkpoint)
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _log.info("Checkpoint saved: %s", path)

    # Enforce retention: keep only the last N per task_id
    _enforce_retention(base_dir, checkpoint.task_id, retention)

    return path


def load_checkpoint(path: Path) -> DecisionCheckpoint:
    """Load a checkpoint from a YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _dict_to_checkpoint(data)


def load_latest_checkpoint(
    task_id: str,
    *,
    directory: Path | None = None,
) -> DecisionCheckpoint | None:
    """Load the most recent checkpoint for a given task_id, or None."""
    base_dir = directory or CHECKPOINT_DIR
    if not base_dir.exists():
        return None
    candidates = sorted(
        base_dir.glob(f"{task_id}_*.yaml"),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        return None
    return load_checkpoint(candidates[0])


def list_checkpoints(
    task_id: str,
    *,
    directory: Path | None = None,
) -> list[Path]:
    """List all checkpoint files for a task, newest first."""
    base_dir = directory or CHECKPOINT_DIR
    if not base_dir.exists():
        return []
    return sorted(
        base_dir.glob(f"{task_id}_*.yaml"),
        key=lambda p: p.name,
        reverse=True,
    )


def _enforce_retention(base_dir: Path, task_id: str, retention: int) -> None:
    """Delete checkpoints beyond the retention limit for a task."""
    files = sorted(base_dir.glob(f"{task_id}_*.yaml"), key=lambda p: p.name, reverse=True)
    for old_file in files[retention:]:
        try:
            old_file.unlink()
            _log.debug("Checkpoint pruned: %s", old_file)
        except OSError as exc:
            _log.warning("Failed to prune checkpoint %s: %s", old_file, exc)


# ---------------------------------------------------------------------------
# Formatting (for terminal / CI output)
# ---------------------------------------------------------------------------

def format_checkpoint_summary(checkpoint: DecisionCheckpoint) -> str:
    """Human-readable one-line summary of a checkpoint."""
    stages_str = " -> ".join(
        f"{s.agent}={s.decision}" for s in checkpoint.stages
    )
    price_str = f" price={checkpoint.final_price}" if checkpoint.final_price is not None else ""
    return (
        f"[{checkpoint.task_id}] {checkpoint.instrument_type} "
        f"({checkpoint.outcome}, {checkpoint.attempts}att, "
        f"{checkpoint.total_tokens}tok{price_str}): {stages_str}"
    )


def format_divergence_report(divergences: list[StageDivergence]) -> str:
    """Format divergences into a human-readable report."""
    if not divergences:
        return "No divergences detected."
    lines = []
    for d in divergences:
        icon = {"decision": "\u2717", "metadata": "\u26a0", "price": "\u26a0"}.get(d.severity, "?")
        lines.append(
            f"  {icon} {d.agent}: {d.old_decision} -> {d.new_decision} [{d.severity}]"
        )
    return "\n".join(lines)
