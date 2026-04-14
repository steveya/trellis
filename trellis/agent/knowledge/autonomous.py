"""Self-maintaining knowledge-aware build wrapper.

This is the top-level entry point for building payoffs with autonomous
knowledge management.  It wraps build_payoff() with:

1. Pre-flight gap check — audit knowledge readiness
2. Build — with gap warnings injected into prompt
3. Post-build reflection — capture lessons, attribute, enrich

Usage:
    from trellis.agent.knowledge.autonomous import build_with_knowledge

    PayoffCls = build_with_knowledge(
        "callable bond with 5% coupon",
        instrument_type="callable_bond",
        market_state=ms,
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from dataclasses import dataclass, field
import os
from typing import Any, Mapping


class BuildTrackingFailure(RuntimeError):
    """Wrap a build failure with tracking metadata collected before the exception."""

    def __init__(self, message: str, *, meta: dict[str, Any], cause: Exception):
        """Store tracked build metadata alongside the original failure cause."""
        super().__init__(message)
        self.meta = meta
        self.cause = cause


@dataclass
class BuildResult:
    """Full result of a knowledge-aware build."""

    payoff_cls: type | None
    success: bool
    attempts: int
    failures: list[str] = field(default_factory=list)
    code: str = ""
    gap_confidence: float = 0.0
    knowledge_gaps: list[str] = field(default_factory=list)
    reflection: dict[str, Any] = field(default_factory=dict)
    agent_observations: list[dict[str, Any]] = field(default_factory=list)
    knowledge_summary: dict[str, Any] = field(default_factory=dict)
    platform_trace_path: str | None = None
    platform_request_id: str | None = None
    analytical_trace_path: str | None = None
    analytical_trace_text_path: str | None = None
    audit_record_path: str | None = None
    execution_module_name: str | None = None
    execution_module_path: str | None = None
    execution_file_path: str | None = None
    admission_target_module_name: str | None = None
    admission_target_module_path: str | None = None
    admission_target_file_path: str | None = None
    blocker_details: dict[str, Any] | None = None
    token_usage_summary: dict[str, Any] = field(default_factory=dict)
    gate_decision: object | None = None  # BuildGateDecision when gate evaluated
    post_build_tracking: dict[str, Any] = field(default_factory=dict)


_PROVIDER_FAILURE_MARKERS = (
    "llm provider",
    "openai ",
    "not_found_error",
    "insufficient_quota",
    "returned invalid json response",
    "returned empty json response",
    "request failed after",
)

_SKIP_POST_BUILD_REFLECTION_ENV = "TRELLIS_SKIP_POST_BUILD_REFLECTION"
_SKIP_POST_BUILD_CONSOLIDATION_ENV = "TRELLIS_SKIP_POST_BUILD_CONSOLIDATION"


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for post-build tracking."""
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str) -> bool:
    """Parse a boolean-like environment flag."""
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _post_build_control_flags() -> dict[str, bool]:
    """Return the active post-build bisection controls."""
    return {
        "skip_reflection": _env_flag(_SKIP_POST_BUILD_REFLECTION_ENV),
        "skip_consolidation": _env_flag(_SKIP_POST_BUILD_CONSOLIDATION_ENV),
    }


def _record_post_build_phase(
    tracking: dict[str, Any],
    phase: str,
    *,
    status: str,
    **details: Any,
) -> None:
    """Append one structured post-build phase marker."""
    entry = {
        "phase": phase,
        "status": status,
        "timestamp": _utc_now_iso(),
    }
    if details:
        entry["details"] = details
    tracking.setdefault("events", []).append(entry)
    tracking["last_phase"] = phase
    tracking["last_status"] = status
    tracking["updated_at"] = entry["timestamp"]


def _llm_stage_metadata(
    *,
    model: str,
    instrument_type: str | None,
    request_metadata: Mapping[str, object] | None,
    comparison_target: str | None,
) -> dict[str, Any]:
    """Attach stable request metadata to LLM stage logs."""
    metadata = {
        "model": model,
        "instrument_type": instrument_type,
        "task_id": (
            str(request_metadata.get("task_id"))
            if isinstance(request_metadata, Mapping) and request_metadata.get("task_id")
            else None
        ),
        "comparison_target": comparison_target,
    }
    return {
        key: value
        for key, value in metadata.items()
        if value not in {None, ""}
    }


def build_with_knowledge(
    description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    market_state=None,
    max_retries: int = 3,
    validation: str = "standard",
    force_rebuild: bool = False,
    fresh_build: bool = False,
    preferred_method: str | None = None,
    comparison_target: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> BuildResult:
    """Build a payoff class while autonomously managing the knowledge lifecycle.

    Phases:
      1. Decompose and gap-check -- break the product into features, audit
         what knowledge is available, and compute a readiness confidence
         score (0.0-1.0).  Low confidence triggers extra retries.
      2. Build -- call ``build_payoff()`` with gap warnings injected into the
         LLM prompt so the code generator knows where knowledge is thin.
      3. Reflect -- after the build (success or failure), run a lightweight
         LLM pass that attributes outcomes to lessons and captures new ones.

    Returns a ``BuildResult`` containing the generated payoff class (or None
    on failure), attempt count, gap confidence, captured lessons, and
    token-usage summary.
    """
    from trellis.agent.knowledge.decompose import decompose, decompose_to_ir
    from trellis.agent.knowledge.gap_check import gap_check, format_gap_warnings
    from trellis.agent.knowledge.methods import normalize_method
    from trellis.agent.knowledge.reflect import reflect_on_build
    from trellis.agent.config import (
        TokenBudgetExceeded,
        enforce_llm_token_budget,
        get_model_for_stage,
        llm_usage_session,
        llm_usage_stage,
        summarize_llm_usage,
    )
    from trellis.agent.platform_traces import attach_platform_trace_token_usage

    effective_description = description
    if comparison_target:
        effective_description = (
            f"{description}\n\n"
            f"Implementation target: {comparison_target}"
        )

    with llm_usage_session() as usage_records:
        # Phase 1: Decompose and gap check
        decomposition_model = get_model_for_stage("decomposition", model)
        with llm_usage_stage(
            "decomposition",
            metadata=_llm_stage_metadata(
                model=decomposition_model,
                instrument_type=instrument_type,
                request_metadata=request_metadata,
                comparison_target=comparison_target,
            ),
        ):
            decomposition = decompose(effective_description, instrument_type, decomposition_model)
        if preferred_method:
            decomposition = replace(
                decomposition,
                method=normalize_method(preferred_method),
            )
        product_ir = decompose_to_ir(effective_description, instrument_type=instrument_type)
        gap_report = gap_check(decomposition)
        enforce_llm_token_budget(stage="decomposition")

        # Pre-flight build gate — block very-low-confidence builds early
        from trellis.agent.build_gate import evaluate_pre_flight_gate
        pre_flight_gate = evaluate_pre_flight_gate(gap_report)

        if pre_flight_gate.decision == "block":
            return BuildResult(
                payoff_cls=None,
                success=False,
                attempts=0,
                gap_confidence=gap_report.confidence,
                knowledge_gaps=gap_report.missing,
                failures=[f"Build gate blocked (pre-flight): {pre_flight_gate.reason}"],
                gate_decision=pre_flight_gate,
                post_build_tracking={
                    "active_flags": _post_build_control_flags(),
                    "events": [],
                },
            )

        # Increase retries if knowledge is thin
        if pre_flight_gate.decision == "narrow_route" or gap_report.confidence < 0.5:
            max_retries = max(max_retries, 4)

        # Phase 2: Build with gap-aware knowledge
        result = BuildResult(
            payoff_cls=None,
            success=False,
            attempts=0,
            gap_confidence=gap_report.confidence,
            knowledge_gaps=gap_report.missing,
            gate_decision=pre_flight_gate,
            post_build_tracking={
                "active_flags": _post_build_control_flags(),
                "events": [],
            },
        )

        try:
            payoff_cls, build_meta = _build_with_tracking(
                description=description,
                build_description=effective_description,
                instrument_type=instrument_type,
                decomposition=decomposition,
                product_ir=product_ir,
                gap_report=gap_report,
                model=model,
                market_state=market_state,
                max_retries=max_retries,
                validation=validation,
                force_rebuild=force_rebuild,
                fresh_build=fresh_build,
                preferred_method=preferred_method,
                request_metadata=request_metadata,
            )
            result.payoff_cls = payoff_cls
            result.success = payoff_cls is not None
            result.attempts = build_meta.get("attempts", 1)
            result.failures = build_meta.get("failures", [])
            result.code = build_meta.get("code", "")
            result.agent_observations = build_meta.get("agent_observations", [])
            result.knowledge_summary = dict(build_meta.get("knowledge_summary", {}) or {})
            result.platform_trace_path = build_meta.get("platform_trace_path")
            result.platform_request_id = build_meta.get("platform_request_id")
            result.analytical_trace_path = build_meta.get("analytical_trace_path")
            result.analytical_trace_text_path = build_meta.get("analytical_trace_text_path")
            result.audit_record_path = build_meta.get("audit_record_path")
            result.execution_module_name = build_meta.get("execution_module_name")
            result.execution_module_path = build_meta.get("execution_module_path")
            result.execution_file_path = build_meta.get("execution_file_path")
            result.admission_target_module_name = build_meta.get("admission_target_module_name")
            result.admission_target_module_path = build_meta.get("admission_target_module_path")
            result.admission_target_file_path = build_meta.get("admission_target_file_path")
            result.blocker_details = build_meta.get("blocker_details")
        except BuildTrackingFailure as exc:
            result.failures = [str(exc.cause)]
            result.attempts = exc.meta.get("attempts", 0)
            result.code = exc.meta.get("code", "")
            result.agent_observations = exc.meta.get("agent_observations", [])
            result.knowledge_summary = dict(exc.meta.get("knowledge_summary", {}) or {})
            result.platform_trace_path = exc.meta.get("platform_trace_path")
            result.platform_request_id = exc.meta.get("platform_request_id")
            result.analytical_trace_path = exc.meta.get("analytical_trace_path")
            result.analytical_trace_text_path = exc.meta.get("analytical_trace_text_path")
            result.audit_record_path = exc.meta.get("audit_record_path")
            result.execution_module_name = exc.meta.get("execution_module_name")
            result.execution_module_path = exc.meta.get("execution_module_path")
            result.execution_file_path = exc.meta.get("execution_file_path")
            result.admission_target_module_name = exc.meta.get("admission_target_module_name")
            result.admission_target_module_path = exc.meta.get("admission_target_module_path")
            result.admission_target_file_path = exc.meta.get("admission_target_file_path")
            result.blocker_details = exc.meta.get("blocker_details")
        except Exception as e:
            result.failures = [str(e)]
        _record_post_build_phase(
            result.post_build_tracking,
            "build_completed",
            status="ok" if result.success else "error",
            attempts=result.attempts,
            success=result.success,
            platform_trace_path=result.platform_trace_path,
            failure_count=len(result.failures),
        )

        # Phase 3: Post-build reflection (skip on obvious provider/config failures)
        post_build_flags = dict(result.post_build_tracking.get("active_flags") or {})
        if post_build_flags.get("skip_reflection"):
            result.reflection = {
                "skipped": True,
                "reason": _SKIP_POST_BUILD_REFLECTION_ENV,
            }
            _record_post_build_phase(
                result.post_build_tracking,
                "reflection_started",
                status="skipped",
                reason=_SKIP_POST_BUILD_REFLECTION_ENV,
            )
            _record_post_build_phase(
                result.post_build_tracking,
                "reflection_completed",
                status="skipped",
                reason=_SKIP_POST_BUILD_REFLECTION_ENV,
            )
        elif _should_skip_reflection(result):
            result.reflection = {
                "skipped": True,
                "reason": "provider_failure_before_build",
            }
            _record_post_build_phase(
                result.post_build_tracking,
                "reflection_started",
                status="skipped",
                reason="provider_failure_before_build",
            )
            _record_post_build_phase(
                result.post_build_tracking,
                "reflection_completed",
                status="skipped",
                reason="provider_failure_before_build",
            )
        else:
            _record_post_build_phase(
                result.post_build_tracking,
                "reflection_started",
                status="running",
            )
            try:
                reflection_model = get_model_for_stage("reflection", model)
                enforce_llm_token_budget(stage="pre_reflection")
                with llm_usage_stage(
                    "reflection",
                    metadata=_llm_stage_metadata(
                        model=reflection_model,
                        instrument_type=instrument_type,
                        request_metadata=request_metadata,
                        comparison_target=comparison_target,
                    ),
                ):
                    reflection = reflect_on_build(
                        description=description,
                        decomposition=decomposition,
                        gap_report=gap_report,
                        retrieved_lesson_ids=gap_report.retrieved_lesson_ids,
                        success=result.success,
                        failures=result.failures,
                        code=result.code,
                        attempt=result.attempts,
                        agent_observations=result.agent_observations,
                        model=reflection_model,
                    )
                result.reflection = reflection
                _record_post_build_phase(
                    result.post_build_tracking,
                    "reflection_completed",
                    status="ok",
                    lessons_attributed=int(reflection.get("lessons_attributed") or 0),
                )
            except TokenBudgetExceeded as exc:
                result.reflection = {"token_budget_exceeded": str(exc)}
                _record_post_build_phase(
                    result.post_build_tracking,
                    "reflection_completed",
                    status="token_budget_exceeded",
                    error=str(exc)[:200],
                )
            except Exception as exc:
                result.reflection = {"error": str(exc)[:200]}
                _record_post_build_phase(
                    result.post_build_tracking,
                    "reflection_completed",
                    status="error",
                    error=str(exc)[:200],
                )

        result.token_usage_summary = summarize_llm_usage(usage_records)
        if result.platform_trace_path and result.token_usage_summary["call_count"] > 0:
            try:
                attach_platform_trace_token_usage(
                    result.platform_trace_path,
                    result.token_usage_summary,
                )
                _record_post_build_phase(
                    result.post_build_tracking,
                    "token_usage_attached",
                    status="ok",
                    call_count=result.token_usage_summary.get("call_count"),
                    total_tokens=result.token_usage_summary.get("total_tokens"),
                )
            except Exception:
                _record_post_build_phase(
                    result.post_build_tracking,
                    "token_usage_attached",
                    status="error",
                    call_count=result.token_usage_summary.get("call_count"),
                )
        else:
            _record_post_build_phase(
                result.post_build_tracking,
                "token_usage_attached",
                status="skipped",
                reason=(
                    "missing_platform_trace"
                    if not result.platform_trace_path
                    else "no_llm_calls"
                ),
                call_count=result.token_usage_summary.get("call_count"),
            )

    # Phase 3b: Emit decision checkpoint (best-effort, never blocks)
    checkpoint_status = _emit_decision_checkpoint(
        result=result,
        decomposition=decomposition,
        instrument_type=instrument_type,
        model=model or "",
    )
    _record_post_build_phase(
        result.post_build_tracking,
        "decision_checkpoint_emitted",
        status=str(checkpoint_status.get("status") or "unknown"),
        checkpoint_path=checkpoint_status.get("path"),
        error=checkpoint_status.get("error"),
    )

    # Phase 4: Post-task consolidation (background, non-blocking)
    if post_build_flags.get("skip_consolidation"):
        _record_post_build_phase(
            result.post_build_tracking,
            "consolidation_dispatched",
            status="skipped",
            reason=_SKIP_POST_BUILD_CONSOLIDATION_ENV,
        )
    else:
        consolidation = _maybe_consolidate(result.reflection, model=model)
        _record_post_build_phase(
            result.post_build_tracking,
            "consolidation_dispatched",
            status=(
                "backgrounded"
                if consolidation is None
                else ("error" if consolidation.error else "ok")
            ),
            tiers=getattr(consolidation, "tiers_run", []),
            trigger_reasons=getattr(consolidation, "trigger_reasons", []),
            error=getattr(consolidation, "error", None),
        )

    return result


def _build_with_tracking(
    description: str,
    instrument_type: str | None,
    decomposition,
    product_ir,
    gap_report,
    model: str | None,
    market_state,
    max_retries: int,
    validation: str,
    force_rebuild: bool,
    fresh_build: bool = False,
    build_description: str | None = None,
    preferred_method: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> tuple[type | None, dict]:
    """Run ``build_payoff()`` while capturing metadata needed by the reflect phase.

    Returns ``(payoff_cls, metadata)`` where *payoff_cls* is the generated
    class (or None), and *metadata* is a dict with keys: ``attempts`` (int),
    ``failures`` (list[str]), ``code`` (str of the last generated module),
    ``platform_trace_path``, and ``platform_request_id``.
    """
    from trellis.agent.knowledge import (
        build_shared_knowledge_payload,
        retrieve_for_product_ir,
        retrieve_for_task,
    )
    from trellis.agent.knowledge.gap_check import format_gap_warnings
    from trellis.agent.knowledge.store import expand_features
    from trellis.agent.executor import build_payoff

    build_description = build_description or description
    _reset_deterministic_planning_caches()

    # Build enhanced knowledge context with gap warnings
    if product_ir is not None:
        knowledge = retrieve_for_product_ir(
            product_ir,
            preferred_method=preferred_method or decomposition.method,
        )
    else:
        knowledge = retrieve_for_task(
            method=decomposition.method,
            features=list(decomposition.features),
            instrument=decomposition.instrument,
        )
    knowledge_payload = build_shared_knowledge_payload(knowledge)
    meta: dict = {
        "knowledge_summary": dict(knowledge_payload.get("summary") or {}),
    }

    def _retrieve_live_payload() -> dict[str, Any]:
        """Render a fresh shared-knowledge payload for each retry attempt."""
        if product_ir is not None:
            latest_knowledge = retrieve_for_product_ir(
                product_ir,
                preferred_method=preferred_method or decomposition.method,
            )
        else:
            latest_knowledge = retrieve_for_task(
                method=decomposition.method,
                features=list(decomposition.features),
                instrument=decomposition.instrument,
            )
        return build_shared_knowledge_payload(latest_knowledge)

    import trellis.agent.executor as executor
    original_record_platform_event = executor._record_platform_event

    def _stage_aware_knowledge_retriever(request) -> str:
        """Serve fresh knowledge with audience/surface awareness on every attempt."""
        try:
            payload = _retrieve_live_payload()
        except Exception:
            payload = knowledge_payload

        audience = getattr(request, "audience", "builder")
        knowledge_surface = getattr(request, "knowledge_surface", "compact")
        prompt_surface = getattr(request, "prompt_surface", "compact")
        expanded = knowledge_surface == "expanded"
        if audience == "review":
            if expanded:
                return payload["review_text_expanded"]
            return payload["review_text_distilled"]
        if expanded:
            text = payload["builder_text_expanded"]
        elif prompt_surface == "import_repair":
            text = payload["builder_text"]
        else:
            text = payload["builder_text_distilled"]
        gap_warnings = format_gap_warnings(gap_report)
        if gap_warnings:
            text += "\n\n" + gap_warnings
        return text

    meta.setdefault("attempts", 0)
    meta.setdefault("failures", [])
    meta.setdefault("code", "")
    try:
        # Track attempts by wrapping _generate_module
        original_generate = executor._generate_module
        attempt_count = [0]
        last_code = [""]
        platform_trace_path = [None]
        platform_request_id = [None]

        def _tracking_generate(*args, **kwargs):
            """Wrap module generation so attempt count and last code are captured."""
            attempt_count[0] += 1
            code = original_generate(*args, **kwargs)
            last_code[0] = _generated_module_code_text(code)
            return code

        def _tracking_record_platform_event(compiled_request, event, **kwargs):
            """Capture trace path metadata while delegating to the original recorder."""
            if compiled_request is not None:
                request = getattr(compiled_request, "request", None)
                request_id = getattr(request, "request_id", None)
                if request_id:
                    from trellis.agent.platform_traces import TRACE_ROOT

                    platform_request_id[0] = request_id
                    platform_trace_path[0] = str((TRACE_ROOT / f"{request_id}.yaml").resolve())
            return original_record_platform_event(compiled_request, event, **kwargs)

        executor._generate_module = _tracking_generate
        executor._record_platform_event = _tracking_record_platform_event

        payoff_cls = build_payoff(
            build_description,
            model=model,
            market_state=market_state,
            instrument_type=instrument_type,
            force_rebuild=force_rebuild,
            fresh_build=fresh_build,
            validation=validation,
            max_retries=max_retries,
            preferred_method=preferred_method,
            request_metadata=request_metadata,
            build_meta=meta,
            gap_report=gap_report,
            knowledge_retriever=_stage_aware_knowledge_retriever,
        )

        meta["attempts"] = attempt_count[0]
        meta["code"] = last_code[0]
        meta["platform_trace_path"] = platform_trace_path[0]
        meta["platform_request_id"] = platform_request_id[0]
        return payoff_cls, meta

    except Exception as e:
        meta["attempts"] = attempt_count[0] if 'attempt_count' in dir() else 0
        meta["failures"] = [str(e)]
        meta["code"] = last_code[0] if 'last_code' in dir() else ""
        meta["platform_trace_path"] = platform_trace_path[0] if 'platform_trace_path' in dir() else None
        meta["platform_request_id"] = platform_request_id[0] if 'platform_request_id' in dir() else None
        raise BuildTrackingFailure(str(e), meta=meta, cause=e) from e
    finally:
        # Restore original functions
        executor._record_platform_event = original_record_platform_event
        if 'original_generate' in dir():
            executor._generate_module = original_generate


def _reset_deterministic_planning_caches() -> None:
    """Clear deterministic planner caches before each autonomous build attempt.

    Comparison-target runs and fresh-build proof passes should not inherit
    route/binding decisions from earlier in-process tasks. Recomputing these
    small deterministic surfaces is cheap relative to the live build.
    """
    from trellis.agent.backend_bindings import clear_backend_binding_catalog_cache
    from trellis.agent.codegen_guardrails import clear_generation_plan_cache
    from trellis.agent.route_registry import clear_route_registry_cache

    clear_generation_plan_cache()
    clear_backend_binding_catalog_cache()
    clear_route_registry_cache()


def _should_skip_reflection(result: BuildResult) -> bool:
    """Skip reflective LLM calls when the failure is provider-side and produced no code."""
    if result.success or not result.failures:
        return False
    if (result.code or "").strip():
        return False
    failures = [failure.lower() for failure in result.failures]
    return all(any(marker in failure for marker in _PROVIDER_FAILURE_MARKERS) for failure in failures)


def _generated_module_code_text(value: object) -> str:
    """Return the source text from a generated-module result or a legacy string."""
    if isinstance(value, str):
        return value
    code = getattr(value, "code", None)
    if isinstance(code, str):
        return code
    if code is not None:
        return str(code)
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# Post-task consolidation skill
# ---------------------------------------------------------------------------

# Thresholds for triggering consolidation tiers
_CANDIDATE_BACKLOG_RATIO = 0.40  # 40%+ candidates → recalibrate
_TRACE_BLOAT_COUNT = 500  # 500+ trace files → compact
_SUPERSEDES_GAP_COUNT = 10  # 10+ unscanned promoted → backfill
_GAP_NOISE_COUNT = 200  # 200+ gap entries → aggregate


@dataclass
class ConsolidationResult:
    """Outcome of a post-task consolidation run."""

    triggered: bool = False
    tiers_run: list[int] = field(default_factory=list)
    tier_results: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    trigger_reasons: list[str] = field(default_factory=list)
    error: str | None = None


def _assess_consolidation_needs() -> tuple[list[int], list[str]]:
    """Determine which consolidation tiers should run based on current state.

    Returns (tiers_to_run, trigger_reasons).  All checks are cheap
    file-system reads — no LLM calls.
    """
    from pathlib import Path

    tiers: list[int] = []
    reasons: list[str] = []

    knowledge_dir = Path(__file__).resolve().parent
    entries_dir = knowledge_dir / "lessons" / "entries"
    traces_dir = knowledge_dir / "traces"

    # --- Tier 1: candidate backlog ---
    if entries_dir.is_dir():
        import yaml

        total = 0
        candidates = 0
        for f in entries_dir.glob("*.yaml"):
            total += 1
            try:
                data = yaml.safe_load(f.read_text())
                if isinstance(data, dict) and data.get("status") == "candidate":
                    candidates += 1
            except Exception:
                pass
        if total > 0 and candidates / total > _CANDIDATE_BACKLOG_RATIO:
            tiers.append(1)
            reasons.append(f"candidate_backlog ({candidates}/{total} = {candidates / total:.0%})")

    # --- Tier 2: trace bloat ---
    if traces_dir.is_dir():
        legacy_trace_count = sum(
            1
            for f in traces_dir.glob("*.yaml")
            if f.is_file() and f.name not in {".gitignore", "consolidation_log.yaml", "gap_registry.yaml"}
        )
        platform_trace_dir = traces_dir / "platform"
        platform_event_log_count = (
            sum(1 for f in platform_trace_dir.glob("*.events.ndjson") if f.is_file())
            if platform_trace_dir.is_dir()
            else 0
        )
        trace_count = legacy_trace_count + platform_event_log_count
        if trace_count > _TRACE_BLOAT_COUNT:
            tiers.append(2)
            reasons.append(
                "trace_bloat "
                f"({trace_count} compactable artifacts: "
                f"{legacy_trace_count} legacy traces, "
                f"{platform_event_log_count} platform event logs)"
            )

    # --- Tier 3: supersedes gap ---
    if entries_dir.is_dir():
        promoted_unscanned = 0
        for f in entries_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text())
                if isinstance(data, dict) and data.get("status") == "promoted":
                    supersedes = data.get("supersedes")
                    if not supersedes or supersedes == []:
                        promoted_unscanned += 1
            except Exception:
                pass
        if promoted_unscanned > _SUPERSEDES_GAP_COUNT:
            tiers.append(3)
            reasons.append(f"supersedes_gap ({promoted_unscanned} unscanned)")

    # --- Tier 4: principle opportunity ---
    # Detected by counting categories with 3+ promoted lessons
    if entries_dir.is_dir():
        from collections import Counter

        category_counts: Counter[str] = Counter()
        for f in entries_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text())
                if isinstance(data, dict) and data.get("status") == "promoted":
                    cat = data.get("category", "unknown")
                    category_counts[cat] += 1
            except Exception:
                pass
        eligible = sum(1 for c in category_counts.values() if c >= 3)
        # Only trigger if there are eligible categories AND no recent draft
        if eligible > 0:
            candidates_dir = traces_dir / "principle_candidates"
            recent_drafts = 0
            if candidates_dir.is_dir():
                import time

                now = time.time()
                for f in candidates_dir.glob("*.yaml"):
                    try:
                        if now - f.stat().st_mtime < 86400 * 7:  # within 7 days
                            recent_drafts += 1
                    except Exception:
                        pass
            if recent_drafts == 0:
                tiers.append(4)
                reasons.append(f"principle_opportunity ({eligible} categories eligible)")

    if not tiers:
        # Always run Tier 1 as a lightweight maintenance pass
        tiers.append(1)
        reasons.append("routine_maintenance")

    return tiers, reasons


def _run_consolidation(
    reflection: dict[str, Any],
    tiers: list[int],
    *,
    model: str | None = None,
) -> ConsolidationResult:
    """Execute the selected consolidation tiers.

    Tier 1: recalibrate candidates (always, cheap)
    Tier 2: compact traces (when trace count > threshold)
    Tier 3: backfill supersedes (dry_run=True — log proposals only)
    Tier 4: draft principle candidates (needs LLM, advisory only)
    """
    import time

    result = ConsolidationResult(triggered=True, tiers_run=list(tiers))
    start = time.monotonic()

    try:
        from trellis.agent.knowledge.promotion import (
            backfill_supersedes,
            compact_traces,
            draft_principle_candidates,
            recalibrate_candidates,
        )

        # Tier 1: recalibrate stuck candidates
        if 1 in tiers:
            try:
                rc = recalibrate_candidates()
                result.tier_results["recalibrate"] = rc
            except Exception as e:
                result.tier_results["recalibrate"] = {"error": str(e)}

        # Tier 2: compact old traces
        if 2 in tiers:
            try:
                ct = compact_traces(older_than_days=30)
                result.tier_results["compact_traces"] = ct
            except Exception as e:
                result.tier_results["compact_traces"] = {"error": str(e)}

        # Tier 3: backfill supersedes (dry_run=True only — no mutations)
        if 3 in tiers:
            try:
                bs = backfill_supersedes(dry_run=True)
                result.tier_results["backfill_supersedes_dry"] = {
                    "proposals": {k: v for k, v in bs.items()} if bs else {},
                    "count": len(bs),
                }
            except Exception as e:
                result.tier_results["backfill_supersedes_dry"] = {"error": str(e)}

        # Tier 4: draft principle candidates (advisory, no auto-promote)
        if 4 in tiers and model:
            try:
                pc = draft_principle_candidates(model=model)
                result.tier_results["principle_candidates"] = {
                    "drafted": len(pc),
                    "categories": [c.get("category", "?") for c in pc],
                }
            except Exception as e:
                result.tier_results["principle_candidates"] = {"error": str(e)}

    except Exception as e:
        result.error = str(e)

    result.duration_seconds = round(time.monotonic() - start, 2)
    _log_consolidation(result)
    return result


def _log_consolidation(result: ConsolidationResult) -> None:
    """Append consolidation outcome to traces/consolidation_log.yaml."""
    from pathlib import Path
    import datetime

    log_path = Path(__file__).resolve().parent / "traces" / "consolidation_log.yaml"
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "triggered": result.triggered,
        "tiers_run": result.tiers_run,
        "trigger_reasons": result.trigger_reasons,
        "tier_results": result.tier_results,
        "duration_seconds": result.duration_seconds,
    }
    if result.error:
        entry["error"] = result.error

    try:
        import yaml

        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if log_path.exists():
            raw = yaml.safe_load(log_path.read_text())
            if isinstance(raw, list):
                existing = raw
        existing.append(entry)
        log_path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=False))
    except Exception:
        pass  # logging failure must not break the build


def _emit_decision_checkpoint(
    *,
    result: BuildResult,
    decomposition: object,
    instrument_type: str | None,
    model: str,
) -> dict[str, Any]:
    """Best-effort: save a structured decision checkpoint for drift detection.

    Never raises — checkpoint emission must not block the pipeline.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        from trellis.agent.checkpoints import capture_checkpoint, save_checkpoint
        from trellis.agent.platform_traces import TRACE_ROOT, load_platform_trace_boundary

        # Extract quant decision from agent_observations
        pricing_plan_proxy = None
        spec_schema_proxy = None
        for obs in result.agent_observations:
            if obs.get("agent") == "quant" and obs.get("kind") == "decision":
                details = obs.get("details", {})
                from types import SimpleNamespace

                pricing_plan_proxy = SimpleNamespace(
                    method=details.get("method", getattr(decomposition, "method", "unknown")),
                    required_market_data=set(details.get("required_market_data", [])),
                    method_modules=details.get("method_modules", []),
                    selection_reason=details.get("selection_reason", ""),
                )
                break

        # Derive task_id from platform request or instrument type
        task_id = "unknown"
        if result.platform_request_id:
            task_id = result.platform_request_id[:12]
        elif instrument_type:
            task_id = instrument_type

        outcome = "pass" if result.success else "fail_build"
        if not result.success and result.failures:
            if any("validation" in f.lower() for f in result.failures):
                outcome = "fail_validate"

        trace_path = result.platform_trace_path
        if not trace_path and result.platform_request_id:
            candidate = TRACE_ROOT / f"{result.platform_request_id}.yaml"
            if candidate.exists():
                trace_path = str(candidate)
        trace_boundary = (
            load_platform_trace_boundary(trace_path)
            if trace_path
            else {}
        )

        checkpoint = capture_checkpoint(
            task_id=task_id,
            instrument_type=instrument_type or "unknown",
            pricing_plan=pricing_plan_proxy,
            code=result.code or None,
            token_summary=result.token_usage_summary,
            semantic_checkpoint=trace_boundary.get("semantic_checkpoint"),
            generation_boundary=trace_boundary.get("generation_boundary"),
            validation_contract=trace_boundary.get("validation_contract"),
            outcome=outcome,
            attempts=result.attempts,
            model=model,
        )
        path = save_checkpoint(checkpoint)
        return {
            "status": "ok",
            "path": str(path) if path is not None else "",
        }
    except Exception as exc:
        _log.debug("_emit_decision_checkpoint: %s", exc)
        return {
            "status": "error",
            "error": str(exc)[:200],
        }


def _maybe_consolidate(
    reflection: dict[str, Any],
    *,
    model: str | None = None,
    background: bool = True,
) -> ConsolidationResult | None:
    """Post-task consolidation entry point.

    Checks conditions and runs consolidation tiers.  By default runs in a
    background daemon thread so it does not block the next pricing task.
    Returns the ConsolidationResult if run synchronously, None if backgrounded.
    """
    tiers, reasons = _assess_consolidation_needs()

    if not tiers:
        return ConsolidationResult(triggered=False)

    # Tier 4 (LLM-based) should not run in background to avoid rate-limit
    # contention with pricing calls.  Defer it if backgrounding.
    bg_tiers = [t for t in tiers if t != 4] if background else tiers
    fg_tier4 = 4 in tiers and not background

    if background and bg_tiers:
        import threading

        def _bg_run():
            r = _run_consolidation(reflection, bg_tiers, model=model)
            r.trigger_reasons = reasons
            return r

        t = threading.Thread(target=_bg_run, daemon=True)
        t.start()

        # If tier 4 was requested but deferred, log it
        if 4 in tiers and 4 not in bg_tiers:
            _log_consolidation(ConsolidationResult(
                triggered=False,
                trigger_reasons=["tier4_deferred_from_background"],
            ))

        return None

    result = _run_consolidation(reflection, tiers, model=model)
    result.trigger_reasons = reasons
    return result
