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

from dataclasses import replace
from dataclasses import dataclass, field
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
    blocker_details: dict[str, Any] | None = None
    token_usage_summary: dict[str, Any] = field(default_factory=dict)


_PROVIDER_FAILURE_MARKERS = (
    "llm provider",
    "openai ",
    "not_found_error",
    "insufficient_quota",
    "returned invalid json response",
    "returned empty json response",
    "request failed after",
)


def build_with_knowledge(
    description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    market_state=None,
    max_retries: int = 3,
    validation: str = "standard",
    force_rebuild: bool = False,
    preferred_method: str | None = None,
    comparison_target: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> BuildResult:
    """Knowledge-aware build: gap check → build → reflect → enrich.

    This is the autonomous entry point that manages the knowledge lifecycle
    around each build.  It replaces direct calls to build_payoff() when
    you want the system to self-maintain its knowledge base.

    Returns a BuildResult with the payoff class and metadata about what
    the knowledge system learned.
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
        with llm_usage_stage("decomposition", metadata={"model": decomposition_model}):
            decomposition = decompose(effective_description, instrument_type, decomposition_model)
        if preferred_method:
            decomposition = replace(
                decomposition,
                method=normalize_method(preferred_method),
            )
        product_ir = decompose_to_ir(effective_description, instrument_type=instrument_type)
        gap_report = gap_check(decomposition)
        enforce_llm_token_budget(stage="decomposition")

        # Increase retries if knowledge is thin
        if gap_report.confidence < 0.5:
            max_retries = max(max_retries, 4)

        # Phase 2: Build with gap-aware knowledge
        result = BuildResult(
            payoff_cls=None,
            success=False,
            attempts=0,
            gap_confidence=gap_report.confidence,
            knowledge_gaps=gap_report.missing,
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
            result.blocker_details = build_meta.get("blocker_details")
        except BuildTrackingFailure as exc:
            result.failures = [str(exc.cause)]
            result.attempts = exc.meta.get("attempts", 0)
            result.code = exc.meta.get("code", "")
            result.agent_observations = exc.meta.get("agent_observations", [])
            result.knowledge_summary = dict(exc.meta.get("knowledge_summary", {}) or {})
            result.platform_trace_path = exc.meta.get("platform_trace_path")
            result.platform_request_id = exc.meta.get("platform_request_id")
            result.blocker_details = exc.meta.get("blocker_details")
        except Exception as e:
            result.failures = [str(e)]

        # Phase 3: Post-build reflection (skip on obvious provider/config failures)
        if _should_skip_reflection(result):
            result.reflection = {
                "skipped": True,
                "reason": "provider_failure_before_build",
            }
        else:
            try:
                reflection_model = get_model_for_stage("reflection", model)
                enforce_llm_token_budget(stage="pre_reflection")
                with llm_usage_stage("reflection", metadata={"model": reflection_model}):
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
            except TokenBudgetExceeded as exc:
                result.reflection = {"token_budget_exceeded": str(exc)}
            except Exception:
                pass

        result.token_usage_summary = summarize_llm_usage(usage_records)
        if result.platform_trace_path and result.token_usage_summary["call_count"] > 0:
            try:
                attach_platform_trace_token_usage(
                    result.platform_trace_path,
                    result.token_usage_summary,
                )
            except Exception:
                pass

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
    build_description: str | None = None,
    preferred_method: str | None = None,
    request_metadata: Mapping[str, object] | None = None,
) -> tuple[type | None, dict]:
    """Run build_payoff() with metadata tracking for reflection.

    Returns (payoff_cls, metadata) where metadata includes attempts,
    failures, and final code.
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
    compact_knowledge_text = knowledge_payload["builder_text"]
    expanded_knowledge_text = knowledge_payload["builder_text_expanded"]
    knowledge_text = compact_knowledge_text
    gap_warnings = format_gap_warnings(gap_report)
    if gap_warnings:
        knowledge_text += "\n\n" + gap_warnings
        expanded_knowledge_text += "\n\n" + gap_warnings

    # Monkey-patch _retrieve_knowledge to return our pre-built context
    # This is a pragmatic approach — avoids deep refactoring of build_payoff
    import trellis.agent.executor as executor
    original_retrieve = executor._retrieve_knowledge
    original_record_platform_event = executor._record_platform_event

    def _patched_retrieve(pricing_plan, inst_type, *, product_ir=None, compact=True):
        """Return the precomputed knowledge context instead of re-querying storage."""
        return knowledge_text if compact else expanded_knowledge_text

    executor._retrieve_knowledge = _patched_retrieve

    meta: dict = {"attempts": 0, "failures": [], "code": ""}
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
            last_code[0] = code
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
            validation=validation,
            max_retries=max_retries,
            preferred_method=preferred_method,
            request_metadata=request_metadata,
            build_meta=meta,
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
        executor._retrieve_knowledge = original_retrieve
        executor._record_platform_event = original_record_platform_event
        if 'original_generate' in dir():
            executor._generate_module = original_generate


def _should_skip_reflection(result: BuildResult) -> bool:
    """Skip reflective LLM calls when the failure is provider-side and produced no code."""
    if result.success or not result.failures:
        return False
    if (result.code or "").strip():
        return False
    failures = [failure.lower() for failure in result.failures]
    return all(any(marker in failure for marker in _PROVIDER_FAILURE_MARKERS) for failure in failures)
