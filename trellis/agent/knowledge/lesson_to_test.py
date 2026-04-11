"""Deterministic lesson-to-regression payload materialization."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from trellis.agent.knowledge.schema import (
    AppliesWhen,
    Lesson,
    LessonRegressionPayload,
    LessonRegressionTemplate,
    LessonStatus,
)
from trellis.agent.knowledge.store import KnowledgeStore

_ELIGIBLE_STATUSES = {
    LessonStatus.PROMOTED,
    LessonStatus.VALIDATED,
}

_BASE_CATEGORY_FAMILIES = {
    "market_data": "dependency_resilience_regression",
    "numerical": "codegen_safety_regression",
    "monte_carlo": "method_contract_regression",
    "finite_differences": "method_contract_regression",
    "backward_induction": "method_contract_regression",
    "convention": "data_contract_regression",
    "volatility": "data_contract_regression",
    "calibration": "data_contract_regression",
    "credit_risk": "data_contract_regression",
    "contract": "data_contract_regression",
    "testing": "harness_regression",
}

_ROUTE_BOUNDARY_MARKERS = (
    "unexpected keyword argument 'market_state'",
    "pricing kernel signature",
    "route_helper",
    "route helper signature",
    "route helper not called",
    "exact binding",
    "helper-backed route",
    "wrapper signature",
    "route authority",
)

_BRIDGE_FALLBACK_MARKERS = (
    "compatibility name",
    "canonical concept",
    "thin wrapper",
    "compatibility bridge",
    "thin compatibility wrapper",
    "fallback drift",
    "bridge drift",
)

_LOWERING_ADMISSIBILITY_MARKERS = (
    "no exact backend binding",
    "no constructive steps",
    "route admissibility failed",
    "admissibility",
    "constructive route",
    "unsupported lane",
    "decision=clarification",
)

_SEMANTIC_CONTRACT_MARKERS = (
    "missing contract fields",
    "semantic_product_shape",
    "underlier_structure",
    "payoff_rule",
    "settlement_rule",
    "observation_schedule",
    "selection_scope",
    "selection_operator",
    "lock_rule",
    "semantic validation",
    "semantic contract",
)

_TEMPLATES = {
    "dependency_resilience_regression": LessonRegressionTemplate(
        family="dependency_resilience_regression",
        target_test_file="tests/test_agent/test_task_runtime.py",
        description="Guard transient dependency failure handling and fallback behavior.",
        assertion_focus=(
            "external_request_retry_present",
            "fallback_path_prevents_hard_abort",
        ),
        fixture_hints=(
            "simulate_transient_dependency_failure",
            "task_runtime_retry_policy",
        ),
        tags=("knowledge", "resilience"),
    ),
    "codegen_safety_regression": LessonRegressionTemplate(
        family="codegen_safety_regression",
        target_test_file="tests/test_agent/test_executor.py",
        description="Guard generated-source syntax and compile-time validation.",
        assertion_focus=(
            "generated_module_compiles",
            "syntax_guard_present",
        ),
        fixture_hints=(
            "compile_generated_module",
            "generated_source_smoke",
        ),
        tags=("knowledge", "codegen"),
    ),
    "method_contract_regression": LessonRegressionTemplate(
        family="method_contract_regression",
        target_test_file="tests/test_agent/test_executor.py",
        description="Guard helper usage and runtime method contracts.",
        assertion_focus=(
            "runtime_contract_preserved",
            "helper_invocation_shape_preserved",
        ),
        fixture_hints=(
            "assemble_generation_plan",
            "method_contract_smoke",
        ),
        tags=("knowledge", "method"),
    ),
    "data_contract_regression": LessonRegressionTemplate(
        family="data_contract_regression",
        target_test_file="tests/test_agent/test_knowledge_store.py",
        description="Guard data contracts, unit normalization, and convention retrieval.",
        assertion_focus=(
            "data_contract_present",
            "normalization_guard_present",
        ),
        fixture_hints=(
            "retrieve_for_task",
            "knowledge_contract_fixture",
        ),
        tags=("knowledge", "contract"),
    ),
    "harness_regression": LessonRegressionTemplate(
        family="harness_regression",
        target_test_file="tests/test_agent/test_reflect_loop.py",
        description="Guard the learning loop and deterministic knowledge artifact capture.",
        assertion_focus=(
            "lesson_pipeline_remains_deterministic",
            "knowledge_artifact_provenance_preserved",
        ),
        fixture_hints=(
            "reflect_on_build",
            "knowledge_root_fixture",
        ),
        tags=("knowledge", "harness"),
    ),
    "semantic_contract_regression": LessonRegressionTemplate(
        family="semantic_contract_regression",
        target_test_file="tests/test_agent/test_semantic_contracts.py",
        description="Guard typed semantic contract fields and semantic validation boundaries.",
        assertion_focus=(
            "semantic_contract_fields_present",
            "semantic_validation_blocks_missing_shape",
        ),
        fixture_hints=(
            "compile_semantic_contract",
            "validate_semantics",
        ),
        tags=("knowledge", "semantic"),
    ),
    "lowering_admissibility_regression": LessonRegressionTemplate(
        family="lowering_admissibility_regression",
        target_test_file="tests/test_agent/test_platform_requests.py",
        description="Guard lowering and admissibility boundaries before generation.",
        assertion_focus=(
            "lowering_gate_blocks_unsupported_lane",
            "constructive_plan_required_for_lane",
        ),
        fixture_hints=(
            "compile_build_request",
            "fallback_lane_plan_fixture",
        ),
        tags=("knowledge", "lowering"),
    ),
    "route_boundary_regression": LessonRegressionTemplate(
        family="route_boundary_regression",
        target_test_file="tests/test_agent/test_semantic_validators.py",
        description="Guard route-helper and exact-binding authority boundaries.",
        assertion_focus=(
            "route_helper_signature_preserved",
            "unsupported_wrapper_plumbing_blocked",
        ),
        fixture_hints=(
            "validate_algorithm_contract",
            "route_helper_contract_fixture",
        ),
        tags=("knowledge", "route"),
    ),
    "bridge_fallback_regression": LessonRegressionTemplate(
        family="bridge_fallback_regression",
        target_test_file="tests/test_agent/test_checkpoints.py",
        description="Guard compatibility bridges and thin-wrapper fallback drift.",
        assertion_focus=(
            "compatibility_bridge_preserved",
            "fallback_drift_detected",
        ),
        fixture_hints=(
            "semantic_checkpoint_fixture",
            "platform_trace_boundary",
        ),
        tags=("knowledge", "bridge"),
    ),
}


def classify_lesson_regression_family(lesson: Lesson) -> str | None:
    """Return the deterministic regression-template family for one lesson."""
    if lesson.status not in _ELIGIBLE_STATUSES:
        return None

    text = _lesson_text(lesson)
    if _contains_any(text, _ROUTE_BOUNDARY_MARKERS):
        return "route_boundary_regression"
    if _contains_any(text, _BRIDGE_FALLBACK_MARKERS):
        return "bridge_fallback_regression"
    if _contains_any(text, _LOWERING_ADMISSIBILITY_MARKERS):
        return "lowering_admissibility_regression"
    if (
        lesson.category in {"semantic", "semantic_contract"}
        or _contains_any(text, _SEMANTIC_CONTRACT_MARKERS)
    ):
        return "semantic_contract_regression"
    return _BASE_CATEGORY_FAMILIES.get(lesson.category, "data_contract_regression")


def materialize_lesson_regression(lesson: Lesson) -> LessonRegressionPayload | None:
    """Materialize one deterministic regression payload from a lesson."""
    family = classify_lesson_regression_family(lesson)
    if family is None:
        return None
    template = _TEMPLATES[family]
    rationale = _build_rationale(lesson, template)
    payload = LessonRegressionPayload(
        lesson_id=lesson.id,
        lesson_title=lesson.title,
        lesson_category=lesson.category,
        lesson_status=lesson.status,
        template_family=template.family,
        target_test_file=template.target_test_file,
        applies_when=lesson.applies_when,
        rationale=rationale,
        assertion_focus=template.assertion_focus,
        fixture_hints=template.fixture_hints,
        tags=template.tags,
        source_trace=lesson.source_trace,
    )
    return replace(
        payload,
        rendered_fragment=render_lesson_regression_fragment(payload),
    )


def materialize_lesson_regressions(
    lessons: Iterable[Lesson],
) -> tuple[LessonRegressionPayload, ...]:
    """Materialize regression payloads for all eligible lessons."""
    payloads: list[LessonRegressionPayload] = []
    for lesson in lessons:
        payload = materialize_lesson_regression(lesson)
        if payload is not None:
            payloads.append(payload)
    return tuple(payloads)


def materialize_store_regressions(
    store: KnowledgeStore,
    *,
    lesson_ids: tuple[str, ...] | None = None,
) -> tuple[LessonRegressionPayload, ...]:
    """Materialize deterministic regression payloads from store-backed lessons."""
    lessons = store.list_lessons(lesson_ids=lesson_ids)
    return materialize_lesson_regressions(lessons)


def render_lesson_regression_fragment(payload: LessonRegressionPayload) -> str:
    """Render one reviewable pytest-style regression fragment."""
    test_name = _test_name(payload.lesson_id, payload.lesson_title)
    methods = ", ".join(f"`{item}`" for item in payload.applies_when.method) or "`any`"
    features = ", ".join(f"`{item}`" for item in payload.applies_when.features) or "`none`"
    instruments = ", ".join(f"`{item}`" for item in payload.applies_when.instrument) or "`generic`"
    docstring_text = _sanitize_fragment_text(
        f"Regression guard for {payload.lesson_id}: {payload.lesson_title}."
    )
    rationale = _sanitize_fragment_text(payload.rationale)
    lines = [
        f"def {test_name}():",
        f'    """{docstring_text}"""',
        f"    # Template family: `{payload.template_family}`",
        f"    # Target file hint: `{payload.target_test_file}`",
        f"    # Applies when: method={methods}; features={features}; instrument={instruments}",
        f"    # Rationale: {rationale}",
    ]
    if payload.assertion_focus:
        lines.append("    # Assertion focus:")
        lines.extend(
            f"    # - {assertion}" for assertion in payload.assertion_focus
        )
    if payload.fixture_hints:
        lines.append("    # Fixture hints:")
        lines.extend(
            f"    # - {fixture}" for fixture in payload.fixture_hints
        )
    return "\n".join(lines)


def _build_rationale(
    lesson: Lesson,
    template: LessonRegressionTemplate,
) -> str:
    method = ", ".join(lesson.applies_when.method) or "any method"
    features = ", ".join(lesson.applies_when.features[:4]) or "generic features"
    return (
        f"{template.description} Triggered by {method} on {features}; "
        f"encode the lesson fix as a deterministic guard instead of relying on replayed reflection."
    )


def _lesson_text(lesson: Lesson) -> str:
    fragments = [
        lesson.title,
        lesson.category,
        lesson.symptom,
        lesson.root_cause,
        lesson.fix,
        lesson.validation,
        *lesson.applies_when.method,
        *lesson.applies_when.features,
        *lesson.applies_when.instrument,
    ]
    return " ".join(fragment for fragment in fragments if fragment).lower()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _identifier_fragment(value: str, *, fallback: str) -> str:
    fragment = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    fragment = re.sub(r"_+", "_", fragment)
    if not fragment:
        fragment = fallback
    if fragment[0].isdigit():
        fragment = f"n_{fragment}"
    return fragment


def _sanitize_fragment_text(value: str) -> str:
    sanitized = str(value).replace('"""', '\\"\\"\\"')
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def _test_name(lesson_id: str, title: str) -> str:
    lesson_id_slug = _identifier_fragment(lesson_id, fallback="lesson")
    slug = _identifier_fragment(title, fallback="lesson_regression")
    return f"test_{lesson_id_slug}_{slug}"
