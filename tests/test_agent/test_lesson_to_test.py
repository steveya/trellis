"""Tests for deterministic lesson-to-regression payload materialization."""

from __future__ import annotations

from trellis.agent.knowledge.schema import AppliesWhen, Lesson, LessonStatus, Severity


def _lesson(
    *,
    lesson_id: str = "lesson_001",
    title: str = "Lesson title",
    category: str = "numerical",
    status: LessonStatus = LessonStatus.PROMOTED,
    method: tuple[str, ...] = ("analytical",),
    features: tuple[str, ...] = ("discounting",),
    instrument: tuple[str, ...] = ("european_option",),
    symptom: str = "symptom",
    root_cause: str = "root cause",
    fix: str = "fix",
    source_trace: str | None = "/tmp/trace.yaml",
) -> Lesson:
    return Lesson(
        id=lesson_id,
        title=title,
        severity=Severity.HIGH,
        category=category,
        applies_when=AppliesWhen(
            method=method,
            features=features,
            instrument=instrument,
        ),
        symptom=symptom,
        root_cause=root_cause,
        fix=fix,
        validation="validation",
        confidence=0.9,
        status=status,
        source_trace=source_trace,
    )


def test_materialize_base_regression_payload_for_promoted_numerical_lesson():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="num_030",
            title="Payoff code generation indentation error",
            category="numerical",
            symptom="Generated payoff module fails to import due to SyntaxError.",
            root_cause="Template indentation and compile validation were missing.",
            fix="Dedent templates and compile generated modules before import.",
        )
    )

    assert payload is not None
    assert payload.lesson_id == "num_030"
    assert payload.template_family == "codegen_safety_regression"
    assert payload.target_test_file == "tests/test_agent/test_executor.py"
    assert "generated_module_compiles" in payload.assertion_focus
    assert "compile_generated_module" in payload.fixture_hints
    assert "test_num_030_payoff_code_generation_indentation_error" in payload.rendered_fragment


def test_materialize_lesson_regression_ignores_candidate_lessons():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="md_130",
            category="market_data",
            status=LessonStatus.CANDIDATE,
        )
    )

    assert payload is None


def test_materialize_store_regressions_only_uses_validated_or_promoted_lessons(monkeypatch):
    from trellis.agent.knowledge.lesson_to_test import materialize_store_regressions
    from trellis.agent.knowledge.schema import LessonIndex
    from trellis.agent.knowledge.store import KnowledgeStore

    store = KnowledgeStore()
    store._lesson_index = [
        LessonIndex(
            id="candidate_lesson",
            title="Candidate lesson",
            severity=Severity.MEDIUM,
            category="market_data",
            applies_when=AppliesWhen(method=("analytical",), features=("discount_curve",)),
            status=LessonStatus.CANDIDATE,
        ),
        LessonIndex(
            id="validated_lesson",
            title="Validated lesson",
            severity=Severity.HIGH,
            category="convention",
            applies_when=AppliesWhen(method=("credit",), features=("credit_risk",)),
            status=LessonStatus.VALIDATED,
        ),
        LessonIndex(
            id="promoted_lesson",
            title="Promoted lesson",
            severity=Severity.HIGH,
            category="numerical",
            applies_when=AppliesWhen(method=("analytical",), features=("discounting",)),
            status=LessonStatus.PROMOTED,
        ),
    ]

    def fake_load(lesson_id: str) -> Lesson:
        if lesson_id == "candidate_lesson":
            return _lesson(
                lesson_id=lesson_id,
                category="market_data",
                status=LessonStatus.CANDIDATE,
            )
        if lesson_id == "validated_lesson":
            return _lesson(
                lesson_id=lesson_id,
                category="convention",
                status=LessonStatus.VALIDATED,
                method=("credit",),
                features=("credit_risk",),
                fix="Add the missing convention contract and unit normalization.",
            )
        return _lesson(
            lesson_id=lesson_id,
            category="numerical",
            status=LessonStatus.PROMOTED,
            fix="Compile generated modules before import.",
        )

    monkeypatch.setattr(store, "_load_lesson", fake_load)

    payloads = materialize_store_regressions(store)

    assert [payload.lesson_id for payload in payloads] == ["promoted_lesson", "validated_lesson"]


def test_materialize_semantic_contract_regression_from_missing_contract_fields():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="sem_006",
            title="missing semantic contract fields",
            category="semantic",
            method=("knowledge_artifact",),
            features=("underlier_structure", "payoff_rule", "settlement_rule"),
            instrument=("structured_note",),
            symptom="missing contract fields: underlier_structure, payoff_rule, settlement_rule",
            root_cause="semantic validation could not proceed because required contract fields were absent",
            fix="Define the smallest new semantic concept before adding any wrapper or route helper.",
        )
    )

    assert payload is not None
    assert payload.template_family == "semantic_contract_regression"
    assert payload.target_test_file == "tests/test_agent/test_semantic_contracts.py"
    assert "semantic_contract_fields_present" in payload.assertion_focus


def test_materialize_bridge_fallback_regression_from_thin_wrapper_guidance():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="sem_018",
            title="keep canonical concept",
            category="semantic",
            method=("knowledge_artifact",),
            features=("underlier_structure", "payoff_rule", "settlement_rule"),
            instrument=("american_put",),
            symptom="missing contract fields: underlier_structure, payoff_rule, settlement_rule",
            root_cause="compatibility naming drift hid the canonical concept boundary",
            fix="Keep the canonical concept and surface the compatibility name as a thin wrapper.",
        )
    )

    assert payload is not None
    assert payload.template_family == "bridge_fallback_regression"
    assert payload.target_test_file == "tests/test_agent/test_checkpoints.py"
    assert "compatibility_bridge_preserved" in payload.assertion_focus


def test_materialize_lowering_admissibility_regression_from_exact_backend_gap():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="con_037",
            title="Missing CDS convention contract",
            category="convention",
            method=("credit",),
            features=("credit_risk", "discounting"),
            instrument=(),
            symptom="Lane `credit` has no exact backend binding and no constructive steps.",
            root_cause="The compiler cannot safely assemble par-spread logic without the lane contract.",
            fix="Bind the credit lane to a constructive route or exact backend contract before generation.",
        )
    )

    assert payload is not None
    assert payload.template_family == "lowering_admissibility_regression"
    assert payload.target_test_file == "tests/test_agent/test_platform_requests.py"
    assert "lowering_gate_blocks_unsupported_lane" in payload.assertion_focus


def test_materialize_route_boundary_regression_from_wrapper_signature_drift():
    from trellis.agent.knowledge.lesson_to_test import materialize_lesson_regression

    payload = materialize_lesson_regression(
        _lesson(
            lesson_id="con_046",
            title="Wrong product shape contract",
            category="convention",
            method=("copula",),
            features=("credit_risk", "path_dependent"),
            instrument=(),
            symptom="price_nth_to_default_basket() got an unexpected keyword argument 'market_state'",
            root_cause="The route wrapper signature did not match the pricing kernel boundary.",
            fix="Define an explicit tranche payoff adapter with the exact pricing kernel signature and remove unsupported market_state plumbing.",
        )
    )

    assert payload is not None
    assert payload.template_family == "route_boundary_regression"
    assert payload.target_test_file == "tests/test_agent/test_semantic_validators.py"
    assert "route_helper_signature_preserved" in payload.assertion_focus
