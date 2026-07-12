from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def test_comparison_target_prefers_per_build_metadata():
    from trellis.agent.executor import _comparison_target_from_build_metadata

    compiled_request = SimpleNamespace(
        request=SimpleNamespace(metadata={"comparison_target": "compiled_target"})
    )

    assert _comparison_target_from_build_metadata(
        compiled_request,
        {"comparison_target": "task_target"},
    ) == "task_target"


def test_build_payoff_reuse_branch_attaches_analytical_trace(monkeypatch, tmp_path):
    from trellis.agent.executor import build_payoff

    build_meta: dict[str, object] = {}
    pricing_plan = SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=set(),
        model_to_build=None,
        reasoning="reuse an existing generated route",
        selection_reason="cached_generated_module",
        assumption_summary=("cached route",),
        sensitivity_support=None,
    )
    product_ir = SimpleNamespace(instrument="european_option")
    compiled_request = SimpleNamespace(
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        request=SimpleNamespace(request_id="executor_build_cached_123"),
        linear_issue_identifier="QUA-372",
        generation_plan=None,
        knowledge_summary={},
    )
    plan = SimpleNamespace(
        steps=[SimpleNamespace(module_path="trellis/instruments/_agent/cached.py")],
        spec_schema=SimpleNamespace(
            spec_name="CachedSpec",
            class_name="CachedPayoff",
            fields=(),
        ),
    )
    generation_plan = SimpleNamespace(
        method="analytical",
        instrument_type="european_option",
        primitive_plan=SimpleNamespace(
            engine_family="analytical",
            blockers=(),
            route="cached",
        ),
    )
    existing = type("ExistingPayoff", (), {})
    trace_id = "executor_build_cached_123"
    emitted_kwargs: dict[str, object] = {}

    def fake_emit_analytical_trace_from_generation_plan(plan, **kwargs):
        emitted_kwargs.update(kwargs)
        json_path = tmp_path / f"{trace_id}.json"
        text_path = tmp_path / f"{trace_id}.md"
        json_path.write_text("{}")
        text_path.write_text("# analytical trace\n")
        return SimpleNamespace(
            trace=SimpleNamespace(trace_id=trace_id),
            json_path=json_path,
            text_path=text_path,
        )

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.planner.plan_build", lambda *args, **kwargs: plan)
    monkeypatch.setattr("trellis.agent.executor._try_import_existing", lambda plan: existing)
    monkeypatch.setattr("trellis.agent.executor.build_generation_plan", lambda **kwargs: generation_plan)
    monkeypatch.setattr(
        "trellis.agent.executor.emit_analytical_trace_from_generation_plan",
        fake_emit_analytical_trace_from_generation_plan,
    )
    monkeypatch.setattr(
        "trellis.agent.executor.render_generation_route_card",
        lambda plan: "route-card",
    )

    result = build_payoff(
        "Cached analytical route",
        compiled_request=compiled_request,
        build_meta=build_meta,
        market_state=SimpleNamespace(
            selected_curve_names={"discount_curve": "usd_ois"},
            available_capabilities=set(),
        ),
        model="gpt-5-mini",
    )

    assert result is existing
    assert build_meta["analytical_trace_id"] == trace_id
    assert build_meta["analytical_trace_path"] == str(tmp_path / f"{trace_id}.json")
    assert build_meta["analytical_trace_text_path"] == str(tmp_path / f"{trace_id}.md")
    assert emitted_kwargs["context"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
    }


def test_build_payoff_blocks_on_semantic_clarification(monkeypatch):
    from trellis.agent.executor import build_payoff

    compiled_request = SimpleNamespace(
        product_ir=None,
        pricing_plan=None,
        request=SimpleNamespace(
            request_id="executor_build_clarify_123",
            request_type="build",
            metadata={
                "semantic_gap": {
                    "requires_clarification": True,
                    "summary": "missing product shape",
                },
                "semantic_extension": {
                    "decision": "clarification",
                    "summary": "ask for the missing product shape before codegen",
                },
            },
        ),
        linear_issue_identifier="QUA-999",
        generation_plan=None,
        knowledge_summary={},
    )

    monkeypatch.setattr(
        "trellis.agent.planner.plan_build",
        lambda *args, **kwargs: pytest.fail("plan_build should not run after semantic clarification"),
    )
    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="requires clarification"):
        build_payoff(
            "Build a pricer for: Finite element method (FEM) vs finite difference for European",
            compiled_request=compiled_request,
            market_state=SimpleNamespace(
                selected_curve_names={},
                available_capabilities=set(),
            ),
            model="gpt-5-mini",
        )


def test_build_payoff_persists_structured_blockers_when_pre_generation_gate_blocks(monkeypatch):
    from trellis.agent.executor import build_payoff
    from trellis.agent.knowledge.schema import BuildGateDecision

    @dataclass
    class _Blocker:
        id: str
        category: str
        primitive_kind: str
        severity: str
        summary: str

    @dataclass
    class _WorkflowItem:
        title: str

    @dataclass
    class _Workflow:
        summary: str
        items: tuple[_WorkflowItem, ...]

    @dataclass
    class _BlockerReport:
        should_block: bool
        summary: str
        blockers: tuple[_Blocker, ...]

    build_meta: dict[str, object] = {}
    pricing_plan = SimpleNamespace(
        method="monte_carlo",
        method_modules=(),
        required_market_data=set(),
        model_to_build=None,
        reasoning="blocked",
        selection_reason="blocked",
        assumption_summary=(),
        sensitivity_support=None,
    )
    blocker = _Blocker(
        id="path_dependent_early_exercise_under_stochastic_vol",
        category="unsupported_composite",
        primitive_kind="exercise_control",
        severity="high",
        summary="Missing exercise/control primitive for path-dependent early exercise under stochastic volatility.",
    )
    generation_plan = SimpleNamespace(
        method="monte_carlo",
        instrument_type="barrier_option",
        primitive_plan=SimpleNamespace(
            route="exercise_monte_carlo",
            route_family="monte_carlo",
            engine_family="monte_carlo",
            blockers=("path_dependent_early_exercise_under_stochastic_vol",),
        ),
        blocker_report=_BlockerReport(
            should_block=True,
            summary="Missing exercise/control primitive for path-dependent early exercise under stochastic volatility.",
            blockers=(blocker,),
        ),
        new_primitive_workflow=_Workflow(
            summary="Implement the missing stochastic-vol exercise primitive.",
            items=(_WorkflowItem(title="exercise primitive"),),
        ),
    )
    compiled_request = SimpleNamespace(
        product_ir=SimpleNamespace(instrument="barrier_option"),
        pricing_plan=pricing_plan,
        request=SimpleNamespace(request_id="executor_build_blocked_123", request_type="build"),
        linear_issue_identifier="QUA-820",
        generation_plan=generation_plan,
        knowledge_summary={},
        semantic_blueprint=None,
    )
    plan = SimpleNamespace(
        steps=[SimpleNamespace(module_path="trellis/instruments/_agent/demo.py")],
        spec_schema=SimpleNamespace(spec_name="DemoSpec", class_name="DemoPayoff", fields=()),
    )

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.planner.plan_build", lambda *args, **kwargs: plan)
    monkeypatch.setattr("trellis.agent.executor.build_generation_plan", lambda **kwargs: generation_plan)
    monkeypatch.setattr("trellis.agent.executor._emit_analytical_trace_metadata", lambda **kwargs: None)
    monkeypatch.setattr(
        "trellis.agent.build_gate.evaluate_pre_generation_gate",
        lambda *args, **kwargs: BuildGateDecision(
            decision="block",
            reason="Hard blockers detected.",
            gap_confidence=1.0,
            gate_source="pre_generation",
        ),
    )

    with pytest.raises(RuntimeError, match="Build gate blocked pre-generation"):
        build_payoff(
            "Blocked build",
            compiled_request=compiled_request,
            build_meta=build_meta,
            market_state=SimpleNamespace(
                selected_curve_names={},
                available_capabilities=set(),
            ),
            model="gpt-5-mini",
        )

    blocker_details = build_meta["blocker_details"]
    assert blocker_details["blocker_codes"] == [
        "path_dependent_early_exercise_under_stochastic_vol"
    ]
    assert blocker_details["blocker_report"]["blockers"][0]["category"] == "unsupported_composite"
    assert blocker_details["new_primitive_workflow"]["items"][0]["title"] == "exercise primitive"


def test_explicit_semantic_method_gap_does_not_fall_back_to_broad_route(monkeypatch):
    from importlib import import_module

    from trellis.agent.executor import build_payoff
    from trellis.agent.semantic_contracts import (
        UnsupportedSemanticMethodError,
        make_rate_style_swaption_contract,
    )

    contract = make_rate_style_swaption_contract(
        description="Bermudan payer swaption with an irregular exercise schedule",
        observation_schedule=("2025-11-15", "2026-05-15", "2026-11-15"),
        preferred_method="rate_tree",
        exercise_style="bermudan",
    )
    build_meta: dict[str, object] = {}

    def _forbid_fallback(*args, **kwargs):
        raise AssertionError("explicit semantic compilation must not use broad fallback")

    decompose_module = import_module("trellis.agent.knowledge.decompose")
    monkeypatch.setattr(decompose_module, "decompose_to_ir", _forbid_fallback)
    monkeypatch.setattr("trellis.agent.planner.plan_build", _forbid_fallback)

    with pytest.raises(
        UnsupportedSemanticMethodError,
        match="Bermudan swaption Monte Carlo",
    ):
        build_payoff(
            "Bermudan payer swaption with an irregular exercise schedule",
            instrument_type="swaption",
            preferred_method="monte_carlo",
            semantic_contract=contract,
            build_meta=build_meta,
        )

    blocker_details = build_meta["blocker_details"]
    assert blocker_details["reason"] == "semantic_method_composition_gap"
    assert blocker_details["semantic_family"] == "rate_style_swaption:bermudan"
    assert blocker_details["requested_method"] == "monte_carlo"
    assert blocker_details["available_capabilities"] == [
        "hull_white_factor_simulation",
        "irregular_exercise_schedule",
        "longstaff_schwartz_continuation",
        "pathwise_swap_value_projection",
    ]
    assert blocker_details["missing_capabilities"] == [
        "exercise_schedule_to_simulation_grid",
        "pathwise_numeraire_discounting",
        "swap_value_paths_to_early_exercise_control",
    ]


def test_record_lesson_maps_fields_into_canonical_payload(monkeypatch):
    from trellis.agent.test_resolution import Lesson, record_lesson

    validated_payload: dict[str, object] = {}
    captured_kwargs: dict[str, object] = {}

    def fake_validate_lesson_payload(payload: dict[str, object]):
        validated_payload.update(payload)
        return SimpleNamespace(valid=True, errors=(), normalized_payload=payload)

    def fake_capture_lesson(**kwargs):
        captured_kwargs.update(kwargs)
        return "mc_999"

    monkeypatch.setattr(
        "trellis.agent.knowledge.promotion.validate_lesson_payload",
        fake_validate_lesson_payload,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.promotion.capture_lesson",
        fake_capture_lesson,
    )

    lesson_id = record_lesson(
        Lesson(
            category="monte_carlo",
            title="Capture canonical lesson fields",
            mistake="The helper used the wrong lesson field name.",
            why="Canonical lessons store root_cause, not why.",
            detect="The capture payload shape no longer matches the contract.",
            fix="Map the distilled lesson into the canonical lesson schema.",
        ),
        method="monte_carlo",
        features=["early_exercise"],
        validation="Resolved during build of American put option",
        confidence=0.5,
    )

    assert lesson_id == "mc_999"
    assert validated_payload["category"] == "monte_carlo"
    assert validated_payload["title"] == "Capture canonical lesson fields"
    assert validated_payload["symptom"] == "The helper used the wrong lesson field name."
    assert validated_payload["root_cause"] == "Canonical lessons store root_cause, not why."
    assert validated_payload["fix"] == "Map the distilled lesson into the canonical lesson schema."
    assert validated_payload["validation"] == "Resolved during build of American put option"
    assert validated_payload["confidence"] == 0.5
    assert validated_payload["applies_when"]["method"] == ["monte_carlo"]
    assert validated_payload["applies_when"]["features"] == ["early_exercise"]
    assert captured_kwargs["root_cause"] == "Canonical lessons store root_cause, not why."
    assert captured_kwargs["symptom"] == "The helper used the wrong lesson field name."


def test_record_resolved_failures_fails_hard_on_missing_lesson_fields(monkeypatch):
    from trellis.agent.executor import _record_resolved_failures

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "openai")
    monkeypatch.setattr(
        "trellis.agent.config.llm_generate_json",
        lambda prompt, model=None: {
            "category": "monte_carlo",
            "title": "Missing why",
            "mistake": "The helper omitted the reason field.",
            "detect": "The payload shape is incomplete.",
            "fix": "Return every required lesson field.",
        },
    )
    monkeypatch.setattr(
        "trellis.agent.test_resolution.record_lesson",
        lambda *args, **kwargs: pytest.fail("record_lesson should not be reached when fields are missing"),
    )

    with pytest.raises(RuntimeError, match="LLM lesson output missing fields"):
        _record_resolved_failures(
            ["example validation failure"],
            "European call option",
            SimpleNamespace(method="analytical"),
            "gpt-5-mini",
        )


def test_record_resolved_failures_skips_without_llm_credentials(monkeypatch, caplog):
    from trellis.agent.executor import _record_resolved_failures

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "openai")
    monkeypatch.setattr(
        "trellis.agent.config.llm_generate_json",
        lambda *args, **kwargs: pytest.fail("llm_generate_json should not run without credentials"),
    )
    monkeypatch.setattr(
        "trellis.agent.test_resolution.record_lesson",
        lambda *args, **kwargs: pytest.fail("record_lesson should not run without credentials"),
    )

    with caplog.at_level("WARNING"):
        _record_resolved_failures(
            ["example validation failure"],
            "European call option",
            SimpleNamespace(method="analytical"),
            "gpt-5-mini",
        )

    assert "Skipping resolved-failure lesson distillation" in caplog.text


def test_diagnose_failure_reads_related_lessons_from_canonical_signatures(monkeypatch):
    from trellis.agent.knowledge.schema import FailureSignature
    from trellis.agent.test_resolution import TestFailure, diagnose_failure

    store = SimpleNamespace(
        _failure_signatures=[
            FailureSignature(
                pattern="longstaff",
                magnitude="significant",
                category="monte_carlo",
                probable_causes=("mc_001",),
                features=("early_exercise",),
                diagnostic_hint="LSM continuation regression is unstable here.",
            )
        ],
        _load_lesson=lambda lesson_id: SimpleNamespace(
            title="LSM high-vol bias with polynomial basis"
        ) if lesson_id == "mc_001" else None,
    )
    monkeypatch.setattr("trellis.agent.knowledge.get_store", lambda: store)

    diagnosis = diagnose_failure(
        TestFailure(
            test_name="test_longstaff_example",
            test_file="tests/test_mc.py",
            error_type="AssertionError",
            error_message="longstaff price above benchmark",
            expected=None,
            actual=None,
            traceback="",
        )
    )

    assert diagnosis.category == "monte_carlo"
    assert diagnosis.related_lessons == ["LSM high-vol bias with polynomial basis"]


def test_generate_skeleton_normalizes_string_defaults_and_keeps_symbolic_defaults():
    from trellis.agent.executor import _generate_skeleton, _render_spec_default_value
    from trellis.agent.planner import FieldDef, SpecSchema

    assert _render_spec_default_value("str", "monte_carlo") == "'monte_carlo'"
    assert _render_spec_default_value("str", "'american'") == "'american'"
    assert _render_spec_default_value("str", '"put"') == "'put'"
    assert _render_spec_default_value("str | None", "None") == "None"

    spec_schema = SpecSchema(
        class_name="DemoPayoff",
        spec_name="DemoSpec",
        requirements=[],
        fields=[
            FieldDef("pricing_method", "str", "Method identifier", "monte_carlo"),
            FieldDef("rate_index", "str | None", "Optional rate index", "None"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.SEMI_ANNUAL"),
        ],
    )

    skeleton = _generate_skeleton(spec_schema, "Demo instrument")

    assert "pricing_method: str = 'monte_carlo'" in skeleton
    assert "rate_index: str | None = None" in skeleton
    assert "frequency: Frequency = Frequency.SEMI_ANNUAL" in skeleton
    assert "from trellis.core.types import Frequency" in skeleton
    assert "from trellis.core.date_utils import generate_schedule, year_fraction" not in skeleton
    assert "from trellis.models.black import black76_call, black76_put" not in skeleton


def test_hydrate_spec_schema_defaults_from_swaption_semantics():
    from trellis.agent.executor import (
        _generate_skeleton,
        _hydrate_spec_schema_defaults_from_semantics,
    )
    from trellis.agent.planner import STATIC_SPECS
    from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

    contract = make_rate_style_swaption_contract(
        description="European payer swaption",
        observation_schedule=("2025-11-15",),
        preferred_method="analytical",
        exercise_style="european",
        term_fields={
            "fixed_leg_day_count": "THIRTY_360",
            "rate_index": "USD-SOFR-3M",
            "payment_frequency": "SEMI_ANNUAL",
        },
    )

    hydrated = _hydrate_spec_schema_defaults_from_semantics(
        STATIC_SPECS["swaption"],
        semantic_contract=contract,
    )

    defaults = {
        field.name: field.default
        for field in hydrated.fields
    }
    assert defaults["day_count"] == "DayCountConvention.THIRTY_360"
    assert defaults["rate_index"] == "USD-SOFR-3M"
    assert defaults["swap_frequency"] == "Frequency.SEMI_ANNUAL"

    skeleton = _generate_skeleton(hydrated, "European payer swaption")
    assert "day_count: DayCountConvention = DayCountConvention.THIRTY_360" in skeleton
    assert "rate_index: str | None = 'USD-SOFR-3M'" in skeleton


def test_deterministic_exact_binding_module_materializes_swaption_helper_wrapper():
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
        EVALUATE_SENTINEL,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.rate_style_swaption.price_swaption_black76",),
        primitive_plan=None,
        method="analytical",
        instrument_type="swaption",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["swaption"],
        "European payer swaption",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "return price_swaption_black76(market_state, spec)" in generated.code
    assert "sigma=0.01" not in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_bermudan_tree_compat_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree",
        ),
        primitive_plan=SimpleNamespace(route="exercise_lattice"),
        method="rate_tree",
        instrument_type="swaption",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["bermudan_swaption"],
        "Bermudan payer swaption",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "return price_bermudan_swaption_tree(market_state, spec)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_cap_strip_helper_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",),
        primitive_plan=None,
        method="analytical",
        instrument_type="cap",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cap"],
        "USD SOFR cap strip analytical exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "price_rate_cap_floor_strip_analytical(" in generated.code
    assert 'instrument_class="cap"' in generated.code
    assert 'model=getattr(spec, "model", None)' in generated.code
    assert 'shift=getattr(spec, "shift", None)' in generated.code
    assert 'sabr=getattr(spec, "sabr", None)' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "binding_ref", "helper_name"),
    [
        (
            "analytical",
            "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",
            "price_rate_cap_floor_strip_analytical",
        ),
        (
            "monte_carlo",
            "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo",
            "price_rate_cap_floor_strip_monte_carlo",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_cap_strip_comparison_wrappers(
    comparison_target,
    binding_ref,
    helper_name,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(binding_ref,),
        primitive_plan=None,
        method=comparison_target,
        instrument_type="cap",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cap"],
        "Callable cap/floor collar comparison exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert f"{helper_name}(" in generated.code
    assert 'coupon_dates=getattr(spec, "coupon_dates", None)' in generated.code
    assert 'cap_strike=getattr(spec, "cap_strike", None)' in generated.code
    assert 'floor_strike=getattr(spec, "floor_strike", None)' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_exact_binding_refs_collect_backend_helper_refs_from_route_authority():
    from trellis.agent.executor import _exact_binding_refs

    refs = _exact_binding_refs(
        SimpleNamespace(
            lane_exact_binding_refs=(),
            backend_helper_refs=(),
            primitive_plan=None,
            route_binding_authority={
                "backend_binding": {
                    "binding_id": "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",
                    "helper_refs": [
                        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo"
                    ],
                }
            },
        )
    )

    assert "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical" in refs
    assert "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo" in refs


def test_exact_binding_refs_collect_backend_helper_refs_from_mapping_plan():
    from trellis.agent.executor import _exact_binding_refs

    refs = _exact_binding_refs(
        {
            "instrument_type": "period_rate_option_strip",
            "route_binding_authority": {
                "backend_binding": {
                    "binding_id": "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical",
                    "helper_refs": [
                        "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo"
                    ],
                    "target_bindings": [
                        {
                            "module": "trellis.models.black",
                            "symbol": "black76_call",
                            "role": "pricing_kernel",
                        }
                    ],
                }
            },
        }
    )

    assert "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical" in refs
    assert "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo" in refs
    assert "trellis.models.black.black76_call" in refs


def test_deterministic_exact_binding_module_materializes_period_rate_strip_from_backend_refs():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        backend_helper_refs=(),
        primitive_plan=None,
        method="analytical",
        instrument_type="period_rate_option_strip",
        route_binding_authority={
            "backend_binding": {
                "helper_refs": [
                    "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical"
                ],
            }
        },
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cap"],
        "Callable cap/floor collar comparison exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="analytical",
    )

    assert generated is not None
    assert "price_rate_cap_floor_strip_analytical(" in generated.code
    assert 'instrument_class=getattr(spec, "instrument_class", None) or "cap"' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_period_rate_strip_from_mapping_refs():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = {
        "method": "monte_carlo",
        "instrument_type": "period_rate_option_strip",
        "route_binding_authority": {
            "backend_binding": {
                "helper_refs": [
                    "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo"
                ],
            }
        },
    }

    skeleton = _generate_skeleton(
        STATIC_SPECS["cap"],
        "Callable cap/floor collar comparison exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="monte_carlo",
    )

    assert generated is not None
    assert "price_rate_cap_floor_strip_monte_carlo(" in generated.code
    assert 'instrument_class=getattr(spec, "instrument_class", None) or "cap"' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_make_test_payoff_reconciles_stale_schema_against_payoff_dataclass(monkeypatch):
    from trellis.agent.executor import _make_test_payoff

    module_name = "test_stale_schema_payoff_module"
    module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    exec(
        """
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class CallableCapFloorCollarSpec:
    notional: float
    comparison_targets: tuple[date, ...] | None

class CallableCapFloorCollar:
    def __init__(self, spec: CallableCapFloorCollarSpec):
        self.spec = spec
""",
        module.__dict__,
    )

    stale_schema = SimpleNamespace(
        spec_name="CallableCapFloorCollarSpec",
        fields=[
            SimpleNamespace(name="notional", type="float", default=None),
            SimpleNamespace(name="discount_curve_name", type="str | None", default=None),
        ],
    )

    payoff = _make_test_payoff(
        module.CallableCapFloorCollar,
        stale_schema,
        date(2024, 11, 15),
        spec_overrides={"discount_curve_name": "usd_ois"},
    )

    assert payoff.spec.notional == pytest.approx(100.0)
    assert payoff.spec.comparison_targets is None
    assert not hasattr(payoff.spec, "discount_curve_name")


@pytest.mark.parametrize(
    ("description", "expected_defaults"),
    [
        (
            "Price a cap strip. Pricing model: shifted_black. Shift: 0.01.",
            {"model": "shifted_black", "shift": pytest.approx(0.01)},
        ),
        (
            "Price a cap strip. Pricing model: sabr. "
            "SABR parameters: alpha=0.025, beta=0.5, nu=0.35, rho=-0.2.",
            {
                "model": "sabr",
                "sabr": {"alpha": 0.025, "beta": 0.5, "nu": 0.35, "rho": -0.2},
            },
        ),
    ],
)
def test_description_spec_defaults_extract_cap_model_fields(
    description,
    expected_defaults,
):
    from trellis.agent.executor import _description_spec_defaults
    from trellis.agent.planner import STATIC_SPECS

    defaults = _description_spec_defaults(
        STATIC_SPECS["cap"],
        description=description,
    )

    for key, expected in expected_defaults.items():
        assert defaults[key] == expected


def test_deterministic_exact_binding_module_materializes_callable_bond_tree_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.callable_bond_tree.price_callable_bond_tree",),
        primitive_plan=None,
        method="rate_tree",
        instrument_type="callable_bond",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["callable_bond"],
        "Callable bond tree",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert 'return price_callable_bond_tree(market_state, spec, model="hull_white")' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_callable_bond_pde_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.callable_bond_pde.price_callable_bond_pde",),
        primitive_plan=None,
        method="pde_solver",
        instrument_type="callable_bond",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["callable_bond"],
        "Callable bond PDE",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "return price_callable_bond_pde(market_state, spec)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("euler", 'scheme="euler"'),
        ("milstein", 'scheme="milstein"'),
        ("exact", 'scheme="exact"'),
        ("log_euler", 'scheme="log_euler"'),
        ("plain_mc", 'variance_reduction="none"'),
        ("antithetic_mc", 'variance_reduction="antithetic"'),
        ("control_variate_mc", 'variance_reduction="control_variate"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_vanilla_equity_mc_from_primitives(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.monte_carlo.single_state_diffusion."
            "price_single_state_terminal_claim_monte_carlo_result",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_monte_carlo"],
        "European option Monte Carlo",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_single_state_terminal_claim_monte_carlo_result(" in generated.code
    assert "terminal_intrinsic_from_resolved(terminal, resolved)" in generated.code
    assert "price_vanilla_equity_option_monte_carlo" not in generated.code
    assert expected_fragment in generated.code
    if comparison_target == "control_variate_mc":
        assert "control_variate_values=" in generated.code
        assert "control_variate_expected=" in generated.code
        assert "from math import exp" in generated.code
    else:
        assert "control_variate_values=" not in generated.code
        assert "control_variate_expected=" not in generated.code
        assert "from math import exp" not in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize("comparison_target", ["exact", "control_variate_mc"])
def test_deterministic_vanilla_equity_mc_primitive_composition_executes(
    comparison_target,
):
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.monte_carlo.single_state_diffusion."
            "price_single_state_terminal_claim_monte_carlo_result",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )
    schema = SPECIALIZED_SPECS["european_option_monte_carlo"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "European option primitive-composed Monte Carlo",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1169>", "exec"), namespace)  # noqa: S102
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2025, 1, 1),
            option_type="call",
            n_paths=30_000,
            n_steps=64,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    assert float(payoff.evaluate(market)) == pytest.approx(10.45, abs=0.25)


def test_admitted_vanilla_equity_mc_adapter_uses_terminal_claim_primitives():
    source = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/europeanoptionmontecarlo.py"
    ).read_text()

    assert "price_single_state_terminal_claim_monte_carlo_result(" in source
    assert "terminal_intrinsic_from_resolved(" in source
    assert "price_vanilla_equity_option_monte_carlo" not in source


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("theta_0.5", "theta=0.5"),
        ("theta_1.0", "theta=1.0"),
    ],
)
def test_deterministic_exact_binding_module_materializes_vanilla_equity_pde_from_primitives(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.pde.event_aware.solve_event_aware_pde",
        ),
        primitive_plan=SimpleNamespace(route="vanilla_equity_theta_pde"),
        method="pde_solver",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European call: theta-method convergence order measurement",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "resolve_single_state_diffusion_inputs(" in generated.code
    assert "terminal_intrinsic_from_resolved(" in generated.code
    assert "EventAwarePDEGridSpec(" in generated.code
    assert "EventAwarePDEOperatorSpec(" in generated.code
    assert "EventAwarePDEBoundarySpec(" in generated.code
    assert "EventAwarePDEProblemSpec(" in generated.code
    assert "build_event_aware_pde_problem(" in generated.code
    assert "solve_event_aware_pde(" in generated.code
    assert "interpolate_pde_values(" in generated.code
    assert "price_vanilla_equity_option_pde" not in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize("comparison_target", ["theta_0.5", "theta_1.0"])
def test_deterministic_vanilla_equity_pde_primitive_composition_executes_with_dividends(
    comparison_target,
):
    from datetime import date as _date
    from math import exp as _exp

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState
    from trellis.models.black import black76_call

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return _exp(-0.05 * float(t))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.pde.event_aware.solve_event_aware_pde",
        ),
        primitive_plan=SimpleNamespace(route="vanilla_equity_theta_pde"),
        method="pde_solver",
        instrument_type="european_option",
    )
    schema = SPECIALIZED_SPECS["european_option_analytical"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "European dividend-paying call primitive-composed PDE",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1170>", "exec"), namespace)  # noqa: S102
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2024, 12, 31),
            option_type="call",
            dividend_yield=0.02,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )
    expected = _exp(-0.05) * black76_call(
        100.0 * _exp(0.03),
        100.0,
        0.20,
        1.0,
    )

    assert float(payoff.evaluate(market)) == pytest.approx(expected, abs=0.08)


def test_admitted_european_analytical_adapter_honors_dividend_yield():
    from datetime import date as _date
    from math import exp as _exp

    from trellis.core.market_state import MarketState
    from trellis.instruments._agent.europeanoptionanalytical import (
        EuropeanOptionAnalyticalPayoff,
        EuropeanOptionSpec,
    )
    from trellis.models.black import black76_call

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return _exp(-0.05 * float(t))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    payoff = EuropeanOptionAnalyticalPayoff(
        EuropeanOptionSpec(
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2024, 12, 31),
            option_type="call",
            dividend_yield=0.02,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )
    expected = _exp(-0.05) * black76_call(
        100.0 * _exp(0.03),
        100.0,
        0.20,
        1.0,
    )

    assert float(payoff.evaluate(market)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("schema_id", "method", "exact_ref", "required_fragments", "excluded_fragments"),
    [
        (
            "fx_barrier_option_analytical",
            "analytical",
            "trellis.models.analytical.barrier.barrier_option_price",
            (
                "resolve_fx_barrier_inputs(",
                "barrier_option_price(",
                "q=resolved.foreign_rate",
            ),
            ("price_fx_barrier_option_analytical(",),
        ),
        (
            "fx_barrier_option_monte_carlo",
            "monte_carlo",
            "trellis.models.monte_carlo.engine.MonteCarloEngine",
            (
                "resolve_fx_barrier_inputs(",
                "GBM(",
                "MonteCarloEngine(",
                "BarrierMonitor(",
                "MonteCarloPathRequirement(",
                "StateAwarePayoff(",
                "terminal_intrinsic(",
                "return_paths=False",
            ),
            (
                "price_fx_barrier_option_monte_carlo(",
                "price_fx_barrier_option_monte_carlo_result(",
            ),
        ),
    ],
)
def test_deterministic_fx_barrier_targets_use_primitive_composition(
    schema_id,
    method,
    exact_ref,
    required_fragments,
    excluded_fragments,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    route_primitives = ()
    if method == "monte_carlo":
        route_primitives = (
            SimpleNamespace(
                module="trellis.models.monte_carlo.path_state",
                symbol="BarrierMonitor",
                role="event_monitor",
                required=True,
            ),
            SimpleNamespace(
                module="trellis.models.monte_carlo.path_state",
                symbol="MonteCarloPathRequirement",
                role="path_requirement",
                required=True,
            ),
            SimpleNamespace(
                module="trellis.models.monte_carlo.path_state",
                symbol="StateAwarePayoff",
                role="payoff_primitive",
                required=True,
            ),
        )
    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(exact_ref,),
        primitive_plan=SimpleNamespace(
            route=f"{method}_fx_barrier" if method == "analytical" else "monte_carlo_fx_barrier",
            primitives=route_primitives,
        ),
        method=method,
        instrument_type="barrier_option",
    )
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            SPECIALIZED_SPECS[schema_id],
            "FX barrier primitive composition",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target=method,
    )

    assert generated is not None
    compile(generated.code, "<qua_1171_source>", "exec")
    for fragment in required_fragments:
        assert fragment in generated.code
    for fragment in excluded_fragments:
        assert fragment not in generated.code
    if method == "monte_carlo":
        assert generated.code.count(
            "from trellis.models.monte_carlo.path_state import"
        ) == 1
    assert EVALUATE_SENTINEL not in generated.code


def test_generated_fx_barrier_analytical_and_mc_agree():
    from datetime import date as _date
    from math import exp

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.instruments.fx import FXRate
    from trellis.models.vol_surface import FlatVol

    def materialize(schema_id, method, exact_ref, route):
        plan = SimpleNamespace(
            lane_exact_binding_refs=(exact_ref,),
            primitive_plan=SimpleNamespace(route=route),
            method=method,
            instrument_type="barrier_option",
        )
        schema = SPECIALIZED_SPECS[schema_id]
        generated = _materialize_deterministic_exact_binding_module(
            _generate_skeleton(
                schema,
                "FX barrier generated parity",
                generation_plan=plan,
            ),
            plan,
            comparison_target=method,
        )
        assert generated is not None
        namespace: dict = {}
        exec(compile(generated.code, f"<qua_1171_{method}>", "exec"), namespace)  # noqa: S102
        return schema, namespace

    analytical_schema, analytical_ns = materialize(
        "fx_barrier_option_analytical",
        "analytical",
        "trellis.models.analytical.barrier.barrier_option_price",
        "analytical_fx_barrier",
    )
    mc_schema, mc_ns = materialize(
        "fx_barrier_option_monte_carlo",
        "monte_carlo",
        "trellis.models.monte_carlo.engine.MonteCarloEngine",
        "monte_carlo_fx_barrier",
    )
    terms = dict(
        notional=1_000_000.0,
        strike=1.10,
        barrier=1.02,
        expiry_date=_date(2025, 11, 15),
        fx_pair="EURUSD",
        foreign_discount_key="EUR-DISC",
        option_type="call",
        barrier_type="down_and_in",
    )
    analytical = analytical_ns[analytical_schema.class_name](
        analytical_ns[analytical_schema.spec_name](**terms)
    )
    mc = mc_ns[mc_schema.class_name](
        mc_ns[mc_schema.spec_name](**terms, n_paths=20_000, n_steps=252)
    )
    market = MarketState(
        as_of=_date(2024, 11, 15),
        settlement=_date(2024, 11, 15),
        discount=YieldCurve.flat(0.045),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.025)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        vol_surface=FlatVol(0.14),
    )

    analytical_price = float(analytical.evaluate(market))
    mc_price = float(mc.evaluate(market))

    assert analytical_price > 0.0
    assert mc_price == pytest.approx(analytical_price, rel=0.08, abs=2_000.0)

    immediate_knock_in = mc_ns[mc_schema.class_name](
        mc_ns[mc_schema.spec_name](
            **{
                **terms,
                "strike": 1.0,
                "barrier": 1.11,
            },
            observations_per_year=1,
            n_paths=1_000,
            n_steps=4,
        )
    )
    deterministic_market = MarketState(
        as_of=_date(2024, 11, 15),
        settlement=_date(2024, 11, 15),
        discount=YieldCurve.flat(0.02),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.0)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        vol_surface=FlatVol(0.0),
    )

    expected_vanilla = 1_000_000.0 * (
        1.10 - exp(-0.02) * 1.0
    )
    assert immediate_knock_in.evaluate(deterministic_market) == pytest.approx(
        expected_vanilla,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment", "expected_import"),
    [
        (
            "psor_pde",
            "price_event_aware_equity_option_pde(",
            "from trellis.models.equity_option_pde import price_event_aware_equity_option_pde",
        ),
    ],
)
@pytest.mark.parametrize("instrument_type", ["american_put", "american_option"])
def test_deterministic_exact_binding_module_materializes_american_put_targets_without_refs(
    instrument_type,
    comparison_target,
    expected_fragment,
    expected_import,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="pde_solver",
        instrument_type=instrument_type,
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["american_put_tree"],
        "American put: PSOR vs tree vs LSM three-way",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_import in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize("comparison_target", ["crr_tree", "high_step_tree_2000"])
@pytest.mark.parametrize("instrument_type", ["american_put", "american_option"])
def test_deterministic_exact_binding_materializes_american_tree_from_primitives(
    instrument_type,
    comparison_target,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="rate_tree",
        instrument_type=instrument_type,
    )
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            SPECIALIZED_SPECS["american_put_tree"],
            "American put primitive-composed tree",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    for fragment in (
        "from trellis.models.resolution.single_state_diffusion import (",
        "resolve_single_state_diffusion_inputs",
        "terminal_intrinsic_from_resolved",
        "from trellis.models.trees.algebra import (",
        "equity_tree",
        "with_control",
        "compile_lattice_recipe",
        "build_lattice",
        "price_on_lattice",
        "dividend_yield=resolved.dividend_yield",
    ):
        assert fragment in generated.code
    assert "event_step_indices(" not in generated.code
    assert 'exercise_style == "bermudan"' not in generated.code
    assert "price_vanilla_equity_option_tree" not in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_american_tree_maps_bermudan_dates_to_exercise_steps():
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.date_utils import year_fraction
    from trellis.core.market_state import MarketState
    from trellis.models.monte_carlo.event_state import event_step_indices

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="rate_tree",
        instrument_type="american_option",
        lane_control_obligations=("exercise_style:bermudan",),
    )
    schema = SPECIALIZED_SPECS["american_put_tree"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "Bermudan put primitive-composed tree",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="crr_tree",
    )

    assert generated is not None
    assert "event_step_indices(" in generated.code
    assert 'exercise_style == "bermudan"' in generated.code
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1168_bermudan>", "exec"), namespace)  # noqa: S102
    settle = _date(2024, 1, 1)
    expiry = _date(2025, 1, 1)
    exercise_dates = (_date(2024, 4, 1), _date(2024, 10, 1), expiry)
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            spot=100.0,
            strike=100.0,
            expiry_date=expiry,
            exercise_style="bermudan",
            exercise_dates=exercise_dates,
            tree_steps=12,
        )
    )
    captured: dict[str, tuple[int, ...]] = {}
    original_with_control = namespace["with_control"]

    def _capture_control(recipe, control_kind, **control_params):
        captured["steps"] = tuple(control_params.get("exercise_steps", ()))
        return original_with_control(recipe, control_kind, **control_params)

    namespace["with_control"] = _capture_control
    market = MarketState(
        as_of=settle,
        settlement=settle,
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    assert float(payoff.evaluate(market)) > 0.0
    maturity = year_fraction(settle, expiry)
    event_times = tuple(year_fraction(settle, item) for item in exercise_dates)
    assert captured["steps"] == event_step_indices(event_times, maturity, 12)


def test_deterministic_american_tree_rejects_empty_bermudan_schedule_window():
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="rate_tree",
        instrument_type="american_option",
        lane_control_obligations=("exercise_style:bermudan",),
    )
    schema = SPECIALIZED_SPECS["american_put_tree"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "Bermudan put primitive-composed tree",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="crr_tree",
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1168_empty_bermudan>", "exec"), namespace)  # noqa: S102
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2025, 1, 1),
            exercise_style="bermudan",
            exercise_dates=(_date(2023, 12, 1), _date(2025, 2, 1)),
            tree_steps=12,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    with pytest.raises(ValueError, match="within the pricing horizon"):
        payoff.evaluate(market)


def test_deterministic_american_tree_primitive_composition_executes():
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="rate_tree",
        instrument_type="american_option",
    )
    schema = SPECIALIZED_SPECS["american_put_tree"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "American put primitive-composed tree",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="crr_tree",
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1168>", "exec"), namespace)  # noqa: S102
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2025, 1, 1),
            tree_steps=800,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    assert float(payoff.evaluate(market)) == pytest.approx(6.09, abs=0.15)


@pytest.mark.parametrize("instrument_type", ["american_put", "american_option"])
def test_deterministic_exact_binding_materializes_american_lsm_from_primitives(
    instrument_type,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type=instrument_type,
    )
    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["american_put_tree"],
        "American put: PSOR vs tree vs LSM three-way",
        generation_plan=generation_plan,
    )

    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="lsm_mc",
    )

    assert generated is not None
    for fragment in (
        "from trellis.models.processes.gbm import GBM",
        "from trellis.models.monte_carlo.engine import MonteCarloEngine",
        "from trellis.models.monte_carlo.lsm import longstaff_schwartz",
        "from trellis.models.monte_carlo.single_state_diffusion import resolve_single_state_monte_carlo_inputs",
        "from trellis.models.resolution.single_state_diffusion import terminal_intrinsic_from_resolved",
        "resolve_single_state_monte_carlo_inputs(",
        "MonteCarloEngine(",
        "longstaff_schwartz(",
        "event_step_indices(",
        'exercise_style == "bermudan"',
        "spec.exercise_dates",
    ):
        assert fragment in generated.code
    assert "price_american_equity_option_lsm_monte_carlo" not in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_american_lsm_maps_bermudan_dates_to_exercise_steps():
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.date_utils import year_fraction
    from trellis.core.market_state import MarketState
    from trellis.models.monte_carlo.event_state import event_step_indices

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="american_option",
    )
    schema = SPECIALIZED_SPECS["american_put_tree"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "Bermudan put primitive-composed LSM",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="lsm_mc",
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1167_bermudan>", "exec"), namespace)  # noqa: S102
    settle = _date(2024, 1, 1)
    expiry = _date(2025, 1, 1)
    exercise_dates = (_date(2024, 4, 1), _date(2024, 10, 1), expiry)
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            spot=100.0,
            strike=100.0,
            expiry_date=expiry,
            exercise_style="bermudan",
            exercise_dates=exercise_dates,
            n_paths=32,
            n_steps=12,
            seed=42,
        )
    )
    captured: dict[str, list[int]] = {}

    def _capture_lsm(_paths, steps, _payoff_fn, **_kwargs):
        captured["steps"] = list(steps)
        return 6.0

    namespace["longstaff_schwartz"] = _capture_lsm
    market = MarketState(
        as_of=settle,
        settlement=settle,
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    assert payoff.evaluate(market) == 6.0
    maturity = year_fraction(settle, expiry)
    event_times = tuple(year_fraction(settle, item) for item in exercise_dates)
    assert captured["steps"] == list(event_step_indices(event_times, maturity, 12))


def test_deterministic_american_lsm_primitive_composition_executes():
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState

    class _FlatDiscount:
        def zero_rate(self, _t: float) -> float:
            return 0.05

        def discount(self, t: float) -> float:
            return float(_np.exp(-0.05 * float(t)))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.20

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="american_option",
    )
    schema = SPECIALIZED_SPECS["american_put_tree"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "American put primitive-composed LSM",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="lsm_mc",
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1167>", "exec"), namespace)  # noqa: S102
    payoff_cls = namespace[schema.class_name]
    spec_cls = namespace[schema.spec_name]
    payoff = payoff_cls(
        spec_cls(
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2025, 1, 1),
            n_paths=8_000,
            n_steps=48,
            seed=42,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )

    assert float(payoff.evaluate(market)) == pytest.approx(6.08, abs=0.75)


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment", "expected_import"),
    [
        (
            "cev_pde",
            "price_cev_option_pde(",
            "from trellis.models.equity_option_pde import price_cev_option_pde",
        ),
        (
            "cev_tree",
            "price_cev_option_tree(",
            "from trellis.models.equity_option_tree import price_cev_option_tree",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_cev_targets_without_refs(
    comparison_target,
    expected_fragment,
    expected_import,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="pde_solver",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["cev_option"],
        "CEV model: CEVOperator PDE vs CEV tree",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_import in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("heston_mc", 'scheme="heston_qe"'),
        ("euler_heston", 'scheme="euler"'),
        ("qe_heston", 'scheme="heston_qe"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_heston_mc_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_monte_carlo"],
        "European Heston Monte Carlo",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_heston_option_monte_carlo(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_heston_adi_helper():
    from trellis.agent.codegen_guardrails import PrimitiveRef
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=SimpleNamespace(
            route="heston_adi_2d",
            primitives=(
                PrimitiveRef(
                    "trellis.models.pde.heston_adi",
                    "price_heston_option_adi_pde_result",
                    "route_helper",
                ),
            ),
        ),
        method="pde_solver",
        instrument_type="heston_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["heston_option"],
        "Heston ADI checked helper",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="heston_adi_pde",
    )

    assert generated is not None
    assert (
        "from trellis.models.pde.heston_adi import price_heston_option_adi_pde_result"
        in generated.code
    )
    assert "price_heston_option_adi_pde_result(market_state, spec).price" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_rannacher"),
    [
        ("cn_rannacher", 'rannacher_timesteps=getattr(spec, "rannacher_timesteps", 2)'),
        ("cn_standard", 'rannacher_timesteps=getattr(spec, "rannacher_timesteps", 0)'),
    ],
)
def test_deterministic_exact_binding_module_materializes_digital_pde_targets(
    comparison_target,
    expected_rannacher,
):
    from trellis.agent.codegen_guardrails import PrimitiveRef
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=SimpleNamespace(
            route="pde_theta_1d",
            primitives=(
                PrimitiveRef(
                    "trellis.models.equity_option_pde",
                    "price_equity_digital_option_pde",
                    "route_helper",
                ),
            ),
        ),
        method="pde_solver",
        instrument_type="digital_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["digital_option"],
        "Digital PDE proof target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "from trellis.models.equity_option_pde import price_equity_digital_option_pde" in generated.code
    assert "price_equity_digital_option_pde(" in generated.code
    assert 'theta=getattr(spec, "theta", 0.5)' in generated.code
    assert expected_rannacher in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "helper_ref", "expected_call"),
    [
        (
            "mc_asian",
            "trellis.models.asian_option.price_arithmetic_asian_option_monte_carlo",
            "price_arithmetic_asian_option_monte_carlo(market_state, spec)",
        ),
        (
            "turnbull_wakeman_approx",
            "trellis.models.asian_option.price_arithmetic_asian_option_analytical",
            "price_arithmetic_asian_option_analytical(market_state, spec)",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_asian_targets(
    comparison_target,
    helper_ref,
    expected_call,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="asian_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["asian_option"],
        "Asian option proof target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_call in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_lookback_mc_target():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    helper_ref = "trellis.models.lookback_option.price_equity_fixed_lookback_option_monte_carlo"
    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="lookback_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["lookback_option"],
        "Lookback option MC proof target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="mc_lookback",
    )

    assert generated is not None
    assert "from trellis.models.lookback_option import price_equity_fixed_lookback_option_monte_carlo" in generated.code
    assert "price_equity_fixed_lookback_option_monte_carlo(market_state, spec)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    "comparison_target",
    [
        "heston_adi_pde",
        "pde_double_barrier",
        "mc_double_barrier",
        "mc_autocall",
        "mc_autocall_qmc",
    ],
)
def test_deterministic_exact_binding_module_does_not_materialize_learning_targets(comparison_target):
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=SimpleNamespace(route="target_helper"),
        method="pde_solver" if comparison_target in {"heston_adi_pde", "pde_double_barrier"} else "monte_carlo",
        instrument_type="barrier_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["barrier_option"],
        "Failed pack helper target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is None


def test_deterministic_exact_binding_module_materializes_cliquet_monte_carlo_helper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="cliquet_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cliquet_option"],
        "Capped and floored cliquet comparison target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="monte_carlo",
    )

    assert generated is not None
    assert "from trellis.models.monte_carlo.event_aware import price_equity_cliquet_option_monte_carlo" in generated.code
    assert "price_equity_cliquet_option_monte_carlo(" in generated.code
    assert 'n_paths=getattr(spec, "n_paths", 120000)' in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_sampling"),
    [
        ("mc_autocall", "pseudo"),
        ("mc_autocall_qmc", "sobol"),
    ],
)
def test_deterministic_exact_binding_module_materializes_autocallable_helper(
    comparison_target,
    expected_sampling,
):
    from trellis.agent.codegen_guardrails import PrimitiveRef
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=SimpleNamespace(
            route="monte_carlo_paths",
            primitives=(
                PrimitiveRef(
                    "trellis.models.autocallable",
                    "price_autocallable_monte_carlo_result",
                    "route_helper",
                ),
            ),
        ),
        method="monte_carlo",
        instrument_type="autocallable",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["autocallable"],
        "Autocallable helper target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "from trellis.models.autocallable import price_autocallable_monte_carlo_result" in generated.code
    assert (
        f'price_autocallable_monte_carlo_result(market_state, spec, sampling="{expected_sampling}").price'
        in generated.code
    )
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("fft", 'method="fft"'),
        ("cos", 'method="cos"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_vanilla_equity_transform_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.equity_option_transforms.price_vanilla_equity_option_transform",),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option transforms",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_vanilla_equity_option_transform(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("heston_fft", 'method="fft"'),
        ("heston_cos", 'method="cos"'),
        ("laguerre_heston", 'method="gauss_laguerre"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_heston_transform_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.transforms.heston.price_heston_option_transform",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option Heston transforms",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_heston_option_transform(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("merton_fft", 'method="fft"'),
        ("merton_cos", 'method="cos"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_merton_transform_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "Merton jump-diffusion option transforms",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert (
        "from trellis.models.merton_jump_diffusion_option import "
        "price_merton_jump_diffusion_option_transform"
    ) in generated.code
    assert 'return {"black_vol_surface", "discount_curve", "jump_parameters"}' in generated.code
    assert "price_merton_jump_diffusion_option_transform(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_merton_monte_carlo_helper_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_monte_carlo",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_monte_carlo"],
        "Merton jump-diffusion option MC",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="merton_mc",
    )

    assert generated is not None
    assert (
        "from trellis.models.merton_jump_diffusion_option import "
        "price_merton_jump_diffusion_option_monte_carlo"
    ) in generated.code
    assert 'return {"black_vol_surface", "discount_curve", "jump_parameters"}' in generated.code
    assert "price_merton_jump_diffusion_option_monte_carlo(market_state, spec" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "helper", "expected_fragment"),
    [
        (
            "vg_cos",
            "price_variance_gamma_option_transform",
            'method="cos"',
        ),
        (
            "madan_carr_chang_reference",
            "price_variance_gamma_option_reference",
            "price_variance_gamma_option_reference(market_state, spec)",
        ),
        (
            "cgmy_cos",
            "price_cgmy_option_transform",
            'method="cos"',
        ),
        (
            "cgmy_reference_values",
            "price_cgmy_option_reference",
            "price_cgmy_option_reference(market_state, spec)",
        ),
        (
            "kou_fft",
            "price_kou_option_transform",
            'method="fft"',
        ),
        (
            "kou_reference_values",
            "price_kou_option_reference",
            "price_kou_option_reference(market_state, spec)",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_levy_transform_helper_wrapper(
    comparison_target,
    helper,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            f"trellis.models.levy_option.{helper}",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "Levy option transform",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert f"from trellis.models.levy_option import {helper}" in generated.code
    assert 'return {"discount_curve", "model_parameters"}' in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "helper"),
    [
        ("vg_mc", "price_variance_gamma_option_monte_carlo"),
        ("cgmy_mc", "price_cgmy_option_monte_carlo"),
        ("kou_mc", "price_kou_option_monte_carlo"),
    ],
)
def test_deterministic_exact_binding_module_materializes_levy_monte_carlo_helper_wrapper(
    comparison_target,
    helper,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            f"trellis.models.levy_option.{helper}",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_monte_carlo"],
        "Levy option MC",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert f"from trellis.models.levy_option import {helper}" in generated.code
    assert 'return {"discount_curve", "model_parameters"}' in generated.code
    assert f"{helper}(market_state, spec" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("bates_fft", 'price_bates_option_transform(market_state, spec, method="fft")'),
        ("bates_mc", "price_bates_option_monte_carlo("),
    ],
)
def test_deterministic_exact_binding_module_materializes_bates_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.bates_option.price_bates_option_transform",
            "trellis.models.bates_option.price_bates_option_monte_carlo",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "Bates option",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "from trellis.models.bates_option import" in generated.code
    assert 'return {"discount_curve", "jump_parameters", "model_parameters"}' in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_sabr_hagan_helper_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.sabr_option.price_sabr_forward_option_hagan",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "SABR Hagan forward option",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="sabr_hagan_analytical",
    )

    assert generated is not None
    assert (
        "from trellis.models.sabr_option import "
        "price_sabr_forward_option_hagan"
    ) in generated.code
    assert 'return {"discount_curve", "model_parameters"}' in generated.code
    assert "price_sabr_forward_option_hagan(market_state, spec)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_sabr_monte_carlo_helper_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.sabr_option.price_sabr_forward_option_monte_carlo",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_monte_carlo"],
        "SABR MC forward option",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="sabr_mc",
    )

    assert generated is not None
    assert (
        "from trellis.models.sabr_option import "
        "price_sabr_forward_option_monte_carlo"
    ) in generated.code
    assert 'return {"discount_curve", "model_parameters"}' in generated.code
    assert "price_sabr_forward_option_monte_carlo(market_state, spec" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_validate_build_uses_heston_model_payload_for_heston_transform(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.planner import FieldDef, SpecSchema
    from trellis.core.types import DayCountConvention
    from trellis.models.transforms.heston import price_heston_option_transform

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)

    @dataclass(frozen=True)
    class HestonExactSpecForValidation:
        notional: float
        spot: float
        strike: float
        expiry_date: date
        option_type: str = "call"
        day_count: DayCountConvention = DayCountConvention.ACT_365

    monkeypatch.setattr(
        sys.modules[__name__],
        "HestonExactSpecForValidation",
        HestonExactSpecForValidation,
        raising=False,
    )

    class HestonExactPayoffForValidation:
        def __init__(self, spec: HestonExactSpecForValidation):
            self._spec = spec

        @property
        def spec(self) -> HestonExactSpecForValidation:
            return self._spec

        @property
        def requirements(self) -> set[str]:
            return {"discount_curve", "model_parameters"}

        def evaluate(self, market_state):
            return price_heston_option_transform(market_state, self._spec, method="fft")

    spec_schema = SpecSchema(
        class_name="HestonExactPayoffForValidation",
        spec_name="HestonExactSpecForValidation",
        requirements=["discount_curve", "model_parameters"],
        fields=[
            FieldDef("notional", "float", "Notional"),
            FieldDef("spot", "float", "Spot"),
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("option_type", "str", "Option type", '"call"'),
            FieldDef("day_count", "DayCountConvention", "Day count", "DayCountConvention.ACT_365"),
        ],
    )
    pricing_plan = SimpleNamespace(
        method="fft_pricing",
        required_market_data={"discount_curve", "model_parameters"},
    )
    product_ir = SimpleNamespace(
        instrument="heston_option",
        model_family="stochastic_volatility",
    )
    compiled_request = SimpleNamespace(
        request=SimpleNamespace(metadata={}),
        validation_contract=None,
        semantic_blueprint=None,
        generation_plan=None,
        execution_plan=SimpleNamespace(route_method="fft_pricing"),
    )

    failures = _validate_build(
        HestonExactPayoffForValidation,
        code="",
        description="Heston option transform route",
        spec_schema=spec_schema,
        validation="smoke",
        compiled_request=compiled_request,
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert not failures


def test_validate_build_uses_merton_jump_payload_for_jump_diffusion_routes(monkeypatch):
    from trellis.agent.executor import _validate_build
    from trellis.agent.planner import FieldDef, SpecSchema
    from trellis.core.types import DayCountConvention
    from trellis.models.merton_jump_diffusion_option import (
        price_merton_jump_diffusion_option_transform,
    )

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)

    @dataclass(frozen=True)
    class MertonExactSpecForValidation:
        notional: float
        spot: float
        strike: float
        expiry_date: date
        option_type: str = "call"
        day_count: DayCountConvention = DayCountConvention.ACT_365

    monkeypatch.setattr(
        sys.modules[__name__],
        "MertonExactSpecForValidation",
        MertonExactSpecForValidation,
        raising=False,
    )

    class MertonExactPayoffForValidation:
        def __init__(self, spec: MertonExactSpecForValidation):
            self._spec = spec

        @property
        def spec(self) -> MertonExactSpecForValidation:
            return self._spec

        @property
        def requirements(self) -> set[str]:
            return {"discount_curve", "black_vol_surface"}

        def evaluate(self, market_state):
            return price_merton_jump_diffusion_option_transform(
                market_state,
                self._spec,
                method="fft",
            )

    spec_schema = SpecSchema(
        class_name="MertonExactPayoffForValidation",
        spec_name="MertonExactSpecForValidation",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional"),
            FieldDef("spot", "float", "Spot"),
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("option_type", "str", "Option type", '"call"'),
            FieldDef("day_count", "DayCountConvention", "Day count", "DayCountConvention.ACT_365"),
        ],
    )
    pricing_plan = SimpleNamespace(
        method="fft_pricing",
        required_market_data={"discount_curve", "black_vol_surface"},
    )
    product_ir = SimpleNamespace(
        instrument="european_option",
        model_family="jump_diffusion",
    )
    compiled_request = SimpleNamespace(
        request=SimpleNamespace(metadata={}),
        validation_contract=None,
        semantic_blueprint=None,
        generation_plan=SimpleNamespace(
            lane_exact_binding_refs=(
                "trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform",
            ),
            route_binding_authority=None,
        ),
        execution_plan=SimpleNamespace(route_method="fft_pricing"),
    )

    failures = _validate_build(
        MertonExactPayoffForValidation,
        code="",
        description="Merton jump-diffusion option transform route",
        spec_schema=spec_schema,
        validation="smoke",
        compiled_request=compiled_request,
        pricing_plan=pricing_plan,
        product_ir=product_ir,
    )

    assert not failures


def test_make_test_payoff_defaults_heston_strike_to_market_spot(monkeypatch):
    from trellis.agent.executor import _make_test_payoff
    from trellis.agent.planner import FieldDef, SpecSchema
    from trellis.core.types import DayCountConvention

    @dataclass(frozen=True)
    class HestonStrikeOnlySpec:
        """Heston European option spec with strike but no scalar spot field."""

        strike: float
        expiry_date: date
        option_type: str = "call"
        day_count: DayCountConvention = DayCountConvention.ACT_365

    monkeypatch.setattr(
        sys.modules[__name__],
        "HestonStrikeOnlySpec",
        HestonStrikeOnlySpec,
        raising=False,
    )

    class HestonStrikeOnlyPayoff:
        def __init__(self, spec: HestonStrikeOnlySpec):
            self._spec = spec

        @property
        def spec(self) -> HestonStrikeOnlySpec:
            return self._spec

    spec_schema = SpecSchema(
        class_name="HestonStrikeOnlyPayoff",
        spec_name="HestonStrikeOnlySpec",
        requirements=["discount_curve", "model_parameters", "spot"],
        fields=[
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("option_type", "str", "Option type", '"call"'),
            FieldDef("day_count", "DayCountConvention", "Day count", "DayCountConvention.ACT_365"),
        ],
    )

    payoff = _make_test_payoff(
        HestonStrikeOnlyPayoff,
        spec_schema,
        date(2024, 11, 15),
        market_state=SimpleNamespace(spot=5890.0),
    )

    assert payoff.spec.strike == pytest.approx(5890.0)


def test_make_test_payoff_uses_unit_notional_for_heston_transform_specs(monkeypatch):
    from trellis.agent.executor import _make_test_payoff
    from trellis.agent.planner import FieldDef, SpecSchema

    @dataclass(frozen=True)
    class HestonNotionalSpec:
        """Heston European option spec with an explicit optional notional."""

        strike: float
        expiry_date: date
        notional: float = 1.0
        is_call: bool = True

    monkeypatch.setattr(
        sys.modules[__name__],
        "HestonNotionalSpec",
        HestonNotionalSpec,
        raising=False,
    )

    class HestonNotionalPayoff:
        def __init__(self, spec: HestonNotionalSpec):
            self._spec = spec

        @property
        def spec(self) -> HestonNotionalSpec:
            return self._spec

    spec_schema = SpecSchema(
        class_name="HestonNotionalPayoff",
        spec_name="HestonNotionalSpec",
        requirements=["discount_curve", "model_parameters", "spot"],
        fields=[
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("notional", "float", "Notional", "1.0"),
            FieldDef("is_call", "bool", "Is call", "True"),
        ],
    )

    payoff = _make_test_payoff(
        HestonNotionalPayoff,
        spec_schema,
        date(2024, 11, 15),
        market_state=SimpleNamespace(spot=5890.0),
    )

    assert payoff.spec.strike == pytest.approx(5890.0)
    assert payoff.spec.notional == pytest.approx(1.0)


def test_deterministic_exact_binding_module_materializes_black_scholes_comparator():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
            "trellis.models.black.black76_asset_or_nothing_call",
            "trellis.models.black.black76_asset_or_nothing_put",
            "trellis.models.black.black76_cash_or_nothing_call",
            "trellis.models.black.black76_cash_or_nothing_put",
            "trellis.models.analytical.terminal_vanilla_from_basis",
            "trellis.core.date_utils.year_fraction",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option analytical comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="black_scholes",
    )

    assert generated is not None
    assert "black76_call(forward, strike, sigma, T)" in generated.code
    assert "black76_put(forward, strike, sigma, T)" in generated.code
    assert "from trellis.core.date_utils import year_fraction" in generated.code
    assert "year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)" in generated.code
    assert "market_state.discount.discount(T)" in generated.code
    assert "market_state.vol_surface.black_vol(max(T, 1e-6), strike)" in generated.code
    assert "terminal_vanilla_from_basis(" not in generated.code
    assert EVALUATE_SENTINEL not in generated.code
    # QUA-862: the Black-Scholes deterministic route now also emits a native
    # ``benchmark_outputs`` method backed by ``equity_vanilla_bs_outputs`` so
    # scorecards can record price + Greeks without a post-hoc bump-and-reprice.
    assert "def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:" in generated.code
    assert "equity_vanilla_bs_outputs(market_state, self._spec)" in generated.code
    assert (
        "from trellis.models.analytical.equity_vanilla_bs import equity_vanilla_bs_outputs"
        in generated.code
    )


@pytest.mark.parametrize(
    ("comparison_target", "expected_method"),
    [
        ("fft", 'method="fft"'),
        ("cos", 'method="cos"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_transform_comparators(
    comparison_target,
    expected_method,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.equity_option_transforms.price_vanilla_equity_option_transform",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option transform comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_vanilla_equity_option_transform(" in generated.code
    assert expected_method in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_method"),
    [
        ("digital_fft", 'method="fft"'),
        ("digital_cos", 'method="cos"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_digital_transform_targets(
    comparison_target,
    expected_method,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.equity_option_transforms.price_equity_digital_option_transform",
        ),
        primitive_plan=None,
        method="fft_pricing",
        instrument_type="digital_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["digital_option"],
        "Digital option transform comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_equity_digital_option_transform(" in generated.code
    assert expected_method in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_call"),
    [
        ("mc_cds", "price_cds_monte_carlo("),
        ("analytical_cds", "price_cds_analytical("),
    ],
)
def test_deterministic_exact_binding_module_materializes_cds_target_aliases(
    comparison_target,
    expected_call,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="monte_carlo" if comparison_target == "mc_cds" else "analytical",
        instrument_type="credit_default_swap",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cds"],
        "CDS comparison target",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_call in generated.code
    assert "build_cds_schedule(" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_uses_metadata_for_cds_target_alias():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(),
        primitive_plan=None,
        method="analytical",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cds"],
        "CDS comparison target with sparse exact plan",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        request_metadata={
            "instrument_type": "credit_default_swap",
            "comparison_target": "analytical_cds",
        },
    )

    assert generated is not None
    assert "price_cds_analytical(" in generated.code
    assert "build_cds_schedule(" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_variance_swap_mc_target():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.monte_carlo.event_aware.price_event_aware_monte_carlo",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="variance_swap",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["variance_swap"],
        "Variance swap MC exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="mc_variance_swap",
    )

    assert generated is not None
    assert "from trellis.models.variance_swap import price_equity_variance_swap_monte_carlo" in generated.code
    assert "price_equity_variance_swap_monte_carlo(" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_route_free_vanilla_black76_body():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option analytical exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=None,
    )

    assert generated is not None
    assert "black76_call(forward, strike, sigma, T)" in generated.code
    assert "black76_put(forward, strike, sigma, T)" in generated.code
    assert "year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)" in generated.code
    assert "return spec.notional * df * undiscounted" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_treats_analytical_black76_target_as_exact():
    """F001-style offline harness lanes must not call the LLM for the generic analytical target."""
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
        ),
        primitive_plan=SimpleNamespace(route="analytical_black76"),
        method="analytical",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option analytical exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="analytical",
    )

    assert generated is not None
    assert "black76_call(forward, strike, sigma, T)" in generated.code
    assert "black76_put(forward, strike, sigma, T)" in generated.code
    assert "equity_vanilla_bs_outputs(market_state, self._spec)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("lane_exact_binding_refs", "expected_fragment"),
    [
        (
            (
                "trellis.models.black.black76_cash_or_nothing_call",
                "trellis.models.black.black76_cash_or_nothing_put",
            ),
            "black76_cash_or_nothing_call(forward, strike, sigma, T)",
        ),
        (
            (
                "trellis.models.black.black76_asset_or_nothing_call",
                "trellis.models.black.black76_asset_or_nothing_put",
            ),
            "black76_asset_or_nothing_call(forward, strike, sigma, T)",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_route_free_digital_black76_body(
    lane_exact_binding_refs,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=lane_exact_binding_refs,
        primitive_plan=None,
        method="analytical",
        instrument_type="digital_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["digital_option"],
        "Digital option analytical exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=None,
    )

    assert generated is not None
    assert expected_fragment in generated.code
    assert "payout_type = str(getattr(spec, \"payout_type\", \"cash_or_nothing\")" in generated.code
    assert "year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_non_black_scholes_route_omits_benchmark_outputs_method():
    """Only the Black-Scholes comparison target injects the native-greeks method (QUA-862)."""
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.analytical.barrier.barrier_option_price",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="barrier_option",
    )
    skeleton = _generate_skeleton(
        STATIC_SPECS["barrier_option"],
        "Barrier option analytical",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=None,
    )
    assert generated is not None
    assert "def benchmark_outputs" not in generated.code
    assert "equity_vanilla_bs_outputs" not in generated.code


def test_deterministic_exact_binding_module_black_scholes_benchmark_outputs_runs_end_to_end():
    """The injected method executes and returns the canonical Greek dict (QUA-862)."""
    from datetime import date as _date

    import numpy as _np

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState

    class _FlatDiscount:
        def __init__(self, rate: float):
            self._rate = float(rate)

        def zero_rate(self, t: float) -> float:
            return self._rate

        def discount(self, t: float) -> float:
            return float(_np.exp(-self._rate * float(t)))

    class _FlatBlackVol:
        def __init__(self, vol: float):
            self._vol = float(vol)

        def black_vol(self, t: float, k: float) -> float:
            return self._vol

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="european_option",
    )
    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option analytical comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="black_scholes",
    )
    assert generated is not None

    namespace: dict = {}
    exec(compile(generated.code, "<qua_862>", "exec"), namespace)  # noqa: S102 -- test harness
    payoff_cls = next(
        obj
        for name, obj in namespace.items()
        if name.endswith("Payoff") and isinstance(obj, type)
    )
    spec_cls = next(
        obj
        for name, obj in namespace.items()
        if name.endswith("Spec") and isinstance(obj, type)
    )
    payoff = payoff_cls(
        spec_cls(
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2025, 11, 15),
            option_type="call",
        )
    )
    market = MarketState(
        as_of=_date(2024, 11, 15),
        settlement=_date(2024, 11, 15),
        discount=_FlatDiscount(0.05),
        vol_surface=_FlatBlackVol(0.25),
    )

    outputs = payoff.benchmark_outputs(market)
    assert set(outputs) == {"price", "delta", "gamma", "vega", "theta"}
    # evaluate() and benchmark_outputs() must agree on the price.
    assert outputs["price"] == pytest.approx(float(payoff.evaluate(market)), rel=1e-10)
    # ATM call spot-check against classical BS numerics at r=5%, σ=25%, T=1.
    assert outputs["delta"] == pytest.approx(0.6274, abs=1e-3)
    assert outputs["vega"] == pytest.approx(37.842, abs=1e-2)


def test_generated_black_scholes_evaluate_and_outputs_honor_dividend_yield():
    from datetime import date as _date
    from math import exp as _exp

    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.core.market_state import MarketState
    from trellis.models.black import black76_call

    class _FlatDiscount:
        def discount(self, t: float) -> float:
            return _exp(-0.05 * float(t))

    class _FlatBlackVol:
        def black_vol(self, _t: float, _strike: float) -> float:
            return 0.25

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="european_option",
    )
    schema = SPECIALIZED_SPECS["european_option_analytical"]
    generated = _materialize_deterministic_exact_binding_module(
        _generate_skeleton(
            schema,
            "Dividend-paying European option analytical comparator",
            generation_plan=generation_plan,
        ),
        generation_plan,
        comparison_target="black_scholes",
    )

    assert generated is not None
    namespace: dict = {}
    exec(compile(generated.code, "<qua_1170_review>", "exec"), namespace)  # noqa: S102
    payoff = namespace[schema.class_name](
        namespace[schema.spec_name](
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=_date(2024, 12, 31),
            option_type="call",
            dividend_yield=0.02,
        )
    )
    market = MarketState(
        as_of=_date(2024, 1, 1),
        settlement=_date(2024, 1, 1),
        discount=_FlatDiscount(),
        vol_surface=_FlatBlackVol(),
    )
    expected = _exp(-0.05) * black76_call(
        100.0 * _exp(0.03),
        100.0,
        0.25,
        1.0,
    )

    assert float(payoff.evaluate(market)) == pytest.approx(expected)
    assert payoff.benchmark_outputs(market)["price"] == pytest.approx(expected)


def test_deterministic_exact_binding_module_fx_vanilla_gk_injects_benchmark_outputs():
    """QUA-878: GK deterministic route injects a native ``benchmark_outputs`` method.

    Mirrors the Black-Scholes test above but for the Garman-Kohlhagen FX
    vanilla route.  Verifies the rendered module imports the GK helper and
    emits the delegating method.
    """
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )

    # Build a minimal inline spec schema rather than depending on
    # STATIC/SPECIALIZED registrations for FX (the FX schema isn't a static
    # spec; it comes from planner inference).  The executor path only needs
    # the skeleton surface + the generation plan.
    from trellis.agent.planner import FieldDef, SpecSchema

    fx_spec_schema = SpecSchema(
        class_name="FXVanillaPayoff",
        spec_name="FXVanillaSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("notional", "float", "Notional"),
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry date"),
            FieldDef("fx_pair", "str", "FX pair"),
            FieldDef("foreign_discount_key", "str", "Foreign discount key"),
            FieldDef("option_type", "str", "call or put", "'call'"),
            FieldDef(
                "day_count",
                "DayCountConvention",
                "Day count convention",
                "DayCountConvention.ACT_365",
            ),
        ],
    )
    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.fx_vanilla.price_fx_vanilla_analytical",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="fx_vanilla",
    )
    skeleton = _generate_skeleton(
        fx_spec_schema,
        "FX vanilla Garman-Kohlhagen analytical",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="black_scholes",
    )
    assert generated is not None
    assert "def benchmark_outputs(self, market_state: MarketState) -> dict[str, float]:" in generated.code
    assert "fx_vanilla_gk_outputs(market_state, self._spec)" in generated.code
    assert (
        "from trellis.models.analytical.fx_vanilla_gk import fx_vanilla_gk_outputs"
        in generated.code
    )
    # Black-Scholes helper must NOT leak into the FX route.
    assert "equity_vanilla_bs_outputs" not in generated.code


def test_deterministic_exact_binding_module_materializes_barrier_helper_with_time_import():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.analytical.barrier.barrier_option_price",
        ),
        primitive_plan=None,
        method="analytical",
        instrument_type="barrier_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["barrier_option"],
        "Barrier option analytical comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "from trellis.core.date_utils import year_fraction" in generated.code
    assert "barrier_option_price(" in generated.code
    assert "year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("helper_ref", "expected_call"),
    [
        (
            "trellis.models.single_barrier_option.price_single_barrier_option_pde_result",
            "price_single_barrier_option_pde_result(market_state, spec).price",
        ),
        (
            "trellis.models.single_barrier_option.price_single_barrier_option_monte_carlo_result",
            "price_single_barrier_option_monte_carlo_result(market_state, spec).price",
        ),
        (
            "trellis.models.double_barrier_option.price_double_barrier_option_pde_result",
            "price_double_barrier_option_pde_result(market_state, spec).price",
        ),
        (
            "trellis.models.double_barrier_option.price_double_barrier_option_monte_carlo_result",
            "price_double_barrier_option_monte_carlo_result(market_state, spec).price",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_barrier_helpers(
    helper_ref,
    expected_call,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="pde_solver",
        instrument_type="barrier_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["barrier_option"],
        "Double barrier checked helper",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert expected_call in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_black_scholes_comparator_satisfies_required_primitive_validation():
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS
    from trellis.agent.semantic_validation import validate_semantics

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.black.black76_call",
            "trellis.models.black.black76_put",
            "trellis.models.black.black76_asset_or_nothing_call",
            "trellis.models.black.black76_asset_or_nothing_put",
            "trellis.models.black.black76_cash_or_nothing_call",
            "trellis.models.black.black76_cash_or_nothing_put",
        ),
        primitive_plan=SimpleNamespace(
            primitives=(
                SimpleNamespace(
                    required=True,
                    role="pricing_kernel",
                    module="trellis.models.black",
                    symbol="black76_call",
                    excluded=False,
                ),
                SimpleNamespace(
                    required=True,
                    role="pricing_kernel",
                    module="trellis.models.black",
                    symbol="black76_put",
                    excluded=False,
                ),
                SimpleNamespace(
                    required=False,
                    role="assembly_helper",
                    module="trellis.models.analytical",
                    symbol="terminal_vanilla_from_basis",
                    excluded=False,
                ),
                SimpleNamespace(
                    required=True,
                    role="time_measure",
                    module="trellis.core.date_utils",
                    symbol="year_fraction",
                    excluded=False,
                ),
            ),
            blockers=(),
        ),
        method="analytical",
        instrument_type="european_option",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["european_option_analytical"],
        "European option analytical comparator",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target="black_scholes",
    )

    assert generated is not None
    report = validate_semantics(generated.code, generation_plan=generation_plan)

    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("ho_lee_tree", 'model="ho_lee"'),
        ("hull_white_tree", 'model="hull_white"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_zcb_option_tree_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.zcb_option_tree.price_zcb_option_tree",),
        primitive_plan=None,
        method="rate_tree",
        instrument_type="zcb_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["zcb_option"],
        "ZCB option tree",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_zcb_option_tree(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("vasicek_tree", 'model="vasicek"'),
        ("cir_tree", 'model="cir"'),
        ("vasicek_analytical", 'model="vasicek"'),
        ("cir_analytical", 'model="cir"'),
    ],
)
def test_deterministic_exact_binding_module_materializes_short_rate_bond_helper_wrappers(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    helper = (
        "trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_tree"
        if comparison_target.endswith("_tree")
        else "trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_analytical"
    )
    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper,),
        primitive_plan=None,
        method="rate_tree" if comparison_target.endswith("_tree") else "analytical",
        instrument_type="short_rate_bond",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["short_rate_bond"],
        "Short-rate zero-coupon bond",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_fragment in generated.code
    assert "price_short_rate_zero_coupon_bond_" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("comparison_target", "expected_fragment"),
    [
        ("gaussian_copula", 'copula_family="gaussian"'),
        ("student_t_copula", 'copula_family="student_t", degrees_of_freedom=5.0, n_paths=40000, seed=42'),
    ],
)
def test_deterministic_exact_binding_module_materializes_credit_basket_tranche_helper_wrapper(
    comparison_target,
    expected_fragment,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.credit_basket_copula.price_credit_basket_tranche",),
        primitive_plan=None,
        method="copula",
        instrument_type="cdo",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cdo"],
        "CDO tranche copula",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert "price_credit_basket_tranche(market_state, spec" in generated.code
    assert expected_fragment in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("helper_ref", "expected_call"),
    [
        (
            "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_recursive",
            "price_credit_portfolio_loss_distribution_recursive(market_state, spec, copula_family=\"gaussian\")",
        ),
        (
            "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_transform_proxy",
            "price_credit_portfolio_loss_distribution_transform_proxy(market_state, spec, copula_family=\"gaussian\")",
        ),
        (
            "trellis.models.credit_basket_copula.price_credit_portfolio_loss_distribution_monte_carlo",
            "price_credit_portfolio_loss_distribution_monte_carlo(market_state, spec, copula_family=\"gaussian\", n_paths=40000, seed=42)",
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_credit_loss_distribution_wrappers(
    helper_ref,
    expected_call,
):
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="copula",
        instrument_type="credit_loss_distribution",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["credit_loss_distribution"],
        "Credit loss distribution",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert expected_call in generated.code
    assert EVALUATE_SENTINEL not in generated.code


@pytest.mark.parametrize(
    ("helper_ref", "expected_call"),
    [
        ("trellis.models.rate_style_swaption.price_swaption_black76", "price_swaption_black76"),
        ("trellis.models.rate_style_swaption_tree.price_swaption_tree", "price_swaption_tree"),
        ("trellis.models.rate_style_swaption.price_swaption_monte_carlo", "price_swaption_monte_carlo"),
    ],
)
def test_deterministic_exact_binding_module_threads_explicit_swaption_comparison_regime(
    helper_ref,
    expected_call,
):
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS
    from trellis.agent.valuation_context import (
        EngineModelSpec,
        PotentialSpec,
        RatesCurveRoleSpec,
        SourceSpec,
    )

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="analytical",
        instrument_type="swaption",
    )
    semantic_blueprint = SimpleNamespace(
        valuation_context=SimpleNamespace(
            engine_model_spec=EngineModelSpec(
                model_family="rates",
                model_name="hull_white_1f",
                state_semantics=("short_rate",),
                potential=PotentialSpec(discount_term="risk_free_rate"),
                sources=(SourceSpec(source_kind="coupon_stream"),),
                calibration_requirements=("bootstrap_curve", "fit_hw_strip"),
                backend_hints=("analytical",),
                parameter_overrides={"mean_reversion": 0.05, "sigma": 0.01},
                rates_curve_roles=RatesCurveRoleSpec(
                    discount_curve_role="discount_curve",
                    forecast_curve_role="forward_curve",
                ),
            )
        )
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["swaption"],
        "European payer swaption",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        semantic_blueprint=semantic_blueprint,
    )

    assert generated is not None
    if expected_call == "price_swaption_monte_carlo":
        assert (
            "price_swaption_monte_carlo("
            "market_state, spec, n_paths=20000, seed=42, mean_reversion=0.05, sigma=0.01)"
        ) in generated.code
    else:
        assert f"{expected_call}(market_state, spec, mean_reversion=0.05, sigma=0.01)" in generated.code


@pytest.mark.parametrize(
    ("helper_ref", "comparison_target", "expected_call"),
    [
        (
            "trellis.models.basket_option.price_basket_option_analytical",
            "stulz_rainbow",
            'price_basket_option_analytical(market_state, spec, comparison_target="stulz_rainbow")',
        ),
        (
            "trellis.models.basket_option.price_basket_option_monte_carlo",
            "mc_spread_2d",
            'price_basket_option_monte_carlo(market_state, spec, comparison_target="mc_spread_2d", seed=42)',
        ),
        (
            "trellis.models.basket_option.price_basket_option_transform_proxy",
            "fft_spread_2d",
            'price_basket_option_transform_proxy(market_state, spec, comparison_target="fft_spread_2d")',
        ),
    ],
)
def test_deterministic_exact_binding_module_materializes_typed_basket_helpers(
    helper_ref,
    comparison_target,
    expected_call,
):
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(helper_ref,),
        primitive_plan=None,
        method="analytical",
        instrument_type="basket_option",
    )
    skeleton = _generate_skeleton(
        STATIC_SPECS["basket_option"],
        "Generic basket option",
        generation_plan=generation_plan,
    )

    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
        comparison_target=comparison_target,
    )

    assert generated is not None
    assert expected_call in generated.code


def test_extract_instrument_type_uses_shared_lower_layer_mapping():
    from trellis.agent.executor import _extract_instrument_type

    assert _extract_instrument_type("European call option on a zero-coupon bond") == "zcb_option"
    assert _extract_instrument_type("CDO tranche: Gaussian copula vs Student-t copula") == "cdo"


def test_extract_instrument_type_does_not_widen_generic_bond_or_swap_wording():
    from trellis.agent.executor import _extract_instrument_type

    assert _extract_instrument_type("Generic bond workflow summary") == "unknown"
    assert _extract_instrument_type("Desk swap exposure summary") == "unknown"


def test_resolve_lower_layer_instrument_type_prefers_request_metadata_over_description():
    from trellis.agent.executor import _resolve_lower_layer_instrument_type

    compiled_request = SimpleNamespace(
        request=SimpleNamespace(
            instrument_type=None,
            metadata={
                "instrument_type": "zcb_option",
                "runtime_contract": {"instrument_type": "zcb_option"},
            },
        )
    )

    resolved = _resolve_lower_layer_instrument_type(
        "Generic European option wording that would otherwise widen the family",
        compiled_request=compiled_request,
        product_ir=SimpleNamespace(instrument="zcb_option"),
    )

    assert resolved == "zcb_option"


def test_resolve_lower_layer_instrument_type_falls_back_to_description_only_when_needed():
    from trellis.agent.executor import _resolve_lower_layer_instrument_type

    resolved = _resolve_lower_layer_instrument_type(
        "CDO tranche: Gaussian copula vs Student-t copula",
        compiled_request=SimpleNamespace(request=SimpleNamespace(instrument_type=None, metadata={})),
        product_ir=None,
    )

    assert resolved == "cdo"


def test_generate_skeleton_prefills_exact_binding_imports_without_generic_noise():
    from trellis.agent.executor import _generate_skeleton
    from trellis.agent.planner import FieldDef, SpecSchema

    spec_schema = SpecSchema(
        class_name="AmericanPutTreePayoff",
        spec_name="AmericanPutTreeSpec",
        requirements=["discount_curve", "black_vol_surface"],
        fields=[
            FieldDef("spot", "float", "Spot"),
            FieldDef("strike", "float", "Strike"),
            FieldDef("expiry_date", "date", "Expiry"),
            FieldDef("option_type", "str", "Option type", '"put"'),
            FieldDef("exercise_style", "str", "Exercise style", '"american"'),
        ],
    )

    skeleton = _generate_skeleton(
        spec_schema,
        "American put: equity tree knowledge-light proving",
        generation_plan=SimpleNamespace(
            method="rate_tree",
            instrument_type="american_put",
            lane_exact_binding_refs=(
                "trellis.models.equity_option_tree.price_vanilla_equity_option_tree",
            ),
            primitive_plan=None,
        ),
    )

    assert (
        "from trellis.models.equity_option_tree import price_vanilla_equity_option_tree"
        in skeleton
    )
    assert "option_type: str = 'put'" in skeleton
    assert "exercise_style: str = 'american'" in skeleton
    assert "from trellis.core.date_utils import generate_schedule, year_fraction" not in skeleton
    assert "from trellis.models.black import black76_call, black76_put" not in skeleton


def test_generate_skeleton_prefills_cds_exact_bindings_from_compiler_plan():
    from trellis.agent.executor import _generate_skeleton
    from trellis.agent.planner import FieldDef, SpecSchema

    spec_schema = SpecSchema(
        class_name="CDSPayoff",
        spec_name="CDSSpec",
        requirements=["credit_curve", "discount_curve"],
        fields=[
            FieldDef("notional", "float", "Notional"),
            FieldDef("spread", "float", "Spread"),
            FieldDef("start_date", "date", "Start"),
            FieldDef("end_date", "date", "End"),
            FieldDef("recovery", "float", "Recovery", "0.4"),
            FieldDef("frequency", "Frequency", "Coupon frequency", "Frequency.QUARTERLY"),
            FieldDef(
                "day_count",
                "DayCountConvention",
                "Day count",
                "DayCountConvention.ACT_360",
            ),
        ],
    )

    skeleton = _generate_skeleton(
        spec_schema,
        "CDS pricing: hazard rate MC vs survival prob analytical",
        generation_plan=SimpleNamespace(
            method="analytical",
            instrument_type="credit_default_swap",
            lane_exact_binding_refs=(
                "trellis.models.credit_default_swap.build_cds_schedule",
                "trellis.models.credit_default_swap.price_cds_analytical",
            ),
            primitive_plan=None,
        ),
    )

    assert (
        "from trellis.models.credit_default_swap import build_cds_schedule, "
        "price_cds_analytical" in skeleton
    )
    assert "from trellis.core.types import DayCountConvention, Frequency" in skeleton
    assert "from trellis.core.date_utils import generate_schedule, year_fraction" not in skeleton
    assert "from trellis.models.black import black76_call, black76_put" not in skeleton


def test_deterministic_exact_binding_module_materializes_cds_analytical_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.credit_default_swap.price_cds_analytical",),
        primitive_plan=None,
        method="analytical",
        instrument_type="credit_default_swap",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["cds"],
        "Single-name CDS analytical exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "from trellis.models.credit_default_swap import build_cds_schedule" in generated.code
    assert "schedule = build_cds_schedule(" in generated.code
    assert 'time_origin=getattr(spec, "valuation_date", None) or spec.start_date' in generated.code
    assert "return price_cds_analytical(" in generated.code
    assert "credit_curve=market_state.credit_curve" in generated.code
    assert "discount_curve=market_state.discount" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_cds_monte_carlo_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import SPECIALIZED_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.models.credit_default_swap.price_cds_monte_carlo",),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="credit_default_swap",
    )

    skeleton = _generate_skeleton(
        SPECIALIZED_SPECS["cds_monte_carlo"],
        "Single-name CDS Monte Carlo exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "from trellis.models.credit_default_swap import build_cds_schedule" in generated.code
    assert "schedule = build_cds_schedule(" in generated.code
    assert "return price_cds_monte_carlo(" in generated.code
    assert 'n_paths=getattr(spec, "n_paths", 250000)' in generated.code
    assert "seed=42" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_ranked_basket_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=(
            "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
            "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo",
        ),
        primitive_plan=None,
        method="monte_carlo",
        instrument_type="rainbow_option",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["rainbow_option"],
        "Asian rainbow average-best-of exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "RankedObservationBasketSpec(" in generated.code
    assert 'constituents=getattr(spec, "constituents", None) or spec.underliers' in generated.code
    assert (
        'aggregation_rule=getattr(spec, "aggregation_rule", None) or "average_locked_levels"'
        in generated.code
    )
    assert "resolved = resolve_basket_semantics(market_state, helper_spec)" in generated.code
    assert "return price_ranked_observation_basket_monte_carlo(helper_spec, resolved)" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_deterministic_exact_binding_module_materializes_nth_to_default_wrapper():
    from trellis.agent.executor import (
        EVALUATE_SENTINEL,
        _generate_skeleton,
        _materialize_deterministic_exact_binding_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        lane_exact_binding_refs=("trellis.instruments.nth_to_default.price_nth_to_default_basket",),
        primitive_plan=None,
        method="copula",
        instrument_type="nth_to_default",
    )

    skeleton = _generate_skeleton(
        STATIC_SPECS["nth_to_default"],
        "Nth-to-default exact binding",
        generation_plan=generation_plan,
    )
    generated = _materialize_deterministic_exact_binding_module(
        skeleton,
        generation_plan,
    )

    assert generated is not None
    assert "T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)" in generated.code
    assert "return price_nth_to_default_basket(" in generated.code
    assert "credit_curve=market_state.credit_curve" in generated.code
    assert "discount_curve=market_state.discount" in generated.code
    assert EVALUATE_SENTINEL not in generated.code


def test_extract_fragment_body_repairs_orphan_indentation():
    from trellis.agent.executor import _extract_fragment_body

    body = _extract_fragment_body(
        [
            "        spec = self._spec",
            "                T = year_fraction(market_state.as_of, spec.expiry_date, spec.day_count)",
            "                if T <= 0.0:",
            "                    return 0.0",
        ]
    )

    assert body.splitlines() == [
        "spec = self._spec",
        "T = year_fraction(market_state.as_of, spec.expiry_date, spec.day_count)",
        "if T <= 0.0:",
        "    return 0.0",
    ]


def test_extract_evaluate_body_from_module_text_repairs_orphan_indentation():
    from trellis.agent.executor import _extract_evaluate_body_from_module_text

    module_text = """
class Demo:
    def evaluate(self, market_state):
        spec = self._spec
                T = year_fraction(market_state.as_of, spec.expiry_date, spec.day_count)
                if T <= 0.0:
                    return 0.0
"""

    body = _extract_evaluate_body_from_module_text(module_text)

    assert body.splitlines() == [
        "spec = self._spec",
        "T = year_fraction(market_state.as_of, spec.expiry_date, spec.day_count)",
        "if T <= 0.0:",
        "    return 0.0",
    ]


def test_extract_fragment_body_repairs_misnested_elif_else():
    from trellis.agent.executor import _extract_fragment_body

    body = _extract_fragment_body(
        [
            "        if opt_type == \"call\":",
            "            pv = df * black76_call(forward, spec.strike, sigma, T)",
            "            elif opt_type == \"put\":",
            "                pv = df * black76_put(forward, spec.strike, sigma, T)",
            "                else:",
            "                    raise ValueError(\"unsupported option_type\")",
        ]
    )

    assert body.splitlines() == [
        "if opt_type == \"call\":",
        "    pv = df * black76_call(forward, spec.strike, sigma, T)",
        "elif opt_type == \"put\":",
        "    pv = df * black76_put(forward, spec.strike, sigma, T)",
        "else:",
        "    raise ValueError(\"unsupported option_type\")",
    ]


def test_extract_fragment_body_dedents_offset_tail_after_first_line():
    from trellis.agent.executor import _extract_fragment_body

    body = _extract_fragment_body(
        [
            "spec = self._spec",
            "        if market_state.discount is None:",
            "            raise ValueError(\"missing discount\")",
            "        if market_state.credit_curve is None:",
            "            raise ValueError(\"missing credit\")",
            "        spread = float(spec.spread)",
            "        return spread",
        ]
    )

    assert body.splitlines() == [
        "spec = self._spec",
        "if market_state.discount is None:",
        "    raise ValueError(\"missing discount\")",
        "if market_state.credit_curve is None:",
        "    raise ValueError(\"missing credit\")",
        "spread = float(spec.spread)",
        "return spread",
    ]


def test_extract_fragment_body_repairs_missing_indent_after_block_opener():
    from trellis.agent.executor import _extract_fragment_body

    body = _extract_fragment_body(
        [
            "        spread = float(spec.spread)",
            "        if spread > 1.0:",
            "        spread *= 1e-4",
            "        return spread",
        ]
    )

    assert body.splitlines() == [
        "spread = float(spec.spread)",
        "if spread > 1.0:",
        "    spread *= 1e-4",
        "return spread",
    ]


def test_make_test_payoff_populates_enum_defaults_for_frequency_and_day_count(monkeypatch):
    from trellis.agent.executor import _make_test_payoff
    from trellis.core.types import DayCountConvention, Frequency

    @dataclass(frozen=True)
    class DemoSpec:
        frequency: Frequency
        day_count: DayCountConvention

    class DemoPayoff:
        def __init__(self, spec):
            self._spec = spec

    module = ModuleType("demo_make_test_payoff_module")
    DemoSpec.__module__ = module.__name__
    DemoPayoff.__module__ = module.__name__
    setattr(module, "DemoSpec", DemoSpec)
    setattr(module, "DemoPayoff", DemoPayoff)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    spec_schema = SimpleNamespace(
        spec_name="DemoSpec",
        fields=[
            SimpleNamespace(name="frequency", type="Frequency", default=None),
            SimpleNamespace(name="day_count", type="DayCountConvention", default=None),
        ],
    )

    payoff = _make_test_payoff(DemoPayoff, spec_schema, date(2024, 11, 15))

    assert payoff._spec.frequency == Frequency.SEMI_ANNUAL
    assert payoff._spec.day_count == DayCountConvention.ACT_360


def test_build_payoff_code_generation_stage_uses_instrument_type_metadata(monkeypatch):
    from trellis.agent.executor import build_payoff

    captured: dict[str, object] = {}
    pricing_plan = SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=set(),
        model_to_build=None,
        reasoning="test analytical route",
        selection_reason="unit_test",
        assumption_summary=(),
        sensitivity_support=None,
    )
    product_ir = SimpleNamespace(instrument="credit_default_swap")
    compiled_request = SimpleNamespace(
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        request=SimpleNamespace(request_id="executor_build_metadata_123", metadata={}),
        linear_issue_identifier=None,
        generation_plan=None,
        knowledge_summary={},
    )
    plan = SimpleNamespace(
        steps=[SimpleNamespace(module_path="trellis/instruments/_agent/test_metadata.py")],
        spec_schema=SimpleNamespace(
            spec_name="TestSpec",
            class_name="TestPayoff",
            fields=(),
            requirements=(),
        ),
    )
    generation_plan = SimpleNamespace(
        method="analytical",
        instrument_type="credit_default_swap",
        primitive_plan=SimpleNamespace(
            engine_family="analytical",
            blockers=(),
            route="test_route",
        ),
        blocker_report=None,
        new_primitive_workflow=None,
    )

    @contextmanager
    def fake_llm_usage_stage(stage, metadata=None):
        captured["stage"] = stage
        captured["metadata"] = metadata or {}
        yield []

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._append_agent_observation", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.planner.plan_build", lambda *args, **kwargs: plan)
    monkeypatch.setattr("trellis.agent.executor._try_import_existing", lambda plan: None)
    monkeypatch.setattr("trellis.agent.executor.build_generation_plan", lambda **kwargs: generation_plan)
    monkeypatch.setattr("trellis.agent.executor._emit_analytical_trace_metadata", lambda **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._reference_modules", lambda *args, **kwargs: ())
    monkeypatch.setattr("trellis.agent.executor._gather_references", lambda modules: [])
    monkeypatch.setattr("trellis.agent.builder.ensure_agent_package", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_default_model", lambda: "gpt-5-mini")
    monkeypatch.setattr("trellis.agent.config.get_model_for_stage", lambda stage, model=None: model or "gpt-5-mini")
    monkeypatch.setattr("trellis.agent.config.llm_usage_stage", fake_llm_usage_stage)
    monkeypatch.setattr("trellis.agent.config.summarize_llm_usage", lambda records: {})
    monkeypatch.setattr("trellis.agent.config.enforce_llm_token_budget", lambda stage=None: None)

    def fake_generate_module(*args, **kwargs):
        raise RuntimeError("stub generation failed")

    monkeypatch.setattr("trellis.agent.executor._generate_module", fake_generate_module)

    with pytest.raises(RuntimeError, match="stub generation failed"):
        build_payoff(
            "CDS pricing metadata regression",
            compiled_request=compiled_request,
            instrument_type="credit_default_swap",
            market_state=SimpleNamespace(
                selected_curve_names={},
                available_capabilities=set(),
            ),
            max_retries=1,
            model="gpt-5-mini",
        )

    assert captured["stage"] == "code_generation"
    assert captured["metadata"]["instrument_type"] == "credit_default_swap"
    assert captured["metadata"]["model"] == "gpt-5-mini"
    assert captured["metadata"]["attempt"] == 1


def test_build_payoff_fresh_build_still_uses_deterministic_exact_binding(monkeypatch):
    from trellis.agent.executor import build_payoff

    pricing_plan = SimpleNamespace(
        method="copula",
        method_modules=("trellis.models.credit_basket_copula",),
        required_market_data={"credit_curve"},
        model_to_build=None,
        reasoning="exact binding available",
        selection_reason="unit_test",
        assumption_summary=(),
        sensitivity_support=None,
    )
    product_ir = SimpleNamespace(instrument="cdo")
    compiled_request = SimpleNamespace(
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        request=SimpleNamespace(
            request_id="executor_build_exact_binding_fresh_123",
            metadata={"comparison_target": "gaussian_copula"},
        ),
        linear_issue_identifier=None,
        generation_plan=None,
        knowledge_summary={},
        semantic_blueprint=None,
    )
    plan = SimpleNamespace(
        steps=[SimpleNamespace(module_path="trellis/instruments/_agent/test_cdo.py")],
        spec_schema=SimpleNamespace(
            spec_name="CDOTrancheSpec",
            class_name="CDOTranchePayoff",
            fields=(),
            requirements=(),
        ),
    )
    generation_plan = SimpleNamespace(
        method="copula",
        instrument_type="cdo",
        primitive_plan=SimpleNamespace(
            engine_family="copula",
            blockers=(),
            route="copula_loss_distribution",
        ),
        blocker_report=None,
        new_primitive_workflow=None,
        lane_exact_binding_refs=("trellis.models.credit_basket_copula.price_credit_basket_tranche",),
    )

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._append_agent_observation", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.planner.plan_build", lambda *args, **kwargs: plan)
    monkeypatch.setattr("trellis.agent.executor._try_import_existing", lambda plan: None)
    monkeypatch.setattr("trellis.agent.executor.build_generation_plan", lambda **kwargs: generation_plan)
    monkeypatch.setattr("trellis.agent.executor._emit_analytical_trace_metadata", lambda **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._reference_modules", lambda *args, **kwargs: ())
    monkeypatch.setattr("trellis.agent.executor._gather_references", lambda modules: [])
    monkeypatch.setattr("trellis.agent.builder.ensure_agent_package", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_default_model", lambda: "gpt-5-mini")
    monkeypatch.setattr("trellis.agent.config.get_model_for_stage", lambda stage, model=None: model or "gpt-5-mini")
    monkeypatch.setattr("trellis.agent.config.summarize_llm_usage", lambda records: {})
    monkeypatch.setattr("trellis.agent.config.enforce_llm_token_budget", lambda stage=None: None)

    def fake_deterministic(*args, **kwargs):
        raise RuntimeError("deterministic exact binding path used")

    monkeypatch.setattr(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        fake_deterministic,
    )
    monkeypatch.setattr(
        "trellis.agent.executor._generate_module",
        lambda *args, **kwargs: pytest.fail("LLM generation should not run for fresh exact bindings"),
    )

    with pytest.raises(RuntimeError, match="deterministic exact binding path used"):
        build_payoff(
            "CDO tranche exact binding regression",
            compiled_request=compiled_request,
            instrument_type="cdo",
            market_state=SimpleNamespace(
                selected_curve_names={},
                available_capabilities={"credit_curve"},
            ),
            max_retries=1,
            model="gpt-5-mini",
            fresh_build=True,
        )


def test_p001_semantic_execution_shim_source_is_thin_adapter():
    from trellis.agent.codegen_guardrails import validate_generated_imports
    from trellis.agent.executor import (
        _generate_skeleton,
        _materialize_semantic_execution_shim_module,
    )
    from trellis.agent.planner import STATIC_SPECS

    generation_plan = SimpleNamespace(
        method="rate_tree",
        instrument_type="rainbow_option",
        primitive_plan=SimpleNamespace(blockers=(), route="exercise_lattice"),
        approved_modules=(
            "trellis.core.market_state",
            "trellis.core.payoff",
            "trellis.core.types",
            "trellis.execution",
        ),
    )
    skeleton = _generate_skeleton(
        STATIC_SPECS["rainbow_option"],
        "P001 Bermudan best-of rainbow proof",
        generation_plan=generation_plan,
    )

    generated = _materialize_semantic_execution_shim_module(
        skeleton,
        generation_plan,
        request_metadata={"task_id": "P001"},
        comparison_target="rate_tree",
    )

    assert generated is not None
    assert "price_bermudan_best_of_basket_from_compat_spec" in generated.code
    assert 'method="lattice"' in generated.code
    assert "build_rate_lattice" not in generated.code
    assert "state_space" not in generated.code
    assert "from trellis.models." not in generated.code
    assert validate_generated_imports(generated.code, generation_plan).ok


def test_build_payoff_fresh_build_bypasses_cached_generated_module_even_without_deterministic_route(
    monkeypatch,
):
    from trellis.agent.executor import build_payoff

    pricing_plan = SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=set(),
        model_to_build=None,
        reasoning="fresh build",
        selection_reason="unit_test",
        assumption_summary=(),
        sensitivity_support=None,
    )
    product_ir = SimpleNamespace(instrument="barrier_option")
    compiled_request = SimpleNamespace(
        product_ir=product_ir,
        pricing_plan=pricing_plan,
        request=SimpleNamespace(request_id="executor_build_fresh_barrier_123", metadata={}),
        linear_issue_identifier=None,
        generation_plan=None,
        knowledge_summary={},
        semantic_blueprint=None,
    )
    plan = SimpleNamespace(
        steps=[SimpleNamespace(module_path="instruments/_agent/barrieroption.py")],
        spec_schema=SimpleNamespace(
            spec_name="BarrierOptionSpec",
            class_name="BarrierOptionPayoff",
            fields=(),
            requirements=(),
        ),
    )
    generation_plan = SimpleNamespace(
        method="analytical",
        instrument_type="barrier_option",
        primitive_plan=SimpleNamespace(
            engine_family="analytical",
            blockers=(),
            route="barrier_analytical",
        ),
        blocker_report=None,
        new_primitive_workflow=None,
        lane_exact_binding_refs=(),
    )

    monkeypatch.setattr("trellis.agent.executor._record_platform_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._append_agent_observation", lambda *args, **kwargs: None)
    monkeypatch.setattr("trellis.agent.planner.plan_build", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        "trellis.agent.executor._try_import_existing",
        lambda plan: type("CachedBarrierPayoff", (), {}),
    )
    monkeypatch.setattr(
        "trellis.agent.executor._is_deterministic_supported_route",
        lambda plan: False,
    )
    monkeypatch.setattr("trellis.agent.executor.build_generation_plan", lambda **kwargs: generation_plan)
    monkeypatch.setattr("trellis.agent.executor._emit_analytical_trace_metadata", lambda **kwargs: None)
    monkeypatch.setattr("trellis.agent.executor._reference_modules", lambda *args, **kwargs: ())
    monkeypatch.setattr("trellis.agent.executor._gather_references", lambda modules: [])
    monkeypatch.setattr("trellis.agent.builder.ensure_agent_package", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_default_model", lambda: "gpt-5-mini")
    monkeypatch.setattr(
        "trellis.agent.config.get_model_for_stage",
        lambda stage, model=None: model or "gpt-5-mini",
    )
    monkeypatch.setattr("trellis.agent.config.summarize_llm_usage", lambda records: {})
    monkeypatch.setattr("trellis.agent.config.enforce_llm_token_budget", lambda stage=None: None)
    monkeypatch.setattr(
        "trellis.agent.executor._materialize_deterministic_exact_binding_module",
        lambda *args, **kwargs: None,
    )

    def fake_generate(*args, **kwargs):
        raise RuntimeError("fresh build generation path used")

    monkeypatch.setattr("trellis.agent.executor._generate_module", fake_generate)

    with pytest.raises(RuntimeError, match="fresh build generation path used"):
        build_payoff(
            "Barrier option fresh build regression",
            compiled_request=compiled_request,
            instrument_type="barrier_option",
            market_state=SimpleNamespace(
                selected_curve_names={},
                available_capabilities={"discount_curve", "black_vol_surface"},
            ),
            max_retries=1,
            model="gpt-5-mini",
            fresh_build=True,
        )


def test_resolve_output_target_uses_benchmark_generated_root_for_financepy_tasks():
    from trellis.agent.executor import _resolve_output_target

    output_file_path, output_module_path, module_name = _resolve_output_target(
        "instruments/_agent/barrieroption.py",
        fresh_build=True,
        request_metadata={
            "task_corpus": "benchmark_financepy",
            "task_id": "F009",
            "comparison_target": "analytical",
        },
    )

    assert "task_runs/financepy_benchmarks/generated/f009/analytical/barrieroption.py" in str(
        output_file_path
    ).replace("\\", "/")
    assert output_module_path == "task_runs/financepy_benchmarks/generated/f009/analytical/barrieroption.py"
    assert module_name == "trellis_benchmarks._fresh.f009.analytical.barrieroption"


def test_resolve_output_target_sanitizes_agent_prompt_filename():
    from trellis.agent.executor import _resolve_output_target

    bad_module_path = (
        "instruments/_agent/buildapricerfor:cranknicolsonrannachersmoothingfordiscontinuouss\n\n"
        "constructmethods:pdesolver\n"
        "comparisontargets:cnrannacher(pdesolver),cnstandard(pdesolver).py"
    )

    _, output_module_path, module_name = _resolve_output_target(
        bad_module_path,
        fresh_build=False,
        request_metadata={"task_id": "T23", "comparison_target": "cn_rannacher"},
    )

    assert output_module_path.startswith("instruments/_agent/")
    assert output_module_path.endswith(".py")
    assert "\n" not in output_module_path
    assert ":" not in output_module_path
    assert "buildapricerfor" not in output_module_path
    assert len(Path(output_module_path).name) <= 75
    assert module_name.startswith("trellis.instruments._agent.")


def test_knowledge_retrieval_stage_maps_builder_retry_reasons():
    from trellis.agent.executor import _knowledge_retrieval_stage

    assert _knowledge_retrieval_stage(
        audience="builder",
        attempt_number=1,
        retry_reason=None,
    ) == "initial_build"
    assert _knowledge_retrieval_stage(
        audience="builder",
        attempt_number=2,
        retry_reason="import_validation",
    ) == "import_validation_failed"
    assert _knowledge_retrieval_stage(
        audience="builder",
        attempt_number=2,
        retry_reason="semantic_validation",
    ) == "semantic_validation_failed"
    assert _knowledge_retrieval_stage(
        audience="builder",
        attempt_number=2,
        retry_reason="lite_review",
    ) == "lite_review_failed"
    assert _knowledge_retrieval_stage(
        audience="builder",
        attempt_number=2,
        retry_reason="actual_market_smoke",
    ) == "actual_market_smoke_failed"
    assert _knowledge_retrieval_stage(
        audience="review",
        attempt_number=1,
        retry_reason=None,
    ) == "critic_review"
