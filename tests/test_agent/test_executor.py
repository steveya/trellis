from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import ModuleType, SimpleNamespace
import sys

import pytest


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


def test_record_lesson_maps_why_to_legacy_explanation(monkeypatch):
    from trellis.agent.test_resolution import Lesson, record_lesson

    captured: dict[str, object] = {}

    def fake_append_lesson(lesson: dict[str, object]) -> None:
        captured.update(lesson)

    monkeypatch.setattr("trellis.agent.experience.append_lesson", fake_append_lesson)

    record_lesson(
        Lesson(
            category="monte_carlo",
            title="Bridge legacy lesson fields",
            mistake="The bridge used the wrong field name.",
            why="Legacy experience expects explanation, not why.",
            detect="The append helper rejects missing explanation.",
            fix="Map why into explanation before appending.",
        )
    )

    assert captured["category"] == "monte_carlo"
    assert captured["title"] == "Bridge legacy lesson fields"
    assert captured["explanation"] == "Legacy experience expects explanation, not why."
    assert captured["fix"] == "Map why into explanation before appending."
    assert captured["symptoms"] == [
        "The bridge used the wrong field name.",
        "The append helper rejects missing explanation.",
    ]
    assert "why" not in captured


def test_record_resolved_failures_fails_hard_on_missing_lesson_fields(monkeypatch):
    from trellis.agent.executor import _record_resolved_failures

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
        lambda lesson: pytest.fail("record_lesson should not be reached when fields are missing"),
    )

    with pytest.raises(RuntimeError, match="LLM lesson output missing fields"):
        _record_resolved_failures(
            ["example validation failure"],
            "European call option",
            SimpleNamespace(method="analytical"),
            "gpt-5-mini",
        )


def test_generate_skeleton_quotes_string_defaults_but_keeps_symbolic_defaults():
    from trellis.agent.executor import _generate_skeleton
    from trellis.agent.planner import FieldDef, SpecSchema

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
