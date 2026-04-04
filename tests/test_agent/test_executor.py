from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from contextlib import contextmanager
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


@pytest.mark.legacy_compat
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
    assert "from trellis.core.types import Frequency" in skeleton
    assert "from trellis.core.date_utils import generate_schedule, year_fraction" not in skeleton
    assert "from trellis.models.black import black76_call, black76_put" not in skeleton


def test_generate_skeleton_prefills_exact_binding_imports_without_generic_noise():
    from trellis.agent.executor import _generate_skeleton
    from trellis.agent.planner import FieldDef, SpecSchema

    spec_schema = SpecSchema(
        class_name="AmericanOptionPayoff",
        spec_name="AmericanPutEquitySpec",
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
