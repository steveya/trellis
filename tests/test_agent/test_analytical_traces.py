from __future__ import annotations

from types import SimpleNamespace


def _black76_pricing_plan():
    return SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=(),
        model_to_build="vanilla_option",
        reasoning="trace black76 analytical assembly",
        modeling_requirements=(),
    )


def test_analytical_trace_round_trip_and_render(tmp_path):
    from trellis.agent.analytical_traces import (
        emit_analytical_trace_from_generation_plan,
        load_analytical_traces,
        render_analytical_trace,
    )
    from trellis.agent.codegen_guardrails import build_generation_plan

    plan = build_generation_plan(
        pricing_plan=_black76_pricing_plan(),
        instrument_type="vanilla_option",
        inspected_modules=("trellis.models.black",),
        product_ir=None,
    )

    artifact = emit_analytical_trace_from_generation_plan(
        plan,
        trace_id="trace_black76",
        task_id="T97",
        issue_id="QUA-346",
        root=tmp_path,
    )

    assert artifact.json_path.exists()
    assert artifact.text_path.exists()

    loaded = load_analytical_traces(root=tmp_path)
    assert len(loaded) == 1

    trace = loaded[0]
    assert trace.trace_type == "analytical"
    assert trace.route.family == "analytical"
    assert trace.route.name == "analytical_black76"
    instruction_resolution = trace.context["generation_plan"]["instruction_resolution"]
    assert instruction_resolution["route"] == "analytical_black76"
    assert instruction_resolution["effective_instruction_count"] >= 1
    assert instruction_resolution["conflicts"] == []
    assert trace.steps[1].outputs["instruction_resolution"]["route"] == "analytical_black76"
    assert [step.kind for step in trace.steps] == [
        "trace",
        "semantic_resolution",
        "instruction_lifecycle",
        "decomposition",
        "assembly",
        "validation",
        "output",
    ]
    assert trace.steps[1].parent_id == trace.steps[0].id
    assert trace.steps[2].parent_id == trace.steps[0].id
    assert trace.steps[2].outputs["instruction_resolution"]["route"] == "analytical_black76"
    assert "black76_call" in artifact.text_path.read_text()
    assert "basis claims" in artifact.text_path.read_text()
    rendered = render_analytical_trace(trace)
    assert "Analytical Trace" in rendered
    assert "analytical_black76" in rendered
    assert "black76_call" in rendered


def test_route_health_snapshot_reports_instruction_counts():
    from trellis.agent.analytical_traces import build_analytical_trace_from_generation_plan, route_health_snapshot
    from trellis.agent.codegen_guardrails import build_generation_plan

    plan = build_generation_plan(
        pricing_plan=_black76_pricing_plan(),
        instrument_type="vanilla_option",
        inspected_modules=("trellis.models.black",),
        product_ir=None,
    )

    trace = build_analytical_trace_from_generation_plan(
        plan,
        trace_id="trace_black76",
        task_id="T97",
        issue_id="QUA-452",
    )
    snapshot = route_health_snapshot(trace)

    assert snapshot["route_id"] == "analytical_black76"
    assert snapshot["route_family"] == "analytical"
    assert snapshot["effective_instruction_count"] >= 1
    assert snapshot["hard_constraint_count"] == 0
    assert snapshot["conflict_count"] == 0
    assert "analytical_black76:schedule-builder" in snapshot["effective_instruction_ids"]
