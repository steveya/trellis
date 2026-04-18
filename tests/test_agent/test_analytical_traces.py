from __future__ import annotations

from types import SimpleNamespace


def _black76_pricing_plan():
    # QUA-909: the Black76 match clause now requires ``black_vol_surface``
    # in the pricing plan's required market data and restricts dispatch to
    # ``european`` / ``bermudan`` exercise styles. The trace fixtures below
    # must therefore declare both, matching what a real analytical pricing
    # plan would carry; otherwise Black76 is filtered out and the
    # generation plan lands on the empty-route fallback.
    return SimpleNamespace(
        method="analytical",
        method_modules=("trellis.models.black",),
        required_market_data=("discount_curve", "black_vol_surface"),
        model_to_build="vanilla_option",
        reasoning="trace black76 analytical assembly",
        modeling_requirements=(),
    )


def _black76_product_ir():
    # QUA-909: the Black76 match clause now filters on both ``payoff_family``
    # and ``exercise``. A bare ``product_ir=None`` argument leaves both at
    # their default values (empty string / "none"), so the positive filters
    # reject Black76 and the generation plan falls back to no route. Supply
    # a minimal European vanilla ``ProductIR`` so these trace fixtures
    # actually exercise the Black76 dispatch path they claim to test.
    from trellis.agent.knowledge.schema import ProductIR

    return ProductIR(
        instrument="european_option",
        payoff_family="vanilla_option",
        exercise_style="european",
        model_family="equity_diffusion",
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
        product_ir=_black76_product_ir(),
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
        product_ir=_black76_product_ir(),
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
    # QUA-909: the canonical European vanilla ``ProductIR`` now dispatches
    # through Black76's ``when: vanilla_option`` clause, which emits the
    # three call/put/digital guidance notes instead of the shared
    # ``build_payment_timeline`` schedule-builder primitive reserved for
    # the ``when: default`` fallback. At least one note must be present on
    # the resulting route-health snapshot.
    assert any(
        iid.startswith("analytical_black76:note:")
        for iid in snapshot["effective_instruction_ids"]
    )


def test_analytical_trace_route_from_dict_preserves_blank_route_name():
    from trellis.agent.analytical_traces import AnalyticalTraceRoute

    route = AnalyticalTraceRoute.from_dict(
        {"family": "analytical", "name": "", "model": "black_scholes"}
    )

    assert route.family == "analytical"
    assert route.name == ""
    assert route.model == "black_scholes"
