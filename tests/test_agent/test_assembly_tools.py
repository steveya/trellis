"""Tests for structured assembly-tool helpers."""

from __future__ import annotations

from types import SimpleNamespace

from trellis.agent.quant import PricingPlan


def test_lookup_primitive_route_for_european_analytical():
    from trellis.agent.assembly_tools import lookup_primitive_route

    result = lookup_primitive_route(
        description="European equity call option",
        instrument_type="european_option",
        preferred_method="analytical",
    )

    assert result.route == "analytical_black76"
    assert "trellis.models.black.black76_call" in result.primitives
    assert "map_spot_discount_and_vol_to_forward_black76" in result.adapters


def test_render_thin_adapter_plan_includes_market_reads_and_primitives():
    from trellis.agent.assembly_tools import render_thin_adapter_plan
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "black_vol_surface"},
        model_to_build="european_option",
        reasoning="test",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir("European equity call option", instrument_type="european_option"),
    )

    text = render_thin_adapter_plan(
        SimpleNamespace(class_name="EuropeanCallPayoff", fields=[]),
        pricing_plan=pricing_plan,
        generation_plan=generation_plan,
    )

    assert "Thin Adapter Plan" in text
    assert "market_state.discount" in text
    assert "market_state.vol_surface.black_vol" in text
    assert "trellis.models.black.black76_call" in text


def test_select_invariant_pack_for_callable_bond_includes_bounding():
    from trellis.agent.assembly_tools import select_invariant_pack

    pack = select_invariant_pack(
        instrument_type="callable_bond",
        method="rate_tree",
    )

    assert "check_non_negativity" in pack.checks
    assert "check_bounded_by_reference" in pack.checks


def test_build_comparison_harness_plan_resolves_targets_and_reference():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    plan = build_comparison_harness_plan(
        {
            "construct": ["lattice", "pde", "monte_carlo", "transforms"],
            "cross_validate": {
                "internal": ["crr_tree", "bs_pde", "mc_exact", "fft", "cos"],
                "analytical": "black_scholes",
                "tolerance_pct": 1.0,
            },
        }
    )

    assert [target.target_id for target in plan.targets] == [
        "crr_tree",
        "bs_pde",
        "mc_exact",
        "fft",
        "cos",
        "black_scholes",
    ]
    assert plan.reference_target == "black_scholes"
    assert plan.tolerance_pct == 1.0


def test_build_cookbook_candidate_payload_extracts_template():
    from trellis.agent.assembly_tools import build_cookbook_candidate_payload

    payload = build_cookbook_candidate_payload(
        method="analytical",
        description="European equity call option",
        code="""class Demo:\n    def evaluate(self, market_state):\n        return 1.0\n""",
    )

    assert payload is not None
    assert payload["method"] == "analytical"
    assert "def evaluate" in payload["template"]
