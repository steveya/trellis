"""Tests for agent build planner."""

import pytest

from trellis.agent.planner import (
    BuildPlan, BuildStep, FieldDef, SpecSchema, STATIC_SPECS, _plan_static, plan_build,
)


class TestSpecSchema:

    def test_swaption_static_spec(self):
        spec = STATIC_SPECS["swaption"]
        assert spec.class_name == "SwaptionPayoff"
        assert spec.spec_name == "SwaptionSpec"
        field_names = [f.name for f in spec.fields]
        assert "notional" in field_names
        assert "strike" in field_names
        assert "expiry_date" in field_names
        assert "swap_start" in field_names
        assert "swap_end" in field_names

    def test_cap_static_spec(self):
        spec = STATIC_SPECS["cap"]
        assert spec.class_name == "AgentCapPayoff"

    def test_frozen(self):
        spec = STATIC_SPECS["swaption"]
        with pytest.raises(AttributeError):
            spec.class_name = "changed"


class TestPlanStatic:

    def test_swaption_plan_has_spec(self):
        plan = _plan_static(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
            {"discount", "forward_rate", "black_vol"},
            set(),
        )
        assert plan.payoff_class_name == "SwaptionPayoff"
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "SwaptionPayoff"
        assert len(plan.steps) == 1
        assert "swaption" in plan.steps[0].module_path

    def test_unknown_instrument_no_spec(self):
        plan = _plan_static(
            "variance swap on VIX",
            {"discount", "black_vol"},
            {"discount", "black_vol"},
            set(),
        )
        assert plan.spec_schema is None  # no static spec → needs LLM Step 1

    def test_generic_class_name(self):
        plan = _plan_static(
            "variance swap on VIX",
            {"discount"},
            {"discount"},
            set(),
        )
        assert plan.payoff_class_name == "VarianceSwapPayoff"

    def test_barrier_option_has_spec(self):
        plan = _plan_static(
            "exotic barrier option",
            {"discount", "black_vol"},
            {"discount", "black_vol"},
            set(),
        )
        assert plan.spec_schema is not None
        assert plan.spec_schema.class_name == "BarrierOptionPayoff"


class TestPlanBuild:

    def test_missing_capabilities_raises(self):
        with pytest.raises(NotImplementedError, match="missing capabilities"):
            plan_build("some exotic", {"discount", "simulated_paths"})

    def test_swaption_returns_plan_with_spec(self):
        plan = plan_build(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
        )
        assert isinstance(plan, BuildPlan)
        assert plan.spec_schema is not None
        assert plan.payoff_class_name == "SwaptionPayoff"

    def test_plan_gap_analysis(self):
        plan = plan_build(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
        )
        assert plan.satisfied == frozenset({"discount", "forward_rate", "black_vol"})
        assert plan.missing == frozenset()
