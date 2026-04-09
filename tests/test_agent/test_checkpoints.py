"""Tests for structured decision checkpoints (QUA-425).

All tests are pure unit tests — no LLM calls, no tokens spent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from trellis.agent.checkpoints import (
    DecisionCheckpoint,
    StageDecision,
    StageDivergence,
    capture_checkpoint,
    diff_checkpoints,
    format_checkpoint_summary,
    format_divergence_report,
    list_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    save_checkpoint,
    _hash_value,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable fake pipeline artifacts
# ---------------------------------------------------------------------------

def _fake_pricing_plan(
    method: str = "rate_tree",
    required_data: set | None = None,
    modules: tuple = ("trellis.models.trees.bdt",),
    reason: str = "static_plan",
):
    return SimpleNamespace(
        method=method,
        required_market_data=required_data or {"discount_curve", "black_vol_surface"},
        method_modules=modules,
        selection_reason=reason,
        reasoning="test reasoning",
        assumption_summary=["test assumption"],
        sensitivity_support=None,
    )


def _fake_spec_schema(
    spec_name: str = "CallableBondSpec",
    class_name: str = "CallableBondPayoff",
    fields: list | None = None,
):
    if fields is None:
        fields = [
            SimpleNamespace(name="face_value"),
            SimpleNamespace(name="coupon_rate"),
            SimpleNamespace(name="call_schedule"),
        ]
    return SimpleNamespace(
        spec_name=spec_name,
        class_name=class_name,
        fields=fields,
    )


SAMPLE_CODE = """\
from trellis.models.trees.bdt import BDTTree
from trellis.core.payoff import Payoff

class CallableBondPayoff(Payoff):
    def evaluate(self, market_state):
        tree = BDTTree()
        return tree.price()
"""


def _semantic_checkpoint(
    *,
    semantic_id: str = "vanilla_option",
    bridge_status: str = "thin_compatibility_wrapper",
    requested_instrument_type: str = "european_option",
):
    return {
        "semantic_id": semantic_id,
        "semantic_version": "c2.1",
        "requested_instrument_type": requested_instrument_type,
        "product_instrument_class": "european_option",
        "payoff_family": "vanilla_option",
        "underlier_structure": "single_underlier",
        "preferred_method": "analytical",
        "required_market_inputs": ["discount_curve", "underlier_spot", "black_vol_surface"],
        "compatibility_bridge_status": bridge_status,
        "matched_wrapper": requested_instrument_type if bridge_status == "thin_compatibility_wrapper" else "",
    }


def _generation_boundary(
    *,
    route_id: str | None = "analytical_black76",
    approved_modules: tuple[str, ...] = ("trellis.models.black",),
):
    exact_fit = bool(route_id)
    binding_ids = {
        "analytical_black76": "trellis.models.black.black76_call",
        "quanto_adjustment_analytical": "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state",
        "analytical_garman_kohlhagen": "trellis.models.fx_vanilla.price_fx_vanilla_analytical",
    }
    helper_refs = {
        "analytical_black76": ["trellis.models.black.black76_call"],
        "quanto_adjustment_analytical": ["trellis.models.quanto_option.price_quanto_option_analytical_from_market_state"],
        "analytical_garman_kohlhagen": ["trellis.models.fx_vanilla.price_fx_vanilla_analytical"],
    }
    binding_id = binding_ids.get(route_id, "trellis.models.black.black76_call")
    return {
        "method": "analytical",
        "approved_modules": list(approved_modules),
        "inspected_modules": ["trellis.models.black"],
        "symbols_to_reuse": ["black76_call"],
        "valuation_context": {
            "market_source": "unbound_market_snapshot",
            "reporting_policy": {
                "reporting_currency": "USD",
            },
        },
        "required_data_spec": {
            "required_input_ids": ["discount_curve", "underlier_spot", "black_vol_surface"],
        },
        "market_binding_spec": {
            "reporting_currency": "USD",
        },
        "construction_identity": {
            "primary_kind": "backend_binding" if exact_fit else "family_ir",
            "primary_label": (
                binding_id
                if exact_fit
                else "VanillaOptionIR"
            ),
            "lane_family": "analytical",
            "plan_kind": "exact_target_binding" if exact_fit else "compiler_boundary_only",
            "family_ir_type": "VanillaOptionIR",
            "backend_binding_id": (
                binding_id
                if exact_fit
                else None
            ),
            "backend_engine_family": "analytical" if exact_fit else None,
            "backend_exact_fit": exact_fit,
            "route_alias": route_id,
            "route_authority_kind": "exact_backend_fit" if exact_fit else None,
            "state_obligations": [],
            "control_obligations": [],
        },
        "lowering": {
            "route_id": route_id,
            "route_family": "analytical" if route_id else None,
            "primitive_routes": [route_id] if route_id else [],
            "route_modules": list(approved_modules),
            "expr_kind": "TerminalVanillaOptionExpr",
            "family_ir_type": "VanillaOptionIR",
            "helper_refs": [],
            "target_bindings": [],
            "lowering_errors": [],
        },
        "primitive_plan": (
            {
                "route": route_id,
                "engine_family": "analytical",
                "route_family": "analytical",
                "adapters": [],
                "blockers": [],
            }
            if route_id
            else {}
        ),
        "route_binding_authority": (
            {
                "route_id": route_id,
                "route_family": "analytical",
                "authority_kind": "exact_backend_fit",
                "backend_binding": {
                    "binding_id": binding_id,
                    "engine_family": "analytical",
                    "exact_backend_fit": True,
                    "helper_refs": helper_refs.get(route_id, ["trellis.models.black.black76_call"]),
                    "primitive_refs": [],
                    "exact_target_refs": [],
                    "approved_modules": list(approved_modules),
                    "admissibility": {
                        "supported_control_styles": ["identity"],
                        "event_support": "none",
                        "phase_sensitivity": "default_phase_order_only",
                        "multicurrency_support": "single_currency_only",
                        "supported_outputs": ["price"],
                        "supports_sensitivity_outputs": True,
                        "supported_state_tags": [],
                        "supported_process_families": [],
                        "supported_path_requirement_kinds": [],
                        "supported_operator_families": [],
                        "supported_event_transform_kinds": [],
                        "supports_calibration": False,
                    },
                    "admissibility_failures": [],
                },
                "validation_bundle_id": "analytical:vanilla_option",
                "canary_task_ids": ["T73"],
            }
            if route_id
            else {}
        ),
    }


def _validation_contract(
    *,
    route_id: str | None = "analytical_black76",
    bundle_id: str = "analytical:vanilla_option",
):
    return {
        "contract_id": "analytical:vanilla_option",
        "bundle_id": bundle_id,
        "route_id": route_id,
        "route_family": "analytical",
        "required_market_data": ["discount_curve", "underlier_spot", "black_vol_surface"],
        "deterministic_checks": [
            {
                "check_id": "closed_form_regression",
                "category": "pricing",
            },
        ],
        "comparison_relations": [],
        "lowering_errors": [],
        "admissibility_failures": [],
        "residual_risks": [],
    }


def _make_checkpoint(**overrides) -> DecisionCheckpoint:
    defaults = dict(
        task_id="T38",
        instrument_type="callable_bond",
        timestamp="2026-03-30T12:00:00+00:00",
        stages=(
            StageDecision(agent="quant", decision="rate_tree", metadata={"method_modules": ["bdt"]}),
            StageDecision(agent="planner", decision="CallableBondSpec", metadata={"field_count": 3}),
            StageDecision(agent="builder", decision="compiled", metadata={"code_lines": 7}, output_hash="abc123"),
            StageDecision(agent="validator", decision="pass", metadata={"final_price": 98.234, "tolerance": 0.5}),
        ),
        outcome="pass",
        total_tokens=5000,
        final_price=98.234,
        tolerance=0.5,
        attempts=1,
        provider="openai",
        model="gpt-5-mini",
    )
    defaults.update(overrides)
    return DecisionCheckpoint(**defaults)


# ---------------------------------------------------------------------------
# StageDecision
# ---------------------------------------------------------------------------

class TestStageDecision:
    def test_frozen(self):
        s = StageDecision(agent="quant", decision="rate_tree")
        with pytest.raises(AttributeError):
            s.agent = "planner"

    def test_defaults(self):
        s = StageDecision(agent="quant", decision="rate_tree")
        assert s.metadata == {}
        assert s.tokens_used == 0
        assert s.input_hash == ""


# ---------------------------------------------------------------------------
# DecisionCheckpoint
# ---------------------------------------------------------------------------

class TestDecisionCheckpoint:
    def test_frozen(self):
        cp = _make_checkpoint()
        with pytest.raises(AttributeError):
            cp.task_id = "T99"

    def test_stage_access(self):
        cp = _make_checkpoint()
        assert len(cp.stages) == 4
        assert cp.stages[0].agent == "quant"


def test_capture_checkpoint_threads_route_binding_authority_into_route_stage():
    checkpoint = capture_checkpoint(
        task_id="T73",
        instrument_type="swaption",
        pricing_plan=_fake_pricing_plan(method="analytical", modules=("trellis.models.black",)),
        generation_boundary=_generation_boundary(route_id="analytical_black76"),
        validation_contract=_validation_contract(route_id="analytical_black76", bundle_id="analytical:swaption"),
        outcome="pass",
    )

    route_stage = next(stage for stage in checkpoint.stages if stage.agent == "route")

    assert route_stage.decision == "trellis.models.black.black76_call"
    assert route_stage.metadata["route_binding_authority"]["authority_kind"] == "exact_backend_fit"
    assert route_stage.metadata["route_binding_authority"]["canary_task_ids"] == ["T73"]
    assert route_stage.metadata["construction_identity"]["route_alias"] == "analytical_black76"


def test_capture_checkpoint_keeps_route_stage_unknown_when_lowering_has_no_route():
    checkpoint = capture_checkpoint(
        task_id="T301",
        instrument_type="range_accrual",
        pricing_plan=_fake_pricing_plan(
            method="analytical",
            modules=("trellis.models.range_accrual", "trellis.models.contingent_cashflows"),
        ),
        semantic_checkpoint=_semantic_checkpoint(
            semantic_id="range_accrual",
            bridge_status="canonical_semantic",
            requested_instrument_type="range_accrual",
        ),
        generation_boundary=_generation_boundary(
            route_id=None,
            approved_modules=(
                "trellis.models.range_accrual",
                "trellis.models.contingent_cashflows",
            ),
        ),
        validation_contract=_validation_contract(
            route_id=None,
            bundle_id="analytical:range_accrual",
        ),
        outcome="fail_build",
    )

    route_stage = next(stage for stage in checkpoint.stages if stage.agent == "route")

    assert route_stage.decision == "VanillaOptionIR"
    assert route_stage.metadata["method"] == "analytical"
    assert route_stage.metadata["lowering"]["route_id"] is None
    assert "route_binding_authority" not in route_stage.metadata


# ---------------------------------------------------------------------------
# capture_checkpoint
# ---------------------------------------------------------------------------

class TestCaptureCheckpoint:
    def test_from_pipeline_artifacts(self):
        cp = capture_checkpoint(
            task_id="T38",
            instrument_type="callable_bond",
            pricing_plan=_fake_pricing_plan(),
            spec_schema=_fake_spec_schema(),
            code=SAMPLE_CODE,
            outcome="pass",
            final_price=98.234,
            tolerance=0.5,
            attempts=2,
            provider="openai",
            model="gpt-5-mini",
        )
        assert cp.task_id == "T38"
        assert cp.outcome == "pass"
        assert cp.final_price == 98.234
        assert cp.attempts == 2

        # Should have 4 stages: quant, planner, builder, validator
        agents = [s.agent for s in cp.stages]
        assert agents == ["quant", "planner", "builder", "validator"]

        # Quant stage
        quant = cp.stages[0]
        assert quant.decision == "rate_tree"
        assert "discount_curve" in quant.metadata["required_market_data"]

        # Planner stage
        planner = cp.stages[1]
        assert planner.decision == "CallableBondSpec"
        assert planner.metadata["field_count"] == 3

        # Builder stage
        builder = cp.stages[2]
        assert builder.decision == "compiled"
        assert builder.metadata["code_lines"] == 7
        assert builder.metadata["import_count"] == 2

        # Validator stage
        validator = cp.stages[3]
        assert validator.decision == "pass"
        assert validator.metadata["final_price"] == 98.234

    def test_minimal_capture(self):
        """Capture with only required fields — no stages produced for missing artifacts."""
        cp = capture_checkpoint(
            task_id="T99",
            instrument_type="unknown",
            outcome="fail_build",
        )
        assert cp.task_id == "T99"
        assert cp.stages == ()
        assert cp.outcome == "fail_build"

    def test_capture_with_build_meta_code(self):
        """Builder stage extracted from build_meta['code'] when code param is None."""
        meta = {"code": SAMPLE_CODE}
        cp = capture_checkpoint(
            task_id="T38",
            instrument_type="callable_bond",
            build_meta=meta,
            outcome="pass",
        )
        builder_stages = [s for s in cp.stages if s.agent == "builder"]
        assert len(builder_stages) == 1
        assert builder_stages[0].metadata["code_lines"] == 7

    def test_capture_with_token_summary(self):
        cp = capture_checkpoint(
            task_id="T38",
            instrument_type="callable_bond",
            token_summary={"total_tokens": 8421},
            outcome="pass",
        )
        assert cp.total_tokens == 8421

    def test_capture_with_failures_in_meta(self):
        meta = {"failures": ["import error: foo not found", "price out of range"]}
        cp = capture_checkpoint(
            task_id="T38",
            instrument_type="callable_bond",
            build_meta=meta,
            outcome="fail_validate",
            final_price=50.0,
        )
        validator = [s for s in cp.stages if s.agent == "validator"][0]
        assert validator.decision == "fail"
        assert validator.metadata["failure_count"] == 2

    def test_capture_semantic_route_boundary(self):
        cp = capture_checkpoint(
            task_id="T74",
            instrument_type="european_option",
            pricing_plan=_fake_pricing_plan(method="analytical", modules=("trellis.models.black",)),
            code=SAMPLE_CODE,
            semantic_checkpoint=_semantic_checkpoint(),
            generation_boundary=_generation_boundary(),
            validation_contract=_validation_contract(),
            outcome="pass",
            final_price=12.34,
            tolerance=0.5,
        )

        agents = [stage.agent for stage in cp.stages]
        assert agents == ["quant", "semantic", "route", "builder", "validator"]

        semantic = [stage for stage in cp.stages if stage.agent == "semantic"][0]
        assert semantic.decision == "vanilla_option"
        assert semantic.metadata["compatibility_bridge_status"] == "thin_compatibility_wrapper"

        route = [stage for stage in cp.stages if stage.agent == "route"][0]
        assert route.decision == "trellis.models.black.black76_call"
        assert route.metadata["valuation_context"]["market_source"] == "unbound_market_snapshot"

        builder = [stage for stage in cp.stages if stage.agent == "builder"][0]
        assert "trellis.models.black" in builder.metadata["approved_modules"]

        validator = [stage for stage in cp.stages if stage.agent == "validator"][0]
        assert validator.decision == "pass"
        assert validator.metadata["bundle_id"] == "analytical:vanilla_option"


# ---------------------------------------------------------------------------
# diff_checkpoints
# ---------------------------------------------------------------------------

class TestDiffCheckpoints:
    def test_identical_checkpoints(self):
        a = _make_checkpoint()
        b = _make_checkpoint()
        assert diff_checkpoints(a, b) == []

    def test_decision_divergence(self):
        a = _make_checkpoint()
        b = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="monte_carlo"),  # changed!
            StageDecision(agent="planner", decision="CallableBondSpec"),
            StageDecision(agent="builder", decision="compiled", output_hash="abc123"),
            StageDecision(agent="validator", decision="pass", metadata={"final_price": 98.234, "tolerance": 0.5}),
        ))
        divs = diff_checkpoints(a, b)
        decision_divs = [d for d in divs if d.severity == "decision"]
        assert len(decision_divs) == 1
        assert decision_divs[0].agent == "quant"
        assert decision_divs[0].old_decision == "rate_tree"
        assert decision_divs[0].new_decision == "monte_carlo"

    def test_metadata_divergence(self):
        a = _make_checkpoint()
        b = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="rate_tree", metadata={"method_modules": ["bdt"]}),
            StageDecision(agent="planner", decision="CallableBondSpec", metadata={"field_count": 3}),
            StageDecision(agent="builder", decision="compiled", metadata={"code_lines": 52}, output_hash="different"),
            StageDecision(agent="validator", decision="pass", metadata={"final_price": 98.234, "tolerance": 0.5}),
        ))
        divs = diff_checkpoints(a, b)
        meta_divs = [d for d in divs if d.severity == "metadata"]
        assert len(meta_divs) == 1
        assert meta_divs[0].agent == "builder"

    def test_price_drift_detection(self):
        a = _make_checkpoint()
        b = _make_checkpoint(
            final_price=97.9,
            stages=(
                StageDecision(agent="quant", decision="rate_tree", metadata={"method_modules": ["bdt"]}),
                StageDecision(agent="planner", decision="CallableBondSpec", metadata={"field_count": 3}),
                StageDecision(agent="builder", decision="compiled", metadata={"code_lines": 7}, output_hash="abc123"),
                StageDecision(agent="validator", decision="pass", metadata={"final_price": 97.9, "tolerance": 0.5}),
            ),
        )
        divs = diff_checkpoints(a, b)
        price_divs = [d for d in divs if d.severity == "price"]
        assert len(price_divs) == 1
        assert "drift_ratio" in price_divs[0].old_metadata
        # drift = |98.234 - 97.9| / 0.5 = 0.668 > 0.5
        assert price_divs[0].old_metadata["drift_ratio"] > 0.5

    def test_no_price_drift_within_tolerance(self):
        a = _make_checkpoint()
        b = _make_checkpoint(
            final_price=98.1,
            stages=(
                StageDecision(agent="quant", decision="rate_tree", metadata={"method_modules": ["bdt"]}),
                StageDecision(agent="planner", decision="CallableBondSpec", metadata={"field_count": 3}),
                StageDecision(agent="builder", decision="compiled", metadata={"code_lines": 7}, output_hash="abc123"),
                StageDecision(agent="validator", decision="pass", metadata={"final_price": 98.1, "tolerance": 0.5}),
            ),
        )
        divs = diff_checkpoints(a, b)
        price_divs = [d for d in divs if d.severity == "price"]
        assert len(price_divs) == 0  # drift ratio = 0.268 < 0.5

    def test_stage_added(self):
        a = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="rate_tree"),
        ))
        b = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="rate_tree"),
            StageDecision(agent="planner", decision="CallableBondSpec"),
        ))
        divs = diff_checkpoints(a, b)
        assert any(d.agent == "planner" and d.old_decision == "(absent)" for d in divs)

    def test_stage_removed(self):
        a = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="rate_tree"),
            StageDecision(agent="planner", decision="CallableBondSpec"),
        ))
        b = _make_checkpoint(stages=(
            StageDecision(agent="quant", decision="rate_tree"),
        ))
        divs = diff_checkpoints(a, b)
        assert any(d.agent == "planner" and d.new_decision == "(absent)" for d in divs)

    def test_semantic_wrapper_drift_is_metadata_divergence(self):
        a = capture_checkpoint(
            task_id="T74",
            instrument_type="european_option",
            semantic_checkpoint=_semantic_checkpoint(bridge_status="thin_compatibility_wrapper"),
            generation_boundary=_generation_boundary(),
            validation_contract=_validation_contract(),
            outcome="pass",
        )
        b = capture_checkpoint(
            task_id="T74",
            instrument_type="vanilla_option",
            semantic_checkpoint=_semantic_checkpoint(
                bridge_status="canonical_semantic",
                requested_instrument_type="vanilla_option",
            ),
            generation_boundary=_generation_boundary(),
            validation_contract=_validation_contract(),
            outcome="pass",
        )

        divs = diff_checkpoints(a, b)
        assert any(d.agent == "semantic" and d.severity == "metadata" for d in divs)

    def test_route_and_approved_module_drift_are_detected(self):
        baseline = capture_checkpoint(
            task_id="T105",
            instrument_type="quanto_option",
            semantic_checkpoint=_semantic_checkpoint(
                semantic_id="quanto_option",
                bridge_status="canonical_semantic",
                requested_instrument_type="quanto_option",
            ),
            generation_boundary=_generation_boundary(
                route_id="quanto_adjustment_analytical",
                approved_modules=(
                    "trellis.models.black",
                    "trellis.models.resolution.quanto",
                ),
            ),
            validation_contract=_validation_contract(
                route_id="quanto_adjustment_analytical",
                bundle_id="analytical:quanto_option",
            ),
            outcome="pass",
        )
        changed = capture_checkpoint(
            task_id="T105",
            instrument_type="quanto_option",
            semantic_checkpoint=_semantic_checkpoint(
                semantic_id="quanto_option",
                bridge_status="canonical_semantic",
                requested_instrument_type="quanto_option",
            ),
            generation_boundary=_generation_boundary(
                route_id="analytical_garman_kohlhagen",
                approved_modules=(
                    "trellis.models.black",
                    "trellis.models.resolution.quanto",
                    "trellis.models.analytical.quanto",
                ),
            ),
            validation_contract=_validation_contract(
                route_id="analytical_garman_kohlhagen",
                bundle_id="analytical:quanto_option",
            ),
            outcome="pass",
        )

        divs = diff_checkpoints(baseline, changed)
        assert any(d.agent == "route" and d.severity == "decision" for d in divs)
        assert any(d.agent == "builder" and d.severity == "metadata" for d in divs)


# ---------------------------------------------------------------------------
# save / load / retention
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        cp = _make_checkpoint()
        path = save_checkpoint(cp, directory=tmp_path)
        assert path.exists()
        assert path.suffix == ".yaml"

        loaded = load_checkpoint(path)
        assert loaded.task_id == cp.task_id
        assert loaded.instrument_type == cp.instrument_type
        assert loaded.outcome == cp.outcome
        assert loaded.total_tokens == cp.total_tokens
        assert loaded.final_price == cp.final_price
        assert len(loaded.stages) == len(cp.stages)
        for orig, loaded_s in zip(cp.stages, loaded.stages):
            assert orig.agent == loaded_s.agent
            assert orig.decision == loaded_s.decision

    def test_save_creates_directory(self, tmp_path):
        cp = _make_checkpoint()
        nested = tmp_path / "deep" / "nested"
        path = save_checkpoint(cp, directory=nested)
        assert path.exists()

    def test_load_latest_checkpoint(self, tmp_path):
        cp1 = _make_checkpoint(timestamp="2026-03-30T10:00:00+00:00")
        cp2 = _make_checkpoint(timestamp="2026-03-30T12:00:00+00:00")
        save_checkpoint(cp1, directory=tmp_path)
        save_checkpoint(cp2, directory=tmp_path)

        latest = load_latest_checkpoint("T38", directory=tmp_path)
        assert latest is not None
        assert latest.timestamp == cp2.timestamp

    def test_load_latest_returns_none_for_missing(self, tmp_path):
        assert load_latest_checkpoint("T99", directory=tmp_path) is None

    def test_retention_policy(self, tmp_path):
        for i in range(15):
            cp = _make_checkpoint(timestamp=f"2026-03-{i+1:02d}T12:00:00+00:00")
            save_checkpoint(cp, directory=tmp_path, retention=5)

        files = list_checkpoints("T38", directory=tmp_path)
        assert len(files) == 5  # only last 5 kept

    def test_list_checkpoints_sorted_newest_first(self, tmp_path):
        for i in [1, 3, 2]:
            cp = _make_checkpoint(timestamp=f"2026-03-0{i}T12:00:00+00:00")
            save_checkpoint(cp, directory=tmp_path, retention=10)

        files = list_checkpoints("T38", directory=tmp_path)
        assert len(files) == 3
        # Newest first
        assert "03" in files[0].name
        assert "01" in files[2].name

    def test_yaml_structure(self, tmp_path):
        cp = _make_checkpoint()
        path = save_checkpoint(cp, directory=tmp_path)
        data = yaml.safe_load(path.read_text())

        assert data["task_id"] == "T38"
        assert data["outcome"] == "pass"
        assert len(data["stages"]) == 4
        assert data["stages"][0]["agent"] == "quant"
        assert data["stages"][0]["decision"] == "rate_tree"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_checkpoint_summary(self):
        cp = _make_checkpoint()
        summary = format_checkpoint_summary(cp)
        assert "T38" in summary
        assert "callable_bond" in summary
        assert "rate_tree" in summary
        assert "pass" in summary
        assert "98.234" in summary

    def test_divergence_report_empty(self):
        report = format_divergence_report([])
        assert "No divergences" in report

    def test_divergence_report_with_items(self):
        divs = [
            StageDivergence(agent="quant", old_decision="rate_tree", new_decision="monte_carlo", severity="decision"),
            StageDivergence(agent="builder", old_decision="compiled", new_decision="compiled", severity="metadata"),
        ]
        report = format_divergence_report(divs)
        assert "quant" in report
        assert "rate_tree" in report
        assert "monte_carlo" in report
        assert "builder" in report


# ---------------------------------------------------------------------------
# _hash_value
# ---------------------------------------------------------------------------

class TestHashValue:
    def test_deterministic(self):
        assert _hash_value("hello") == _hash_value("hello")

    def test_different_inputs(self):
        assert _hash_value("hello") != _hash_value("world")

    def test_truncated_to_16(self):
        h = _hash_value("test")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# _emit_decision_checkpoint integration
# ---------------------------------------------------------------------------

class TestEmitDecisionCheckpoint:
    def test_emit_from_build_result_success(self, tmp_path, monkeypatch):
        """Emit a checkpoint from a successful BuildResult."""
        from trellis.agent.knowledge.autonomous import BuildResult, _emit_decision_checkpoint
        import trellis.agent.checkpoints as cp_mod

        # Redirect checkpoint dir to tmp_path
        monkeypatch.setattr(cp_mod, "CHECKPOINT_DIR", tmp_path)

        result = BuildResult(
            payoff_cls=object,
            success=True,
            attempts=2,
            code=SAMPLE_CODE,
            agent_observations=[
                {
                    "agent": "quant",
                    "kind": "decision",
                    "message": "Selected pricing method `rate_tree`",
                    "details": {
                        "method": "rate_tree",
                        "required_market_data": ["discount_curve", "black_vol_surface"],
                        "method_modules": ["trellis.models.trees.bdt"],
                        "selection_reason": "static_plan",
                    },
                },
            ],
            token_usage_summary={"total_tokens": 5000, "call_count": 7},
            platform_request_id="req_abc123def456",
        )
        decomposition = SimpleNamespace(method="rate_tree")

        _emit_decision_checkpoint(
            result=result,
            decomposition=decomposition,
            instrument_type="callable_bond",
            model="gpt-5-mini",
        )

        files = list(tmp_path.glob("*.yaml"))
        assert len(files) == 1

        loaded = load_checkpoint(files[0])
        assert loaded.outcome == "pass"
        assert loaded.attempts == 2
        assert loaded.total_tokens == 5000

        quant = [s for s in loaded.stages if s.agent == "quant"]
        assert len(quant) == 1
        assert quant[0].decision == "rate_tree"

        builder = [s for s in loaded.stages if s.agent == "builder"]
        assert len(builder) == 1
        assert builder[0].decision == "compiled"

    def test_emit_from_build_result_failure(self, tmp_path, monkeypatch):
        """Emit a checkpoint from a failed BuildResult — must not raise."""
        from trellis.agent.knowledge.autonomous import BuildResult, _emit_decision_checkpoint
        import trellis.agent.checkpoints as cp_mod

        monkeypatch.setattr(cp_mod, "CHECKPOINT_DIR", tmp_path)

        result = BuildResult(
            payoff_cls=None,
            success=False,
            attempts=3,
            failures=["validation failed: price out of range"],
            code="",
            agent_observations=[],
            token_usage_summary={"total_tokens": 3000},
        )
        decomposition = SimpleNamespace(method="monte_carlo")

        # Should not raise
        _emit_decision_checkpoint(
            result=result,
            decomposition=decomposition,
            instrument_type="barrier_option",
            model="gpt-5-mini",
        )

        files = list(tmp_path.glob("*.yaml"))
        assert len(files) == 1
        loaded = load_checkpoint(files[0])
        assert loaded.outcome == "fail_validate"

    def test_emit_never_raises(self, monkeypatch):
        """Even with broken input, _emit_decision_checkpoint must not raise."""
        from trellis.agent.knowledge.autonomous import BuildResult, _emit_decision_checkpoint
        import trellis.agent.checkpoints as cp_mod

        # Point to a path that will fail (e.g., read-only)
        monkeypatch.setattr(cp_mod, "CHECKPOINT_DIR", Path("/nonexistent/readonly/dir"))

        result = BuildResult(payoff_cls=None, success=False, attempts=0)
        decomposition = SimpleNamespace(method="unknown")

        # Must not raise — best-effort
        _emit_decision_checkpoint(
            result=result,
            decomposition=decomposition,
            instrument_type=None,
            model="",
        )

    def test_emit_uses_platform_trace_boundary(self, tmp_path, monkeypatch):
        """Checkpoint emission should pull semantic/route/validation boundary from the trace."""
        from trellis.agent.knowledge.autonomous import BuildResult, _emit_decision_checkpoint
        from trellis.agent.platform_requests import compile_build_request
        from trellis.agent.platform_traces import record_platform_trace
        import trellis.agent.checkpoints as cp_mod

        checkpoint_dir = tmp_path / "checkpoints"
        monkeypatch.setattr(cp_mod, "CHECKPOINT_DIR", checkpoint_dir)

        compiled = compile_build_request(
            "Himalaya-style ranked observation basket on AAPL, MSFT, NVDA with observation dates "
            "2025-01-15, 2025-02-15, 2025-03-15. At each observation choose the best performer "
            "among the remaining constituents, remove it, lock the simple return, and settle the "
            "average locked returns at maturity.",
            instrument_type="basket_option",
        )
        trace_path = record_platform_trace(
            compiled,
            success=True,
            outcome="build_completed",
            root=tmp_path / "platform",
        )
        result = BuildResult(
            payoff_cls=object,
            success=True,
            attempts=1,
            code=SAMPLE_CODE,
            agent_observations=[
                {
                    "agent": "quant",
                    "kind": "decision",
                    "details": {
                        "method": compiled.pricing_plan.method,
                        "required_market_data": list(compiled.pricing_plan.required_market_data),
                        "method_modules": list(compiled.pricing_plan.method_modules),
                        "selection_reason": compiled.pricing_plan.selection_reason,
                    },
                },
            ],
            token_usage_summary={"total_tokens": 1234},
            platform_trace_path=str(trace_path),
            platform_request_id=compiled.request.request_id,
        )

        _emit_decision_checkpoint(
            result=result,
            decomposition=SimpleNamespace(method=compiled.pricing_plan.method),
            instrument_type="basket_option",
            model="gpt-5-mini",
        )

        files = list(checkpoint_dir.glob("*.yaml"))
        assert len(files) == 1
        loaded = load_checkpoint(files[0])
        semantic = [stage for stage in loaded.stages if stage.agent == "semantic"][0]
        route = [stage for stage in loaded.stages if stage.agent == "route"][0]
        builder = [stage for stage in loaded.stages if stage.agent == "builder"][0]
        validator = [stage for stage in loaded.stages if stage.agent == "validator"][0]

        assert semantic.decision == "ranked_observation_basket"
        assert semantic.metadata["compatibility_bridge_status"] == "thin_compatibility_wrapper"
        assert route.decision == "trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo"
        assert "trellis.models.monte_carlo.semantic_basket" in builder.metadata["approved_modules"]
        assert validator.metadata["route_id"] == "correlated_basket_monte_carlo"
        assert validator.metadata["bundle_id"] == "monte_carlo:basket_option"
