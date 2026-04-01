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
        required_market_data=required_data or {"discount_curve", "vol_surface"},
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
        assert cp.stages[0].decision == "rate_tree"


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
                        "required_market_data": ["discount_curve", "vol_surface"],
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
