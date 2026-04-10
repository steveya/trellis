"""Tests for the canary task runner (QUA-424).

All tests use mocked task execution — no LLM calls, no tokens spent.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pytest
import yaml

# Import paths relative to the repo — the runner is a script, so we import
# its functions after inserting sys.path.
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_canary import (
    CORE_FAMILIES,
    check_drift_after_run,
    display_dry_run,
    filter_canaries,
    load_canary_set,
    promote_golden_after_run,
    run_canaries,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CANARY_YAML = """\
version: 1
total_budget_usd: 2.50
refresh_cadence: weekly

canary_set:
  - id: T01
    engine_family: lattice
    complexity: simple
    estimated_cost_usd: 0.12
    rationale: "ZCB option — simplest lattice task."
    covers:
      - rate_tree
      - cross_validation

  - id: T38
    engine_family: credit
    complexity: complex
    estimated_cost_usd: 0.15
    rationale: "CDS pricing — historically fragile."
    covers:
      - credit_default_swap
      - hazard_rate

  - id: T25
    engine_family: monte_carlo
    complexity: simple
    estimated_cost_usd: 0.10
    rationale: "GBM call convergence."
    covers:
      - gbm
      - euler_scheme

  - id: T13
    engine_family: pde
    complexity: simple
    estimated_cost_usd: 0.12
    rationale: "European call theta-method convergence."
    covers:
      - theta_method
      - crank_nicolson

  - id: T39
    engine_family: transforms
    complexity: simple
    estimated_cost_usd: 0.12
    rationale: "FFT vs COS baseline."
    covers:
      - fft
      - cos_method
"""


@pytest.fixture
def canary_file(tmp_path):
    path = tmp_path / "CANARY_TASKS.yaml"
    path.write_text(SAMPLE_CANARY_YAML)
    return path


# ---------------------------------------------------------------------------
# load_canary_set
# ---------------------------------------------------------------------------

class TestLoadCanarySet:
    def test_load_basic(self, canary_file):
        canaries, meta = load_canary_set(canary_file)
        assert len(canaries) == 5
        assert meta["version"] == 1
        assert meta["total_budget_usd"] == 2.50
        assert meta["refresh_cadence"] == "weekly"

    def test_canary_fields(self, canary_file):
        canaries, _ = load_canary_set(canary_file)
        t01 = canaries[0]
        assert t01["id"] == "T01"
        assert t01["engine_family"] == "lattice"
        assert t01["complexity"] == "simple"
        assert t01["estimated_cost_usd"] == 0.12
        assert "rate_tree" in t01["covers"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_canary_set(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# filter_canaries
# ---------------------------------------------------------------------------

class TestFilterCanaries:
    def test_no_filter(self, canary_file):
        canaries, _ = load_canary_set(canary_file)
        filtered = filter_canaries(canaries)
        assert len(filtered) == 5

    def test_filter_by_task_id(self, canary_file):
        canaries, _ = load_canary_set(canary_file)
        filtered = filter_canaries(canaries, task_id="T38")
        assert len(filtered) == 1
        assert filtered[0]["id"] == "T38"

    def test_filter_by_nonexistent_task(self, canary_file):
        canaries, _ = load_canary_set(canary_file)
        with pytest.raises(ValueError, match="not in the canary set"):
            filter_canaries(canaries, task_id="T999")

    def test_filter_core_subset(self, canary_file):
        canaries, _ = load_canary_set(canary_file)
        filtered = filter_canaries(canaries, subset="core")
        families = {c["engine_family"] for c in filtered}
        # Should include lattice, monte_carlo, pde, credit — but not transforms
        assert families <= CORE_FAMILIES
        assert "transforms" not in families
        assert len(filtered) == 4  # T01(lattice), T38(credit), T25(mc), T13(pde)


# ---------------------------------------------------------------------------
# display_dry_run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_output(self, canary_file, capsys):
        canaries, meta = load_canary_set(canary_file)
        display_dry_run(canaries, meta)
        output = capsys.readouterr().out
        assert "CANARY DRY RUN" in output
        assert "5 tasks" in output
        assert "T01" in output
        assert "T38" in output
        assert "lattice" in output
        assert "credit" in output
        assert "$2.50" in output  # budget

    def test_dry_run_shows_estimated_cost(self, canary_file, capsys):
        canaries, meta = load_canary_set(canary_file)
        display_dry_run(canaries, meta)
        output = capsys.readouterr().out
        assert "Total estimated cost" in output


# ---------------------------------------------------------------------------
# run_canaries
# ---------------------------------------------------------------------------

class TestRunCanaries:
    def test_run_canaries_loads_full_task_registry(self, monkeypatch):
        seen: dict[str, object] = {}

        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            seen["status"] = status
            return [{"id": "T01", "status": "done"}]

        def fake_run_task(task, market_state, **kwargs):
            seen["task_id"] = task["id"]
            seen["max_retries"] = kwargs.get("max_retries")
            return {"task_id": task["id"], "success": True}

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )

        results = run_canaries(
            [{"id": "T01", "engine_family": "lattice", "complexity": "simple"}],
            {"total_budget_usd": 1.0},
        )

        assert seen["status"] is None
        assert seen["task_id"] == "T01"
        assert seen["max_retries"] == 3
        assert results[0]["success"] is True

    def test_run_canaries_executes_done_task_entries(self, monkeypatch):
        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            if status is None:
                return [{"id": "T02", "status": "done"}]
            return []

        def fake_run_task(task, market_state, **kwargs):
            return {
                "task_id": task["id"],
                "success": True,
                "token_usage_summary": {"total_tokens": 123},
            }

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )

        results = run_canaries(
            [{"id": "T02", "engine_family": "lattice", "complexity": "simple"}],
            {"total_budget_usd": 1.0},
        )

        assert results[0]["task_id"] == "T02"
        assert results[0]["success"] is True
        assert results[0]["token_usage_summary"] == {"total_tokens": 123}
        assert results[0]["canary_id"] == "T02"
        assert results[0]["engine_family"] == "lattice"
        assert results[0]["complexity"] == "simple"
        assert results[0]["canary_batch_id"].startswith("canary_")
        assert results[0]["canary_batch_history_path"].endswith(".json")
        assert results[0]["canary_batch_latest_path"].endswith(
            "/task_runs/canary_batches/latest/live__full_curated__standard__default.json"
        )

    def test_run_canaries_prefers_curated_canary_description(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            return [
                {
                    "id": "T25",
                    "title": "GBM call: all 4 schemes convergence order",
                    "construct": "monte_carlo",
                    "cross_validate": {
                        "internal": ["euler", "milstein", "exact", "log_euler"],
                        "analytical": "black_scholes",
                    },
                }
            ]

        def fake_run_task(task, market_state, **kwargs):
            captured["description"] = task.get("description")
            captured["construct"] = task.get("construct")
            return {"task_id": task["id"], "success": True}

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )

        run_canaries(
            [
                {
                    "id": "T25",
                    "engine_family": "monte_carlo",
                    "complexity": "simple",
                    "description": "European call option on a non-dividend-paying stock.",
                }
            ],
            {"total_budget_usd": 1.0},
        )

        assert captured["description"] == "European call option on a non-dividend-paying stock."
        assert captured["construct"] == "monte_carlo"

    def test_run_canaries_replay_mode_skips_missing_cassettes(self, monkeypatch, tmp_path):
        seen = {"run_task_calls": 0}

        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            return [{"id": "T13", "title": "European call: theta-method convergence order"}]

        def fake_run_task(task, market_state, **kwargs):
            seen["run_task_calls"] += 1
            return {"task_id": task["id"], "success": True}

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )

        results = run_canaries(
            [{"id": "T13", "engine_family": "pde", "complexity": "simple"}],
            {"total_budget_usd": 1.0},
            replay=True,
            cassette_dir=tmp_path,
        )

        assert seen["run_task_calls"] == 0
        assert results[0]["canary_id"] == "T13"
        assert results[0]["engine_family"] == "pde"
        assert results[0]["success"] is False
        assert results[0]["skipped"] is True
        assert results[0]["reason"] == "missing_cassette"
        assert results[0]["error"] == (
            "Missing cassette for T13 at "
            f"{tmp_path / 'T13.yaml'}. "
            "Record it with scripts/record_cassettes.py --task T13"
        )
        assert results[0]["canary_batch_id"].startswith("canary_")
        assert results[0]["canary_batch_latest_path"].endswith(
            "/task_runs/canary_batches/latest/cassette_replay__full_curated__standard__default.json"
        )

    def test_run_canaries_replay_mode_uses_full_task_cassette(self, monkeypatch, tmp_path):
        from trellis.agent.cassette import _prompt_hash

        prompt = "runner replay prompt"
        cassette_path = tmp_path / "T38.yaml"
        cassette_path.write_text(
            yaml.safe_dump(
                {
                    "meta": {
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "total_calls": 1,
                    },
                    "calls": [
                        {
                            "seq": 0,
                            "function": "llm_generate",
                            "stage": "critic",
                            "prompt_hash": _prompt_hash(prompt),
                            "response_text": "runner replay response",
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        seen: dict[str, object] = {}

        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            return [{"id": "T38", "title": "CDS pricing canary"}]

        def fake_run_task(task, market_state, **kwargs):
            from trellis.agent import critic

            seen["llm_text"] = critic.llm_generate(prompt)
            return {
                "task_id": task["id"],
                "success": True,
                "execution_mode": "cassette_replay",
                "token_usage_summary": {"total_tokens": 0},
            }

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )

        results = run_canaries(
            [{"id": "T38", "engine_family": "credit", "complexity": "complex"}],
            {"total_budget_usd": 1.0},
            replay=True,
            cassette_dir=tmp_path,
        )

        assert seen["llm_text"] == "runner replay response"
        assert results[0]["success"] is True
        assert results[0]["execution_mode"] == "cassette_replay"
        assert results[0]["token_usage_summary"]["total_tokens"] == 0

    def test_run_canaries_persists_canary_batch_telemetry(self, monkeypatch, tmp_path):
        seen: dict[str, object] = {}

        def fake_build_market_state():
            return object()

        def fake_load_tasks(*, status="pending", path=None):
            return [{"id": "T13", "title": "European call: theta-method convergence order"}]

        def fake_run_task(task, market_state, **kwargs):
            return {
                "task_id": task["id"],
                "success": True,
                "elapsed_seconds": 4.2,
                "attempts": 2,
                "token_usage_summary": {"total_tokens": 321},
                "task_run_history_path": "/tmp/task_runs/history/T13/live.json",
            }

        def fake_persist_canary_batch_record(
            *,
            canaries,
            meta,
            results,
            model,
            validation,
            knowledge_light,
            replay,
            requested_task_id,
            requested_subset,
            root,
            started_at,
            finished_at,
        ):
            seen["canaries"] = canaries
            seen["meta"] = meta
            seen["results"] = results
            seen["model"] = model
            seen["validation"] = validation
            seen["knowledge_light"] = knowledge_light
            seen["replay"] = replay
            seen["requested_task_id"] = requested_task_id
            seen["requested_subset"] = requested_subset
            seen["root"] = root
            seen["started_at"] = started_at
            seen["finished_at"] = finished_at
            return {
                "batch_id": "canary_20260410T120000Z",
                "history_path": str(tmp_path / "task_runs" / "canary_batches" / "history" / "canary_20260410T120000Z.json"),
                "latest_path": str(tmp_path / "task_runs" / "canary_batches" / "latest" / "live__full_curated__standard__default.json"),
            }

        monkeypatch.setattr(
            "trellis.agent.task_runtime.build_market_state",
            fake_build_market_state,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.load_tasks",
            fake_load_tasks,
        )
        monkeypatch.setattr(
            "trellis.agent.task_runtime.run_task",
            fake_run_task,
        )
        monkeypatch.setattr(
            "trellis.agent.task_run_store.persist_canary_batch_record",
            fake_persist_canary_batch_record,
        )

        results = run_canaries(
            [{"id": "T13", "engine_family": "pde", "complexity": "simple"}],
            {"total_budget_usd": 1.0, "version": 7},
            model="gpt-5.4-mini",
            validation="standard",
        )

        assert seen["replay"] is False
        assert seen["requested_task_id"] is None
        assert seen["requested_subset"] is None
        assert seen["knowledge_light"] is False
        assert seen["model"] == "gpt-5.4-mini"
        assert seen["validation"] == "standard"
        assert seen["results"][0]["task_id"] == "T13"
        assert results[0]["canary_batch_id"] == "canary_20260410T120000Z"
        assert results[0]["canary_batch_history_path"].endswith(
            "/task_runs/canary_batches/history/canary_20260410T120000Z.json"
        )


# ---------------------------------------------------------------------------
# CANARY_TASKS.yaml validity
# ---------------------------------------------------------------------------

class TestCanaryFileValidity:
    """Validate the real CANARY_TASKS.yaml in the repo."""

    def test_real_canary_file_loads(self):
        real_path = ROOT / "CANARY_TASKS.yaml"
        if not real_path.exists():
            pytest.skip("CANARY_TASKS.yaml not found in repo root")
        canaries, meta = load_canary_set(real_path)
        assert len(canaries) >= 10, f"Expected at least 10 canaries, got {len(canaries)}"
        assert meta["total_budget_usd"] <= 5.0, "Budget should be reasonable"

    def test_real_canary_file_has_required_fields(self):
        real_path = ROOT / "CANARY_TASKS.yaml"
        if not real_path.exists():
            pytest.skip("CANARY_TASKS.yaml not found in repo root")
        canaries, _ = load_canary_set(real_path)
        for c in canaries:
            assert "id" in c, f"Missing id in canary: {c}"
            assert "engine_family" in c, f"Missing engine_family in {c['id']}"
            assert "rationale" in c, f"Missing rationale in {c['id']}"

    def test_real_canary_file_covers_engine_families(self):
        real_path = ROOT / "CANARY_TASKS.yaml"
        if not real_path.exists():
            pytest.skip("CANARY_TASKS.yaml not found in repo root")
        canaries, _ = load_canary_set(real_path)
        families = {c["engine_family"] for c in canaries}
        # Must cover at least these core families
        assert "lattice" in families
        assert "monte_carlo" in families
        assert "pde" in families
        assert "credit" in families
        assert "analytical" in families

    def test_real_canary_ids_exist_in_tasks_yaml(self):
        real_path = ROOT / "CANARY_TASKS.yaml"
        tasks_path = ROOT / "TASKS.yaml"
        if not real_path.exists() or not tasks_path.exists():
            pytest.skip("CANARY_TASKS.yaml or TASKS.yaml not found")
        canaries, _ = load_canary_set(real_path)
        tasks = yaml.safe_load(tasks_path.read_text())
        task_ids = {t["id"] for t in tasks}
        for c in canaries:
            assert c["id"] in task_ids, f"Canary {c['id']} not found in TASKS.yaml"

    def test_real_canary_file_pins_migrated_semantic_routes(self):
        from trellis.agent.task_runtime import task_to_instrument_type

        real_path = ROOT / "CANARY_TASKS.yaml"
        tasks_path = ROOT / "TASKS.yaml"
        if not real_path.exists() or not tasks_path.exists():
            pytest.skip("CANARY_TASKS.yaml or TASKS.yaml not found")

        canaries, _ = load_canary_set(real_path)
        tasks = yaml.safe_load(tasks_path.read_text())
        task_lookup = {task["id"]: task for task in tasks}
        canary_lookup = {canary["id"]: canary for canary in canaries}

        expected_ids = {"T02", "T13", "T73", "T105"}
        assert expected_ids <= set(canary_lookup)

        instrument_types = {
            task_to_instrument_type(task_lookup[task_id])
            for task_id in expected_ids
        }
        assert instrument_types == {
            "callable_bond",
            "european_option",
            "swaption",
            "quanto_option",
        }

        assert "semantic_contract" in canary_lookup["T105"]["covers"]
        assert "helper_route" in canary_lookup["T105"]["covers"]


# ---------------------------------------------------------------------------
# Drift detection integration (QUA-426)
# ---------------------------------------------------------------------------

class TestDriftIntegration:
    """Test --check-drift and --update-golden integration in the runner."""

    def test_check_drift_no_checkpoints(self, capsys):
        """No checkpoints available → skip drift check gracefully."""
        results = [{"canary_id": "T01", "success": True}]
        canaries = [{"id": "T01", "engine_family": "lattice"}]
        exit_code = check_drift_after_run(results, canaries)
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "no checkpoint found" in output

    def test_check_drift_with_golden_and_checkpoint(self, tmp_path, monkeypatch):
        """With a golden + matching checkpoint → drift report is printed."""
        from trellis.agent.checkpoints import DecisionCheckpoint, StageDecision
        from trellis.agent.golden_traces import save_golden, GOLDEN_DIR

        cp = DecisionCheckpoint(
            task_id="T01",
            instrument_type="zcb_option",
            timestamp="2026-03-30T12:00:00+00:00",
            stages=(
                StageDecision(agent="quant", decision="rate_tree", metadata={}),
            ),
            outcome="pass",
            total_tokens=1000,
        )
        # Save golden to the default dir (monkeypatched)
        golden_dir = tmp_path / "golden"
        monkeypatch.setattr("run_canary.detect_drift_for_canary", lambda tid, cp, **kw: __import__(
            "trellis.agent.golden_traces", fromlist=["TaskDriftReport"]
        ).TaskDriftReport(task_id=tid, engine_family=kw.get("engine_family", ""), has_golden=True))

        # Mock load_latest_checkpoint to return our checkpoint
        monkeypatch.setattr("run_canary.load_latest_checkpoint", lambda tid: cp)

        results = [{"canary_id": "T01", "success": True}]
        canaries = [{"id": "T01", "engine_family": "lattice"}]
        exit_code = check_drift_after_run(results, canaries)
        assert exit_code == 0  # stable = no blocking drift

    def test_promote_golden_with_checkpoints(self, tmp_path, monkeypatch):
        """Promote passing checkpoints to golden traces."""
        from trellis.agent.checkpoints import DecisionCheckpoint, StageDecision
        from trellis.agent.golden_traces import load_golden

        cp = DecisionCheckpoint(
            task_id="T01",
            instrument_type="zcb_option",
            timestamp="2026-03-30T12:00:00+00:00",
            stages=(
                StageDecision(agent="quant", decision="rate_tree", metadata={}),
            ),
            outcome="pass",
            total_tokens=1000,
        )

        monkeypatch.setattr("run_canary.load_latest_checkpoint", lambda tid: cp)
        # Redirect golden writes to tmp_path
        monkeypatch.setattr(
            "run_canary.update_golden_from_results",
            lambda results, checkpoints, **kw: (
                __import__("trellis.agent.golden_traces", fromlist=["update_golden_from_results"])
                .update_golden_from_results(results, checkpoints, directory=tmp_path)
            ),
        )

        results = [{"canary_id": "T01", "success": True}]
        canaries = [{"id": "T01", "engine_family": "lattice"}]
        promote_golden_after_run(results, canaries)

        loaded = load_golden("T01", directory=tmp_path)
        assert loaded is not None
        assert loaded.task_id == "T01"

    def test_promote_golden_fails_on_failure(self, tmp_path, monkeypatch, capsys):
        """Don't promote golden if any canary failed."""
        from trellis.agent.checkpoints import DecisionCheckpoint, StageDecision
        from trellis.agent.golden_traces import load_golden

        cp = DecisionCheckpoint(
            task_id="T01",
            instrument_type="zcb_option",
            timestamp="2026-03-30T12:00:00+00:00",
            stages=(
                StageDecision(agent="quant", decision="rate_tree", metadata={}),
            ),
            outcome="pass",
            total_tokens=1000,
        )

        monkeypatch.setattr("run_canary.load_latest_checkpoint", lambda tid: cp)
        monkeypatch.setattr(
            "run_canary.update_golden_from_results",
            lambda results, checkpoints, **kw: (
                __import__("trellis.agent.golden_traces", fromlist=["update_golden_from_results"])
                .update_golden_from_results(results, checkpoints, directory=tmp_path)
            ),
        )

        results = [
            {"canary_id": "T01", "success": True},
            {"canary_id": "T38", "success": False},
        ]
        canaries = [
            {"id": "T01", "engine_family": "lattice"},
            {"id": "T38", "engine_family": "credit"},
        ]
        promote_golden_after_run(results, canaries)

        output = capsys.readouterr().out
        assert "NOT updated" in output
        assert load_golden("T01", directory=tmp_path) is None

    def test_parse_args_drift_flags(self):
        from run_canary import _parse_args

        args = _parse_args(["--check-drift", "--update-golden"])
        assert args.check_drift is True
        assert args.update_golden is True

    def test_parse_args_replay_flags(self):
        from run_canary import _parse_args

        args = _parse_args(
            [
                "--replay",
                "--cassette-dir",
                "/tmp/cassettes",
                "--cassette-stale-policy",
                "error",
            ]
        )
        assert args.replay is True
        assert args.cassette_dir == "/tmp/cassettes"
        assert args.cassette_stale_policy == "error"

        args_none = _parse_args([])
        assert args_none.check_drift is False
        assert args_none.update_golden is False

    def test_parse_args_knowledge_light_flag(self):
        from run_canary import _parse_args

        args = _parse_args(["--knowledge-light"])
        assert args.knowledge_light is True

        args_none = _parse_args([])
        assert args_none.knowledge_light is False
