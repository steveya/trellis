"""Tests for task runtime helpers used by task rerun and benchmarking scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import ModuleType, SimpleNamespace
import sys

import pytest


def test_run_task_passes_force_rebuild_and_validation():
    """run_task should forward execution mode into the knowledge-aware builder."""
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []
    fake_market_state = object()
    wait_log_path = "/tmp/task_runtime_waits.jsonl"

    class FakeResult:
        success = True
        attempts = 2
        gap_confidence = 0.75
        knowledge_gaps = ["vol_surface"]
        payoff_cls = type("FakePayoff", (), {})
        failures = []
        reflection = {"lesson_captured": "vol_001"}

    timer_values = iter([100.0, 104.5])

    def fake_timer() -> float:
        return next(timer_values)

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("TRELLIS_LLM_WAIT_LOG_PATH", wait_log_path)

    result = run_task(
        {"id": "T13", "title": "European call: theta-method convergence order"},
        market_state=fake_market_state,
        model="test-model",
        force_rebuild=False,
        fresh_build=True,
        validation="fast",
        build_fn=fake_build,
        timer=fake_timer,
        now_fn=lambda: datetime(2026, 3, 24, 12, 0, 0),
    )

    assert calls[0]["description"] == "Build a pricer for: European call: theta-method convergence order"
    assert calls[0]["instrument_type"] == "european_option"
    assert calls[0]["request_metadata"]["task_id"] == "T13"
    assert calls[0]["request_metadata"]["task_title"] == "European call: theta-method convergence order"
    assert calls[0]["request_metadata"]["runtime_contract"]["snapshot_reference"]["source"] == "default"
    assert calls[0]["request_metadata"]["runtime_contract"]["evaluation_tags"] == (
        "task_runtime",
        "market:default",
    )
    assert calls[0]["request_metadata"]["runtime_contract"]["simulation_identity"]["seed_source"] == (
        "derived_from_request_and_snapshot"
    )
    assert calls[0]["request_metadata"]["runtime_contract"]["simulation_identity"]["sample_source"]["kind"] == (
        "market_snapshot"
    )
    assert calls[0]["request_metadata"]["runtime_contract"]["simulation_identity"]["sample_indexing"]["kind"] == (
        "path_index"
    )
    assert calls[0]["model"] == "test-model"
    assert calls[0]["market_state"] is fake_market_state
    assert calls[0]["max_retries"] == 3
    assert calls[0]["validation"] == "fast"
    assert calls[0]["force_rebuild"] is False
    assert calls[0]["fresh_build"] is True
    assert result["runtime_controls"]["llm_wait_log_path"] == wait_log_path
    assert result["success"] is True
    assert result["elapsed_seconds"] == 4.5
    assert result["payoff_class"] == "FakePayoff"
    monkeypatch.undo()


def test_run_task_replays_the_same_simulation_identity_for_the_same_request_and_snapshot():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 1.0
        knowledge_gaps = []
        payoff_cls = type("FakeSemanticPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    task = {
        "id": "T997",
        "title": "Himalaya ranked observation basket",
        "description": (
            "AAPL, MSFT, and NVDA with observation dates 2025-01-15, "
            "2025-02-15, 2025-03-15. At each observation choose the best "
            "performer among remaining constituents, remove it, lock the "
            "simple return, and settle the average locked returns at maturity."
        ),
    }

    run_task(task, market_state=object(), build_fn=fake_build)
    run_task(task, market_state=object(), build_fn=fake_build)

    first_identity = calls[0]["request_metadata"]["runtime_contract"]["simulation_identity"]
    second_identity = calls[1]["request_metadata"]["runtime_contract"]["simulation_identity"]

    assert first_identity == second_identity
    assert first_identity["seed_source"] == "derived_from_request_and_snapshot"
    assert first_identity["simulation_stream_id"].startswith("T997:")


def test_run_task_uses_an_explicit_simulation_seed_when_one_is_provided():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 1.0
        knowledge_gaps = []
        payoff_cls = type("FakeSemanticPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    run_task(
        {
            "id": "T996",
            "title": "Himalaya ranked observation basket",
            "simulation_seed": 271828,
            "description": (
                "AAPL, MSFT, and NVDA with observation dates 2025-01-15, "
                "2025-02-15, 2025-03-15. At each observation choose the best "
                "performer among remaining constituents, remove it, lock the "
                "simple return, and settle the average locked returns at maturity."
            ),
        },
        market_state=object(),
        build_fn=fake_build,
    )

    identity = calls[0]["request_metadata"]["runtime_contract"]["simulation_identity"]
    assert identity["seed"] == 271828
    assert identity["seed_source"] == "task.simulation_seed"
    assert identity["simulation_stream_id"].startswith("T996:")


def test_run_task_drafts_semantic_contract_metadata_for_ranked_observation_basket():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.9
        knowledge_gaps = []
        payoff_cls = type("FakeSemanticPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    result = run_task(
        {
            "id": "T999",
            "title": "Himalaya ranked observation basket",
            "description": (
                "AAPL, MSFT, and NVDA with observation dates 2025-01-15, "
                "2025-02-15, 2025-03-15. At each observation choose the best "
                "performer among remaining constituents, remove it, lock the "
                "simple return, and settle the average locked returns at maturity."
            ),
        },
        market_state=object(),
        build_fn=fake_build,
    )

    assert calls
    assert calls[0]["request_metadata"]["semantic_contract"]["semantic_id"] == "ranked_observation_basket"
    assert calls[0]["request_metadata"]["semantic_contract"]["product"]["instrument_class"] == "basket_path_payoff"
    assert calls[0]["request_metadata"]["semantic_contract"]["product"]["selection_operator"] == "best_of_remaining"
    assert calls[0]["request_metadata"]["semantic_contract"]["product"]["observation_schedule"] == [
        "2025-01-15",
        "2025-02-15",
        "2025-03-15",
    ]
    assert result["success"] is True


def test_run_task_bootstraps_sparse_himalaya_request_into_semantic_basket():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.9
        knowledge_gaps = []
        payoff_cls = type("FakeSemanticPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    result = run_task(
        {
            "id": "T998",
            "title": "Himalaya ranked observation basket",
        },
        market_state=object(),
        build_fn=fake_build,
    )

    assert calls
    assert "AAPL, MSFT, and NVDA" in calls[0]["description"]
    assert "2025-01-15" in calls[0]["description"]
    assert calls[0]["request_metadata"]["semantic_contract"]["semantic_id"] == "ranked_observation_basket"
    assert calls[0]["request_metadata"]["runtime_contract"]["description"].startswith(
        "Build a pricer for: Himalaya ranked observation basket"
    )
    assert result["semantic_contract_id"] == "ranked_observation_basket"
    assert result["success"] is True


def test_run_task_attaches_runtime_contract_snapshot_and_trace_metadata():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.9
        knowledge_gaps = []
        payoff_cls = type("FakeSemanticPayoff", (), {})
        failures = []
        reflection = {}
        platform_request_id = "executor_build_20260325_deadbeef"
        platform_trace_path = "/tmp/executor_build_20260325_deadbeef.yaml"

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    result = run_task(
        {
            "id": "T998",
            "title": "Himalaya ranked observation basket",
            "description": (
                "AAPL, MSFT, and NVDA with observation dates 2025-01-15, "
                "2025-02-15, 2025-03-15. At each observation choose the best "
                "performer among remaining constituents, remove it, lock the "
                "simple return, and settle the average locked returns at maturity."
            ),
            "market": {
                "source": "mock",
                "as_of": "2024-11-15",
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
                "vol_surface": "usd_rates_smile",
            },
        },
        market_state=object(),
        build_fn=fake_build,
    )

    runtime_contract = calls[0]["request_metadata"]["runtime_contract"]
    assert runtime_contract["task_id"] == "T998"
    assert runtime_contract["snapshot_reference"]["source"] == "mock"
    assert runtime_contract["snapshot_reference"]["as_of"] == "2024-11-15"
    assert runtime_contract["snapshot_reference"]["selected_components"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "vol_surface": "usd_rates_smile",
    }
    assert runtime_contract["snapshot_reference"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "usd_ig",
    }
    assert runtime_contract["market_provenance"]["source_kind"] == "synthetic_snapshot"
    assert runtime_contract["market_provenance"]["prior_family"] == "embedded_market_regime"
    assert runtime_contract["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "usd_ig",
    }
    assert "task_runtime" in runtime_contract["evaluation_tags"]
    assert "semantic:ranked_observation_basket" in runtime_contract["evaluation_tags"]
    assert result["runtime_contract"]["trace_identifier"] == "executor_build_20260325_deadbeef"
    assert result["runtime_contract"]["trace_path"] == "/tmp/executor_build_20260325_deadbeef.yaml"
    assert result["runtime_contract"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "usd_ig",
    }
    assert result["market_context"]["selected_curve_names"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "credit_curve": "usd_ig",
    }
    assert result["market_context"]["provenance"]["source_kind"] == "synthetic_snapshot"
    assert result["market_context"]["provenance"]["prior_seed"] == runtime_contract["market_provenance"]["prior_seed"]


def test_artifacts_from_payload_collects_analytical_trace_paths():
    from trellis.agent.task_runtime import _artifacts_from_payload

    artifacts = _artifacts_from_payload(
        {
            "platform_request_id": "executor_build_demo",
            "platform_trace_path": "/tmp/platform.yaml",
            "analytical_trace_path": "/tmp/analytical.json",
            "analytical_trace_text_path": "/tmp/analytical.md",
            "reflection": {},
        }
    )

    assert artifacts["platform_request_ids"] == ["executor_build_demo"]
    assert artifacts["platform_trace_paths"] == ["/tmp/platform.yaml"]
    assert artifacts["analytical_trace_paths"] == [
        "/tmp/analytical.json",
    ]
    assert artifacts["analytical_trace_text_paths"] == ["/tmp/analytical.md"]


def test_run_task_persists_latest_record(monkeypatch):
    """Each task run should write a canonical latest/history record."""
    from trellis.agent.task_runtime import run_task

    persisted: dict[str, object] = {}

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.8
        knowledge_gaps = []
        payoff_cls = type("FakePayoff", (), {})
        failures = []
        reflection = {}
        post_build_tracking = {"last_phase": "consolidation_dispatched", "last_status": "backgrounded"}

    def fake_build(**kwargs):
        return FakeResult()

    def fake_persist(task, result):
        persisted["task"] = task
        persisted["result"] = dict(result)
        return {
            "history_path": "/tmp/task_runs/history/T13/run.json",
            "latest_path": "/tmp/task_runs/latest/T13.json",
            "latest_index_path": "/tmp/task_results_latest.json",
            "diagnosis_packet_path": "/tmp/task_runs/diagnostics/history/T13/run.json",
            "diagnosis_dossier_path": "/tmp/task_runs/diagnostics/history/T13/run.md",
            "latest_diagnosis_packet_path": "/tmp/task_runs/diagnostics/latest/T13.json",
            "latest_diagnosis_dossier_path": "/tmp/task_runs/diagnostics/latest/T13.md",
            "diagnosis_headline": "Demo task completed successfully.",
            "diagnosis_failure_bucket": "success",
            "diagnosis_decision_stage": "completed",
            "diagnosis_next_action": "No action required.",
            "diagnosis_persist_error": "",
            "diagnosis_persist_skipped": "",
        }

    monkeypatch.setattr(
        "trellis.agent.task_run_store.persist_task_run_record",
        fake_persist,
    )

    result = run_task(
        {"id": "T13", "title": "European call: theta-method convergence order"},
        market_state=object(),
        build_fn=fake_build,
    )

    assert persisted["task"]["id"] == "T13"
    assert persisted["result"]["task_id"] == "T13"
    assert result["task_run_history_path"] == "/tmp/task_runs/history/T13/run.json"
    assert result["task_run_latest_path"] == "/tmp/task_runs/latest/T13.json"
    assert result["task_run_latest_index_path"] == "/tmp/task_results_latest.json"
    assert result["task_diagnosis_packet_path"] == "/tmp/task_runs/diagnostics/history/T13/run.json"
    assert result["task_diagnosis_dossier_path"] == "/tmp/task_runs/diagnostics/history/T13/run.md"
    assert result["task_diagnosis_latest_packet_path"] == "/tmp/task_runs/diagnostics/latest/T13.json"
    assert result["task_diagnosis_latest_dossier_path"] == "/tmp/task_runs/diagnostics/latest/T13.md"
    assert result["task_diagnosis_headline"] == "Demo task completed successfully."
    assert result["task_diagnosis_persist_error"] == ""
    assert result["task_diagnosis_persist_skipped"] == ""


def test_run_task_uses_single_construct_as_preferred_method():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.8
        knowledge_gaps = []
        payoff_cls = type("FakePDEPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    result = run_task(
        {
            "id": "T13",
            "title": "European call: theta-method convergence order",
            "construct": "pde",
        },
        market_state=object(),
        build_fn=fake_build,
    )

    assert calls[0]["description"].startswith(
        "Build a pricer for: European call: theta-method convergence order"
    )
    assert "Construct methods: pde_solver" in calls[0]["description"]
    assert "Comparison targets: pde_solver (pde_solver)" in calls[0]["description"]
    assert calls[0]["instrument_type"] == "european_option"
    assert calls[0]["preferred_method"] == "pde_solver"
    assert calls[0]["request_metadata"]["task_id"] == "T13"
    assert calls[0]["request_metadata"]["task_title"] == "European call: theta-method convergence order"
    assert calls[0]["request_metadata"]["preferred_method"] == "pde_solver"
    assert calls[0]["request_metadata"]["runtime_contract"]["evaluation_tags"] == (
        "task_runtime",
        "construct:pde",
        "market:default",
    )
    assert calls[0]["model"] == "gpt-5.4-mini"
    assert calls[0]["market_state"] is not None
    assert calls[0]["max_retries"] == 3
    assert calls[0]["validation"] == "standard"
    assert calls[0]["force_rebuild"] is True
    assert calls[0]["fresh_build"] is False
    assert result["comparison_task"] is False
    assert result["preferred_method"] == "pde_solver"
    assert result["success"] is True


def test_run_task_builds_each_construct_method_for_comparison_task():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []
    price_map = {
        "crr_tree": 10.10,
        "bs_pde": 10.00,
        "mc_exact": 10.05,
        "fft": 9.98,
        "cos": 10.01,
        "black_scholes": 10.00,
    }

    class FakeResult:
        def __init__(self, target: str, method: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.75
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": price_map[target]})
            self.failures = []
            self.reflection = {"lesson_captured": f"{target}_001"}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult(kwargs["comparison_target"], kwargs["preferred_method"])

    result = run_task(
        {
            "id": "T74",
            "title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)",
            "construct": ["lattice", "pde", "monte_carlo", "transforms"],
            "cross_validate": {
                "internal": ["crr_tree", "bs_pde", "mc_exact", "fft", "cos"],
                "analytical": "black_scholes",
            },
            "new_component": None,
        },
        market_state=object(),
        fresh_build=True,
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert [call["preferred_method"] for call in calls] == [
        "rate_tree",
        "pde_solver",
        "monte_carlo",
        "fft_pricing",
        "fft_pricing",
        "analytical",
    ]
    assert [call["comparison_target"] for call in calls] == [
        "crr_tree",
        "bs_pde",
        "mc_exact",
        "fft",
        "cos",
        "black_scholes",
    ]
    assert all(call["fresh_build"] is True for call in calls)
    assert calls[0]["request_metadata"]["task_id"] == "T74"
    assert calls[0]["request_metadata"]["task_title"] == "European equity call: 5-way (tree, PDE, MC, FFT, COS)"
    assert calls[0]["request_metadata"]["comparison_target"] == "crr_tree"
    assert calls[0]["request_metadata"]["preferred_method"] == "rate_tree"
    assert calls[0]["request_metadata"]["runtime_contract"]["snapshot_reference"]["source"] == "default"
    assert "comparison" in calls[0]["request_metadata"]["runtime_contract"]["evaluation_tags"]
    assert calls[-1]["request_metadata"]["task_id"] == "T74"
    assert calls[-1]["request_metadata"]["task_title"] == "European equity call: 5-way (tree, PDE, MC, FFT, COS)"
    assert calls[-1]["request_metadata"]["comparison_target"] == "black_scholes"
    assert calls[-1]["request_metadata"]["preferred_method"] == "analytical"
    assert calls[-1]["request_metadata"]["runtime_contract"]["snapshot_reference"]["source"] == "default"
    assert "comparison" in calls[-1]["request_metadata"]["runtime_contract"]["evaluation_tags"]
    assert result["comparison_task"] is True
    assert result["construct_methods"] == [
        "rate_tree",
        "pde_solver",
        "monte_carlo",
        "fft_pricing",
    ]
    assert result["comparison_targets"] == [
        "crr_tree",
        "bs_pde",
        "mc_exact",
        "fft",
        "cos",
        "black_scholes",
    ]
    assert result["cross_validate"] == {
        "internal": ["crr_tree", "bs_pde", "mc_exact", "fft", "cos"],
        "analytical": "black_scholes",
    }
    assert set(result["method_results"]) == {
        "crr_tree",
        "bs_pde",
        "mc_exact",
        "fft",
        "cos",
        "black_scholes",
    }
    assert result["cross_validation"]["status"] == "passed"
    assert result["cross_validation"]["reference_target"] == "black_scholes"
    assert result["cross_validation"]["tolerance_pct"] == 5.0
    assert result["success"] is True


def test_cross_validate_comparison_task_prices_reused_fx_modules():
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
        build_market_state_for_task,
    )
    from trellis.instruments._agent.fxvanillaanalytical import FXVanillaAnalyticalPayoff
    from trellis.instruments._agent.fxvanillamontecarlo import FXVanillaMonteCarloPayoff

    task = {
        "id": "E25",
        "title": "FX option (EURUSD): GK analytical vs MC",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
            "fx_rate": "EURUSD",
        },
        "cross_validate": {
            "internal": ["gk_mc"],
            "analytical": "garman_kohlhagen",
        },
    }
    market_state, _ = build_market_state_for_task(task)

    class FakeResult:
        def __init__(self, payoff_cls):
            self.success = True
            self.payoff_cls = payoff_cls

    live_results = {
        "gk_mc": FakeResult(FXVanillaMonteCarloPayoff),
        "garman_kohlhagen": FakeResult(FXVanillaAnalyticalPayoff),
    }
    comparison_targets = [
        ComparisonBuildTarget("gk_mc", "monte_carlo"),
        ComparisonBuildTarget("garman_kohlhagen", "analytical", is_reference=True),
    ]

    result = _cross_validate_comparison_task(
        comparison_targets,
        live_results,
        market_state,
        configured_targets=task["cross_validate"],
    )

    assert result["status"] == "passed"
    assert result["reference_target"] == "garman_kohlhagen"
    assert set(result["successful_targets"]) == {"gk_mc", "garman_kohlhagen"}
    assert result["price_errors"] == {}
    assert result["prices"]["gk_mc"] > 0.0
    assert result["prices"]["garman_kohlhagen"] > 0.0


def test_cross_validate_comparison_task_prices_reused_quanto_modules():
    from trellis.agent.task_runtime import (
        ComparisonBuildTarget,
        _cross_validate_comparison_task,
        build_market_state_for_task,
    )
    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionAnalyticalPayoff
    from trellis.instruments._agent.quantooptionmontecarlo import QuantoOptionMonteCarloPayoff

    task = {
        "id": "T105",
        "title": "Quanto option: quanto-adjusted BS vs MC cross-currency",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
            "fx_rate": "EURUSD",
            "underlier_spot": "SPX",
            "model_parameters": "heston_equity",
        },
        "market_assertions": {
            "requires": [
                "discount_curve",
                "forward_curve",
                "fx_rates",
                "spot",
                "model_parameters",
            ],
            "selected": {
                "discount_curve": "usd_ois",
                "forecast_curve": "EUR-DISC",
                "fx_rate": "EURUSD",
                "underlier_spot": "SPX",
                "model_parameters": "heston_equity",
            },
        },
        "cross_validate": {
            "internal": ["quanto_bs", "mc_quanto"],
            "tolerance_pct": 5.0,
        },
    }
    market_state, _ = build_market_state_for_task(task)

    class FakeResult:
        def __init__(self, payoff_cls):
            self.success = True
            self.payoff_cls = payoff_cls

    live_results = {
        "quanto_bs": FakeResult(QuantoOptionAnalyticalPayoff),
        "mc_quanto": FakeResult(QuantoOptionMonteCarloPayoff),
    }
    comparison_targets = [
        ComparisonBuildTarget("quanto_bs", "analytical", is_reference=True),
        ComparisonBuildTarget("mc_quanto", "monte_carlo"),
    ]

    result = _cross_validate_comparison_task(
        comparison_targets,
        live_results,
        market_state,
        configured_targets=task["cross_validate"],
    )

    assert result["status"] == "passed"
    assert result["reference_target"] == "quanto_bs"
    assert set(result["successful_targets"]) == {"quanto_bs", "mc_quanto"}
    assert result["price_errors"] == {}
    assert result["prices"]["quanto_bs"] > 0.0
    assert result["prices"]["mc_quanto"] > 0.0


def test_load_tasks_excludes_framework_inventory():
    from trellis.agent.task_runtime import load_tasks

    tasks = load_tasks(status=None)
    task_ids = {task["id"] for task in tasks}

    assert "T90" in task_ids
    assert "T94" in task_ids
    assert "E21" in task_ids
    assert "T91" not in task_ids
    assert "E01" not in task_ids


def test_load_framework_tasks_returns_meta_inventory():
    from trellis.agent.task_runtime import load_framework_tasks

    tasks = load_framework_tasks(status=None)
    task_ids = {task["id"] for task in tasks}

    assert {"T91", "T92", "T93", "E01", "E20"} <= task_ids
    assert "T94" not in task_ids
    assert "E21" not in task_ids


def test_task_to_instrument_type_maps_quanto_to_family_contract_key():
    from trellis.agent.task_runtime import task_to_instrument_type

    instrument_type = task_to_instrument_type(
        {"id": "T105", "title": "Quanto option: quanto-adjusted BS vs MC cross-currency"}
    )

    assert instrument_type == "quanto_option"


def test_run_task_uses_internal_cross_validate_targets_for_single_construct_family():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        def __init__(self, target: str, method: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.7
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 99.5})
            self.failures = []
            self.reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult(kwargs["comparison_target"], kwargs["preferred_method"])

    result = run_task(
        {
            "id": "T10",
            "title": "Tree convergence study: price oscillation and Richardson extrapolation",
            "construct": "lattice",
            "cross_validate": {
                "internal": ["raw_tree", "extrapolated_tree"],
                "analytical": "black_scholes",
            },
        },
        market_state=object(),
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert result["comparison_task"] is True
    assert result["comparison_targets"] == ["raw_tree", "extrapolated_tree", "black_scholes"]
    assert [call["preferred_method"] for call in calls] == [
        "rate_tree",
        "rate_tree",
        "analytical",
    ]


def test_run_task_includes_cross_validate_context_in_comparison_description():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        def __init__(self, target: str, method: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.8
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 101.0})
            self.failures = []
            self.reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult(kwargs["comparison_target"], kwargs["preferred_method"])

    run_task(
        {
            "id": "T24X",
            "title": "Finite element method (FEM) vs finite difference for European",
            "construct": "pde",
            "new_component": "fem_1d_solver",
            "cross_validate": {
                "internal": ["fem_solver", "fd_theta_method"],
                "analytical": "black_scholes",
            },
        },
        market_state=object(),
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert calls
    first_description = calls[0]["description"]
    assert "Build a pricer for: Finite element method (FEM) vs finite difference for European" in first_description
    assert "Construct methods:" in first_description
    assert "Comparison targets:" in first_description
    assert "Cross-validation harness:" in first_description
    assert "analytical benchmark: black_scholes" in first_description
    assert "New component: fem_1d_solver" in first_description


def test_run_task_maps_early_exercise_policy_targets_to_monte_carlo_family():
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []

    class FakeResult:
        def __init__(self, target: str, method: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.7
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 99.5})
            self.failures = []
            self.reflection = {}

    def fake_build(**kwargs):
        calls.append(kwargs)
        return FakeResult(kwargs["comparison_target"], kwargs["preferred_method"])

    result = run_task(
        {
            "id": "T07",
            "title": "American put: tree vs primal-dual MC vs stochastic mesh",
            "construct": ["lattice", "monte_carlo"],
            "cross_validate": {
                "internal": ["hw_tree_bermudan", "primal_dual_mc", "stochastic_mesh"],
            },
        },
        market_state=object(),
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert result["comparison_task"] is True
    assert [call["preferred_method"] for call in calls] == [
        "rate_tree",
        "monte_carlo",
        "monte_carlo",
    ]


def test_run_task_aggregates_artifact_references_for_comparison_tasks():
    from trellis.agent.task_runtime import run_task

    class FakeResult:
        def __init__(self, target: str, method: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.6
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 100.0})
            self.failures = []
            self.agent_observations = [{"agent": "critic", "summary": f"{target} concern"}]
            self.knowledge_summary = {
                "lesson_ids": [f"{target}_lesson"],
                "lesson_titles": [f"{target} lesson"],
                "principle_ids": ["P1"],
                "cookbook_method": method,
                "retrieval_stages": ["initial_build", "validation_failed"],
                "retrieval_sources": ["callback"],
            }
            self.platform_request_id = f"executor_build_{target}"
            self.platform_trace_path = f"/tmp/{target}_platform.yaml"
            self.reflection = {
                "lesson_captured": f"{target}_lesson",
                "knowledge_trace_saved": f"/tmp/{target}_knowledge.yaml",
                "cookbook_candidate_saved": f"/tmp/{target}_cookbook.yaml",
                "knowledge_gap_log_saved": "/tmp/knowledge_gaps.yaml",
            }

    def fake_build(**kwargs):
        return FakeResult(kwargs["comparison_target"], kwargs["preferred_method"])

    result = run_task(
        {
            "id": "E28",
            "title": "European equity call: transform-family separation (FFT vs COS)",
            "construct": ["transforms"],
            "cross_validate": {
                "internal": ["fft", "cos"],
                "analytical": "black_scholes",
            },
        },
        market_state=object(),
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert result["artifacts"]["platform_request_ids"] == [
        "executor_build_black_scholes",
        "executor_build_cos",
        "executor_build_fft",
    ]
    assert result["artifacts"]["platform_trace_paths"] == [
        "/tmp/black_scholes_platform.yaml",
        "/tmp/cos_platform.yaml",
        "/tmp/fft_platform.yaml",
    ]
    assert result["artifacts"]["knowledge_trace_paths"] == [
        "/tmp/black_scholes_knowledge.yaml",
        "/tmp/cos_knowledge.yaml",
        "/tmp/fft_knowledge.yaml",
    ]
    assert result["artifacts"]["cookbook_candidate_paths"] == [
        "/tmp/black_scholes_cookbook.yaml",
        "/tmp/cos_cookbook.yaml",
        "/tmp/fft_cookbook.yaml",
    ]
    assert result["artifacts"]["knowledge_gap_log_paths"] == ["/tmp/knowledge_gaps.yaml"]
    assert result["knowledge_summary"]["lesson_ids"] == [
        "black_scholes_lesson",
        "cos_lesson",
        "fft_lesson",
    ]
    assert result["knowledge_summary"]["cookbook_methods"] == [
        "analytical",
        "fft_pricing",
    ]
    assert result["knowledge_summary"]["retrieval_stages"] == [
        "initial_build",
        "validation_failed",
    ]
    assert result["knowledge_summary"]["retrieval_sources"] == ["callback"]
    assert result["learning"]["knowledge_outcome"] == "captured_knowledge"
    assert "lesson(s)" in result["learning"]["knowledge_outcome_reason"]
    assert result["agent_observation_count"] == 3
    assert result["method_results"]["fft"]["artifacts"]["platform_request_ids"] == [
        "executor_build_fft"
    ]
    assert result["method_results"]["fft"]["artifacts"]["knowledge_trace_paths"] == [
        "/tmp/fft_knowledge.yaml"
    ]
    assert result["method_results"]["fft"]["knowledge_summary"]["lesson_ids"] == ["fft_lesson"]
    assert result["method_results"]["fft"]["learning"]["knowledge_outcome"] == "captured_knowledge"


def test_run_task_records_promotion_candidates_for_fresh_build_comparison_success(monkeypatch):
    from trellis.agent.task_runtime import run_task

    captured: dict[str, object] = {}

    class FakeResult:
        def __init__(self, target: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.7
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 100.0})
            self.failures = []
            self.code = f"class {target.title()}Payoff: pass\n"
            self.agent_observations = []
            self.knowledge_summary = {}
            self.platform_request_id = f"executor_build_{target}"
            self.platform_trace_path = f"/tmp/{target}_trace.yaml"
            self.blocker_details = None
            self.reflection = {}

    def fake_build(**kwargs):
        return FakeResult(kwargs["comparison_target"])

    def fake_record_candidates(**kwargs):
        captured.update(kwargs)
        return {
            "black_scholes": "/tmp/black_scholes_candidate.yaml",
            "mc_exact": "/tmp/mc_exact_candidate.yaml",
        }

    monkeypatch.setattr(
        "trellis.agent.task_runtime._record_promotion_candidates",
        fake_record_candidates,
    )

    result = run_task(
        {
            "id": "T105",
            "title": "Quanto option: analytical vs MC proving run",
            "construct": ["analytical", "monte_carlo"],
            "cross_validate": {
                "analytical": "black_scholes",
                "internal": ["mc_exact"],
            },
        },
        market_state=object(),
        fresh_build=True,
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert captured["task"]["id"] == "T105"
    assert captured["instrument_type"] == "quanto_option"
    assert captured["cross_validation"]["status"] == "passed"
    assert sorted(captured["method_results"]) == ["black_scholes", "mc_exact"]
    assert result["reflection"]["promotion_candidate_saved"] == [
        "/tmp/black_scholes_candidate.yaml",
        "/tmp/mc_exact_candidate.yaml",
    ]
    assert result["artifacts"]["promotion_candidate_paths"] == [
        "/tmp/black_scholes_candidate.yaml",
        "/tmp/mc_exact_candidate.yaml",
    ]
    assert (
        result["method_results"]["black_scholes"]["reflection"]["promotion_candidate_saved"]
        == "/tmp/black_scholes_candidate.yaml"
    )
    assert (
        result["method_results"]["mc_exact"]["artifacts"]["promotion_candidate_paths"]
        == ["/tmp/mc_exact_candidate.yaml"]
    )


def test_run_task_skips_promotion_candidates_without_fresh_build(monkeypatch):
    from trellis.agent.task_runtime import run_task

    calls: list[dict[str, object]] = []

    class FakeResult:
        def __init__(self, target: str):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.7
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": 100.0})
            self.failures = []
            self.code = f"class {target.title()}Payoff: pass\n"
            self.agent_observations = []
            self.knowledge_summary = {}
            self.platform_request_id = f"executor_build_{target}"
            self.platform_trace_path = f"/tmp/{target}_trace.yaml"
            self.blocker_details = None
            self.reflection = {}

    def fake_build(**kwargs):
        return FakeResult(kwargs["comparison_target"])

    monkeypatch.setattr(
        "trellis.agent.task_runtime._record_promotion_candidates",
        lambda **kwargs: calls.append(kwargs) or {},
    )

    result = run_task(
        {
            "id": "T105",
            "title": "Quanto option: analytical vs MC proving run",
            "construct": ["analytical", "monte_carlo"],
            "cross_validate": {
                "analytical": "black_scholes",
                "internal": ["mc_exact"],
            },
        },
        market_state=object(),
        fresh_build=False,
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert calls == []
    assert result["artifacts"]["promotion_candidate_paths"] == []
    assert "promotion_candidate_saved" not in result["reflection"]


def test_run_task_skips_promotion_candidates_when_cross_validation_fails(monkeypatch):
    from trellis.agent.task_runtime import run_task

    calls: list[dict[str, object]] = []

    class FakeResult:
        def __init__(self, target: str, price: float):
            self.success = True
            self.attempts = 1
            self.gap_confidence = 0.7
            self.knowledge_gaps = []
            self.payoff_cls = type(f"{target.title()}Payoff", (), {"price": price})
            self.failures = []
            self.code = f"class {target.title()}Payoff: pass\n"
            self.agent_observations = []
            self.knowledge_summary = {}
            self.platform_request_id = f"executor_build_{target}"
            self.platform_trace_path = f"/tmp/{target}_trace.yaml"
            self.blocker_details = None
            self.reflection = {}

    def fake_build(**kwargs):
        target = kwargs["comparison_target"]
        price = 100.0 if target == "black_scholes" else 130.0
        return FakeResult(target, price)

    monkeypatch.setattr(
        "trellis.agent.task_runtime._record_promotion_candidates",
        lambda **kwargs: calls.append(kwargs) or {},
    )

    result = run_task(
        {
            "id": "T105",
            "title": "Quanto option: analytical vs MC proving run",
            "construct": ["analytical", "monte_carlo"],
            "cross_validate": {
                "analytical": "black_scholes",
                "internal": ["mc_exact"],
                "tolerance_pct": 0.5,
            },
        },
        market_state=object(),
        fresh_build=True,
        build_fn=fake_build,
        payoff_factory=lambda payoff_cls, spec_schema, settle: payoff_cls(),
        price_fn=lambda payoff, market_state: payoff.price,
    )

    assert result["cross_validation"]["status"] == "failed"
    assert calls == []
    assert result["artifacts"]["promotion_candidate_paths"] == []


def test_build_result_payload_includes_blocker_details(tmp_path):
    from trellis.agent.task_runtime import _build_result_payload

    trace_path = tmp_path / "executor_build_blocked.yaml"
    trace_path.write_text(
        """
request_id: executor_build_blocked
status: failed
outcome: request_failed
details:
  source_sanitization:
    source_status: sanitized
    fence_removed: true
events:
  - event: builder_attempt_generated
    status: ok
    timestamp: 2026-03-25T18:00:00+00:00
    details:
      parse_status: compiled
  - event: builder_attempt_failed
    status: error
    timestamp: 2026-03-25T18:00:01+00:00
    details:
      source_sanitization:
        source_status: rejected
      parse_status: parse_failed
      correlation_preflight:
        correlation_status: regularized
""".strip()
    )

    class FakeResult:
        success = False
        attempts = 1
        gap_confidence = 0.2
        knowledge_gaps = ["exercise control missing"]
        payoff_cls = None
        failures = ["blocked"]
        agent_observations = [{"agent": "critic", "summary": "double discounting"}]
        knowledge_summary = {
            "lesson_ids": ["mc_007"],
            "cookbook_method": "monte_carlo",
            "retrieval_stages": ["semantic_validation_failed"],
            "retrieval_sources": ["compiled_request_payload"],
        }
        platform_trace_path = str(trace_path)
        platform_request_id = "executor_build_blocked"
        analytical_trace_path = "/tmp/blocked_analytical_trace.json"
        analytical_trace_text_path = "/tmp/blocked_analytical_trace.md"
        blocker_details = {
            "blocker_codes": ["missing_symbol:demo"],
            "new_primitive_workflow": {"summary": "library_repair"},
        }
        post_build_tracking = {
            "last_phase": "reflection_completed",
            "last_status": "ok",
            "active_flags": {"skip_reflection": False},
        }
        reflection = {}

    payload = _build_result_payload(FakeResult(), preferred_method="analytical")

    assert payload["blocker_details"]["blocker_codes"] == ["missing_symbol:demo"]
    assert payload["agent_observation_count"] == 1
    assert payload["agent_observations"][0]["agent"] == "critic"
    assert payload["knowledge_summary"]["lesson_ids"] == ["mc_007"]
    assert payload["learning"]["retrieval_stages"] == ["semantic_validation_failed"]
    assert payload["learning"]["retrieval_sources"] == ["compiled_request_payload"]
    assert payload["artifacts"]["platform_trace_paths"] == [str(trace_path)]
    assert payload["artifacts"]["analytical_trace_paths"] == [
        "/tmp/blocked_analytical_trace.json"
    ]
    assert payload["artifacts"]["analytical_trace_text_paths"] == [
        "/tmp/blocked_analytical_trace.md"
    ]
    assert payload["build_observability"]["source_status"] == "rejected"
    assert payload["build_observability"]["parse_status"] == "parse_failed"
    assert payload["build_observability"]["correlation_status"] == "regularized"
    assert payload["learning"]["knowledge_outcome"] == "blocked_without_learning"
    assert payload["post_build_tracking"]["last_phase"] == "reflection_completed"


def test_build_result_payload_surfaces_lesson_contract():
    from trellis.agent.task_runtime import _build_result_payload

    class FakeResult:
        success = False
        attempts = 1
        gap_confidence = 0.2
        knowledge_gaps = ["exercise control missing"]
        payoff_cls = None
        failures = ["blocked"]
        agent_observations = []
        knowledge_summary = {}
        platform_trace_path = None
        platform_request_id = "executor_build_blocked"
        analytical_trace_path = None
        analytical_trace_text_path = None
        blocker_details = None
        reflection = {
            "lesson_contract": {
                "contract": "lesson_payload.v1",
                "valid": True,
                "normalized_payload": {"title": "Use antithetic control variates"},
                "errors": [],
                "warnings": [],
            },
            "lesson_promotion_outcome": "skipped",
        }

    payload = _build_result_payload(FakeResult(), preferred_method="analytical")

    assert payload["reflection"]["lesson_contract"]["valid"] is True
    assert payload["reflection"]["lesson_promotion_outcome"] == "skipped"
    assert payload["learning"]["lesson_contract_outcome"] == "validated"
    assert payload["learning"]["lesson_contract_count"] == 1


def test_aggregate_failures_flattens_nested_method_results():
    from trellis.agent.task_runtime import _aggregate_failures

    failures = _aggregate_failures(
        {
            "success": False,
            "cross_validation": {"status": "insufficient_results"},
            "method_results": {
                "psor_pde": {
                    "success": False,
                    "failures": [
                        "OpenAI json request failed after 3 attempts for model 'gpt-5-mini': TimeoutError: OpenAI request exceeded 30.0s",
                    ],
                },
                "lsm_mc": {
                    "success": False,
                    "failures": ["name 'AMERICAN' is not defined"],
                },
            },
        }
    )

    assert any("psor_pde" in failure and "TimeoutError" in failure for failure in failures)
    assert any("lsm_mc" in failure and "AMERICAN" in failure for failure in failures)
    assert "cross_validation status: insufficient_results" in failures


def test_task_to_instrument_type_detects_european_equity_call():
    from trellis.agent.task_runtime import task_to_instrument_type

    assert (
        task_to_instrument_type(
            {"id": "T74", "title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)"}
        )
        == "european_option"
    )


def test_task_to_instrument_type_detects_basket_option():
    from trellis.agent.task_runtime import task_to_instrument_type

    assert (
        task_to_instrument_type(
            {"id": "T999", "title": "Worst-of basket option on two equities"}
        )
        == "basket_option"
    )


def test_task_to_instrument_type_detects_bare_european_shape():
    from trellis.agent.task_runtime import task_to_instrument_type

    assert task_to_instrument_type(
        {"id": "T24", "title": "Finite element method (FEM) vs finite difference for European"}
    ) == "european_option"


def test_build_market_state_uses_mock_snapshot_defaults():
    from trellis.agent.task_runtime import build_market_state

    market_state = build_market_state()

    assert "discount_curve" in market_state.available_capabilities
    assert "forward_curve" in market_state.available_capabilities
    assert "black_vol_surface" in market_state.available_capabilities
    assert market_state.forecast_curves is not None
    assert market_state.fx_rates is not None
    assert "spot" in market_state.available_capabilities
    assert "state_space" in market_state.available_capabilities
    assert "local_vol_surface" in market_state.available_capabilities
    assert "jump_parameters" in market_state.available_capabilities
    assert "model_parameters" in market_state.available_capabilities


def test_build_market_state_for_task_selects_named_mock_components():
    from trellis.agent.task_runtime import build_market_state_for_task

    market_state, market_context = build_market_state_for_task({
        "id": "E22",
        "title": "Cap/floor: Black caplet stack vs MC rate simulation",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
            "vol_surface": "usd_rates_smile",
        },
        "market_assertions": {
            "requires": ["discount_curve", "forward_curve", "black_vol_surface"],
            "selected": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
                "vol_surface": "usd_rates_smile",
            },
        },
    })

    assert market_context["source"] == "mock"
    assert market_context["as_of"] == "2024-11-15"
    assert market_context["selected_components"] == {
        "discount_curve": "usd_ois",
        "forecast_curve": "USD-SOFR-3M",
        "vol_surface": "usd_rates_smile",
    }
    assert market_context["selected_curve_names"]["discount_curve"] == "usd_ois"
    assert market_context["selected_curve_names"]["forecast_curve"] == "USD-SOFR-3M"
    assert market_context["selected_curve_names"]["credit_curve"] == "usd_ig"
    assert set(market_state.forecast_curves) == {"USD-SOFR-3M"}
    assert market_state.discount is not None
    assert market_state.vol_surface is not None
    assert market_state.forward_curve is not None
    assert market_state.forward_curve.forward_rate(0.5, 1.0) == pytest.approx(
        market_state.forecast_forward_curve("USD-SOFR-3M").forward_rate(0.5, 1.0)
    )


def test_build_market_state_for_credit_task_injects_default_credit_curve_when_unspecified():
    from trellis.agent.task_runtime import build_market_state_for_task
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol

    fallback_market_state = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.05, max_tenor=31.0),
        vol_surface=FlatVol(0.20),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "credit_curve": "usd_ig",
        },
    )

    market_state, market_context = build_market_state_for_task({
        "id": "T38",
        "title": "CDS pricing: hazard rate MC vs survival prob analytical",
        "construct": ["monte_carlo", "credit"],
    }, fallback_market_state=fallback_market_state)

    assert market_state.credit_curve is not None
    assert "credit_curve" in market_state.available_capabilities
    assert market_context["selected_curve_names"]["credit_curve"] == "default_flat_credit_curve_2pct"
    assert market_context["provenance"]["credit_curve"]["source"] == "runtime_default"
    assert market_context["provenance"]["credit_curve"]["hazard_rate"] == 0.02


def test_build_market_state_for_credit_task_injects_default_credit_curve_when_snapshot_lacks_it(monkeypatch):
    from trellis.agent.task_runtime import build_market_state_for_task
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol

    class FakeSnapshot(SimpleNamespace):
        source = "mock"
        as_of = date(2024, 11, 15)
        metadata = {}
        provenance = {"source": "mock"}

        def to_market_state(self, **kwargs):
            return MarketState(
                as_of=date(2024, 11, 15),
                settlement=date(2024, 11, 15),
                discount=YieldCurve.flat(0.05, max_tenor=31.0),
                vol_surface=FlatVol(0.20),
                selected_curve_names={"discount_curve": "usd_ois"},
            )

    monkeypatch.setattr(
        "trellis.agent.task_runtime.build_market_snapshot_for_task",
        lambda task: FakeSnapshot(),
    )

    market_state, market_context = build_market_state_for_task({
        "id": "T38",
        "title": "CDS pricing: hazard rate MC vs survival prob analytical",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
        },
    })

    assert market_state.credit_curve is not None
    assert "credit_curve" in market_state.available_capabilities
    assert market_context["selected_components"]["credit_curve"] == market_state.credit_curve
    assert market_context["selected_curve_names"]["credit_curve"] == "default_flat_credit_curve_2pct"
    assert market_context["provenance"]["credit_curve"]["source"] == "runtime_default"


def test_build_market_state_for_task_bridges_selected_fx_rate_to_spot():
    from trellis.agent.task_runtime import build_market_state_for_task

    market_state, market_context = build_market_state_for_task({
        "id": "E25",
        "title": "FX option (EURUSD): GK analytical vs MC",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
            "fx_rate": "EURUSD",
        },
        "market_assertions": {
            "requires": ["discount_curve", "forward_curve", "fx_rates", "spot"],
            "selected": {
                "discount_curve": "usd_ois",
                "forecast_curve": "EUR-DISC",
                "fx_rate": "EURUSD",
            },
        },
    })

    assert market_context["selected_components"]["fx_rate"] == "EURUSD"
    assert market_context["selected_curve_names"]["discount_curve"] == "usd_ois"
    assert market_context["selected_curve_names"]["forecast_curve"] == "EUR-DISC"
    assert set(market_state.fx_rates) == {"EURUSD"}
    assert market_state.spot == pytest.approx(market_state.fx_rates["EURUSD"].spot)
    assert market_state.underlier_spots["EURUSD"] == pytest.approx(market_state.spot)


def test_build_market_state_for_task_selects_state_space():
    from trellis.agent.task_runtime import build_market_state_for_task

    market_state, market_context = build_market_state_for_task({
        "id": "S01",
        "title": "Scenario-weighted pricing stress task",
        "market": {
            "source": "mock",
            "as_of": "2024-11-15",
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
            "fx_rate": "EURUSD",
            "state_space": "macro_regime",
        },
        "market_assertions": {
            "requires": ["discount_curve", "forward_curve", "fx_rates", "spot", "state_space"],
            "selected": {
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
                "fx_rate": "EURUSD",
                "state_space": "macro_regime",
            },
        },
    })

    assert market_context["selected_components"]["state_space"] == "macro_regime"
    assert market_context["selected_curve_names"]["discount_curve"] == "usd_ois"
    assert market_state.state_space is not None
    assert set(market_state.state_space.state_names) == {
        "base",
        "bull_repricing",
        "stress_repricing",
    }


def test_run_task_rejects_framework_task_with_structured_contract_error():
    from trellis.agent.task_runtime import run_task

    def fake_build(**kwargs):
        raise AssertionError("framework task should fail before build dispatch")

    result = run_task(
        {
            "id": "F01",
            "title": "Automatic model selection: agent picks best method per instrument",
            "construct": "framework",
            "new_component": "model_selection_agent",
            "market": {
                "source": "mock",
                "as_of": "2024-11-15",
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
                "fx_rate": "EURUSD",
                "state_space": "macro_regime",
            },
            "cross_validate": {
                "internal": ["agent_selected_method", "manual_best_method"],
                "external": ["quantlib"],
            },
        },
        market_state=object(),
        build_fn=fake_build,
    )

    assert result["success"] is False
    assert result["task_contract_error"]["code"] == "framework_task_not_priceable"
    assert "does not make sense as a pricing-task run" in result["error"]
    assert "framework or meta-development work" in result["task_contract_error"]["explanation"]
    assert "market context" in result["task_contract_error"]["explanation"]
    assert "framework-eval harness" in result["task_contract_error"]["suggestion"]


def test_build_market_state_for_task_rejects_market_assertion_mismatch():
    from trellis.agent.task_runtime import build_market_state_for_task

    with pytest.raises(ValueError, match="selected component mismatch"):
        build_market_state_for_task({
            "id": "E21",
            "title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)",
            "market": {
                "source": "mock",
                "as_of": "2024-11-15",
                "discount_curve": "usd_ois",
                "vol_surface": "usd_rates_smile",
                "underlier_spot": "SPX",
            },
            "market_assertions": {
                "selected": {"discount_curve": "eur_ois"},
            },
        })


def test_run_task_uses_task_specific_market_state():
    from trellis.agent.task_runtime import run_task

    seen_market_states: list[object] = []

    class FakeResult:
        success = True
        attempts = 1
        gap_confidence = 0.9
        knowledge_gaps = []
        payoff_cls = type("TaskSpecificPayoff", (), {})
        failures = []
        reflection = {}

    def fake_build(**kwargs):
        seen_market_states.append(kwargs["market_state"])
        return FakeResult()

    result = run_task(
        {
            "id": "E21",
            "title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)",
            "construct": "pde",
            "market": {
                "source": "mock",
                "as_of": "2024-11-15",
                "discount_curve": "usd_ois",
                "vol_surface": "usd_rates_smile",
                "underlier_spot": "SPX",
            },
            "market_assertions": {
                "requires": ["discount_curve", "black_vol_surface", "spot"],
                "selected": {
                    "discount_curve": "usd_ois",
                    "vol_surface": "usd_rates_smile",
                    "underlier_spot": "SPX",
                },
            },
        },
        market_state=object(),
        build_fn=fake_build,
    )

    assert result["success"] is True
    assert result["market_context"]["selected_components"] == {
        "discount_curve": "usd_ois",
        "vol_surface": "usd_rates_smile",
        "underlier_spot": "SPX",
    }
    assert len(seen_market_states) == 1
    assert getattr(seen_market_states[0], "spot", None) is not None


def test_prepare_existing_task_requires_resolved_instrument_type(monkeypatch):
    """Offline benchmarking should refuse tasks without a deterministic type."""
    from trellis.agent.task_runtime import prepare_existing_task

    with pytest.raises(ValueError, match="deterministic instrument type"):
        prepare_existing_task({"id": "T18", "title": "Log-space PDE for rate instruments"})


@dataclass(frozen=True)
class _PreparedTask:
    task_id: str
    title: str
    description: str
    instrument_type: str
    requirements: set[str]
    payoff_cls: type
    spec_schema: object


def test_prepare_existing_task_loads_cached_module(monkeypatch):
    """Offline preparation uses the fallback path when compile_build_request misses."""
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("CallableBondPayoff", (), {})
    fake_schema = object()

    # Simulate a registry miss — compile degrades to select_pricing_method
    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("no plan for instrument")),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.select_pricing_method",
        lambda description, instrument_type, model=None: SimpleNamespace(
            required_market_data={"discount_curve", "black_vol_surface"},
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.plan_build",
        lambda description, requirements, model="o3-mini", instrument_type=None, preferred_method=None: SimpleNamespace(
            spec_schema=fake_schema,
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._try_import_existing",
        lambda plan: FakePayoff,
    )

    prepared = prepare_existing_task(
        {"id": "T20", "title": "Callable bond with short call schedule"},
        model="test-model",
    )

    assert prepared.task_id == "T20"
    assert prepared.instrument_type == "callable_bond"
    assert prepared.payoff_cls is FakePayoff
    assert prepared.spec_schema is fake_schema
    assert prepared.requirements == {"discount_curve", "black_vol_surface"}
    assert prepared.compiled_request is None  # fallback path sets compiled=None


def test_prepare_existing_task_uses_quanto_family_compiled_requirements(monkeypatch):
    """Quanto offline preparation should reuse the family-compiled request surface."""
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("QuantoOptionAnalyticalPayoff", (), {})
    fake_schema = object()
    plan_calls: list[dict] = []

    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda description, instrument_type=None, market_snapshot=None, settlement=None, model=None, preferred_method=None, measures=None, metadata=None: SimpleNamespace(
            pricing_plan=SimpleNamespace(
                required_market_data={
                    "discount_curve",
                    "forward_curve",
                    "black_vol_surface",
                    "fx_rates",
                    "spot",
                    "model_parameters",
                },
                method="analytical",
            ),
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.plan_build",
        lambda description, requirements, model="o3-mini", instrument_type=None, preferred_method=None: plan_calls.append(
            {
                "description": description,
                "requirements": set(requirements),
                "instrument_type": instrument_type,
                "preferred_method": preferred_method,
            }
        ) or SimpleNamespace(spec_schema=fake_schema),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._try_import_existing",
        lambda plan: FakePayoff,
    )

    prepared = prepare_existing_task(
        {"id": "T105", "title": "Quanto option: quanto-adjusted BS vs MC cross-currency"},
        model="test-model",
    )

    assert prepared.task_id == "T105"
    assert prepared.instrument_type == "quanto_option"
    assert prepared.payoff_cls is FakePayoff
    assert prepared.spec_schema is fake_schema
    assert prepared.requirements == {
        "discount_curve",
        "forward_curve",
        "black_vol_surface",
        "fx_rates",
        "spot",
        "model_parameters",
    }
    assert plan_calls == [{
        "description": "Build a pricer for: Quanto option: quanto-adjusted BS vs MC cross-currency",
        "requirements": {
            "discount_curve",
            "forward_curve",
            "black_vol_surface",
            "fx_rates",
            "spot",
            "model_parameters",
        },
        "instrument_type": "quanto_option",
        "preferred_method": "analytical",
    }]


def test_prepare_existing_task_infers_schema_for_matching_generic_module(monkeypatch):
    """Generic cached modules should be benchmarkable when they clearly match the task."""
    from dataclasses import dataclass
    from datetime import date
    from types import ModuleType

    from trellis.agent.task_runtime import prepare_existing_task

    @dataclass(frozen=True)
    class FFTvsCOSSpec:
        s0: float
        strike: float
        expiry_date: date
        is_call: bool

    class FFTvsCOSPricer:
        def __init__(self, spec: FFTvsCOSSpec):
            self._spec = spec

        def evaluate(self, market_state):
            return 0.0

    module = ModuleType("trellis.instruments._agent.buildapayoff")
    module.__doc__ = "Agent-generated payoff: Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities."
    FFTvsCOSSpec.__module__ = module.__name__
    FFTvsCOSPricer.__module__ = module.__name__
    setattr(module, "FFTvsCOSSpec", FFTvsCOSSpec)
    setattr(module, "FFTvsCOSPricer", FFTvsCOSPricer)

    monkeypatch.setattr(
        "trellis.agent.task_runtime.select_pricing_method",
        lambda description, instrument_type, model=None: SimpleNamespace(
            required_market_data={"discount_curve", "black_vol_surface"},
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.plan_build",
        lambda description, requirements, model="o3-mini", instrument_type=None, preferred_method=None: SimpleNamespace(
            spec_schema=None,
            steps=[SimpleNamespace(module_path="instruments/_agent/buildapayoff.py")],
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._try_import_existing",
        lambda plan: None,
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.import_module",
        lambda module_name: module,
    )

    prepared = prepare_existing_task(
        {"id": "T39", "title": "FFT vs COS: GBM calls/puts across strikes and maturities"},
        model="test-model",
    )

    assert prepared.payoff_cls is FFTvsCOSPricer
    assert prepared.spec_schema.spec_name == "FFTvsCOSSpec"
    assert [field.name for field in prepared.spec_schema.fields] == [
        "s0",
        "strike",
        "expiry_date",
        "is_call",
    ]


def test_prepare_existing_task_rejects_mismatched_generic_module(monkeypatch):
    """Generic cached modules should not be reused for unrelated task titles."""
    from dataclasses import dataclass
    from datetime import date
    from types import ModuleType

    from trellis.agent.task_runtime import prepare_existing_task

    @dataclass(frozen=True)
    class FFTvsCOSSpec:
        s0: float
        strike: float
        expiry_date: date
        is_call: bool

    class FFTvsCOSPricer:
        def __init__(self, spec: FFTvsCOSSpec):
            self._spec = spec

        def evaluate(self, market_state):
            return 0.0

    module = ModuleType("trellis.instruments._agent.buildapayoff")
    module.__doc__ = "Agent-generated payoff: Build a pricer for: FFT vs COS: GBM calls/puts across strikes and maturities."
    FFTvsCOSSpec.__module__ = module.__name__
    FFTvsCOSPricer.__module__ = module.__name__
    setattr(module, "FFTvsCOSSpec", FFTvsCOSSpec)
    setattr(module, "FFTvsCOSPricer", FFTvsCOSPricer)

    monkeypatch.setattr(
        "trellis.agent.task_runtime.select_pricing_method",
        lambda description, instrument_type, model=None: SimpleNamespace(
            required_market_data={"discount_curve", "black_vol_surface"},
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.plan_build",
        lambda description, requirements, model="o3-mini", instrument_type=None, preferred_method=None: SimpleNamespace(
            spec_schema=None,
            steps=[SimpleNamespace(module_path="instruments/_agent/buildapayoff.py")],
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._try_import_existing",
        lambda plan: FFTvsCOSPricer,
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.import_module",
        lambda module_name: module,
    )

    with pytest.raises(FileNotFoundError, match="No cached agent module found"):
        prepare_existing_task(
            {"id": "T99", "title": "Chooser option: Rubinstein formula vs MC"},
            model="test-model",
        )


def test_benchmark_existing_task_uses_cached_payoff(monkeypatch):
    """Benchmarking should instantiate and price an existing payoff without rebuild."""
    from trellis.agent.task_runtime import PreparedTask, benchmark_existing_task

    FakePayoff = type("AsianOptionPayoff", (), {})
    fake_schema = object()
    fake_market_state = object()
    prepared = PreparedTask(
        task_id="T22",
        title="Asian option",
        description="Build a pricer for: Asian option",
        instrument_type="asian_option",
        requirements={"discount_curve", "black_vol_surface"},
        payoff_cls=FakePayoff,
        spec_schema=fake_schema,
    )

    monkeypatch.setattr(
        "trellis.agent.task_runtime.prepare_existing_task",
        lambda task, model="o3-mini": prepared,
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._make_test_payoff",
        lambda payoff_cls, spec_schema, settle: "payoff-instance",
    )

    observed: list[tuple[object, object]] = []
    timer_values = iter([0.0, 0.05, 1.0, 1.2, 2.0, 2.25, 3.0, 3.3])

    def fake_timer() -> float:
        return next(timer_values)

    def fake_price(payoff, market_state):
        observed.append((payoff, market_state))
        return 123.45

    result = benchmark_existing_task(
        {"id": "T22", "title": "Asian option"},
        market_state=fake_market_state,
        repeats=2,
        warmups=1,
        timer=fake_timer,
        price_fn=fake_price,
    )

    assert observed == [
        ("payoff-instance", fake_market_state),
        ("payoff-instance", fake_market_state),
        ("payoff-instance", fake_market_state),
    ]
    assert result["task_id"] == "T22"
    assert result["payoff_class"] == "AsianOptionPayoff"
    assert result["warmups"] == 1
    assert result["repeats"] == 2
    assert result["mean_seconds"] == pytest.approx(0.225)
    assert result["min_seconds"] == pytest.approx(0.2)
    assert result["max_seconds"] == pytest.approx(0.25)


def test_benchmark_existing_task_supports_generic_cached_transform_task():
    """A real cached generic transform task should benchmark without rebuild."""
    from trellis.agent.task_runtime import benchmark_existing_task, build_market_state

    result = benchmark_existing_task(
        {
            "id": "T39",
            "title": "FFT vs COS: GBM calls/puts across strikes and maturities",
        },
        market_state=build_market_state(),
        repeats=1,
        warmups=0,
    )

    assert result["task_id"] == "T39"
    assert result["instrument_type"] == "generic"
    assert result["payoff_class"] == "FFTvsCOSPricer"
    assert result["mean_seconds"] >= 0.0
    assert result["last_price"] > 0.0


def test_make_test_payoff_uses_basket_specific_string_defaults(monkeypatch):
    """Basket smoke tests should get valid observation/date defaults."""
    from trellis.agent.task_runtime import _make_test_payoff

    class BasketSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module_name = "dummy_basket_adapter"
    module = ModuleType(module_name)
    module.HimalayaBasketSpec = BasketSpec
    monkeypatch.setitem(sys.modules, module_name, module)

    BasketPayoff = type(
        "BasketPayoff",
        (),
        {
            "__init__": lambda self, spec: setattr(self, "spec", spec),
            "__module__": module_name,
        },
    )

    spec_schema = SimpleNamespace(
        spec_name="HimalayaBasketSpec",
        fields=[
            SimpleNamespace(name="underlyings", type="str", default=None),
            SimpleNamespace(name="observation_dates", type="str", default=None),
            SimpleNamespace(name="expiry_date", type="date", default=None),
            SimpleNamespace(name="start_date", type="date", default=None),
            SimpleNamespace(name="notional", type="float", default=None),
        ],
    )

    payoff = _make_test_payoff(BasketPayoff, spec_schema, date(2024, 11, 15))

    assert payoff.spec.kwargs["underlyings"] == "AAPL,MSFT,NVDA"
    assert payoff.spec.kwargs["observation_dates"] == "2026-04-01,2026-05-01,2026-06-01"
    assert payoff.spec.kwargs["start_date"] == date(2024, 11, 15)
    assert payoff.spec.kwargs["expiry_date"] == date(2025, 11, 15)


def test_make_test_payoff_uses_atm_strike_for_spot_based_options(monkeypatch):
    """Spot-based option smoke tests should default to an at-the-money strike."""
    from trellis.agent.task_runtime import _make_test_payoff

    class OptionSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module_name = "dummy_option_adapter"
    module = ModuleType(module_name)
    module.EuropeanOptionSpec = OptionSpec
    monkeypatch.setitem(sys.modules, module_name, module)

    OptionPayoff = type(
        "EuropeanOptionAnalyticalPayoff",
        (),
        {
            "__init__": lambda self, spec: setattr(self, "spec", spec),
            "__module__": module_name,
        },
    )

    spec_schema = SimpleNamespace(
        spec_name="EuropeanOptionSpec",
        fields=[
            SimpleNamespace(name="notional", type="float", default=None),
            SimpleNamespace(name="spot", type="float", default=None),
            SimpleNamespace(name="strike", type="float", default=None),
            SimpleNamespace(name="expiry_date", type="date", default=None),
            SimpleNamespace(name="option_type", type="str", default=None),
        ],
    )

    payoff = _make_test_payoff(OptionPayoff, spec_schema, date(2024, 11, 15))

    assert payoff.spec.kwargs["notional"] == 10.0
    assert payoff.spec.kwargs["spot"] == 100.0
    assert payoff.spec.kwargs["strike"] == 100.0
    assert payoff.spec.kwargs["option_type"] == "call"


# ---------------------------------------------------------------------------
# QUA-421: exception narrowing + compiled_request threading
# ---------------------------------------------------------------------------

def _make_fake_compiled(
    *,
    requires_clarification: bool = False,
    summary: str = "",
    required_market_data=None,
):
    """Build a minimal fake compiled-request object for test stubs."""
    plan = SimpleNamespace(
        required_market_data=required_market_data or {"discount_curve"},
        method="analytical",
    )
    gap = {
        "requires_clarification": requires_clarification,
        "summary": summary,
        "gap_types": ["missing_semantic_contract_field"] if requires_clarification else [],
    }
    metadata = {"semantic_gap": gap}
    request = SimpleNamespace(metadata=metadata)
    return SimpleNamespace(pricing_plan=plan, request=request, product_ir=None)


def _stub_plan_and_import(monkeypatch, FakePayoff, fake_schema):
    """Patch plan_build and _try_import_existing to return controlled fixtures."""
    monkeypatch.setattr(
        "trellis.agent.task_runtime.plan_build",
        lambda description, requirements, model="o3-mini", instrument_type=None, preferred_method=None: SimpleNamespace(
            spec_schema=fake_schema,
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._try_import_existing",
        lambda plan: FakePayoff,
    )


def test_prepare_existing_task_logs_warning_on_compile_failure(monkeypatch, caplog):
    """AttributeError from compile_build_request logs a WARNING and falls back."""
    import logging
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("EuropeanOptionAnalyticalPayoff", (), {})
    fake_schema = object()

    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda *a, **kw: (_ for _ in ()).throw(AttributeError("registry miss")),
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime.select_pricing_method",
        lambda *a, **kw: SimpleNamespace(required_market_data={"discount_curve"}, method="analytical"),
    )
    _stub_plan_and_import(monkeypatch, FakePayoff, fake_schema)

    with caplog.at_level(logging.WARNING, logger="trellis.agent.task_runtime"):
        prepared = prepare_existing_task(
            {"id": "T01", "title": "European call on AAPL"},
            model="test-model",
        )

    assert prepared.payoff_cls is FakePayoff
    assert prepared.compiled_request is None  # fallback path
    assert any("registry miss" in r.message for r in caplog.records)
    assert any("falling back to select_pricing_method" in r.message for r in caplog.records)


def test_prepare_existing_task_propagates_runtime_error(monkeypatch):
    """RuntimeError from compile_build_request must propagate — not be swallowed."""
    from trellis.agent.task_runtime import prepare_existing_task

    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("broken environment")),
    )

    with pytest.raises(RuntimeError, match="broken environment"):
        prepare_existing_task(
            {"id": "T01", "title": "European call on AAPL"},
            model="test-model",
        )


def test_prepare_existing_task_warns_on_ambiguous_description(monkeypatch, caplog):
    """requires_clarification=True in compiled metadata emits WARNING, cached payoff still returned."""
    import logging
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("EuropeanOptionAnalyticalPayoff", (), {})
    fake_schema = object()
    ambiguous_compiled = _make_fake_compiled(
        requires_clarification=True,
        summary="ambiguous: cds vs nth_to_default",
    )

    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda *a, **kw: ambiguous_compiled,
    )
    _stub_plan_and_import(monkeypatch, FakePayoff, fake_schema)

    with caplog.at_level(logging.WARNING, logger="trellis.agent.task_runtime"):
        prepared = prepare_existing_task(
            {"id": "T38", "title": "CDS pricing: hazard rate MC vs survival prob analytical"},
            model="test-model",
        )

    # Cached payoff is still returned — ambiguity is a warning, not a block
    assert prepared.payoff_cls is FakePayoff
    assert prepared.compiled_request is ambiguous_compiled
    # Warning must mention the ambiguity summary
    warning_messages = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
    assert "ambiguous" in warning_messages
    assert "T38" in warning_messages


def test_prepared_task_carries_compiled_request(monkeypatch):
    """PreparedTask.compiled_request is populated from compile_build_request result."""
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("EuropeanOptionAnalyticalPayoff", (), {})
    fake_schema = object()
    mock_compiled = _make_fake_compiled()

    monkeypatch.setattr(
        "trellis.agent.task_runtime.compile_build_request",
        lambda *a, **kw: mock_compiled,
    )
    _stub_plan_and_import(monkeypatch, FakePayoff, fake_schema)

    prepared = prepare_existing_task(
        {"id": "T01", "title": "European call on AAPL"},
        model="test-model",
    )

    assert prepared.compiled_request is mock_compiled
