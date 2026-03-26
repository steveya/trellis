"""Tests for task runtime helpers used by task rerun and benchmarking scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import pytest


def test_run_task_passes_force_rebuild_and_validation():
    """run_task should forward execution mode into the knowledge-aware builder."""
    from trellis.agent.task_runtime import run_task

    calls: list[dict] = []
    fake_market_state = object()

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

    result = run_task(
        {"id": "T13", "title": "European call: theta-method convergence order"},
        market_state=fake_market_state,
        model="test-model",
        force_rebuild=False,
        validation="fast",
        build_fn=fake_build,
        timer=fake_timer,
        now_fn=lambda: datetime(2026, 3, 24, 12, 0, 0),
    )

    assert calls == [{
        "description": "Build a pricer for: European call: theta-method convergence order",
        "instrument_type": "european_option",
        "request_metadata": {
            "task_id": "T13",
            "task_title": "European call: theta-method convergence order",
        },
        "model": "test-model",
        "market_state": fake_market_state,
        "max_retries": 3,
        "validation": "fast",
        "force_rebuild": False,
    }]
    assert result["success"] is True
    assert result["elapsed_seconds"] == 4.5
    assert result["payoff_class"] == "FakePayoff"


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

    def fake_build(**kwargs):
        return FakeResult()

    def fake_persist(task, result):
        persisted["task"] = task
        persisted["result"] = dict(result)
        return {
            "history_path": "/tmp/task_runs/history/T13/run.json",
            "latest_path": "/tmp/task_runs/latest/T13.json",
            "latest_index_path": "/tmp/task_results_latest.json",
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

    assert calls == [{
        "description": "Build a pricer for: European call: theta-method convergence order",
        "instrument_type": "european_option",
        "preferred_method": "pde_solver",
        "request_metadata": {
            "task_id": "T13",
            "task_title": "European call: theta-method convergence order",
            "preferred_method": "pde_solver",
        },
        "model": "gpt-5-mini",
        "market_state": calls[0]["market_state"],
        "max_retries": 3,
        "validation": "standard",
        "force_rebuild": True,
    }]
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
    assert calls[0]["request_metadata"] == {
        "task_id": "T74",
        "task_title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)",
        "comparison_target": "crr_tree",
        "preferred_method": "rate_tree",
    }
    assert calls[-1]["request_metadata"] == {
        "task_id": "T74",
        "task_title": "European equity call: 5-way (tree, PDE, MC, FFT, COS)",
        "comparison_target": "black_scholes",
        "preferred_method": "analytical",
    }
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
    assert result["agent_observation_count"] == 3
    assert result["method_results"]["fft"]["artifacts"]["platform_request_ids"] == [
        "executor_build_fft"
    ]
    assert result["method_results"]["fft"]["artifacts"]["knowledge_trace_paths"] == [
        "/tmp/fft_knowledge.yaml"
    ]
    assert result["method_results"]["fft"]["knowledge_summary"]["lesson_ids"] == ["fft_lesson"]


def test_build_result_payload_includes_blocker_details():
    from trellis.agent.task_runtime import _build_result_payload

    class FakeResult:
        success = False
        attempts = 1
        gap_confidence = 0.2
        knowledge_gaps = ["exercise control missing"]
        payoff_cls = None
        failures = ["blocked"]
        agent_observations = [{"agent": "critic", "summary": "double discounting"}]
        knowledge_summary = {"lesson_ids": ["mc_007"], "cookbook_method": "monte_carlo"}
        platform_trace_path = "/tmp/blocked_trace.yaml"
        platform_request_id = "executor_build_blocked"
        blocker_details = {
            "blocker_codes": ["missing_symbol:demo"],
            "new_primitive_workflow": {"summary": "library_repair"},
        }
        reflection = {}

    payload = _build_result_payload(FakeResult(), preferred_method="analytical")

    assert payload["blocker_details"]["blocker_codes"] == ["missing_symbol:demo"]
    assert payload["agent_observation_count"] == 1
    assert payload["agent_observations"][0]["agent"] == "critic"
    assert payload["knowledge_summary"]["lesson_ids"] == ["mc_007"]
    assert payload["artifacts"]["platform_trace_paths"] == ["/tmp/blocked_trace.yaml"]


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
    assert set(market_state.forecast_curves) == {"USD-SOFR-3M"}
    assert market_state.discount is not None
    assert market_state.vol_surface is not None
    assert market_state.forward_curve is not None
    assert market_state.forward_curve.forward_rate(0.5, 1.0) == pytest.approx(
        market_state.forecast_forward_curve("USD-SOFR-3M").forward_rate(0.5, 1.0)
    )


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
    """Offline preparation should use the cached agent module and never rebuild."""
    from trellis.agent.task_runtime import prepare_existing_task

    FakePayoff = type("CallableBondPayoff", (), {})
    fake_schema = object()

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
