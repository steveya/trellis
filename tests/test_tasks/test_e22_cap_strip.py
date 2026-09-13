from __future__ import annotations

from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest


def _task():
    from trellis.agent.task_manifests import load_task_manifest

    return next(
        t for t in load_task_manifest("TASKS_PROOF_LEGACY.yaml") if t["id"] == "E22"
    )


def test_e22_authored_contract_is_admitted_and_title_independent():
    from trellis.agent.task_manifest_validation import assert_executable_task_selection
    from trellis.agent.task_runtime import (
        benchmark_spec_overrides,
        task_to_semantic_contract,
    )

    task = _task()
    assert_executable_task_selection([task])
    renamed = {**task, "title": "Uninformative title"}
    assert_executable_task_selection([renamed])
    assert benchmark_spec_overrides(task) == benchmark_spec_overrides(renamed)
    original = task_to_semantic_contract(task)
    changed = task_to_semantic_contract(renamed)
    assert original.product == changed.product
    assert original.product.term_fields["n_paths"] == 100_000
    assert original.product.term_fields["model_time_day_count"] == "ACT/365"
    assert original.product.timeline.settlement_dates == tuple(
        task["benchmark_contract"]["payment_dates"]
    )


def test_e22_structured_bridge_and_method_rebuild_preserve_the_contract(monkeypatch):
    from trellis.agent.semantic_contracts import specialize_semantic_contract_for_method
    from trellis.agent.task_runtime import (
        _effective_task_description,
        task_to_semantic_contract,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Authored E22 must not be synthesized from prose")

    monkeypatch.setattr(
        "trellis.agent.semantic_contracts.draft_semantic_contract", forbidden
    )
    monkeypatch.setattr(
        "trellis.agent.task_runtime._bootstrap_rate_cap_floor_description", forbidden
    )
    task = _task()
    assert _effective_task_description(task) == _effective_task_description(
        {**task, "title": "Changed"}
    )
    contract = task_to_semantic_contract(task)
    rebuilt = specialize_semantic_contract_for_method(
        contract, preferred_method="monte_carlo"
    )
    assert dict(rebuilt.product.term_fields) == dict(contract.product.term_fields)
    assert rebuilt.product.timeline == contract.product.timeline


def test_e22_generated_schema_keeps_explicit_schedule_and_controls():
    from trellis.agent.executor import (
        _generate_skeleton,
        _hydrate_spec_schema_defaults_from_semantics,
    )
    from trellis.agent.planner import STATIC_SPECS
    from trellis.agent.task_runtime import task_to_semantic_contract

    schema = _hydrate_spec_schema_defaults_from_semantics(
        STATIC_SPECS["period_rate_option_strip"],
        semantic_contract=task_to_semantic_contract(_task()),
    )
    defaults = {field.name: field.default for field in schema.fields}
    assert defaults["n_paths"] == "100000"
    assert defaults["seed"] == "42"
    assert defaults["day_count"] == "DayCountConvention.ACT_360"
    assert defaults["frequency"] == "Frequency.QUARTERLY"
    assert "date(2025, 2, 15)" in defaults["fixing_dates"]
    assert "date(2030, 2, 15)" in defaults["payment_dates"]
    generated = _generate_skeleton(schema, "E22 authored cap")
    compile(generated, "<E22 skeleton>", "exec")


def test_cap_strip_smoke_fixture_does_not_invent_callable_or_collar_terms():
    from trellis.agent.executor import _generate_skeleton, _make_test_payoff
    from trellis.agent.planner import STATIC_SPECS

    schema = STATIC_SPECS["period_rate_option_strip"]
    namespace = {}
    exec(_generate_skeleton(schema, "Non-callable cap"), namespace)
    payoff = _make_test_payoff(namespace[schema.class_name], schema, date(2024, 11, 15))
    assert payoff._spec.call_price is None
    assert payoff._spec.exercise_dates is None
    assert payoff._spec.cap_strike is None
    assert payoff._spec.floor_strike is None


def test_e22_admits_materialized_market_and_rejects_market_override():
    from trellis.agent.market_scenarios import market_scenario_contract_from_task
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _task()
    scenario = market_scenario_contract_from_task(task)
    task["market"] = {
        "source": scenario.source,
        "as_of": scenario.as_of.isoformat(),
        **dict(scenario.selected_components),
        "scenario_contract": scenario.to_payload(),
        "scenario_digest": scenario.scenario_digest,
        "scenario_schema_version": scenario.schema_version,
        "scenario_constructor_kind": scenario.constructor_kind,
        "benchmark_inputs": scenario.financepy_inputs(),
    }
    assert_executable_task_selection([task])
    task["market"]["benchmark_inputs"]["black_vol"] = 0.9
    with pytest.raises(
        TaskManifestValidationError, match="legacy.cap_strip_invalid_contract"
    ):
        assert_executable_task_selection([task])


@pytest.mark.parametrize(
    "key", ["notional", "payment_dates", "fixing_rule", "model_time_day_count", "seed"]
)
def test_e22_rejects_missing_authored_terms(key):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _task()
    del task["benchmark_contract"][key]
    with pytest.raises(
        TaskManifestValidationError, match="legacy.cap_strip_invalid_contract"
    ):
        assert_executable_task_selection([task])


def test_e22_validates_referenced_scenario_at_the_requested_root(tmp_path):
    from pathlib import Path

    import yaml

    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    source = Path(__file__).resolve().parents[2] / "MARKET_SCENARIOS.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["scenarios"]["usd_rates_smile"]["constructor"]["black_vol"] = 0.3
    (tmp_path / "MARKET_SCENARIOS.yaml").write_text(yaml.safe_dump(payload))
    with pytest.raises(
        TaskManifestValidationError, match="legacy.cap_strip_invalid_contract"
    ):
        assert_executable_task_selection([_task()], root=tmp_path)


@pytest.mark.parametrize("field", ["seed", "simulation_seed"])
def test_e22_rejects_top_level_seed_aliases_that_conflict_with_authored_controls(field):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = {**_task(), field: 7}
    with pytest.raises(
        TaskManifestValidationError, match="legacy.cap_strip_invalid_contract"
    ):
        assert_executable_task_selection([task])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("benchmark_contract", "strike", 0.05),
        ("benchmark_contract", "n_paths", 20_000),
        ("benchmark_contract", "seed", 7),
        ("benchmark_contract", "n_steps", 64),
        ("benchmark_contract", "model_time_day_count", "ACT/360"),
        ("benchmark_contract", "business_day_adjustment", "following"),
        ("benchmark_contract", "fixing_dates", ["2025-02-17"]),
        ("cross_validate", "tolerance_pct", 5.0),
    ],
)
def test_e22_rejects_drift_and_unexercised_controls(section, key, value):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = deepcopy(_task())
    task[section][key] = value
    with pytest.raises(TaskManifestValidationError) as exc:
        assert_executable_task_selection([task])
    assert "legacy.cap_strip_invalid_contract" in str(exc.value)


def test_e22_generated_monte_carlo_exercises_authored_controls():
    from trellis.agent.executor import _deterministic_exact_binding_evaluate_body

    plan = SimpleNamespace(
        method="monte_carlo",
        instrument_type="period_rate_option_strip",
        lane_exact_binding_refs=(
            "trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo",
        ),
        backend_helper_refs=(),
        primitive_plan=None,
    )
    body = _deterministic_exact_binding_evaluate_body(
        plan, comparison_target="monte_carlo"
    )
    assert body
    captured = {}

    def record(**kwargs):
        captured.update(kwargs)
        return 123.0

    namespace = {"price_rate_cap_floor_strip_monte_carlo": record}
    exec(
        "def evaluate(self, market_state):\n"
        + "\n".join("    " + line for line in body.splitlines()),
        namespace,
    )
    spec = SimpleNamespace(n_paths=1234, seed=17)
    assert namespace["evaluate"](SimpleNamespace(_spec=spec), object()) == 123.0
    assert captured["n_paths"] == 1234
    assert captured["seed"] == 17


def test_e22_authored_black_caplets_match_seeded_forward_marginals():
    from trellis.agent.task_runtime import (
        benchmark_spec_overrides,
        build_market_state_for_task,
    )
    from trellis.models.rate_cap_floor import (
        price_rate_cap_floor_strip_analytical,
        price_rate_cap_floor_strip_monte_carlo,
    )
    from trellis.core.types import DayCountConvention

    task = _task()
    market, _ = build_market_state_for_task(task)
    assert market.discount.curve_day_count == DayCountConvention.ACT_ACT_ISDA
    assert (
        market.forecast_curves["USD-SOFR-3M"].curve_day_count
        == DayCountConvention.ACT_ACT_ISDA
    )
    overrides = benchmark_spec_overrides(task)
    assert len(overrides["accrual_dates"]) == 21
    assert overrides["fixing_dates"] == overrides["accrual_dates"][:-1]
    assert overrides["payment_dates"] == overrides["accrual_dates"][1:]
    assert overrides["start_date"] == date(2025, 2, 15)
    keys = (
        "notional",
        "strike",
        "start_date",
        "end_date",
        "frequency",
        "day_count",
        "rate_index",
        "accrual_dates",
        "fixing_dates",
        "calendar_name",
        "business_day_adjustment",
    )
    kwargs = {k: overrides[k] for k in keys}
    kwargs["coupon_dates"] = overrides["payment_dates"]
    analytical = price_rate_cap_floor_strip_analytical(market, **kwargs)
    sampled = price_rate_cap_floor_strip_monte_carlo(
        market, **kwargs, n_paths=overrides["n_paths"], seed=overrides["seed"]
    )
    assert analytical == pytest.approx(27531.179158915023, rel=1e-12)
    assert sampled == pytest.approx(27538.523575656935, rel=1e-12)
    assert (
        abs(sampled / analytical - 1.0) * 100 < task["cross_validate"]["tolerance_pct"]
    )
    assert sampled == price_rate_cap_floor_strip_monte_carlo(
        market, **kwargs, n_paths=overrides["n_paths"], seed=overrides["seed"]
    )


@pytest.mark.global_workflow
def test_e22_loaded_contract_prices_both_declared_lanes_end_to_end(
    monkeypatch, tmp_path
):
    """Defend the authored contract across compilation, hydration and comparison."""
    import sys
    from pathlib import Path

    import trellis.agent.analytical_traces as analytical_traces
    import trellis.agent.executor as executor
    import trellis.agent.model_audit as model_audit
    import trellis.agent.platform_requests as platform_requests
    import trellis.agent.platform_traces as platform_traces
    from trellis.agent.offline_agents import offline_local_agent_run_scope
    from trellis.agent.task_runtime import run_task
    from trellis.engine.payoff_pricer import price_payoff

    task = {**_task(), "title": "Uninformative label"}
    observed = []

    def observed_price(payoff, market):
        observed.append((payoff._spec, market.settlement))
        return price_payoff(payoff, market)

    def write_generated_module(module_path: str, code: str) -> Path:
        output_path = tmp_path / "generated" / module_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)
        return output_path

    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "TRELLIS_PACKAGE_ROOT", tmp_path / "trellis")
    monkeypatch.setattr(executor, "_REPO_REVISION", "test")
    monkeypatch.setattr(executor, "write_module", write_generated_module)
    monkeypatch.setattr(
        analytical_traces, "TRACE_ROOT", tmp_path / "traces" / "analytical"
    )
    monkeypatch.setattr(platform_traces, "TRACE_ROOT", tmp_path / "traces" / "platform")
    monkeypatch.setattr(model_audit, "_AUDIT_DIR", tmp_path / "audits")
    monkeypatch.setattr(
        platform_requests, "_record_semantic_extension_artifact", lambda *a, **kw: None
    )
    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "1")
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", "1")

    module_name = "trellis.instruments._agent._fresh.periodrateoptionstrip"
    previous_module = sys.modules.get(module_name)
    try:
        with offline_local_agent_run_scope():
            result = run_task(
                task,
                market_state=None,
                price_fn=observed_price,
                fresh_build=True,
                recovery_mode="strict",
                execution_mode_override="deterministic_replay",
                task_run_storage_root=tmp_path / "task-runs",
                task_run_storage_layout="standalone",
            )
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    assert result["success"] is True, result.get("failures")
    assert result["passed_expectation"] is True
    assert result["attempts"] == 0
    assert result["token_usage_summary"]["call_count"] == 0
    assert result["instrument_type"] == "cap"
    assert result["runtime_contract"]["simulation_seed"] == 42
    assert (
        result["runtime_contract"]["simulation_identity"]["seed_source"]
        == "task.benchmark_contract.seed"
    )
    comparison = result["cross_validation"]
    assert comparison["status"] == "passed"
    assert comparison["prices"] == pytest.approx(
        {
            "analytical": 27531.179158915023,
            "monte_carlo": 27538.523575656935,
        },
        rel=1e-12,
    )
    assert comparison["reference_target"] == "analytical"
    assert comparison["tolerance_pct"] == 0.5
    assert comparison["failed_targets"] == []
    for target in ("analytical", "monte_carlo"):
        binding = comparison["artifact_coherence"][target]
        assert binding["status"] == "bound_unique_artifact"
        assert (
            binding["selected_semantic_axes"]["payoff_family"]
            == "period_rate_option_strip"
        )
        acceptance = comparison["target_acceptance"][target]
        assert acceptance["output_currency"] == "USD"
        assert acceptance["output_unit"] == "currency_amount"
        assert acceptance["tolerance_unit"] == "percent_of_reference_price"
        assert acceptance["tolerance_pct"] == 0.5
    assert comparison["artifact_coherence"]["monte_carlo"][
        "exercised_spec_overrides"
    ] == {
        "n_paths": 100000,
        "seed": 42,
    }
    assert observed
    for spec, settlement in observed:
        assert settlement == date(2024, 11, 15)
        assert spec.notional == 1000000.0
        assert spec.strike == 0.04
        assert spec.n_paths == 100000
        assert spec.seed == 42
        assert len(spec.accrual_dates) == 21
        assert spec.fixing_dates == spec.accrual_dates[:-1]
        assert spec.payment_dates == spec.accrual_dates[1:]
