"""T89 defends a same-payoff duration identity, not an external price oracle."""

from copy import deepcopy
from dataclasses import replace
import math

import pytest

from trellis.agent.benchmark_contracts import benchmark_spec_overrides
from trellis.agent.task_manifests import load_task_manifest
from trellis.agent.task_runtime import build_market_state_for_task
from trellis.analytics.measures import Duration, OASDuration
from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec


def _task():
    return next(task for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml") if task["id"] == "T89")


@pytest.fixture
def isolated_task_artifacts(monkeypatch, tmp_path):
    import sys
    import trellis.agent.analytical_traces as analytical_traces
    import trellis.agent.executor as executor
    import trellis.agent.model_audit as model_audit
    import trellis.agent.platform_requests as platform_requests
    import trellis.agent.platform_traces as platform_traces

    def write_generated_module(module_path, code):
        output = tmp_path / "generated" / module_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(code)
        return output

    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "TRELLIS_PACKAGE_ROOT", tmp_path / "trellis")
    monkeypatch.setattr(executor, "_REPO_REVISION", "test")
    monkeypatch.setattr(executor, "write_module", write_generated_module)
    monkeypatch.setattr(analytical_traces, "TRACE_ROOT", tmp_path / "traces" / "analytical")
    monkeypatch.setattr(platform_traces, "TRACE_ROOT", tmp_path / "traces" / "platform")
    monkeypatch.setattr(model_audit, "_AUDIT_DIR", tmp_path / "audits")
    monkeypatch.setattr(platform_requests, "_record_semantic_extension_artifact", lambda *args, **kwargs: None)
    monkeypatch.setenv("TRELLIS_OFFLINE_LOCAL_AGENTS", "1")
    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")
    module_name = "trellis.instruments._agent._fresh.callablebond"
    previous = sys.modules.get(module_name)
    yield
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous


def test_t89_authors_duration_separately_from_holder_price_and_ignores_title():
    from trellis.agent.task_runtime import _effective_task_description, task_to_semantic_contract

    task = _task()
    assert task["proof_fixture_id"] == "usd_fixed_coupon_callable_bond_5pct_2025_2035_v1"
    assert task["market_scenario_id"] == "usd_callable_fixed_5pct_proof"
    comparison = task["cross_validate"]
    assert comparison["reference_target"] == "same_payoff_parallel_duration"
    assert comparison["output_tolerances_pct"] == {"effective_duration": 0.000001}
    assert comparison["output_unit"] == "currency_amount"
    assert comparison["analytics_contract"]["output_unit"] == "years"
    assert comparison["analytics_contract"]["bump_bps"] == 25.0
    assert comparison["analytics_contract"]["market_price"] is None
    assert comparison["analytics_contract"]["constant_oas_bps"] == 0.0
    renamed = {**task, "title": "Unrelated title containing neither bond nor duration"}
    assert _effective_task_description(task) == _effective_task_description(renamed)
    assert benchmark_spec_overrides(task) == benchmark_spec_overrides(renamed)
    assert task_to_semantic_contract(renamed).product.observation_schedule == (
        "2028-01-15", "2030-01-15", "2032-01-15",
    )


def test_duration_bridge_executes_authored_measures_on_dated_curve(monkeypatch):
    from trellis.agent.task_analytics import evaluate_comparison_analytics
    from trellis.curves.yield_curve import YieldCurve

    task = _task()
    market, _ = build_market_state_for_task(task)
    market = replace(market, forecast_curves={"independent_forecast": YieldCurve.flat(0.07)})
    observed_markets = []

    class ObservedPayoff(CallableBondPayoff):
        def evaluate(self, market_state):
            observed_markets.append(market_state)
            return super().evaluate(market_state)

    payoff = ObservedPayoff(CallableBondSpec(**benchmark_spec_overrides(task)))
    anchor = payoff.evaluate(market)
    up = payoff.evaluate(replace(market, discount=market.discount.shift(25.0), forward_curve=None))
    down = payoff.evaluate(replace(market, discount=market.discount.shift(-25.0), forward_curve=None))
    expected = (down - up) / (2 * 0.0025 * anchor)
    observed_markets.clear()
    def forbidden_oas_solve(*args, **kwargs):
        raise AssertionError("constant-zero-OAS proof must not solve a market-price OAS")
    monkeypatch.setattr("trellis.analytics.measures._callable_market_price_oas_bps", forbidden_oas_solve)
    calls = []
    for measure_cls in (Duration, OASDuration):
        original = measure_cls.compute
        def record(self, actual_payoff, actual_market, _original=original, **ctx):
            calls.append((type(self).__name__, self.bump_bps, actual_payoff, actual_market, ctx["base_price"]))
            return _original(self, actual_payoff, actual_market, **ctx)
        monkeypatch.setattr(measure_cls, "compute", record)
    for target in task["cross_validate"]["internal"]:
        output = evaluate_comparison_analytics(
            payoff, market, target_id=target,
            contract=task["cross_validate"]["analytics_contract"], anchor_price=anchor,
        )["effective_duration"]
        assert output["value"] == pytest.approx(expected, rel=1e-12)
        assert output["unit"] == "years"
        assert output["metadata"]["derivative_method"] == "finite_difference"
        assert output["metadata"]["resolved_derivative_method"] == "parallel_curve_bump"
        assert output["metadata"]["anchor_price"] == pytest.approx(anchor)
    assert [(name, bump) for name, bump, *_ in calls] == [("OASDuration", 25.0), ("Duration", 25.0)]
    assert all(actual is payoff and ms is market and base == anchor for _, _, actual, ms, base in calls)
    assert expected == pytest.approx(5.533555119264439, abs=1e-9)
    assert len(observed_markets) == 4
    assert sorted(ms.discount.flat_rate for ms in observed_markets) == pytest.approx([0.0475, 0.0475, 0.0525, 0.0525])
    for shifted in observed_markets:
        assert shifted.forecast_curves == market.forecast_curves
        assert shifted.model_parameters == market.model_parameters
        assert shifted.model_parameter_sets == market.model_parameter_sets
        assert shifted.selected_curve_names == market.selected_curve_names
        assert shifted.discount.value_date == market.discount.value_date
        assert shifted.discount.curve_day_count is market.discount.curve_day_count
        assert shifted.forward_curve.discount_date(payoff.spec.end_date) == pytest.approx(
            shifted.discount.discount_date(payoff.spec.end_date),
        )


@pytest.mark.parametrize("mutation", [
    lambda task: task["cross_validate"]["analytics_contract"].pop("output_unit"),
    lambda task: task["cross_validate"]["analytics_contract"].update(output_unit="percent"),
    lambda task: task["cross_validate"]["analytics_contract"].update(bump_bps=1.0),
    lambda task: task["cross_validate"]["analytics_contract"].update(market_price=100.0),
    lambda task: task["cross_validate"]["analytics_contract"].update(constant_oas_bps=10.0),
    lambda task: task["cross_validate"].pop("output_tolerances_pct"),
    lambda task: task["cross_validate"].update(reference_target="modified_duration"),
    lambda task: task["benchmark_contract"].update(call_dates=["2028-01-15"]),
])
def test_t89_rejects_contract_drift(mutation):
    from trellis.agent.task_manifest_validation import _validate_legacy_task

    task = deepcopy(_task())
    mutation(task)
    issues = _validate_legacy_task("TASKS_PROOF_LEGACY.yaml", task, "tasks[0]")
    assert any(issue.code == "legacy.callable_bond_invalid_contract" for issue in issues)


@pytest.mark.global_workflow
def test_t89_fresh_offline_run_requires_duration_and_authored_tree_controls(monkeypatch, tmp_path, isolated_task_artifacts):
    from trellis.agent.task_runtime import run_task
    from trellis.models.trees import lattice

    monkeypatch.setenv("TRELLIS_OFFLINE_LOCAL_AGENTS", "1")
    observed = []
    original = lattice.build_generic_lattice

    def observe(*args, **kwargs):
        observed.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(lattice, "build_generic_lattice", observe)
    task = {**_task(), "title": "Unrelated renamed proof"}
    result = run_task(task, None, fresh_build=True, task_run_storage_root=tmp_path)
    assert result["success"], result
    report = result["cross_validation"]
    assert report["status"] == "passed"
    assert report["output_validation"]["effective_duration"]["output_unit"] == "years"
    for target in task["cross_validate"]["internal"]:
        assert report["outputs"][target]["price"] == pytest.approx(96.8252972854, abs=1e-7)
        assert report["outputs"][target]["effective_duration"] == pytest.approx(5.533555119264439, abs=1e-8)
        assert report["output_metadata"][target]["effective_duration"]["bump_bps"] == 25.0
        assert report["target_acceptance"][target]["output_acceptance"]["effective_duration"]["output_unit"] == "years"
        assert result["method_results"][target]["generated_artifact"]["is_fresh_build"]
    assert observed
    assert all(call["n_steps"] == 200 for call in observed)
    assert all(call["a"] == 0.1 and call["sigma"] == 0.01 for call in observed)
    observed_rates = {
        round(-math.log(float(call["discount_curve"].discount(1.0))), 4)
        for call in observed
    }
    assert {0.0475, 0.05, 0.0525} <= observed_rates


@pytest.mark.parametrize("value, metadata", [
    (True, {}), (float("inf"), {}), (float("nan"), {}),
    (5.5, {"status": "failed"}), (5.5, {"output_unit": "days"}),
])
def test_duration_bridge_rejects_raw_bad_measure_before_float_conversion(monkeypatch, value, metadata):
    from trellis.agent.task_analytics import evaluate_comparison_analytics
    from trellis.analytics.result import ScalarRiskMeasureOutput

    raw = ScalarRiskMeasureOutput(value, metadata=metadata) if metadata else value
    monkeypatch.setattr(OASDuration, "compute", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError):
        evaluate_comparison_analytics(
            object(), object(), target_id="oas_bump_duration",
            contract=_task()["cross_validate"]["analytics_contract"], anchor_price=96.0,
        )


@pytest.mark.global_workflow
@pytest.mark.parametrize("mode", ["mismatch", "failed_status", "missing", "exception"])
def test_t89_real_run_fails_duration_defect_while_prices_agree(monkeypatch, tmp_path, isolated_task_artifacts, mode):
    from trellis.agent.task_runtime import run_task
    from trellis.analytics.result import ScalarRiskMeasureOutput

    def bad_measure(*args, **kwargs):
        if mode == "exception":
            raise ValueError("injected duration failure")
        if mode == "missing":
            return None
        if mode == "failed_status":
            return ScalarRiskMeasureOutput(5.533555119264439, metadata={"status": "failed"})
        return 6.0

    monkeypatch.setattr(OASDuration, "compute", bad_measure)
    result = run_task(_task(), None, fresh_build=True, task_run_storage_root=tmp_path)
    assert result["success"] is False
    assert result["passed_expectation"] is False
    report = result["cross_validation"]
    assert len(set(report["prices"].values())) == 1
    assert report["status"] == "failed"
    assert report["output_validation"]["effective_duration"]["status"] != "passed"
    assert all(method["success"] for method in result["method_results"].values())


@pytest.mark.parametrize("task_id", ["T89", " T89 "])
@pytest.mark.parametrize("field", ["analytics_contract", "output_tolerances_pct"])
@pytest.mark.parametrize("provenance", ["authored", "removed", "spoofed"])
def test_t89_direct_run_rejects_missing_duration_requirement_before_build(
    field, task_id, provenance, monkeypatch, tmp_path,
):
    from trellis.agent.task_runtime import run_task

    task = deepcopy(_task())
    task["id"] = task_id
    task["cross_validate"].pop(field)
    if provenance == "removed":
        task.pop("task_corpus", None)
        task.pop("task_definition_manifest", None)
    elif provenance == "spoofed":
        task.update(task_corpus="extension", task_definition_manifest="TASKS_EXTENSION.yaml")
    calls = []
    def forbidden_market(*args, **kwargs):
        calls.append("market")
        raise AssertionError("invalid T89 must fail before market construction")
    monkeypatch.setattr("trellis.agent.task_runtime.build_market_state_for_task", forbidden_market)
    result = run_task(task, None, build_fn=lambda **kwargs: calls.append(kwargs), task_run_storage_root=tmp_path)
    assert result["success"] is False
    assert not calls
    assert "legacy.callable_bond_invalid_contract" in result["error"]


@pytest.mark.parametrize("task_id", ["T89", " T89 "])
@pytest.mark.parametrize("removed_field", [None, "analytics_contract", "output_tolerances_pct"])
def test_t89_cannot_bypass_pricing_contract_through_fpml_dispatch(
    task_id, removed_field, monkeypatch, tmp_path,
):
    from trellis.agent.task_runtime import run_task

    task = deepcopy(_task())
    task.update(id=task_id, task_kind="fpml_conformance")
    if removed_field:
        task["cross_validate"].pop(removed_field)
    calls = []
    def forbidden_market(*args, **kwargs):
        calls.append("market")
        raise AssertionError("reserved T89 must not reach FpML market construction")
    monkeypatch.setattr("trellis.agent.task_runtime.build_market_state_for_task", forbidden_market)
    result = run_task(task, None, build_fn=lambda **kwargs: calls.append(kwargs), task_run_storage_root=tmp_path)
    assert result["success"] is False
    assert not calls
    assert "legacy.callable_bond_invalid_contract" in result["error"]


@pytest.mark.parametrize("mutation", [
    lambda output: output.clear(),
    lambda output: output["effective_duration"].update(value=float("inf")),
    lambda output: output["effective_duration"].update(value=float("nan")),
    lambda output: output["effective_duration"].update(value=True),
    lambda output: output["effective_duration"].update(value=6.0),
    lambda output: output["effective_duration"].update(value=5.50000011),
    lambda output: output["effective_duration"].update(unit="days"),
    lambda output: output["effective_duration"].update(status="failed"),
    lambda output: output["effective_duration"]["metadata"].update(bump_bps=1.0),
    lambda output: output["effective_duration"]["metadata"].update(unit="days"),
    lambda output: output["effective_duration"]["metadata"].update(output_unit="days"),
    lambda output: output["effective_duration"]["metadata"].update(status="failed"),
    lambda output: output["effective_duration"]["metadata"].update(derivative_method_category="autograd"),
])
def test_duration_comparison_fails_bad_output_even_when_prices_agree(monkeypatch, mutation):
    from types import SimpleNamespace
    from trellis.agent import task_analytics
    from trellis.agent.task_runtime import _cross_validate_comparison_task, _task_comparison_targets

    task = _task()
    targets = _task_comparison_targets(task, ["rate_tree"])
    market, _ = build_market_state_for_task(task)
    payoff = CallableBondPayoff(CallableBondSpec(**benchmark_spec_overrides(task)))
    # A reported native price cannot replace the actual bound-payoff anchor.
    payoff.benchmark_outputs = lambda _: {"price": 1.0, "effective_duration": 5.5}
    live = {target.target_id: SimpleNamespace(success=True, payoff_cls=type(payoff)) for target in targets}
    monkeypatch.setattr("trellis.agent.task_runtime._comparison_target_binding_report", lambda target, _: {
        "status": "bound_unique_artifact", "artifact_identity": target.target_id, "failures": [],
    })
    def outputs(actual_payoff, actual_market, *, target_id, contract, anchor_price):
        output = {"effective_duration": {
            "value": 5.5, "unit": "years", "status": "passed", "metadata": {
                "measure": contract["measures"][target_id], "bump_bps": 25.0,
                "derivative_method": "finite_difference", "resolved_derivative_method": "parallel_curve_bump",
                "market_price": None, "constant_oas_bps": 0.0, "anchor_price": anchor_price,
                "denominator": contract["denominator"], "risk_coordinate": contract["risk_coordinate"],
                "reference_role": contract["reference_role"],
            },
        }}
        # Mutate the reference as well: +inf must not make the tolerance infinite.
        if target_id == "same_payoff_parallel_duration":
            mutation(output)
        return output
    monkeypatch.setattr(task_analytics, "evaluate_comparison_analytics", outputs)
    report = _cross_validate_comparison_task(
        targets, live, market, configured_targets=task["cross_validate"],
        payoff_factory=lambda *args: payoff, price_fn=lambda *args: 96.0,
    )
    assert report["prices"] == {target.target_id: 96.0 for target in targets}
    assert report["status"] == "failed"
    assert report["output_validation"]["effective_duration"]["status"] != "passed"
    if report["output_validation"]["effective_duration"]["status"] == "failed":
        assert report["output_validation"]["effective_duration"]["deviations_pct"]["oas_bump_duration"] > 0.000001


@pytest.mark.parametrize("value, metadata", [
    (True, {}), (float("inf"), {}), (float("nan"), {}), (0.0, {}),
    (96.0, {"status": "failed"}), (96.0, {"output_unit": "years"}),
])
def test_t89_rejects_invalid_evaluator_anchor_before_analytics(monkeypatch, value, metadata):
    from types import SimpleNamespace
    from trellis.agent import task_analytics
    from trellis.agent.task_runtime import _cross_validate_comparison_task, _task_comparison_targets
    from trellis.analytics.result import ScalarRiskMeasureOutput

    task = _task()
    targets = _task_comparison_targets(task, ["rate_tree"])
    market, _ = build_market_state_for_task(task)
    payoff = CallableBondPayoff(CallableBondSpec(**benchmark_spec_overrides(task)))
    live = {target.target_id: SimpleNamespace(success=True, payoff_cls=type(payoff)) for target in targets}
    monkeypatch.setattr("trellis.agent.task_runtime._comparison_target_binding_report", lambda target, _: {
        "status": "bound_unique_artifact", "artifact_identity": target.target_id, "failures": [],
    })
    analytics_calls = []
    monkeypatch.setattr(task_analytics, "evaluate_comparison_analytics", lambda *args, **kwargs: analytics_calls.append(kwargs))
    raw = ScalarRiskMeasureOutput(value, metadata=metadata) if metadata else value
    report = _cross_validate_comparison_task(
        targets, live, market, configured_targets=task["cross_validate"],
        payoff_factory=lambda *args: payoff, price_fn=lambda *args: raw,
    )
    assert not analytics_calls
    assert not report["prices"]
    assert report["status"] != "passed"
