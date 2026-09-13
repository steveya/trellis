"""T73: authored European swaption comparison proof.

The proof compares three Trellis-native lanes over one exact contract and one
named market regime. External-library parity is intentionally outside this
bounded legacy proof.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import (
    price_swaption_black76,
    price_swaption_monte_carlo,
    resolve_swaption_monte_carlo_problem,
)
from trellis.models.rate_style_swaption_tree import price_swaption_tree
from trellis.models.vol_surface import FlatVol


VALUATION_DATE = date(2024, 11, 15)
MEAN_REVERSION = 0.05
SIGMA = 0.01


def _authored_t73_task():
    from trellis.agent.task_manifests import load_task_manifest

    return deepcopy(
        next(
            task
            for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
            if task["id"] == "T73"
        )
    )


def test_t73_authors_exercise_value_without_contractual_settlement_claims():
    from trellis.agent.task_runtime import benchmark_spec_overrides, task_to_semantic_contract

    task = _authored_t73_task()
    terms = task["benchmark_contract"]
    assert terms["exercise_value_convention"] == "positive_payer_underlying_swap_npv"
    assert "settlement_type" not in terms
    assert "settlement_timing" not in terms
    overrides = benchmark_spec_overrides(task)
    assert overrides["exercise_value_convention"] == terms["exercise_value_convention"]
    assert "settlement_type" not in overrides
    product = task_to_semantic_contract(task).product
    assert product.settlement_rule == product.maturity_settlement_rule == "exercise_value_only"
    assert product.obligations[0].settlement_kind == "valuation"
    assert product.obligations[0].amount_expression == terms["exercise_value_convention"]


@pytest.mark.parametrize(
    "field,value",
    [("settlement_type", "cash"), ("settlement_type", "physical"),
     ("settlement_timing", "exercise_date"), ("settlement_method", "par_yield_cash_annuity")],
)
def test_t73_rejects_contractual_settlement_declarations(field, value):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError, assert_executable_task_selection,
    )

    task = _authored_t73_task()
    task["benchmark_contract"][field] = value
    with pytest.raises(TaskManifestValidationError, match="legacy.swaption_invalid_contract"):
        assert_executable_task_selection([task])


@pytest.mark.parametrize("field", ("seed", "simulation_seed"))
def test_t73_rejects_task_level_seed_aliases(field):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _authored_t73_task()
    task[field] = 7

    with pytest.raises(
        TaskManifestValidationError, match="legacy.swaption_invalid_contract"
    ):
        assert_executable_task_selection([task])


@pytest.mark.parametrize("renamed", (False, True))
@pytest.mark.parametrize("conflicting_defaults", (False, True))
def test_t73_simulation_seed_comes_from_executed_target(renamed, conflicting_defaults):
    from trellis.agent.task_runtime import _explicit_simulation_seed

    task = _authored_t73_task()
    if renamed:
        task["title"] = "Uninformative label"
    market_context = {}
    if conflicting_defaults:
        # Even a direct runtime caller that bypasses manifest admission cannot
        # report these generic defaults as the seed of the authored MC target.
        task.update(seed=7, simulation_seed=8)
        market_context = {"metadata": {"seed": 9, "simulation_seed": 10}}

    seed, source = _explicit_simulation_seed(task, market_context)

    assert (
        seed
        == task["cross_validate"]["target_contracts"]["hw_mc"]["spec_overrides"]["seed"]
    )
    assert seed == 42
    assert source == "task.cross_validate.target_contracts.hw_mc.spec_overrides.seed"


def test_t73_seed_resolution_matches_normalized_manifest_id():
    from trellis.agent.task_manifest_validation import assert_executable_task_selection
    from trellis.agent.task_runtime import _explicit_simulation_seed

    task = _authored_t73_task()
    task["id"] = " T73 "
    assert_executable_task_selection([task])

    assert _explicit_simulation_seed(task, {}) == (
        42,
        "task.cross_validate.target_contracts.hw_mc.spec_overrides.seed",
    )


@pytest.mark.parametrize(
    ("task", "market_context", "expected"),
    (
        ({"simulation_seed": 7, "seed": 8}, {}, (7, "task.simulation_seed")),
        (
            {"benchmark_contract": {"seed": 42}},
            {},
            (42, "task.benchmark_contract.seed"),
        ),
        (
            {"extension_contract": {"seed": 42}},
            {},
            (42, "task.extension_contract.seed"),
        ),
        ({}, {"metadata": {"seed": 9}}, (9, "market.metadata.seed")),
        ({}, {}, (None, "")),
    ),
)
def test_t73_seed_resolution_preserves_other_task_defaults(
    task, market_context, expected
):
    from trellis.agent.task_runtime import _explicit_simulation_seed

    assert _explicit_simulation_seed(task, market_context) == expected


@dataclass(frozen=True)
class _T73SwaptionSpec:
    notional: float = 1_000_000.0
    strike: float = 0.03
    expiry_date: date = date(2025, 11, 15)
    swap_start: date = date(2025, 11, 15)
    swap_end: date = date(2030, 11, 15)
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    float_frequency: Frequency = Frequency.QUARTERLY
    float_day_count: DayCountConvention = DayCountConvention.ACT_360
    model_time_day_count: DayCountConvention = DayCountConvention.THIRTY_360
    rate_index: str = "USD-SOFR-3M"
    is_payer: bool = True
    tree_steps: int = 120
    n_paths: int = 20_000
    n_steps: int = 64
    seed: int = 42


def _t73_market_state() -> MarketState:
    return MarketState(
        as_of=VALUATION_DATE,
        settlement=VALUATION_DATE,
        discount=YieldCurve.flat(0.04, max_tenor=10.0),
        forecast_curves={
            "USD-SOFR-3M": YieldCurve.flat(0.0425, max_tenor=10.0),
        },
        vol_surface=FlatVol(0.20),
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        },
        model_parameters={
            "family": "hull_white_1f",
            "mean_reversion": MEAN_REVERSION,
            "sigma": SIGMA,
        },
    )


def test_t73_uses_distinct_fixed_and_floating_leg_timelines():
    spec = _T73SwaptionSpec()
    problem = resolve_swaption_monte_carlo_problem(
        _t73_market_state(),
        spec,
        n_steps=spec.n_steps,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
    )

    assert problem.event_timeline is not None
    settlement = next(
        event
        for event in problem.event_timeline.events
        if event.name == "swaption_settlement"
    )
    assert len(settlement.payload["payment_times"]) == 10
    assert len(settlement.payload["floating_payment_times"]) == 20
    assert len(settlement.payload["floating_accrual_fractions"]) == 20


def test_t73_black_tree_and_seeded_monte_carlo_meet_authored_tolerances():
    market_state = _t73_market_state()
    spec = _T73SwaptionSpec()

    black_price = price_swaption_black76(
        market_state,
        spec,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
    )
    tree_price = price_swaption_tree(
        market_state,
        spec,
        model="hull_white",
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
        n_steps=spec.tree_steps,
    )
    monte_carlo_price = price_swaption_monte_carlo(
        market_state,
        spec,
        mean_reversion=MEAN_REVERSION,
        sigma=SIGMA,
        n_paths=spec.n_paths,
        n_steps=spec.n_steps,
        seed=spec.seed,
    )

    assert black_price > 0.0
    assert tree_price > 0.0
    assert monte_carlo_price > 0.0
    assert black_price == pytest.approx(tree_price, rel=0.001)
    assert monte_carlo_price == pytest.approx(tree_price, rel=0.03)


@pytest.mark.global_workflow
def test_t73_loaded_contract_prices_declared_lanes_and_records_executed_seed(
    monkeypatch, tmp_path
):
    """Exercise real compilation, spec hydration, pricing and replay metadata."""
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

    task = {**_authored_t73_task(), "title": "Uninformative label"}
    observed_specs = []

    def observed_price(payoff, market):
        observed_specs.append(payoff._spec)
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

    module_name = "trellis.instruments._agent._fresh.swaption"
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

    runtime_contract = result["runtime_contract"]
    identity = runtime_contract["simulation_identity"]
    assert runtime_contract["simulation_seed"] == identity["seed"] == 42
    assert identity["seed_source"] == (
        "task.cross_validate.target_contracts.hw_mc.spec_overrides.seed"
    )
    assert result["success"] is True, result.get("failures")
    assert result["passed_expectation"] is True
    assert result["attempts"] == 0
    assert result["token_usage_summary"]["call_count"] == 0
    comparison = result["cross_validation"]
    assert comparison["status"] == "passed"
    assert comparison["reference_target"] == "hw_tree"
    assert comparison["tolerance_pct"] is None
    assert comparison["target_acceptance"]["black76"]["tolerance_pct"] == 0.1
    assert comparison["target_acceptance"]["hw_mc"]["tolerance_pct"] == 3.0
    assert comparison["failed_targets"] == []
    assert comparison["prices"] == pytest.approx(
        {
            "black76": 56963.09506754951,
            "hw_tree": 56963.095066215574,
            "hw_mc": 56299.43142852026,
        },
        rel=1e-10,
    )
    for target in ("black76", "hw_tree", "hw_mc"):
        assert (
            comparison["artifact_coherence"][target]["status"]
            == "bound_unique_artifact"
        )
    assert comparison["artifact_coherence"]["hw_mc"]["exercised_spec_overrides"] == {
        "n_paths": 20_000,
        "n_steps": 64,
        "seed": identity["seed"],
    }
    mc_specs = [spec for spec in observed_specs if hasattr(spec, "n_paths")]
    assert mc_specs
    assert all(spec.seed == identity["seed"] for spec in mc_specs)
    assert all(spec.n_paths == 20_000 and spec.n_steps == 64 for spec in mc_specs)
