from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_load_market_scenario_contracts_exposes_schema_v2_and_digest():
    from trellis.agent.market_scenarios import load_market_scenario_contracts

    contracts = load_market_scenario_contracts(root=ROOT)
    contract = contracts["flat_usd_equity_vanilla"]

    assert contract.schema_version == 2
    assert contract.constructor_kind == "single_asset_equity"
    assert contract.scenario_digest
    assert contract.financepy_inputs()["stock_price"] == 100.0
    assert contract.financepy_inputs()["domestic_rate"] == 0.05


def test_construct_market_state_for_scenario_populates_multi_asset_support():
    from trellis.agent.market_scenarios import (
        construct_market_state_for_scenario,
        load_market_scenario_contracts,
    )
    from trellis.agent.task_runtime import build_market_state

    contract = load_market_scenario_contracts(root=ROOT)["equity_rainbow_two_asset"]
    base_market_state = build_market_state()
    market_state, metadata = construct_market_state_for_scenario(
        contract,
        base_market_state,
        task_id="F008",
    )

    assert metadata["market_scenario_construction"] is True
    assert metadata["scenario_construction_kind"] == "multi_asset_equity"
    assert market_state.underlier_spots["AAPL"] == 100.0
    assert market_state.underlier_spots["MSFT"] == 95.0
    assert market_state.forecast_curves["AAPL-DISC"].zero_rate(1.0) == 0.0
    assert market_state.model_parameters["underlier_vols"]["AAPL"] == 0.2
    assert market_state.model_parameters["underlier_vols"]["MSFT"] == 0.25
    assert market_state.model_parameters["correlation_source"]["kind"] == "explicit"
    assert market_state.market_provenance["market_scenario"]["scenario_id"] == "equity_rainbow_two_asset"


def test_build_market_scenario_coverage_report_counts_usage_and_unknown_refs():
    from trellis.agent.market_scenarios import (
        build_market_scenario_coverage_report,
        load_market_scenario_contracts,
    )

    contracts = load_market_scenario_contracts(root=ROOT)
    report = build_market_scenario_coverage_report(
        pricing_tasks=[
            {"id": "F001", "task_corpus": "benchmark_financepy", "market_scenario_id": "flat_usd_equity_vanilla"},
            {"id": "F002", "task_corpus": "extension", "market_scenario_id": "flat_fx_gk"},
            {"id": "T013", "task_corpus": "proof_legacy"},
        ],
        negative_tasks=[
            {"id": "N001", "task_corpus": "negative", "market_scenario_id": "negative_request_only"},
            {"id": "N002", "task_corpus": "negative"},
        ],
        canaries=[
            {"id": "F001"},
            {"id": "N001"},
            {"id": "T013"},
            {"id": "MISSING"},
        ],
        scenario_contracts=contracts,
    )

    assert report["constructor_counts"]["single_asset_equity"] >= 1
    assert report["task_counts_by_corpus"]["proof_legacy"] == 1
    assert report["usage_by_scenario"]["flat_usd_equity_vanilla"]["pricing"] == 1
    assert report["usage_by_scenario"]["negative_request_only"]["negative"] == 1
    assert {"task_id": "N002", "task_corpus": "negative"} in report["missing_task_scenarios"]
    assert all(item["task_id"] != "T013" for item in report["missing_task_scenarios"])
    assert any(item["task_id"] == "MISSING" for item in report["unknown_scenario_refs"])
