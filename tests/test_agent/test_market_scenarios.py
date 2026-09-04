from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import yaml


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


def test_p005_usd_rates_scenario_isolates_named_hull_white_parameters():
    from trellis.agent.market_scenarios import load_market_scenario_contracts

    contracts = load_market_scenario_contracts(root=ROOT)
    shared_contract = contracts["usd_rates_smile"]
    contract = contracts["usd_rates_smile_physical_bermudan"]

    assert shared_contract.model_parameter_sets == {}
    assert contract.model_parameter_sets == {
        "usd_rates_smile_hw1f": {
            "parameter_set_name": "usd_rates_smile_hw1f",
            "model_family": "hull_white",
            "mean_reversion": 0.05,
            "sigma": 0.01,
            "source_kind": "calibrated",
            "calibration_source": "usd_rates_smile_mock_calibration_v1",
        }
    }
    assert contract.selected_components == shared_contract.selected_components
    assert contract.domestic_rate == shared_contract.domestic_rate
    assert contract.forecast_rate == shared_contract.forecast_rate
    assert contract.forecast_curve_name == shared_contract.forecast_curve_name
    assert contract.black_vol == shared_contract.black_vol
    assert contract.shifted_black_vol == shared_contract.shifted_black_vol
    assert contract.shift == shared_contract.shift
    assert contract.sabr == shared_contract.sabr
    assert "model_parameters" not in contract.selected_components
    assert contract.to_payload()["model_parameter_sets"] == contract.model_parameter_sets
    assert (
        replace(contract, model_parameter_sets={}).with_digest().scenario_digest
        != contract.scenario_digest
    )


def test_embedded_market_scenario_round_trip_preserves_named_model_parameter_sets():
    from trellis.agent.market_scenarios import (
        load_market_scenario_contracts,
        market_scenario_contract_from_task,
    )

    original = load_market_scenario_contracts(root=ROOT)[
        "usd_rates_smile_physical_bermudan"
    ]
    reconstructed = market_scenario_contract_from_task(
        {
            "market_scenario_id": original.scenario_id,
            "market": original.to_market_spec(),
        }
    )

    assert reconstructed is not None
    assert reconstructed.scenario_digest == original.scenario_digest
    assert reconstructed.model_parameter_sets == original.model_parameter_sets


def test_construct_market_state_materializes_named_model_parameters_without_global_selection():
    from trellis.agent.market_scenarios import (
        construct_market_state_for_scenario,
        load_market_scenario_contracts,
    )
    from trellis.core.market_state import MarketState

    contract = load_market_scenario_contracts(root=ROOT)[
        "usd_rates_smile_physical_bermudan"
    ]
    pack_only_contract = replace(
        contract,
        shifted_black_vol=None,
        shift=None,
        sabr={},
    )
    isolated_state, _ = construct_market_state_for_scenario(
        pack_only_contract,
        MarketState(
            as_of=date(2024, 11, 15),
            settlement=date(2024, 11, 15),
        ),
    )
    assert isolated_state.model_parameters == {}
    assert isolated_state.model_parameter_sets == contract.model_parameter_sets

    base_state = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        model_parameters={},
        model_parameter_sets={"existing": {"model_family": "custom"}},
    )

    market_state, metadata = construct_market_state_for_scenario(
        contract,
        base_state,
        task_id="P005",
    )

    assert "hull_white" not in market_state.model_parameters
    assert "mean_reversion" not in market_state.model_parameters
    assert "sigma" not in market_state.model_parameters
    assert market_state.model_parameter_sets == {
        "existing": {"model_family": "custom"},
        **contract.model_parameter_sets,
    }
    assert metadata["scenario_applied_inputs"]["model_parameter_sets"] == [
        "usd_rates_smile_hw1f"
    ]
    scenario_provenance = market_state.market_provenance["market_scenario"]
    assert scenario_provenance["model_parameter_sets"] == contract.model_parameter_sets
    assert scenario_provenance["applied_inputs"]["model_parameter_sets"] == [
        "usd_rates_smile_hw1f"
    ]


def test_shared_usd_rates_scenario_does_not_implicitly_enable_hull_white():
    from trellis.agent.market_scenarios import (
        construct_market_state_for_scenario,
        load_market_scenario_contracts,
    )
    from trellis.core.market_state import MarketState
    from trellis.models.hull_white_parameters import resolve_hull_white_parameters

    contract = load_market_scenario_contracts(root=ROOT)["usd_rates_smile"]
    market_state, _ = construct_market_state_for_scenario(
        contract,
        MarketState(
            as_of=date(2024, 11, 15),
            settlement=date(2024, 11, 15),
        ),
    )

    assert market_state.model_parameter_sets is None
    with pytest.raises(ValueError, match="Hull-White sigma must be provided"):
        resolve_hull_white_parameters(market_state)


@pytest.mark.parametrize(
    "model_parameter_sets",
    [
        ["not-a-map"],
        {"": {"model_family": "hull_white"}},
        {"valid_name": ["not-a-parameter-map"]},
    ],
)
def test_load_market_scenario_contracts_rejects_malformed_named_parameter_sets(
    tmp_path,
    model_parameter_sets,
):
    from trellis.agent.market_scenarios import load_market_scenario_contracts

    manifest = {
        "version": 2,
        "scenarios": {
            "invalid_parameters": {
                "source": "mock",
                "as_of": "2024-11-15",
                "constructor": {
                    "kind": "flat_rates",
                    "model_parameter_sets": model_parameter_sets,
                },
            }
        },
    }
    (tmp_path / "MARKET_SCENARIOS.yaml").write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_parameter_sets"):
        load_market_scenario_contracts(root=tmp_path)


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


def test_construct_hybrid_equity_fx_scenario_keeps_named_factors_distinct():
    from trellis.agent.market_scenarios import (
        construct_market_state_for_scenario,
        load_market_scenario_contracts,
    )
    from trellis.agent.task_runtime import build_market_state

    contract = load_market_scenario_contracts(root=ROOT)["eur_equity_usd_quanto"]
    market_state, metadata = construct_market_state_for_scenario(
        contract,
        build_market_state(),
        task_id="T105",
    )

    assert contract.constructor_kind == "hybrid_equity_fx"
    assert contract.financepy_inputs()["stock_price"] == 100.0
    assert contract.financepy_inputs()["spot_fx"] == 1.10
    assert metadata["scenario_construction_kind"] == "hybrid_equity_fx"
    assert market_state.underlier_spots["SX5E"] == 100.0
    assert market_state.fx_rates["EURUSD"].spot == 1.10
    assert market_state.vol_surfaces["sx5e_implied_vol"].black_vol(1.0, 100.0) == 0.20
    assert market_state.vol_surfaces["eurusd_implied_vol"].black_vol(1.0, 1.10) == 0.12
    assert market_state.vol_surface is market_state.vol_surfaces["sx5e_implied_vol"]
    assert market_state.model_parameters["correlation_source"]["value"] == 0.25
    assert "rho" not in market_state.model_parameters
    assert "model_family" not in market_state.model_parameters
    assert market_state.market_provenance["market_scenario"]["scenario_id"] == (
        "eur_equity_usd_quanto"
    )


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
