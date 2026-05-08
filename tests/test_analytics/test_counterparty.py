"""Tests for institutional counterparty valuation workflows."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import numpy as np
import pytest

from trellis.book import FutureValueCube


def test_counterparty_semantic_contract_records_collateral_and_netting_context():
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        CounterpartySemanticContract,
        NettingSet,
        validate_counterparty_semantic_contract,
    )

    agreement = CollateralAgreement(
        agreement_id="csa_alpha",
        collateral_currency="USD",
        threshold=100.0,
        minimum_transfer_amount=25.0,
        independent_amount=10.0,
        margin_period_of_risk_days=10,
        call_frequency_days=1,
        valuation_lag_days=1,
    )
    netting_set = NettingSet(
        netting_set_id="ns_alpha",
        counterparty_id="bank_alpha",
        position_names=("payer_swap", "receiver_swap"),
        collateral_agreement_id="csa_alpha",
        exposure_currency="USD",
    )
    contract = CounterpartySemanticContract(
        contract_id="cp_alpha",
        netting_sets=(netting_set,),
        collateral_agreements=(agreement,),
        covered_position_names=("payer_swap", "receiver_swap"),
    )

    report = validate_counterparty_semantic_contract(contract)
    payload = contract.to_dict()

    assert report.ok is True
    assert report.missing_fields == ()
    assert report.warnings == ()
    assert payload["contract_id"] == "cp_alpha"
    assert payload["netting_sets"][0]["position_names"] == ["payer_swap", "receiver_swap"]
    assert payload["collateral_agreements"][0]["margin_period_of_risk_days"] == 10
    assert payload["runtime_semantics"]["closeout_convention"] == "replacement_cost"

    with pytest.raises(FrozenInstanceError):
        agreement.threshold = 0.0


def test_counterparty_semantic_contract_warns_on_missing_operational_fields():
    from trellis.analytics.counterparty import (
        CounterpartySemanticContract,
        NettingSet,
        validate_counterparty_semantic_contract,
    )

    contract = CounterpartySemanticContract(
        contract_id="cp_incomplete",
        netting_sets=(
            NettingSet(
                netting_set_id="ns_missing",
                counterparty_id="bank_missing",
                position_names=("swap_a",),
            ),
        ),
        covered_position_names=("swap_a", "swap_b"),
    )

    report = validate_counterparty_semantic_contract(contract)

    assert report.ok is False
    assert "collateral_agreements" in report.missing_fields
    assert "netting_sets.ns_missing.exposure_currency" in report.missing_fields
    assert "covered_position_names.swap_b" in report.missing_fields
    assert "netting_sets.ns_missing.collateral_agreement_id" in report.warnings
    assert report.to_dict()["ok"] is False


def test_project_collateral_state_applies_margin_period_and_valuation_lag():
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        NettingSet,
        project_collateral_state,
    )

    cube = FutureValueCube(
        values=np.asarray(
            [
                [
                    [100.0, 20.0],
                    [140.0, 30.0],
                    [80.0, 120.0],
                ],
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
            dtype=float,
        ),
        position_names=("swap_a", "swap_b"),
        observation_times=(0.0, 1.0 / 12.0, 2.0 / 12.0),
        observation_dates=(
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 3, 1),
        ),
    )
    agreement = CollateralAgreement(
        agreement_id="csa_alpha",
        collateral_currency="USD",
        threshold=50.0,
        minimum_transfer_amount=10.0,
        independent_amount=5.0,
        margin_period_of_risk_days=31,
        valuation_lag_days=29,
    )
    netting_set = NettingSet(
        netting_set_id="ns_alpha",
        counterparty_id="bank_alpha",
        position_names=("swap_a", "swap_b"),
        collateral_agreement_id="csa_alpha",
        exposure_currency="USD",
    )

    projection = project_collateral_state(
        cube,
        netting_set=netting_set,
        collateral_agreement=agreement,
    )

    np.testing.assert_allclose(
        projection.netted_values,
        np.asarray([[100.0, 20.0], [140.0, 30.0], [80.0, 120.0]]),
    )
    np.testing.assert_allclose(
        projection.closeout_values,
        np.asarray([[140.0, 30.0], [80.0, 120.0], [80.0, 120.0]]),
    )
    np.testing.assert_allclose(
        projection.collateral_balance,
        np.asarray([[55.0, 5.0], [55.0, 5.0], [95.0, 5.0]]),
    )
    np.testing.assert_allclose(
        projection.collateralized_exposure,
        np.asarray([[85.0, 25.0], [25.0, 115.0], [0.0, 115.0]]),
    )
    assert projection.metadata["margin_period_of_risk_days"] == 31
    assert projection.metadata["valuation_lag_days"] == 29
    assert projection.to_dict()["netting_set_id"] == "ns_alpha"


def test_aggregate_netting_set_exposures_builds_closeout_inputs():
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        NettingSet,
        aggregate_netting_set_exposures,
        project_collateral_state,
    )

    cube = FutureValueCube(
        values=np.asarray(
            [
                [[100.0, 20.0], [140.0, 30.0], [80.0, 120.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[-50.0, 10.0], [60.0, -20.0], [10.0, 30.0]],
            ],
            dtype=float,
        ),
        position_names=("swap_a", "swap_b", "swap_c"),
        observation_times=(0.0, 1.0 / 12.0, 2.0 / 12.0),
        observation_dates=(
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 3, 1),
        ),
    )
    agreement = CollateralAgreement(
        agreement_id="csa_alpha",
        collateral_currency="USD",
        threshold=50.0,
        minimum_transfer_amount=10.0,
        independent_amount=5.0,
        margin_period_of_risk_days=31,
        valuation_lag_days=29,
    )
    alpha = NettingSet(
        netting_set_id="ns_alpha",
        counterparty_id="bank_alpha",
        position_names=("swap_a", "swap_b"),
        collateral_agreement_id="csa_alpha",
        exposure_currency="USD",
    )
    beta = NettingSet(
        netting_set_id="ns_beta",
        counterparty_id="bank_beta",
        position_names=("swap_c",),
        exposure_currency="USD",
    )
    alpha_projection = project_collateral_state(
        cube,
        netting_set=alpha,
        collateral_agreement=agreement,
    )

    exposure_cube = aggregate_netting_set_exposures(
        cube,
        netting_sets=(alpha, beta),
        collateral_projections={"ns_alpha": alpha_projection},
    )

    assert exposure_cube.netting_set_ids == ("ns_alpha", "ns_beta")
    np.testing.assert_allclose(
        exposure_cube.netted_values_for_set("ns_alpha"),
        alpha_projection.netted_values,
    )
    np.testing.assert_allclose(
        exposure_cube.exposure_values_for_set("ns_alpha"),
        alpha_projection.collateralized_exposure,
    )
    np.testing.assert_allclose(
        exposure_cube.netted_values_for_set("ns_beta"),
        np.asarray([[-50.0, 10.0], [60.0, -20.0], [10.0, 30.0]]),
    )
    np.testing.assert_allclose(
        exposure_cube.exposure_values_for_set("ns_beta"),
        np.asarray([[0.0, 10.0], [60.0, 0.0], [10.0, 30.0]]),
    )
    closeout = exposure_cube.closeout_input_for_set("ns_alpha")
    assert closeout["netting_set_id"] == "ns_alpha"
    assert closeout["collateralized"] is True
    assert closeout["exposure_currency"] == "USD"
    np.testing.assert_allclose(closeout["collateral_balance"], alpha_projection.collateral_balance)
    np.testing.assert_allclose(
        exposure_cube.portfolio_exposure_values(),
        alpha_projection.collateralized_exposure
        + np.asarray([[0.0, 10.0], [60.0, 0.0], [10.0, 30.0]]),
    )
    assert exposure_cube.to_dict()["netting_set_ids"] == ["ns_alpha", "ns_beta"]


def test_compute_exposure_metrics_reports_ee_epe_and_pfe():
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        NettingSet,
        aggregate_netting_set_exposures,
        compute_exposure_metrics,
        project_collateral_state,
    )

    cube = FutureValueCube(
        values=np.asarray(
            [
                [[100.0, 20.0], [140.0, 30.0], [80.0, 120.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[-50.0, 10.0], [60.0, -20.0], [10.0, 30.0]],
            ],
            dtype=float,
        ),
        position_names=("swap_a", "swap_b", "swap_c"),
        observation_times=(0.0, 1.0 / 12.0, 2.0 / 12.0),
        observation_dates=(
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 3, 1),
        ),
    )
    agreement = CollateralAgreement(
        agreement_id="csa_alpha",
        collateral_currency="USD",
        threshold=50.0,
        minimum_transfer_amount=10.0,
        independent_amount=5.0,
        margin_period_of_risk_days=31,
        valuation_lag_days=29,
    )
    alpha = NettingSet(
        netting_set_id="ns_alpha",
        counterparty_id="bank_alpha",
        position_names=("swap_a", "swap_b"),
        collateral_agreement_id="csa_alpha",
        exposure_currency="USD",
    )
    beta = NettingSet(
        netting_set_id="ns_beta",
        counterparty_id="bank_beta",
        position_names=("swap_c",),
        exposure_currency="USD",
    )
    exposure_cube = aggregate_netting_set_exposures(
        cube,
        netting_sets=(alpha, beta),
        collateral_projections={
            "ns_alpha": project_collateral_state(
                cube,
                netting_set=alpha,
                collateral_agreement=agreement,
            )
        },
    )

    metrics = compute_exposure_metrics(exposure_cube, pfe_levels=(0.5, 0.95))

    np.testing.assert_allclose(metrics.expected_exposure, (60.0, 100.0, 77.5))
    assert metrics.epe == pytest.approx(84.375)
    np.testing.assert_allclose(metrics.pfe[0.5], (60.0, 100.0, 77.5))
    np.testing.assert_allclose(metrics.pfe[0.95], (82.5, 113.5, 138.25))
    np.testing.assert_allclose(
        metrics.netting_set_metrics["ns_beta"]["expected_exposure"],
        (5.0, 30.0, 20.0),
    )
    assert metrics.to_dict()["aggregation_level"] == "portfolio"


def test_price_counterparty_xva_consumes_semantic_exposure_stack():
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        CounterpartySemanticContract,
        NettingSet,
        XVAAssumptionSet,
        price_counterparty_xva,
    )

    cube = FutureValueCube(
        values=np.asarray(
            [
                [[100.0, 20.0], [140.0, 30.0], [80.0, 120.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[-50.0, 10.0], [60.0, -20.0], [10.0, 30.0]],
            ],
            dtype=float,
        ),
        position_names=("swap_a", "swap_b", "swap_c"),
        observation_times=(0.0, 1.0 / 12.0, 2.0 / 12.0),
        observation_dates=(
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 3, 1),
        ),
    )
    agreement = CollateralAgreement(
        agreement_id="csa_alpha",
        collateral_currency="USD",
        threshold=50.0,
        minimum_transfer_amount=10.0,
        independent_amount=5.0,
        margin_period_of_risk_days=31,
        valuation_lag_days=29,
    )
    alpha = NettingSet(
        netting_set_id="ns_alpha",
        counterparty_id="bank_alpha",
        position_names=("swap_a", "swap_b"),
        collateral_agreement_id="csa_alpha",
        exposure_currency="USD",
    )
    beta = NettingSet(
        netting_set_id="ns_beta",
        counterparty_id="bank_beta",
        position_names=("swap_c",),
        exposure_currency="USD",
    )
    contract = CounterpartySemanticContract(
        contract_id="cp_alpha",
        netting_sets=(alpha, beta),
        collateral_agreements=(agreement,),
        covered_position_names=("swap_a", "swap_b", "swap_c"),
    )

    result = price_counterparty_xva(
        cube,
        counterparty_contract=contract,
        assumptions=XVAAssumptionSet(
            counterparty_hazard_rate=0.02,
            own_hazard_rate=0.01,
            counterparty_recovery_rate=0.40,
            own_recovery_rate=0.40,
            funding_spread=0.005,
            discount_rate=0.0,
        ),
        pfe_levels=(0.95,),
    )

    assert result.exposure_metrics.epe == pytest.approx(84.375)
    np.testing.assert_allclose(result.expected_negative_exposure, (25.0, 10.0, 7.5))
    assert result.cva == pytest.approx(0.16875)
    assert result.dva == pytest.approx(0.013125)
    assert result.fva == pytest.approx(0.0703125)
    assert result.total_xva == pytest.approx(0.2259375)
    assert result.metadata["counterparty_contract_id"] == "cp_alpha"
    assert result.to_dict()["assumptions"]["funding_spread"] == 0.005
