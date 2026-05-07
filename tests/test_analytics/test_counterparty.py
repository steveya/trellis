"""Tests for institutional counterparty valuation workflows."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


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

