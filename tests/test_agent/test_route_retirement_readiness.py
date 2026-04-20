from __future__ import annotations

import pytest

from trellis.agent.knowledge.decompose import decompose_to_dynamic_contract_ir


def test_dynamic_route_retirement_readiness_ledger_seeds_dynamic_cohorts():
    from trellis.agent.route_retirement_readiness import (
        dynamic_route_retirement_readiness_ledger,
        get_dynamic_route_retirement_readiness,
        is_route_retirement_ready,
        missing_route_retirement_gates,
    )

    ledger = dynamic_route_retirement_readiness_ledger()

    assert tuple(entry.cohort_id for entry in ledger) == (
        "automatic_event_state",
        "discrete_control",
        "continuous_singular_control",
    )

    automatic = get_dynamic_route_retirement_readiness("automatic_event_state")
    discrete = get_dynamic_route_retirement_readiness("discrete_control")
    continuous = get_dynamic_route_retirement_readiness("continuous_singular_control")

    assert automatic.proving_families == ("autocallable", "tarn")
    assert "callable_cms_range_accrual" in automatic.honest_block_relatives
    assert discrete.proving_families == ("callable_bond", "swing_option")
    assert "autocallable" in discrete.honest_block_relatives
    assert continuous.proving_families == ("gmwb_financial_control",)
    assert "insurance_overlay" in continuous.honest_block_relatives

    for entry in ledger:
        assert entry.representation_closure.ready is True
        assert entry.decomposition_closure.ready is True
        assert entry.lowering_admission.ready is True
        assert entry.masked_authority_readiness.ready is True
        assert missing_route_retirement_gates(entry) == (
            "parity_or_benchmark",
            "provenance_readiness",
        )
        assert is_route_retirement_ready(entry) is False


@pytest.mark.parametrize(
    ("description", "instrument_type", "expected_lane", "expected_ref"),
    [
        (
            "Phoenix autocallable note on SPX notional 1000000 coupon 8% "
            "autocall barrier 100% observation dates 2025-07-15, 2026-01-15, "
            "2026-07-15, 2027-01-15 maturity 2027-01-15",
            "autocallable",
            "automatic_event_state",
            "autocallable_note",
        ),
        (
            "Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue "
            "2025-01-15 maturity 2030-01-15 semiannual day count ACT/ACT "
            "call dates 2027-01-15, 2028-01-15, 2029-01-15",
            "callable_bond",
            "discrete_control",
            "callable_bond",
        ),
        (
            "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
            "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15",
            "gmwb",
            "continuous_control",
            "gmwb_financial_control",
        ),
    ],
)
def test_masked_authority_harness_supports_dynamic_later_family_probes(
    description: str,
    instrument_type: str,
    expected_lane: str,
    expected_ref: str,
):
    from trellis.agent.route_retirement_readiness import (
        capture_dynamic_lane_probe_authority_snapshot,
        default_masked_authority_variants,
        require_masked_authority_invariant,
    )

    contract = decompose_to_dynamic_contract_ir(description, instrument_type=instrument_type)

    baseline = require_masked_authority_invariant(
        default_masked_authority_variants(),
        lambda variant: capture_dynamic_lane_probe_authority_snapshot(
            contract,
            variant=variant,
        ),
    )

    assert baseline.selection_surface == "dynamic_lane_probe"
    assert baseline.lane_family == expected_lane
    assert baseline.authoritative_ref == expected_ref


def test_masked_authority_harness_reuses_phase4_route_free_build_surface():
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.route_retirement_readiness import (
        capture_compiled_request_authority_snapshot,
        default_masked_authority_variants,
        require_masked_authority_invariant,
    )

    description = "European call on AAPL strike 150 expiring 2025-11-15"

    baseline = require_masked_authority_invariant(
        default_masked_authority_variants(),
        lambda variant: capture_compiled_request_authority_snapshot(
            compile_build_request(
                description,
                instrument_type="european_option",
                preferred_method="analytical",
                metadata={
                    "route_id": variant.route_id,
                    "route_family": variant.route_family,
                    "product_instrument": variant.product_instrument,
                    **dict(variant.wrapper_metadata),
                },
            )
        ),
    )

    assert baseline.selection_surface == "platform_request"
    assert baseline.authoritative_ref == "black76_vanilla_call"
    assert baseline.lane_family == "analytical"
    assert baseline.binding_id == "trellis.models.black.black76_call"
