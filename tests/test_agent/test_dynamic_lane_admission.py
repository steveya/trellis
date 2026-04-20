from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    ControlProgram,
    DecisionEvent,
    DynamicContractIR,
    EventProgram,
    EventTimeBucket,
    ObservationEvent,
    PaymentEvent,
    StateFieldSpec,
    StateSchema,
    StateUpdateSpec,
    TerminationRule,
)
from trellis.agent.knowledge.decompose import decompose_to_dynamic_contract_ir
from trellis.agent.static_leg_contract import SettlementRule


def test_automatic_event_state_lane_admits_autocallable_fixture():
    from trellis.agent.dynamic_lane_admission import (
        AutomaticEventStateLaneAdmission,
        compile_dynamic_lane_admission,
    )

    contract = decompose_to_dynamic_contract_ir(
        "Phoenix autocallable note on SPX notional 1000000 coupon 8% "
        "autocall barrier 100% observation dates 2025-07-15, 2026-01-15, "
        "2026-07-15, 2027-01-15 maturity 2027-01-15",
        instrument_type="autocallable",
    )

    admission = compile_dynamic_lane_admission(contract)

    assert isinstance(admission, AutomaticEventStateLaneAdmission)
    assert admission.semantic_family == "autocallable"
    assert admission.state_fields == ("coupon_memory",)
    assert admission.termination_rule_labels
    assert admission.candidate_numerical_lanes == ("event_aware_monte_carlo",)
    assert admission.benchmark_plan.cohort_id == "autocallable_note"


def test_discrete_control_lane_admits_callable_and_inventory_control():
    from trellis.agent.dynamic_lane_admission import (
        DiscreteControlLaneAdmission,
        compile_dynamic_lane_admission,
    )

    callable_bond = decompose_to_dynamic_contract_ir(
        "Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue "
        "2025-01-15 maturity 2030-01-15 semiannual day count ACT/ACT "
        "call dates 2027-01-15, 2028-01-15, 2029-01-15",
        instrument_type="callable_bond",
    )
    swing = decompose_to_dynamic_contract_ir(
        "Swing option on power notional 1000000 rights 3 exercise dates "
        "2025-01-31, 2025-02-28, 2025-03-31, 2025-04-30 maturity 2025-04-30",
        instrument_type="swing_option",
    )

    callable_admission = compile_dynamic_lane_admission(callable_bond)
    swing_admission = compile_dynamic_lane_admission(swing)

    assert isinstance(callable_admission, DiscreteControlLaneAdmission)
    assert callable_admission.controller_role == "issuer"
    assert callable_admission.candidate_numerical_lanes == ("exercise_lattice", "event_aware_pde")
    assert callable_admission.benchmark_plan.cohort_id == "callable_bond"

    assert isinstance(swing_admission, DiscreteControlLaneAdmission)
    assert swing_admission.controller_role == "holder"
    assert swing_admission.inventory_fields == ("remaining_rights",)
    assert swing_admission.candidate_numerical_lanes == ("control_lsmc",)
    assert swing_admission.benchmark_plan.cohort_id == "swing_option"


def test_continuous_control_lane_requires_magnitude_semantics():
    from trellis.agent.dynamic_lane_admission import (
        ContinuousControlLaneAdmission,
        compile_dynamic_lane_admission,
    )

    contract = decompose_to_dynamic_contract_ir(
        "GMWB contract premium 100000 guarantee base 100000 account value 100000 "
        "withdrawal dates 2026-01-15, 2027-01-15, 2028-01-15",
        instrument_type="gmwb",
    )

    admission = compile_dynamic_lane_admission(contract)

    assert isinstance(admission, ContinuousControlLaneAdmission)
    assert admission.controlled_state_fields == ("account_value", "guarantee_base")
    assert admission.action_domains == ("continuous",)
    assert admission.magnitude_action_names == ("withdraw",)
    assert admission.candidate_numerical_lanes == ("qvi_pde", "control_dynamic_programming")
    assert admission.benchmark_plan.cohort_id == "gmwb_financial_control"


def test_dynamic_lane_admission_fails_closed_for_deferred_hybrids():
    from trellis.agent.dynamic_lane_admission import (
        DynamicLaneAdmissionError,
        compile_dynamic_lane_admission,
    )

    observe = ObservationEvent(
        label="observe_cms_spread",
        schedule_role="observation_dates",
        observed_terms=("cms_10y_2y_spread",),
    )
    coupon = PaymentEvent(
        label="pay_coupon",
        schedule_role="payment_dates",
        cashflow_formula="coupon_if_spread_in_range",
    )
    call = DecisionEvent(
        label="issuer_call",
        schedule_role="call_dates",
        action_set=(
            ActionSpec("call", "terminate"),
            ActionSpec("continue", "continue"),
        ),
        controller_role="issuer",
    )
    contract = DynamicContractIR(
        base_contract=None,
        semantic_family="callable_cms_range_accrual",
        base_track="quoted_observable",
        state_schema=StateSchema(
            fields=(StateFieldSpec("coupon_memory", "float", 0.0, tags=("quoted_hybrid",)),)
        ),
        event_program=EventProgram(
            buckets=(
                EventTimeBucket(
                    event_date=date(2027, 1, 15),
                    phase_sequence=("observation", "payment", "decision"),
                    events=(observe, coupon, call),
                ),
            ),
            termination_rules=(
                TerminationRule(
                    label="terminate_on_call",
                    trigger="action == call",
                    settlement_expression="par_redemption",
                    event_label="issuer_call",
                ),
            ),
        ),
        control_program=ControlProgram(
            controller_role="issuer",
            decision_style="bermudan",
            decision_event_labels=("issuer_call",),
            admissible_actions=(
                ActionSpec("call", "terminate"),
                ActionSpec("continue", "continue"),
            ),
        ),
        settlement=SettlementRule(payout_currency="USD"),
    )

    with pytest.raises(DynamicLaneAdmissionError, match="quoted-observable"):
        compile_dynamic_lane_admission(contract)
