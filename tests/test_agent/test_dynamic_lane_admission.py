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
from trellis.agent.dynamic_lane_admission import (
    DynamicLaneAdmissionError,
    _automatic_benchmark_plan,
    _discrete_benchmark_plan,
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


def test_dynamic_lane_admission_fails_closed_for_deferred_insurance_overlays():
    from trellis.agent.dynamic_lane_admission import (
        compile_dynamic_lane_admission,
    )

    withdraw = ActionSpec(
        "withdraw",
        "withdraw",
        action_domain="continuous",
        quantity_source="withdrawal_amount",
        bounds_expression="0 <= withdrawal_amount <= guarantee_base",
        state_updates=(
            StateUpdateSpec("account_value", "account_value - withdrawal_amount"),
            StateUpdateSpec("guarantee_base", "guarantee_base - withdrawal_amount"),
        ),
    )
    contract = DynamicContractIR(
        base_contract=None,
        semantic_family="gmwb",
        base_track="payoff_expression",
        state_schema=StateSchema(
            fields=(
                StateFieldSpec(
                    "account_value",
                    "float",
                    100_000.0,
                    tags=("financial_state", "continuous_control"),
                ),
                StateFieldSpec(
                    "guarantee_base",
                    "float",
                    100_000.0,
                    tags=("financial_state", "continuous_control"),
                ),
                StateFieldSpec(
                    "policy_status",
                    "enum",
                    "alive",
                    tags=("policy_state", "insurance_overlay"),
                ),
            ),
        ),
        event_program=EventProgram(
            buckets=(
                EventTimeBucket(
                    event_date=date(2026, 1, 15),
                    phase_sequence=("decision", "payment"),
                    events=(
                        DecisionEvent(
                            label="withdraw_2026-01-15",
                            schedule_role="withdrawal_dates",
                            action_set=(withdraw,),
                            controller_role="holder",
                        ),
                        PaymentEvent(
                            label="withdrawal_cashflow_2026-01-15",
                            schedule_role="payment_dates",
                            cashflow_formula="withdrawal_amount",
                        ),
                    ),
                ),
            ),
        ),
        control_program=ControlProgram(
            controller_role="holder",
            decision_style="continuous_withdrawal",
            decision_event_labels=("withdraw_2026-01-15",),
            admissible_actions=(withdraw,),
            inventory_fields=("guarantee_base",),
        ),
        settlement=SettlementRule(payout_currency="USD"),
    )

    with pytest.raises(DynamicLaneAdmissionError, match="insurance-style overlay"):
        compile_dynamic_lane_admission(contract)


@pytest.mark.parametrize(
    ("planner", "family", "match"),
    (
        (_automatic_benchmark_plan, "unsupported", "automatic event/state cohort"),
        (_discrete_benchmark_plan, "unsupported", "discrete-control cohort"),
    ),
)
def test_benchmark_plans_fail_closed_for_unknown_families(planner, family, match):
    with pytest.raises(DynamicLaneAdmissionError, match=match):
        planner(family)
