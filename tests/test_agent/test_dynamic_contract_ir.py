from __future__ import annotations

from datetime import date

import pytest

from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    ControlProgram,
    DecisionEvent,
    DynamicContractIR,
    DynamicContractIRWellFormednessError,
    EventProgram,
    EventTimeBucket,
    StateFieldSpec,
    StateResetEvent,
    StateSchema,
    StateUpdateSpec,
    TerminationRule,
)
from trellis.agent.static_leg_contract import SettlementRule


class TestDynamicContractIR:
    def test_dynamic_wrapper_allows_bounded_control_program(self):
        redeem = ActionSpec("redeem", "terminate")
        continue_ = ActionSpec("continue", "continue")
        decision = DecisionEvent(
            label="issuer_call_1",
            schedule_role="call_date",
            action_set=(redeem, continue_),
            controller_role="issuer",
        )
        contract = DynamicContractIR(
            base_contract=None,
            event_program=EventProgram(
                buckets=(
                    EventTimeBucket(
                        event_date=date(2027, 1, 15),
                        phase_sequence=("decision", "termination"),
                        events=(decision,),
                    ),
                ),
                termination_rules=(
                    TerminationRule(
                        label="redeem_if_called",
                        trigger="action == redeem",
                        settlement_expression="par_redemption",
                        event_label="issuer_call_1",
                    ),
                ),
            ),
            control_program=ControlProgram(
                controller_role="issuer",
                decision_style="bermudan",
                decision_event_labels=("issuer_call_1",),
                admissible_actions=(redeem, continue_),
            ),
            settlement=SettlementRule(payout_currency="USD"),
        )

        assert contract.control_program is not None
        assert contract.event_program.buckets[0].events[0].label == "issuer_call_1"

    def test_state_updates_must_reference_declared_fields(self):
        with pytest.raises(DynamicContractIRWellFormednessError):
            DynamicContractIR(
                base_contract=None,
                state_schema=StateSchema(
                    fields=(StateFieldSpec(name="coupon_memory", domain="float", initial_value=0.0),)
                ),
                event_program=EventProgram(
                    buckets=(
                        EventTimeBucket(
                            event_date=date(2027, 1, 15),
                            phase_sequence=("state_update",),
                            events=(
                                StateResetEvent(
                                    label="bad_update",
                                    schedule_role="coupon_date",
                                    state_updates=(
                                        StateUpdateSpec("missing_field", "0.0"),
                                    ),
                                ),
                            ),
                        ),
                    )
                ),
            )

    def test_decision_events_require_a_control_program(self):
        with pytest.raises(DynamicContractIRWellFormednessError):
            DynamicContractIR(
                base_contract=None,
                event_program=EventProgram(
                    buckets=(
                        EventTimeBucket(
                            event_date=date(2027, 1, 15),
                            phase_sequence=("decision",),
                            events=(
                                DecisionEvent(
                                    label="issuer_call_1",
                                    schedule_role="call_date",
                                    action_set=(ActionSpec("redeem", "terminate"),),
                                    controller_role="issuer",
                                ),
                            ),
                        ),
                    )
                ),
            )
