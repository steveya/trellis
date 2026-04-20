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
    PaymentEvent,
    StateFieldSpec,
    StateSchema,
    StateUpdateSpec,
)
from trellis.agent.static_leg_contract import SettlementRule


def _build_financial_control_core() -> DynamicContractIR:
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
    return DynamicContractIR(
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


class TestInsuranceOverlayContractIR:
    def test_overlay_wrapper_keeps_policy_state_separate_from_financial_core(self):
        from trellis.agent.insurance_overlay_contract import (
            InsuranceOverlayContractIR,
            OverlayCompositionRule,
            OverlayFeeEvent,
            OverlayParameterSet,
            OverlayParameterSpec,
            OverlayTransitionEvent,
            PolicyStateSchema,
        )

        contract = InsuranceOverlayContractIR(
            core_contract=_build_financial_control_core(),
            semantic_family="gmwb",
            policy_state_schema=PolicyStateSchema(
                fields=(
                    StateFieldSpec(
                        "policy_status",
                        "enum",
                        "alive",
                        tags=("policy_state", "insurance_overlay"),
                    ),
                ),
            ),
            overlay_events=(
                OverlayTransitionEvent(
                    label="mortality_transition",
                    schedule_role="overlay_monitoring",
                    trigger_expression="mortality_event_occurs",
                    state_updates=(StateUpdateSpec("policy_status", "dead"),),
                ),
                OverlayFeeEvent(
                    label="rider_fee",
                    schedule_role="overlay_fee_dates",
                    fee_formula="rider_fee_rate * account_value",
                ),
            ),
            overlay_parameters=OverlayParameterSet(
                parameters=(
                    OverlayParameterSpec("rider_fee_rate", "fee_rate", 0.01),
                    OverlayParameterSpec("mortality_hazard", "hazard_rate", "deferred"),
                ),
            ),
            composition_rule=OverlayCompositionRule(
                composition_style="policy_state_gates_financial_control",
                policy_state_field="policy_status",
                notes=("keep the executable continuous-control lane overlay-free",),
            ),
        )

        assert contract.semantic_family == "gmwb"
        assert contract.core_contract.semantic_family == "gmwb"
        assert contract.policy_state_schema.field_names == ("policy_status",)
        assert tuple(event.label for event in contract.overlay_events) == (
            "mortality_transition",
            "rider_fee",
        )
        assert contract.overlay_parameters.parameter_names == (
            "rider_fee_rate",
            "mortality_hazard",
        )

    def test_overlay_wrapper_rejects_policy_state_leaking_into_core_contract(self):
        from trellis.agent.insurance_overlay_contract import (
            InsuranceOverlayContractIR,
            InsuranceOverlayContractWellFormednessError,
            OverlayCompositionRule,
            OverlayTransitionEvent,
            PolicyStateSchema,
        )

        base = _build_financial_control_core()
        contaminated_core = DynamicContractIR(
            base_contract=base.base_contract,
            semantic_family=base.semantic_family,
            base_track=base.base_track,
            state_schema=StateSchema(
                fields=base.state_schema.fields
                + (
                    StateFieldSpec(
                        "policy_status",
                        "enum",
                        "alive",
                        tags=("policy_state",),
                    ),
                ),
            ),
            event_program=base.event_program,
            control_program=base.control_program,
            settlement=base.settlement,
        )

        with pytest.raises(InsuranceOverlayContractWellFormednessError, match="overlay-free"):
            InsuranceOverlayContractIR(
                core_contract=contaminated_core,
                semantic_family="gmwb",
                policy_state_schema=PolicyStateSchema(
                    fields=(
                        StateFieldSpec(
                            "policy_status",
                            "enum",
                            "alive",
                            tags=("policy_state", "insurance_overlay"),
                        ),
                    ),
                ),
                overlay_events=(
                    OverlayTransitionEvent(
                        label="mortality_transition",
                        schedule_role="overlay_monitoring",
                        trigger_expression="mortality_event_occurs",
                        state_updates=(StateUpdateSpec("policy_status", "dead"),),
                    ),
                ),
                composition_rule=OverlayCompositionRule(
                    composition_style="policy_state_gates_financial_control",
                    policy_state_field="policy_status",
                ),
            )

    def test_overlay_events_must_update_declared_policy_state_fields(self):
        from trellis.agent.insurance_overlay_contract import (
            InsuranceOverlayContractIR,
            InsuranceOverlayContractWellFormednessError,
            OverlayCompositionRule,
            OverlayTransitionEvent,
            PolicyStateSchema,
        )

        with pytest.raises(InsuranceOverlayContractWellFormednessError, match="unknown policy-state field"):
            InsuranceOverlayContractIR(
                core_contract=_build_financial_control_core(),
                semantic_family="gmwb",
                policy_state_schema=PolicyStateSchema(
                    fields=(
                        StateFieldSpec(
                            "policy_status",
                            "enum",
                            "alive",
                            tags=("policy_state", "insurance_overlay"),
                        ),
                    ),
                ),
                overlay_events=(
                    OverlayTransitionEvent(
                        label="mortality_transition",
                        schedule_role="overlay_monitoring",
                        trigger_expression="mortality_event_occurs",
                        state_updates=(StateUpdateSpec("missing_field", "dead"),),
                    ),
                ),
                composition_rule=OverlayCompositionRule(
                    composition_style="policy_state_gates_financial_control",
                    policy_state_field="policy_status",
                ),
            )

    def test_composition_rule_policy_state_field_must_reference_declared_field(self):
        from trellis.agent.insurance_overlay_contract import (
            InsuranceOverlayContractIR,
            InsuranceOverlayContractWellFormednessError,
            OverlayCompositionRule,
            OverlayTransitionEvent,
            PolicyStateSchema,
        )

        with pytest.raises(
            InsuranceOverlayContractWellFormednessError,
            match="policy_state_field",
        ):
            InsuranceOverlayContractIR(
                core_contract=_build_financial_control_core(),
                semantic_family="gmwb",
                policy_state_schema=PolicyStateSchema(
                    fields=(
                        StateFieldSpec(
                            "policy_status",
                            "enum",
                            "alive",
                            tags=("policy_state", "insurance_overlay"),
                        ),
                    ),
                ),
                overlay_events=(
                    OverlayTransitionEvent(
                        label="mortality_transition",
                        schedule_role="overlay_monitoring",
                        trigger_expression="mortality_event_occurs",
                        state_updates=(StateUpdateSpec("policy_status", "dead"),),
                    ),
                ),
                composition_rule=OverlayCompositionRule(
                    composition_style="policy_state_gates_financial_control",
                    policy_state_field="undeclared_field",
                ),
            )

    def test_duplicate_overlay_event_labels_are_rejected(self):
        from trellis.agent.insurance_overlay_contract import (
            InsuranceOverlayContractIR,
            InsuranceOverlayContractWellFormednessError,
            OverlayCompositionRule,
            OverlayTransitionEvent,
            PolicyStateSchema,
        )

        with pytest.raises(
            InsuranceOverlayContractWellFormednessError,
            match="duplicate overlay event label",
        ):
            InsuranceOverlayContractIR(
                core_contract=_build_financial_control_core(),
                semantic_family="gmwb",
                policy_state_schema=PolicyStateSchema(
                    fields=(
                        StateFieldSpec(
                            "policy_status",
                            "enum",
                            "alive",
                            tags=("policy_state", "insurance_overlay"),
                        ),
                    ),
                ),
                overlay_events=(
                    OverlayTransitionEvent(
                        label="mortality_transition",
                        schedule_role="overlay_monitoring",
                        trigger_expression="mortality_event_occurs",
                        state_updates=(StateUpdateSpec("policy_status", "dead"),),
                    ),
                    OverlayTransitionEvent(
                        label="mortality_transition",
                        schedule_role="overlay_monitoring",
                        trigger_expression="second_mortality_event",
                        state_updates=(StateUpdateSpec("policy_status", "dead"),),
                    ),
                ),
                composition_rule=OverlayCompositionRule(
                    composition_style="policy_state_gates_financial_control",
                    policy_state_field="policy_status",
                ),
            )
