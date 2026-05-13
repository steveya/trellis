"""Tests for semantic portfolio-AAD lane admission."""

from __future__ import annotations

import json
from datetime import date

from trellis.agent.contract_ir import (
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    Gt,
    Indicator,
    Max,
    Observation,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    Underlying,
)
from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    ControlProgram,
    DecisionEvent,
    DynamicContractIR,
    EventProgram,
    EventTimeBucket,
)
from trellis.analytics.portfolio_aad_admission import admit_portfolio_aad_lane


EXPIRY = date(2025, 11, 15)
OBSERVATIONS = (
    date(2025, 2, 15),
    date(2025, 5, 15),
    date(2025, 8, 15),
    EXPIRY,
)


def _terminal_call_ir() -> ContractIR:
    schedule = Singleton(EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("AAPL", "gbm")),
    )


def _terminal_put_ir() -> ContractIR:
    schedule = Singleton(EXPIRY)
    return ContractIR(
        payoff=Max((Sub(Strike(150.0), Spot("AAPL")), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("AAPL", "gbm")),
    )


def test_terminal_vanilla_contract_ir_admits_existing_flat_vol_lane():
    admission = admit_portfolio_aad_lane(
        _terminal_call_ir(),
        market_parameterization="flat_vol",
        product_family="vanilla_equity_option",
    )

    payload = admission.to_payload()

    assert admission.admitted is True
    assert admission.support_status == "supported"
    assert admission.lane_id == "vanilla_equity_option_flat_vol"
    assert admission.contract_shape == "terminal_vanilla_option"
    assert payload["derivative_method_category"] == "portfolio_aad"
    assert payload["factor_requirements"][0]["coordinate_type"] == "flat_vol"
    assert json.loads(json.dumps(payload)) == payload


def test_terminal_vanilla_contract_ir_admits_grid_vol_lane():
    admission = admit_portfolio_aad_lane(
        _terminal_put_ir(),
        market_parameterization="grid_vol",
        product_family="vanilla_equity_option",
    )

    assert admission.admitted is True
    assert admission.support_status == "supported"
    assert admission.lane_id == "vanilla_equity_option_grid_vol"
    assert admission.reason == "supported_terminal_vanilla_grid_vol_aad"
    assert admission.factor_requirements[0].coordinate_type == "black_vol"


def test_dynamic_early_exercise_contract_admits_flat_vol_control_policy():
    base = _terminal_put_ir()
    action = ActionSpec(action_name="exercise", action_type="exercise")
    decision = DecisionEvent(
        label="holder_exercise",
        schedule_role="exercise",
        action_set=(action,),
        controller_role="holder",
    )
    dynamic_contract = DynamicContractIR(
        base_contract=base,
        semantic_family="american_vanilla_option",
        base_track="payoff_expression",
        event_program=EventProgram(
            buckets=(
                EventTimeBucket(
                    event_date=EXPIRY,
                    phase_sequence=("decision",),
                    events=(decision,),
                ),
            )
        ),
        control_program=ControlProgram(
            controller_role="holder",
            decision_style="optimal_stopping",
            decision_event_labels=("holder_exercise",),
            admissible_actions=(action,),
        ),
    )

    admission = admit_portfolio_aad_lane(dynamic_contract)

    assert admission.admitted is True
    assert admission.support_status == "supported"
    assert admission.reason == "supported_early_exercise_control_policy_flat_vol_aad"
    assert admission.semantic_contract_type == "DynamicContractIR"
    assert admission.lane_id == "early_exercise_control_policy_flat_vol"
    assert admission.metadata["decision_style"] == "optimal_stopping"
    assert admission.metadata["derivative_policy"] == (
        "hard_exercise_projection_smooth_interior"
    )


def test_dynamic_early_exercise_contract_keeps_grid_vol_control_policy_planned():
    base = _terminal_put_ir()
    action = ActionSpec(action_name="exercise", action_type="exercise")
    decision = DecisionEvent(
        label="holder_exercise",
        schedule_role="exercise",
        action_set=(action,),
        controller_role="holder",
    )
    dynamic_contract = DynamicContractIR(
        base_contract=base,
        semantic_family="american_vanilla_option",
        base_track="payoff_expression",
        event_program=EventProgram(
            buckets=(
                EventTimeBucket(
                    event_date=EXPIRY,
                    phase_sequence=("decision",),
                    events=(decision,),
                ),
            )
        ),
        control_program=ControlProgram(
            controller_role="holder",
            decision_style="optimal_stopping",
            decision_event_labels=("holder_exercise",),
            admissible_actions=(action,),
        ),
    )

    admission = admit_portfolio_aad_lane(
        dynamic_contract,
        market_parameterization="grid_vol",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "early_exercise_grid_vol_aad_pending"
    assert admission.factor_requirements[0].coordinate_type == "black_vol"


def test_path_dependent_arithmetic_summary_is_explicitly_pending():
    averaging_schedule = FiniteSchedule(OBSERVATIONS)
    contract = ContractIR(
        payoff=Max(
            (
                Sub(ArithmeticMean(Spot("AAPL"), averaging_schedule), Strike(100.0)),
                Constant(0.0),
            )
        ),
        exercise=Exercise("european", Singleton(EXPIRY)),
        observation=Observation("path_dependent", averaging_schedule),
        underlying=Underlying(EquitySpot("AAPL", "gbm")),
    )

    admission = admit_portfolio_aad_lane(contract)

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "path_dependent_aad_pending"
    assert admission.contract_shape == "path_dependent_smooth_summary"


def test_discontinuous_event_monitor_is_unsupported_not_planned():
    schedule = Singleton(EXPIRY)
    payoff = Scaled(
        Indicator(Gt(Spot("AAPL"), Constant(80.0))),
        Max((Sub(Spot("AAPL"), Strike(100.0)), Constant(0.0))),
    )
    contract = ContractIR(
        payoff=payoff,
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(EquitySpot("AAPL", "gbm")),
    )

    admission = admit_portfolio_aad_lane(contract)

    assert admission.admitted is False
    assert admission.support_status == "unsupported"
    assert admission.reason == "unsupported_discontinuous_event_monitor"


def test_hybrid_composite_underlying_reports_correlation_coordinate_requirement():
    schedule = Singleton(EXPIRY)
    contract = ContractIR(
        payoff=Max((Sub(Spot("AAPL"), Strike(100.0)), Constant(0.0))),
        exercise=Exercise("european", schedule),
        observation=Observation("terminal", schedule),
        underlying=Underlying(
            CompositeUnderlying(
                (
                    EquitySpot("AAPL", "gbm"),
                    EquitySpot("EURUSD", "gbm"),
                )
            )
        ),
    )

    admission = admit_portfolio_aad_lane(
        contract,
        market_parameterization="correlation_scalar",
    )

    assert admission.admitted is False
    assert admission.support_status == "planned"
    assert admission.reason == "hybrid_factor_aad_pending"
    assert admission.factor_requirements[0].coordinate_type == "correlation"
    assert admission.factor_requirements[0].risk_class == "hybrid"
