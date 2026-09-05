from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from trellis.agent.benchmark_contracts import (
    benchmark_preferred_method,
    benchmark_request_description,
    benchmark_spec_overrides,
    canonical_benchmark_instrument_type,
)
from trellis.agent.task_manifests import load_task_manifest
from trellis.agent.task_runtime import prepare_existing_task
from trellis.core.types import DayCountConvention, Frequency


ROOT = Path(__file__).resolve().parents[2]


def _benchmark_tasks() -> dict[str, dict]:
    return {
        task["id"]: task
        for task in load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml", root=ROOT)
    }


def _extension_tasks() -> dict[str, dict]:
    return {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
    }


def _legacy_tasks() -> dict[str, dict]:
    return {
        task["id"]: task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=ROOT)
    }


def _physical_bermudan_task() -> dict:
    return {
        "id": "P005-UNIT",
        "title": "Strict physical Bermudan swaption",
        "instrument_type": "physical_bermudan_swaption",
        "construct": ["rate_tree", "monte_carlo"],
        "extension_contract": {
            "product": "physical_bermudan_swaption",
            "payer_receiver": "payer",
            "settlement_type": "physical",
            "currency": "USD",
            "notional": 1_000_000.0,
            "fixed_coupon": 0.03,
            "exercise_dates": ["2025-11-15", "2026-05-15", "2026-11-15"],
            "exercise_to_swap_start": [
                ["2025-11-15", "2025-11-19"],
                ["2026-05-15", "2026-05-19"],
                ["2026-11-15", "2026-11-18"],
            ],
            "maturity_date": "2030-11-15",
            "discount_curve_id": "usd_ois",
            "forecast_curve_id": "USD-SOFR-3M",
            "hull_white_parameter_source": "usd_rates_smile_hw1f",
            "fixed_frequency": "semi_annual",
            "fixed_day_count": "30/360",
            "fixed_calendar_name": "USSettlement",
            "fixed_business_day_adjustment": "modified_following",
            "fixed_stub_rule": "short_final",
            "fixed_roll_convention": "none",
            "fixed_payment_lag_business_days": 2,
            "float_frequency": "quarterly",
            "floating_day_count": "ACT/360",
            "floating_calendar_name": "USSettlement",
            "floating_business_day_adjustment": "modified_following",
            "floating_stub_rule": "short_final",
            "floating_roll_convention": "none",
            "floating_fixing_lag_business_days": 2,
            "floating_reset_lag_business_days": 2,
            "floating_payment_lag_business_days": 2,
            "floating_rate_index": "USD-SOFR-3M",
            "floating_compounding": "simple",
            "floating_gearing": 1.0,
            "floating_spread": 0.0,
            "model_time_day_count": "ACT/365F",
            "model_time_calendar_name": "USSettlement",
            "projection_policy": "static_additive_forward_basis",
            "lattice_steps": 2_195,
            "lattice_date_tolerance_days": 0,
        },
    }


def test_canonical_benchmark_instrument_type_maps_broad_runtime_families():
    tasks = _benchmark_tasks()

    assert canonical_benchmark_instrument_type(tasks["F002"]) == "european_option"
    assert canonical_benchmark_instrument_type(tasks["F003"]) == "cap"
    assert canonical_benchmark_instrument_type(tasks["F008"]) == "basket_option"
    assert canonical_benchmark_instrument_type(tasks["F010"]) == "digital_option"
    assert canonical_benchmark_instrument_type(tasks["F011"]) == "lookback_option"
    assert canonical_benchmark_instrument_type(tasks["F012"]) == "chooser_option"
    assert canonical_benchmark_instrument_type(tasks["F013"]) == "compound_option"
    assert canonical_benchmark_instrument_type(tasks["F014"]) == "cliquet_option"
    assert canonical_benchmark_instrument_type(tasks["F015"]) == "variance_swap"


def test_canonical_benchmark_instrument_type_supports_period_rate_option_strip_product():
    task = {
        "benchmark_contract": {
            "product": "period_rate_option_strip",
        }
    }

    assert canonical_benchmark_instrument_type(task) == "period_rate_option_strip"


def test_physical_bermudan_extension_contract_maps_to_strict_runtime_identity():
    task = _physical_bermudan_task()

    assert canonical_benchmark_instrument_type(task) == "physical_bermudan_swaption"

    description = benchmark_request_description(task, root=ROOT)
    assert description is not None
    assert "strict physical Bermudan swaption" in description
    assert "Exercise/start mapping: 2025-11-15->2025-11-19" in description
    assert "No analytical, European, Black, or cash-settlement fallback" in description


def test_physical_bermudan_spec_overrides_preserve_every_authored_term_and_alias():
    overrides = benchmark_spec_overrides(_physical_bermudan_task(), root=ROOT)

    assert overrides["fixed_rate"] == pytest.approx(0.03)
    assert overrides["floating_frequency"] == "quarterly"
    assert overrides["swap_maturity"] == date(2030, 11, 15)
    assert overrides["exercise_dates"] == (
        date(2025, 11, 15),
        date(2026, 5, 15),
        date(2026, 11, 15),
    )
    assert overrides["exercise_to_swap_start"] == (
        (date(2025, 11, 15), date(2025, 11, 19)),
        (date(2026, 5, 15), date(2026, 5, 19)),
        (date(2026, 11, 15), date(2026, 11, 18)),
    )
    assert overrides["settlement_type"] == "physical"
    assert overrides["discount_curve_id"] == "usd_ois"
    assert overrides["forecast_curve_id"] == "USD-SOFR-3M"
    assert overrides["hull_white_parameter_source"] == "usd_rates_smile_hw1f"
    assert overrides["fixed_payment_lag_business_days"] == 2
    assert overrides["floating_fixing_lag_business_days"] == 2
    assert overrides["floating_reset_lag_business_days"] == 2
    assert overrides["floating_payment_lag_business_days"] == 2
    assert overrides["lattice_steps"] == 2_195
    assert overrides["lattice_date_tolerance_days"] == 0


def test_physical_bermudan_spec_overrides_reject_any_malformed_exercise_date():
    task = _physical_bermudan_task()
    task["extension_contract"]["exercise_dates"][-1] = "not-an-iso-date"
    task["extension_contract"]["exercise_to_swap_start"] = task[
        "extension_contract"
    ]["exercise_to_swap_start"][:2]

    with pytest.raises(ValueError, match=r"exercise_dates\[2\].*ISO"):
        benchmark_spec_overrides(task, root=ROOT)


def test_benchmark_preferred_method_uses_single_declared_construct():
    tasks = _benchmark_tasks()

    assert benchmark_preferred_method(tasks["F002"]) == "analytical"
    assert benchmark_preferred_method(tasks["F014"]) == "analytical"


def test_benchmark_request_description_makes_cap_request_explicit():
    tasks = _benchmark_tasks()

    description = benchmark_request_description(tasks["F003"], root=ROOT)

    assert description is not None
    assert "Instrument class: cap." in description
    assert "Rate index: USD-SOFR-3M." in description
    assert "Pricing model: black." in description
    assert "cap/floor" not in description.lower()


@pytest.mark.parametrize("task_id", ["T30", "T96"])
def test_legacy_lookback_contract_is_explicit_and_runtime_bindable(task_id):
    task = _legacy_tasks()[task_id]
    contract = task["benchmark_contract"]
    cross_validate = task["cross_validate"]

    description = benchmark_request_description(task, root=ROOT)
    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert task["task_disposition"] == "executable_pricing"
    assert "Goldman-Sosin-Gatto" not in task["title"]
    assert "Conze-Viswanathan" in task["title"]
    assert task["instrument_type"] == "lookback_option"
    assert task["market_scenario_id"] == "equity_barrier_smile"
    assert task["validation_policy"] == "invariants_and_cross_method"
    assert canonical_benchmark_instrument_type(task) == "lookback_option"
    assert contract["n_paths"] == 80_000
    assert contract["n_steps"] == 96
    assert contract["seed"] == 42
    assert cross_validate["reference_target"] == "conze_viswanathan_analytical"
    assert cross_validate["relations"] == {"mc_lookback": "within_tolerance"}
    assert cross_validate["tolerance_pct"] == pytest.approx(1.25)
    assert set(cross_validate["target_contracts"]) == {
        "mc_lookback",
        "conze_viswanathan_analytical",
    }
    assert description is not None
    assert "Notional: 1.0." in description
    assert "Lookback type: fixed_strike." in description
    assert "Monitoring style: continuous." in description
    assert "Exercise style: european." in description
    assert "Day count: act/365." in description
    assert "Monte Carlo controls: n_paths=80000, n_steps=96, seed=42." in description
    assert overrides["lookback_type"] == "fixed_strike"
    assert overrides["monitoring_style"] == "continuous"
    assert overrides["day_count"] is DayCountConvention.ACT_365
    assert overrides["n_paths"] == 80_000
    assert overrides["n_steps"] == 96
    assert overrides["seed"] == 42


def test_financepy_lookback_contract_declares_continuous_monitoring():
    task = _benchmark_tasks()["F011"]

    assert task["benchmark_contract"]["monitoring_style"] == "continuous"


def test_benchmark_request_description_surfaces_cap_model_specific_terms():
    tasks = _benchmark_tasks()

    shifted_description = benchmark_request_description(tasks["F004"], root=ROOT)
    sabr_description = benchmark_request_description(tasks["F005"], root=ROOT)

    assert shifted_description is not None
    assert "Pricing model: shifted_black." in shifted_description
    assert "Shift: 0.01." in shifted_description

    assert sabr_description is not None
    assert "Pricing model: sabr." in sabr_description
    assert "SABR parameters: alpha=0.025, beta=0.5, nu=0.35, rho=-0.2." in sabr_description


def test_p003_surfaces_discrete_monitoring_and_authored_numerical_controls():
    task = _extension_tasks()["P003"]

    description = benchmark_request_description(task, root=ROOT)
    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert description is not None
    assert "Monitoring: discrete (252 observations/year)." in description
    assert "Rebate: 0.0." in description
    assert "Monte Carlo controls: n_paths=120000, n_steps=252, seed=42." in description
    assert overrides["monitoring"] == "discrete"
    assert overrides["observations_per_year"] == 252
    assert overrides["rebate"] == pytest.approx(0.0)
    assert overrides["n_paths"] == 120_000
    assert overrides["n_steps"] == 252
    assert overrides["seed"] == 42


def test_p004_overrides_preserve_authored_callable_collar_schedule_and_control():
    task = _extension_tasks()["P004"]
    contract = task["extension_contract"]

    description = benchmark_request_description(task, root=ROOT)
    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert description is not None
    assert "Style: callable." in description
    assert "Irregular schedule: true." in description
    assert "Controller side: collar_payer." in description
    assert "Call action: terminate_remaining_strip." in description
    assert "Current-period treatment: fixed_unpaid_cashflow_survives." in description
    assert "Call settlement: cash USD 0.0 on exercise_date." in description
    assert "Accrual dates: 2024-11-15, 2025-01-15" in description
    assert "Fixing dates: 2024-11-15, 2025-01-15" in description
    assert "Payment dates: 2025-01-15, 2025-04-15" in description
    assert overrides["accrual_dates"] == tuple(
        date.fromisoformat(value) for value in contract["accrual_dates"]
    )
    assert overrides["fixing_dates"] == tuple(
        date.fromisoformat(value) for value in contract["fixing_dates"]
    )
    assert overrides["payment_dates"] == tuple(
        date.fromisoformat(value) for value in contract["payment_dates"]
    )
    assert len(overrides["accrual_dates"]) == len(overrides["fixing_dates"]) + 1
    assert len(overrides["accrual_dates"]) == len(overrides["payment_dates"]) + 1
    assert overrides["accrual_dates"][1] == date(2025, 1, 15)
    assert overrides["exercise_dates"] == tuple(
        date.fromisoformat(value) for value in contract["callable_dates"]
    )
    assert overrides["irregular_schedule"] is True
    assert overrides["exercise_style"] == "callable"
    assert overrides["collar_direction"] == "pay_cap_receive_floor"
    assert overrides["controller_side"] == "collar_payer"
    assert overrides["call_action"] == "terminate_remaining_strip"
    assert overrides["current_period_treatment"] == "fixed_unpaid_cashflow_survives"
    assert overrides["call_settlement"] == {
        "type": "cash",
        "amount": 0.0,
        "currency": "USD",
        "timing": "exercise_date",
    }
    assert overrides["rate_index"] == "USD-SOFR-3M"


def test_extension_request_description_surfaces_bermudan_swaption_contract():
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
    }

    description = benchmark_request_description(tasks["P005"], root=ROOT)

    assert description is not None
    assert "Price a strict physical Bermudan swaption" in description
    assert (
        "Exercise/start mapping: 2025-11-15->2025-11-19, "
        "2026-05-15->2026-05-19, 2026-11-15->2026-11-18."
        in description
    )
    assert "Co-terminal swap maturity: 2030-11-15." in description
    assert "Fixed leg: semiannual 30/360." in description
    assert "Floating leg: quarterly ACT/360 simple." in description
    assert "No analytical, European, Black, or cash-settlement fallback" in description


def test_extension_rainbow_overrides_keep_expiry_after_final_exercise_date():
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
    }

    rainbow = benchmark_spec_overrides(tasks["P001"], root=ROOT)

    assert rainbow["expiry_date"] == date(2025, 12, 15)
    assert rainbow["observation_dates"] == (
        date(2025, 3, 15),
        date(2025, 6, 15),
        date(2025, 9, 15),
        date(2025, 12, 15),
    )
    assert rainbow["exercise_dates"] == (
        date(2025, 6, 15),
        date(2025, 9, 15),
        date(2025, 12, 15),
    )


def test_extension_rainbow_overrides_bind_underliers_from_market_scenario():
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
    }

    rainbow = benchmark_spec_overrides(tasks["P001"], root=ROOT)

    assert rainbow["underliers"] == "AAPL,MSFT"
    assert rainbow["constituents"] == "AAPL,MSFT"
    assert rainbow["spots"] == "100.0,95.0"
    assert rainbow["vols"] == "0.2,0.25"
    assert rainbow["risk_free_rate"] == pytest.approx(0.05)


def test_t102_terminal_basket_contract_renders_and_overrides_without_hidden_defaults():
    task = _legacy_tasks()["T102"]

    description = benchmark_request_description(task, root=ROOT)
    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert description is not None
    assert "Build a pricer for: Two-asset European terminal best-of call" in description
    assert "Underliers: SPX, NDX." in description
    assert "Spots: 100.0, 95.0." in description
    assert "Volatilities: 0.2, 0.2." in description
    assert "Dividend yields: 0.0, 0.0." in description
    assert "Correlation: 1.0,0.35;0.35,1.0." in description
    assert "Expiry date: 2025-11-15." in description
    assert "Day count: ACT/365." in description
    assert "Monte Carlo controls: n_paths=40000, n_steps=1, seed=42, method=exact." in description
    assert overrides == {
        "notional": pytest.approx(10.0),
        "strike": pytest.approx(100.0),
        "option_type": "call",
        "payoff": "best_of_call",
        "style": "european",
        "underliers": "SPX,NDX",
        "constituents": "SPX,NDX",
        "spots": "100.0,95.0",
        "vols": "0.2,0.2",
        "dividend_yields": "0.0,0.0",
        "correlation": "1.0,0.35;0.35,1.0",
        "expiry_date": date(2025, 11, 15),
        "basket_style": "best_of",
        "risk_free_rate": pytest.approx(0.05),
        "n_paths": 40_000,
        "n_steps": 1,
        "seed": 42,
        "day_count": DayCountConvention.ACT_365,
        "mc_method": "exact",
    }


def test_t102_terminal_basket_rendering_and_overrides_ignore_title():
    task = _legacy_tasks()["T102"]
    renamed = deepcopy(task)
    renamed["title"] = "An unrelated display label that carries no economics"

    assert benchmark_request_description(renamed, root=ROOT) == benchmark_request_description(
        task,
        root=ROOT,
    )
    assert benchmark_spec_overrides(renamed, root=ROOT) == benchmark_spec_overrides(
        task,
        root=ROOT,
    )


def test_t102_terminal_basket_harness_reports_authored_acceptance_and_mc_controls():
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    plan = build_comparison_harness_plan(_legacy_tasks()["T102"])
    targets = {target.target_id: target for target in plan.targets}

    assert plan.reference_target == "stulz_rainbow"
    assert plan.tolerance_pct == pytest.approx(2.0)
    assert targets["stulz_rainbow"].is_reference is True
    assert targets["mc_rainbow"].relation == "within_tolerance"
    assert targets["mc_rainbow"].contract.spec_overrides == {
        "n_paths": 40_000,
        "n_steps": 1,
        "seed": 42,
        "mc_method": "exact",
    }
    assert _legacy_tasks()["T102"]["financepy_binding_id"] == (
        "financepy.equity.rainbow.stulz"
    )
    assert _legacy_tasks()["T102"]["benchmark_contract"]["num_assets"] == 2


def test_weighted_nth_to_default_overrides_preserve_name_exposure_and_spread():
    task = _extension_tasks()["P006"]

    description = benchmark_request_description(task, root=ROOT)
    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert description is not None
    assert "decimal annual market credit-spread quote" in description
    assert "not a running coupon" in description
    assert "Maturity: 2029-11-15 (authored tenor 5Y)." in description
    assert "Market curves: discount=usd_ois, credit=usd_ig." in description
    assert "Gaussian equicorrelation: 0.3." in description
    assert "Day count: ACT/360." in description
    assert "Settlement: terminal_if_rank_triggers_by_maturity." in description
    assert "Valuation measure: terminal_protection_leg_pv." in description
    assert "Marginal credit policy: homogeneous_representative_spread." in description
    assert "Recovery policy: homogeneous_common." in description
    assert "Correlation policy: gaussian_equicorrelation." in description
    assert "Discounting policy: deterministic." in description
    assert "Copula family: gaussian." in description
    assert (
        "Spread-to-hazard mapping: credit_triangle_spread_over_one_minus_recovery."
        in description
    )
    assert "Premium leg: none." in description
    assert "Spread-risk bump: 0.0001." in description
    assert overrides["basket_names"] == ("A", "B", "C", "D")
    assert overrides["basket_weights"] == pytest.approx((0.4, 0.2, 0.2, 0.2))
    assert overrides["spread"] == pytest.approx(0.025)
    assert overrides["notional"] == pytest.approx(5_000_000.0)
    assert overrides["recovery"] == pytest.approx(0.4)
    assert overrides["correlation"] == pytest.approx(0.3)
    assert overrides["day_count"] is DayCountConvention.ACT_360
    assert overrides["n_names"] == 4
    assert overrides["n_th"] == 2
    assert overrides["end_date"] == date(2029, 11, 15)
    assert "recovery_rate" not in overrides


@pytest.mark.parametrize(
    "field",
    (
        "product",
        "currency",
        "notional",
        "maturity_tenor",
        "nth",
        "basket_names",
        "basket_weights",
        "spread",
        "recovery_rate",
        "copula_family",
        "correlation",
        "day_count",
        "settlement_rule",
        "valuation_measure",
        "marginal_credit_policy",
        "recovery_policy",
        "correlation_policy",
        "discounting_policy",
        "spread_quote_convention",
        "spread_to_hazard_mapping",
        "premium_leg",
        "spread_risk_bump",
    ),
)
def test_p006_strict_contract_rejects_missing_authored_economics(field):
    task = _extension_tasks()["P006"]
    del task["extension_contract"][field]

    with pytest.raises(ValueError, match=field):
        benchmark_spec_overrides(task, root=ROOT)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"nth": 0}, "nth"),
        ({"basket_names": ["A", "A", "C", "D"]}, "unique"),
        ({"basket_weights": [0.4, 0.2, 0.2]}, "same length"),
        ({"basket_weights": [0.4, 0.2, 0.2, 0.1]}, "sum to 1"),
        ({"correlation": 1.0}, "correlation"),
        ({"day_count": "ACT/365F"}, "day_count"),
        ({"maturity_tenor": "five years"}, "maturity_tenor"),
        ({"copula_family": "student_t"}, "copula_family"),
        ({"premium_leg": "running"}, "premium_leg"),
        ({"spread_risk_bump": 0.001}, "spread_risk_bump"),
    ),
)
def test_p006_strict_contract_rejects_malformed_or_unsupported_shapes(
    updates,
    message,
):
    task = _extension_tasks()["P006"]
    task["extension_contract"].update(updates)

    with pytest.raises(ValueError, match=message):
        benchmark_spec_overrides(task, root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("notional", 4_000_000.0),
        ("nth", 1),
        ("basket_names", ["W", "X", "Y", "Z"]),
        ("basket_weights", [0.25, 0.25, 0.25, 0.25]),
        ("spread", 0.03),
        ("recovery_rate", 0.35),
        ("correlation", 0.25),
    ),
)
def test_p006_strict_contract_rejects_in_range_economic_identity_drift(field, value):
    task = _extension_tasks()["P006"]
    task["extension_contract"][field] = value

    with pytest.raises(ValueError, match=field):
        benchmark_spec_overrides(task, root=ROOT)


@pytest.mark.parametrize(
    "extra_field",
    (
        "correlation_matrix",
        "name_level_credit_curves",
        "recovery_vector",
        "unexpected_default",
    ),
)
def test_p006_strict_contract_rejects_extra_keys(extra_field):
    task = _extension_tasks()["P006"]
    task["extension_contract"][extra_field] = ["unsupported"]

    with pytest.raises(ValueError, match=extra_field):
        benchmark_spec_overrides(task, root=ROOT)


def test_p006_strict_contract_rejects_market_scenario_substitution():
    task = _extension_tasks()["P006"]
    task["market_scenario_id"] = "flat_usd_equity_vanilla"

    with pytest.raises(ValueError, match="usd_credit_ig"):
        benchmark_spec_overrides(task, root=ROOT)


def test_generic_nth_to_default_keeps_legacy_optional_default_behavior():
    task = {
        "id": "GENERIC-NTD",
        "title": "Generic first-to-default",
        "extension_contract": {
            "product": "nth_to_default",
            "basket_names": ["A", "B"],
            "maturity_tenor": "1Y",
        },
    }

    overrides = benchmark_spec_overrides(task, root=ROOT)

    assert overrides["n_th"] == 1
    assert overrides["basket_names"] == ("A", "B")
    assert "correlation" not in overrides
    assert "day_count" not in overrides


def test_extension_rainbow_overrides_reject_vector_underlier_mismatch():
    tasks = {
        task["id"]: task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
    }
    bad_task = {
        **tasks["P001"],
        "extension_contract": {
            **tasks["P001"]["extension_contract"],
            "underliers": ["AAPL"],
        },
    }

    with pytest.raises(ValueError, match="underlier count"):
        benchmark_spec_overrides(bad_task, root=ROOT)


def test_benchmark_spec_overrides_cover_fx_rates_cap_and_swaption_contracts():
    tasks = _benchmark_tasks()

    fx = benchmark_spec_overrides(tasks["F002"], root=ROOT)
    assert fx["fx_pair"] == "EURUSD"
    assert fx["foreign_discount_key"] == "EUR-DISC"
    assert fx["expiry_date"] == date(2025, 11, 15)

    cap = benchmark_spec_overrides(tasks["F003"], root=ROOT)
    assert cap["start_date"] == date(2024, 11, 15)
    assert cap["end_date"] == date(2029, 11, 15)
    assert cap["frequency"] is Frequency.QUARTERLY
    assert cap["day_count"] is DayCountConvention.ACT_360
    assert cap["rate_index"] == "USD-SOFR-3M"
    assert cap["instrument_class"] == "cap"
    assert cap["model"] == "black"

    swaption = benchmark_spec_overrides(tasks["F006"], root=ROOT)
    assert swaption["expiry_date"] == date(2025, 11, 15)
    assert swaption["swap_start"] == date(2025, 11, 15)
    assert swaption["swap_end"] == date(2030, 11, 15)
    assert swaption["swap_frequency"] is Frequency.SEMI_ANNUAL
    assert swaption["day_count"] is DayCountConvention.THIRTY_E_360
    assert swaption["is_payer"] is True

    cds = benchmark_spec_overrides(tasks["F007"], root=ROOT)
    assert cds["valuation_date"] == date(2024, 11, 15)
    assert cds["pricing_method"] == "analytical"
    assert cds["start_date"] == date(2024, 9, 20)
    assert cds["end_date"] == date(2029, 12, 20)
    assert cds["frequency"] is Frequency.QUARTERLY
    assert cds["day_count"] is DayCountConvention.ACT_360

    digital = benchmark_spec_overrides(tasks["F010"], root=ROOT)
    assert digital["cash_payoff"] == pytest.approx(10.0)
    assert digital["payout_type"] == "cash_or_nothing"

    lookback = benchmark_spec_overrides(tasks["F011"], root=ROOT)
    assert lookback["running_extreme"] == pytest.approx(100.0)

    chooser = benchmark_spec_overrides(tasks["F012"], root=ROOT)
    assert chooser["call_strike"] == pytest.approx(100.0)
    assert chooser["put_strike"] == pytest.approx(100.0)

    compound = benchmark_spec_overrides(tasks["F013"], root=ROOT)
    assert compound["outer_option_type"] == "call"
    assert compound["inner_option_type"] == "call"

    cliquet = benchmark_spec_overrides(tasks["F014"], root=ROOT)
    assert cliquet["observation_dates"] == (
        date(2024, 11, 18),
        date(2025, 2, 17),
        date(2025, 5, 19),
        date(2025, 8, 18),
        date(2025, 11, 17),
    )
    assert cliquet["day_count"] is DayCountConvention.THIRTY_E_360
    assert cliquet["time_day_count"] is DayCountConvention.ACT_365

    bounded_cliquet = benchmark_spec_overrides(_extension_tasks()["P007"], root=ROOT)
    assert bounded_cliquet["local_cap"] == pytest.approx(0.08)
    assert bounded_cliquet["local_floor"] == pytest.approx(0.0)
    assert bounded_cliquet["global_cap"] == pytest.approx(0.20)
    assert bounded_cliquet["global_floor"] == pytest.approx(0.0)

    variance_swap = benchmark_spec_overrides(tasks["F015"], root=ROOT)
    assert variance_swap["strike_variance"] == pytest.approx(0.04)
    assert variance_swap["replication_strikes"] == "60.0,80.0,100.0,120.0,140.0"
    assert variance_swap["replication_volatilities"] == "0.26,0.24,0.22,0.23,0.25"

    shifted_cap = benchmark_spec_overrides(tasks["F004"], root=ROOT)
    assert shifted_cap["model"] == "shifted_black"
    assert shifted_cap["shift"] == pytest.approx(0.01)

    sabr_cap = benchmark_spec_overrides(tasks["F005"], root=ROOT)
    assert sabr_cap["model"] == "sabr"
    assert sabr_cap["sabr"] == {
        "alpha": 0.025,
        "beta": 0.5,
        "rho": -0.2,
        "nu": 0.35,
    }

    barrier = benchmark_spec_overrides(tasks["F009"], root=ROOT)
    assert barrier["observations_per_year"] == 252


@pytest.mark.parametrize(
    ("task_id", "expected_spec_name", "expected_module_suffix"),
    [
        ("F002", "FXVanillaOptionSpec", "fxvanillaanalytical"),
        ("F003", "AgentCapSpec", "agentcap"),
        ("F009", "BarrierOptionSpec", "barrieroption"),
        ("F008", "BasketOptionSpec", "basketoption"),
        ("F010", "DigitalOptionSpec", "digitaloption"),
        ("F011", "LookbackOptionSpec", "lookbackoption"),
        ("F012", "ChooserOptionSpec", "chooseroption"),
        ("F013", "CompoundOptionSpec", "compoundoption"),
        ("F014", "CliquetOptionSpec", "cliquetoption"),
        ("F015", "VarianceSwapSpec", "varianceswap"),
    ],
)
def test_prepare_existing_task_uses_benchmark_normalization(
    task_id: str,
    expected_spec_name: str,
    expected_module_suffix: str,
):
    task = _benchmark_tasks()[task_id]

    prepared = prepare_existing_task(task)

    assert prepared.spec_schema is not None
    assert prepared.spec_schema.spec_name == expected_spec_name
    assert prepared.payoff_cls.__module__.endswith(expected_module_suffix)
