from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml


def _write_yaml(root: Path, name: str, payload) -> None:
    (root / name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _codes(report) -> set[str]:
    return {issue.code for issue in report.blocking_issues}


def _legacy_expected_block_task(**overrides):
    task = {
        "id": "T09",
        "title": "Step-up callable bond (varying coupon schedule)",
        "status": "done",
        "task_definition_manifest": "TASKS_PROOF_LEGACY.yaml",
        "description": "Do not substitute a flat coupon.",
        "task_disposition": "expected_honest_block",
        "disposition_reason": "Variable coupon economics are not supported.",
        "expected_outcome": "honest_block",
        "expected_blocker_ids": [
            "semantic_product_contract_gap:variable_coupon_schedule"
        ],
        "honest_block_contract": {
            "reason": "callable_bond_variable_coupon_schedule_missing",
            "summary": "A dated step-up coupon schedule is required.",
            "packet_type": "semantic_product_contract_gap",
            "missing_capabilities": ["variable_coupon_schedule"],
            "suggested_action": "Implement the dated coupon schedule primitive.",
            "follow_on_issue": "QUA-1251",
        },
    }
    task.update(overrides)
    return task


def _legacy_lookback_task(task_id: str = "T30"):
    from trellis.agent.task_manifests import load_task_manifest

    return next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == task_id
    )


def _extension_task(task_id: str):
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    return next(task for task in payload["tasks"] if task["id"] == task_id)


def _audit_single_extension_task(tmp_path: Path, task):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {
                str(task["market_scenario_id"]): {"description": "test market"}
            },
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": 1, "tasks": [task]},
    )
    return audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )


def test_missing_and_malformed_manifests_fail_closed(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    missing = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )
    assert _codes(missing) == {"manifest.missing"}

    _write_yaml(tmp_path, "TASKS_EXTENSION.yaml", "not-a-task-manifest")
    malformed = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )
    assert "manifest.invalid_top_level" in _codes(malformed)


def test_duplicate_task_ids_are_rejected_across_manifests(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    task = {"id": "P900", "title": "Duplicate", "status": "pending"}
    _write_yaml(tmp_path, "TASKS_EXTENSION.yaml", {"version": 1, "tasks": [task]})
    _write_yaml(tmp_path, "TASKS_NEGATIVE.yaml", {"version": 1, "tasks": [task]})

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml", "TASKS_NEGATIVE.yaml"),
    )

    duplicates = [
        issue for issue in report.blocking_issues if issue.code == "task.duplicate_id"
    ]
    assert len(duplicates) == 2
    assert {issue.manifest for issue in duplicates} == {
        "TASKS_EXTENSION.yaml",
        "TASKS_NEGATIVE.yaml",
    }


def test_financepy_contract_requires_references_economics_and_acceptance(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(tmp_path, "MARKET_SCENARIOS.yaml", {"version": 1, "scenarios": {}})
    _write_yaml(tmp_path, "FINANCEPY_BINDINGS.yaml", {"version": 1, "bindings": {}})
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "F900",
                    "title": "Incomplete parity task",
                    "status": "pending",
                    "description": "Price a product.",
                    "instrument_type": "european_option",
                    "construct": "analytical",
                    "market_scenario_id": "missing-market",
                    "financepy_binding_id": "missing-binding",
                    "benchmark_contract": {},
                    "cross_validate": {"external": ["financepy"]},
                }
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_BENCHMARK_FINANCEPY.yaml",),
    )

    assert {
        "reference.unknown_market_scenario",
        "reference.unknown_financepy_binding",
        "financepy.missing_economic_contract",
        "financepy.missing_tolerance",
    } <= _codes(report)


def test_contract_envelopes_require_typed_meaningful_content(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {"version": 1, "scenarios": {"scenario": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "FINANCEPY_BINDINGS.yaml",
        {"version": 1, "bindings": {"binding": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "F900",
                    "title": "Near-empty contract",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": ["analytical"],
                    "market_scenario_id": "scenario",
                    "financepy_binding_id": "binding",
                    "benchmark_contract": {
                        "product": "equity_vanilla",
                        "terms": {"strike": ""},
                    },
                    "cross_validate": {"tolerance_pct": 1.0},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "P900",
                    "title": "Bad method contract",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": ["analytical", ""],
                    "market_scenario_id": "scenario",
                    "extension_contract": {
                        "product": "equity_vanilla",
                        "terms": [""],
                    },
                    "validation_policy": "invented_policy",
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_NEGATIVE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "N900",
                    "title": "Empty clarification",
                    "status": "pending",
                    "description": "Price it.",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "clarification_requested",
                    "clarification_contract": {"missing_fields": [""]},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_FPML_CONFORMANCE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "FPC900",
                    "title": "Wrong task kind",
                    "status": "pending",
                    "task_kind": "other",
                    "expected_outcome": "honest_block",
                    "market_scenario_id": "scenario",
                    "fpml": {
                        "fixture": "fixture.xml",
                        "source_view": "confirmation",
                        "source_version": "5-12",
                    },
                    "expected_blocker_ids": [""],
                }
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=(
            "TASKS_BENCHMARK_FINANCEPY.yaml",
            "TASKS_EXTENSION.yaml",
            "TASKS_NEGATIVE.yaml",
            "TASKS_FPML_CONFORMANCE.yaml",
        ),
    )

    assert {
        "financepy.invalid_construct",
        "financepy.missing_economic_contract",
        "extension.invalid_construct",
        "extension.missing_economic_contract",
        "extension.invalid_validation_policy",
        "negative.missing_fields",
        "fpml.invalid_task_kind",
        "fpml.missing_blockers",
    } <= _codes(report)


@pytest.mark.parametrize(
    "missing_field",
    ["monitoring", "observations_per_year", "rebate", "n_paths", "n_steps", "seed"],
)
def test_p003_requires_explicit_monitoring_and_numerical_controls(
    tmp_path, missing_field
):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    p003 = next(task for task in payload["tasks"] if task["id"] == "P003")
    p003["extension_contract"].pop(missing_field, None)
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"fx_barrier_smile": {"description": "test FX market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [p003]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert "extension.fx_barrier_missing_field" in _codes(report)


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("monitoring", "continuous", "extension.fx_barrier_invalid_monitoring"),
        ("observations_per_year", 12, "extension.fx_barrier_invalid_controls"),
        ("rebate", 1.0, "extension.fx_barrier_invalid_controls"),
        ("n_paths", 50_000, "extension.fx_barrier_invalid_controls"),
        ("n_steps", 365, "extension.fx_barrier_invalid_controls"),
        ("seed", -1, "extension.fx_barrier_invalid_controls"),
    ],
)
def test_p003_rejects_contract_drift(tmp_path, field, value, expected_code):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    p003 = next(task for task in payload["tasks"] if task["id"] == "P003")
    p003["extension_contract"][field] = value
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"fx_barrier_smile": {"description": "test FX market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [p003]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert expected_code in _codes(report)


def test_callable_collar_contract_requires_exact_schedule_and_honest_block(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    p004 = next(task for task in payload["tasks"] if task["id"] == "P004")
    p004["extension_contract"].pop("fixing_dates", None)
    p004.pop("expected_outcome", None)
    p004.pop("expected_blocker_ids", None)
    p004.pop("honest_block_contract", None)
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"usd_rates_smile": {"description": "test rates market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [p004]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert {
        "extension.callable_collar_missing_field",
        "extension.callable_collar_missing_honest_block",
    } <= _codes(report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("product", "rate_cap_floor"), ("style", "european")],
)
def test_p004_cannot_bypass_callable_collar_validation_by_changing_identity(
    tmp_path, field, replacement
):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    p004 = next(task for task in payload["tasks"] if task["id"] == "P004")
    p004["extension_contract"][field] = replacement
    p004.pop("expected_outcome", None)
    p004.pop("expected_blocker_ids", None)
    p004.pop("honest_block_contract", None)
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"usd_rates_smile": {"description": "test rates market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [p004]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert {
        "extension.callable_collar_invalid_semantics",
        "extension.callable_collar_missing_honest_block",
    } <= _codes(report)


@pytest.mark.parametrize(
    "bad_schedule",
    ["fixing_order", "payment_before_period_end", "yaml_timestamp"],
)
def test_p004_callable_collar_rejects_invalid_period_chronology(tmp_path, bad_schedule):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    p004 = next(task for task in payload["tasks"] if task["id"] == "P004")
    if bad_schedule == "fixing_order":
        p004["extension_contract"]["fixing_dates"][1] = p004["extension_contract"][
            "fixing_dates"
        ][0]
    elif bad_schedule == "payment_before_period_end":
        p004["extension_contract"]["payment_dates"][0] = "2025-01-14"
    else:
        p004["extension_contract"]["fixing_dates"][0] = yaml.safe_load(
            "2024-11-15T00:00:00"
        )
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"usd_rates_smile": {"description": "test rates market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [p004]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert "extension.callable_collar_invalid_schedule" in _codes(report)


def test_callable_collar_exact_values_are_scoped_to_p004(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "TASKS_EXTENSION.yaml").read_text(encoding="utf-8"))
    task = next(task for task in payload["tasks"] if task["id"] == "P004")
    task["id"] = "P900"
    task["extension_contract"]["rate_index"] = "EUR-EURIBOR-3M"
    task["extension_contract"]["collar_direction"] = "receive_cap_pay_floor"
    task["extension_contract"]["controller_side"] = "collar_receiver"
    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {"usd_rates_smile": {"description": "test rates market"}},
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {"version": payload["version"], "tasks": [task]},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )

    assert not {
        issue.code
        for issue in report.blocking_issues
        if issue.code.startswith("extension.callable_collar_")
    }


@pytest.mark.parametrize(
    "missing_field",
    (
        "exercise_to_swap_start",
        "maturity_date",
        "settlement_type",
        "currency",
        "discount_curve_id",
        "forecast_curve_id",
        "hull_white_parameter_source",
        "fixed_day_count",
        "fixed_coupon",
        "fixed_frequency",
        "fixed_calendar_name",
        "fixed_business_day_adjustment",
        "fixed_stub_rule",
        "fixed_roll_convention",
        "fixed_payment_lag_business_days",
        "float_frequency",
        "floating_day_count",
        "floating_calendar_name",
        "floating_business_day_adjustment",
        "floating_stub_rule",
        "floating_roll_convention",
        "floating_fixing_lag_business_days",
        "floating_reset_lag_business_days",
        "floating_payment_lag_business_days",
        "floating_rate_index",
        "floating_compounding",
        "floating_gearing",
        "floating_spread",
        "model_time_day_count",
        "model_time_calendar_name",
        "projection_policy",
        "lattice_steps",
        "lattice_date_tolerance_days",
    ),
)
def test_p005_requires_every_physical_swap_tail_input(tmp_path, missing_field):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    task["extension_contract"].pop(missing_field, None)

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_missing_field" in _codes(report)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda task: task.__setitem__("market_scenario_id", "usd_rates_smile"),
        lambda task: task.__setitem__("instrument_type", "swaption"),
        lambda task: task.__setitem__("construct", ["lattice", "monte_carlo"]),
        lambda task: task["extension_contract"].__setitem__("product", "swaption"),
        lambda task: task["extension_contract"].__setitem__(
            "discount_curve_id", "USD-OIS"
        ),
        lambda task: task["extension_contract"].__setitem__(
            "exercise_to_swap_start",
            [
                ["2025-11-15", "2025-11-15"],
                ["2026-05-15", "2026-05-19"],
                ["2026-11-15", "2026-11-18"],
            ],
        ),
        lambda task: task["extension_contract"].__setitem__(
            "fixed_day_count", "ACT/ACT ICMA"
        ),
        lambda task: task["extension_contract"].__setitem__("lattice_steps", 1098),
        lambda task: task["extension_contract"].__setitem__(
            "lattice_date_tolerance_days", 1
        ),
        lambda task: task["cross_validate"].__setitem__("tolerance_pct", 4.99),
    ),
)
def test_p005_rejects_identity_economic_and_acceptance_drift(tmp_path, mutation):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    mutation(task)

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_invalid_contract" in _codes(report)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lattice_date_tolerance_days", False),
        ("floating_gearing", True),
        ("floating_spread", False),
    ),
)
def test_p005_rejects_booleans_for_numeric_contract_fields(tmp_path, field, value):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    task["extension_contract"][field] = value

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_invalid_contract" in _codes(report)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda task: task["cross_validate"]["target_contracts"][
            "physical_bermudan_swaption_lattice"
        ].__setitem__("route_id", "exercise_lattice"),
        lambda task: task["cross_validate"]["target_contracts"][
            "physical_bermudan_swaption_lattice"
        ].__setitem__(
            "backend_binding_id",
            "trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree",
        ),
        lambda task: task["cross_validate"]["target_contracts"][
            "physical_bermudan_swaption_monte_carlo"
        ].__setitem__("route_id", "exercise_monte_carlo"),
        lambda task: task["cross_validate"]["target_contracts"][
            "physical_bermudan_swaption_monte_carlo"
        ].__setitem__(
            "backend_binding_id", "trellis.models.monte_carlo.engine.MonteCarloEngine"
        ),
        lambda task: task["cross_validate"].__setitem__(
            "internal", ["physical_bermudan_swaption_lattice"]
        ),
    ),
)
def test_p005_rejects_route_fallbacks_and_executable_mc_claims(tmp_path, mutation):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    mutation(task)

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_invalid_targets" in _codes(report)


def test_p005_reports_malformed_internal_targets_without_raising(tmp_path):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    task["cross_validate"]["internal"] = 1

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_invalid_targets" in _codes(report)


def test_p005_requests_price_without_native_greeks(tmp_path):
    from copy import deepcopy

    task = deepcopy(_extension_task("P005"))
    task["description"] += " Return native Greeks."

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.bermudan_swaption_invalid_output_request" in _codes(report)


def test_p006_exact_terminal_protection_contract_passes_manifest_audit(tmp_path):
    report = _audit_single_extension_task(tmp_path, _extension_task("P006"))

    assert not {
        issue.code
        for issue in report.blocking_issues
        if issue.code.startswith("extension.nth_to_default_")
    }


@pytest.mark.parametrize(
    "description",
    (
        "Price an nth-to-default basket with custom weights.",
        "Price the P006 basket including a running premium leg.",
        "Price the P006 basket and return DV01.",
    ),
)
def test_p006_rejects_vague_or_conflicting_description(tmp_path, description):
    from copy import deepcopy

    task = deepcopy(_extension_task("P006"))
    task["description"] = description

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.nth_to_default_invalid_description" in _codes(report)


@pytest.mark.parametrize(
    "missing_field",
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
def test_p006_requires_every_terminal_protection_input(tmp_path, missing_field):
    from copy import deepcopy

    task = deepcopy(_extension_task("P006"))
    task["extension_contract"].pop(missing_field, None)

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.nth_to_default_missing_field" in _codes(report)


@pytest.mark.parametrize(
    "field,value",
    (
        ("currency", "EUR"),
        ("notional", 0.0),
        ("notional", True),
        ("maturity_tenor", "10Y"),
        ("nth", 0),
        ("nth", 5),
        ("nth", True),
        ("basket_names", ["A", "A", "C", "D"]),
        ("basket_names", ["A", "B", "C"]),
        ("basket_weights", [0.4, 0.2, 0.4]),
        ("basket_weights", [0.4, 0.2, 0.2, 0.1]),
        ("basket_weights", [0.4, 0.2, 0.2, -0.2]),
        ("spread", 1.0),
        ("spread", True),
        ("recovery_rate", 1.0),
        ("recovery_rate", True),
        ("copula_family", "student_t"),
        ("correlation", 1.0),
        ("correlation", True),
        ("day_count", "ACT/365F"),
        ("settlement_rule", "payment_at_default"),
        ("valuation_measure", "trade_npv"),
        ("marginal_credit_policy", "heterogeneous_name_curves"),
        ("recovery_policy", "name_level"),
        ("correlation_policy", "correlation_matrix"),
        ("discounting_policy", "stochastic"),
        ("spread_quote_convention", "running_coupon"),
        ("spread_to_hazard_mapping", "bootstrap"),
        ("premium_leg", "running"),
        ("spread_risk_bump", 0.0),
        ("spread_risk_bump", True),
    ),
)
def test_p006_rejects_economic_and_policy_drift(tmp_path, field, value):
    from copy import deepcopy

    task = deepcopy(_extension_task("P006"))
    task["extension_contract"][field] = value

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.nth_to_default_invalid_contract" in _codes(report)


@pytest.mark.parametrize(
    "extra_field",
    (
        "correlation_matrix",
        "name_level_credit_curves",
        "recovery_vector",
        "running_premium_leg",
        "unexpected_default",
    ),
)
def test_p006_rejects_unsupported_or_extra_contract_keys(tmp_path, extra_field):
    from copy import deepcopy

    task = deepcopy(_extension_task("P006"))
    task["extension_contract"][extra_field] = [0.1, 0.2]

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.nth_to_default_invalid_contract" in _codes(report)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda task: task["cross_validate"].__setitem__(
            "internal", ["weighted_rank_analytical"]
        ),
        lambda task: task["cross_validate"].__setitem__("tolerance_pct", 2.99),
        lambda task: task["cross_validate"].__setitem__(
            "output_tolerances_pct", {"spread_cs01": 14.99}
        ),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_analytical"
        ].__setitem__("route_id", "nth_to_default_analytical"),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_analytical"
        ].__setitem__(
            "backend_binding_id", "trellis.instruments.nth_to_default.price_nth_to_default_basket"
        ),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_monte_carlo"
        ].__setitem__("route_id", "nth_to_default_monte_carlo"),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_monte_carlo"
        ]["spec_overrides"].__setitem__("n_paths", 100_000),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_monte_carlo"
        ]["spec_overrides"].__setitem__("n_paths", True),
        lambda task: task["cross_validate"]["target_contracts"][
            "weighted_rank_monte_carlo"
        ]["spec_overrides"].__setitem__("seed", 7),
    ),
)
def test_p006_rejects_target_binding_control_and_tolerance_drift(tmp_path, mutation):
    from copy import deepcopy

    task = deepcopy(_extension_task("P006"))
    mutation(task)

    report = _audit_single_extension_task(tmp_path, task)

    assert "extension.nth_to_default_invalid_targets" in _codes(report)


def test_tolerances_accept_exact_zero_and_reject_non_finite_values(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {"version": 1, "scenarios": {"scenario": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "FINANCEPY_BINDINGS.yaml",
        {"version": 1, "bindings": {"binding": {"description": "test"}}},
    )
    base = {
        "title": "Tolerance contract",
        "status": "pending",
        "description": "Price it.",
        "instrument_type": "option",
        "construct": "analytical",
        "market_scenario_id": "scenario",
        "financepy_binding_id": "binding",
        "benchmark_contract": {"product": "equity_vanilla", "spot": 100.0},
    }
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {"id": "F900", **base, "cross_validate": {"tolerance_pct": 0.0}},
                {"id": "F901", **base, "cross_validate": {"tolerance_pct": math.nan}},
                {"id": "F902", **base, "cross_validate": {"tolerance_pct": math.inf}},
                {
                    "id": "F903",
                    **base,
                    "benchmark_contract": {
                        "product": "equity_vanilla",
                        "spot": 100.0,
                        "strike": math.nan,
                    },
                    "cross_validate": {"tolerance_pct": 0.0},
                },
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_BENCHMARK_FINANCEPY.yaml",),
    )

    tolerance_issues = [
        issue for issue in report.blocking_issues if issue.code == "financepy.missing_tolerance"
    ]
    assert {issue.task_id for issue in tolerance_issues} == {"F901", "F902"}
    economic_issues = [
        issue
        for issue in report.blocking_issues
        if issue.code == "financepy.missing_economic_contract"
    ]
    assert {issue.task_id for issue in economic_issues} == {"F903"}


def test_extension_negative_fpml_and_framework_contract_shapes_are_strict(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {
                "scenario": {
                    "source": "mock",
                    "as_of": "2024-11-15",
                    "description": "Test scenario.",
                    "selected_components": {},
                    "constructor": {"kind": "test"},
                }
            },
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "P900",
                    "title": "Incomplete extension",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": "analytical",
                    "market_scenario_id": "scenario",
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_NEGATIVE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "N900",
                    "title": "Bad negative row",
                    "status": "pending",
                    "description": "Price an option.",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "clarification_requested",
                    "clarification_contract": {},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_FPML_CONFORMANCE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "FPC900",
                    "title": "Bad FpML row",
                    "status": "pending",
                    "task_kind": "fpml_conformance",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "pricing_success",
                    "fpml": {},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "FRAMEWORK_TASKS.yaml",
        [
            {
                "id": "E900",
                "title": "Bad framework row",
                "status": "pending",
                "construct": "framework",
                "trigger_after": [""],
            }
        ],
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=(
            "TASKS_EXTENSION.yaml",
            "TASKS_NEGATIVE.yaml",
            "TASKS_FPML_CONFORMANCE.yaml",
            "FRAMEWORK_TASKS.yaml",
        ),
    )

    assert {
        "extension.missing_economic_contract",
        "extension.missing_validation_policy",
        "negative.missing_fields",
        "fpml.missing_source_contract",
        "fpml.missing_native_contract",
        "fpml.missing_tolerance",
        "framework.missing_delivery_contract",
    } <= _codes(report)


def test_legacy_debt_must_match_checked_baseline(tmp_path):
    from trellis.agent.task_manifest_validation import (
        audit_task_manifests,
        legacy_issue_digest,
        legacy_task_fingerprint,
    )

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "T900",
                    "title": "Title-only proof",
                    "status": "pending",
                    "construct": "analytical",
                    "new_component": "example",
                    "cross_validate": {"analytical": "example"},
                }
            ],
        },
    )

    first = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert "legacy.baseline_missing" in _codes(first)
    assert first.legacy_issues

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY_BASELINE.yaml",
        {
            "version": 1,
            "manifest": "TASKS_PROOF_LEGACY.yaml",
            "issue_count": len(first.legacy_issues),
            "issue_digest": legacy_issue_digest(first.legacy_issues),
            "task_fingerprint": legacy_task_fingerprint(
                yaml.safe_load(
                    (tmp_path / "TASKS_PROOF_LEGACY.yaml").read_text()
                )["tasks"]
            ),
            "policy": "exact_issue_identity_and_task_content",
        },
    )
    accepted = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert not accepted.blocking_issues
    assert accepted.legacy_issues

    payload = yaml.safe_load((tmp_path / "TASKS_PROOF_LEGACY.yaml").read_text())
    payload["tasks"][0]["title"] = "Semantically changed title-only proof"
    _write_yaml(tmp_path, "TASKS_PROOF_LEGACY.yaml", payload)
    changed = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert "legacy.baseline_mismatch" in _codes(changed)


def test_manifest_paths_cannot_escape_the_repository_root(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("../TASKS_EXTENSION.yaml",),
    )

    assert _codes(report) == {"manifest.outside_root"}


def test_legacy_baseline_path_cannot_escape_the_repository_root(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY.yaml",
        {"version": 1, "tasks": []},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
        legacy_baseline_name="../TASKS_PROOF_LEGACY_BASELINE.yaml",
    )

    assert _codes(report) == {"legacy.baseline_outside_root"}


def test_incomplete_selected_legacy_task_stops_before_market_or_build(monkeypatch, tmp_path):
    from scripts import run_tasks
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    task = {
        "id": "T900",
        "title": "Sparse legacy request",
        "status": "pending",
        "task_definition_manifest": "TASKS_PROOF_LEGACY.yaml",
    }
    calls: list[str] = []

    def unexpected_market():
        calls.append("market")
        return object()

    def unexpected_build(*args, **kwargs):
        calls.append("build")
        return {}

    monkeypatch.setattr(run_tasks, "build_market_state", unexpected_market)
    monkeypatch.setattr(run_tasks, "run_task", unexpected_build)

    with pytest.raises(TaskManifestValidationError, match="legacy.missing_disposition"):
        run_tasks.run_block([task], str(tmp_path / "results.json"))

    assert calls == []


def test_validated_legacy_expected_honest_block_is_admitted_for_runtime():
    from trellis.agent.task_manifest_validation import assert_executable_task_selection

    assert_executable_task_selection([_legacy_expected_block_task()])


@pytest.mark.parametrize("task_id", ("T30", "T96"))
def test_authored_legacy_lookback_comparison_is_admitted_for_runtime(task_id):
    from trellis.agent.task_manifest_validation import assert_executable_task_selection

    assert_executable_task_selection([_legacy_lookback_task(task_id)])


def test_legacy_lookback_validation_uses_the_callers_repository_root(tmp_path):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )
    from trellis.agent.task_manifests import load_task_manifest

    repository_root = Path(__file__).resolve().parents[2]
    scenarios = yaml.safe_load(
        (repository_root / "MARKET_SCENARIOS.yaml").read_text(encoding="utf-8")
    )
    scenarios["scenarios"]["equity_barrier_smile"]["description"] = (
        "Custom-root lookback proof scenario."
    )
    legacy = yaml.safe_load(
        (repository_root / "TASKS_PROOF_LEGACY.yaml").read_text(encoding="utf-8")
    )
    task = next(item for item in legacy["tasks"] if item["id"] == "T30")
    _write_yaml(tmp_path, "MARKET_SCENARIOS.yaml", scenarios)
    _write_yaml(tmp_path, "TASKS_PROOF_LEGACY.yaml", [task])

    materialized = load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=tmp_path)

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection(materialized)
    assert "legacy.lookback_invalid_contract" in _codes(exc_info.value.report)
    assert_executable_task_selection(materialized, root=tmp_path)


@pytest.mark.parametrize("task_id", ("T30", "T96"))
def test_fixed_lookback_proofs_cannot_be_reclassified_as_honest_blocks(task_id):
    from copy import deepcopy

    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = deepcopy(_legacy_lookback_task(task_id))
    task.update(
        {
            "task_disposition": "expected_honest_block",
            "disposition_reason": "Attempt to bypass the fixed pricing proof.",
            "expected_outcome": "honest_block",
            "expected_blocker_ids": ["semantic_product_contract_gap:lookback"],
            "honest_block_contract": {
                "reason": "lookback_missing",
                "summary": "Skip the authored pricing proof.",
                "packet_type": "semantic_product_contract_gap",
                "missing_capabilities": ["lookback"],
                "suggested_action": "Block instead of executing.",
            },
        }
    )

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection([task])

    assert "legacy.lookback_invalid_contract" in _codes(exc_info.value.report)


@pytest.mark.parametrize(
    ("mutation"),
    (
        lambda task: task["benchmark_contract"].__setitem__("monitoring_style", "discrete"),
        lambda task: task["benchmark_contract"].__setitem__("n_steps", 95),
        lambda task: task["benchmark_contract"].__setitem__("dividend_rate", 0.25),
        lambda task: task["cross_validate"].__setitem__(
            "analytical", "conze_viswanathan_analytical"
        ),
        lambda task: task.update(
            {
                "expected_outcome": "honest_block",
                "expected_blocker_ids": ["semantic_product_contract_gap:lookback"],
                "honest_block_contract": {
                    "reason": "lookback_missing",
                    "summary": "Skip the authored pricing proof.",
                    "packet_type": "semantic_product_contract_gap",
                    "missing_capabilities": ["lookback"],
                    "suggested_action": "Block instead of executing.",
                },
            }
        ),
        lambda task: task["cross_validate"].__setitem__("reference_target", "mc_lookback"),
        lambda task: task["cross_validate"].__setitem__("tolerance_pct", 5.0),
        lambda task: task["cross_validate"]["target_contracts"].pop(
            "conze_viswanathan_analytical"
        ),
        lambda task: task["cross_validate"]["target_contracts"][
            "mc_lookback"
        ].__setitem__("spec_overrides", {"n_paths": 2}),
        lambda task: task.__setitem__(
            "market",
            {
                "scenario_contract": {
                    "scenario_id": "equity_barrier_smile",
                    "constructor_kind": "flat",
                    "black_vol": 0.01,
                }
            },
        ),
        lambda task: task.__setitem__(
            "comparison_regime",
            {"regime_family": "short_rate", "flat_sigma": 0.01},
        ),
    ),
)
def test_legacy_lookback_comparison_rejects_semantic_contract_drift(mutation):
    from copy import deepcopy

    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = deepcopy(_legacy_lookback_task())
    mutation(task)

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection([task])

    assert "legacy.lookback_invalid_contract" in _codes(exc_info.value.report)


def test_run_block_reports_real_t09_as_non_actionable_honest_block(
    monkeypatch, tmp_path
):
    from scripts import run_tasks
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import run_task as runtime_run_task

    task = next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == "T09"
    )
    output_path = tmp_path / "t09-results.json"

    monkeypatch.setattr(run_tasks, "build_market_state", lambda: object())

    def run_t09(selected_task, market_state, **kwargs):
        return runtime_run_task(
            selected_task,
            market_state,
            task_run_storage_root=tmp_path / "task-runs",
            **kwargs,
        )

    monkeypatch.setattr(run_tasks, "run_task", run_t09)

    run_tasks.run_block([task], str(output_path))

    summary = json.loads((tmp_path / "t09-results_summary.json").read_text())
    assert {
        key: summary["totals"][key]
        for key in (
            "successes",
            "expectation_passes",
            "honest_blocks",
            "actionable_failures",
        )
    } == {
        "successes": 0,
        "expectation_passes": 1,
        "honest_blocks": 1,
        "actionable_failures": 0,
    }


@pytest.mark.parametrize(
    "missing_field",
    ("reason", "summary", "packet_type", "missing_capabilities", "suggested_action"),
)
def test_legacy_expected_honest_block_requires_complete_contract(missing_field):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _legacy_expected_block_task()
    task["honest_block_contract"].pop(missing_field)

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection([task])

    assert "legacy.invalid_honest_block" in _codes(exc_info.value.report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_blocker_ids", ["semantic_product_contract_gap:other"]),
        ("missing_capabilities", ["other"]),
        ("follow_on_issue", "QUA-9999"),
    ],
)
def test_t09_expected_block_requires_variable_coupon_repair_contract(field, value):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _legacy_expected_block_task()
    if field == "expected_blocker_ids":
        task[field] = value
    else:
        task["honest_block_contract"][field] = value

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection([task])

    assert "legacy.t09_invalid_honest_block" in _codes(exc_info.value.report)


@pytest.mark.parametrize("disposition", ("research_hold", "proof_hold", "rewrite_candidate"))
def test_legacy_hold_dispositions_remain_non_executable(disposition):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_executable_task_selection,
    )

    task = _legacy_expected_block_task(
        id="T900",
        task_disposition=disposition,
        disposition_reason="Retained outside executable pricing.",
    )

    with pytest.raises(TaskManifestValidationError) as exc_info:
        assert_executable_task_selection([task])

    assert "legacy.non_executable_disposition" in _codes(exc_info.value.report)


def test_t03_t83_t85_declare_current_nonpricing_contracts():
    from trellis.agent.task_manifests import load_task_manifest

    expected = {
        "T03": (
            "research_hold",
            {
                "experiment_specification",
                "valuation_date",
                "discount_curve",
                "short_rate_model_parameters",
                "tree_grid",
                "convergence_metric",
                "acceptance_tolerance",
            },
        ),
        "T83": (
            "proof_hold",
            {
                "bond_cashflows",
                "settlement",
                "discount_curve",
                "krd_bucket_grid",
                "bump_size",
                "interpolation_policy",
                "acceptance_tolerance",
            },
        ),
        "T85": (
            "rewrite_candidate",
            {
                "bond_cashflows",
                "settlement",
                "dirty_or_clean_price",
                "accrued_interest_convention",
                "yield_compounding_convention",
                "yield_frequency",
                "day_count_convention",
            },
        ),
    }
    loaded_tasks = load_task_manifest("TASKS_PROOF_LEGACY.yaml")
    tasks = {
        task["id"]: task
        for task in loaded_tasks
        if task["id"] in expected
    }

    assert not [task["id"] for task in loaded_tasks if task.get("blocked_by")]
    assert set(tasks) == set(expected)
    for task_id, (disposition, missing_inputs) in expected.items():
        task = tasks[task_id]
        assert task["task_disposition"] == disposition
        assert set(task["missing_inputs"]) == missing_inputs
        assert task["disposition_reason"].strip()
        assert "blocked_by" not in task
        assert task["status"] == "blocked"


def test_real_t03_t83_t85_holds_stop_before_market_or_build(monkeypatch, tmp_path):
    from scripts import run_tasks
    from trellis.agent.task_manifest_validation import TaskManifestValidationError
    from trellis.agent.task_manifests import load_task_manifest

    tasks = [
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] in {"T03", "T83", "T85"}
    ]
    calls: list[str] = []

    monkeypatch.setattr(
        run_tasks,
        "build_market_state",
        lambda: calls.append("market"),
    )
    monkeypatch.setattr(
        run_tasks,
        "run_task",
        lambda *args, **kwargs: calls.append("build"),
    )

    with pytest.raises(TaskManifestValidationError) as exc_info:
        run_tasks.run_block(tasks, str(tmp_path / "results.json"))

    issues = exc_info.value.report.blocking_issues
    assert {issue.code for issue in issues} == {"legacy.non_executable_disposition"}
    assert {issue.task_id for issue in issues} == {"T03", "T83", "T85"}
    assert calls == []


@pytest.mark.parametrize("task_id", ("T03", "T83", "T85"))
def test_shared_task_runtime_rejects_governed_holds_before_build(task_id):
    from trellis.agent import task_runtime
    from trellis.agent.task_manifest_validation import TaskManifestValidationError
    from trellis.agent.task_manifests import load_task_manifest

    task = next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == task_id
    )
    calls: list[str] = []

    with pytest.raises(TaskManifestValidationError) as exc_info:
        task_runtime.run_task(
            task,
            object(),
            build_fn=lambda *args, **kwargs: calls.append("build"),
        )

    issues = exc_info.value.report.blocking_issues
    assert {issue.code for issue in issues} == {"legacy.non_executable_disposition"}
    assert {issue.task_id for issue in issues} == {task_id}
    assert calls == []


def test_analytical_stress_defaults_exclude_governed_nonpricing_tasks():
    from scripts.run_analytical_pricing_stress_set import (
        ANALYTICAL_PRICING_STRESS_TASK_IDS,
    )
    from trellis.agent.task_manifests import load_task_manifest

    held_task_ids = {
        task["id"]
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task.get("task_disposition")
        in {"research_hold", "proof_hold", "rewrite_candidate"}
    }

    assert held_task_ids.isdisjoint(ANALYTICAL_PRICING_STRESS_TASK_IDS)


def test_analytical_stress_explicit_holds_stop_before_market_or_build(
    monkeypatch,
    tmp_path,
):
    from scripts import run_analytical_pricing_stress_set as stress_runner
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    calls: list[str] = []
    monkeypatch.setattr(
        stress_runner,
        "build_market_state",
        lambda: calls.append("market"),
    )
    monkeypatch.setattr(
        stress_runner,
        "run_task",
        lambda *args, **kwargs: calls.append("build"),
    )

    with pytest.raises(TaskManifestValidationError) as exc_info:
        stress_runner.run_analytical_pricing_stress_set(
            model="test-model",
            validation="standard",
            force_rebuild=False,
            task_ids=["T03", "T83", "T85"],
            output_file=tmp_path / "results.json",
            report_json_file=tmp_path / "report.json",
            report_md_file=tmp_path / "report.md",
        )

    issues = exc_info.value.report.blocking_issues
    assert {issue.code for issue in issues} == {"legacy.non_executable_disposition"}
    assert {issue.task_id for issue in issues} == {"T03", "T83", "T85"}
    assert calls == []


def test_task_loader_overwrites_manifest_provenance_claims(tmp_path):
    from trellis.agent.task_manifests import load_task_manifest

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY.yaml",
        {
            "version": 7,
            "tasks": [
                {
                    "id": "T900",
                    "title": "Spoofed provenance",
                    "status": "pending",
                    "task_corpus": "extension",
                    "task_definition_version": 999,
                    "task_definition_manifest": "TASKS_EXTENSION.yaml",
                }
            ],
        },
    )

    task = load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=tmp_path)[0]

    assert task["task_corpus"] == "proof_legacy"
    assert task["task_definition_version"] == 7
    assert task["task_definition_manifest"] == "TASKS_PROOF_LEGACY.yaml"


def test_assert_valid_task_manifests_raises_with_actionable_issue_keys(tmp_path):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_valid_task_manifests,
    )

    with pytest.raises(TaskManifestValidationError, match="manifest.missing") as exc:
        assert_valid_task_manifests(
            root=tmp_path,
            manifest_names=("TASKS_NEGATIVE.yaml",),
        )

    assert exc.value.report.blocking_issues


def test_pricing_loader_runs_manifest_preflight_before_loading(monkeypatch, tmp_path):
    from trellis.agent import task_manifests

    called = []

    def fail_preflight(**kwargs):
        called.append(kwargs)
        raise RuntimeError("preflight stopped loading")

    monkeypatch.setattr(task_manifests, "assert_valid_task_manifests", fail_preflight)

    with pytest.raises(RuntimeError, match="preflight stopped loading"):
        task_manifests.load_pricing_tasks(root=tmp_path)

    assert called == [{"root": tmp_path, "manifest_names": task_manifests.PRICING_TASK_CORPORA}]


def test_active_lookup_preflights_pricing_and_negative_corpora_together(monkeypatch, tmp_path):
    from trellis.agent import task_manifests

    called = []

    def record_preflight(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(task_manifests, "assert_valid_task_manifests", record_preflight)
    monkeypatch.setattr(task_manifests, "load_task_manifest", lambda *args, **kwargs: [])

    assert task_manifests.load_active_task_lookup(root=tmp_path) == {}
    assert called == [
        {
            "root": tmp_path,
            "manifest_names": (
                *task_manifests.PRICING_TASK_CORPORA,
                task_manifests.NEGATIVE_TASKS_MANIFEST,
            ),
        }
    ]


def test_financepy_benchmark_loader_uses_fail_closed_preflight(tmp_path):
    from trellis.agent.financepy_benchmark import load_financepy_benchmark_tasks
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    with pytest.raises(TaskManifestValidationError, match="manifest.missing"):
        load_financepy_benchmark_tasks(root=tmp_path)


def test_checked_in_task_manifests_pass_fail_closed_gate():
    from trellis.agent.task_manifest_validation import assert_valid_task_manifests
    from trellis.agent.task_manifests import ALL_TASK_CORPORA, ROOT

    report = assert_valid_task_manifests(root=ROOT, manifest_names=ALL_TASK_CORPORA)

    assert not report.blocking_issues
    assert report.legacy_issues
