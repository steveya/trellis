from __future__ import annotations

import json


def _legacy_tasks() -> dict[str, dict]:
    from trellis.agent.task_manifests import load_task_manifest

    return {
        str(task["id"]): task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
    }


def _financepy_tasks() -> dict[str, dict]:
    from trellis.agent.task_manifests import load_task_manifest

    return {
        str(task["id"]): task
        for task in load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml")
    }


def _target_buckets(report) -> dict[str, str]:
    return {target.target_id: target.bucket for target in report.target_problems}


def _target_payload(report, target_id: str) -> dict:
    payload = report.to_payload()
    for target in payload["targets"]:
        if target["target_id"] == target_id:
            return target
    raise AssertionError(f"missing target {target_id}")


def test_classifies_recent_stochastic_vol_task_pack_into_stable_buckets():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    tasks = _legacy_tasks()
    expected = {
        "T20": {
            "heston_adi_pde": "stochastic_vol_pde",
            "heston_mc": "stochastic_vol_monte_carlo",
        },
        "T28": {
            "euler_heston": "stochastic_vol_monte_carlo",
            "qe_heston": "stochastic_vol_monte_carlo",
            "heston_fft": "stochastic_vol_transform",
        },
        "T40": {
            "heston_fft": "stochastic_vol_transform",
            "heston_cos": "stochastic_vol_transform",
            "heston_mc": "stochastic_vol_monte_carlo",
        },
        "T67": {
            "calibrated_heston_fft": "calibration_to_surface",
            "market_prices": "calibration_to_surface",
        },
        "T76": {
            "heston_analytical": "stochastic_vol_transform",
            "heston_mc": "stochastic_vol_monte_carlo",
            "heston_pde": "stochastic_vol_pde",
            "heston_fft": "stochastic_vol_transform",
            "heston_cos": "stochastic_vol_transform",
        },
        "T114": {
            "laguerre_heston": "stochastic_vol_transform",
            "fft_heston": "stochastic_vol_transform",
        },
        "T44": {
            "bates_fft": "affine_jump_stochastic_vol",
            "bates_mc": "affine_jump_stochastic_vol",
        },
        "T60": {
            "slv_mc": "slv_lsv",
            "heston_mc": "stochastic_vol_monte_carlo",
        },
        "T117": {
            "lsv_pde": "slv_lsv",
            "lsv_mc": "slv_lsv",
        },
        "E27": {
            "american_pathdep_pde": "unsupported_path_dependent_control",
            "american_pathdep_mc": "unsupported_path_dependent_control",
            "american_pathdep_fft": "unsupported_path_dependent_control",
        },
    }

    for task_id, target_expectations in expected.items():
        report = classify_stochastic_vol_task(tasks[task_id])
        assert report is not None, task_id
        assert _target_buckets(report) == target_expectations


def test_ambient_market_capabilities_do_not_declare_stochastic_vol_semantics():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    assert classify_stochastic_vol_task(_financepy_tasks()["F012"]) is None
    assert classify_stochastic_vol_task(_legacy_tasks()["T105"]) is None
    assert (
        classify_stochastic_vol_task(
            {
                "id": "TAMBIENT",
                "title": "Black-Scholes European equity option",
                "construct": "analytical",
                "market": {
                    "available_capabilities": [
                        "discount_curve",
                        "spx_heston_implied_vol",
                    ],
                    "model_parameters": "heston_equity",
                    "metadata": {"surface_builder": "heston_implied_vol"},
                },
                "cross_validate": {"internal": ["black_scholes"]},
            }
        )
        is None
    )


def test_explicit_semantic_model_contract_declares_stochastic_vol_semantics():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    for model_family, target_id, expected_bucket, expected_process in (
        ("heston", "fft", "stochastic_vol_transform", "heston"),
        ("rough_heston", "fft", "stochastic_vol_transform", "rough_heston"),
        ("bates", "fft", "stochastic_vol_transform", "bates"),
        ("slv_lsv", "pde", "slv_lsv", "slv_lsv"),
    ):
        report = classify_stochastic_vol_task(
            {
                "id": "TEXPLICIT",
                "title": "European equity option comparison",
                "semantic_contract": {
                    "product": {"model_family": model_family},
                },
                "cross_validate": {"internal": [target_id]},
            }
        )

        assert report is not None
        target = _target_payload(report, target_id)
        assert target["bucket"] == expected_bucket
        assert target["process_family"] == expected_process


def test_inherited_bates_model_preserves_route_only_target_shape():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    report = classify_stochastic_vol_task(
        {
            "id": "TBATESROUTES",
            "title": "European equity option comparison",
            "model_contract": {"model_family": "bates"},
            "cross_validate": {"internal": ["pde", "market_prices"]},
        }
    )

    assert report is not None
    pde = _target_payload(report, "pde")
    assert pde["bucket"] == "stochastic_vol_pde"
    assert pde["solver_target"] == "pde"
    assert pde["process_family"] == "bates"
    assert pde["validation_bundle"] == "bates:pde"

    calibration = _target_payload(report, "market_prices")
    assert calibration["bucket"] == "calibration_to_surface"
    assert calibration["solver_target"] == "surface_calibration"
    assert calibration["process_family"] == "bates"
    assert calibration["calibration_problem"]["status"] == "calibration_blocked"


def test_heston_parameter_semantics_distinguish_parameters_from_surface_bumps():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    tasks = _legacy_tasks()

    t20 = classify_stochastic_vol_task(tasks["T20"])
    heston_pde = _target_payload(t20, "heston_adi_pde")
    assert heston_pde["model_parameter_semantics"] == {
        "model_family": "heston",
        "model_parameter_source": "explicit_model_parameters",
        "black_vol_surface_role": "market_input_not_model_calibration",
        "requires_calibration_bridge": False,
    }
    assert heston_pde["market_bindings"]["requires_model_parameters"] is True
    assert heston_pde["market_bindings"]["requires_black_vol_surface"] is False

    t67 = classify_stochastic_vol_task(tasks["T67"])
    calibrated = _target_payload(t67, "calibrated_heston_fft")
    assert calibrated["model_parameter_semantics"]["model_parameter_source"] == (
        "calibration_to_market_surface"
    )
    assert calibrated["model_parameter_semantics"]["requires_calibration_bridge"] is True
    assert calibrated["market_bindings"]["requires_model_parameters"] is False
    assert calibrated["market_bindings"]["requires_black_vol_surface"] is True
    assert calibrated["calibration_problem"]["status"] == "calibration_supported"
    assert calibrated["calibration_problem"]["workflow_id"] == "heston_smile"
    assert calibrated["calibration_problem"]["output_parameter_source"] == (
        "calibrated_model_parameter_set"
    )

    market_prices = _target_payload(t67, "market_prices")
    assert market_prices["calibration_problem"]["status"] == "calibration_needed"
    assert market_prices["calibration_problem"]["input_quote_family"] == "option_price"


def test_repair_packets_name_missing_primitives_not_generated_text():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    tasks = _legacy_tasks()

    qe_target = _target_payload(classify_stochastic_vol_task(tasks["T28"]), "qe_heston")
    assert qe_target["repair_packet"] is None
    assert qe_target["solver_target"] == "monte_carlo_qe"
    assert qe_target["validation_bundle"] == "heston:monte_carlo"

    bates_target = _target_payload(classify_stochastic_vol_task(tasks["T44"]), "bates_fft")
    assert bates_target["repair_packet"] is None

    bates_barrier_packet = _target_payload(
        classify_stochastic_vol_task(
            {
                "id": "TBATESBARRIER",
                "title": "Bates barrier option: MC",
                "cross_validate": {"internal": ["bates_barrier_mc"]},
            }
        ),
        "bates_barrier_mc",
    )["repair_packet"]
    assert bates_barrier_packet["missing_primitive"] == (
        "bates_nonvanilla_route_contract"
    )

    slv_packet = _target_payload(classify_stochastic_vol_task(tasks["T117"]), "lsv_pde")[
        "repair_packet"
    ]
    assert slv_packet["missing_primitive"] == "leverage_function_contract"

    laguerre_packet = _target_payload(
        classify_stochastic_vol_task(tasks["T114"]),
        "laguerre_heston",
    )["repair_packet"]
    assert laguerre_packet["missing_primitive"] == (
        "heston_gauss_laguerre_transform_kernel"
    )

    e27_packet = _target_payload(
        classify_stochastic_vol_task(tasks["E27"]),
        "american_pathdep_mc",
    )["repair_packet"]
    assert e27_packet["unsupported_class"] == (
        "path_dependent_early_exercise_under_stochastic_vol"
    )


def test_heston_laguerre_target_exposes_quadrature_transform_contract():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    target = _target_payload(
        classify_stochastic_vol_task(_legacy_tasks()["T114"]),
        "laguerre_heston",
    )
    contract = target["quadrature_transform_contract"]

    assert target["bucket"] == "stochastic_vol_transform"
    assert target["process_family"] == "heston"
    assert target["solver_target"] == "gauss_laguerre_quadrature"
    assert target["validation_bundle"] == "heston:transform"
    assert target["market_bindings"]["requires_model_parameters"] is True
    assert target["market_bindings"]["requires_black_vol_surface"] is False
    assert target["repair_packet"]["missing_primitive"] == (
        "heston_gauss_laguerre_transform_kernel"
    )
    assert contract["quadrature_family"] == "gauss_laguerre"
    assert contract["characteristic_function"] == (
        "heston_log_spot_characteristic_function"
    )
    assert contract["required_model_parameters"] == [
        "kappa",
        "theta",
        "xi",
        "rho",
        "v0",
    ]
    assert contract["integration_requirements"] == [
        "gauss_laguerre_nodes_weights",
        "heston_characteristic_function_binding",
        "damping_or_contour_policy",
        "oscillatory_integrand_stabilization",
    ]
    assert "cross_validate_against_fft_cos_when_admitted" in contract[
        "validation_requirements"
    ]
    assert contract["missing_components"] == [
        "heston_gauss_laguerre_transform_kernel",
        "gauss_laguerre_heston_validation_bundle",
    ]
    assert contract["supported_now"] is False


def test_bates_targets_expose_affine_jump_process_contract():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    report = classify_stochastic_vol_task(_legacy_tasks()["T44"])

    for target_id, solver_target in {
        "bates_fft": "affine_jump_transform",
        "bates_mc": "affine_jump_monte_carlo",
    }.items():
        target = _target_payload(report, target_id)
        contract = target["affine_jump_process"]

        assert target["process_family"] == "bates"
        assert target["solver_target"] == solver_target
        assert target["market_bindings"]["requires_model_parameters"] is True
        assert target["market_bindings"]["requires_jump_parameters"] is True
        assert target["validation_bundle"] == "bates:affine_jump_stochastic_vol"
        assert contract["base_process_family"] == "heston"
        assert contract["jump_family"] == "compound_poisson_lognormal"
        assert contract["required_model_parameters"] == [
            "kappa",
            "theta",
            "xi",
            "rho",
            "v0",
        ]
        assert contract["required_jump_parameters"] == [
            "jump_intensity",
            "jump_mean",
            "jump_variance",
        ]
        assert contract["jump_parameter_aliases"]["jump_intensity"] == [
            "lam",
            "lambda",
        ]
        assert contract["jump_parameter_aliases"]["jump_variance"] == [
            "jump_var",
            "jump_vol^2",
            "jump_vol",
        ]
        assert contract["transform_capability"] == "bates_characteristic_function"
        assert contract["monte_carlo_capability"] == "bates_jump_stochastic_vol_process"
        assert target["repair_packet"] is None
        assert target["unsupported_features"] == []
        assert contract["supported_now"] is True
        assert contract["missing_primitives"] == []
        assert "consume_jump_parameters" in contract["validation_requirements"]


def test_slv_lsv_targets_expose_leverage_function_contracts():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    tasks = _legacy_tasks()
    expectations = {
        ("T60", "slv_mc"): (
            "leverage_function_monte_carlo",
            "slv_lsv_monte_carlo_solver",
        ),
        ("T117", "lsv_pde"): (
            "leverage_function_pde",
            "slv_lsv_pde_solver",
        ),
        ("T117", "lsv_mc"): (
            "leverage_function_monte_carlo",
            "slv_lsv_monte_carlo_solver",
        ),
    }

    for (task_id, target_id), (solver_target, missing_solver) in expectations.items():
        target = _target_payload(classify_stochastic_vol_task(tasks[task_id]), target_id)
        contract = target["leverage_function_contract"]

        assert target["process_family"] == "slv_lsv"
        assert target["solver_target"] == solver_target
        assert target["market_bindings"]["requires_model_parameters"] is True
        assert target["market_bindings"]["requires_black_vol_surface"] is True
        assert target["market_bindings"]["requires_local_vol_surface"] is True
        assert target["validation_bundle"] == "slv_lsv:leverage_function"
        assert contract["leverage_function_kind"] == "spot_time_surface"
        assert contract["required_market_inputs"] == [
            "local_vol_surface",
            "black_vol_surface",
            "underlier_spot",
            "discount_curve",
        ]
        assert contract["required_model_inputs"] == [
            "heston_model_parameters",
            "leverage_function_surface",
        ]
        assert contract["calibration_requirements"] == [
            "recorded_leverage_calibration_problem",
            "local_vol_surface_authority",
            "stochastic_vol_process_coupling",
        ]
        assert contract["interpolation_domain"] == ["time", "spot"]
        assert "leverage_bounds" in contract["diagnostics"]
        assert missing_solver in contract["missing_components"]
        assert contract["supported_now"] is False


def test_path_dependent_heston_targets_expose_composite_control_contract():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    report = classify_stochastic_vol_task(_legacy_tasks()["E27"])
    expectations = {
        "american_pathdep_pde": (
            "path_dependent_control_pde",
            "path_dependent_heston_pde_solver",
        ),
        "american_pathdep_mc": (
            "path_dependent_control_monte_carlo",
            "path_dependent_heston_monte_carlo_solver",
        ),
        "american_pathdep_fft": (
            "path_dependent_control_transform",
            "path_dependent_heston_transform_blocker",
        ),
    }

    for target_id, (solver_target, missing_solver) in expectations.items():
        target = _target_payload(report, target_id)
        contract = target["path_dependent_control_contract"]

        assert target["bucket"] == "unsupported_path_dependent_control"
        assert target["process_family"] == "heston"
        assert target["solver_target"] == solver_target
        assert target["payoff_class"] == "path_dependent_early_exercise"
        assert target["validation_bundle"] == "heston:path_dependent_control"
        assert target["market_bindings"]["requires_model_parameters"] is True
        assert target["market_bindings"]["requires_black_vol_surface"] is False
        assert target["unsupported_features"] == [
            "path_dependent_early_exercise_under_stochastic_vol",
        ]
        assert contract["composite_class"] == (
            "american_asian_barrier_under_stochastic_vol"
        )
        assert contract["state_requirements"] == [
            "spot_state",
            "variance_state",
            "path_summary_state",
            "exercise_state",
        ]
        assert contract["path_state_requirements"] == [
            "running_average_state",
            "barrier_status_state",
            "monitoring_grid_state",
        ]
        assert contract["event_monitor_requirements"] == [
            "barrier_monitor",
            "exercise_schedule",
            "monitoring_grid",
        ]
        assert contract["payoff_summary_requirements"] == [
            "asian_average_summary",
            "barrier_survival_indicator",
            "exercise_intrinsic_value",
        ]
        assert "early_exercise_policy" in contract["control_requirements"]
        assert "heston_path_state_coupling" in contract[
            "stochastic_vol_coupling_requirements"
        ]
        assert missing_solver in contract["missing_components"]
        assert contract["supported_now"] is False
        assert contract["expected_honest_block"] is True
        assert contract["model_validator_policy"] == "skip_expected_honest_block"


def test_stochastic_vol_problem_payload_is_json_stable():
    from trellis.agent.computational_problem_ir import stochastic_vol_problem_payload

    payload = stochastic_vol_problem_payload(_legacy_tasks()["T28"])

    assert payload is not None
    assert payload["task_id"] == "T28"
    assert payload["task_bucket"] == "stochastic_vol_mixed"
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_single_target_stochastic_vol_task_uses_construct_hint():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    report = classify_stochastic_vol_task(
        {
            "id": "TX",
            "title": "Heston equity option via 2D PDE",
            "construct": "pde",
            "new_component": "adi_2d_solver",
        }
    )

    assert report is not None
    assert report.task_bucket == "stochastic_vol_pde"
    assert report.target_problems[0].solver_target == "pde_adi"
