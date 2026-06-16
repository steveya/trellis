from __future__ import annotations

import json


def _legacy_tasks() -> dict[str, dict]:
    from trellis.agent.task_manifests import load_task_manifest

    return {
        str(task["id"]): task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
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


def test_repair_packets_name_missing_primitives_not_generated_text():
    from trellis.agent.computational_problem_ir import classify_stochastic_vol_task

    tasks = _legacy_tasks()

    qe_target = _target_payload(classify_stochastic_vol_task(tasks["T28"]), "qe_heston")
    assert qe_target["repair_packet"] is None
    assert qe_target["solver_target"] == "monte_carlo_qe"
    assert qe_target["validation_bundle"] == "heston:monte_carlo"

    bates_packet = _target_payload(classify_stochastic_vol_task(tasks["T44"]), "bates_fft")[
        "repair_packet"
    ]
    assert bates_packet["missing_primitive"] == "bates_affine_jump_stochastic_vol_kernel"

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
