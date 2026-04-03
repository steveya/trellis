"""Deterministic preflight coverage for the market-connector stress-task tranche."""

from __future__ import annotations


def _stress_tasks() -> dict[str, dict]:
    from trellis.agent.task_runtime import load_tasks

    return {
        task["id"]: task
        for task in load_tasks("E21", "E28", status=None)
        if task["id"].startswith("E")
    }


def _expected_comparison_targets(task: dict) -> tuple[str, ...]:
    cross_validate = dict(task.get("cross_validate") or {})
    internal = tuple(cross_validate.get("internal") or ())
    analytical = cross_validate.get("analytical")
    if analytical:
        return internal + (str(analytical),)
    return internal


def test_stress_task_tranche_is_present_in_task_inventory():
    tasks = _stress_tasks()

    assert set(tasks) == {"E21", "E22", "E23", "E24", "E25", "E26", "E27", "E28"}


def test_stress_task_manifest_matches_task_inventory():
    from trellis.agent.evals import load_stress_task_manifest

    tasks = _stress_tasks()
    manifest = load_stress_task_manifest()

    assert set(manifest) == set(tasks)


def test_stress_task_manifest_matches_task_contracts():
    from trellis.agent.evals import load_stress_task_manifest

    tasks = _stress_tasks()
    manifest = load_stress_task_manifest()

    for task_id, task in tasks.items():
        expectation = manifest[task_id]
        assert tuple(expectation.get("required_mock_capabilities") or ()) == tuple(
            (task.get("market_assertions") or {}).get("requires") or ()
        ), task_id
        assert tuple(expectation.get("comparison_targets") or ()) == _expected_comparison_targets(
            task
        ), task_id


def test_compare_ready_stress_tasks_have_mock_connector_coverage():
    from trellis.agent.evals import grade_stress_task_preflight, load_stress_task_manifest

    tasks = _stress_tasks()
    manifest = load_stress_task_manifest()

    for task_id, expectation in manifest.items():
        if expectation["outcome_class"] != "compare_ready":
            continue
        report = grade_stress_task_preflight(
            tasks[task_id],
            expectation,
        )
        assert report["task_contract_alignment"].passed, task_id
        assert report["market_capability_alignment"].passed, task_id
        assert report["comparison_target_inventory"].passed, task_id
        assert report["comparison_target_separation"].passed, task_id


def test_honest_block_stress_tasks_are_not_missing_mock_capabilities():
    from trellis.agent.evals import grade_stress_task_preflight, load_stress_task_manifest

    tasks = _stress_tasks()
    manifest = load_stress_task_manifest()

    for task_id, expectation in manifest.items():
        if expectation["outcome_class"] != "honest_block":
            continue
        report = grade_stress_task_preflight(
            tasks[task_id],
            expectation,
        )
        assert report["task_contract_alignment"].passed, task_id
        assert report["market_capability_alignment"].passed, task_id
        assert report["comparison_target_inventory"].passed, task_id
