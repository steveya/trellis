"""Contract tests for the fresh-generated FinancePy pilot boundary enforcer."""

from __future__ import annotations

import pytest

from trellis.agent.financepy_benchmark import (
    FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS,
    financepy_benchmark_execution_policy,
)
from trellis.agent.fresh_generated_boundary import (
    FreshGeneratedBoundaryCheck,
    FreshGeneratedBoundaryError,
    enforce_fresh_generated_boundary,
)


PILOT_TASK_IDS = sorted(FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS)


def _fresh_artifact(**overrides):
    artifact = {
        "module_name": "trellis_benchmarks._fresh.f001.analytical.europeanoptionanalytical",
        "module_path": (
            "task_runs/financepy_benchmarks/generated/f001/analytical/"
            "europeanoptionanalytical.py"
        ),
        "file_path": (
            "/tmp/trellis/task_runs/financepy_benchmarks/generated/f001/analytical/"
            "europeanoptionanalytical.py"
        ),
        "is_fresh_build": True,
        "trellis_imports": (
            "trellis.core.market_state",
            "trellis.models.black",
        ),
    }
    artifact.update(overrides)
    return artifact


CLEAN_GENERATED_SOURCE = """\
from trellis.core.market_state import MarketState
from trellis.models.black import black76_call


def price(market_state: MarketState) -> float:
    return black76_call(0.0, 0.0, 0.0, 0.0)
"""

AGENT_IMPORT_SOURCE = """\
from trellis.instruments._agent.europeanoptionanalytical import price as legacy_price


def price(*args, **kwargs):
    return legacy_price(*args, **kwargs)
"""


def test_enforce_fresh_generated_boundary_accepts_clean_fresh_artifact():
    check = enforce_fresh_generated_boundary(
        {"id": "F001"},
        _fresh_artifact(),
        execution_policy="fresh_generated",
        generated_source=CLEAN_GENERATED_SOURCE,
    )
    assert isinstance(check, FreshGeneratedBoundaryCheck)
    assert check.status == "enforced"
    assert check.task_id == "F001"
    assert check.policy == "fresh_generated"
    assert check.violations == ()


def test_enforce_fresh_generated_boundary_skips_non_fresh_policy():
    check = enforce_fresh_generated_boundary(
        {"id": "F010"},
        None,
        execution_policy="cached_existing",
        generated_source=None,
    )
    assert check.status == "not_applicable"
    assert check.violations == ()


def test_enforce_fresh_generated_boundary_rejects_missing_artifact():
    with pytest.raises(FreshGeneratedBoundaryError) as exc_info:
        enforce_fresh_generated_boundary(
            {"id": "F001"},
            None,
            execution_policy="fresh_generated",
        )
    message = str(exc_info.value)
    assert "F001" in message
    assert "fresh" in message.lower()


def test_enforce_fresh_generated_boundary_rejects_non_fresh_build_flag():
    with pytest.raises(FreshGeneratedBoundaryError):
        enforce_fresh_generated_boundary(
            {"id": "F002"},
            _fresh_artifact(is_fresh_build=False),
            execution_policy="fresh_generated",
            generated_source=CLEAN_GENERATED_SOURCE,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("module_path", "trellis/instruments/_agent/europeanoptionanalytical.py"),
        ("module_path", "instruments/_agent/europeanoptionanalytical.py"),
        ("module_name", "trellis.instruments._agent.europeanoptionanalytical"),
    ],
)
def test_enforce_fresh_generated_boundary_rejects_agent_module_target(field, value):
    with pytest.raises(FreshGeneratedBoundaryError) as exc_info:
        enforce_fresh_generated_boundary(
            {"id": "F003"},
            _fresh_artifact(**{field: value}),
            execution_policy="fresh_generated",
            generated_source=CLEAN_GENERATED_SOURCE,
        )
    assert "_agent" in str(exc_info.value)


def test_enforce_fresh_generated_boundary_rejects_agent_import_in_generated_source():
    with pytest.raises(FreshGeneratedBoundaryError) as exc_info:
        enforce_fresh_generated_boundary(
            {"id": "F007"},
            _fresh_artifact(),
            execution_policy="fresh_generated",
            generated_source=AGENT_IMPORT_SOURCE,
        )
    message = str(exc_info.value)
    assert "trellis.instruments._agent" in message


def test_enforce_fresh_generated_boundary_rejects_agent_import_in_artifact_metadata():
    with pytest.raises(FreshGeneratedBoundaryError):
        enforce_fresh_generated_boundary(
            {"id": "F009"},
            _fresh_artifact(
                trellis_imports=(
                    "trellis.core.market_state",
                    "trellis.instruments._agent.barrieroption",
                )
            ),
            execution_policy="fresh_generated",
        )


def test_enforce_fresh_generated_boundary_returns_violation_report_without_raising():
    check = enforce_fresh_generated_boundary(
        {"id": "F012"},
        _fresh_artifact(is_fresh_build=False),
        execution_policy="fresh_generated",
        generated_source=CLEAN_GENERATED_SOURCE,
        raise_on_violation=False,
    )
    assert check.status == "violated"
    assert check.violations
    assert check.task_id == "F012"


@pytest.mark.parametrize("task_id", PILOT_TASK_IDS)
def test_pilot_subset_defaults_to_fresh_generated_execution_policy(task_id):
    assert financepy_benchmark_execution_policy({"id": task_id}) == "fresh_generated"


@pytest.mark.parametrize("task_id", PILOT_TASK_IDS)
def test_pilot_task_enforcement_accepts_clean_fresh_artifact(task_id):
    artifact = _fresh_artifact(
        module_name=f"trellis_benchmarks._fresh.{task_id.lower()}.analytical.clean",
        module_path=(
            f"task_runs/financepy_benchmarks/generated/{task_id.lower()}/analytical/clean.py"
        ),
    )
    check = enforce_fresh_generated_boundary(
        {"id": task_id},
        artifact,
        execution_policy="fresh_generated",
        generated_source=CLEAN_GENERATED_SOURCE,
    )
    assert check.status == "enforced"
    assert check.task_id == task_id


@pytest.mark.parametrize("task_id", PILOT_TASK_IDS)
def test_pilot_task_enforcement_rejects_agent_fallback(task_id):
    artifact = _fresh_artifact(
        module_name=f"trellis.instruments._agent.{task_id.lower()}_adapter",
        module_path=f"trellis/instruments/_agent/{task_id.lower()}_adapter.py",
        is_fresh_build=False,
    )
    with pytest.raises(FreshGeneratedBoundaryError):
        enforce_fresh_generated_boundary(
            {"id": task_id},
            artifact,
            execution_policy="fresh_generated",
            generated_source=AGENT_IMPORT_SOURCE,
        )
