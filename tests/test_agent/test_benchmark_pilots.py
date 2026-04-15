"""Tests for the cross-corpus benchmark pilot registry."""

from __future__ import annotations

import pytest

from trellis.agent.benchmark_pilots import (
    BenchmarkPilot,
    PILOT_REGISTRY,
    get_pilot,
    get_pilot_task_ids,
    is_pilot_task,
    pilot_corpora,
)


def test_pilot_registry_exposes_financepy_pilot():
    pilot = get_pilot("financepy")
    assert isinstance(pilot, BenchmarkPilot)
    assert pilot.corpus == "financepy"
    assert set(pilot.task_ids) == {"F001", "F002", "F003", "F007", "F009", "F012"}
    assert pilot.execution_policy == "fresh_generated"


def test_get_pilot_task_ids_returns_sorted_tuple():
    ids = get_pilot_task_ids("financepy")
    assert ids == ("F001", "F002", "F003", "F007", "F009", "F012")


def test_get_pilot_task_ids_is_case_insensitive():
    assert get_pilot_task_ids("FinancePy") == get_pilot_task_ids("financepy")


def test_get_pilot_raises_for_unknown_corpus():
    with pytest.raises(KeyError):
        get_pilot("totally_unregistered_corpus")


def test_is_pilot_task_matches_membership():
    assert is_pilot_task("financepy", "F001") is True
    assert is_pilot_task("financepy", "F015") is False
    assert is_pilot_task("financepy", "") is False


def test_pilot_corpora_lists_registry_keys():
    assert "financepy" in pilot_corpora()


def test_financepy_benchmark_execution_policy_delegates_to_registry():
    from trellis.agent.financepy_benchmark import (
        FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS,
    )

    assert set(FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS) == set(
        get_pilot_task_ids("financepy")
    )


def test_pilot_parity_scorecard_uses_registry():
    from trellis.agent.pilot_parity_scorecard import PILOT_SCORECARD_TASK_IDS

    assert tuple(PILOT_SCORECARD_TASK_IDS) == get_pilot_task_ids("financepy")


def test_registry_entries_are_frozen():
    pilot = PILOT_REGISTRY["financepy"]
    with pytest.raises(AttributeError):
        pilot.corpus = "other"  # type: ignore[misc]
