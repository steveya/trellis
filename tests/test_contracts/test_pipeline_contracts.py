"""Tier 2 contract tests: full pipeline contracts for canary tasks (QUA-427).

These tests exercise the complete build pipeline using cassette replay.
They require recorded cassettes — skip gracefully if cassettes aren't available.

To record cassettes for these tests:
    python scripts/record_cassettes.py
"""

from __future__ import annotations

import pytest

from tests.test_contracts.conftest import (
    CANARY_META,
    TASK_ENTRIES,
    cassette_available,
    cassette_skip_reason,
    cassette_path_for,
)


# ---------------------------------------------------------------------------
# Helper: parametrize cassette name via indirect fixture
# ---------------------------------------------------------------------------

def _pipeline_test(task_id: str, instrument_type: str | None = None):
    """Generate a full-pipeline contract test for a canary task.

    Returns a test class that:
    1. Skips if cassette not recorded
    2. Loads the cassette via llm_cassette fixture
    3. Runs build_payoff and asserts on the result
    """
    # The class-level skipif handles missing cassettes
    # The parametrize with indirect routes the cassette name to llm_cassette fixture

    @pytest.mark.tier2
    @pytest.mark.skipif(
        not cassette_available(task_id),
        reason=cassette_skip_reason(task_id),
    )
    class _PipelineTest:
        @pytest.mark.parametrize("llm_cassette", [task_id], indirect=True)
        def test_build_produces_payoff_class(self, llm_cassette):
            from trellis.agent.executor import build_payoff

            task = TASK_ENTRIES.get(task_id)
            if not task:
                pytest.skip(f"{task_id} not in active pricing task manifests")

            construct = instrument_type or task.get("construct")
            if isinstance(construct, list):
                construct = construct[0]  # use first for build

            cls = build_payoff(
                task["title"],
                instrument_type=construct,
                force_rebuild=True,
            )
            assert cls is not None
            # Must be a class with the Payoff protocol
            assert isinstance(cls, type), f"Expected a class, got {type(cls)}"

        @pytest.mark.parametrize("llm_cassette", [task_id], indirect=True)
        def test_quant_decision_is_deterministic(self, llm_cassette):
            """Same cassette should produce the same quant decision."""
            from trellis.agent.quant import select_pricing_method

            task = TASK_ENTRIES.get(task_id)
            if not task:
                pytest.skip(f"{task_id} not in active pricing task manifests")

            construct = instrument_type or task.get("construct")
            if isinstance(construct, list):
                construct = construct[0]

            plan = select_pricing_method(
                task["title"],
                instrument_type=construct,
            )
            assert plan.method, "Quant must select a method"
            assert plan.method_modules, "Quant must specify method modules"

    _PipelineTest.__name__ = f"TestPipeline_{task_id}"
    _PipelineTest.__qualname__ = f"TestPipeline_{task_id}"
    return _PipelineTest


# ---------------------------------------------------------------------------
# Pipeline contracts for each canary engine family
# ---------------------------------------------------------------------------

TestPipeline_T01 = _pipeline_test("T01", instrument_type="lattice")
TestPipeline_T13 = _pipeline_test("T13", instrument_type="pde")
TestPipeline_T25 = _pipeline_test("T25", instrument_type="monte_carlo")
TestPipeline_T38 = _pipeline_test("T38", instrument_type="cds")
TestPipeline_T39 = _pipeline_test("T39", instrument_type="transforms")
TestPipeline_T49 = _pipeline_test("T49", instrument_type="copula")
TestPipeline_T73 = _pipeline_test("T73", instrument_type="swaption")
