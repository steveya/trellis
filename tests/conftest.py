"""Root-level test configuration and shared fixtures.

Provides the ``llm_cassette`` fixture for deterministic LLM replay testing.
See QUA-423 for design context.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Cassette directories
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CASSETTES_DIR = REPO_ROOT / "cassettes"
_GLOBAL_WORKFLOW_TEST_PATHS = {
    "tests/test_session.py",
    "tests/test_pipeline.py",
    "tests/test_contracts/test_pipeline_contracts.py",
}


# ---------------------------------------------------------------------------
# Custom pytest marks
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tier2: Tier 2 contract tests (cassette replay, no tokens)")
    config.addinivalue_line("markers", "tier3: Tier 3 canary tests (live LLM, expensive)")
    config.addinivalue_line("markers", "integration: Full integration tests requiring live LLM")
    config.addinivalue_line("markers", "crossval: Cross-validation tests against independent libraries or engines")
    config.addinivalue_line("markers", "verification: Numerical or analytical verification tests against trusted references")
    config.addinivalue_line("markers", "global_workflow: End-to-end or user-facing workflow tests spanning multiple modules")
    config.addinivalue_line("markers", "legacy_compat: Compatibility tests that defend deprecated or legacy-only behavior")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-tag the major test strata so they stay filterable as the suite grows."""
    for item in items:
        try:
            rel_path = Path(str(item.fspath)).resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue

        if rel_path.startswith("tests/test_crossval/"):
            item.add_marker(pytest.mark.crossval)
        if rel_path.startswith("tests/test_verification/"):
            item.add_marker(pytest.mark.verification)
        if rel_path in _GLOBAL_WORKFLOW_TEST_PATHS:
            item.add_marker(pytest.mark.global_workflow)


# ---------------------------------------------------------------------------
# llm_cassette fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_cassette(request, monkeypatch):
    """Record or replay LLM interactions for deterministic agent testing.

    **Replay mode** (default): replays stored responses from a YAML cassette.
    Zero LLM calls, zero tokens.

    **Record mode** (``TRELLIS_CASSETTE_RECORD=1``): makes live LLM calls
    and writes the interactions to a cassette file for future replay.

    The cassette name is derived from the test name by default, or can be
    specified via ``@pytest.mark.parametrize("llm_cassette", ["name"], indirect=True)``.
    """
    from trellis.agent.cassette import llm_cassette_session

    # Determine cassette name
    if hasattr(request, "param") and request.param:
        cassette_name = request.param
    else:
        cassette_name = request.node.name

    cassette_dir = Path(os.environ.get("TRELLIS_CASSETTE_DIR", str(CASSETTES_DIR)))
    cassette_path = cassette_dir / f"{cassette_name}.yaml"

    if os.environ.get("TRELLIS_CASSETTE_RECORD", "0") == "1":
        with llm_cassette_session(
            cassette_path,
            mode="record",
            name=cassette_name,
        ) as recorder:
            yield recorder
    else:
        stale_policy = os.environ.get("TRELLIS_CASSETTE_STALE_POLICY", "warn")
        with llm_cassette_session(
            cassette_path,
            mode="replay",
            stale_policy=stale_policy,
            name=cassette_name,
        ) as replayer:
            yield replayer
