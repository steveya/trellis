"""Tier 2 contract test: cassette freshness check (QUA-427).

Warns when cassettes are stale (> 14 days old) so they get re-recorded
before prompt drift causes silent failures.
"""

from __future__ import annotations

import pytest

from tests.test_contracts.conftest import (
    CASSETTES_DIR,
    CASSETTE_MAX_AGE_DAYS,
    cassette_age_days,
)


@pytest.mark.tier2
def test_cassettes_are_fresh():
    """All committed cassettes should be less than CASSETTE_MAX_AGE_DAYS old."""
    if not CASSETTES_DIR.exists():
        pytest.skip("No cassettes directory found — record cassettes first")

    cassette_files = list(CASSETTES_DIR.glob("*.yaml"))
    if not cassette_files:
        pytest.skip("No cassette files found — record cassettes first")

    stale = []
    for path in sorted(cassette_files):
        age = cassette_age_days(path)
        if age is not None and age > CASSETTE_MAX_AGE_DAYS:
            stale.append(f"{path.name} ({age} days old)")

    if stale:
        msg = (
            f"{len(stale)} stale cassette(s) (> {CASSETTE_MAX_AGE_DAYS} days):\n"
            + "\n".join(f"  - {s}" for s in stale)
            + "\n\nRe-record with: TRELLIS_CASSETTE_RECORD=1 pytest tests/test_contracts/ -m tier2"
        )
        pytest.fail(msg)


@pytest.mark.tier2
def test_cassette_files_are_valid_yaml():
    """All cassette files must be parseable YAML with meta and calls sections."""
    if not CASSETTES_DIR.exists():
        pytest.skip("No cassettes directory")

    import yaml

    cassette_files = list(CASSETTES_DIR.glob("*.yaml"))
    if not cassette_files:
        pytest.skip("No cassette files")

    for path in sorted(cassette_files):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), f"{path.name}: not a dict"
        assert "meta" in raw, f"{path.name}: missing 'meta' section"
        assert "calls" in raw, f"{path.name}: missing 'calls' section"
        assert isinstance(raw["calls"], list), f"{path.name}: 'calls' must be a list"
        meta = raw["meta"]
        assert "recorded_at" in meta, f"{path.name}: meta missing 'recorded_at'"
        assert "total_calls" in meta, f"{path.name}: meta missing 'total_calls'"
        assert meta["total_calls"] == len(raw["calls"]), (
            f"{path.name}: total_calls ({meta['total_calls']}) != actual ({len(raw['calls'])})"
        )
