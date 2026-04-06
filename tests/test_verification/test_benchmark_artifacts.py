"""Verification tests for checked benchmark artifact portability."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "docs" / "benchmarks"


def test_checked_benchmark_json_artifacts_do_not_embed_machine_local_paths():
    repo_root_text = str(REPO_ROOT)
    for path in sorted(BENCHMARK_ROOT.glob("*.json")):
        payload = json.loads(path.read_text())
        assert "json_path" not in payload, path.name
        assert "text_path" not in payload, path.name
        assert repo_root_text not in path.read_text(), path.name
