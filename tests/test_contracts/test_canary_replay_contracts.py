"""Tier 2 contract tests for full-task canary replay (QUA-458).

These tests exercise the real canary runner in replay mode against committed
full-task cassettes. They are intentionally narrower than the older
build-pipeline cassette contracts: the acceptance target here is ``run_task``
plus diagnosis-packet parity with zero live token spend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_contracts.conftest import (
    CANARY_META,
    FULL_TASK_CASSETTES_DIR,
    full_task_cassette_available,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _full_task_replay_test(task_id: str):
    @pytest.mark.tier2
    @pytest.mark.skipif(
        not full_task_cassette_available(task_id),
        reason=(
            f"Full-task cassette not recorded for {task_id}. "
            f"Run: /Users/steveyang/miniforge3/bin/python3 scripts/record_cassettes.py --task {task_id}"
        ),
    )
    class _ReplayTest:
        def test_canary_replay_runs_zero_token_with_diagnosis_artifacts(self, tmp_path):
            canary = CANARY_META.get(task_id)
            assert canary is not None, f"{task_id} missing from CANARY_TASKS.yaml"
            output_path = tmp_path / f"{task_id}.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_canary.py"),
                "--task",
                task_id,
                "--replay",
                "--cassette-dir",
                str(FULL_TASK_CASSETTES_DIR),
                "--output",
                str(output_path),
        ]
            env = {
                "HOME": os.environ.get("HOME", ""),
                "PATH": os.environ.get("PATH", ""),
                "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                "TERM": os.environ.get("TERM", "xterm-256color"),
                "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
                "PYTHONHASHSEED": "0",
            }
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr

            results = json.loads(output_path.read_text())
            assert len(results) == 1
            result = results[0]
            assert result["success"] is True
            assert result["execution_mode"] == "cassette_replay"
            assert result["llm_cassette"]["path"] == str(
                FULL_TASK_CASSETTES_DIR / f"{task_id}.yaml"
            )
            assert (result.get("token_usage_summary") or {}).get("total_tokens") == 0
            assert Path(result["task_diagnosis_packet_path"]).exists()
            assert Path(result["task_diagnosis_dossier_path"]).exists()
            assert Path(result["task_diagnosis_latest_packet_path"]).exists()
            assert Path(result["task_diagnosis_latest_dossier_path"]).exists()

    _ReplayTest.__name__ = f"TestCanaryReplay_{task_id}"
    _ReplayTest.__qualname__ = f"TestCanaryReplay_{task_id}"
    return _ReplayTest


TestCanaryReplay_T13 = _full_task_replay_test("T13")
TestCanaryReplay_T38 = _full_task_replay_test("T38")
