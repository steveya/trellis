from __future__ import annotations

import json
from pathlib import Path


def test_should_run_canary_flags_runtime_and_model_changes():
    from trellis.testing.gates import should_run_canary

    decision = should_run_canary(
        [
            "docs/developer/test_gates.md",
            "trellis/agent/task_runtime.py",
            "trellis/models/zcb_option.py",
        ]
    )

    assert decision.run_canary is True
    assert decision.subset == "core"
    assert "trellis/agent/" in decision.reasons
    assert "trellis/models/" in decision.reasons


def test_should_run_canary_skips_docs_only_changes():
    from trellis.testing.gates import should_run_canary

    decision = should_run_canary(
        [
            "docs/developer/task_and_eval_loops.rst",
            "doc/plan/active__backlog-burn-down-execution.md",
        ]
    )

    assert decision.run_canary is False
    assert decision.subset == ""
    assert decision.reasons == ()


def test_should_run_canary_ignores_ephemeral_artifacts():
    from trellis.testing.gates import should_run_canary

    decision = should_run_canary(
        [
            "task_results_latest.json",
            "task_runs/history/run_123.json",
            "trellis/agent/knowledge/traces/platform/session_price.events.ndjson",
        ]
    )

    assert decision.run_canary is False
    assert decision.changed_paths == ()


def test_load_changed_paths_from_git_status_handles_rename_and_untracked(monkeypatch, tmp_path):
    from trellis.testing.gates import load_changed_paths_from_git_status

    class _Result:
        stdout = " M tests/test_cli/test_should_run_canary.py\nR  old/path.py -> new/path.py\n?? scripts/should_run_canary.py\n"

    monkeypatch.setattr(
        "trellis.testing.gates.subprocess.run",
        lambda *args, **kwargs: _Result(),
    )

    paths = load_changed_paths_from_git_status(tmp_path)

    assert paths == [
        "tests/test_cli/test_should_run_canary.py",
        "new/path.py",
        "scripts/should_run_canary.py",
    ]


def test_should_run_canary_cli_supports_explicit_paths(capsys):
    from trellis.testing import gates

    exit_code = gates.main(
        [
            "--path",
            "docs/developer/test_gates.md",
            "--path",
            "scripts/run_canary.py",
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["run_canary"] is True
    assert output["subset"] == "core"
    assert "scripts/run_canary.py" in output["matched_paths"]
