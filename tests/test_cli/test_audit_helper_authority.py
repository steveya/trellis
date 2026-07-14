from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_helper_authority_cli_emits_json_for_explicit_root(capsys, tmp_path):
    from tests.test_agent.test_helper_authority_audit import _fixture_root
    from trellis.agent import helper_authority_audit

    root = _fixture_root(tmp_path)
    exit_code = helper_authority_audit.main(["--root", str(root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["summary"]["route_only_reference_count"] == 1
    assert payload["summary"]["binding_only_reference_count"] == 1


def test_helper_authority_cli_can_fail_on_route_binding_drift(capsys, tmp_path):
    from tests.test_agent.test_helper_authority_audit import _fixture_root
    from trellis.agent import helper_authority_audit

    root = _fixture_root(tmp_path)
    exit_code = helper_authority_audit.main(
        ["--root", str(root), "--fail-on-drift"]
    )

    assert exit_code == 1
    assert "route_only_references=1" in capsys.readouterr().out


def test_helper_authority_script_runs_against_the_worktree(tmp_path):
    from tests.test_agent.test_helper_authority_audit import _fixture_root

    root = Path(__file__).resolve().parents[2]
    fixture = _fixture_root(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/audit_helper_authority.py"),
            "--root",
            str(fixture),
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"]["adapter_authority_call_count"] == 3
