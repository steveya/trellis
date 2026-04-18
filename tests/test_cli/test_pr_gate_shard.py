from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "pr_gate_shard.py"
RELEASE_ONLY_DIRS = (
    "tests/test_contracts",
    "tests/test_crossval",
    "tests/test_tasks",
    "tests/test_verification",
)


def _run_shard(index: int, count: int) -> set[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--count", str(count), "--index", str(index)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def _eligible_pr_gate_files() -> set[str]:
    files: set[str] = set()
    for path in (REPO_ROOT / "tests").rglob("test_*.py"):
        relpath = path.relative_to(REPO_ROOT).as_posix()
        if any(
            relpath == excluded or relpath.startswith(f"{excluded}/")
            for excluded in RELEASE_ONLY_DIRS
        ):
            continue
        files.add(relpath)
    return files


def test_pr_gate_shards_cover_each_core_test_exactly_once() -> None:
    shard_count = 4
    seen: set[str] = set()

    for index in range(1, shard_count + 1):
        shard_paths = _run_shard(index=index, count=shard_count)
        assert shard_paths
        assert seen.isdisjoint(shard_paths)
        seen.update(shard_paths)

    assert seen == _eligible_pr_gate_files()


def test_pr_gate_shards_skip_release_only_suites() -> None:
    shard_count = 4

    for index in range(1, shard_count + 1):
        for relpath in _run_shard(index=index, count=shard_count):
            assert not any(
                relpath == excluded or relpath.startswith(f"{excluded}/")
                for excluded in RELEASE_ONLY_DIRS
            )
