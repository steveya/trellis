"""Gate helpers for local PR/canary/release workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess


_CANARY_PREFIX_REASONS: dict[str, str] = {
    "trellis/agent/": "agent runtime or compiler path changed",
    "trellis/core/": "core pricing interfaces changed",
    "trellis/curves/": "curve construction changed",
    "trellis/models/": "pricing model code changed",
    "trellis/instruments/": "instrument adapter changed",
    "tests/test_agent/": "agent regression coverage changed",
    "tests/test_contracts/": "contract coverage changed",
    "tests/test_tasks/": "task regression coverage changed",
}
_CANARY_FILE_REASONS: dict[str, str] = {
    "TASKS.yaml": "task manifest changed",
    "CANARY_TASKS.yaml": "canary manifest changed",
    "scripts/run_canary.py": "canary runner changed",
    "scripts/canary_common.py": "canary payload merge changed",
    "scripts/record_cassettes.py": "cassette recording path changed",
}
_DEFAULT_BASE = "HEAD"
_IGNORED_PREFIXES = (
    "task_runs/",
    "trellis/agent/knowledge/traces/",
)
_IGNORED_EXACT = {
    "task_results_latest.json",
}


@dataclass(frozen=True)
class CanaryTriggerDecision:
    """Decision about whether the focused canary gate should run."""

    run_canary: bool
    subset: str
    reasons: tuple[str, ...]
    matched_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]


def load_changed_paths_from_git_status(repo_root: Path) -> list[str]:
    """Return changed local paths from ``git status --porcelain``."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    paths: list[str] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        line = raw_line[3:]
        if " -> " in line:
            line = line.split(" -> ", 1)[1]
        path = line.strip()
        if path:
            paths.append(path)
    return paths


def should_run_canary(paths: list[str]) -> CanaryTriggerDecision:
    """Classify whether the focused canary gate should run for changed paths."""
    normalized = tuple(_normalize_path(path) for path in paths if _normalize_path(path))
    matched_paths: list[str] = []
    reasons: list[str] = []

    for path in normalized:
        exact_reason = _CANARY_FILE_REASONS.get(path)
        if exact_reason is not None:
            matched_paths.append(path)
            reasons.append(path)
            continue
        for prefix, _ in _CANARY_PREFIX_REASONS.items():
            if path.startswith(prefix):
                matched_paths.append(path)
                reasons.append(prefix)
                break

    deduped_reasons = tuple(dict.fromkeys(reasons))
    return CanaryTriggerDecision(
        run_canary=bool(matched_paths),
        subset="core" if matched_paths else "",
        reasons=deduped_reasons,
        matched_paths=tuple(matched_paths),
        changed_paths=normalized,
    )


def format_decision(decision: CanaryTriggerDecision) -> str:
    """Render a compact human-readable trigger summary."""
    lines = [
        f"run_canary={'yes' if decision.run_canary else 'no'}",
        f"subset={decision.subset or 'none'}",
        f"changed_paths={len(decision.changed_paths)}",
    ]
    if decision.reasons:
        lines.append("reasons=" + ", ".join(decision.reasons))
    if decision.matched_paths:
        lines.append("matched_paths=" + ", ".join(decision.matched_paths))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``scripts/should_run_canary.py``."""
    parser = argparse.ArgumentParser(description="Decide whether the focused canary gate should run.")
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        default=[],
        help="Explicit changed path to classify. Repeatable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the decision as JSON.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repo root used when loading changed paths from git status.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for canary-trigger classification."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    changed_paths = args.paths or load_changed_paths_from_git_status(root)
    decision = should_run_canary(changed_paths)

    if args.json:
        print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    else:
        print(format_decision(decision))
    return 0


def _normalize_path(path: str) -> str:
    text = str(path).strip()
    if not text:
        return ""
    if text.startswith("./"):
        text = text[2:]
    normalized = text.replace("\\", "/")
    if normalized in _IGNORED_EXACT:
        return ""
    if normalized.startswith("task_results_") and normalized.endswith(".json"):
        return ""
    if any(normalized.startswith(prefix) for prefix in _IGNORED_PREFIXES):
        return ""
    return normalized
