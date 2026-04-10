"""Stale-test hygiene helpers shared by CLI tooling and pytest enforcement."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Callable


_TICKET_RE = re.compile(r"\b(?:QUA|CR)-\d+\b")
_DEFAULT_QUARANTINE_DAYS = 30
_DEFAULT_ANCIENT_DAYS = 90


@dataclass(frozen=True)
class HygieneFinding:
    """One skip/xfail/quarantine occurrence discovered in a test file."""

    path: str
    line: int
    kind: str
    source: str
    reason: str
    ticket: str
    age_days: int
    bucket: str
    requires_ticket: bool


def git_last_touch_age_days(path: Path, *, repo_root: Path, now: datetime | None = None) -> int:
    """Approximate finding age from git last-touch time for the owning file."""
    now = now or datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-1",
                "--format=%ct",
                "--",
                str(path.relative_to(repo_root)),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout.strip()
        if raw:
            touched_at = datetime.fromtimestamp(int(raw), tz=timezone.utc)
            return max((now - touched_at).days, 0)
    except Exception:
        pass

    touched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max((now - touched_at).days, 0)


def bucket_age(age_days: int, *, quarantine_days: int, ancient_days: int) -> str:
    """Classify a marker by approximate age bucket."""
    if age_days >= ancient_days:
        return "ancient"
    if age_days >= quarantine_days:
        return "stale"
    return "quarantine"


def scan_repo(
    root: Path,
    *,
    age_days_fn: Callable[[Path], int] | None = None,
    quarantine_days: int = _DEFAULT_QUARANTINE_DAYS,
    ancient_days: int = _DEFAULT_ANCIENT_DAYS,
) -> list[HygieneFinding]:
    """Scan ``tests/`` under one repo root for skip/xfail/quarantine markers."""
    test_root = root / "tests"
    if not test_root.exists():
        return []

    age_days_fn = age_days_fn or (lambda path: git_last_touch_age_days(path, repo_root=root))
    findings: list[HygieneFinding] = []
    for path in sorted(test_root.rglob("*.py")):
        findings.extend(
            scan_test_file(
                path,
                age_days=age_days_fn(path),
                quarantine_days=quarantine_days,
                ancient_days=ancient_days,
            )
        )
    return findings


def scan_test_file(
    path: Path,
    *,
    age_days: int,
    quarantine_days: int = _DEFAULT_QUARANTINE_DAYS,
    ancient_days: int = _DEFAULT_ANCIENT_DAYS,
) -> list[HygieneFinding]:
    """Scan one test file for skip/xfail/quarantine markers."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    bucket = bucket_age(
        age_days,
        quarantine_days=quarantine_days,
        ancient_days=ancient_days,
    )

    findings: list[HygieneFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                finding = _finding_from_marker_expr(
                    decorator,
                    path=path,
                    age_days=age_days,
                    bucket=bucket,
                    source_kind="decorator",
                )
                if finding is not None:
                    findings.append(finding)
        if isinstance(node, ast.Assign):
            for finding in _findings_from_pytestmark_assign(
                node,
                path=path,
                age_days=age_days,
                bucket=bucket,
            ):
                findings.append(finding)
        if isinstance(node, ast.Call):
            finding = _finding_from_runtime_call(
                node,
                path=path,
                age_days=age_days,
                bucket=bucket,
            )
            if finding is not None:
                findings.append(finding)
    return sorted(findings, key=lambda item: (item.path, item.line, item.kind))


def stale_unticketed_xfails(
    findings: list[HygieneFinding],
    *,
    ancient_only: bool = True,
) -> list[HygieneFinding]:
    """Return xfail findings that are old enough to trip local enforcement."""
    out: list[HygieneFinding] = []
    for finding in findings:
        if finding.kind != "xfail" or finding.ticket:
            continue
        if ancient_only and finding.bucket != "ancient":
            continue
        out.append(finding)
    return out


def format_report(
    findings: list[HygieneFinding],
    *,
    root: Path,
) -> str:
    """Render a compact human-readable stale-test hygiene report."""
    if not findings:
        return "No skip/xfail/quarantine findings."

    bucket_counts = {"quarantine": 0, "stale": 0, "ancient": 0}
    kind_counts: dict[str, int] = {}
    for finding in findings:
        bucket_counts[finding.bucket] = bucket_counts.get(finding.bucket, 0) + 1
        kind_counts[finding.kind] = kind_counts.get(finding.kind, 0) + 1

    lines = [
        "Stale test hygiene report",
        f"Findings: {len(findings)}",
        "Buckets: "
        f"quarantine={bucket_counts.get('quarantine', 0)}, "
        f"stale={bucket_counts.get('stale', 0)}, "
        f"ancient={bucket_counts.get('ancient', 0)}",
        "Kinds: "
        + ", ".join(f"{kind}={count}" for kind, count in sorted(kind_counts.items())),
        "",
    ]
    for finding in findings:
        rel_path = Path(finding.path).relative_to(root).as_posix()
        suffix = f" ticket={finding.ticket}" if finding.ticket else ""
        reason = f" reason={finding.reason}" if finding.reason else ""
        lines.append(
            f"[{finding.bucket}] {finding.kind:13s} {rel_path}:{finding.line} "
            f"age={finding.age_days}d{suffix}{reason}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the stale-test hygiene tool."""
    parser = argparse.ArgumentParser(description="Report stale skip/xfail/quarantine test markers.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--quarantine-days", type=int, default=_DEFAULT_QUARANTINE_DAYS)
    parser.add_argument("--ancient-days", type=int, default=_DEFAULT_ANCIENT_DAYS)
    parser.add_argument(
        "--fail-on-ancient-unticketed-xfail",
        action="store_true",
        help="Exit non-zero when ancient xfails do not carry a linked ticket id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``scripts/test_hygiene.py``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    findings = scan_repo(
        root,
        quarantine_days=args.quarantine_days,
        ancient_days=args.ancient_days,
    )
    print(format_report(findings, root=root))

    violations = stale_unticketed_xfails(findings)
    if args.fail_on_ancient_unticketed_xfail and violations:
        print("")
        print("Ancient xfails without linked ticket ids:")
        for finding in violations:
            rel_path = Path(finding.path).relative_to(root).as_posix()
            print(f"- {rel_path}:{finding.line} ({finding.age_days}d) {finding.reason or 'no reason'}")
        return 1
    return 0


def _finding_from_marker_expr(
    expr: ast.expr,
    *,
    path: Path,
    age_days: int,
    bucket: str,
    source_kind: str,
) -> HygieneFinding | None:
    kind = _marker_kind(expr)
    if kind is None:
        return None
    reason = _marker_reason(expr)
    ticket = _extract_ticket(reason)
    return HygieneFinding(
        path=str(path),
        line=int(getattr(expr, "lineno", 1) or 1),
        kind=kind,
        source=source_kind,
        reason=reason,
        ticket=ticket,
        age_days=age_days,
        bucket=bucket,
        requires_ticket=(kind == "xfail"),
    )


def _findings_from_pytestmark_assign(
    node: ast.Assign,
    *,
    path: Path,
    age_days: int,
    bucket: str,
) -> list[HygieneFinding]:
    targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
    if "pytestmark" not in targets:
        return []
    value = node.value
    expressions: list[ast.expr] = []
    if isinstance(value, (ast.Tuple, ast.List)):
        expressions.extend(item for item in value.elts if isinstance(item, ast.expr))
    elif isinstance(value, ast.expr):
        expressions.append(value)
    return [
        finding
        for expr in expressions
        if (finding := _finding_from_marker_expr(
            expr,
            path=path,
            age_days=age_days,
            bucket=bucket,
            source_kind="pytestmark",
        )) is not None
    ]


def _finding_from_runtime_call(
    node: ast.Call,
    *,
    path: Path,
    age_days: int,
    bucket: str,
) -> HygieneFinding | None:
    kind = _runtime_kind(node)
    if kind is None:
        return None
    reason = _runtime_reason(node, kind=kind)
    ticket = _extract_ticket(reason)
    return HygieneFinding(
        path=str(path),
        line=int(getattr(node, "lineno", 1) or 1),
        kind=kind,
        source="call",
        reason=reason,
        ticket=ticket,
        age_days=age_days,
        bucket=bucket,
        requires_ticket=(kind == "xfail"),
    )


def _marker_kind(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Call):
        name = _attr_path(expr.func)
    else:
        name = _attr_path(expr)
    mapping = {
        "pytest.mark.skip": "skip",
        "pytest.mark.skipif": "skipif",
        "pytest.mark.xfail": "xfail",
        "pytest.mark.legacy_compat": "legacy_compat",
    }
    return mapping.get(name)


def _runtime_kind(node: ast.Call) -> str | None:
    mapping = {
        "pytest.skip": "skip",
        "pytest.xfail": "xfail",
        "pytest.importorskip": "importorskip",
    }
    return mapping.get(_attr_path(node.func))


def _marker_reason(expr: ast.expr) -> str:
    if not isinstance(expr, ast.Call):
        return ""
    return _call_reason(expr, default_positional_index=0)


def _runtime_reason(node: ast.Call, *, kind: str) -> str:
    if kind == "importorskip":
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            return str(node.args[0].value)
        return ""
    return _call_reason(node, default_positional_index=0)


def _call_reason(node: ast.Call, *, default_positional_index: int) -> str:
    for keyword in node.keywords:
        if keyword.arg == "reason":
            return _constant_str(keyword.value)
    if len(node.args) > default_positional_index:
        return _constant_str(node.args[default_positional_index])
    return ""


def _constant_str(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value)
    return ""


def _attr_path(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _extract_ticket(reason: str) -> str:
    match = _TICKET_RE.search(reason or "")
    return match.group(0) if match else ""
