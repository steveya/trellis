"""Validate task manifests before any pricing task can be selected."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.task_manifest_validation import (  # noqa: E402
    audit_task_manifests,
    legacy_issue_digest,
)
from trellis.agent.task_manifests import ALL_TASK_CORPORA  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        default=[],
        choices=ALL_TASK_CORPORA,
        help="Validate one manifest name. May be repeated; defaults to all corpora.",
    )
    parser.add_argument(
        "--show-legacy",
        action="store_true",
        help="Print every checked legacy-debt finding instead of only its summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable report.",
    )
    return parser.parse_args(argv)


def _issue_payload(issue) -> dict[str, str]:
    return {
        "key": issue.key,
        "manifest": issue.manifest,
        "task_id": issue.task_id,
        "code": issue.code,
        "path": issue.path,
        "message": issue.message,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    manifests = tuple(args.manifests or ALL_TASK_CORPORA)
    report = audit_task_manifests(root=ROOT, manifest_names=manifests)
    payload = {
        "manifests": list(manifests),
        "blocking_issue_count": len(report.blocking_issues),
        "blocking_issues": [_issue_payload(issue) for issue in report.blocking_issues],
        "legacy_issue_count": len(report.legacy_issues),
        "legacy_issue_digest": legacy_issue_digest(report.legacy_issues),
        "legacy_task_fingerprint": report.legacy_task_fingerprint,
        "legacy_issues": (
            [_issue_payload(issue) for issue in report.legacy_issues]
            if args.show_legacy
            else []
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Task manifest validation: "
            f"{len(manifests)} manifest(s), "
            f"{len(report.blocking_issues)} blocking issue(s), "
            f"{len(report.legacy_issues)} checked legacy debt issue(s)"
        )
        print(f"Legacy debt digest: {payload['legacy_issue_digest']}")
        for issue in report.blocking_issues:
            print(f"ERROR {issue.key}: {issue.message}")
        if args.show_legacy:
            for issue in report.legacy_issues:
                print(f"KNOWN {issue.key}: {issue.message}")
    return 1 if report.blocking_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
