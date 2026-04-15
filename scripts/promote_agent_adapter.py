"""Explicit post-benchmark admission CLI for _agent adapter promotion.

The FinancePy pilot (QUA-864) separates benchmark execution from adapter
admission.  The benchmark runner (QUA-865) records a fresh-build candidate; the
review gate (QUA-866) blocks the benchmark path from silently using admitted
``_agent`` code.  This CLI (QUA-867) is the only supported way to move a
validated fresh-build artifact into ``trellis/instruments/_agent/`` and it
fails closed on any provenance mismatch.

Usage:

    python scripts/promote_agent_adapter.py \
        --candidate trellis/agent/knowledge/traces/promotion_candidates/<file>.yaml \
        [--dry-run] [--promoted-by <name>]

Exit code is zero only when the admission (or dry-run) succeeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        help="Path to a single promotion-candidate YAML emitted by the benchmark runner.",
    )
    parser.add_argument(
        "--candidate-glob",
        help=(
            "Glob pattern (relative to --candidate-root, default "
            "trellis/agent/knowledge/traces/promotion_candidates/) for batch "
            "admission across multiple candidates.  Mutually exclusive with --candidate."
        ),
    )
    parser.add_argument(
        "--candidate-root",
        default=str(
            ROOT / "trellis" / "agent" / "knowledge" / "traces" / "promotion_candidates"
        ),
        help="Directory used to resolve --candidate-glob.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(ROOT),
        help="Repository root used to resolve admission target paths (defaults to the repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every provenance check without touching the _agent tree.",
    )
    parser.add_argument(
        "--promoted-by",
        default="",
        help="Optional operator identifier recorded in the admission log.",
    )
    return parser.parse_args(argv)


def _promote_one(
    candidate_path: Path,
    *,
    repo_root: Path,
    dry_run: bool,
    promoted_by: str,
) -> tuple[int, dict]:
    """Promote one candidate and return (exit_code, payload)."""
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    try:
        result = promote_benchmark_candidate(
            candidate_path,
            repo_root=repo_root,
            dry_run=dry_run,
            promoted_by=promoted_by,
        )
    except PromotionAdmissionError as exc:
        return 2, {
            "status": "rejected",
            "candidate_path": str(candidate_path),
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return 1, {
            "status": "error",
            "candidate_path": str(candidate_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return 0, dict(result)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if bool(args.candidate) == bool(args.candidate_glob):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "exactly one of --candidate or --candidate-glob is required",
                },
                indent=2,
            )
        )
        return 1

    repo_root = Path(args.repo_root).expanduser()

    if args.candidate:
        exit_code, payload = _promote_one(
            Path(args.candidate).expanduser(),
            repo_root=repo_root,
            dry_run=bool(args.dry_run),
            promoted_by=args.promoted_by,
        )
        print(json.dumps(payload, indent=2, default=str))
        return exit_code

    candidate_root = Path(args.candidate_root).expanduser()
    # Path.glob accepts `..` segments in the pattern and would happily match
    # paths outside `candidate_root`.  Reject parent-traversal up front and
    # filter any matches that resolve outside the resolved candidate root, so
    # batch admission cannot accidentally operate on YAMLs anywhere else on
    # disk.  (PR #590 round-4 Copilot review.)
    if ".." in Path(args.candidate_glob).parts:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": (
                        "--candidate-glob must be relative to --candidate-root "
                        "and must not contain '..' segments"
                    ),
                    "candidate_root": str(candidate_root),
                    "candidate_glob": args.candidate_glob,
                },
                indent=2,
            )
        )
        return 1
    candidate_root_resolved = candidate_root.resolve()
    candidate_paths: list[Path] = []
    for match in sorted(candidate_root.glob(args.candidate_glob)):
        try:
            match.resolve().relative_to(candidate_root_resolved)
        except ValueError:
            continue
        candidate_paths.append(match)
    if not candidate_paths:
        print(
            json.dumps(
                {
                    "status": "no_match",
                    "candidate_root": str(candidate_root),
                    "candidate_glob": args.candidate_glob,
                    "results": [],
                },
                indent=2,
            )
        )
        return 1

    results: list[dict] = []
    rejection_seen = False
    error_seen = False
    for path in candidate_paths:
        per_exit, per_payload = _promote_one(
            path,
            repo_root=repo_root,
            dry_run=bool(args.dry_run),
            promoted_by=args.promoted_by,
        )
        results.append(per_payload)
        if per_exit == 2:
            rejection_seen = True
            # On apply (not dry-run) a single rejection halts the batch so
            # downstream candidates do not get partially admitted under the
            # assumption the earlier ones succeeded.
            if not args.dry_run:
                break
        elif per_exit == 1:
            error_seen = True
            if not args.dry_run:
                break

    summary = {
        "status": (
            "errored" if error_seen
            else "rejected" if rejection_seen
            else ("would_promote_all" if args.dry_run else "promoted_all")
        ),
        "dry_run": bool(args.dry_run),
        "candidate_root": str(candidate_root),
        "candidate_glob": args.candidate_glob,
        "candidate_count": len(candidate_paths),
        "processed_count": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2, default=str))
    if error_seen:
        return 1
    if rejection_seen:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
