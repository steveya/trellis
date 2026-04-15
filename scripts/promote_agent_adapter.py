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
        required=True,
        help="Path to a promotion-candidate YAML emitted by the benchmark runner.",
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


def main(argv: list[str] | None = None) -> int:
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    candidate_path = Path(args.candidate).expanduser()
    repo_root = Path(args.repo_root).expanduser()

    try:
        result = promote_benchmark_candidate(
            candidate_path,
            repo_root=repo_root,
            dry_run=bool(args.dry_run),
            promoted_by=args.promoted_by,
        )
    except PromotionAdmissionError as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "candidate_path": str(candidate_path),
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(
            json.dumps(
                {
                    "status": "error",
                    "candidate_path": str(candidate_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
