"""Review fresh-build promotion candidates against deterministic gate criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review one or more promotion candidate snapshots.",
    )
    parser.add_argument(
        "candidate_paths",
        nargs="*",
        help="Explicit candidate YAML paths to review.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="Review the latest N promotion candidates when explicit paths are not provided.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write promotion review artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the reviews as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from trellis.agent.knowledge.promotion import (
        list_promotion_candidate_paths,
        review_promotion_candidate,
    )
    from trellis.cli_paths import resolve_repo_path

    parser = _build_parser()
    args = parser.parse_args(argv)

    candidate_paths = [str(resolve_repo_path(path)) for path in args.candidate_paths]
    if not candidate_paths and args.latest > 0:
        candidate_paths = list_promotion_candidate_paths(limit=args.latest)
    if not candidate_paths:
        parser.error("Provide candidate_paths or use --latest N.")

    reviews = [
        review_promotion_candidate(path, persist=not args.no_persist)
        for path in candidate_paths
    ]
    if args.json:
        print(json.dumps(reviews, indent=2, default=str))
    else:
        for review in reviews:
            print(f"[{review['status'].upper()}] {review['comparison_target']} -> {review['recommended_module_path']}")
            print(f"  candidate: {review['candidate_path']}")
            if review.get("review_path"):
                print(f"  review: {review['review_path']}")
            for check in review["checks"]:
                status = "OK" if check["passed"] else "FAIL"
                print(f"  - {status} {check['name']}: {check['detail']}")
    return 0 if all(review["approved"] for review in reviews) else 1


if __name__ == "__main__":
    sys.exit(main())
