"""Adopt approved promotion candidates into checked-in route modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adopt one or more approved promotion reviews.",
    )
    parser.add_argument(
        "review_paths",
        nargs="*",
        help="Explicit promotion review YAML paths to adopt.",
    )
    parser.add_argument(
        "--latest-approved",
        type=int,
        default=0,
        help="Adopt the latest N approved promotion reviews when explicit paths are not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the adoption plan without writing route files.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not persist promotion-adoption artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit adoption decisions as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from trellis.agent.knowledge.promotion import (
        adopt_promotion_candidate,
        list_promotion_review_paths,
    )
    from trellis.cli_paths import resolve_repo_path

    parser = _build_parser()
    args = parser.parse_args(argv)

    review_paths = [str(resolve_repo_path(path)) for path in args.review_paths]
    if not review_paths and args.latest_approved > 0:
        review_paths = list_promotion_review_paths(status="approved", limit=args.latest_approved)
    if not review_paths:
        parser.error("Provide review_paths or use --latest-approved N.")

    adoptions = [
        adopt_promotion_candidate(
            path,
            dry_run=args.dry_run,
            persist=not args.no_persist,
        )
        for path in review_paths
    ]
    if args.json:
        print(json.dumps(adoptions, indent=2, default=str))
    else:
        for adoption in adoptions:
            print(
                f"[{str(adoption['status']).upper()}] "
                f"{adoption['comparison_target']} -> {adoption['target_module_path']}"
            )
            print(f"  review: {adoption['review_path']}")
            print(f"  target: {adoption['target_file_path']}")
            if adoption.get("adoption_path"):
                print(f"  artifact: {adoption['adoption_path']}")
            for check in adoption["checks"]:
                status = "OK" if check["passed"] else "FAIL"
                print(f"  - {status} {check['name']}: {check['detail']}")
    return 0 if all(adoption["status"] in {"ready", "adopted"} for adoption in adoptions) else 1


if __name__ == "__main__":
    sys.exit(main())
