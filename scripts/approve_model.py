"""CLI tool to approve or reject a model audit record.

Usage:
    python scripts/approve_model.py <audit_record_path> \\
        --reviewer "john.doe" \\
        --status approved \\
        --notes "Validated against QuantLib reference"

Status values:
    approved                  — model passes review; cleared for production use
    conditionally_approved    — approved with caveats (document in --notes)
    rejected                  — model fails review; must be rebuilt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Approve or reject a Trellis model audit record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audit_path",
        help="Path to the .json audit record file",
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="Reviewer identifier (name or user ID)",
    )
    parser.add_argument(
        "--status",
        choices=["approved", "conditionally_approved", "rejected"],
        default="approved",
        help="Approval decision (default: approved)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes to attach to the approval record",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the merged audit record (with sidecar data) after writing",
    )

    args = parser.parse_args(argv)

    audit_path = Path(args.audit_path)
    if not audit_path.exists():
        print(f"error: audit record not found: {audit_path}", file=sys.stderr)
        return 1

    try:
        from trellis.agent.model_audit import approve_model, load_model_audit_record
    except ImportError as exc:
        print(f"error: could not import trellis.agent.model_audit: {exc}", file=sys.stderr)
        return 1

    try:
        sidecar = approve_model(
            audit_path,
            reviewer=args.reviewer,
            status=args.status,
            notes=args.notes,
        )
    except Exception as exc:
        print(f"error: failed to write approval sidecar: {exc}", file=sys.stderr)
        return 1

    print(f"approval written → {sidecar}")

    if args.show:
        merged = load_model_audit_record(audit_path)
        print(json.dumps(merged, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
