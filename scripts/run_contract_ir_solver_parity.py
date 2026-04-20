"""Emit the checked Phase 3 ContractIR structural-compiler parity ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from trellis.agent.contract_ir_solver_parity import (
        build_contract_ir_solver_parity_report,
        default_parity_artifact_paths,
        save_contract_ir_solver_parity_report,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    default_json, default_md = default_parity_artifact_paths()
    json_path = Path(args.json_out).expanduser() if args.json_out else default_json
    md_path = Path(args.md_out).expanduser() if args.md_out else default_md

    report = build_contract_ir_solver_parity_report()
    save_contract_ir_solver_parity_report(
        report,
        json_path=json_path,
        markdown_path=md_path,
    )
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "totals": report["totals"],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
