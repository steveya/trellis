#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

RELEASE_ONLY_DIRS: Final[tuple[str, ...]] = (
    "tests/test_contracts",
    "tests/test_crossval",
    "tests/test_tasks",
    "tests/test_verification",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_release_only(relpath: str) -> bool:
    return any(
        relpath == excluded or relpath.startswith(f"{excluded}/")
        for excluded in RELEASE_ONLY_DIRS
    )


def iter_pr_gate_files(root: Path) -> list[Path]:
    tests_root = root / "tests"
    return [
        path
        for path in sorted(tests_root.rglob("test_*.py"))
        if not _is_release_only(path.relative_to(root).as_posix())
    ]


def test_weight(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )
    return max(1, count)


@dataclass
class Shard:
    index: int
    total_weight: int = 0
    file_count: int = 0
    paths: list[Path] = field(default_factory=list)


def build_shards(paths: list[Path], count: int) -> list[Shard]:
    shards = [Shard(index=index) for index in range(count)]
    weighted_paths = sorted(
        ((test_weight(path), path) for path in paths),
        key=lambda item: (-item[0], item[1].as_posix()),
    )
    for weight, path in weighted_paths:
        shard = min(shards, key=lambda item: (item.total_weight, item.file_count, item.index))
        shard.paths.append(path)
        shard.total_weight += weight
        shard.file_count += 1
    for shard in shards:
        shard.paths.sort(key=lambda path: path.as_posix())
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the deterministic pytest file list for one PR-gate shard."
    )
    parser.add_argument("--count", type=int, required=True, help="Total shard count.")
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="1-based shard index to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.index <= 0 or args.index > args.count:
        raise SystemExit("--index must be between 1 and --count")

    root = repo_root()
    shards = build_shards(iter_pr_gate_files(root), args.count)
    for path in shards[args.index - 1].paths:
        print(path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
