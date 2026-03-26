"""Structured blocker taxonomy and missing-primitive planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PrimitiveBlocker:
    """Single structured blocker derived from raw primitive/blocker signals."""

    id: str
    category: str
    primitive_kind: str
    severity: str
    summary: str
    target_package: str | None = None
    suggested_modules: tuple[str, ...] = ()
    required_tests: tuple[str, ...] = ()
    docs_to_update: tuple[str, ...] = ()
    knowledge_files_to_update: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockerReport:
    """Structured set of blockers for a route or product."""

    blockers: tuple[PrimitiveBlocker, ...]
    should_block: bool
    summary: str


_BLOCKERS_PATH = Path(__file__).resolve().parent / "knowledge" / "canonical" / "blockers.yaml"
_BLOCKER_DEFS: dict[str, dict] | None = None


def plan_blockers(
    raw_blockers: tuple[str, ...] | list[str],
    *,
    product_ir=None,
) -> BlockerReport:
    """Turn raw blocker tokens into a structured blocker report."""
    blockers = tuple(
        _blocker_from_token(token, product_ir=product_ir)
        for token in dict.fromkeys(raw_blockers)
    )
    if not blockers:
        return BlockerReport(
            blockers=(),
            should_block=False,
            summary="No missing-primitive blockers.",
        )
    return BlockerReport(
        blockers=blockers,
        should_block=True,
        summary="; ".join(blocker.summary for blocker in blockers),
    )


def render_blocker_report(report: BlockerReport) -> str:
    """Render a structured blocker report for prompt or exception text."""
    if not report.blockers:
        return "No missing-primitive blockers."

    lines = ["## Structured blocker report"]
    for blocker in report.blockers:
        lines.append(f"- `{blocker.id}`")
        lines.append(f"  - Category: `{blocker.category}`")
        lines.append(f"  - Primitive kind: `{blocker.primitive_kind}`")
        lines.append(f"  - Severity: `{blocker.severity}`")
        lines.append(f"  - Summary: {blocker.summary}")
        if blocker.target_package:
            lines.append(f"  - Recommended package: `{blocker.target_package}`")
        if blocker.suggested_modules:
            lines.append(
                "  - Suggested modules: "
                + ", ".join(f"`{module}`" for module in blocker.suggested_modules)
            )
        if blocker.required_tests:
            lines.append(
                "  - Required tests: "
                + ", ".join(f"`{target}`" for target in blocker.required_tests)
            )
        if blocker.docs_to_update:
            lines.append(
                "  - Docs to update: "
                + ", ".join(f"`{target}`" for target in blocker.docs_to_update)
            )
        if blocker.knowledge_files_to_update:
            lines.append(
                "  - Knowledge files to update: "
                + ", ".join(f"`{target}`" for target in blocker.knowledge_files_to_update)
            )
    return "\n".join(lines)


def _blocker_from_token(token: str, *, product_ir=None) -> PrimitiveBlocker:
    """Convert one raw blocker token into a structured remediation work item."""
    defs = _load_blocker_defs()
    if token in defs:
        entry = defs[token]
        return PrimitiveBlocker(
            id=token,
            category=entry["category"],
            primitive_kind=entry["primitive_kind"],
            severity=entry["severity"],
            summary=entry["summary"],
            target_package=entry.get("target_package"),
            suggested_modules=tuple(entry.get("suggested_modules", [])),
            required_tests=tuple(entry.get("required_tests", [])),
            docs_to_update=tuple(entry.get("docs_to_update", [])),
            knowledge_files_to_update=tuple(entry.get("knowledge_files_to_update", [])),
        )

    if token.startswith("missing_module:"):
        module = token.split(":", 1)[1]
        return PrimitiveBlocker(
            id=token,
            category="implementation_gap",
            primitive_kind="module_availability",
            severity="high",
            summary=f"The planned route depends on module `{module}`, which does not exist yet.",
            target_package=_parent_package(module),
            suggested_modules=(module,),
            required_tests=("tests/test_agent/test_import_registry.py",),
            docs_to_update=("docs/api/models.rst",),
            knowledge_files_to_update=("trellis/agent/knowledge/import_registry.py",),
        )

    if token.startswith("missing_symbol:"):
        qualified = token.split(":", 1)[1]
        module, _, symbol = qualified.rpartition(".")
        return PrimitiveBlocker(
            id=token,
            category="export_or_registry_gap",
            primitive_kind="symbol_availability",
            severity="high",
            summary=(
                f"The planned route depends on `{qualified}`, but that symbol is not "
                "currently exported from the module."
            ),
            target_package=module or None,
            suggested_modules=(qualified,),
            required_tests=("tests/test_agent/test_import_registry.py",),
            docs_to_update=("docs/api/models.rst",),
            knowledge_files_to_update=("trellis/agent/knowledge/import_registry.py",),
        )

    model_family = getattr(product_ir, "model_family", None) if product_ir is not None else None
    return PrimitiveBlocker(
        id=token,
        category="unknown_gap",
        primitive_kind="unknown",
        severity="high",
        summary=(
            f"Unclassified blocker `{token}` was produced for model family "
            f"`{model_family or 'unknown'}`. Add a structured blocker definition "
            "before attempting generation."
        ),
    )


def _load_blocker_defs() -> dict[str, dict]:
    """Load and cache the canonical blocker taxonomy definitions from YAML."""
    global _BLOCKER_DEFS
    if _BLOCKER_DEFS is None:
        data = yaml.safe_load(_BLOCKERS_PATH.read_text()) or []
        _BLOCKER_DEFS = {entry["id"]: entry for entry in data}
    return _BLOCKER_DEFS


def _parent_package(module: str) -> str | None:
    """Return the parent package for a dotted module path, if any."""
    if "." not in module:
        return None
    return module.rsplit(".", 1)[0]
