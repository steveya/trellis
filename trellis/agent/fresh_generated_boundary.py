"""Fresh-generated FinancePy pilot boundary enforcer.

The FinancePy parity pilot runs a bounded subset of benchmark tasks (F001, F002,
F003, F007, F009, F012) with ``execution_policy='fresh_generated'`` so each
run proves that the current agent can assemble a pricing engine from Trellis
primitives rather than relying on checked-in ``trellis/instruments/_agent``
adapters.  The runner already persists provenance metadata that marks whether
an artifact is a fresh ephemeral build.  This module adds the explicit
fail-closed check: if the pilot critical path ever resolves to the admitted
``_agent`` tree, the benchmark must stop loudly instead of silently pricing
against admitted code.

See ``doc/plan/active__fresh-generated-financepy-pilot.md`` (QUA-864, QUA-866)
for the methodology.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


ADMITTED_AGENT_IMPORT_PREFIX = "trellis.instruments._agent"
_ADMITTED_AGENT_PATH_PREFIXES = (
    "trellis/instruments/_agent/",
    "instruments/_agent/",
)


class FreshGeneratedBoundaryError(RuntimeError):
    """Raised when the fresh-generated pilot benchmark path crosses into ``_agent``."""


@dataclass(frozen=True)
class FreshGeneratedBoundaryCheck:
    """Structured boundary-check result attached to each pilot benchmark record.

    ``status`` is one of:

    * ``not_applicable`` -- execution policy is not ``fresh_generated``
    * ``enforced`` -- pilot critical path verified against the admitted surface
    * ``violated`` -- pilot critical path leaked into the admitted surface
    """

    status: str
    policy: str
    task_id: str
    reason: str = ""
    violations: tuple[str, ...] = ()
    generated_module: str = ""
    generated_module_path: str = ""
    inspected_imports: tuple[str, ...] = field(default_factory=tuple)

    def as_record(self) -> dict[str, Any]:
        """Serialize the check into a JSON-ready payload for benchmark records."""
        return {
            "status": self.status,
            "policy": self.policy,
            "task_id": self.task_id,
            "reason": self.reason,
            "violations": list(self.violations),
            "generated_module": self.generated_module,
            "generated_module_path": self.generated_module_path,
            "inspected_imports": list(self.inspected_imports),
        }


def _is_admitted_agent_module_name(module_name: str) -> bool:
    text = str(module_name or "").strip()
    if not text:
        return False
    return text == ADMITTED_AGENT_IMPORT_PREFIX or text.startswith(
        ADMITTED_AGENT_IMPORT_PREFIX + "."
    )


def _is_admitted_agent_module_path(module_path: str) -> bool:
    normalized = str(module_path or "").replace("\\", "/").strip()
    if not normalized:
        return False
    return any(normalized.startswith(prefix) for prefix in _ADMITTED_AGENT_PATH_PREFIXES)


def _extract_trellis_imports_from_source(source: str) -> tuple[str, ...]:
    """Extract absolute ``trellis.*`` imports from generated source text.

    The parser is intentionally permissive: unparseable source returns an empty
    tuple so the caller can degrade to artifact-metadata-only inspection
    without masking other downstream syntax errors.
    """
    text = str(source or "")
    if not text.strip():
        return ()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name or "").strip()
                if name.startswith("trellis."):
                    imports.append(name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                continue
            module = str(node.module or "").strip()
            if not module.startswith("trellis."):
                continue
            imports.append(module)
    return tuple(imports)


def _collect_imports(
    generated_artifact: Mapping[str, Any] | None,
    generated_source: str | None,
) -> tuple[str, ...]:
    """Merge declared artifact imports with imports parsed from source."""
    collected: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        token = str(name or "").strip()
        if not token or token in seen:
            return
        seen.add(token)
        collected.append(token)

    declared: Iterable[Any]
    if isinstance(generated_artifact, Mapping):
        declared_value = generated_artifact.get("trellis_imports") or ()
        if isinstance(declared_value, (list, tuple)):
            declared = declared_value
        else:
            declared = ()
    else:
        declared = ()
    for name in declared:
        _add(str(name))

    if generated_source:
        for name in _extract_trellis_imports_from_source(generated_source):
            _add(name)
    return tuple(collected)


def enforce_fresh_generated_boundary(
    task: Mapping[str, Any],
    generated_artifact: Mapping[str, Any] | None,
    *,
    execution_policy: str,
    generated_source: str | None = None,
    raise_on_violation: bool = True,
) -> FreshGeneratedBoundaryCheck:
    """Fail closed if the fresh-generated pilot path leaks into admitted ``_agent`` code.

    ``raise_on_violation=False`` returns the violation report instead of raising
    so the benchmark runner can embed the reason in the comparison summary and
    still mark the run failed.
    """
    task_id = str((task or {}).get("id") or "").strip()
    policy = str(execution_policy or "").strip().lower()

    if policy != "fresh_generated":
        return FreshGeneratedBoundaryCheck(
            status="not_applicable",
            policy=policy or "unspecified",
            task_id=task_id,
            reason="execution policy is not fresh_generated",
        )

    violations: list[str] = []
    module_name = ""
    module_path = ""
    inspected_imports: tuple[str, ...] = ()

    if not isinstance(generated_artifact, Mapping) or not generated_artifact:
        violations.append("missing generated artifact provenance for fresh-generated benchmark")
    else:
        module_name = str(generated_artifact.get("module_name") or "").strip()
        module_path = str(generated_artifact.get("module_path") or "").strip()
        if not bool(generated_artifact.get("is_fresh_build")):
            violations.append(
                "fresh-generated benchmark resolved to a non-fresh adapter "
                f"(module_name={module_name or 'unknown'}, module_path={module_path or 'unknown'})"
            )
        if _is_admitted_agent_module_name(module_name):
            violations.append(
                f"fresh-generated artifact module name is under admitted _agent tree: {module_name}"
            )
        if _is_admitted_agent_module_path(module_path):
            violations.append(
                f"fresh-generated artifact module path is under admitted _agent tree: {module_path}"
            )

        inspected_imports = _collect_imports(generated_artifact, generated_source)
        for imported_module in inspected_imports:
            if _is_admitted_agent_module_name(imported_module):
                violations.append(
                    "generated critical path imports from admitted _agent tree: "
                    f"{imported_module}"
                )
    if not inspected_imports:
        inspected_imports = _collect_imports(generated_artifact, generated_source)

    if violations:
        reason = "; ".join(violations)
        check = FreshGeneratedBoundaryCheck(
            status="violated",
            policy=policy,
            task_id=task_id,
            reason=reason,
            violations=tuple(violations),
            generated_module=module_name,
            generated_module_path=module_path,
            inspected_imports=inspected_imports,
        )
        if raise_on_violation:
            raise FreshGeneratedBoundaryError(
                f"QUA-866: fresh-generated boundary violation for task {task_id or 'unknown'}: {reason}"
            )
        return check

    return FreshGeneratedBoundaryCheck(
        status="enforced",
        policy=policy,
        task_id=task_id,
        reason="fresh-generated artifact verified outside admitted _agent tree",
        generated_module=module_name,
        generated_module_path=module_path,
        inspected_imports=inspected_imports,
    )


__all__ = (
    "ADMITTED_AGENT_IMPORT_PREFIX",
    "FreshGeneratedBoundaryCheck",
    "FreshGeneratedBoundaryError",
    "enforce_fresh_generated_boundary",
)
