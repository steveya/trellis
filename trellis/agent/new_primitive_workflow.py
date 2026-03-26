"""Concrete workflow planning for adding new pricing primitives."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.agent.blocker_planning import BlockerReport, PrimitiveBlocker


@dataclass(frozen=True)
class NewPrimitiveWorkItem:
    """Single actionable work item for closing a blocker."""

    blocker_id: str
    action_kind: str
    target_package: str | None
    suggested_modules: tuple[str, ...]
    summary: str
    mathematical_contract: str
    tests_to_add: tuple[str, ...]
    docs_to_update: tuple[str, ...]
    knowledge_files_to_update: tuple[str, ...]


@dataclass(frozen=True)
class NewPrimitiveWorkflow:
    """Structured workflow plan for closing one or more blockers."""

    summary: str
    items: tuple[NewPrimitiveWorkItem, ...]


def plan_new_primitive_workflow(
    blocker_report: BlockerReport,
    *,
    product_ir=None,
) -> NewPrimitiveWorkflow:
    """Plan the concrete engineering workflow needed to close blockers."""
    items = tuple(
        _work_item_from_blocker(blocker, product_ir=product_ir)
        for blocker in blocker_report.blockers
    )
    summary = (
        "No new primitive workflow required."
        if not items
        else " -> ".join(item.action_kind for item in items)
    )
    return NewPrimitiveWorkflow(summary=summary, items=items)


def render_new_primitive_workflow(workflow: NewPrimitiveWorkflow) -> str:
    """Render the workflow as markdown for docs or exception text."""
    if not workflow.items:
        return "No new primitive workflow required."

    lines = ["## New primitive workflow"]
    for item in workflow.items:
        lines.append(f"- Blocker: `{item.blocker_id}`")
        lines.append(f"  - Action kind: `{item.action_kind}`")
        if item.target_package:
            lines.append(f"  - Target package: `{item.target_package}`")
        if item.suggested_modules:
            lines.append(
                "  - Suggested modules: "
                + ", ".join(f"`{module}`" for module in item.suggested_modules)
            )
        lines.append(f"  - Summary: {item.summary}")
        lines.append(f"  - Mathematical contract: {item.mathematical_contract}")
        if item.tests_to_add:
            lines.append(
                "  - Tests to add: "
                + ", ".join(f"`{test}`" for test in item.tests_to_add)
            )
        if item.docs_to_update:
            lines.append(
                "  - Docs to update: "
                + ", ".join(f"`{target}`" for target in item.docs_to_update)
            )
        if item.knowledge_files_to_update:
            lines.append(
                "  - Knowledge files to update: "
                + ", ".join(f"`{target}`" for target in item.knowledge_files_to_update)
            )
    return "\n".join(lines)


def _work_item_from_blocker(blocker: PrimitiveBlocker, *, product_ir=None) -> NewPrimitiveWorkItem:
    """Translate one blocker into a concrete engineering work item."""
    if blocker.category == "numerical_substrate_gap":
        return NewPrimitiveWorkItem(
            blocker_id=blocker.id,
            action_kind="new_foundational_primitive",
            target_package=blocker.target_package,
            suggested_modules=blocker.suggested_modules,
            summary=blocker.summary,
            mathematical_contract=_mathematical_contract_for(blocker, product_ir=product_ir),
            tests_to_add=blocker.required_tests,
            docs_to_update=blocker.docs_to_update,
            knowledge_files_to_update=blocker.knowledge_files_to_update,
        )

    if blocker.category in {"implementation_gap", "export_or_registry_gap"}:
        return NewPrimitiveWorkItem(
            blocker_id=blocker.id,
            action_kind="library_repair",
            target_package=blocker.target_package,
            suggested_modules=blocker.suggested_modules,
            summary=blocker.summary,
            mathematical_contract=(
                "Preserve the existing numerical contract; repair the missing "
                "module/export so the planned route can reuse the intended primitive."
            ),
            tests_to_add=blocker.required_tests,
            docs_to_update=blocker.docs_to_update,
            knowledge_files_to_update=blocker.knowledge_files_to_update,
        )

    return NewPrimitiveWorkItem(
        blocker_id=blocker.id,
        action_kind="taxonomy_extension",
        target_package=blocker.target_package,
        suggested_modules=blocker.suggested_modules,
        summary=blocker.summary,
        mathematical_contract=(
            "Define the missing primitive contract before implementation. "
            "The blocker is currently unclassified."
        ),
        tests_to_add=blocker.required_tests,
        docs_to_update=blocker.docs_to_update,
        knowledge_files_to_update=blocker.knowledge_files_to_update,
    )


def _mathematical_contract_for(blocker: PrimitiveBlocker, *, product_ir=None) -> str:
    """Describe the numerical contract that a new primitive must satisfy."""
    if blocker.primitive_kind == "exercise_control":
        model_family = getattr(product_ir, "model_family", "unknown") if product_ir is not None else "unknown"
        state_dependence = getattr(product_ir, "state_dependence", "unknown") if product_ir is not None else "unknown"
        exercise_style = getattr(product_ir, "exercise_style", "unknown") if product_ir is not None else "unknown"
        return (
            "Define a continuation/exercise control primitive that accepts the "
            f"`{model_family}` state representation, handles `{state_dependence}` "
            f"payoff state, and enforces `{exercise_style}` exercise semantics "
            "without inventing a new engine API."
        )
    if blocker.primitive_kind == "module_availability":
        return "Create the missing module with the planned public contract so existing routes can import it."
    if blocker.primitive_kind == "symbol_availability":
        return "Expose the missing symbol without changing the intended numerical contract of the route."
    return "Specify and validate the missing primitive contract before code generation."
