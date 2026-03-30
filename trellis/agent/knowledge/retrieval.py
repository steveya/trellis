"""Format retrieved knowledge as markdown for prompt injection and traces."""

from __future__ import annotations

from typing import Any

from trellis.agent.knowledge.schema import (
    AdapterLifecycleStatus,
    CookbookEntry,
    DataContractEntry,
    FailureSignature,
    Lesson,
    MethodRequirements,
    Principle,
    ProductDecomposition,
    ProductIR,
)
from trellis.agent.knowledge.promotion import (
    detect_adapter_lifecycle_records,
    format_adapter_lifecycle_warnings,
    resolve_adapter_lifecycle_records,
    summarize_adapter_lifecycle_records,
)
from trellis.agent.knowledge.store import identify_superseded_basket_lesson_ids
from trellis.agent.knowledge.api_map import format_api_map_for_prompt

_COMPACT_LIMITS = {
    "builder": {
        "principles": 4,
        "contracts": 2,
        "requirements": 6,
        "lessons": 3,
        "unresolved_primitives": 5,
        "template_chars": 1800,
        "registry_chars": 1600,
    },
    "review": {
        "principles": 4,
        "requirements": 6,
        "lessons": 3,
    },
    "routing": {
        "principles": 4,
        "lessons": 4,
    },
}

_DISTILLED_LIMITS = {
    "builder": {
        "principles": 3,
        "requirements": 3,
        "lessons": 2,
    },
    "review": {
        "principles": 3,
        "requirements": 3,
        "lessons": 2,
    },
    "routing": {
        "principles": 3,
        "lessons": 2,
    },
}


def _adapter_lifecycle_warnings(*, compact: bool) -> str:
    """Load warning-only stale adapter findings for prompt rendering."""
    try:
        records = resolve_adapter_lifecycle_records(detect_adapter_lifecycle_records())
        active_records = [
            record
            for record in records
            if record.status != AdapterLifecycleStatus.ARCHIVED
        ]
        return format_adapter_lifecycle_warnings(active_records, compact=compact)
    except Exception:
        return ""


def _adapter_lifecycle_summary() -> dict[str, Any]:
    """Summarize stale adapter findings for trace payloads."""
    try:
        records = resolve_adapter_lifecycle_records(detect_adapter_lifecycle_records())
    except Exception:
        return {
            "stale_adapter_count": 0,
            "stale_adapter_ids": [],
            "deprecated_adapter_count": 0,
            "deprecated_adapter_ids": [],
            "archived_adapter_count": 0,
            "archived_adapter_ids": [],
            "fresh_replacements": [],
            "adapter_lifecycle": {
                "summary": {
                    "status_counts": {
                        "fresh": 0,
                        "stale": 0,
                        "deprecated": 0,
                        "archived": 0,
                    },
                    "stale_adapter_count": 0,
                    "stale_adapter_ids": [],
                    "deprecated_adapter_count": 0,
                    "deprecated_adapter_ids": [],
                    "archived_adapter_count": 0,
                    "archived_adapter_ids": [],
                    "fresh_adapter_count": 0,
                    "fresh_adapter_ids": [],
                    "fresh_replacements": [],
                    "records": [],
                },
                "records": [],
            },
        }
    summary = summarize_adapter_lifecycle_records(records)
    return {
        "stale_adapter_count": summary["stale_adapter_count"],
        "stale_adapter_ids": summary["stale_adapter_ids"],
        "deprecated_adapter_count": summary["deprecated_adapter_count"],
        "deprecated_adapter_ids": summary["deprecated_adapter_ids"],
        "archived_adapter_count": summary["archived_adapter_count"],
        "archived_adapter_ids": summary["archived_adapter_ids"],
        "fresh_replacements": summary["fresh_replacements"],
        "adapter_lifecycle": {
            "summary": summary,
            "records": summary["records"],
        },
    }


def format_knowledge_for_prompt(knowledge: dict[str, Any], *, compact: bool = False) -> str:
    """Format a ``retrieve_for_task()`` result dict as a single markdown string.

    The output is structured as numbered sections in this order:
    API map, import registry, adapter freshness warnings, principles,
    cookbook template, product semantics, data contracts, modeling
    requirements, lessons (ranked), product notes, unresolved primitives,
    and matched failure signatures.

    When ``compact=True``, each section is truncated to fit within the
    limits defined in ``_COMPACT_LIMITS`` (fewer items, shorter templates).
    """
    sections: list[str] = []

    # 0. API map (small family-level orientation before the full registry)
    try:
        api_map = format_api_map_for_prompt(compact=compact)
        if api_map:
            sections.append(api_map)
    except Exception:
        pass

    # 1. Import registry (always included — eliminates hallucination)
    try:
        from trellis.agent.knowledge.import_registry import get_import_registry
        registry = get_import_registry()
        if registry:
            if compact:
                registry = _truncate_text(
                    registry,
                    _COMPACT_LIMITS["builder"]["registry_chars"],
                    label="import registry",
                )
            sections.append(registry)
    except Exception:
        pass

    adapter_warnings = _adapter_lifecycle_warnings(compact=compact)
    if adapter_warnings:
        sections.append(adapter_warnings)

    # 2. Principles (always first — hot tier)
    principles: list[Principle] = knowledge.get("principles", [])
    if principles:
        lines = ["## Key Principles\n"]
        selected = _take_limited(principles, _COMPACT_LIMITS["builder"]["principles"] if compact else None)
        for p in selected:
            lines.append(f"- **{p.id}**: {p.rule}")
        omission = _omission_notice("principles", len(principles), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    # 3. Pricing method + cookbook
    cookbook: CookbookEntry | None = knowledge.get("cookbook")
    if cookbook:
        template = cookbook.template
        if compact:
            template = _truncate_text(
                template,
                _COMPACT_LIMITS["builder"]["template_chars"],
                label="cookbook template",
            )
        sections.append(
            f"## Pricing Method: {cookbook.method}\n\n"
            f"{cookbook.description}\n\n"
            f"{template}"
        )

    # 3b. Product semantics from ProductIR
    product_ir: ProductIR | None = knowledge.get("product_ir")
    if product_ir is not None:
        lines = [
            "## Product Semantics\n",
            f"- Instrument: `{product_ir.instrument}`",
            f"- Payoff family: `{product_ir.payoff_family}`",
            f"- Exercise style: `{product_ir.exercise_style}`",
            f"- State dependence: `{product_ir.state_dependence}`",
            f"- Schedule dependence: `{product_ir.schedule_dependence}`",
            f"- Model family: `{product_ir.model_family}`",
        ]
        if product_ir.payoff_traits:
            lines.append(
                "- Payoff traits: "
                + ", ".join(f"`{trait}`" for trait in product_ir.payoff_traits)
            )
        if product_ir.candidate_engine_families:
            lines.append(
                "- Candidate engine families: "
                + ", ".join(f"`{family}`" for family in product_ir.candidate_engine_families)
            )
        if getattr(product_ir, "route_families", ()):
            lines.append(
                "- Exact route families: "
                + ", ".join(f"`{family}`" for family in product_ir.route_families)
            )
        sections.append("\n".join(lines))

    # 4. Data contracts
    contracts: list[DataContractEntry] = knowledge.get("data_contracts", [])
    if contracts:
        lines = ["## DATA CONTRACTS (input conventions)\n"]
        selected = _take_limited(contracts, _COMPACT_LIMITS["builder"]["contracts"] if compact else None)
        for c in selected:
            lines.append(f"### {c.name}")
            lines.append(f"- Source: `{c.source}`")
            lines.append(f"- Convention: {c.convention}")
            lines.append(f"- Typical range: {c.typical_range}")
            lines.append(f"- **Your model expects**: {c.model_expects}")
            lines.append(f"- **Conversion**: `{c.conversion}`")
            lines.append(f"- Model range after conversion: {c.model_range}")
            if c.warning:
                lines.append(f"- **WARNING**: {c.warning}")
            lines.append("")
        omission = _omission_notice("data contracts", len(contracts), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    # 5. Method requirements
    reqs: MethodRequirements | None = knowledge.get("method_requirements")
    if reqs and reqs.requirements:
        lines = ["## MODELING REQUIREMENTS (you MUST satisfy all of these)\n"]
        selected = _take_limited(reqs.requirements, _COMPACT_LIMITS["builder"]["requirements"] if compact else None)
        for i, req in enumerate(selected, 1):
            lines.append(f"{i}. {req}\n")
        omission = _omission_notice("requirements", len(reqs.requirements), len(selected))
        if omission:
            lines.append(omission)
        lines.append("These are not optional. Failure to satisfy them produces incorrect prices.")
        sections.append("\n".join(lines))

    # 6. Lessons (ranked by relevance + severity)
    lessons: list[Lesson] = knowledge.get("lessons", [])
    if lessons:
        lines = ["## Lessons (ranked by relevance)\n"]
        selected = _take_limited(lessons, _COMPACT_LIMITS["builder"]["lessons"] if compact else None)
        for lesson in selected:
            lines.append(
                f"### [{lesson.severity.value.upper()}] {lesson.title}"
            )
            if lesson.symptom:
                lines.append(f"**Symptom:** {lesson.symptom}")
            lines.append(f"**Why:** {lesson.root_cause.strip()}")
            lines.append(f"**Fix:** {lesson.fix.strip()}")
            lines.append("")
        omission = _omission_notice("lessons", len(lessons), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    # 7. Decomposition notes (for novel/composite products)
    decomp: ProductDecomposition | None = knowledge.get("decomposition")
    if decomp and decomp.notes:
        sections.append(
            f"## Product Notes\n\n"
            f"Features: {', '.join(decomp.features)}\n"
            f"Notes: {decomp.notes}"
        )

    unresolved_primitives: tuple[str, ...] | list[str] = knowledge.get("unresolved_primitives", ())
    if unresolved_primitives:
        lines = ["## Unresolved Primitives\n"]
        selected = _take_limited(
            list(unresolved_primitives),
            _COMPACT_LIMITS["builder"]["unresolved_primitives"] if compact else None,
        )
        lines.extend(f"- `{primitive}`" for primitive in selected)
        omission = _omission_notice("unresolved primitives", len(unresolved_primitives), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    # 8. Matched failure signatures (during retry loop)
    sigs: list[FailureSignature] = knowledge.get("matched_signatures", [])
    if sigs:
        lines = ["## KNOWN FAILURE PATTERNS MATCHING YOUR ERRORS\n"]
        for sig in sigs:
            lines.append(
                f"- **{sig.category}** ({sig.magnitude}): {sig.diagnostic_hint}"
            )
            if sig.probable_causes:
                lines.append(
                    f"  Related lessons: {', '.join(sig.probable_causes)}"
                )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def summarize_knowledge_for_trace(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Build a compact trace/task summary for one shared-knowledge payload."""
    product_ir: ProductIR | None = knowledge.get("product_ir")
    decomposition: ProductDecomposition | None = knowledge.get("decomposition")
    cookbook: CookbookEntry | None = knowledge.get("cookbook")
    reqs: MethodRequirements | None = knowledge.get("method_requirements")
    contracts: list[DataContractEntry] = knowledge.get("data_contracts", [])
    principles: list[Principle] = knowledge.get("principles", [])
    lessons: list[Lesson] = knowledge.get("lessons", [])
    unresolved_primitives: tuple[str, ...] | list[str] = knowledge.get("unresolved_primitives", ())

    summary: dict[str, Any] = {
        "principle_ids": [principle.id for principle in principles],
        "lesson_ids": [lesson.id for lesson in lessons],
        "lesson_titles": [lesson.title for lesson in lessons],
        "lesson_count": len(lessons),
        "cookbook_method": cookbook.method if cookbook else None,
        "data_contracts": [contract.name for contract in contracts],
        "requirement_count": len(reqs.requirements) if reqs else 0,
        "requirement_method": reqs.method if reqs else None,
        "unresolved_primitives": list(unresolved_primitives),
    }
    summary.update(_adapter_lifecycle_summary())

    basket_instrument = False
    if product_ir is not None:
        basket_instrument = product_ir.instrument == "basket_path_payoff"
    elif decomposition is not None:
        basket_instrument = decomposition.instrument == "basket_path_payoff"
    if not basket_instrument:
        basket_instrument = any(
            "ranked_observation"
            in getattr(getattr(lesson, "applies_when", None), "features", ())
            for lesson in lessons
        )
    if basket_instrument:
        superseded_ids = identify_superseded_basket_lesson_ids()
        summary["superseded_lesson_ids"] = superseded_ids
        summary["superseded_lesson_count"] = len(superseded_ids)

    if product_ir is not None:
        summary.update(
            {
                "instrument": product_ir.instrument,
                "payoff_family": product_ir.payoff_family,
                "exercise_style": product_ir.exercise_style,
                "model_family": product_ir.model_family,
                "route_families": list(getattr(product_ir, "route_families", ()) or ()),
            }
        )
    elif decomposition is not None:
        summary.update(
            {
                "instrument": decomposition.instrument,
                "payoff_family": None,
                "exercise_style": None,
                "model_family": None,
            }
        )

    return summary


def format_review_knowledge_for_prompt(
    knowledge: dict[str, Any],
    *,
    audience: str = "reviewer",
    compact: bool = False,
) -> str:
    """Format shared knowledge as markdown for review-stage LLM prompts.

    Args:
        knowledge: The dict returned by ``retrieve_for_task()``.
        audience: Label for the review role (e.g. "reviewer", "arbiter",
            "model_validator").  Used as a heading prefix in the requirements
            section so the LLM knows which perspective to adopt.
        compact: When True, truncate each section per ``_COMPACT_LIMITS``.
    """
    sections: list[str] = []

    principles: list[Principle] = knowledge.get("principles", [])
    if principles:
        lines = ["## Shared Review Principles\n"]
        selected = _take_limited(principles, _COMPACT_LIMITS["review"]["principles"] if compact else None)
        for principle in selected:
            lines.append(f"- **{principle.id}**: {principle.rule}")
        omission = _omission_notice("principles", len(principles), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    product_ir: ProductIR | None = knowledge.get("product_ir")
    if product_ir is not None:
        lines = [
            "## Product Semantics\n",
            f"- Instrument: `{product_ir.instrument}`",
            f"- Exercise style: `{product_ir.exercise_style}`",
            f"- State dependence: `{product_ir.state_dependence}`",
            f"- Model family: `{product_ir.model_family}`",
        ]
        if product_ir.payoff_traits:
            lines.append(
                "- Payoff traits: "
                + ", ".join(f"`{trait}`" for trait in product_ir.payoff_traits)
            )
        sections.append("\n".join(lines))

    reqs: MethodRequirements | None = knowledge.get("method_requirements")
    if reqs and reqs.requirements:
        lines = [f"## {audience.upper()} CHECKPOINTS\n"]
        selected = _take_limited(reqs.requirements, _COMPACT_LIMITS["review"]["requirements"] if compact else None)
        for requirement in selected:
            lines.append(f"- {requirement}")
        omission = _omission_notice("requirements", len(reqs.requirements), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    lessons: list[Lesson] = knowledge.get("lessons", [])
    if lessons:
        lines = ["## Shared Failure Memory\n"]
        selected = _take_limited(lessons, _COMPACT_LIMITS["review"]["lessons"] if compact else None)
        for lesson in selected:
            lines.append(f"### [{lesson.severity.value.upper()}] {lesson.title}")
            if lesson.symptom:
                lines.append(f"**Symptom:** {lesson.symptom}")
            lines.append(f"**Why:** {lesson.root_cause.strip()}")
            lines.append(f"**Fix:** {lesson.fix.strip()}")
            lines.append("")
        omission = _omission_notice("lessons", len(lessons), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    adapter_warnings = _adapter_lifecycle_warnings(compact=compact)
    if adapter_warnings:
        sections.append(adapter_warnings)

    sigs: list[FailureSignature] = knowledge.get("matched_signatures", [])
    if sigs:
        lines = ["## Known Failure Patterns\n"]
        for sig in sigs:
            lines.append(
                f"- **{sig.category}** ({sig.magnitude}): {sig.diagnostic_hint}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_decomposition_knowledge_for_prompt(knowledge: dict[str, Any], *, compact: bool = False) -> str:
    """Format shared knowledge for decomposition / routing prompts.

    The API map is surfaced first so route selection starts from the compact
    family-level navigation layer before the broader routing context.
    """
    sections: list[str] = []

    try:
        api_map = format_api_map_for_prompt(compact=compact)
        if api_map:
            sections.append(api_map)
    except Exception:
        pass

    principles: list[Principle] = knowledge.get("principles", [])
    if principles:
        lines = ["## Shared Routing Principles\n"]
        selected = _take_limited(principles, _COMPACT_LIMITS["routing"]["principles"] if compact else None)
        for principle in selected:
            lines.append(f"- **{principle.id}**: {principle.rule}")
        omission = _omission_notice("principles", len(principles), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    lessons: list[Lesson] = knowledge.get("lessons", [])
    if lessons:
        lines = ["## Prior Lessons From Similar Products\n"]
        selected = _take_limited(lessons, _COMPACT_LIMITS["routing"]["lessons"] if compact else None)
        for lesson in selected:
            lines.append(
                f"- [{lesson.severity.value.upper()}] {lesson.title}: {lesson.fix.strip()}"
            )
        omission = _omission_notice("lessons", len(lessons), len(selected))
        if omission:
            lines.append(omission)
        sections.append("\n".join(lines))

    adapter_warnings = _adapter_lifecycle_warnings(compact=compact)
    if adapter_warnings:
        sections.append(adapter_warnings)

    decomp: ProductDecomposition | None = knowledge.get("decomposition")
    if decomp and decomp.notes:
        sections.append(
            "## Similar Product Notes\n\n"
            f"Features: {', '.join(decomp.features)}\n"
            f"Notes: {decomp.notes}"
        )

    return "\n\n".join(sections)


def format_distilled_knowledge_for_prompt(
    knowledge: dict[str, Any],
    *,
    audience: str = "builder",
) -> str:
    """Format the smallest useful reusable memory card for repeated tasks."""
    sections: list[str] = []
    product_ir: ProductIR | None = knowledge.get("product_ir")
    cookbook: CookbookEntry | None = knowledge.get("cookbook")
    reqs: MethodRequirements | None = knowledge.get("method_requirements")
    principles: list[Principle] = knowledge.get("principles", [])
    lessons: list[Lesson] = knowledge.get("lessons", [])
    unresolved_primitives: tuple[str, ...] | list[str] = knowledge.get("unresolved_primitives", ())

    if audience == "builder":
        limits = _DISTILLED_LIMITS["builder"]
        lines = []
        api_map = format_api_map_for_prompt(compact=True)
        if api_map:
            lines.append(api_map)
        lines.append("## Distilled Build Memory\n")
        if product_ir is not None:
            lines.append(
                f"- Product: `{product_ir.instrument}` / `{product_ir.payoff_family}` / "
                f"`{product_ir.exercise_style}`"
            )
        if cookbook is not None:
            lines.append(f"- Default method family: `{cookbook.method}`")
            if cookbook.description:
                lines.append(f"- Method intent: {cookbook.description}")
        if reqs and reqs.requirements:
            lines.append("- Non-negotiable requirements:")
            for requirement in _take_limited(reqs.requirements, limits["requirements"]):
                lines.append(f"  - {requirement}")
        if lessons:
            lines.append("- Repeated fixes to reuse:")
            for lesson in _take_limited(lessons, limits["lessons"]):
                lines.append(f"  - `{lesson.title}` -> {lesson.fix.strip()}")
        if unresolved_primitives:
            lines.append(
                "- Open primitive gaps: "
                + ", ".join(f"`{primitive}`" for primitive in _take_limited(list(unresolved_primitives), 3))
            )
        adapter_summary = _adapter_lifecycle_summary()
        stale_ids = adapter_summary["stale_adapter_ids"]
        if stale_ids:
            lines.append(
                "- Stale adapters: "
                + ", ".join(f"`{adapter_id}`" for adapter_id in _take_limited(stale_ids, 3))
            )
        sections.append("\n".join(lines))

    elif audience == "review":
        limits = _DISTILLED_LIMITS["review"]
        lines = ["## Distilled Review Memory\n"]
        if principles:
            lines.append("- Review principles:")
            for principle in _take_limited(principles, limits["principles"]):
                lines.append(f"  - `{principle.id}`: {principle.rule}")
        if reqs and reqs.requirements:
            lines.append("- Review checkpoints:")
            for requirement in _take_limited(reqs.requirements, limits["requirements"]):
                lines.append(f"  - {requirement}")
        if lessons:
            lines.append("- Known failure traps:")
            for lesson in _take_limited(lessons, limits["lessons"]):
                lines.append(f"  - `{lesson.title}` -> {lesson.root_cause.strip()}")
        adapter_summary = _adapter_lifecycle_summary()
        stale_ids = adapter_summary["stale_adapter_ids"]
        if stale_ids:
            lines.append(
                "- Stale adapters: "
                + ", ".join(f"`{adapter_id}`" for adapter_id in _take_limited(stale_ids, 3))
            )
        sections.append("\n".join(lines))

    elif audience == "routing":
        limits = _DISTILLED_LIMITS["routing"]
        lines = []
        api_map = format_api_map_for_prompt(compact=True)
        if api_map:
            lines.append(api_map)
        lines.append("## Distilled Routing Memory\n")
        if product_ir is not None:
            lines.append(
                "- Route cues: "
                f"`{product_ir.instrument}`, `{product_ir.model_family}`, "
                + ", ".join(f"`{family}`" for family in product_ir.candidate_engine_families[:4])
            )
            if getattr(product_ir, "route_families", ()):
                lines.append(
                    "- Exact route families: "
                    + ", ".join(f"`{family}`" for family in product_ir.route_families[:4])
                )
        if principles:
            lines.append("- Routing principles:")
            for principle in _take_limited(principles, limits["principles"]):
                lines.append(f"  - `{principle.id}`: {principle.rule}")
        if lessons:
            lines.append("- Similar-product lessons:")
            for lesson in _take_limited(lessons, limits["lessons"]):
                lines.append(f"  - `{lesson.title}` -> {lesson.fix.strip()}")
        adapter_summary = _adapter_lifecycle_summary()
        stale_ids = adapter_summary["stale_adapter_ids"]
        if stale_ids:
            lines.append(
                "- Stale adapters: "
                + ", ".join(f"`{adapter_id}`" for adapter_id in _take_limited(stale_ids, 3))
            )
        sections.append("\n".join(lines))
    else:
        raise ValueError(f"Unsupported distilled-knowledge audience: {audience}")

    return "\n\n".join(section for section in sections if section.strip())


def build_shared_knowledge_payload(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Build compact-first and expanded shared-knowledge views for reuse."""
    builder_text_distilled = format_distilled_knowledge_for_prompt(knowledge, audience="builder")
    builder_text = format_knowledge_for_prompt(knowledge, compact=True)
    builder_text_expanded = format_knowledge_for_prompt(knowledge, compact=False)
    review_text_distilled = format_distilled_knowledge_for_prompt(knowledge, audience="review")
    review_text = format_review_knowledge_for_prompt(knowledge, compact=True)
    review_text_expanded = format_review_knowledge_for_prompt(knowledge, compact=False)
    routing_text_distilled = format_distilled_knowledge_for_prompt(knowledge, audience="routing")
    routing_text = format_decomposition_knowledge_for_prompt(knowledge, compact=True)
    routing_text_expanded = format_decomposition_knowledge_for_prompt(knowledge, compact=False)

    summary = summarize_knowledge_for_trace(knowledge)
    summary["prompt_sizes"] = {
        "builder": {
            "distilled_chars": len(builder_text_distilled),
            "compact_chars": len(builder_text),
            "expanded_chars": len(builder_text_expanded),
        },
        "review": {
            "distilled_chars": len(review_text_distilled),
            "compact_chars": len(review_text),
            "expanded_chars": len(review_text_expanded),
        },
        "routing": {
            "distilled_chars": len(routing_text_distilled),
            "compact_chars": len(routing_text),
            "expanded_chars": len(routing_text_expanded),
        },
    }

    return {
        "knowledge": knowledge,
        "builder_text_distilled": builder_text_distilled,
        "builder_text": builder_text,
        "builder_text_expanded": builder_text_expanded,
        "review_text_distilled": review_text_distilled,
        "review_text": review_text,
        "review_text_expanded": review_text_expanded,
        "routing_text_distilled": routing_text_distilled,
        "routing_text": routing_text,
        "routing_text_expanded": routing_text_expanded,
        "summary": summary,
    }


def _take_limited(items, limit: int | None):
    """Return the first ``limit`` items, or all items when limit is not set."""
    if limit is None:
        return list(items)
    return list(items)[: max(limit, 0)]


def _omission_notice(label: str, total: int, shown: int) -> str:
    """Render a compact omission notice when a prompt view was truncated."""
    omitted = total - shown
    if omitted <= 0:
        return ""
    return f"- [omitted {omitted} additional {label}]"


def _truncate_text(text: str, max_chars: int, *, label: str) -> str:
    """Trim a large block of prompt text and keep the truncation explicit."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + f"\n\n[truncated {label}]"
