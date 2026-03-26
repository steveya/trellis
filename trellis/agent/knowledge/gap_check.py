"""Pre-flight knowledge audit — detect gaps before building.

Checks what knowledge is available for a task and produces a
confidence score + gap report.  Injected into the build prompt
so the agent knows where its knowledge is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trellis.agent.knowledge.schema import ProductDecomposition, RetrievalSpec
from trellis.agent.knowledge.store import KnowledgeStore, expand_features


@dataclass
class GapReport:
    """Result of a pre-flight knowledge audit."""

    has_decomposition: bool = False
    decomposition_learned: bool = False
    has_cookbook: bool = False
    cookbook_method: str | None = None
    lesson_count: int = 0
    has_contracts: bool = False
    has_requirements: bool = False
    missing: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_lesson_ids: list[str] = field(default_factory=list)


def gap_check(
    decomposition: ProductDecomposition,
    store: KnowledgeStore | None = None,
) -> GapReport:
    """Audit knowledge readiness before a build.

    Returns a GapReport with a confidence score (0.0–1.0) and a list
    of human-readable gap descriptions for prompt injection.
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    report = GapReport()

    # 1. Decomposition
    static_decomp = store._decompositions.get(decomposition.instrument)
    report.has_decomposition = static_decomp is not None
    report.decomposition_learned = decomposition.learned

    if not report.has_decomposition and not decomposition.learned:
        report.missing.append(
            f"No static decomposition for '{decomposition.instrument}' — "
            f"using features {list(decomposition.features)}"
        )

    # 2. Cookbook
    cookbook = store._load_cookbook(decomposition.method)
    report.has_cookbook = cookbook is not None
    report.cookbook_method = decomposition.method

    if not report.has_cookbook:
        report.missing.append(
            f"No cookbook template for method '{decomposition.method}' — "
            f"agent must rely on reference implementations and lessons"
        )

    # 3. Lessons
    expanded = expand_features(list(decomposition.features), store._features)
    spec = RetrievalSpec(
        method=decomposition.method,
        features=expanded,
        instrument=decomposition.instrument,
    )
    result = store.retrieve_for_task(spec)
    lessons = result.get("lessons", [])
    report.lesson_count = len(lessons)
    report.retrieved_lesson_ids = [l.id for l in lessons]

    if report.lesson_count == 0:
        report.missing.append(
            "No relevant lessons found — this is uncharted territory"
        )
    elif report.lesson_count < 3:
        report.missing.append(
            f"Only {report.lesson_count} relevant lessons — limited prior experience"
        )

    # 4. Data contracts
    contracts = result.get("data_contracts", [])
    report.has_contracts = len(contracts) > 0

    if not report.has_contracts:
        report.missing.append(
            f"No data contracts for method '{decomposition.method}' — "
            f"watch for unit/convention mismatches"
        )

    # 5. Method requirements
    reqs = result.get("method_requirements")
    report.has_requirements = reqs is not None and bool(reqs.requirements)

    if not report.has_requirements:
        report.missing.append(
            f"No modeling requirements for method '{decomposition.method}'"
        )

    # Compute confidence score
    report.confidence = _score(report)

    return report


def _score(report: GapReport) -> float:
    """Compute a 0.0–1.0 confidence score from the gap report."""
    score = 0.0

    # Decomposition: 0.2 for static, 0.1 for learned
    if report.has_decomposition:
        score += 0.2
    elif not report.decomposition_learned:
        score += 0.05  # LLM fallback
    else:
        score += 0.1

    # Cookbook: 0.25
    if report.has_cookbook:
        score += 0.25

    # Lessons: up to 0.3 (0.05 per lesson, capped)
    score += min(0.3, report.lesson_count * 0.05)

    # Contracts: 0.15
    if report.has_contracts:
        score += 0.15

    # Requirements: 0.1
    if report.has_requirements:
        score += 0.1

    return round(min(1.0, score), 2)


def format_gap_warnings(report: GapReport) -> str:
    """Format gap report as warnings for prompt injection."""
    if not report.missing:
        return ""

    lines = [
        f"## KNOWLEDGE GAPS (confidence: {report.confidence:.0%})\n",
        "The following knowledge is MISSING for this task. "
        "Proceed with extra caution in these areas:\n",
    ]
    for gap in report.missing:
        lines.append(f"- **WARNING**: {gap}")

    if not report.has_cookbook:
        lines.append(
            "\nSince no cookbook is available, you MUST rely on "
            "reference implementations and the lessons above. "
            "Follow the patterns in the reference code closely."
        )

    return "\n".join(lines)
