"""Pre-flight knowledge audit — detect gaps before building.

Checks what knowledge is available for a task and produces a
confidence score + gap report.  Injected into the build prompt
so the agent knows where its knowledge is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trellis.agent.knowledge.schema import (
    ProductDecomposition,
    RetrievalSpec,
    SimilarProductMatch,
)
from trellis.agent.knowledge.store import KnowledgeStore, expand_features


@dataclass
class RouteGap:
    """Describes a missing or immature route for a (method, product) pair."""

    kind: str  # "no_known_route" | "route_not_yet_promoted"
    message: str
    candidate_routes: tuple[str, ...] = ()


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
    has_promoted_route: bool = False
    route_gap: RouteGap | None = None
    missing: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_lesson_ids: list[str] = field(default_factory=list)
    similar_products: tuple[SimilarProductMatch, ...] = ()


def gap_check(
    decomposition: ProductDecomposition,
    store: KnowledgeStore | None = None,
) -> GapReport:
    """Audit knowledge readiness before a build.

    Checks five knowledge dimensions -- decomposition, cookbook template,
    relevant lessons, data contracts, and modeling requirements -- and
    returns a GapReport.  The confidence score (0.0-1.0) is a weighted
    sum: decomposition 0.2, cookbook 0.25, lessons up to 0.3 (0.05 each,
    capped at 6), contracts 0.15, requirements 0.1.  The ``missing`` list
    contains human-readable gap descriptions that are injected into the
    builder prompt so the LLM knows where its knowledge is thin.
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

    # 6. Route availability
    report.route_gap = _check_route_gap(decomposition)
    report.has_promoted_route = report.route_gap is None

    if report.route_gap is not None:
        if report.route_gap.kind == "no_known_route":
            report.missing.append(
                f"No known route for ({decomposition.method}, "
                f"{decomposition.instrument}) — build will be ad-hoc"
            )
        elif report.route_gap.kind == "route_not_yet_promoted":
            report.missing.append(
                f"Found candidate route(s) for ({decomposition.method}, "
                f"{decomposition.instrument}) but none promoted yet: "
                f"{', '.join(report.route_gap.candidate_routes)}"
            )

    # Compute confidence score
    report.confidence = _score(report)

    if (
        report.route_gap is not None
        or report.confidence < 0.7
        or not report.has_decomposition
    ):
        report.similar_products = tuple(
            store.find_similar_products(
                RetrievalSpec(
                    method=decomposition.method,
                    features=list(decomposition.features),
                    instrument=decomposition.instrument,
                )
            )
        )

    return report


def _check_route_gap(decomposition: ProductDecomposition) -> RouteGap | None:
    """Check whether a promoted route exists for this (method, instrument) pair.

    Constructs a minimal ProductIR from the decomposition to query the route
    registry.  Returns None if a promoted route exists, or a RouteGap
    describing what's missing.
    """
    try:
        from trellis.agent.knowledge.schema import ProductIR
        from trellis.agent.route_registry import (
            load_route_registry,
            match_candidate_routes,
        )

        minimal_ir = ProductIR(
            instrument=decomposition.instrument,
            payoff_family=decomposition.instrument,
        )
        registry = load_route_registry()

        # Check for promoted routes
        promoted = match_candidate_routes(
            registry, decomposition.method, minimal_ir, promoted_only=True,
        )
        if promoted:
            return None

        # Check for candidate/validated routes
        analysis_registry = load_route_registry(include_discovered=True)
        all_matches = match_candidate_routes(
            analysis_registry, decomposition.method, minimal_ir, promoted_only=False,
        )
        candidates = [r for r in all_matches if r.status in ("candidate", "validated")]
        if candidates:
            return RouteGap(
                kind="route_not_yet_promoted",
                message=f"Found {len(candidates)} candidate route(s) but none promoted yet.",
                candidate_routes=tuple(c.id for c in candidates),
            )

        return RouteGap(
            kind="no_known_route",
            message=f"No known route for ({decomposition.method}, {decomposition.instrument}).",
        )
    except Exception:
        # Registry not available — don't block the build
        return None


def _score(report: GapReport) -> float:
    """Compute a 0.0-1.0 confidence score from the gap report.

    Each knowledge dimension contributes a fixed weight:
      - Static decomposition: +0.20 (or +0.10 if learned, +0.05 if LLM fallback)
      - Cookbook template:     +0.20
      - Lessons:              +0.05 per lesson, up to +0.25
      - Data contracts:       +0.10
      - Method requirements:  +0.10
      - Promoted route:       +0.15
    A score below 0.5 triggers extra build retries in the autonomous loop.
    """
    score = 0.0

    # Decomposition: 0.2 for static, 0.1 for learned
    if report.has_decomposition:
        score += 0.2
    elif not report.decomposition_learned:
        score += 0.05  # LLM fallback
    else:
        score += 0.1

    # Cookbook: 0.20
    if report.has_cookbook:
        score += 0.20

    # Lessons: up to 0.25 (0.05 per lesson, capped)
    score += min(0.25, report.lesson_count * 0.05)

    # Contracts: 0.10
    if report.has_contracts:
        score += 0.10

    # Requirements: 0.10
    if report.has_requirements:
        score += 0.10

    # Promoted route: 0.15
    if report.has_promoted_route:
        score += 0.15

    return round(min(1.0, score), 2)


def format_gap_warnings(report: GapReport) -> str:
    """Format gap report as warnings for prompt injection."""
    missing = list(getattr(report, "missing", []) or [])
    similar_products = tuple(getattr(report, "similar_products", ()) or ())

    if not missing and not similar_products:
        return ""

    lines = [f"## KNOWLEDGE GAPS (confidence: {float(getattr(report, 'confidence', 0.0)):.0%})\n"]
    if missing:
        lines.append(
            "The following knowledge is MISSING for this task. "
            "Proceed with extra caution in these areas:\n",
        )
        for gap in missing:
            lines.append(f"- **WARNING**: {gap}")

    if similar_products:
        lines.append("\n## Similar Products")
        for match in similar_products[:3]:
            shared = ", ".join(f"`{feature}`" for feature in match.shared_features[:4])
            route_hint = (
                " via "
                + ", ".join(f"`{route}`" for route in match.promoted_routes[:2])
                if match.promoted_routes
                else ""
            )
            lines.append(
                f"- `{match.instrument}` ({match.score:.0%} match): shared features {shared}{route_hint}"
            )

    if not report.has_cookbook:
        lines.append(
            "\nSince no cookbook is available, you MUST rely on "
            "reference implementations and the lessons above. "
            "Follow the patterns in the reference code closely."
        )

    return "\n".join(lines)
