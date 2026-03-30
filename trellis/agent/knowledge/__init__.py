"""Trellis Knowledge System — production-grade knowledge infrastructure.

Public API:

    from trellis.agent.knowledge import retrieve_for_task, get_store

    knowledge = retrieve_for_task(
        method="rate_tree",
        features=["callable", "fixed_coupons"],
        instrument="callable_bond",
    )

    from trellis.agent.knowledge import format_knowledge_for_prompt
    prompt_text = format_knowledge_for_prompt(knowledge)
"""

from __future__ import annotations

from typing import Any

from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.decompose import (
    clear_decomposition_cache,
    decomposition_cache_stats,
    decompose,
    decompose_to_ir,
    retrieval_spec_from_ir,
)
from trellis.agent.knowledge.schema import ProductIR, RetrievalSpec
from trellis.agent.knowledge.store import KnowledgeStore
from trellis.agent.knowledge.retrieval import (
    build_shared_knowledge_payload,
    format_decomposition_knowledge_for_prompt,
    format_knowledge_for_prompt,
    format_review_knowledge_for_prompt,
    summarize_knowledge_for_trace,
)
from trellis.agent.knowledge.import_registry import (
    get_package_map,
    get_repo_facts,
    get_repo_revision,
    get_symbol_map,
    get_test_map,
    suggest_tests_for_symbol,
)
from trellis.agent.knowledge.api_map import (
    format_api_map_for_prompt,
    get_api_map,
)


# Module-level singleton — hot tier loads at first access
_store: KnowledgeStore | None = None


def get_store() -> KnowledgeStore:
    """Get (or create) the global KnowledgeStore singleton."""
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def retrieve_for_task(
    method: str,
    features: list[str] | None = None,
    instrument: str | None = None,
    error_signatures: list[str] | None = None,
    include_benchmarks: bool = False,
    max_lessons: int = 7,
) -> dict[str, Any]:
    """Retrieve all relevant knowledge for a pricing task.

    This is the primary public API — replaces the scattered calls to
    get_experience_for_task(), get_cookbook(), format_contracts_for_prompt(),
    and modeling_requirements lookup.

    Features are the primary retrieval axis.  Lessons are matched via
    feature union (any matching feature contributes), ranked by relevance
    score and severity.

    Parameters
    ----------
    method
        Pricing method (e.g., "rate_tree", "monte_carlo", "analytical").
    features
        Product features (e.g., ["callable", "fixed_coupons"]).
        Expanded transitively via the feature taxonomy.
    instrument
        Optional instrument key for decomposition lookup.
    error_signatures
        Error messages to match against known failure patterns.
    include_benchmarks
        Whether to include benchmark suites in the result.
    max_lessons
        Maximum number of lessons to return (default 7).
    """
    spec = RetrievalSpec(
        method=normalize_method(method),
        features=features or [],
        instrument=instrument,
        error_signatures=error_signatures or [],
        max_lessons=max_lessons,
        include_benchmarks=include_benchmarks,
    )
    return get_store().retrieve_for_task(spec)


def retrieve_for_product_ir(
    product_ir: ProductIR,
    *,
    preferred_method: str | None = None,
    error_signatures: list[str] | None = None,
    include_benchmarks: bool = False,
    max_lessons: int = 7,
) -> dict[str, Any]:
    """Retrieve knowledge using ``ProductIR`` as the primary semantic input."""
    spec = retrieval_spec_from_ir(
        product_ir,
        preferred_method=preferred_method,
    )
    spec = RetrievalSpec(
        method=spec.method,
        features=spec.features,
        instrument=spec.instrument,
        exercise_style=spec.exercise_style,
        state_dependence=spec.state_dependence,
        schedule_dependence=spec.schedule_dependence,
        model_family=spec.model_family,
        candidate_engine_families=spec.candidate_engine_families,
        unresolved_primitives=spec.unresolved_primitives,
        error_signatures=error_signatures or [],
        max_lessons=max_lessons,
        include_benchmarks=include_benchmarks,
    )
    knowledge = get_store().retrieve_for_task(spec)
    knowledge["product_ir"] = product_ir
    knowledge["unresolved_primitives"] = tuple(product_ir.unresolved_primitives)
    return knowledge


def reload() -> None:
    """Force reload all knowledge tiers."""
    global _store
    clear_decomposition_cache()
    if _store is not None:
        _store.reload()
    else:
        _store = KnowledgeStore()


def build_with_knowledge(
    description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    market_state=None,
    **kwargs,
):
    """Knowledge-aware build: gap check → build → reflect → enrich.

    See autonomous.py for full documentation.
    """
    from trellis.agent.knowledge.autonomous import build_with_knowledge as _bwk
    return _bwk(
        description, instrument_type=instrument_type,
        model=model, market_state=market_state, **kwargs,
    )


__all__ = [
    "get_store",
    "get_repo_revision",
    "get_symbol_map",
    "get_package_map",
    "get_test_map",
    "get_repo_facts",
    "suggest_tests_for_symbol",
    "get_api_map",
    "format_api_map_for_prompt",
    "retrieve_for_task",
    "retrieve_for_product_ir",
    "build_shared_knowledge_payload",
    "format_knowledge_for_prompt",
    "format_review_knowledge_for_prompt",
    "format_decomposition_knowledge_for_prompt",
    "summarize_knowledge_for_trace",
    "reload",
    "build_with_knowledge",
    "clear_decomposition_cache",
    "decompose",
    "decomposition_cache_stats",
    "decompose_to_ir",
    "retrieval_spec_from_ir",
    "KnowledgeStore",
    "ProductIR",
    "RetrievalSpec",
]
