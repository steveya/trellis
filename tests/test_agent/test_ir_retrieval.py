"""Tests for IR-native retrieval and prompt formatting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_retrieval_spec_from_american_put_ir_contains_semantic_fields():
    from trellis.agent.knowledge.decompose import decompose_to_ir, retrieval_spec_from_ir

    ir = decompose_to_ir("American put option on equity")
    spec = retrieval_spec_from_ir(ir, preferred_method="monte_carlo")

    assert spec.method == "monte_carlo"
    assert spec.instrument == "american_put"
    assert spec.exercise_style == "american"
    assert spec.state_dependence == "terminal_markov"
    assert spec.model_family == "equity_diffusion"
    assert "early_exercise" in spec.features
    assert "exercise" in spec.candidate_engine_families


def test_retrieve_for_product_ir_surfaces_ir_summary_and_unresolved_primitives():
    from trellis.agent.knowledge import format_knowledge_for_prompt, retrieve_for_product_ir
    from trellis.agent.knowledge.decompose import decompose_to_ir

    ir = decompose_to_ir("American Asian barrier option under Heston with early exercise")
    knowledge = retrieve_for_product_ir(ir, preferred_method="monte_carlo")

    assert knowledge["product_ir"] == ir
    assert knowledge["unresolved_primitives"] == ir.unresolved_primitives

    text = format_knowledge_for_prompt(knowledge)
    assert "## Product Semantics" in text
    assert "Exercise style: `american`" in text
    assert "Model family: `stochastic_volatility`" in text
    assert "## Unresolved Primitives" in text
    assert "path_dependent_early_exercise_under_stochastic_vol" in text


def test_retrieve_for_product_ir_surfaces_exact_route_families():
    from trellis.agent.knowledge import format_knowledge_for_prompt, retrieve_for_product_ir
    from trellis.agent.knowledge.decompose import decompose_to_ir

    ir = decompose_to_ir("Callable bond with semiannual coupon and call schedule")
    knowledge = retrieve_for_product_ir(ir, preferred_method="rate_tree")

    text = format_knowledge_for_prompt(knowledge)
    assert "Exact route families" in text
    assert "rate_lattice" in text


def test_ir_native_retrieval_ranks_early_exercise_lessons_for_american_put():
    from trellis.agent.knowledge import retrieve_for_product_ir
    from trellis.agent.knowledge.decompose import decompose_to_ir

    ir = decompose_to_ir("American put option on equity")
    knowledge = retrieve_for_product_ir(ir, preferred_method="monte_carlo")

    assert knowledge["lessons"]
    top_titles = {lesson.title.lower() for lesson in knowledge["lessons"][:3]}
    assert any("exercise" in title or "lsm" in title for title in top_titles)


def test_autonomous_build_tracking_passes_stage_aware_knowledge_retriever():
    from trellis.agent.knowledge.autonomous import _build_with_tracking
    from trellis.agent.knowledge.decompose import decompose_to_ir

    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )
    decomposition = SimpleNamespace(
        method="monte_carlo",
        features=("early_exercise",),
        instrument="american_put",
    )
    gap_report = SimpleNamespace(
        confidence=1.0,
        missing=[],
        retrieved_lesson_ids=[],
    )
    observed: dict[str, object] = {}

    def _fake_build_payoff(*args, **kwargs):
        import trellis.agent.executor as executor

        observed["executor_retrieve"] = executor._retrieve_knowledge
        request = executor.KnowledgeRetrievalRequest(
            audience="builder",
            stage="initial_build",
            attempt_number=1,
            knowledge_surface="compact",
            prompt_surface="compact",
            retry_reason=None,
            instrument_type="american_option",
            pricing_method="monte_carlo",
            product_ir=product_ir,
            compiled_request=None,
        )
        text = kwargs["knowledge_retriever"](request)
        assert "## Distilled Build Memory" in text
        assert "## Pricing Method" not in text

        class DummyPayoff:
            pass

        return DummyPayoff

    import trellis.agent.executor as executor

    original_retrieve = executor._retrieve_knowledge
    with patch("trellis.agent.executor.build_payoff", side_effect=_fake_build_payoff):
        payoff_cls, meta = _build_with_tracking(
            description="American put option on equity",
            instrument_type="american_option",
            decomposition=decomposition,
            product_ir=product_ir,
            gap_report=gap_report,
            model=None,
            market_state=None,
            max_retries=1,
            validation="fast",
            force_rebuild=True,
        )

    assert payoff_cls.__name__ == "DummyPayoff"
    assert meta["failures"] == []
    assert observed["executor_retrieve"] is original_retrieve
    assert "lesson_ids" in meta["knowledge_summary"]


def test_autonomous_build_tracking_refreshes_knowledge_between_attempts(monkeypatch):
    from trellis.agent.knowledge.autonomous import _build_with_tracking
    from trellis.agent.knowledge.decompose import decompose_to_ir

    product_ir = decompose_to_ir(
        "American put option on equity",
        instrument_type="american_option",
    )
    decomposition = SimpleNamespace(
        method="monte_carlo",
        features=("early_exercise",),
        instrument="american_put",
    )
    gap_report = SimpleNamespace(
        confidence=1.0,
        missing=[],
        retrieved_lesson_ids=[],
    )

    retrieval_calls = {"count": 0}
    observed_texts: list[str] = []

    def fake_retrieve_for_product_ir(*args, **kwargs):
        retrieval_calls["count"] += 1
        return {"marker": retrieval_calls["count"]}

    def fake_build_shared_payload(knowledge):
        marker = knowledge["marker"]
        return {
            "builder_text_distilled": f"distilled {marker}",
            "builder_text": f"compact {marker}",
            "builder_text_expanded": f"expanded {marker}",
            "review_text_distilled": f"review distilled {marker}",
            "review_text": f"review compact {marker}",
            "review_text_expanded": f"review expanded {marker}",
            "summary": {"lesson_ids": [f"lesson_{marker}"]},
        }

    def _fake_build_payoff(*args, **kwargs):
        import trellis.agent.executor as executor

        request = executor.KnowledgeRetrievalRequest(
            audience="builder",
            stage="semantic_validation_failed",
            attempt_number=2,
            knowledge_surface="expanded",
            prompt_surface="semantic_repair",
            retry_reason="semantic_validation",
            instrument_type="american_option",
            pricing_method="monte_carlo",
            product_ir=product_ir,
            compiled_request=None,
        )
        observed_texts.append(kwargs["knowledge_retriever"](request))
        observed_texts.append(kwargs["knowledge_retriever"](request))

        class DummyPayoff:
            pass

        return DummyPayoff

    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieve_for_product_ir",
        fake_retrieve_for_product_ir,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.build_shared_knowledge_payload",
        fake_build_shared_payload,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.format_gap_warnings",
        lambda report: "gap warning",
    )

    with patch("trellis.agent.executor.build_payoff", side_effect=_fake_build_payoff):
        payoff_cls, meta = _build_with_tracking(
            description="American put option on equity",
            instrument_type="american_option",
            decomposition=decomposition,
            product_ir=product_ir,
            gap_report=gap_report,
            model=None,
            market_state=None,
            max_retries=2,
            validation="fast",
            force_rebuild=True,
        )

    assert payoff_cls.__name__ == "DummyPayoff"
    assert meta["failures"] == []
    assert observed_texts[0] != observed_texts[1]
    assert observed_texts[0].startswith("expanded 2")
    assert observed_texts[1].startswith("expanded 3")
    assert observed_texts[0].endswith("gap warning")
    assert observed_texts[1].endswith("gap warning")
