from __future__ import annotations

from types import SimpleNamespace


def test_evaluate_prompt_fallback_uses_unified_shared_knowledge(monkeypatch):
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    captured = {}

    def fake_retrieve_for_task(*, method, instrument, features=None, **kwargs):
        captured["method"] = method
        captured["instrument"] = instrument
        return {"lessons": [], "principles": []}

    def fake_format_knowledge_for_prompt(knowledge, compact=False):
        captured["knowledge"] = knowledge
        captured["compact"] = compact
        return "## Shared Knowledge\n- Use shared retrieval."

    monkeypatch.setattr("trellis.agent.knowledge.retrieve_for_task", fake_retrieve_for_task)
    monkeypatch.setattr(
        "trellis.agent.knowledge.format_knowledge_for_prompt",
        fake_format_knowledge_for_prompt,
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.black"],
            required_market_data={"discount_curve"},
            model_to_build="european_option",
            reasoning="Known vanilla route.",
        ),
        knowledge_context="",
    )

    assert captured["method"] == "analytical"
    assert captured["instrument"] == "european_option"
    assert captured["compact"] is True
    assert "## Shared Knowledge" in prompt
    assert "Use shared retrieval." in prompt


def test_executor_knowledge_context_escalates_after_first_attempt(monkeypatch):
    from trellis.agent.executor import (
        _builder_knowledge_context_for_attempt,
        _review_knowledge_context_for_attempt,
    )

    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieval.build_shared_knowledge_payload",
        lambda knowledge: {
            "builder_text": "compact builder",
            "builder_text_expanded": "expanded builder",
            "review_text": "compact review",
            "review_text_expanded": "expanded review",
        },
    )

    compiled_request = SimpleNamespace(
        knowledge={"lessons": []},
        knowledge_text="compact builder",
        review_knowledge_text="compact review",
    )

    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=1,
        retry_reason=None,
        compiled_request=compiled_request,
        product_ir=None,
    )
    retry_builder_text, retry_builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=2,
        retry_reason="validation",
        compiled_request=compiled_request,
        product_ir=None,
    )
    review_text, review_surface = _review_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=1,
        compiled_request=compiled_request,
        product_ir=None,
    )
    retry_review_text, retry_review_surface = _review_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=2,
        compiled_request=compiled_request,
        product_ir=None,
    )

    assert (builder_text, builder_surface) == ("compact builder", "compact")
    assert (retry_builder_text, retry_builder_surface) == ("expanded builder", "expanded")
    assert (review_text, review_surface) == ("compact review", "compact")
    assert (retry_review_text, retry_review_surface) == ("expanded review", "expanded")


def test_executor_import_retry_keeps_compact_builder_knowledge(monkeypatch):
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieval.build_shared_knowledge_payload",
        lambda knowledge: {
            "builder_text": "compact builder",
            "builder_text_expanded": "expanded builder",
        },
    )

    compiled_request = SimpleNamespace(
        knowledge={"lessons": []},
        knowledge_text="compact builder",
    )

    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=2,
        retry_reason="import_validation",
        compiled_request=compiled_request,
        product_ir=None,
    )

    assert (builder_text, builder_surface) == ("compact builder", "compact")


def test_evaluate_prompt_compact_surface_uses_route_card_and_truncated_references():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black", "trellis.core.date_utils"),
        symbols_to_reuse=("black76_call", "black76_put", "year_fraction"),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
                PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
            ),
            adapters=("map_spot_to_forward",),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Black helper": "x" * 2000},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.black"],
            required_market_data={"discount_curve"},
            model_to_build="european_option",
            reasoning="Known vanilla route.",
        ),
        knowledge_context="## Shared Knowledge\n- Compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "## Structured Route Card" in prompt
    assert "## Primitive Lookup" in prompt
    assert "## Thin Adapter Plan" in prompt
    assert "## Invariant Pack" in prompt
    assert "## Structured Generation Plan" not in prompt
    assert "[truncated reference]" in prompt


def test_evaluate_prompt_import_repair_surface_uses_import_card_without_references():
    from trellis.agent.codegen_guardrails import GenerationPlan
    from trellis.agent.prompts import evaluate_prompt

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black", "trellis.core.date_utils"),
        symbols_to_reuse=("black76_call", "black76_put", "year_fraction"),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Black helper": "x" * 2000},
        knowledge_context="",
        generation_plan=plan,
        prompt_surface="import_repair",
    )

    assert "## Import Repair Card" in prompt
    assert "## Reference implementations" in prompt
    assert "### Black helper" not in prompt


def test_evaluate_prompt_semantic_repair_surface_uses_semantic_card():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),),
            adapters=("map_spot_to_forward",),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Black helper": "x" * 2000},
        knowledge_context="",
        generation_plan=plan,
        prompt_surface="semantic_repair",
    )

    assert "## Semantic Repair Card" in prompt
    assert "## Structured Generation Plan" not in prompt
    assert "[truncated reference]" in prompt


def test_evaluate_prompt_expanded_surface_uses_full_generation_plan():
    from trellis.agent.codegen_guardrails import GenerationPlan
    from trellis.agent.prompts import evaluate_prompt

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black",),
        symbols_to_reuse=("black76_call",),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Black helper": "def x():\n    return 1\n"},
        knowledge_context="",
        generation_plan=plan,
        prompt_surface="expanded",
    )

    assert "## Structured Generation Plan" in prompt


def test_evaluate_prompt_compact_surface_trims_fx_reference_budget():
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={
            "FX helper 1": "a" * 1200,
            "FX helper 2": "b" * 1200,
            "FX helper 3": "c" * 1200,
        },
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.black"],
            required_market_data={"discount_curve", "forward_curve", "fx_rates", "spot"},
            model_to_build="european_option",
            reasoning="FX vanilla route.",
        ),
        knowledge_context="## Shared Knowledge\n- FX compact route",
        prompt_surface="compact",
    )

    assert "### FX helper 1" in prompt
    assert "### FX helper 2" in prompt
    assert "### FX helper 3" not in prompt
    assert "[omitted 1 additional reference modules]" in prompt


def test_prompt_mentions_approved_early_exercise_policy_classes():
    from trellis.agent.prompts import evaluate_prompt

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        knowledge_context="",
        prompt_surface="compact",
    )

    assert "longstaff_schwartz" in prompt
    assert "tsitsiklis_van_roy" in prompt
    assert "primal_dual_mc" in prompt
    assert "stochastic_mesh" in prompt
    assert "## Structured Route Card" not in prompt
