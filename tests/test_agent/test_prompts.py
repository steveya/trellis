from __future__ import annotations

from types import SimpleNamespace


def test_system_prompt_mentions_api_map_navigation():
    from trellis.agent.prompts import system_prompt

    prompt = system_prompt()
    assert "inspect_api_map" in prompt
    assert "API map" in prompt or "API Map" in prompt
    assert "find_symbol" in prompt


def test_evaluate_prompt_fallback_uses_unified_shared_knowledge(monkeypatch):
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    captured = {}

    def fake_retrieve_for_task(*, method, instrument, features=None, **kwargs):
        captured["method"] = method
        captured["instrument"] = instrument
        return {"lessons": [], "principles": []}

    def fake_build_shared_knowledge_payload(knowledge, *, pricing_method=None, **kwargs):
        captured["knowledge"] = knowledge
        captured["pricing_method"] = pricing_method
        return {
            "builder_text_distilled": "## Generated Skills\n- Use shared retrieval.",
        }

    monkeypatch.setattr("trellis.agent.knowledge.retrieve_for_task", fake_retrieve_for_task)
    monkeypatch.setattr(
        "trellis.agent.knowledge.build_shared_knowledge_payload",
        fake_build_shared_knowledge_payload,
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
    assert captured["pricing_method"] == "analytical"
    assert "## Generated Skills" in prompt
    assert "Use shared retrieval." in prompt


def test_evaluate_prompt_renders_selection_basis_and_assumptions():
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

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
            selection_reason="simplest_valid_default",
            assumption_summary=(
                "simplest_valid_assumption_set",
                "closed_form_or_quasi_closed_form_route",
                "no_path_sampling_required",
            ),
        ),
        knowledge_context="",
    )

    assert "Selection basis and assumptions:" in prompt
    assert "simplest_valid_default" in prompt
    assert "closed_form_or_quasi_closed_form_route" in prompt


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
    assert retry_builder_surface == "compact"
    assert retry_builder_text.startswith("compact builder")
    assert "## Retry Focus" in retry_builder_text
    assert (review_text, review_surface) == ("compact review", "compact")
    assert retry_review_surface == "expanded"
    assert retry_review_text.startswith("expanded review")


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

    assert builder_surface == "compact"
    assert builder_text.startswith("compact builder")
    assert "Use only import-registry and API-map-backed modules and symbols." in builder_text


def test_executor_semantic_retry_expands_builder_knowledge_and_records_stage(monkeypatch):
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieval.build_shared_knowledge_payload",
        lambda knowledge: {
            "builder_text": "compact builder",
            "builder_text_distilled": "distilled builder",
            "builder_text_expanded": "expanded builder",
        },
    )

    compiled_request = SimpleNamespace(
        knowledge={"lessons": []},
        knowledge_text="compact builder",
    )
    build_meta: dict[str, object] = {"knowledge_summary": {}}

    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=None,
        instrument_type="european_option",
        attempt_number=2,
        retry_reason="semantic_validation",
        compiled_request=compiled_request,
        product_ir=None,
        build_meta=build_meta,
    )

    assert builder_surface == "expanded"
    assert builder_text.startswith("expanded builder")
    assert "Match the runtime contract exactly" in builder_text
    assert build_meta["knowledge_summary"]["retrieval_stages"] == ["semantic_validation_failed"]
    assert build_meta["knowledge_summary"]["retrieval_sources"] == ["compiled_request_payload"]


def test_executor_knowledge_light_profile_reuses_compiled_request_text_even_on_expanded_retry():
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    build_meta: dict[str, object] = {"knowledge_summary": {}}
    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="analytical"),
        instrument_type="quanto_option",
        attempt_number=2,
        retry_reason="semantic_validation",
        compiled_request=SimpleNamespace(
            knowledge={"knowledge_profile": "knowledge_light"},
            knowledge_text="## Knowledge-Light Mode\n- Compiler first.",
            review_knowledge_text="## Knowledge-Light Review Mode\n- Review compiler boundary.",
            knowledge_summary={"knowledge_profile": "knowledge_light"},
        ),
        product_ir="demo-ir",
        build_meta=build_meta,
    )

    assert builder_surface == "expanded"
    assert builder_text.startswith("## Knowledge-Light Mode")
    assert build_meta["knowledge_summary"]["retrieval_sources"] == ["compiled_request"]


def test_executor_stage_aware_callback_overrides_cached_knowledge():
    from trellis.agent.executor import (
        KnowledgeRetrievalRequest,
        _builder_knowledge_context_for_attempt,
    )

    observed: dict[str, object] = {}

    def fake_retriever(request: KnowledgeRetrievalRequest) -> str:
        observed["request"] = request
        return f"{request.stage}:{request.knowledge_surface}"

    build_meta: dict[str, object] = {"knowledge_summary": {}}
    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="pde_solver"),
        instrument_type="european_option",
        attempt_number=2,
        retry_reason="semantic_validation",
        compiled_request=SimpleNamespace(
            knowledge={"lessons": []},
            knowledge_text="stale compact",
        ),
        product_ir="demo-ir",
        build_meta=build_meta,
        knowledge_retriever=fake_retriever,
    )

    assert builder_surface == "expanded"
    assert builder_text.startswith("semantic_validation_failed:expanded")
    assert "## Retry Focus" in builder_text
    request = observed["request"]
    assert request.stage == "semantic_validation_failed"
    assert request.knowledge_surface == "expanded"
    assert request.pricing_method == "pde_solver"
    assert build_meta["knowledge_summary"]["retrieval_sources"] == ["callback"]


def test_executor_validation_retry_stays_compact():
    from trellis.agent.executor import (
        KnowledgeRetrievalRequest,
        _builder_knowledge_context_for_attempt,
    )

    observed: dict[str, object] = {}

    def fake_retriever(request: KnowledgeRetrievalRequest) -> str:
        observed["request"] = request
        return f"{request.stage}:{request.knowledge_surface}"

    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="monte_carlo"),
        instrument_type="credit_default_swap",
        attempt_number=2,
        retry_reason="validation",
        compiled_request=SimpleNamespace(
            knowledge={"lessons": []},
            knowledge_text="stale compact",
        ),
        product_ir="demo-ir",
        build_meta={"knowledge_summary": {}},
        knowledge_retriever=fake_retriever,
    )

    assert builder_surface == "compact"
    assert builder_text.startswith("validation_failed:compact")
    request = observed["request"]
    assert request.stage == "validation_failed"
    assert request.knowledge_surface == "compact"


def test_executor_actual_market_retry_expands_and_records_selected_artifacts(monkeypatch):
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    monkeypatch.setattr(
        "trellis.agent.knowledge.load_skill_index",
        lambda: SimpleNamespace(
            records=(
                SimpleNamespace(
                    skill_id="route_hint:quanto_runtime_contract",
                    kind="route_hint",
                    title="Quanto runtime contract",
                    summary="Bind domestic and foreign curves from the live market_state instead of widening fallbacks.",
                    source_artifact="quanto_runtime_contract",
                    source_path="",
                    instrument_types=("quanto_option",),
                    method_families=("analytical",),
                    route_families=(),
                    failure_buckets=(),
                    concepts=(),
                    tags=(),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                ),
            ),
        ),
    )

    build_meta: dict[str, object] = {"knowledge_summary": {}}
    builder_text, builder_surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="analytical"),
        instrument_type="quanto_option",
        attempt_number=2,
        retry_reason="actual_market_smoke",
        compiled_request=None,
        product_ir=SimpleNamespace(instrument="quanto_option", route_families=()),
        build_meta=build_meta,
        knowledge_retriever=lambda request: f"{request.stage}:{request.knowledge_surface}",
    )

    assert builder_surface == "expanded"
    assert builder_text.startswith("actual_market_smoke_failed:expanded")
    assert "## Stage-Aware Skills" in builder_text
    assert "Quanto runtime contract" in builder_text
    assert "Recover against the real task market contract" in builder_text
    assert build_meta["knowledge_summary"]["retrieval_stages"] == ["actual_market_smoke_failed"]
    assert build_meta["knowledge_summary"]["selected_artifact_ids"] == [
        "route_hint:quanto_runtime_contract"
    ]


def test_prompt_skill_selection_skips_historical_route_notes_even_on_exact_route_match(monkeypatch):
    from trellis.agent.knowledge.skills import select_prompt_skill_artifacts

    monkeypatch.setattr(
        "trellis.agent.knowledge.skills.load_skill_index",
        lambda: SimpleNamespace(
            records=(
                SimpleNamespace(
                    skill_id="route_hint:pde_theta_1d:note:1",
                    kind="historical_note",
                    title="Legacy PDE route note",
                    summary="Construct Grid and theta_method_1d manually.",
                    source_artifact="pde_theta_1d",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("pde_solver",),
                    route_families=("pde_solver",),
                    failure_buckets=(),
                    concepts=(),
                    tags=("route:pde_theta_1d",),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=50,
                    instruction_type="route_hint",
                    source_kind="route_card",
                    lineage_status="derived",
                    lineage_evidence=("route.match_method_to_cookbook",),
                ),
                SimpleNamespace(
                    skill_id="route_hint:pde_theta_1d:route-helper",
                    kind="route_hint",
                    title="PDE helper contract",
                    summary="Use the checked helper surface instead of bespoke solver glue.",
                    source_artifact="pde_theta_1d",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("pde_solver",),
                    route_families=("pde_solver",),
                    failure_buckets=(),
                    concepts=(),
                    tags=("module:trellis.models.equity_option_pde",),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                    lineage_status="derived",
                    lineage_evidence=("route.match_method_to_cookbook",),
                ),
            ),
        ),
    )

    artifacts = select_prompt_skill_artifacts(
        "European option PDE retry",
        audience="builder",
        stage="validation_failed",
        instrument_type="european_option",
        pricing_method="pde_solver",
        route_ids=("pde_theta_1d",),
        route_families=("pde_solver",),
    )

    assert [artifact["id"] for artifact in artifacts] == [
        "route_hint:pde_theta_1d:route-helper",
    ]


def test_prompt_skill_selection_prefers_hard_constraints_over_exact_route_note_matches(monkeypatch):
    from trellis.agent.knowledge.skills import select_prompt_skill_artifacts

    monkeypatch.setattr(
        "trellis.agent.knowledge.skills.load_skill_index",
        lambda: SimpleNamespace(
            records=(
                SimpleNamespace(
                    skill_id="route_hint:local_vol_monte_carlo:note:1",
                    kind="route_hint",
                    title="Route-local note",
                    summary="Legacy local-vol route note.",
                    source_artifact="local_vol_monte_carlo",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("monte_carlo",),
                    route_families=("local_vol",),
                    failure_buckets=(),
                    concepts=(),
                    tags=("route:local_vol_monte_carlo",),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=50,
                    instruction_type="route_hint",
                    source_kind="route_card",
                    lineage_status="derived",
                    lineage_evidence=("route.match_method_to_cookbook",),
                ),
                SimpleNamespace(
                    skill_id="route_hint:local_vol_monte_carlo:route-helper",
                    kind="route_hint",
                    title="Local-vol helper",
                    summary="Use the approved local-vol helper directly.",
                    source_artifact="local_vol_monte_carlo",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("monte_carlo",),
                    route_families=("local_vol",),
                    failure_buckets=(),
                    concepts=(),
                    tags=(),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                    lineage_status="derived",
                    lineage_evidence=("route.match_method_to_cookbook",),
                ),
            ),
        ),
    )

    artifacts = select_prompt_skill_artifacts(
        "Local-vol retry",
        audience="builder",
        stage="validation_failed",
        instrument_type="european_option",
        pricing_method="monte_carlo",
        route_ids=("local_vol_monte_carlo",),
        route_families=("local_vol",),
    )

    assert [artifact["id"] for artifact in artifacts][:2] == [
        "route_hint:local_vol_monte_carlo:route-helper",
        "route_hint:local_vol_monte_carlo:note:1",
    ]


def test_executor_stage_aware_skills_prefer_hard_constraints_over_route_note_matches(monkeypatch):
    from trellis.agent.executor import KnowledgeRetrievalRequest, _stage_aware_skill_artifacts

    monkeypatch.setattr(
        "trellis.agent.knowledge.load_skill_index",
        lambda: SimpleNamespace(
            records=(
                SimpleNamespace(
                    skill_id="route_hint:local_vol_monte_carlo:note:1",
                    kind="route_hint",
                    title="Route-local note",
                    summary="Legacy local-vol route note.",
                    source_artifact="local_vol_monte_carlo",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("monte_carlo",),
                    route_families=("local_vol",),
                    failure_buckets=(),
                    concepts=(),
                    tags=("route:local_vol_monte_carlo",),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=50,
                    instruction_type="route_hint",
                    source_kind="route_card",
                ),
                SimpleNamespace(
                    skill_id="route_hint:local_vol_monte_carlo:route-helper",
                    kind="route_hint",
                    title="Local-vol helper",
                    summary="Use the approved local-vol helper directly.",
                    source_artifact="local_vol_monte_carlo",
                    source_path="",
                    instrument_types=("european_option",),
                    method_families=("monte_carlo",),
                    route_families=("local_vol",),
                    failure_buckets=(),
                    concepts=(),
                    tags=(),
                    origin="canonical",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=100,
                    instruction_type="hard_constraint",
                    source_kind="route_card",
                ),
            ),
        ),
    )

    artifacts = _stage_aware_skill_artifacts(
        KnowledgeRetrievalRequest(
            audience="builder",
            stage="validation_failed",
            attempt_number=2,
            knowledge_surface="compact",
            prompt_surface="builder",
            retry_reason="validation",
            instrument_type="european_option",
            pricing_method="monte_carlo",
            product_ir=SimpleNamespace(instrument="european_option", route_families=("local_vol",)),
            compiled_request=SimpleNamespace(
                generation_plan=SimpleNamespace(
                    primitive_plan=SimpleNamespace(route="local_vol_monte_carlo", route_family="local_vol"),
                ),
            ),
        )
    )

    assert [artifact["id"] for artifact in artifacts][:2] == [
        "route_hint:local_vol_monte_carlo:route-helper",
        "route_hint:local_vol_monte_carlo:note:1",
    ]


def test_executor_stage_aware_skills_skip_duplicate_guidance(monkeypatch):
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    monkeypatch.setattr(
        "trellis.agent.knowledge.load_skill_index",
        lambda: SimpleNamespace(
            records=(
                SimpleNamespace(
                    skill_id="lesson:dup",
                    kind="lesson",
                    title="Duplicate route guidance",
                    summary="Use the checked-in CDS helper directly inside evaluate.",
                    source_artifact="dup",
                    source_path="",
                    instrument_types=("credit_default_swap",),
                    method_families=("monte_carlo",),
                    route_families=(),
                    failure_buckets=(),
                    concepts=(),
                    tags=(),
                    origin="captured",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=10,
                    instruction_type="route_hint",
                    source_kind="lesson_entry",
                ),
            ),
        ),
    )

    text, _ = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="monte_carlo"),
        instrument_type="credit_default_swap",
        attempt_number=2,
        retry_reason="semantic_validation",
        compiled_request=None,
        product_ir=SimpleNamespace(instrument="credit_default_swap"),
        knowledge_retriever=lambda request: (
            "Use the checked-in CDS helper directly inside evaluate."
        ),
    )

    assert "## Stage-Aware Skills" not in text


def test_executor_stage_aware_skills_respect_compact_budget(monkeypatch):
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    long_summary = (
        "Keep the generated adapter thin, bind the checked-in helper directly, "
        "preserve the route contract, avoid fallback abstractions, and keep the "
        "market-binding semantics explicit across every retry attempt."
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.load_skill_index",
        lambda: SimpleNamespace(
            records=tuple(
                SimpleNamespace(
                    skill_id=f"lesson:{idx}",
                    kind="lesson",
                    title=f"Guidance {idx}",
                    summary=f"{long_summary} ({idx})",
                    source_artifact=f"lesson_{idx}",
                    source_path="",
                    instrument_types=("credit_default_swap",),
                    method_families=("monte_carlo",),
                    route_families=(),
                    failure_buckets=(),
                    concepts=(),
                    tags=(),
                    origin="captured",
                    parents=(),
                    supersedes=(),
                    status="promoted",
                    confidence=1.0,
                    updated_at="",
                    precedence_rank=10 - idx,
                    instruction_type="route_hint",
                    source_kind="lesson_entry",
                )
                for idx in range(5)
            ),
        ),
    )

    build_meta: dict[str, object] = {"knowledge_summary": {}}
    text, surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="monte_carlo"),
        instrument_type="credit_default_swap",
        attempt_number=2,
        retry_reason="validation",
        compiled_request=None,
        product_ir=SimpleNamespace(instrument="credit_default_swap"),
        build_meta=build_meta,
        knowledge_retriever=lambda request: "base retry knowledge",
    )

    assert surface == "compact"
    assert "## Stage-Aware Skills" in text
    assert len(build_meta["knowledge_summary"]["selected_artifact_ids"]) < 5


def test_executor_credit_default_swap_retry_adds_disambiguation_guidance():
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    def fake_retriever(request) -> str:
        return "base credit knowledge"

    text, surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="monte_carlo"),
        instrument_type="credit_default_swap",
        attempt_number=2,
        retry_reason="import_validation",
        compiled_request=None,
        product_ir=SimpleNamespace(instrument="credit_default_swap"),
        knowledge_retriever=fake_retriever,
    )

    assert surface == "compact"
    assert "single-name CDS / credit_default_swap" in text
    assert "Do not reinterpret CDS here as nth_to_default" in text
    assert "Do not import copula or Gaussian-copula machinery" in text
    assert "Use only import-registry and API-map-backed modules and symbols." in text
    assert "Do not import `trellis.models.processes.gbm`" in text
    assert "Do not import or instantiate `MonteCarloEngine`" in text
    assert "Single-name CDS Monte Carlo does not need an equity price process" in text
    assert "np.random.default_rng" in text
    assert "build_period_schedule" in text or "build_cds_schedule" in text
    assert "use `prev_date` for `year_fraction(prev_date, pay_date, ...)` and `prev_t` for survival/default-time thresholds" in text
    assert "persistent `alive` indicator" in text
    assert "Update `alive` before premium accrual" in text
    assert "spread = float(spec.spread)" in text
    assert "spread *= 1e-4" in text
    assert "Do not hard-code `n_paths=50000`" in text
    assert "`spec.n_paths`" in text
    assert "seed=42" in text
    assert "`100` and `0.01` must produce the same CDS PV" in text


def test_executor_european_black_scholes_retry_keeps_plain_vanilla_shape():
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    text, surface = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="analytical"),
        instrument_type="european_option",
        attempt_number=2,
        retry_reason="code_generation",
        compiled_request=SimpleNamespace(
            request=SimpleNamespace(metadata={"comparison_target": "black_scholes"})
        ),
        product_ir=SimpleNamespace(instrument="european_option"),
        knowledge_retriever=lambda request: "base european analytical knowledge",
    )

    assert surface == "compact"
    assert "plain Black-Scholes / Black76 comparator lane" in text
    assert "black76_call" in text and "black76_put" in text
    assert "Do not import or call `terminal_vanilla_from_basis`" in text
    assert "Do not emit only an `evaluate()` fragment" in text


def test_executor_nth_to_default_disambiguation_keeps_credit_family_distinct():
    from trellis.agent.executor import _builder_knowledge_context_for_attempt

    text, _ = _builder_knowledge_context_for_attempt(
        pricing_plan=SimpleNamespace(method="copula"),
        instrument_type="nth_to_default",
        attempt_number=1,
        retry_reason=None,
        compiled_request=None,
        product_ir=SimpleNamespace(instrument="nth_to_default"),
        knowledge_retriever=lambda request: "base nth-to-default knowledge",
    )

    assert "Treat this request as nth_to_default / multi-name credit" in text
    assert "single-name CDS" in text


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

    assert "## Structured Lane Card" in prompt
    assert "## Backend Lookup (Secondary To Lane Obligations)" in prompt
    assert "## Thin Adapter Plan" in prompt
    assert "## Invariant Pack" in prompt
    assert "## Structured Generation Plan" not in prompt
    assert "[truncated reference]" in prompt


def test_evaluate_prompt_compact_surface_mentions_black76_basis_helpers():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.black",),
        approved_modules=("trellis.models.black", "trellis.core.date_utils"),
        symbols_to_reuse=(
            "black76_call",
            "black76_put",
            "black76_asset_or_nothing_call",
            "black76_asset_or_nothing_put",
            "black76_cash_or_nothing_call",
            "black76_cash_or_nothing_put",
            "terminal_vanilla_from_basis",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
                PrimitiveRef("trellis.models.black", "black76_put", "pricing_kernel"),
                PrimitiveRef("trellis.models.black", "black76_asset_or_nothing_call", "pricing_kernel", required=False),
                PrimitiveRef("trellis.models.black", "black76_asset_or_nothing_put", "pricing_kernel", required=False),
                PrimitiveRef("trellis.models.analytical", "terminal_vanilla_from_basis", "assembly_helper", required=False),
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
            reasoning="Known digital route.",
        ),
        knowledge_context="## Shared Knowledge\n- Compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "black76_cash_or_nothing_call" in prompt
    assert "black76_asset_or_nothing_call" in prompt
    assert "terminal_vanilla_from_basis" in prompt
    assert "basis claims" in prompt


def test_evaluate_prompt_plain_european_analytical_surface_prefers_direct_black76():
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
        reference_sources={"Black helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.black"],
            required_market_data={"discount_curve"},
            model_to_build="european_option",
            reasoning="Plain vanilla comparator route.",
        ),
        knowledge_context="## Shared Knowledge\n- Compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "compute `T`, `df`, `sigma`, `forward`, then call `black76_call` or `black76_put` directly" in prompt
    assert "Do not use `terminal_vanilla_from_basis`" in prompt
    assert "Return a complete module with the spec class and payoff class" in prompt


def test_evaluate_prompt_compact_surface_mentions_fx_route_helper():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="european_option",
        inspected_modules=("trellis.models.fx_vanilla", "trellis.models.analytical.fx", "trellis.models.black"),
        approved_modules=("trellis.models.fx_vanilla", "trellis.models.analytical.fx", "trellis.models.black", "trellis.core.date_utils"),
        symbols_to_reuse=(
            "ResolvedGarmanKohlhagenInputs",
            "garman_kohlhagen_price_raw",
            "price_fx_vanilla_analytical",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_garman_kohlhagen",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.fx_vanilla", "price_fx_vanilla_analytical", "route_helper"),
                PrimitiveRef("trellis.models.analytical.fx", "garman_kohlhagen_price_raw", "pricing_kernel"),
            ),
            adapters=("map_fx_spot_and_curves_to_garman_kohlhagen_inputs",),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"FX helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.analytical.fx"],
            required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot"},
            model_to_build="european_option",
            reasoning="FX vanilla route.",
        ),
        knowledge_context="## Shared Knowledge\n- FX compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "price_fx_vanilla_analytical" in prompt
    assert "garman_kohlhagen_price_raw" in prompt
    assert "ResolvedGarmanKohlhagenInputs" in prompt
    assert "Garman-Kohlhagen" in prompt
    assert "route helper" in prompt


def test_evaluate_prompt_compact_surface_mentions_swaption_helper_route():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="swaption",
        inspected_modules=("trellis.models.rate_style_swaption",),
        approved_modules=("trellis.models.rate_style_swaption",),
        symbols_to_reuse=(
            "price_swaption_black76",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="analytical_black76",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.rate_style_swaption", "price_swaption_black76", "route_helper"),
            ),
            adapters=(),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Swaption helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.black"],
            required_market_data={"discount_curve", "black_vol_surface", "forward_curve"},
            model_to_build="swaption",
            reasoning="European swaption route.",
        ),
        knowledge_context="## Shared Knowledge\n- Swaption compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "price_swaption_black76" in prompt
    assert "Hull-White-implied Black vol" in prompt
    assert "annuity" in prompt


def test_evaluate_prompt_compact_surface_mentions_jamshidian_raw_helper():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="zcb_option",
        inspected_modules=("trellis.models.zcb_option", "trellis.models.analytical.jamshidian"),
        approved_modules=("trellis.models.zcb_option", "trellis.models.analytical.jamshidian"),
        symbols_to_reuse=(
            "price_zcb_option_jamshidian",
            "resolve_zcb_option_hw_inputs",
            "ResolvedJamshidianInputs",
            "zcb_option_hw_raw",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="zcb_option_analytical",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.zcb_option", "price_zcb_option_jamshidian", "route_helper"),
            ),
            adapters=("reuse_checked_in_zcb_option_analytical_helper",),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Jamshidian helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.models.zcb_option"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="zcb_option",
            reasoning="Jamshidian route.",
        ),
        knowledge_context="## Shared Knowledge\n- Jamshidian compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "price_zcb_option_jamshidian" in prompt
    assert "resolve_zcb_option_hw_inputs" in prompt
    assert "ResolvedJamshidianInputs" in prompt
    assert "zcb_option_hw_raw" in prompt


def test_evaluate_prompt_compact_surface_mentions_current_pde_contract():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=["trellis.models.pde.theta_method"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="european_option",
            reasoning="PDE route.",
        ),
        instrument_type="european_option",
        inspected_modules=("trellis.models.pde.theta_method",),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"PDE helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=["trellis.models.pde.theta_method"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="european_option",
            reasoning="PDE route.",
        ),
        knowledge_context="## Shared Knowledge\n- PDE route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "## Structured Lane Card" in prompt
    assert "trellis.models.equity_option_pde" in prompt
    assert "price_vanilla_equity_option_pde" in prompt
    assert "Map `implementation_target=theta_0.5` to `theta=0.5`" in prompt
    assert "Grid + BlackScholesOperator + theta_method_1d" in prompt
    assert "Import Repair Card" not in prompt


def test_evaluate_prompt_cds_surface_mentions_credit_curve_contract():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    product_ir = decompose_to_ir(
        "CDS pricing: hazard rate MC vs survival prob analytical",
        instrument_type="cds",
    )
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.core.date_utils"],
            required_market_data={"discount_curve", "credit_curve"},
            model_to_build="credit_default_swap",
            reasoning="CDS route.",
        ),
        instrument_type="cds",
        inspected_modules=("trellis.core.date_utils",),
        product_ir=product_ir,
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"CDS helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=["trellis.core.date_utils"],
            required_market_data={"discount_curve", "credit_curve"},
            model_to_build="credit_default_swap",
            reasoning="CDS route.",
        ),
        knowledge_context="## Shared Knowledge\n- CDS compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "credit_curve" in prompt
    assert "survival probability" in prompt.lower() or "survival_probability" in prompt
    assert "single-name CDS analytical routes" in prompt
    assert "150 bp -> 0.015" in prompt
    assert "explicit payment/default schedule" in prompt
    assert "build_cds_schedule" in prompt
    assert "price_cds_analytical" in prompt
    assert "Route family: `credit_default_swap`" in prompt
    assert "Route family: `event_triggered_two_legged_contract`" not in prompt
    assert "market_state.discount.discount(t)" in prompt
    assert "spec.start_date` as the time origin" in prompt
    assert "accrued-on-default premium adjustment" in prompt
    assert "Do not average adjacent discount factors" in prompt
    assert "price_cds_analytical" in prompt


def test_distilled_builder_memory_keeps_legacy_cds_labels_and_omits_nearest_products():
    from trellis.agent.knowledge.retrieval import format_distilled_knowledge_for_prompt
    from trellis.agent.knowledge.schema import ProductIR, SimilarProductMatch

    text = format_distilled_knowledge_for_prompt(
        {
            "product_ir": ProductIR(
                instrument="cds",
                payoff_family="event_triggered_two_legged_contract",
            ),
            "similar_products": [
                SimilarProductMatch(instrument="bond", method="analytical", score=0.6),
                SimilarProductMatch(instrument="autocallable", method="monte_carlo", score=0.53),
            ],
        },
        audience="builder",
    )

    assert "- Product: `cds` / `cds` / `none`" in text
    assert "Nearest known products" not in text


def test_evaluate_prompt_cds_monte_carlo_surface_mentions_get_numpy_and_schedule_loop():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    product_ir = decompose_to_ir(
        "CDS pricing: hazard rate MC vs survival prob analytical",
        instrument_type="cds",
    )
    plan = build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.core.date_utils", "trellis.core.differentiable"],
            required_market_data={"discount_curve", "credit_curve"},
            model_to_build="nth_to_default",
            reasoning="CDS route.",
        ),
        instrument_type="cds",
        inspected_modules=("trellis.core.date_utils", "trellis.core.differentiable"),
        product_ir=product_ir,
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"CDS helper": "x" * 1500},
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.core.date_utils", "trellis.core.differentiable"],
            required_market_data={"discount_curve", "credit_curve"},
            model_to_build="nth_to_default",
            reasoning="CDS route.",
        ),
        knowledge_context="## Shared Knowledge\n- CDS compact route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "get_numpy" in prompt
    assert "np = get_numpy()" in prompt
    assert "explicit payment/default schedule" in prompt
    assert "hazard_rate" in prompt or "survival_probability" in prompt
    assert "150 bp -> 0.015" in prompt
    assert "spread = float(spec.spread)" in prompt
    assert "spread *= 1e-4" in prompt
    assert "`100` and `0.01`" in prompt
    assert "Do not import or instantiate `MonteCarloEngine`" in prompt
    assert "np.random.default_rng" in prompt
    assert "build_cds_schedule" in prompt
    assert "price_cds_monte_carlo" in prompt
    assert "many paths" in prompt
    assert "scalar `alive`" in prompt
    assert "1.0 - s_pay / s_prev" in prompt
    assert "1.0 - exp(-hazard * dt)" in prompt
    assert "Do not discount protection at sampled default times `tau`" in prompt
    assert "Use `spec.start_date` as the time origin for Monte Carlo schedule times" in prompt
    assert "keep `prev_date` and `prev_t` as separate variables" in prompt
    assert "persistent `alive` indicator" in prompt
    assert "Update `alive` immediately after drawing `default_in_interval`" in prompt
    assert "Do not hard-code `n_paths=50000`" in prompt
    assert "`spec.n_paths`" in prompt
    assert "seed=42" in prompt
    assert "market_state.discount.discount(t)" in prompt


def test_executor_credit_default_swap_analytical_retry_pins_discount_and_time_origin():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="analytical",
        instrument_type="credit_default_swap",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="credit_default_swap"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "spec.start_date" in text
    assert "accrued-on-default premium adjustment" in text
    assert "Do not average adjacent discount factors" in text
    assert "0.5 * (prev_discount + discount)" in text
    assert "market_state.discount.discount(pay_t)" in text
    assert "from trellis.models import black" in text


def test_evaluate_prompt_cds_analytical_prefers_route_bound_modules_over_generic_family_modules():
    from trellis.agent.codegen_guardrails import build_generation_plan
    from trellis.agent.knowledge.decompose import decompose_to_ir
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    pricing_plan = PricingPlan(
        method="analytical",
        method_modules=["trellis.models.black"],
        required_market_data={"discount_curve", "credit_curve"},
        model_to_build="credit_default_swap",
        reasoning="Known CDS helper route.",
    )
    generation_plan = build_generation_plan(
        pricing_plan=pricing_plan,
        instrument_type="credit_default_swap",
        inspected_modules=("trellis.models.black",),
        product_ir=decompose_to_ir(
            "CDS pricing: hazard rate MC vs survival prob analytical",
            instrument_type="credit_default_swap",
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=pricing_plan,
        knowledge_context="## Shared Knowledge\n- CDS analytical route",
        generation_plan=generation_plan,
        prompt_surface="compact",
    )

    marker = "Route-bound modules to import and use:"
    assert marker in prompt
    modules_block = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    assert "`trellis.models.credit_default_swap`" in modules_block
    assert "`trellis.models.black`" not in modules_block
    assert "Do not import a generic parent package such as `from trellis.models import ...`" in prompt


def test_evaluate_prompt_route_less_semantic_request_prefers_compiler_inspected_modules():
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.prompts import evaluate_prompt

    compiled = compile_build_request(
        (
            "Range accrual note on SOFR paying 5.25% when SOFR stays between 1.50% "
            "and 3.25% on 2026-01-15, 2026-04-15, 2026-07-15, and 2026-10-15."
        ),
        instrument_type="range_accrual",
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=compiled.pricing_plan,
        knowledge_context="## Shared Knowledge\n- Range accrual semantic route pending.",
        generation_plan=compiled.generation_plan,
        prompt_surface="compact",
    )

    marker = "Modules to import and use:"
    assert marker in prompt
    modules_block = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    assert "`trellis.models.range_accrual`" in modules_block
    assert "`trellis.models.contingent_cashflows`" in modules_block
    assert "`trellis.models.black`" not in modules_block


def test_executor_callable_bond_rate_tree_retry_pins_vol_surface_and_control_policy():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="rate_tree",
        instrument_type="callable_bond",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="callable_bond"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "market_state.vol_surface.black_vol" in text
    assert "market_state.discount.zero_rate" in text
    assert "price_callable_bond_tree" in text
    assert "sigma_hw = black_vol" in text
    assert "build_generic_lattice" in text
    assert 'MODEL_REGISTRY["bdt"]' in text
    assert "lattice_step_from_time" in text
    assert "lattice_steps_from_timeline" in text
    assert "build_payment_timeline" in text
    assert 'resolve_lattice_exercise_policy("issuer_call"' in text


def test_executor_bermudan_swaption_rate_tree_retry_pins_helper_and_bermudan_control():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="rate_tree",
        instrument_type="bermudan_swaption",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="bermudan_swaption"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_bermudan_swaption_tree" in text
    assert "market_state.vol_surface.black_vol" in text
    assert "market_state.discount.zero_rate" in text
    assert 'resolve_lattice_exercise_policy("bermudan"' in text
    assert "price_callable_bond_tree" in text


def test_executor_bermudan_swaption_analytical_retry_pins_lower_bound_helper():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="analytical",
        instrument_type="bermudan_swaption",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="bermudan_swaption"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_bermudan_swaption_black76_lower_bound" in text
    assert "European swaption exercisable only on the final Bermudan date" in text
    assert "Do not sum one European Black76 price per exercise date" in text


def test_executor_swaption_analytical_retry_pins_helper_backed_route():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="analytical",
        instrument_type="swaption",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="swaption"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_swaption_black76" in text
    assert "Hull-White-implied Black vol" in text
    assert "annuity" in text


def test_executor_swaption_rate_tree_retry_pins_helper_backed_route():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="rate_tree",
        instrument_type="swaption",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="swaption"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_swaption_tree" in text
    assert "single-exercise European" in text
    assert "swap_start == expiry_date" in text


def test_executor_swaption_monte_carlo_retry_pins_event_aware_route():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="monte_carlo",
        instrument_type="swaption",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="swaption"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_swaption_monte_carlo" in text
    assert "thin adapter" in text or "Keep the route thin" in text
    assert "do not hardcode `sigma = 0.01`" in text
    assert "do not synthesize a GBM equity path" in text


def test_executor_zcb_option_analytical_retry_mentions_jamshidian_raw_lane():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="analytical",
        instrument_type="zcb_option",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="zcb_option"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_zcb_option_jamshidian" in text
    assert "resolve_zcb_option_hw_inputs" in text
    assert "ResolvedJamshidianInputs" in text
    assert "zcb_option_hw_raw" in text


def test_evaluate_prompt_american_tree_surface_mentions_equity_tree_helper_and_longstaff_schwartz():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="rate_tree",
        instrument_type="american_option",
        inspected_modules=(
            "trellis.models.trees.binomial",
            "trellis.models.trees.backward_induction",
            "trellis.models.monte_carlo.engine",
        ),
        approved_modules=(
            "trellis.models.trees.binomial",
            "trellis.models.trees.backward_induction",
            "trellis.models.monte_carlo.engine",
        ),
        symbols_to_reuse=("BinomialTree", "backward_induction", "longstaff_schwartz"),
        proposed_tests=("tests/test_tasks/test_t07_american_put_3way.py",),
        primitive_plan=PrimitivePlan(
            route="exercise_lattice",
            engine_family="tree",
            route_family="equity_tree",
            primitives=(
                PrimitiveRef(
                    "trellis.models.equity_option_tree",
                    "price_vanilla_equity_option_tree",
                    "route_helper",
                ),
            ),
            adapters=(),
            blockers=(),
            notes=(
                "For American and Bermudan equity options, prefer `price_vanilla_equity_option_tree(...)` from `trellis.models.equity_option_tree`.",
                "If you need the lower-level lattice path, use `build_spot_lattice(..., model=\"crr\"|\"jarrow_rudd\")` with `lattice_backward_induction(...)`.",
            ),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=[
                "trellis.models.equity_option_tree",
                "trellis.models.trees.lattice",
            ],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="american_option",
            reasoning="American equity tree route.",
        ),
        knowledge_context="## Shared Knowledge\n- American tree route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "price_vanilla_equity_option_tree" in prompt
    assert "build_spot_lattice" in prompt
    assert "lattice_backward_induction" in prompt
    assert "trellis.models.equity_option_tree" in prompt
    assert "Route family: `equity_tree`" in prompt
    assert 'model="crr"' in prompt
    assert "longstaff_schwartz" in prompt


def test_evaluate_prompt_american_pde_surface_mentions_exercise_values_and_rannacher():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="pde_solver",
        instrument_type="american_option",
        inspected_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        approved_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        symbols_to_reuse=("Grid", "BlackScholesOperator", "theta_method_1d"),
        proposed_tests=("tests/test_tasks/test_t07_american_put_3way.py",),
        primitive_plan=PrimitivePlan(
            route="pde_theta_1d",
            engine_family="pde_solver",
            primitives=(
                PrimitiveRef("trellis.models.pde.grid", "Grid", "grid"),
                PrimitiveRef("trellis.models.pde.operator", "BlackScholesOperator", "spatial_operator"),
                PrimitiveRef("trellis.models.pde.theta_method", "theta_method_1d", "time_stepping"),
            ),
            adapters=("define_operator_boundary_terminal_conditions",),
            blockers=(),
            notes=(
                "Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
                "Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
                "For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
                "For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
            ),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=["trellis.models.pde.theta_method"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="american_option",
            reasoning="American equity PDE route.",
        ),
        knowledge_context="## Shared Knowledge\n- American PDE route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "exercise_values" in prompt
    assert "exercise_fn=max" in prompt
    assert "rannacher_timesteps" in prompt


def test_evaluate_prompt_callable_bond_rate_tree_surface_mentions_lattice_control_helper():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="rate_tree",
        instrument_type="callable_bond",
        inspected_modules=(
            "trellis.models.trees.lattice",
            "trellis.models.trees.models",
            "trellis.models.trees.control",
        ),
        approved_modules=(
            "trellis.models.trees.lattice",
            "trellis.models.trees.models",
            "trellis.models.trees.control",
        ),
        symbols_to_reuse=(
            "build_generic_lattice",
            "build_rate_lattice",
            "lattice_backward_induction",
            "MODEL_REGISTRY",
            "lattice_steps_from_timeline",
            "resolve_lattice_exercise_policy",
        ),
        proposed_tests=("tests/test_tasks/test_t02_bdt_callable.py",),
        primitive_plan=PrimitivePlan(
            route="exercise_lattice",
            engine_family="lattice",
            route_family="rate_lattice",
            primitives=(
                PrimitiveRef("trellis.models.trees.lattice", "build_generic_lattice", "generic_lattice_builder", required=False),
                PrimitiveRef("trellis.models.trees.lattice", "build_rate_lattice", "lattice_builder"),
                PrimitiveRef(
                    "trellis.models.trees.lattice",
                    "lattice_backward_induction",
                    "backward_induction",
                ),
                PrimitiveRef(
                    "trellis.models.trees.models",
                    "MODEL_REGISTRY",
                    "model_registry",
                    required=False,
                ),
                PrimitiveRef(
                    "trellis.models.trees.control",
                    "resolve_lattice_exercise_policy",
                    "control_policy",
                    required=False,
                ),
                PrimitiveRef(
                    "trellis.models.trees.control",
                    "lattice_steps_from_timeline",
                    "step_mapper",
                    required=False,
                ),
            ),
            adapters=("resolve_schedule_dependent_lattice_control_policy",),
            blockers=(),
            notes=(
                "Use lattice_backward_induction with a checked-in lattice exercise policy.",
                "Prefer resolve_lattice_exercise_policy(...) and lattice_steps_from_timeline(...).",
                "For BDT or explicit Hull-White comparisons, use build_generic_lattice(..., MODEL_REGISTRY[...], ...).",
            ),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="rate_tree",
            method_modules=[
                "trellis.models.trees.lattice",
                "trellis.models.trees.models",
                "trellis.models.trees.control",
            ],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="callable_bond",
            reasoning="Callable bond rate-tree route.",
        ),
        knowledge_context="## Shared Knowledge\n- Callable lattice route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "resolve_lattice_exercise_policy" in prompt
    assert "lattice_steps_from_timeline" in prompt
    assert "build_generic_lattice" in prompt
    assert "MODEL_REGISTRY" in prompt
    assert 'exercise_policy=exercise_policy' in prompt
    assert '"issuer_call"' in prompt
    assert "market_state.vol_surface.black_vol" in prompt
    assert "market_state.discount.zero_rate" in prompt


def test_evaluate_prompt_barrier_pde_surface_mentions_grid_operator_and_rannacher():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="pde_solver",
        instrument_type="barrier_option",
        inspected_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        approved_modules=(
            "trellis.models.pde.grid",
            "trellis.models.pde.operator",
            "trellis.models.pde.theta_method",
        ),
        symbols_to_reuse=("Grid", "BlackScholesOperator", "theta_method_1d"),
        proposed_tests=("tests/test_tasks/test_t09_barrier.py",),
        primitive_plan=PrimitivePlan(
            route="pde_theta_1d",
            engine_family="pde_solver",
            primitives=(
                PrimitiveRef("trellis.models.pde.grid", "Grid", "grid"),
                PrimitiveRef("trellis.models.pde.operator", "BlackScholesOperator", "spatial_operator"),
                PrimitiveRef("trellis.models.pde.theta_method", "theta_method_1d", "time_stepping"),
            ),
            adapters=("define_operator_boundary_terminal_conditions",),
            blockers=(),
            notes=(
                "Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
                "Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
                "For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
            ),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="pde_solver",
            method_modules=["trellis.models.pde.theta_method"],
            required_market_data={"discount_curve", "black_vol_surface"},
            model_to_build="barrier_option",
            reasoning="Barrier PDE route.",
        ),
        knowledge_context="## Shared Knowledge\n- Barrier PDE route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "Grid" in prompt
    assert "BlackScholesOperator" in prompt
    assert "theta_method_1d" in prompt
    assert "rannacher_timesteps" in prompt
    assert "trellis.models.pde.grid" in prompt
    assert "trellis.models.pde.operator" in prompt
    assert "lower_bc_fn" in prompt
    assert "upper_bc_fn" in prompt


def test_executor_european_option_pde_retry_pins_helper_surface_and_theta_mapping():
    from types import SimpleNamespace

    from trellis.agent.executor import KnowledgeRetrievalRequest, _route_specific_retry_lines

    request = KnowledgeRetrievalRequest(
        audience="builder",
        attempt_number=2,
        knowledge_surface="compact",
        prompt_surface="compact",
        retry_reason="validation",
        pricing_method="pde_solver",
        instrument_type="european_option",
        stage="validation_failed",
        product_ir=SimpleNamespace(instrument="european_option"),
    )

    text = "\n".join(_route_specific_retry_lines(request))

    assert "price_vanilla_equity_option_pde" in text
    assert "theta_0.5" in text
    assert "theta_1.0" in text
    assert "Grid" in text


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


def test_evaluate_prompt_compact_surface_shows_semantic_valuation_and_validation_boundary():
    from trellis.agent.platform_requests import compile_build_request
    from trellis.agent.prompts import evaluate_prompt

    compiled = compile_build_request(
        "Quanto option on SAP in USD with EUR underlier currency expiring 2025-11-15",
        instrument_type="quanto_option",
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={"Quanto helper": "def resolve_quanto_inputs(...):\n    pass\n"},
        knowledge_context="## Shared Knowledge\n- Quanto route",
        pricing_plan=compiled.pricing_plan,
        generation_plan=compiled.generation_plan,
        prompt_surface="compact",
    )

    assert "## Structured Lane Card" in prompt
    assert "- Semantic contract: `quanto_option`" in prompt
    assert "- Valuation context:" in prompt
    assert "market_source=`unbound_market_snapshot`" in prompt
    assert "- Lane boundary:" in prompt
    assert "family=`analytical`" in prompt
    assert "- Lowering boundary:" in prompt
    assert "route_alias=`quanto_adjustment_analytical`" not in prompt
    assert "- Validation contract:" in prompt
    assert "bundle=`analytical:quanto_option`" in prompt
    assert "quanto_adjustment_applied" in prompt


def test_evaluate_prompt_quanto_analytical_surface_includes_resolution_guidance():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="analytical",
        instrument_type="quanto_option",
        inspected_modules=(
            "trellis.models.black",
            "trellis.models.quanto_option",
            "trellis.models.resolution.quanto",
            "trellis.models.analytical.quanto",
        ),
        approved_modules=(
            "trellis.models.black",
            "trellis.models.quanto_option",
            "trellis.models.resolution.quanto",
            "trellis.models.analytical.quanto",
        ),
        symbols_to_reuse=(
            "black76_call",
            "resolve_quanto_inputs",
            "price_quanto_option_analytical_from_market_state",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="quanto_adjustment_analytical",
            engine_family="analytical",
            primitives=(
                PrimitiveRef("trellis.models.black", "black76_call", "pricing_kernel"),
                PrimitiveRef(
                    "trellis.models.resolution.quanto",
                    "resolve_quanto_inputs",
                    "market_binding",
                ),
                PrimitiveRef(
                    "trellis.models.quanto_option",
                    "price_quanto_option_analytical_from_market_state",
                    "route_helper",
                ),
            ),
            adapters=("reuse_shared_quanto_market_binding",),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={
            "Quanto helper": "def resolve_quanto_inputs(...):\n    pass\n",
            "Quanto analytical helper": "def price_quanto_option_analytical_from_market_state(...):\n    pass\n",
        },
        pricing_plan=PricingPlan(
            method="analytical",
            method_modules=[
                "trellis.models.black",
                "trellis.models.quanto_option",
                "trellis.models.resolution.quanto",
                "trellis.models.analytical.quanto",
            ],
            required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"},
            model_to_build="quanto_option",
            reasoning="product_ir_compiler",
        ),
        knowledge_context="## Shared Knowledge\n- Quanto analytical route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "resolve_quanto_inputs" in prompt
    assert "price_quanto_option_analytical_from_market_state" in prompt
    assert "Do not reimplement spot / FX / curve / correlation lookup" in prompt
    assert "Do not reimplement the quanto-adjusted analytical pricing body" in prompt
    assert "quanto-adjusted forward" in prompt
    assert "trellis.models.analytical.support" in prompt
    assert "normalized_option_type" in prompt
    assert "discounted_value" in prompt


def test_evaluate_prompt_quanto_monte_carlo_surface_includes_joint_state_guidance():
    from trellis.agent.codegen_guardrails import GenerationPlan, PrimitivePlan, PrimitiveRef
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    plan = GenerationPlan(
        method="monte_carlo",
        instrument_type="quanto_option",
        inspected_modules=(
            "trellis.models.monte_carlo.engine",
            "trellis.models.monte_carlo.quanto",
            "trellis.models.quanto_option",
            "trellis.models.processes.correlated_gbm",
            "trellis.models.resolution.quanto",
        ),
        approved_modules=(
            "trellis.models.monte_carlo.engine",
            "trellis.models.monte_carlo.quanto",
            "trellis.models.quanto_option",
            "trellis.models.processes.correlated_gbm",
            "trellis.models.resolution.quanto",
        ),
        symbols_to_reuse=(
            "MonteCarloEngine",
            "CorrelatedGBM",
            "resolve_quanto_inputs",
            "price_quanto_option_monte_carlo_from_market_state",
        ),
        proposed_tests=("tests/test_agent/test_build_loop.py",),
        primitive_plan=PrimitivePlan(
            route="correlated_gbm_monte_carlo",
            engine_family="monte_carlo",
            primitives=(
                PrimitiveRef("trellis.models.processes.correlated_gbm", "CorrelatedGBM", "state_process"),
                PrimitiveRef("trellis.models.monte_carlo.engine", "MonteCarloEngine", "path_simulation"),
                PrimitiveRef(
                    "trellis.models.resolution.quanto",
                    "resolve_quanto_inputs",
                    "market_binding",
                ),
                PrimitiveRef(
                    "trellis.models.quanto_option",
                    "price_quanto_option_monte_carlo_from_market_state",
                    "route_helper",
                ),
            ),
            adapters=("reuse_shared_quanto_market_binding", "reuse_shared_quanto_mc_route_helper"),
            blockers=(),
        ),
    )

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={
            "Quanto helper": "def resolve_quanto_inputs(...):\n    pass\n",
            "Quanto MC helper": "def price_quanto_option_monte_carlo_from_market_state(...):\n    pass\n",
        },
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=[
                "trellis.models.monte_carlo.engine",
                "trellis.models.monte_carlo.quanto",
                "trellis.models.quanto_option",
                "trellis.models.processes.correlated_gbm",
                "trellis.models.resolution.quanto",
            ],
            required_market_data={"discount_curve", "forward_curve", "black_vol_surface", "fx_rates", "spot", "model_parameters"},
            model_to_build="quanto_option",
            reasoning="product_ir_compiler",
        ),
        knowledge_context="## Shared Knowledge\n- Quanto MC route",
        generation_plan=plan,
        prompt_surface="compact",
    )

    assert "price_quanto_option_monte_carlo_from_market_state" in prompt
    assert "np.array([resolved.spot, resolved.fx_spot]" in prompt
    assert "Do not seed a multi-asset correlated GBM" in prompt
    assert "Do not reimplement process / engine / payoff / discount wiring" in prompt


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


def test_evaluate_prompt_mentions_fxrate_scalar_extraction_rule():
    from trellis.agent.prompts import evaluate_prompt
    from trellis.agent.quant import PricingPlan

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.models.monte_carlo.engine"],
            required_market_data={"discount_curve", "forward_curve", "fx_rates", "spot"},
            model_to_build="european_option",
            reasoning="FX MC route.",
        ),
        knowledge_context="## Shared Knowledge\n- FX MC route",
        prompt_surface="compact",
    )

    assert "`market_state.fx_rates[pair]` returns an `FXRate` wrapper" in prompt


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
    assert "## Structured Lane Card" not in prompt


def test_prompt_warns_against_wall_clock_dates():
    from trellis.agent.prompts import evaluate_prompt

    prompt = evaluate_prompt(
        skeleton_code="class Demo:\n    def evaluate(self, market_state):\n        pass\n",
        spec_schema=SimpleNamespace(class_name="Demo", fields=[]),
        reference_sources={},
        knowledge_context="",
        prompt_surface="compact",
    )

    assert "Never use wall-clock dates such as `date.today()`" in prompt
