from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace


def test_generated_module_code_text_handles_structured_results():
    from trellis.agent.knowledge.autonomous import _generated_module_code_text

    assert _generated_module_code_text("print('ok')") == "print('ok')"
    assert _generated_module_code_text(SimpleNamespace(code="print('ok')")) == "print('ok')"


def test_build_with_knowledge_respects_preferred_method(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    observed: dict[str, object] = {}

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    def fake_build_with_tracking(**kwargs):
        observed["decomposition_method"] = kwargs["decomposition"].method
        observed["instrument_type"] = kwargs["instrument_type"]
        return type("FakePayoff", (), {}), {"attempts": 1, "failures": [], "code": "pass"}

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        fake_build_with_tracking,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {},
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
        preferred_method="monte_carlo",
    )

    assert observed["decomposition_method"] == "monte_carlo"
    assert observed["instrument_type"] == "european_option"
    assert result.success is True


def test_build_with_knowledge_preserves_platform_trace_metadata(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        lambda **kwargs: (
            type("FakePayoff", (), {}),
            {
                "attempts": 1,
                "failures": [],
                "code": "pass",
                "knowledge_summary": {"lesson_ids": ["mc_007"]},
                "platform_trace_path": "/tmp/platform_trace.yaml",
                "platform_request_id": "executor_build_20260325_deadbeef",
                "analytical_trace_path": "/tmp/analytical_trace.json",
                "analytical_trace_text_path": "/tmp/analytical_trace.md",
            },
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {},
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
    )

    assert result.success is True
    assert result.knowledge_summary == {"lesson_ids": ["mc_007"]}
    assert result.platform_trace_path == "/tmp/platform_trace.yaml"
    assert result.platform_request_id == "executor_build_20260325_deadbeef"
    assert result.analytical_trace_path == "/tmp/analytical_trace.json"
    assert result.analytical_trace_text_path == "/tmp/analytical_trace.md"


def test_build_with_knowledge_forwards_request_metadata(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    observed: dict[str, object] = {}
    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )

    def fake_build_with_tracking(**kwargs):
        observed["request_metadata"] = kwargs["request_metadata"]
        return type("FakePayoff", (), {}), {"attempts": 1, "failures": [], "code": "pass"}

    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        fake_build_with_tracking,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {},
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
        request_metadata={"task_id": "E23", "task_title": "European equity call under local vol: PDE vs MC"},
    )

    assert result.success is True
    assert observed["request_metadata"] == {
        "task_id": "E23",
        "task_title": "European equity call under local vol: PDE vs MC",
    }


def test_build_with_knowledge_resets_deterministic_planning_caches(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    decomposition = ProductDecomposition(
        instrument="basket_option",
        features=("vanilla",),
        method="fft_pricing",
        learned=False,
    )
    observed: dict[str, object] = {"cache_reset_calls": 0}

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="basket_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._reset_deterministic_planning_caches",
        lambda: observed.__setitem__(
            "cache_reset_calls",
            int(observed["cache_reset_calls"]) + 1,
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {},
    )

    def fake_build_payoff(*args, **kwargs):
        return type("FakeBasketPayoff", (), {})

    monkeypatch.setattr("trellis.agent.executor.build_payoff", fake_build_payoff)
    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieve_for_product_ir",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.build_shared_knowledge_payload",
        lambda knowledge: {
            "summary": {},
            "builder_text_distilled": "",
            "builder_text": "",
            "builder_text_expanded": "",
            "review_text_distilled": "",
            "review_text_expanded": "",
        },
    )

    result = build_with_knowledge(
        "Spread option (Kirk approximation) vs 2D MC vs 2D FFT",
        instrument_type="basket_option",
        preferred_method="fft_pricing",
        comparison_target="fft_spread_2d",
    )

    assert result.success is True
    assert observed["cache_reset_calls"] == 1


def test_build_with_tracking_forwards_retry_overlays_to_build_payoff(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import _build_with_tracking

    observed: dict[str, object] = {}
    overlay = {
        "candidate_id": "retry-autocall-001",
        "target_id": "mc_autocall",
        "patch_type": "cookbook_patch",
        "retryable": True,
        "repair_obligations": [
            {
                "kind": "required_primitive",
                "primitive": "trellis.models.autocallable.price_autocallable_monte_carlo",
                "module": "trellis.models.autocallable",
                "symbol": "price_autocallable_monte_carlo",
                "available": True,
            }
        ],
    }
    decomposition = ProductDecomposition(
        instrument="autocallable",
        features=("path_dependent",),
        method="monte_carlo",
        learned=False,
    )

    def fake_build_payoff(*args, **kwargs):
        observed["knowledge_overlays"] = kwargs.get("knowledge_overlays")
        observed["request_metadata"] = kwargs.get("request_metadata")
        return type("FakeAutocallPayoff", (), {})

    monkeypatch.setattr("trellis.agent.executor.build_payoff", fake_build_payoff)
    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieve_for_product_ir",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.build_shared_knowledge_payload",
        lambda knowledge: {
            "summary": {},
            "builder_text_distilled": "",
            "builder_text": "",
            "builder_text_expanded": "",
            "review_text_distilled": "",
            "review_text_expanded": "",
        },
    )

    payoff_cls, meta = _build_with_tracking(
        description="Autocallable note",
        build_description="Autocallable note",
        instrument_type="autocallable",
        decomposition=decomposition,
        product_ir=SimpleNamespace(instrument="autocallable"),
        gap_report=GapReport(confidence=0.8, retrieved_lesson_ids=[]),
        model="test-model",
        market_state=None,
        max_retries=1,
        validation="standard",
        force_rebuild=True,
        fresh_build=False,
        preferred_method="monte_carlo",
        request_metadata={"task_id": "T107"},
        knowledge_overlays=[overlay],
    )

    assert payoff_cls.__name__ == "FakeAutocallPayoff"
    assert observed["knowledge_overlays"] == [overlay]
    assert observed["request_metadata"] == {"task_id": "T107"}
    assert meta["intra_run_learning"]["overlay_candidate_ids"] == [
        "retry-autocall-001"
    ]


def test_build_with_tracking_scopes_live_knowledge_to_compiled_route(monkeypatch):
    from trellis.agent.knowledge.autonomous import _build_with_tracking
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition

    observed: list[dict[str, object]] = []
    decomposition = ProductDecomposition(
        instrument="cliquet_option",
        features=("scheduled_observation_returns",),
        method="analytical",
        learned=False,
    )
    product_ir = SimpleNamespace(
        instrument="cliquet_option",
        route_families=("analytical",),
    )

    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._reset_deterministic_planning_caches",
        lambda: None,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.retrieve_for_product_ir",
        lambda *args, **kwargs: [],
    )

    def fake_shared_payload(knowledge, **kwargs):
        observed.append(dict(kwargs))
        route_ids = tuple(kwargs.get("route_ids") or ())
        scoped = route_ids == ("analytical_black76",)
        text = "scoped primitive knowledge" if scoped else "unscoped SABR helper"
        selected_ids = ["principle:P10"] if scoped else [
            "route_hint:sabr_hagan_analytical:route-helper"
        ]
        return {
            "summary": {"selected_artifact_ids": selected_ids},
            "builder_text_distilled": text,
            "builder_text": text,
            "builder_text_expanded": text,
            "review_text_distilled": text,
            "review_text_expanded": text,
        }

    monkeypatch.setattr(
        "trellis.agent.knowledge.build_shared_knowledge_payload",
        fake_shared_payload,
    )

    def fake_build_payoff(*args, **kwargs):
        generation_plan = SimpleNamespace(
            lowering_route_id=None,
            primitive_plan=SimpleNamespace(route="analytical_black76"),
        )
        request = SimpleNamespace(
            audience="builder",
            knowledge_surface="compact",
            prompt_surface="compact",
            pricing_method="analytical",
            product_ir=product_ir,
            compiled_request=SimpleNamespace(generation_plan=generation_plan),
        )
        assert kwargs["knowledge_retriever"](request) == "scoped primitive knowledge"
        return type("FakeCliquetPayoff", (), {})

    monkeypatch.setattr("trellis.agent.executor.build_payoff", fake_build_payoff)

    payoff_cls, meta = _build_with_tracking(
        description="Cliquet option",
        instrument_type="cliquet_option",
        decomposition=decomposition,
        product_ir=product_ir,
        gap_report=GapReport(confidence=0.8, retrieved_lesson_ids=[]),
        model="test-model",
        market_state=None,
        max_retries=1,
        validation="standard",
        force_rebuild=True,
        preferred_method="analytical",
    )

    assert payoff_cls.__name__ == "FakeCliquetPayoff"
    assert observed[-1] == {
        "pricing_method": "analytical",
        "route_ids": ("analytical_black76",),
        "route_families": ("analytical",),
    }
    assert meta["knowledge_summary"]["selected_artifact_ids"] == ["principle:P10"]


def test_build_with_knowledge_preserves_platform_trace_metadata_on_failure(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import BuildTrackingFailure, build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )

    def fake_build_with_tracking(**kwargs):
        raise BuildTrackingFailure(
            "blocked",
            meta={
                "attempts": 1,
                "failures": ["blocked"],
                "code": "",
                "agent_observations": [],
                "knowledge_summary": {"lesson_ids": ["mc_007"]},
                "platform_trace_path": "/tmp/blocked_trace.yaml",
                "platform_request_id": "executor_build_20260325_blocked",
                "analytical_trace_path": "/tmp/blocked_analytical_trace.json",
                "analytical_trace_text_path": "/tmp/blocked_analytical_trace.md",
                "blocker_details": {"blocker_codes": ["missing_symbol:demo"]},
            },
            cause=RuntimeError("blocked for missing primitive"),
        )

    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        fake_build_with_tracking,
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {},
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
    )

    assert result.success is False
    assert result.failures == ["blocked for missing primitive"]
    assert result.knowledge_summary == {"lesson_ids": ["mc_007"]}
    assert result.platform_trace_path == "/tmp/blocked_trace.yaml"
    assert result.platform_request_id == "executor_build_20260325_blocked"
    assert result.analytical_trace_path == "/tmp/blocked_analytical_trace.json"
    assert result.analytical_trace_text_path == "/tmp/blocked_analytical_trace.md"
    assert result.blocker_details == {"blocker_codes": ["missing_symbol:demo"]}


def test_build_with_knowledge_skips_reflection_on_provider_failure_before_build(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import BuildTrackingFailure, build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )

    def fake_build_with_tracking(**kwargs):
        raise BuildTrackingFailure(
            "provider failed",
            meta={
                "attempts": 0,
                "failures": [
                    "LLM provider 'anthropic' model 'claude-sonnet-4-6' returned invalid JSON response: Expecting value at line 1 column 1"
                ],
                "code": "",
                "agent_observations": [],
                "knowledge_summary": {},
                "platform_trace_path": "/tmp/provider_trace.yaml",
                "platform_request_id": "executor_build_provider",
            },
            cause=RuntimeError(
                "LLM provider 'anthropic' model 'claude-sonnet-4-6' returned invalid JSON response: Expecting value at line 1 column 1"
            ),
        )

    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        fake_build_with_tracking,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("reflection should have been skipped")

    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        fail_if_called,
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
    )

    assert result.success is False
    assert result.reflection == {
        "skipped": True,
        "reason": "provider_failure_before_build",
    }


def test_build_with_knowledge_skips_reflection_on_provider_failure_without_code(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import BuildTrackingFailure, build_with_knowledge
    decompose_mod = import_module("trellis.agent.knowledge.decompose")

    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.8, retrieved_lesson_ids=[]),
    )

    def fake_build_with_tracking(**kwargs):
        raise BuildTrackingFailure(
            "provider failed during code generation",
            meta={
                "attempts": 1,
                "failures": [
                    "OpenAI text request failed after 1 attempts for model 'gpt-5': TimeoutError: OpenAI request exceeded 12.0s"
                ],
                "code": "",
                "agent_observations": [],
                "knowledge_summary": {},
                "platform_trace_path": "/tmp/provider_trace.yaml",
                "platform_request_id": "executor_build_provider_codegen",
            },
            cause=RuntimeError(
                "OpenAI text request failed after 1 attempts for model 'gpt-5': TimeoutError: OpenAI request exceeded 12.0s"
            ),
        )

    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        fake_build_with_tracking,
    )

    def fail_if_called(**kwargs):
        raise AssertionError("reflection should have been skipped")

    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        fail_if_called,
    )

    result = build_with_knowledge(
        "FX vanilla option",
        instrument_type="european_option",
    )

    assert result.success is False
    assert result.reflection == {
        "skipped": True,
        "reason": "provider_failure_before_build",
    }


def test_build_with_knowledge_records_post_build_phase_markers(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge

    decompose_mod = import_module("trellis.agent.knowledge.decompose")
    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.9, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        lambda **kwargs: (
            type("FakePayoff", (), {}),
            {
                "attempts": 1,
                "failures": [],
                "code": "pass",
                "platform_trace_path": "/tmp/platform_trace.yaml",
                "platform_request_id": "executor_build_demo",
            },
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.reflect.reflect_on_build",
        lambda **kwargs: {"lessons_attributed": 1},
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._emit_decision_checkpoint",
        lambda **kwargs: {"status": "ok", "path": "/tmp/checkpoint.yaml"},
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._maybe_consolidate",
        lambda *args, **kwargs: None,
    )

    result = build_with_knowledge("European option", instrument_type="european_option")

    tracking = result.post_build_tracking
    assert tracking["last_phase"] == "consolidation_dispatched"
    assert tracking["last_status"] == "backgrounded"
    phases = [event["phase"] for event in tracking["events"]]
    assert phases == [
        "build_completed",
        "reflection_started",
        "reflection_completed",
        "token_usage_attached",
        "decision_checkpoint_emitted",
        "consolidation_dispatched",
    ]


def test_build_with_knowledge_can_skip_reflection_via_env(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge

    decompose_mod = import_module("trellis.agent.knowledge.decompose")
    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )

    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.9, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        lambda **kwargs: (
            type("FakePayoff", (), {}),
            {
                "attempts": 1,
                "failures": [],
                "code": "pass",
                "platform_trace_path": "/tmp/platform_trace.yaml",
                "platform_request_id": "executor_build_demo",
            },
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._emit_decision_checkpoint",
        lambda **kwargs: {"status": "ok", "path": "/tmp/checkpoint.yaml"},
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._maybe_consolidate",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "1")

    def fail_if_called(**kwargs):
        raise AssertionError("reflection should have been skipped")

    monkeypatch.setattr("trellis.agent.knowledge.reflect.reflect_on_build", fail_if_called)

    result = build_with_knowledge("European option", instrument_type="european_option")

    assert result.reflection == {
        "skipped": True,
        "reason": "TRELLIS_SKIP_POST_BUILD_REFLECTION",
    }
    assert result.post_build_tracking["active_flags"]["skip_reflection"] is True
    phases = {
        event["phase"]: event["status"]
        for event in result.post_build_tracking["events"]
    }
    assert phases["reflection_started"] == "skipped"
    assert phases["reflection_completed"] == "skipped"


def test_build_with_knowledge_skips_post_build_stages_from_task_policy(monkeypatch):
    from trellis.agent.knowledge.gap_check import GapReport
    from trellis.agent.knowledge.schema import ProductDecomposition
    from trellis.agent.knowledge.autonomous import build_with_knowledge
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    decompose_mod = import_module("trellis.agent.knowledge.decompose")
    decomposition = ProductDecomposition(
        instrument="european_option",
        features=("vanilla",),
        method="analytical",
        learned=False,
    )
    policy = determine_post_build_learning_policy(
        execution_mode="deterministic_replay",
        artifact_policy="cached_existing",
        recovery_mode="strict",
    )

    monkeypatch.delenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", raising=False)
    monkeypatch.delenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", raising=False)
    monkeypatch.setattr(decompose_mod, "decompose", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(
        decompose_mod,
        "decompose_to_ir",
        lambda *args, **kwargs: SimpleNamespace(instrument="european_option"),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.gap_check.gap_check",
        lambda decomposition: GapReport(confidence=0.9, retrieved_lesson_ids=[]),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._build_with_tracking",
        lambda **kwargs: (
            type("FakePayoff", (), {}),
            {
                "attempts": 1,
                "failures": [],
                "code": "pass",
                "platform_trace_path": None,
                "platform_request_id": "executor_build_cached",
            },
        ),
    )
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._emit_decision_checkpoint",
        lambda **kwargs: {"status": "ok", "path": "/tmp/checkpoint.yaml"},
    )

    def fail_reflection(**kwargs):
        raise AssertionError("policy-skipped reflection must not run")

    def fail_consolidation(*args, **kwargs):
        raise AssertionError("policy-skipped consolidation must not run")

    monkeypatch.setattr("trellis.agent.knowledge.reflect.reflect_on_build", fail_reflection)
    monkeypatch.setattr(
        "trellis.agent.knowledge.autonomous._maybe_consolidate",
        fail_consolidation,
    )

    result = build_with_knowledge(
        "European option",
        instrument_type="european_option",
        request_metadata={"post_build_learning_policy": policy.to_payload()},
    )

    assert result.success is True
    assert result.token_usage_summary["call_count"] == 0
    assert result.token_usage_summary["total_tokens"] == 0
    assert result.reflection == {
        "skipped": True,
        "reason": "execution_mode_deterministic_replay",
        "policy_reasons": [
            "execution_mode_deterministic_replay",
            "recovery_mode_strict",
        ],
    }
    tracking = result.post_build_tracking
    assert tracking["policy"] == policy.to_payload()
    assert tracking["active_flags"] == {
        "skip_reflection": True,
        "skip_consolidation": True,
        "reflection_policy_allowed": False,
        "consolidation_policy_allowed": False,
        "reflection_env_override": False,
        "consolidation_env_override": False,
    }
    events = {event["phase"]: event for event in tracking["events"]}
    assert events["reflection_started"]["status"] == "skipped"
    assert events["reflection_started"]["details"]["reason"] == (
        "execution_mode_deterministic_replay"
    )
    assert events["reflection_started"]["details"]["policy_reasons"] == [
        "execution_mode_deterministic_replay",
        "recovery_mode_strict",
    ]
    assert events["consolidation_dispatched"]["status"] == "skipped"
    assert events["consolidation_dispatched"]["details"]["reason"] == (
        "execution_mode_deterministic_replay"
    )


def test_environment_skip_remains_distinct_from_policy_skip(monkeypatch):
    from trellis.agent.knowledge.autonomous import _post_build_control_flags
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    policy = determine_post_build_learning_policy(
        execution_mode="live",
        artifact_policy="forced_rebuild",
        recovery_mode="assisted",
    )
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "1")
    monkeypatch.delenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", raising=False)

    controls = _post_build_control_flags(policy)

    assert controls == {
        "skip_reflection": True,
        "skip_consolidation": False,
        "reflection_policy_allowed": True,
        "consolidation_policy_allowed": True,
        "reflection_env_override": True,
        "consolidation_env_override": False,
    }


def test_policy_denial_remains_primary_when_environment_skip_is_also_set(monkeypatch):
    from trellis.agent.knowledge.autonomous import (
        _post_build_control_flags,
        _post_build_skip_reason,
    )
    from trellis.agent.post_build_policy import determine_post_build_learning_policy

    policy = determine_post_build_learning_policy(
        execution_mode="deterministic_replay",
        artifact_policy="cached_existing",
        recovery_mode="strict",
    )
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "1")
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", "1")
    controls = _post_build_control_flags(policy)

    assert _post_build_skip_reason(
        "reflection",
        policy=policy,
        controls=controls,
    ) == (
        "execution_mode_deterministic_replay",
        ["execution_mode_deterministic_replay", "recovery_mode_strict"],
    )
    assert _post_build_skip_reason(
        "consolidation",
        policy=policy,
        controls=controls,
    ) == (
        "execution_mode_deterministic_replay",
        ["execution_mode_deterministic_replay", "recovery_mode_strict"],
    )
    assert controls["reflection_env_override"] is True
    assert controls["consolidation_env_override"] is True
