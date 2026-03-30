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
