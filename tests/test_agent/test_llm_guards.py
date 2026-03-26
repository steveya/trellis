from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import pytest


def test_validate_llm_text_response_rejects_blank():
    from trellis.agent.config import _validate_llm_text_response

    with pytest.raises(RuntimeError, match="empty text response"):
        _validate_llm_text_response("   ", provider="openai", model="gpt-5-mini")


def test_parse_llm_json_response_rejects_blank_and_invalid_json():
    from trellis.agent.config import _parse_llm_json_response

    with pytest.raises(RuntimeError, match="empty JSON response"):
        _parse_llm_json_response("", provider="anthropic", model="claude-sonnet-4-6")

    with pytest.raises(RuntimeError, match="invalid JSON response"):
        _parse_llm_json_response("not json", provider="anthropic", model="claude-sonnet-4-6")


def test_parse_llm_json_response_extracts_fenced_json():
    from trellis.agent.config import _parse_llm_json_response

    data = _parse_llm_json_response(
        "```json\n{\"ok\": true, \"n\": 2}\n```",
        provider="anthropic",
        model="claude-sonnet-4-6",
    )

    assert data == {"ok": True, "n": 2}


def test_parse_llm_json_response_extracts_json_from_prose():
    from trellis.agent.config import _parse_llm_json_response

    data = _parse_llm_json_response(
        "Here is the JSON you requested: {\"ok\": true, \"kind\": \"spec\"}",
        provider="anthropic",
        model="claude-sonnet-4-6",
    )

    assert data == {"ok": True, "kind": "spec"}


def test_generate_module_retries_after_empty_response(monkeypatch):
    from trellis.agent.executor import _generate_module
    from trellis.agent.planner import FieldDef, SpecSchema

    calls = {"count": 0}

    def fake_llm_generate(prompt: str, model: str | None = None) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "   "
        return "def ok():\n    return 1\n"

    monkeypatch.setattr("trellis.agent.config.llm_generate", fake_llm_generate)

    code = _generate_module(
        skeleton="",
        spec_schema=SpecSchema(
            class_name="Foo",
            spec_name="FooSpec",
            requirements=[],
            fields=[FieldDef(name="spot", type="float", description="", default=None)],
        ),
        reference_sources={},
        model="gpt-5-mini",
        max_retries=2,
    )

    assert code == "def ok():\n    return 1\n"
    assert calls["count"] == 2


def test_openai_request_with_retry_retries_then_succeeds(monkeypatch):
    from trellis.agent.config import _openai_request_with_retry

    attempts = {"count": 0}
    monkeypatch.setattr("trellis.agent.config.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "trellis.agent.config._run_with_wall_clock_timeout",
        lambda request_fn, timeout_seconds: request_fn(),
    )

    def flaky_request():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("timed out")
        return {"ok": True}

    result = _openai_request_with_retry(
        flaky_request,
        model="gpt-5-mini",
        response_kind="json",
        timeout_seconds=5.0,
    )

    assert result == {"ok": True}
    assert attempts["count"] == 2


def test_openai_request_with_retry_raises_after_bounded_retries(monkeypatch):
    from trellis.agent.config import _openai_request_with_retry

    monkeypatch.setattr("trellis.agent.config.time.sleep", lambda _: None)
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setattr(
        "trellis.agent.config._run_with_wall_clock_timeout",
        lambda request_fn, timeout_seconds: request_fn(),
    )

    attempts = {"count": 0}

    def always_fail():
        attempts["count"] += 1
        raise TimeoutError("still timed out")

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        _openai_request_with_retry(
            always_fail,
            model="gpt-5-mini",
            response_kind="json",
            timeout_seconds=5.0,
        )

    assert attempts["count"] == 2


def test_openai_generate_json_uses_timeout_and_retries(monkeypatch):
    from trellis.agent.config import _openai_generate_json

    monkeypatch.setattr("trellis.agent.config.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "trellis.agent.config._run_with_wall_clock_timeout",
        lambda request_fn, timeout_seconds: request_fn(),
    )
    monkeypatch.setenv("OPENAI_JSON_TIMEOUT_SECONDS", "7")

    calls: list[float] = []

    class FakeMessage:
        content = '{"ok": true}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_create(**kwargs):
        calls.append(kwargs["timeout_seconds"])
        if len(calls) == 1:
            raise TimeoutError("slow")
        return FakeResponse()

    monkeypatch.setattr(
        "trellis.agent.config._openai_chat_completion_create",
        fake_create,
    )

    text, usage = _openai_generate_json("return json", "gpt-5-mini")

    assert text == '{"ok": true}'
    assert usage["prompt_tokens"] is None
    assert calls == [7.0, 7.0]


def test_run_with_wall_clock_timeout_times_out_in_worker_thread():
    from trellis.agent.config import _run_with_wall_clock_timeout

    outcome: dict[str, object] = {}

    def _worker():
        try:
            _run_with_wall_clock_timeout(lambda: time.sleep(0.2), 0.01)
        except Exception as exc:  # pragma: no cover - assertion happens after join
            outcome["exception"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("exception"), TimeoutError)


def test_openai_chat_completion_create_disables_sdk_retries(monkeypatch):
    from trellis.agent.config import _openai_chat_completion_create

    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: {"kwargs": kwargs},
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    response = _openai_chat_completion_create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=128,
        timeout_seconds=9.0,
    )

    assert captured == {"timeout": 9.0, "max_retries": 0}
    assert response["kwargs"]["model"] == "gpt-5-mini"


def test_llm_usage_session_tracks_stage_token_totals(monkeypatch):
    from trellis.agent.config import (
        llm_generate,
        llm_usage_session,
        llm_usage_stage,
        summarize_llm_usage,
    )

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "openai")
    monkeypatch.setattr(
        "trellis.agent.config._openai_generate",
        lambda prompt, model: (
            "ok",
            {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        ),
    )

    with llm_usage_session() as records:
        with llm_usage_stage("critic"):
            assert llm_generate("hello", model="gpt-5-mini") == "ok"

    summary = summarize_llm_usage(records)

    assert summary["call_count"] == 1
    assert summary["calls_with_usage"] == 1
    assert summary["calls_without_usage"] == 0
    assert summary["prompt_tokens"] == 11
    assert summary["completion_tokens"] == 5
    assert summary["total_tokens"] == 16
    assert summary["by_stage"]["critic"]["total_tokens"] == 16
    assert summary["by_provider"]["openai"]["total_tokens"] == 16


def test_llm_usage_summary_marks_missing_provider_usage(monkeypatch):
    from trellis.agent.config import (
        llm_generate_json,
        llm_usage_session,
        llm_usage_stage,
        summarize_llm_usage,
    )

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "anthropic")
    monkeypatch.setattr(
        "trellis.agent.config._anthropic_generate_json",
        lambda prompt, model: ("{}", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}),
    )

    with llm_usage_session() as records:
        with llm_usage_stage("decomposition"):
            assert llm_generate_json("{}", model="claude-sonnet-4-6") == {}

    summary = summarize_llm_usage(records)

    assert summary["call_count"] == 1
    assert summary["calls_with_usage"] == 0
    assert summary["calls_without_usage"] == 1
    assert summary["by_stage"]["decomposition"]["calls_without_usage"] == 1


def test_get_model_for_stage_uses_tiered_default_for_anthropic(monkeypatch):
    from trellis.agent.config import get_model_for_stage

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "anthropic")

    assert get_model_for_stage("decomposition", "claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert get_model_for_stage("code_generation", "claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_get_model_for_stage_uses_tiered_default_for_openai(monkeypatch):
    from trellis.agent.config import get_model_for_stage

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "openai")

    assert get_model_for_stage("decomposition", "gpt-5-mini") == "gpt-5-mini"
    assert get_model_for_stage("code_generation", "gpt-5-mini") == "gpt-5-mini"
    assert get_model_for_stage("model_validator", "gpt-5-mini") == "gpt-5"


def test_enforce_llm_token_budget_raises_after_stage(monkeypatch):
    from trellis.agent.config import (
        TokenBudgetExceeded,
        enforce_llm_token_budget,
        llm_generate,
        llm_usage_session,
        llm_usage_stage,
    )

    monkeypatch.setattr("trellis.agent.config.load_env", lambda: None)
    monkeypatch.setattr("trellis.agent.config.get_provider", lambda: "openai")
    monkeypatch.setattr(
        "trellis.agent.config._openai_generate",
        lambda prompt, model: (
            "ok",
            {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
        ),
    )

    with llm_usage_session():
        with llm_usage_stage("critic"):
            llm_generate("hello", model="gpt-5-mini")
        with pytest.raises(TokenBudgetExceeded, match="critic"):
            enforce_llm_token_budget(stage="critic", budget_tokens=100)
