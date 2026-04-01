"""Tests for the LLM cassette record/replay system (QUA-423).

All tests are pure unit tests — no LLM calls, no tokens spent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from trellis.agent.cassette import (
    CassetteMissingError,
    CassetteNotFoundError,
    CassetteRecorder,
    CassetteReplayer,
    CassetteStaleError,
    _prompt_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm_generate(prompt: str, model: str | None = None) -> str:
    """Fake llm_generate that returns a deterministic response."""
    return f"response_for_{prompt[:20]}"


def _fake_llm_generate_json(prompt: str, model: str | None = None) -> dict:
    """Fake llm_generate_json that returns a deterministic response."""
    return {"method": "rate_tree", "prompt_len": len(prompt)}


def _make_cassette_yaml(calls: list[dict], meta: dict | None = None) -> str:
    """Build cassette YAML content from call specs."""
    if meta is None:
        meta = {
            "recorded_at": "2026-03-30T12:00:00+00:00",
            "total_calls": len(calls),
        }
    return yaml.dump({"meta": meta, "calls": calls}, default_flow_style=False)


# ---------------------------------------------------------------------------
# _prompt_hash
# ---------------------------------------------------------------------------

class TestPromptHash:
    def test_deterministic(self):
        assert _prompt_hash("hello") == _prompt_hash("hello")

    def test_different_prompts(self):
        assert _prompt_hash("hello") != _prompt_hash("world")

    def test_is_hex_sha256(self):
        h = _prompt_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# CassetteRecorder
# ---------------------------------------------------------------------------

class TestCassetteRecorder:
    def _set_stage(self, stage: str):
        """Set the LLM usage stage context var. Returns reset token."""
        from trellis.agent.config import _LLM_USAGE_STAGE
        return _LLM_USAGE_STAGE.set(stage)

    def test_record_single_call(self, tmp_path):
        token = self._set_stage("quant")
        try:
            path = tmp_path / "test.yaml"
            recorder = CassetteRecorder(path, name="test_cassette")

            wrapped = recorder.wrap_generate(_fake_llm_generate)
            result = wrapped("Price a callable bond", model="test-model")

            assert result == "response_for_Price a callable bon"
            assert len(recorder) == 1
            assert recorder.calls[0].function == "llm_generate"
            assert recorder.calls[0].stage == "quant"
            assert recorder.calls[0].model == "test-model"
        finally:
            from trellis.agent.config import _LLM_USAGE_STAGE
            _LLM_USAGE_STAGE.reset(token)

    def test_record_json_call(self, tmp_path):
        token = self._set_stage("planner")
        try:
            path = tmp_path / "test.yaml"
            recorder = CassetteRecorder(path)

            wrapped = recorder.wrap_generate_json(_fake_llm_generate_json)
            result = wrapped("Design spec for callable bond")

            assert result == {"method": "rate_tree", "prompt_len": 29}
            assert len(recorder) == 1
            assert recorder.calls[0].function == "llm_generate_json"
            parsed = json.loads(recorder.calls[0].response_text)
            assert parsed["method"] == "rate_tree"
        finally:
            from trellis.agent.config import _LLM_USAGE_STAGE
            _LLM_USAGE_STAGE.reset(token)

    def test_record_multiple_calls(self, tmp_path):
        from trellis.agent.config import _LLM_USAGE_STAGE

        path = tmp_path / "multi.yaml"
        recorder = CassetteRecorder(path)
        gen = recorder.wrap_generate(_fake_llm_generate)
        gen_json = recorder.wrap_generate_json(_fake_llm_generate_json)

        token = _LLM_USAGE_STAGE.set("quant")
        gen("prompt 1")
        _LLM_USAGE_STAGE.reset(token)

        token = _LLM_USAGE_STAGE.set("planner")
        gen_json("prompt 2")
        _LLM_USAGE_STAGE.reset(token)

        token = _LLM_USAGE_STAGE.set("builder")
        gen("prompt 3")
        _LLM_USAGE_STAGE.reset(token)

        assert len(recorder) == 3
        assert [c.stage for c in recorder.calls] == ["quant", "planner", "builder"]
        assert [c.function for c in recorder.calls] == [
            "llm_generate", "llm_generate_json", "llm_generate",
        ]

    def test_flush_writes_yaml(self, tmp_path):
        token = self._set_stage("quant")
        try:
            path = tmp_path / "flush.yaml"
            recorder = CassetteRecorder(path, name="flush_test")
            wrapped = recorder.wrap_generate(_fake_llm_generate)
            wrapped("test prompt")

            written_path = recorder.flush(provider="openai", model="gpt-5-mini")
            assert written_path == path
            assert path.exists()

            data = yaml.safe_load(path.read_text())
            assert data["meta"]["total_calls"] == 1
            assert data["meta"]["provider"] == "openai"
            assert data["meta"]["model"] == "gpt-5-mini"
            assert data["meta"]["name"] == "flush_test"
            assert len(data["calls"]) == 1
            assert data["calls"][0]["seq"] == 0
            assert data["calls"][0]["function"] == "llm_generate"
        finally:
            from trellis.agent.config import _LLM_USAGE_STAGE
            _LLM_USAGE_STAGE.reset(token)

    def test_flush_creates_parent_dirs(self, tmp_path):
        token = self._set_stage("quant")
        try:
            path = tmp_path / "nested" / "deep" / "cassette.yaml"
            recorder = CassetteRecorder(path)
            wrapped = recorder.wrap_generate(_fake_llm_generate)
            wrapped("test")
            recorder.flush()
            assert path.exists()
        finally:
            from trellis.agent.config import _LLM_USAGE_STAGE
            _LLM_USAGE_STAGE.reset(token)

    def test_store_prompts_flag(self, tmp_path):
        token = self._set_stage("quant")
        try:
            # With store_prompts=True (default)
            path_with = tmp_path / "with_prompts.yaml"
            rec_with = CassetteRecorder(path_with, store_prompts=True)
            rec_with.wrap_generate(_fake_llm_generate)("my secret prompt")
            assert rec_with.calls[0].prompt_text == "my secret prompt"

            # With store_prompts=False
            path_without = tmp_path / "without_prompts.yaml"
            rec_without = CassetteRecorder(path_without, store_prompts=False)
            rec_without.wrap_generate(_fake_llm_generate)("my secret prompt")
            assert rec_without.calls[0].prompt_text is None
        finally:
            from trellis.agent.config import _LLM_USAGE_STAGE
            _LLM_USAGE_STAGE.reset(token)


# ---------------------------------------------------------------------------
# CassetteReplayer
# ---------------------------------------------------------------------------

class TestCassetteReplayer:
    def _write_cassette(self, tmp_path, calls, meta=None) -> Path:
        path = tmp_path / "replay.yaml"
        path.write_text(_make_cassette_yaml(calls, meta))
        return path

    def test_replay_text(self, tmp_path):
        prompt = "Price a callable bond"
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash(prompt),
                "response_text": "Use rate_tree method",
            },
        ])
        replayer = CassetteReplayer(path)
        result = replayer.generate(prompt)
        assert result == "Use rate_tree method"
        assert replayer.consumed == 1
        assert replayer.remaining == 0

    def test_replay_json(self, tmp_path):
        prompt = "Design spec"
        response = {"method": "rate_tree", "needs": ["vol_surface"]}
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate_json",
                "stage": "planner",
                "prompt_hash": _prompt_hash(prompt),
                "response_text": json.dumps(response),
            },
        ])
        replayer = CassetteReplayer(path)
        result = replayer.generate_json(prompt)
        assert result == response

    def test_replay_sequence(self, tmp_path):
        p1, p2 = "prompt one", "prompt two"
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash(p1),
                "response_text": "response one",
            },
            {
                "seq": 1,
                "function": "llm_generate_json",
                "stage": "planner",
                "prompt_hash": _prompt_hash(p2),
                "response_text": '{"key": "value"}',
            },
        ])
        replayer = CassetteReplayer(path)
        assert replayer.total_calls == 2

        r1 = replayer.generate(p1)
        assert r1 == "response one"
        assert replayer.consumed == 1

        r2 = replayer.generate_json(p2)
        assert r2 == {"key": "value"}
        assert replayer.consumed == 2

    def test_exhausted_raises_missing(self, tmp_path):
        prompt = "only one call"
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash(prompt),
                "response_text": "ok",
            },
        ])
        replayer = CassetteReplayer(path)
        replayer.generate(prompt)  # consume the only call

        with pytest.raises(CassetteMissingError, match="exhausted"):
            replayer.generate("another prompt")

    def test_function_mismatch_raises_stale(self, tmp_path):
        prompt = "test"
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash(prompt),
                "response_text": "ok",
            },
        ])
        replayer = CassetteReplayer(path)
        with pytest.raises(CassetteStaleError, match="expected llm_generate_json"):
            replayer.generate_json(prompt)

    def test_prompt_hash_mismatch_warns(self, tmp_path):
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": "deadbeef" * 8,  # wrong hash
                "response_text": "ok",
            },
        ])
        replayer = CassetteReplayer(path, stale_policy="warn")
        with pytest.warns(match="prompt hash mismatch"):
            result = replayer.generate("different prompt")
        assert result == "ok"  # still returns the recorded response

    def test_prompt_hash_mismatch_error_policy(self, tmp_path):
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": "deadbeef" * 8,
                "response_text": "ok",
            },
        ])
        replayer = CassetteReplayer(path, stale_policy="error")
        with pytest.raises(CassetteStaleError, match="prompt hash mismatch"):
            replayer.generate("different prompt")

    def test_assert_all_consumed_passes_when_empty(self, tmp_path):
        prompt = "test"
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash(prompt),
                "response_text": "ok",
            },
        ])
        replayer = CassetteReplayer(path)
        replayer.generate(prompt)
        replayer.assert_all_consumed()  # should not raise

    def test_assert_all_consumed_fails_with_remaining(self, tmp_path):
        path = self._write_cassette(tmp_path, [
            {
                "seq": 0,
                "function": "llm_generate",
                "stage": "quant",
                "prompt_hash": _prompt_hash("p1"),
                "response_text": "ok",
            },
            {
                "seq": 1,
                "function": "llm_generate",
                "stage": "planner",
                "prompt_hash": _prompt_hash("p2"),
                "response_text": "ok2",
            },
        ])
        replayer = CassetteReplayer(path)
        replayer.generate("p1")
        with pytest.raises(CassetteStaleError, match="1 unconsumed"):
            replayer.assert_all_consumed()

    def test_not_found_raises(self, tmp_path):
        with pytest.raises(CassetteNotFoundError, match="not found"):
            CassetteReplayer(tmp_path / "nonexistent.yaml")

    def test_meta_accessible(self, tmp_path):
        path = self._write_cassette(
            tmp_path, [],
            meta={"recorded_at": "2026-03-30T12:00:00+00:00", "provider": "openai", "total_calls": 0},
        )
        replayer = CassetteReplayer(path)
        assert replayer.meta["provider"] == "openai"
        assert replayer.total_calls == 0


# ---------------------------------------------------------------------------
# Round-trip: record → replay
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_record_then_replay(self, tmp_path):
        """Record a session, then replay it and get identical results."""
        from trellis.agent.config import _LLM_USAGE_STAGE

        path = tmp_path / "roundtrip.yaml"

        # --- Record phase ---
        recorder = CassetteRecorder(path, name="roundtrip")
        gen = recorder.wrap_generate(_fake_llm_generate)
        gen_json = recorder.wrap_generate_json(_fake_llm_generate_json)

        token = _LLM_USAGE_STAGE.set("quant")
        text_result = gen("select method for callable bond")
        _LLM_USAGE_STAGE.reset(token)

        token = _LLM_USAGE_STAGE.set("planner")
        json_result = gen_json("design spec for callable bond")
        _LLM_USAGE_STAGE.reset(token)

        recorder.flush(provider="openai", model="gpt-5-mini")

        # --- Replay phase ---
        replayer = CassetteReplayer(path)
        assert replayer.total_calls == 2

        replayed_text = replayer.generate("select method for callable bond")
        assert replayed_text == text_result

        replayed_json = replayer.generate_json("design spec for callable bond")
        assert replayed_json == json_result

        replayer.assert_all_consumed()

    def test_record_then_replay_with_changed_prompt_warns(self, tmp_path):
        """If prompts change between record and replay, warn but still return data."""
        from trellis.agent.config import _LLM_USAGE_STAGE

        path = tmp_path / "stale.yaml"

        # Record with one prompt
        token = _LLM_USAGE_STAGE.set("quant")
        recorder = CassetteRecorder(path)
        gen = recorder.wrap_generate(_fake_llm_generate)
        gen("original prompt text")
        recorder.flush()
        _LLM_USAGE_STAGE.reset(token)

        # Replay with a different prompt
        replayer = CassetteReplayer(path, stale_policy="warn")
        with pytest.warns(match="prompt hash mismatch"):
            result = replayer.generate("modified prompt text")
        # Should still return the recorded response
        assert result == "response_for_original prompt text"
