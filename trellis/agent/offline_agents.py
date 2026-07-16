"""Local-agent execution guards for offline task reruns."""

from __future__ import annotations

from contextlib import contextmanager
import os


_OFFLINE_LLM_ERROR = (
    "offline_local_agents forbids live LLM {kind} calls; "
    "use deterministic helpers, local validation, or cassette replay"
)


@contextmanager
def offline_local_agent_llm_guard():
    """Fail fast if a local-agent rerun attempts a live LLM call."""
    from trellis.agent.config import llm_override_scope

    def _blocked_llm_text(*_args, **_kwargs):
        raise RuntimeError(_OFFLINE_LLM_ERROR.format(kind="text"))

    def _blocked_llm_json(*_args, **_kwargs):
        raise RuntimeError(_OFFLINE_LLM_ERROR.format(kind="JSON"))

    with llm_override_scope(
        generate=_blocked_llm_text,
        generate_json=_blocked_llm_json,
        source="offline_local_agents",
    ):
        yield


@contextmanager
def offline_local_agent_run_scope():
    """Run a task with local deterministic agents and no live-learning side effects."""
    skip_flags = {
        "TRELLIS_OFFLINE_LOCAL_AGENTS": "1",
    }
    previous = {name: os.environ.get(name) for name in skip_flags}
    try:
        os.environ.update(skip_flags)
        with offline_local_agent_llm_guard():
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = ["offline_local_agent_llm_guard", "offline_local_agent_run_scope"]
