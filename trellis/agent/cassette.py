"""LLM cassette record/replay system for deterministic agent testing.

Records LLM interactions (prompts + responses) during live runs and replays
them deterministically in tests.  Zero tokens consumed during replay.

**Recording** — activated by ``TRELLIS_CASSETTE_RECORD=1``:

    recorder = CassetteRecorder(Path("cassettes/T38.yaml"))
    # monkey-patch llm_generate / llm_generate_json with recorder wrappers
    recorder.flush()  # writes YAML

**Replay** — activated by ``TRELLIS_CASSETTE_REPLAY=1``:

    replayer = CassetteReplayer(Path("cassettes/T38.yaml"))
    # monkey-patch with replayer.generate / replayer.generate_json
    replayer.assert_all_consumed()

Designed for QUA-423.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

_log = logging.getLogger(__name__)
_LLM_CASSETTE_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_cassette_context",
    default=None,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CassetteError(Exception):
    """Base class for cassette-related errors."""


class CassetteStaleError(CassetteError):
    """Raised when a replayed prompt's hash doesn't match the recorded one."""


class CassetteMissingError(CassetteError):
    """Raised when replay exhausts recorded calls (more LLM calls than recorded)."""


class CassetteNotFoundError(CassetteError):
    """Raised when a cassette file does not exist in replay mode."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CassetteCall:
    """One recorded LLM interaction."""

    seq: int
    function: str                   # "llm_generate" or "llm_generate_json"
    stage: str                      # from _LLM_USAGE_STAGE context
    prompt_hash: str                # sha256 hex digest of the prompt text
    response_text: str              # raw response from LLM
    prompt_text: str | None = None  # full prompt (optional, for debugging)
    model: str | None = None


@dataclass
class CassetteMeta:
    """Metadata header for a cassette file."""

    recorded_at: str
    provider: str | None = None
    model: str | None = None
    total_calls: int = 0
    name: str | None = None


def _prompt_hash(prompt: str) -> str:
    """SHA-256 hex digest of a prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _call_to_dict(call: CassetteCall) -> dict[str, Any]:
    d: dict[str, Any] = {
        "seq": call.seq,
        "function": call.function,
        "stage": call.stage,
        "prompt_hash": call.prompt_hash,
        "response_text": call.response_text,
    }
    if call.prompt_text is not None:
        d["prompt_text"] = call.prompt_text
    if call.model is not None:
        d["model"] = call.model
    return d


def _dict_to_call(d: dict[str, Any]) -> CassetteCall:
    return CassetteCall(
        seq=d["seq"],
        function=d["function"],
        stage=d.get("stage", "unscoped"),
        prompt_hash=d["prompt_hash"],
        response_text=d["response_text"],
        prompt_text=d.get("prompt_text"),
        model=d.get("model"),
    )


def _meta_to_dict(meta: CassetteMeta) -> dict[str, Any]:
    d: dict[str, Any] = {"recorded_at": meta.recorded_at}
    if meta.provider:
        d["provider"] = meta.provider
    if meta.model:
        d["model"] = meta.model
    d["total_calls"] = meta.total_calls
    if meta.name:
        d["name"] = meta.name
    return d


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class CassetteRecorder:
    """Records LLM calls during a live session and writes them to a YAML file.

    Usage::

        recorder = CassetteRecorder(path, store_prompts=True)
        wrapped_generate = recorder.wrap(real_llm_generate, "llm_generate")
        # ... use wrapped_generate in place of llm_generate ...
        recorder.flush()
    """

    def __init__(
        self,
        path: Path,
        *,
        store_prompts: bool = True,
        name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.store_prompts = store_prompts
        self.name = name
        self._calls: list[CassetteCall] = []
        self._provider: str | None = None
        self._model: str | None = None

    # -- wrapping helpers ---------------------------------------------------

    def wrap(
        self,
        real_fn: Callable[..., Any],
        function_name: str,
    ) -> Callable[..., Any]:
        """Return a wrapper that calls *real_fn* and records the interaction."""

        def wrapper(prompt: str, model: str | None = None, **kwargs: Any) -> Any:
            # Import here to avoid circular deps at module level
            from trellis.agent.config import _LLM_USAGE_STAGE

            stage = _LLM_USAGE_STAGE.get() or "unscoped"
            result = real_fn(prompt, model=model, **kwargs)
            # For llm_generate, result is str; for llm_generate_json, result is dict.
            if isinstance(result, dict):
                response_text = json.dumps(result, ensure_ascii=False)
            else:
                response_text = str(result)
            call = CassetteCall(
                seq=len(self._calls),
                function=function_name,
                stage=stage,
                prompt_hash=_prompt_hash(prompt),
                response_text=response_text,
                prompt_text=prompt if self.store_prompts else None,
                model=model,
            )
            self._calls.append(call)
            return result

        return wrapper

    def wrap_generate(self, real_fn: Callable[..., str]) -> Callable[..., str]:
        """Wrap ``llm_generate``."""
        return self.wrap(real_fn, "llm_generate")

    def wrap_generate_json(self, real_fn: Callable[..., dict]) -> Callable[..., dict]:
        """Wrap ``llm_generate_json``."""
        return self.wrap(real_fn, "llm_generate_json")

    # -- persistence --------------------------------------------------------

    def flush(self, *, provider: str | None = None, model: str | None = None) -> Path:
        """Write all recorded calls to *self.path* as YAML.  Returns the path."""
        self._provider = provider or self._provider
        self._model = model or self._model
        meta = CassetteMeta(
            recorded_at=datetime.now(timezone.utc).isoformat(),
            provider=self._provider,
            model=self._model,
            total_calls=len(self._calls),
            name=self.name,
        )
        data = {
            "meta": _meta_to_dict(meta),
            "calls": [_call_to_dict(c) for c in self._calls],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        _log.info("Cassette recorded: %s (%d calls)", self.path, len(self._calls))
        return self.path

    @property
    def calls(self) -> list[CassetteCall]:
        return list(self._calls)

    def __len__(self) -> int:
        return len(self._calls)


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------

class CassetteReplayer:
    """Replays recorded LLM responses in sequence.  Zero LLM calls made.

    Usage::

        replayer = CassetteReplayer(path)
        text = replayer.generate(prompt)       # returns recorded text
        data = replayer.generate_json(prompt)   # returns recorded dict
        replayer.assert_all_consumed()
    """

    # Staleness policy: "warn" (default) logs a warning; "error" raises.
    STALE_POLICY_WARN = "warn"
    STALE_POLICY_ERROR = "error"

    def __init__(
        self,
        path: Path,
        *,
        stale_policy: str = STALE_POLICY_WARN,
    ) -> None:
        self.path = Path(path)
        self.stale_policy = stale_policy
        if not self.path.exists():
            raise CassetteNotFoundError(f"Cassette file not found: {self.path}")
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self._meta = raw.get("meta", {})
        self._calls = [_dict_to_call(d) for d in raw.get("calls", [])]
        self._cursor = 0

    # -- replay interface ---------------------------------------------------

    def _next_call(self, prompt: str, expected_function: str) -> CassetteCall:
        """Advance cursor, validate, and return the next recorded call."""
        if self._cursor >= len(self._calls):
            raise CassetteMissingError(
                f"Cassette exhausted after {len(self._calls)} calls; "
                f"the code made an additional '{expected_function}' call "
                f"(prompt hash {_prompt_hash(prompt)[:12]}...)."
            )
        call = self._calls[self._cursor]
        self._cursor += 1

        # Function type mismatch is always an error — indicates structural change
        if call.function != expected_function:
            raise CassetteStaleError(
                f"Cassette call #{call.seq}: expected {expected_function}, "
                f"recorded {call.function}. The call sequence has changed."
            )

        # Prompt hash check for staleness
        current_hash = _prompt_hash(prompt)
        if current_hash != call.prompt_hash:
            msg = (
                f"Cassette call #{call.seq} ({call.stage}): prompt hash mismatch. "
                f"Recorded {call.prompt_hash[:12]}..., "
                f"current {current_hash[:12]}.... "
                f"The prompt template may have changed — consider re-recording."
            )
            if self.stale_policy == self.STALE_POLICY_ERROR:
                raise CassetteStaleError(msg)
            else:
                warnings.warn(msg, stacklevel=3)
                _log.warning(msg)

        return call

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        """Replay a ``llm_generate`` call.  Returns the recorded text response."""
        call = self._next_call(prompt, "llm_generate")
        return call.response_text

    def generate_json(self, prompt: str, model: str | None = None, **kwargs: Any) -> dict:
        """Replay a ``llm_generate_json`` call.  Returns the recorded dict."""
        call = self._next_call(prompt, "llm_generate_json")
        return json.loads(call.response_text)

    # -- assertions ---------------------------------------------------------

    def assert_all_consumed(self) -> None:
        """Raise if there are unconsumed recorded calls (fewer LLM calls than expected)."""
        remaining = len(self._calls) - self._cursor
        if remaining > 0:
            stages = [c.stage for c in self._calls[self._cursor:]]
            raise CassetteStaleError(
                f"Cassette has {remaining} unconsumed call(s) "
                f"(stages: {stages}). The code made fewer LLM calls than recorded."
            )

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    @property
    def consumed(self) -> int:
        return self._cursor

    @property
    def remaining(self) -> int:
        return len(self._calls) - self._cursor


# ---------------------------------------------------------------------------
# Convenience: load cassette from path
# ---------------------------------------------------------------------------

def load_cassette(path: str | Path) -> dict[str, Any]:
    """Load a cassette YAML file and return the raw dict."""
    p = Path(path)
    if not p.exists():
        raise CassetteNotFoundError(f"Cassette not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def current_llm_cassette_context() -> dict[str, Any] | None:
    """Return metadata for the active cassette scope, if any."""
    current = _LLM_CASSETTE_CONTEXT.get()
    if current is None:
        return None
    return dict(current)


@contextmanager
def llm_cassette_session(
    path: str | Path,
    *,
    mode: str,
    stale_policy: str = CassetteReplayer.STALE_POLICY_WARN,
    store_prompts: bool = True,
    name: str | None = None,
):
    """Run the current LLM call path against a recorder or replayer.

    This scopes cassette control through ``trellis.agent.config.llm_generate``
    and ``llm_generate_json`` so callers that imported those functions at
    module scope still respect the active cassette session.
    """
    from trellis.agent.config import (
        _llm_generate_json_live,
        _llm_generate_live,
        get_default_model,
        get_provider,
        llm_override_scope,
    )

    cassette_path = Path(path)
    cassette_name = name or cassette_path.stem
    if mode == "record":
        handler: CassetteRecorder | CassetteReplayer = CassetteRecorder(
            cassette_path,
            store_prompts=store_prompts,
            name=cassette_name,
        )
        generate = handler.wrap_generate(_llm_generate_live)
        generate_json = handler.wrap_generate_json(_llm_generate_json_live)
    elif mode == "replay":
        handler = CassetteReplayer(cassette_path, stale_policy=stale_policy)
        generate = handler.generate
        generate_json = handler.generate_json
    else:
        raise ValueError(f"Unsupported cassette mode: {mode!r}")

    metadata = {
        "mode": mode,
        "name": cassette_name,
        "path": str(cassette_path),
    }
    if mode == "replay":
        metadata["stale_policy"] = stale_policy

    token = _LLM_CASSETTE_CONTEXT.set(metadata)
    try:
        with llm_override_scope(
            generate=generate,
            generate_json=generate_json,
        ):
            yield handler
    finally:
        _LLM_CASSETTE_CONTEXT.reset(token)
        if mode == "record":
            assert isinstance(handler, CassetteRecorder)
            handler.flush(
                provider=get_provider(),
                model=get_default_model(),
            )
        else:
            assert isinstance(handler, CassetteReplayer)
            handler.assert_all_consumed()
