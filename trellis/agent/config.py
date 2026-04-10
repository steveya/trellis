"""Agent configuration: API keys, LLM provider, environment setup."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_LOADED = False

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-6",
}
STAGE_DEFAULT_MODEL = {
    "openai": {
        "decomposition": "gpt-5.4-mini",
        "spec_design": "gpt-5.4-mini",
        "code_generation": "gpt-5.4-mini",
        "critic": "gpt-5.4-mini",
        "model_validator": "gpt-5.4-mini",
        "reflection": "gpt-5.4-mini",
    },
    "anthropic": {
        "decomposition": "claude-sonnet-4-6",
        "spec_design": "claude-sonnet-4-6",
        "code_generation": "claude-sonnet-4-6",
        "critic": "claude-sonnet-4-6",
        "model_validator": "claude-sonnet-4-6",
        "reflection": "claude-sonnet-4-6",
    },
}

ALLOWED_FIELD_TYPES = frozenset({
    "float", "int", "str", "bool", "date", "str | None",
    "float | None", "int | None",
    "tuple[date, ...]", "tuple[date, ...] | None",
    "Frequency", "DayCountConvention",
})

OPENAI_TEXT_TIMEOUT_SECONDS = 45.0
OPENAI_JSON_TIMEOUT_SECONDS = 30.0
OPENAI_MAX_RETRIES = 2

_LLM_USAGE_SESSION: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "llm_usage_session",
    default=None,
)
_LLM_USAGE_STAGE: ContextVar[str | None] = ContextVar(
    "llm_usage_stage",
    default=None,
)
_LLM_USAGE_STAGE_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "llm_usage_stage_collector",
    default=None,
)
_LLM_USAGE_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_usage_metadata",
    default=None,
)
_LLM_REQUEST_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_request_context",
    default=None,
)
_LLM_OVERRIDE_HANDLERS: ContextVar["LLMOverrideHandlers | None"] = ContextVar(
    "llm_override_handlers",
    default=None,
)
LLM_WAIT_LOG_ROOT = Path(__file__).resolve().parents[2] / "task_runs" / "llm_waits"


@dataclass(frozen=True)
class LLMOverrideHandlers:
    """Temporary override functions for LLM text and JSON requests."""

    generate: Callable[..., str]
    generate_json: Callable[..., dict]


class TokenBudgetExceeded(RuntimeError):
    """Raised when cumulative LLM token usage in a tracked scope exceeds the configured limit.

    The agent wraps build stages in token-tracking scopes; this exception
    halts the pipeline early to prevent runaway API costs.
    """


def load_env():
    """Load .env file from the project root if it exists."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    env_path = _find_env_file()
    if env_path is None:
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


def _find_env_file() -> Path | None:
    """Search upward from the package tree for the nearest project ``.env`` file."""
    current = Path(__file__).parent.parent.parent
    for _ in range(5):
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def get_provider() -> str:
    """Return the active LLM provider after loading environment overrides."""
    load_env()
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)


def get_default_model() -> str:
    """Return the default model name for the currently selected provider."""
    return DEFAULT_MODEL.get(get_provider(), DEFAULT_MODEL["openai"])


def get_model_for_stage(stage: str, requested_model: str | None = None) -> str:
    """Resolve the preferred model for one logical LLM stage."""
    provider = get_provider()
    default_model = DEFAULT_MODEL.get(provider, DEFAULT_MODEL["openai"])
    requested_model = requested_model or default_model

    env_key = f"TRELLIS_MODEL_{stage.upper()}"
    provider_env_key = f"TRELLIS_{provider.upper()}_MODEL_{stage.upper()}"
    override = os.environ.get(provider_env_key) or os.environ.get(env_key)
    if override:
        return override

    if requested_model != default_model:
        return requested_model

    provider_defaults = STAGE_DEFAULT_MODEL.get(provider, {})
    return provider_defaults.get(stage, requested_model)


def get_task_token_budget() -> int | None:
    """Return the configured per-task token budget, if any."""
    return _int_env("TRELLIS_TASK_TOKEN_BUDGET")


def get_batch_token_budget() -> int | None:
    """Return the configured per-batch token budget, if any."""
    return _int_env("TRELLIS_BATCH_TOKEN_BUDGET")


def issue_tracker_sync_enabled() -> bool:
    """Return whether external request-issue sync should run."""
    load_env()
    raw = os.environ.get("TRELLIS_SYNC_REQUEST_ISSUES", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def current_llm_usage_records() -> list[dict[str, Any]]:
    """Return the current in-scope LLM usage record list, if any."""
    return list(_LLM_USAGE_SESSION.get() or [])


def current_llm_usage_summary() -> dict[str, Any]:
    """Summarize the current in-scope LLM usage records."""
    return summarize_llm_usage(_LLM_USAGE_SESSION.get() or [])


def enforce_llm_token_budget(
    *,
    stage: str,
    budget_tokens: int | None = None,
    scope: str = "task",
) -> None:
    """Raise when the active LLM usage scope exceeds the configured budget."""
    budget = budget_tokens if budget_tokens is not None else get_task_token_budget()
    if budget is None or budget <= 0:
        return
    summary = current_llm_usage_summary()
    total_tokens = int(summary.get("total_tokens", 0) or 0)
    if total_tokens <= budget:
        return
    raise TokenBudgetExceeded(
        f"LLM {scope} token budget exceeded after stage '{stage}': "
        f"{total_tokens} > {budget}"
    )


@contextmanager
def llm_usage_session():
    """Collect per-call LLM usage records within the current execution scope."""
    records: list[dict[str, Any]] = []
    token = _LLM_USAGE_SESSION.set(records)
    try:
        yield records
    finally:
        _LLM_USAGE_SESSION.reset(token)


@contextmanager
def llm_usage_stage(stage: str, *, metadata: dict[str, Any] | None = None):
    """Assign a logical stage label to LLM calls made inside the scope."""
    records: list[dict[str, Any]] = []
    stage_token = _LLM_USAGE_STAGE.set(stage)
    collector_token = _LLM_USAGE_STAGE_COLLECTOR.set(records)
    metadata_token = _LLM_USAGE_METADATA.set(dict(metadata or {}))
    try:
        yield records
    finally:
        _LLM_USAGE_STAGE.reset(stage_token)
        _LLM_USAGE_STAGE_COLLECTOR.reset(collector_token)
        _LLM_USAGE_METADATA.reset(metadata_token)


def summarize_llm_usage(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Aggregate raw per-call usage records into task/trace friendly totals."""
    summary: dict[str, Any] = {
        "call_count": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
        "by_provider": {},
    }
    if not records:
        return summary

    def _empty_counter() -> dict[str, int]:
        return {
            "call_count": 0,
            "calls_with_usage": 0,
            "calls_without_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    for record in records:
        stage = str(record.get("stage") or "unscoped")
        provider = str(record.get("provider") or "unknown")
        stage_bucket = summary["by_stage"].setdefault(stage, _empty_counter())
        provider_bucket = summary["by_provider"].setdefault(provider, _empty_counter())

        for bucket in (summary, stage_bucket, provider_bucket):
            bucket["call_count"] += 1

        prompt_tokens = record.get("prompt_tokens")
        completion_tokens = record.get("completion_tokens")
        total_tokens = record.get("total_tokens")
        has_usage = any(value is not None for value in (prompt_tokens, completion_tokens, total_tokens))

        if has_usage:
            for bucket in (summary, stage_bucket, provider_bucket):
                bucket["calls_with_usage"] += 1
                bucket["prompt_tokens"] += int(prompt_tokens or 0)
                bucket["completion_tokens"] += int(completion_tokens or 0)
                bucket["total_tokens"] += int(total_tokens or 0)
        else:
            for bucket in (summary, stage_bucket, provider_bucket):
                bucket["calls_without_usage"] += 1

    return summary


def llm_generate(
    prompt: str,
    model: str | None = None,
    *,
    max_retries: int | None = None,
) -> str:
    """Call the LLM and return text response."""
    override = _LLM_OVERRIDE_HANDLERS.get()
    if override is not None:
        return override.generate(prompt, model=model, max_retries=max_retries)
    return _llm_generate_live(prompt, model=model, max_retries=max_retries)


def _llm_generate_live(
    prompt: str,
    model: str | None = None,
    *,
    max_retries: int | None = None,
) -> str:
    """Call the live LLM provider and return text response."""
    load_env()
    provider = get_provider()
    model = model or get_default_model()

    if provider == "openai":
        text, usage = _openai_generate(prompt, model, max_retries=max_retries)
    elif provider == "anthropic":
        text, usage = _anthropic_generate(prompt, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
    normalized = _validate_llm_text_response(text, provider=provider, model=model)
    _record_llm_usage(
        provider=provider,
        model=model,
        response_kind="text",
        prompt=prompt,
        usage=usage,
        response_text=text,
    )
    return normalized


def llm_generate_json(
    prompt: str,
    model: str | None = None,
    *,
    max_retries: int | None = None,
) -> dict:
    """Call LLM with JSON response format and return parsed dict."""
    override = _LLM_OVERRIDE_HANDLERS.get()
    if override is not None:
        return override.generate_json(prompt, model=model, max_retries=max_retries)
    return _llm_generate_json_live(prompt, model=model, max_retries=max_retries)


def _llm_generate_json_live(
    prompt: str,
    model: str | None = None,
    *,
    max_retries: int | None = None,
) -> dict:
    """Call the live LLM provider for a JSON response and return parsed data."""
    load_env()
    provider = get_provider()
    model = model or get_default_model()

    if provider == "openai":
        text, usage = _openai_generate_json(prompt, model, max_retries=max_retries)
    elif provider == "anthropic":
        text, usage = _anthropic_generate_json(prompt, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
    _record_llm_usage(
        provider=provider,
        model=model,
        response_kind="json",
        prompt=prompt,
        usage=usage,
        response_text=text,
    )
    return _parse_llm_json_response(text, provider=provider, model=model)


@contextmanager
def llm_override_scope(
    *,
    generate: Callable[..., str],
    generate_json: Callable[..., dict],
):
    """Temporarily override LLM text/JSON calls in the current context."""
    token = _LLM_OVERRIDE_HANDLERS.set(
        LLMOverrideHandlers(
            generate=generate,
            generate_json=generate_json,
        )
    )
    try:
        yield
    finally:
        _LLM_OVERRIDE_HANDLERS.reset(token)


def _validate_llm_text_response(text: str | None, *, provider: str, model: str) -> str:
    """Validate and normalize raw text returned by an LLM."""
    normalized = (text or "").strip()
    if not normalized:
        raise RuntimeError(
            f"LLM provider '{provider}' model '{model}' returned empty text response"
        )
    return normalized


def _parse_llm_json_response(text: str | None, *, provider: str, model: str) -> dict:
    """Validate and parse a JSON response with actionable error messages."""
    normalized = (text or "").strip()
    if not normalized:
        raise RuntimeError(
            f"LLM provider '{provider}' model '{model}' returned empty JSON response"
        )
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(normalized)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        preview = _llm_response_preview(normalized)
        raise RuntimeError(
            f"LLM provider '{provider}' model '{model}' returned invalid JSON response: "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}; "
            f"response preview={preview!r}"
        ) from exc


def _openai_generate(
    prompt: str,
    model: str,
    *,
    max_retries: int | None = None,
) -> tuple[str, dict[str, int | None]]:
    """Request a plain-text completion from the OpenAI chat-completions API."""
    timeout_seconds = _openai_timeout_seconds(
        "OPENAI_TEXT_TIMEOUT_SECONDS",
        OPENAI_TEXT_TIMEOUT_SECONDS,
    )
    response = _openai_request_with_retry(
        lambda: _openai_chat_completion_create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16384,
            timeout_seconds=timeout_seconds,
        ),
        model=model,
        response_kind="text",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return response.choices[0].message.content, _extract_openai_usage(response)


def _openai_generate_json(
    prompt: str,
    model: str,
    *,
    max_retries: int | None = None,
) -> tuple[str, dict[str, int | None]]:
    """Request a JSON-object completion from OpenAI and parse the response."""
    timeout_seconds = _openai_timeout_seconds(
        "OPENAI_JSON_TIMEOUT_SECONDS",
        OPENAI_JSON_TIMEOUT_SECONDS,
    )
    response = _openai_request_with_retry(
        lambda: _openai_chat_completion_create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
            response_format={"type": "json_object"},
            timeout_seconds=timeout_seconds,
        ),
        model=model,
        response_kind="json",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return response.choices[0].message.content, _extract_openai_usage(response)


def _openai_chat_completion_create(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    timeout_seconds: float,
    response_format: dict[str, str] | None = None,
):
    """Invoke the OpenAI SDK with the chat-completions parameters Trellis expects."""
    import openai

    client = openai.OpenAI(timeout=timeout_seconds, max_retries=0)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


def _openai_request_with_retry(
    request_fn,
    *,
    model: str,
    response_kind: str,
    timeout_seconds: float,
    max_retries: int | None = None,
):
    """Run an OpenAI request with exponential-backoff retries and hard timeout wrapping."""
    retry_budget = _openai_retry_count() if max_retries is None else max(max_retries, 0)
    max_attempts = retry_budget + 1
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            request_context = {
                "provider": "openai",
                "model": model,
                "response_kind": response_kind,
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
            token = _LLM_REQUEST_CONTEXT.set(request_context)
            try:
                return _run_with_wall_clock_timeout(request_fn, timeout_seconds)
            finally:
                _LLM_REQUEST_CONTEXT.reset(token)
        except Exception as exc:  # pragma: no cover - exercised via tests with stubs
            last_error = exc
            if attempt >= max_attempts:
                break
            time.sleep(0.25 * (2 ** (attempt - 1)))

    assert last_error is not None
    raise RuntimeError(
        f"OpenAI {response_kind} request failed after {max_attempts} attempts for "
        f"model '{model}': {type(last_error).__name__}: {last_error}"
    ) from last_error


def _run_with_wall_clock_timeout(request_fn, timeout_seconds: float):
    """Enforce a hard wall-clock timeout around a blocking SDK call."""
    if timeout_seconds <= 0:
        return request_fn()

    wait_id = f"{time.time_ns()}-{threading.get_ident()}"
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    completed = threading.Event()
    started_at = time.monotonic()
    _write_llm_wait_event(
        event="wait_started",
        wait_id=wait_id,
        timeout_seconds=timeout_seconds,
    )

    def _target() -> None:
        try:
            result["value"] = request_fn()
        except BaseException as exc:  # pragma: no cover - exercised via worker-thread timeout test
            error["exception"] = exc
        finally:
            completed.set()

    worker = threading.Thread(
        target=_target,
        name="trellis-openai-request",
        daemon=True,
    )
    worker.start()
    if not completed.wait(timeout_seconds):
        _write_llm_wait_event(
            event="wait_timeout",
            wait_id=wait_id,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )
        raise TimeoutError(f"OpenAI request exceeded {timeout_seconds:.1f}s")
    if "exception" in error:
        _write_llm_wait_event(
            event="wait_error",
            wait_id=wait_id,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            error=error["exception"],
        )
        raise error["exception"]
    _write_llm_wait_event(
        event="wait_completed",
        wait_id=wait_id,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=round(time.monotonic() - started_at, 3),
    )
    return result.get("value")


def _llm_wait_log_path() -> Path:
    """Return the per-process wait log path."""
    override = os.environ.get("TRELLIS_LLM_WAIT_LOG_PATH", "").strip()
    if override:
        return Path(override)
    return LLM_WAIT_LOG_ROOT / f"{os.getpid()}.jsonl"


def _write_llm_wait_event(
    *,
    event: str,
    wait_id: str,
    timeout_seconds: float,
    elapsed_seconds: float | None = None,
    error: BaseException | None = None,
) -> None:
    """Append a live LLM wait event for in-flight diagnosis."""
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "wait_id": wait_id,
        "pid": os.getpid(),
        "thread_name": threading.current_thread().name,
        "timeout_seconds": timeout_seconds,
        "stage": _LLM_USAGE_STAGE.get(),
        "stage_metadata": dict(_LLM_USAGE_METADATA.get() or {}),
        "request_context": dict(_LLM_REQUEST_CONTEXT.get() or {}),
    }
    if elapsed_seconds is not None:
        entry["elapsed_seconds"] = elapsed_seconds
    if error is not None:
        entry["error_type"] = type(error).__name__
        entry["error"] = str(error)[:200]
    try:
        path = _llm_wait_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
    except Exception:
        pass


def _openai_timeout_seconds(env_var: str, default: float) -> float:
    """Read a positive timeout override from the environment or fall back to default."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, 1.0)


def _int_env(name: str) -> int | None:
    """Read a positive integer from the environment, returning None when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(value, 0)


def _openai_retry_count() -> int:
    """Read the OpenAI retry budget from the environment with sane fallback rules."""
    raw = os.environ.get("OPENAI_MAX_RETRIES")
    if raw is None:
        return OPENAI_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError:
        return OPENAI_MAX_RETRIES
    return max(value, 0)


def _anthropic_generate(prompt: str, model: str) -> tuple[str, dict[str, int | None]]:
    """Request a plain-text response from Anthropic's messages API."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        response.content[0].text if response.content else "",
        _extract_anthropic_usage(response),
    )


def _anthropic_generate_json(prompt: str, model: str) -> tuple[str, dict[str, int | None]]:
    """Request a JSON object from Anthropic's messages API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            "Return exactly one valid JSON object. Do not include markdown fences, "
            "prose, or leading/trailing commentary."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        response.content[0].text if response.content else "",
        _extract_anthropic_usage(response),
    )


def _extract_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from a text response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()

    start = stripped.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start:index + 1]
    return None


def _llm_response_preview(text: str, limit: int = 240) -> str:
    """Return a compact one-line preview of a provider response."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


AUDIT_LLM_PROMPTS = os.environ.get("TRELLIS_AUDIT_LLM_PROMPTS", "0") == "1"


def _record_llm_usage(
    *,
    provider: str,
    model: str,
    response_kind: str,
    prompt: str,
    usage: dict[str, int | None] | None,
    response_text: str | None = None,
) -> None:
    """Append one LLM usage record into the active session and stage collectors."""
    record = {
        "stage": _LLM_USAGE_STAGE.get() or "unscoped",
        "provider": provider,
        "model": model,
        "response_kind": response_kind,
        "prompt_chars": len(prompt),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "total_tokens": (usage or {}).get("total_tokens"),
    }
    metadata = _LLM_USAGE_METADATA.get()
    if metadata:
        record["metadata"] = dict(metadata)
    if AUDIT_LLM_PROMPTS:
        record["prompt_text"] = prompt
        if response_text is not None:
            record["response_text"] = response_text

    session_records = _LLM_USAGE_SESSION.get()
    if session_records is not None:
        session_records.append(record)

    stage_records = _LLM_USAGE_STAGE_COLLECTOR.get()
    if stage_records is not None:
        stage_records.append(record)


def _extract_openai_usage(response) -> dict[str, int | None]:
    """Normalize token usage from an OpenAI chat-completions response."""
    usage = getattr(response, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_anthropic_usage(response) -> dict[str, int | None]:
    """Normalize token usage from an Anthropic messages response."""
    usage = getattr(response, "usage", None)
    prompt_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
    completion_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _usage_value(usage, *keys: str) -> int | None:
    """Read a usage attribute from an SDK object or dict-like payload."""
    if usage is None:
        return None
    for key in keys:
        value = getattr(usage, key, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None
