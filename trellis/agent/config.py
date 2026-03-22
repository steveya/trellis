"""Agent configuration: API keys, LLM provider, environment setup."""

from __future__ import annotations

import json
import os
from pathlib import Path

_LOADED = False

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = {
    "openai": "o3-mini",
    "anthropic": "claude-sonnet-4-6",
}

ALLOWED_FIELD_TYPES = frozenset({
    "float", "int", "str", "bool", "date", "str | None",
    "Frequency", "DayCountConvention",
})


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
    load_env()
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)


def get_default_model() -> str:
    return DEFAULT_MODEL.get(get_provider(), DEFAULT_MODEL["openai"])


def llm_generate(prompt: str, model: str | None = None) -> str:
    """Call the LLM and return text response."""
    load_env()
    provider = get_provider()
    model = model or get_default_model()

    if provider == "openai":
        return _openai_generate(prompt, model)
    elif provider == "anthropic":
        return _anthropic_generate(prompt, model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def llm_generate_json(prompt: str, model: str | None = None) -> dict:
    """Call LLM with JSON response format and return parsed dict."""
    load_env()
    provider = get_provider()
    model = model or get_default_model()

    if provider == "openai":
        return _openai_generate_json(prompt, model)
    elif provider == "anthropic":
        text = _anthropic_generate(prompt, model)
        return json.loads(text)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _openai_generate(prompt: str, model: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=16384,
    )
    return response.choices[0].message.content.strip()


def _openai_generate_json(prompt: str, model: str) -> dict:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _anthropic_generate(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
