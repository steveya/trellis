"""Local streamable HTTP transport wrapper for the Trellis MCP shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from trellis.mcp.demo import build_demo_provider_registry, enable_local_demo_mode
from trellis.mcp.server import TrellisMcpServer, bootstrap_mcp_server
from trellis.platform.providers import ProviderRegistry

try:  # pragma: no cover - exercised indirectly when dependency exists
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.prompts import Prompt
    from mcp.server.fastmcp.prompts.base import PromptArgument
except ImportError as exc:  # pragma: no cover - import guard
    FastMCP = None  # type: ignore[assignment]
    Prompt = None  # type: ignore[assignment]
    PromptArgument = None  # type: ignore[assignment]
    _FASTMCP_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _FASTMCP_IMPORT_ERROR = None


_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_DEFAULT_HTTP_PATH = "/mcp"
_RESOURCE_PARAM_PATTERN = re.compile(r"{([^{}]+)}")
_PROMPT_ARGUMENT_NAMES: dict[str, tuple[str, ...]] = {
    "compare_model_versions": ("model_id",),
    "configure_market_data": ("session_id",),
    "explain_model_selection": (),
    "persist_current_model": ("model_id",),
    "price_trade": ("session_id",),
    "price_trade_audit": ("run_id",),
    "validate_candidate_model": ("model_id", "version"),
}


def _require_fastmcp() -> None:
    if FastMCP is None:
        raise ImportError(
            "The local Trellis MCP HTTP transport requires the optional `mcp` package. "
            "Install `trellis[mcp]` or `mcp>=1.27`."
        ) from _FASTMCP_IMPORT_ERROR


def _normalize_http_path(path: str | None) -> str:
    text = str(path or _DEFAULT_HTTP_PATH).strip() or _DEFAULT_HTTP_PATH
    if not text.startswith("/"):
        text = f"/{text}"
    return text.rstrip("/") or "/"


def _resource_param_names(uri_template: str) -> tuple[str, ...]:
    return tuple(_RESOURCE_PARAM_PATTERN.findall(uri_template))


def _render_prompt_message(payload: Mapping[str, object]) -> str:
    tools = payload.get("tools") or ()
    resources = payload.get("resources") or ()
    tool_lines = "\n".join(f"- `{name}`" for name in tools) or "- none"
    resource_lines = "\n".join(f"- `{uri}`" for uri in resources) or "- none"
    workflow = str(payload.get("prompt", "")).strip()
    return (
        f"{str(payload.get('description', '')).strip()}\n\n"
        f"Recommended tools:\n{tool_lines}\n\n"
        f"Relevant resources:\n{resource_lines}\n\n"
        f"Workflow:\n{workflow}"
    ).strip()


def _build_transport_tool_handler(shell: TrellisMcpServer, name: str):
    def handler(arguments: dict[str, Any] | None = None) -> Mapping[str, object]:
        return shell.call_tool(name, arguments)

    return handler


def _build_transport_resource_handler(shell: TrellisMcpServer, uri_template: str):
    param_names = _resource_param_names(uri_template)

    def handler(**params):
        uri = uri_template.format(**{name: str(params.get(name, "")).strip() for name in param_names})
        return shell.read_resource(uri)

    return handler


def _build_transport_prompt_handler(shell: TrellisMcpServer, name: str):
    def handler(**arguments):
        payload = shell.get_prompt(name, arguments)
        return _render_prompt_message(payload)

    return handler


def _build_prompt_arguments(name: str):
    if PromptArgument is None:
        return None
    return [
        PromptArgument(
            name=argument_name,
            description=f"Trellis prompt argument `{argument_name}`.",
            required=False,
        )
        for argument_name in _PROMPT_ARGUMENT_NAMES.get(name, ())
    ]


def _register_transport_tools(shell: TrellisMcpServer, transport) -> None:
    for definition in shell.tool_registry.iter_definitions():
        transport._tool_manager.add_tool(
            _build_transport_tool_handler(shell, definition.name),
            name=definition.name,
            description=definition.description,
            meta={"trellis_transport": "streamable_http"},
        )


def _register_transport_resources(shell: TrellisMcpServer, transport) -> None:
    for uri_template in shell.list_resources():
        transport._resource_manager.add_template(
            _build_transport_resource_handler(shell, uri_template),
            uri_template=uri_template,
            name=uri_template,
            description=f"Read the governed Trellis resource `{uri_template}`.",
            mime_type="application/json",
            meta={"trellis_transport": "streamable_http"},
        )


def _register_transport_prompts(shell: TrellisMcpServer, transport) -> None:
    for prompt_name in shell.list_prompts():
        prompt_payload = shell.get_prompt(prompt_name, {})
        prompt = Prompt(
            name=prompt_name,
            description=str(prompt_payload.get("description", "")).strip(),
            arguments=_build_prompt_arguments(prompt_name),
            fn=_build_transport_prompt_handler(shell, prompt_name),
        )
        transport._prompt_manager.add_prompt(prompt)


@dataclass(frozen=True)
class TrellisHttpMcpServer:
    """Runnable local HTTP transport around the transport-neutral Trellis shell."""

    shell: TrellisMcpServer
    transport: Any
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    streamable_http_path: str = _DEFAULT_HTTP_PATH

    @property
    def endpoint_url(self) -> str:
        """Return the local streamable HTTP endpoint."""
        return f"http://{self.host}:{self.port}{self.streamable_http_path}"

    def streamable_http_app(self):
        """Return the FastMCP ASGI app for local mounting or tests."""
        return self.transport.streamable_http_app()

    def run(self) -> None:
        """Run the local streamable HTTP MCP server."""
        self.transport.run(transport="streamable-http")

    def list_tools(self) -> tuple[str, ...]:
        """Return the shell tool surface."""
        return self.shell.list_tools()

    def list_resources(self) -> tuple[str, ...]:
        """Return the shell resource surface."""
        return self.shell.list_resources()

    def list_prompts(self) -> tuple[str, ...]:
        """Return the shell prompt surface."""
        return self.shell.list_prompts()


def bootstrap_http_mcp_server(
    *,
    state_root: Path | str | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    provider_registry: ProviderRegistry | None = None,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    streamable_http_path: str = _DEFAULT_HTTP_PATH,
    debug: bool = False,
    log_level: str = "INFO",
    demo_mode: bool = False,
    demo_session_id: str = "demo",
) -> TrellisHttpMcpServer:
    """Bootstrap the local streamable HTTP transport over the Trellis MCP shell."""
    _require_fastmcp()
    resolved_provider_registry = (
        provider_registry
        if provider_registry is not None
        else build_demo_provider_registry() if demo_mode else None
    )
    shell = bootstrap_mcp_server(
        state_root=state_root,
        config_path=config_path,
        env=env,
        provider_registry=resolved_provider_registry,
    )
    if demo_mode:
        enable_local_demo_mode(shell.services, session_id=demo_session_id)
    normalized_path = _normalize_http_path(streamable_http_path)
    transport = FastMCP(
        name=shell.services.config.server.name,
        host=str(host or _DEFAULT_HOST).strip() or _DEFAULT_HOST,
        port=int(port),
        streamable_http_path=normalized_path,
        debug=debug,
        log_level=str(log_level or "INFO").strip() or "INFO",
    )
    _register_transport_tools(shell, transport)
    _register_transport_resources(shell, transport)
    _register_transport_prompts(shell, transport)
    return TrellisHttpMcpServer(
        shell=shell,
        transport=transport,
        host=str(host or _DEFAULT_HOST).strip() or _DEFAULT_HOST,
        port=int(port),
        streamable_http_path=normalized_path,
    )


__all__ = [
    "TrellisHttpMcpServer",
    "bootstrap_http_mcp_server",
]
