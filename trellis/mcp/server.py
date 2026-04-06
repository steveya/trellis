"""Thin transport-neutral MCP server shell for Trellis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trellis.mcp.prompts import PromptRegistry, build_prompt_registry
from trellis.mcp.resources import ResourceRegistry, build_resource_registry
from trellis.mcp.tool_registry import ToolRegistry, build_tool_registry
from trellis.platform.providers import ProviderRegistry
from trellis.platform.services import PlatformServiceContainer, bootstrap_platform_services


@dataclass(frozen=True)
class TrellisMcpServer:
    """Minimal MCP server shell built around the shared platform services."""

    services: PlatformServiceContainer
    tool_registry: ToolRegistry
    resource_registry: ResourceRegistry
    prompt_registry: PromptRegistry

    def list_tools(self) -> tuple[str, ...]:
        """Return the currently registered tool names."""
        return self.tool_registry.list_tools()

    def call_tool(self, name: str, arguments: Mapping[str, object] | None = None) -> Mapping[str, object]:
        """Dispatch one tool call through the shared registry."""
        return self.tool_registry.call_tool(name, arguments)

    def list_resources(self) -> tuple[str, ...]:
        """Return the currently registered resource names."""
        return self.resource_registry.list_resources()

    def read_resource(self, uri: str) -> object:
        """Resolve one Trellis MCP resource URI."""
        return self.resource_registry.read_resource(uri)

    def list_prompts(self) -> tuple[str, ...]:
        """Return the currently registered prompt names."""
        return self.prompt_registry.list_prompts()

    def get_prompt(self, name: str, arguments: Mapping[str, object] | None = None) -> Mapping[str, object]:
        """Resolve one Trellis MCP prompt workflow."""
        return self.prompt_registry.get_prompt(name, arguments)

    def describe_host_packaging(self) -> Mapping[str, object]:
        """Return the common host-packaging manifest for supported MCP clients."""
        common_contract = {
            "bootstrap_entrypoint": "trellis.mcp.server.bootstrap_mcp_server",
            "transport": self.services.config.server.transport,
            "tools": list(self.list_tools()),
            "resources": list(self.list_resources()),
            "prompts": list(self.list_prompts()),
        }
        return {
            "server_name": self.services.config.server.name,
            "common_contract": common_contract,
            "hosts": {
                "claude": {
                    "transport": common_contract["transport"],
                    "uses_common_contract": True,
                },
                "codex": {
                    "transport": common_contract["transport"],
                    "uses_common_contract": True,
                },
                "chatgpt": {
                    "transport": common_contract["transport"],
                    "uses_common_contract": True,
                },
            },
        }


def bootstrap_mcp_server(
    *,
    state_root: Path | str | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> TrellisMcpServer:
    """Bootstrap the thin MCP shell over the shared service container."""
    services = bootstrap_platform_services(
        state_root=state_root,
        config_path=config_path,
        env=env,
        provider_registry=provider_registry,
    )
    return TrellisMcpServer(
        services=services,
        tool_registry=build_tool_registry(services),
        resource_registry=build_resource_registry(services),
        prompt_registry=build_prompt_registry(),
    )


__all__ = [
    "TrellisMcpServer",
    "bootstrap_mcp_server",
]
