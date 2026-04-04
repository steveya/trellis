"""Thin MCP-facing adapters over the transport-neutral Trellis platform services."""

from trellis.mcp.errors import TrellisMcpError
from trellis.mcp.server import TrellisMcpServer, bootstrap_mcp_server

__all__ = [
    "TrellisMcpError",
    "TrellisMcpServer",
    "bootstrap_mcp_server",
]
