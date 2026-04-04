#!/Users/steveyang/miniforge3/bin/python3
"""Run the Trellis MCP server over local streamable HTTP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trellis.mcp.http_transport import bootstrap_http_mcp_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Trellis MCP server on a local streamable HTTP endpoint.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the local MCP server.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for the local MCP server.")
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Streamable HTTP path exposed to Codex and Claude Code.",
    )
    parser.add_argument(
        "--state-root",
        default=None,
        help="Optional Trellis state root. Defaults to TRELLIS_STATE_ROOT or the repo-local .trellis_state.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional explicit Trellis bootstrap config path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="FastMCP log level.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable FastMCP debug mode.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Enable the local sandbox+mock prompt-flow demo bootstrap.",
    )
    parser.add_argument(
        "--demo-session-id",
        default="demo",
        help="Session id to seed when --demo is enabled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    server = bootstrap_http_mcp_server(
        state_root=None if args.state_root is None else Path(args.state_root),
        config_path=None if args.config_path is None else Path(args.config_path),
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
        log_level=args.log_level,
        debug=bool(args.debug),
        demo_mode=bool(args.demo),
        demo_session_id=args.demo_session_id,
    )
    print(f"Trellis MCP listening on {server.endpoint_url}")
    server.run()


if __name__ == "__main__":
    main()
