"""Thin MCP prompt workflows over the stable Trellis tool and resource surface."""

from __future__ import annotations

from trellis.mcp.errors import TrellisMcpError


class PromptRegistry:
    """Small in-process registry for Trellis MCP prompt workflows."""

    def list_prompts(self) -> tuple[str, ...]:
        return (
            "compare_model_versions",
            "configure_market_data",
            "explain_model_selection",
            "persist_current_model",
            "price_trade",
            "price_trade_audit",
            "validate_candidate_model",
        )

    def get_prompt(self, name: str, arguments=None):
        normalized_name = str(name or "").strip()
        payload = dict(arguments or {})
        builders = {
            "price_trade": self._price_trade,
            "price_trade_audit": self._price_trade_audit,
            "persist_current_model": self._persist_current_model,
            "compare_model_versions": self._compare_model_versions,
            "explain_model_selection": self._explain_model_selection,
            "configure_market_data": self._configure_market_data,
            "validate_candidate_model": self._validate_candidate_model,
        }
        try:
            builder = builders[normalized_name]
        except KeyError as exc:
            raise TrellisMcpError(
                code="unknown_prompt",
                message=f"Unknown Trellis MCP prompt: {normalized_name!r}",
                details={"prompt_name": normalized_name},
            ) from exc
        return builder(payload)

    @staticmethod
    def _price_trade(arguments):
        session_id = str(arguments.get("session_id", "default")).strip() or "default"
        return {
            "name": "price_trade",
            "description": "Guide one host through the governed trade-pricing flow without bypassing the canonical tool surface.",
            "tools": [
                "trellis.session.get_context",
                "trellis.providers.list",
                "trellis.providers.configure",
                "trellis.trade.parse",
                "trellis.model.match",
                "trellis.price.trade",
            ],
            "resources": [
                "trellis://runs/{run_id}",
                "trellis://runs/{run_id}/audit",
            ],
            "prompt": (
                f"Use governed session {session_id!r}. Confirm provider bindings first, "
                "parse the trade, inspect deterministic model matching, then call "
                "`trellis.price.trade`. If the run succeeds or blocks, inspect the persisted "
                "run and audit resources rather than rerunning ad hoc helper code."
            ),
        }

    @staticmethod
    def _price_trade_audit(arguments):
        run_id = str(arguments.get("run_id", "{run_id}")).strip() or "{run_id}"
        return {
            "name": "price_trade_audit",
            "description": "Inspect one persisted governed pricing run through the canonical audit surfaces.",
            "tools": ["trellis.run.get", "trellis.run.get_audit"],
            "resources": [
                f"trellis://runs/{run_id}",
                f"trellis://runs/{run_id}/audit",
                f"trellis://runs/{run_id}/inputs",
                f"trellis://runs/{run_id}/outputs",
            ],
            "prompt": (
                "Read the persisted run summary first, then the canonical audit bundle. "
                "Use the run input/output resources for narrow inspection and avoid "
                "replaying valuation unless the audit data is actually incomplete."
            ),
        }

    @staticmethod
    def _persist_current_model(arguments):
        model_id = str(arguments.get("model_id", "{model_id}")).strip() or "{model_id}"
        return {
            "name": "persist_current_model",
            "description": "Persist a governed model revision with explicit lineage instead of mutating the prior version.",
            "tools": ["trellis.model.persist", "trellis.model.versions.list"],
            "resources": [
                f"trellis://models/{model_id}",
                f"trellis://models/{model_id}/versions",
            ],
            "prompt": (
                "Write the next governed model version through `trellis.model.persist`, "
                "then inspect the stored version history to confirm lineage and sidecar artifacts."
            ),
        }

    @staticmethod
    def _compare_model_versions(arguments):
        model_id = str(arguments.get("model_id", "{model_id}")).strip() or "{model_id}"
        return {
            "name": "compare_model_versions",
            "description": "Compare two governed model versions using the canonical stored diff surface.",
            "tools": ["trellis.model.diff"],
            "resources": [
                f"trellis://models/{model_id}/versions",
                f"trellis://models/{model_id}/versions/{{version}}/contract",
                f"trellis://models/{model_id}/versions/{{version}}/code",
                f"trellis://models/{model_id}/versions/{{version}}/validation-report",
            ],
            "prompt": (
                "Call `trellis.model.diff` for the exact version pair, then use the "
                "stored contract/code/validation resources only when deeper artifact inspection is needed."
            ),
        }

    @staticmethod
    def _explain_model_selection(arguments):
        return {
            "name": "explain_model_selection",
            "description": "Explain deterministic governed model selection for one trade request.",
            "tools": ["trellis.trade.parse", "trellis.model.explain_match"],
            "resources": ["trellis://models/{model_id}", "trellis://policies/{policy_id}"],
            "prompt": (
                "Parse the trade, inspect the deterministic match explanation, and only then "
                "follow any referenced model or policy resources for deeper review."
            ),
        }

    @staticmethod
    def _configure_market_data(arguments):
        session_id = str(arguments.get("session_id", "default")).strip() or "default"
        return {
            "name": "configure_market_data",
            "description": "Guide explicit governed market-data binding for one session.",
            "tools": [
                "trellis.session.get_context",
                "trellis.providers.list",
                "trellis.providers.configure",
                "trellis.run_mode.set",
            ],
            "resources": [
                "trellis://providers/{provider_id}",
                "trellis://policies/{policy_id}",
            ],
            "prompt": (
                f"Inspect governed session {session_id!r}, list visible providers, then persist "
                "explicit provider bindings and run mode changes without relying on hidden source defaults."
            ),
        }

    @staticmethod
    def _validate_candidate_model(arguments):
        model_id = str(arguments.get("model_id", "{model_id}")).strip() or "{model_id}"
        version = str(arguments.get("version", "{version}")).strip() or "{version}"
        return {
            "name": "validate_candidate_model",
            "description": "Run deterministic validation and explicit lifecycle review for one governed candidate version.",
            "tools": ["trellis.model.validate", "trellis.model.promote"],
            "resources": [
                f"trellis://models/{model_id}",
                f"trellis://models/{model_id}/versions/{version}/validation-report",
            ],
            "prompt": (
                "Persist the deterministic validation report first. Only apply an explicit "
                "lifecycle transition after reviewing the stored validation resource."
            ),
        }


def build_prompt_registry() -> PromptRegistry:
    """Build the current prompt registry."""
    return PromptRegistry()


__all__ = [
    "PromptRegistry",
    "build_prompt_registry",
]
