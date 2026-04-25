from __future__ import annotations

import scripts.run_tasks as runner


def test_parse_args_uses_environment_default_model(monkeypatch):
    monkeypatch.setattr(runner, "get_default_model", lambda: "gpt-4.1")

    args = runner._parse_args([])

    assert args.model == "gpt-4.1"
