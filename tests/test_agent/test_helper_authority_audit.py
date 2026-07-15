from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    routes = {
        "routes": [
            {
                "id": "promoted_route",
                "status": "promoted",
                "primitives": [
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_example",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "barrier_option_price",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_optional",
                        "role": "route_helper",
                        "required": False,
                    },
                ],
                "conditional_primitives": [
                    {
                        "when": {
                            "payoff_family": "example",
                            "methods": ["monte_carlo"],
                        },
                        "primitives": [
                            {
                                "module": "trellis.models.example",
                                "symbol": "price_example_monte_carlo",
                                "role": "route_helper",
                            }
                        ],
                    }
                ],
            },
            {
                "id": "candidate_route",
                "status": "candidate",
                "primitives": [
                    {
                        "module": "trellis.models.candidate",
                        "symbol": "price_candidate",
                        "role": "route_helper",
                    }
                ],
            },
        ]
    }
    bindings = {
        "bindings": [
            {
                "route_id": "promoted_route",
                "primitives": [
                    {
                        "module": "trellis.models.example",
                        "symbol": "price_example",
                        "role": "route_helper",
                    },
                    {
                        "module": "trellis.models.example",
                        "symbol": "barrier_option_price",
                        "role": "route_helper",
                    }
                ],
                "conditional_primitives": [
                    {
                        "when": "default",
                        "primitives": [
                            {
                                "module": "trellis.models.binding_only",
                                "symbol": "price_binding_only",
                                "role": "route_helper",
                            }
                        ],
                    }
                ],
            },
            {
                "route_id": "candidate_route",
                "primitives": [
                    {
                        "module": "trellis.models.candidate",
                        "symbol": "price_candidate",
                        "role": "route_helper",
                    }
                ],
            },
        ]
    }
    _write_yaml(
        tmp_path / "trellis/agent/knowledge/canonical/routes.yaml",
        routes,
    )
    _write_yaml(
        tmp_path / "trellis/agent/knowledge/canonical/backend_bindings.yaml",
        bindings,
    )
    adapter = tmp_path / "trellis/instruments/_agent/example.py"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text(
        """
from trellis.models.example import price_example as delegated_price
from trellis.models.example import barrier_option_price as delegated_barrier
from trellis.models.unused import price_unused
import trellis.models.binding_only as binding
import trellis.models.direct


def price_local():
    return 0.0


def evaluate():
    return delegated_price(None, None) + delegated_barrier(None, None) + binding.price_binding_only(None, None) + trellis.models.direct.price_direct(None, None) + price_local()
""".lstrip(),
        encoding="utf-8",
    )
    return tmp_path


def test_audit_preserves_required_route_and_binding_authority_drift(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(_fixture_root(tmp_path))

    assert report.promoted_route_count == 1
    assert [(item.route_id, item.condition, item.symbol) for item in report.route_authority] == [
        ("promoted_route", "base", "barrier_option_price"),
        ("promoted_route", "base", "price_example"),
        (
            "promoted_route",
            '{"methods":["monte_carlo"],"payoff_family":"example"}',
            "price_example_monte_carlo",
        ),
    ]
    assert [(item.route_id, item.condition, item.symbol) for item in report.binding_authority] == [
        ("promoted_route", "base", "barrier_option_price"),
        ("promoted_route", "base", "price_example"),
        ("promoted_route", '"default"', "price_binding_only"),
    ]
    assert [item.symbol for item in report.route_only_authority] == [
        "price_example_monte_carlo"
    ]
    assert [item.symbol for item in report.binding_only_authority] == [
        "price_binding_only"
    ]


def test_audit_resolves_import_aliases_and_ignores_unused_or_local_price_calls(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(_fixture_root(tmp_path))

    assert [item.symbol for item in report.adapter_calls] == [
        "price_binding_only",
        "barrier_option_price",
        "price_example",
        "price_direct",
    ]
    assert all(
        item.matches_required_authority
        for item in report.adapter_calls
        if item.symbol != "price_direct"
    )
    direct = next(item for item in report.adapter_calls if item.symbol == "price_direct")
    assert direct.module == "trellis.models.direct"
    assert direct.matches_required_authority is False
    assert [item.symbol for item in report.adapter_calls if item.is_price_call] == [
        "price_binding_only",
        "price_example",
        "price_direct",
    ]
    example = next(item for item in report.adapter_calls if item.symbol == "price_example")
    assert example.local_name == "delegated_price"
    assert example.path == "trellis/instruments/_agent/example.py"
    assert example.line == 13


def test_helper_authority_report_has_stable_machine_readable_shape(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    payload = build_helper_authority_report(_fixture_root(tmp_path)).to_dict()

    assert payload["schema_version"] == 1
    assert payload["summary"] == {
        "promoted_route_count": 1,
        "route_authority_route_count": 1,
        "route_authority_reference_count": 3,
        "binding_authority_route_count": 1,
        "binding_authority_reference_count": 3,
        "route_only_reference_count": 1,
        "binding_only_reference_count": 1,
        "adapter_price_call_file_count": 1,
        "adapter_price_call_count": 3,
        "adapter_authority_call_file_count": 1,
        "adapter_authority_call_count": 3,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_helper_authority_human_report_surfaces_drift_and_adapter_authority(tmp_path):
    from trellis.agent.helper_authority_audit import (
        build_helper_authority_report,
        render_helper_authority_report,
    )

    rendered = render_helper_authority_report(
        build_helper_authority_report(_fixture_root(tmp_path))
    )

    assert "Helper authority audit" in rendered
    assert "route_authority_references=3" in rendered
    assert "binding_authority_references=3" in rendered
    assert "route_only_references=1" in rendered
    assert "binding_only_references=1" in rendered
    assert "price_example_monte_carlo" in rendered
    assert "price_binding_only" in rendered
    assert "barrier_option_price" in rendered
    assert "trellis/instruments/_agent/example.py:13" in rendered


def test_current_repository_helper_authority_report_is_internally_consistent():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)

    assert report.promoted_route_count > 0
    assert all(item.required for item in report.route_authority)
    assert all(item.required for item in report.binding_authority)
    assert all((root / item.path).is_file() for item in report.adapter_calls)
    assert report.to_dict()["summary"]["route_authority_reference_count"] == len(
        report.route_authority
    )


def test_current_repository_retires_arithmetic_asian_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    asian_symbols = {
        "price_asian_option_monte_carlo",
        "price_arithmetic_asian_option_analytical",
        "price_arithmetic_asian_option_monte_carlo",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in asian_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/asianoption.py"
        and item.symbol in asian_symbols
    ]


def test_current_repository_retires_analytical_digital_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_digital_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/digitaloption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_chooser_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_chooser_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/chooseroption.py"
        and item.symbol == helper_symbol
    ]
