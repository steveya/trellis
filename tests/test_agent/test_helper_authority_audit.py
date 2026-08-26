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


def _fixture_root_with_indirect_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
from trellis.models.example import barrier_option_price


def accept_callback(callback):
    return callback


delegated_price = price_example


def evaluate():
    return accept_callback(barrier_option_price)
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_from_imported_module_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models import example


def accept_callback(callback):
    return callback


delegated_price = example.price_example


def callback_evaluate():
    return accept_callback(example.barrier_option_price)


def direct_evaluate():
    return example.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_imported_module_alias_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


delegated_module = helpers


def dynamic_evaluate():
    return getattr(helpers, "price_example")()


def delegated_evaluate():
    return delegated_module.price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_same_named_non_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.other import price_example


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_relative_import_authority(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from ...models.example import price_example
from ...models.example import barrier_option_price


delegated_barrier = barrier_option_price


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_authority_call_attribute(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.price_example.__call__()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_nested_non_authority_shadow(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def shadowed():
    from trellis.models.other import price_example
    delegated = price_example
    return price_example()


def parameter_shadow(price_example):
    return price_example


delegated = price_example


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_nested_authority_shadow(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.other import price_example


def authoritative():
    from trellis.models.example import price_example
    delegated = price_example
    return price_example()


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_wildcard_authority_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import *


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_same_scope_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example as non_authority_last
from trellis.models.other import price_example as non_authority_last
from trellis.models.other import price_example as authority_last
from trellis.models.example import price_example as authority_last


non_authority_callback = non_authority_last
authority_callback = authority_last


def evaluate():
    return non_authority_last() + authority_last()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_late_global_authority_import(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
def evaluate():
    global price_example
    return price_example()


from trellis.models.example import price_example
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_deferred_enclosing_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def evaluate():
    return price_example()


early_value = evaluate()
from trellis.models.other import price_example
late_value = evaluate()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_immediate_comprehension_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
early_values = [price_example() for _ in range(1)]
from trellis.models.other import price_example
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_authority_module_chain(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.__dict__["price_example"]()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_dynamic_authority_getattribute(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
import trellis.models.example as helpers


def evaluate():
    return helpers.__getattribute__("price_example")()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_late_class_rebinding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example


def local_price():
    return 0.0


class Adapter:
    delegated = price_example()
    price_example = local_price
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_ordinary_rebindings(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example
from trellis.models.example import barrier_option_price


def local_price():
    return 0.0


price_example = local_price


def barrier_option_price():
    return 0.0


def evaluate():
    return price_example() + barrier_option_price()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_annotation_only_references(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example

price_example: object


class Adapter:
    price_example: object
    delegated = price_example()


def evaluate():
    return price_example()
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _fixture_root_with_deleted_class_binding(tmp_path: Path) -> Path:
    root = _fixture_root(tmp_path)
    adapter = root / "trellis/instruments/_agent/example.py"
    adapter.write_text(
        """
from trellis.models.example import price_example as helper


def local_price():
    return 0.0


class Adapter:
    helper = local_price
    del helper
    delegated = helper()
""".lstrip(),
        encoding="utf-8",
    )
    return root


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

    assert payload["schema_version"] == 2
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
        "adapter_indirect_authority_use_file_count": 0,
        "adapter_indirect_authority_use_count": 0,
    }
    assert payload["adapter_indirect_authority_uses"] == []
    assert json.loads(json.dumps(payload)) == payload


def test_audit_rejects_assignment_aliases_and_callback_authority(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_indirect_authority(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (9, "price_example", "price_example", "indirect_reference"),
        (
            13,
            "barrier_option_price",
            "barrier_option_price",
            "indirect_reference",
        ),
    ]
    assert report.has_adapter_authority is True
    assert report.summary["adapter_indirect_authority_use_file_count"] == 1
    assert report.summary["adapter_indirect_authority_use_count"] == 2


def test_audit_resolves_authority_through_from_imported_modules(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_from_imported_module_authority(tmp_path)
    )

    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_calls
    ] == [
        (
            16,
            "example.price_example",
            "trellis.models.example",
            "price_example",
        )
    ]
    assert [
        (item.line, item.local_name, item.module, item.symbol)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            8,
            "example.price_example",
            "trellis.models.example",
            "price_example",
        ),
        (
            12,
            "example.barrier_option_price",
            "trellis.models.example",
            "barrier_option_price",
        ),
    ]
    assert report.has_adapter_authority is True
    assert report.summary["adapter_indirect_authority_use_file_count"] == 1
    assert report.summary["adapter_indirect_authority_use_count"] == 2


def test_audit_rejects_imported_authority_modules_used_as_values(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_imported_module_alias_authority(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (
            item.line,
            item.local_name,
            item.module,
            item.symbol,
            item.use_kind,
        )
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            4,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        ),
        (
            8,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_does_not_match_same_named_symbol_from_other_module(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_same_named_non_authority(tmp_path)
    )

    assert len(report.adapter_calls) == 1
    assert report.adapter_calls[0].module == "trellis.models.other"
    assert report.adapter_calls[0].symbol == "price_example"
    assert report.adapter_calls[0].is_price_call is True
    assert report.adapter_calls[0].matches_required_authority is False
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_normalizes_relative_authority_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_relative_import_authority(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [("trellis.models.example", "price_example", True)]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "barrier_option_price",
            "indirect_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_rejects_authority_reached_through_call_attribute(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_authority_call_attribute(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_keeps_nested_non_authority_import_scoped(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_nested_non_authority_shadow(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("trellis.models.other", "price_example", False),
        ("trellis.models.example", "price_example", True),
    ]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_keeps_nested_authority_import_scoped(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_nested_authority_shadow(tmp_path)
    )

    assert [
        (item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("trellis.models.example", "price_example", True),
        ("trellis.models.other", "price_example", False),
    ]
    assert [
        (item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_rejects_wildcard_imports_from_authority_modules(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_wildcard_authority_import(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            1,
            "*",
            "trellis.models.example",
            "*",
            "wildcard_import",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_respects_same_scope_import_rebinding_order(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_same_scope_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        ("authority_last", "trellis.models.example", True),
        ("non_authority_last", "trellis.models.other", False),
    ]
    assert [
        (item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            "authority_last",
            "trellis.models.example",
            "price_example",
            "indirect_reference",
        )
    ]


def test_audit_resolves_late_module_import_for_explicit_global(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_late_global_authority_import(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        )
    ]
    assert report.adapter_indirect_authority_uses == ()


def test_audit_retains_possible_enclosing_imports_for_deferred_calls(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_deferred_enclosing_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        ),
        (
            "price_example",
            "trellis.models.other",
            "price_example",
            False,
        ),
    ]
    assert report.has_adapter_authority is True


def test_audit_uses_creation_position_for_immediate_comprehensions(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_immediate_comprehension_rebinding(tmp_path)
    )

    assert [
        (item.local_name, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (
            "price_example",
            "trellis.models.example",
            "price_example",
            True,
        )
    ]


def test_audit_retains_authority_module_root_in_dynamic_attribute_chain(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_authority_module_chain(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_retains_authority_module_root_for_dynamic_getattribute(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_dynamic_authority_getattribute(tmp_path)
    )

    assert report.adapter_calls == ()
    assert [
        (item.line, item.local_name, item.module, item.symbol, item.use_kind)
        for item in report.adapter_indirect_authority_uses
    ] == [
        (
            5,
            "helpers",
            "trellis.models.example",
            "*",
            "indirect_module_reference",
        )
    ]
    assert report.has_adapter_authority is True


def test_audit_resolves_early_class_reference_before_late_rebinding(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_late_class_rebinding(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [(9, "trellis.models.example", "price_example", True)]
    assert report.has_adapter_authority is True


def test_audit_honors_ordinary_rebindings_after_imports(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_ordinary_rebindings(tmp_path)
    )

    assert report.adapter_calls == ()
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False


def test_audit_preserves_imports_across_annotation_only_statements(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_annotation_only_references(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [
        (8, "trellis.models.example", "price_example", True),
        (12, "trellis.models.example", "price_example", True),
    ]
    assert report.has_adapter_authority is True


def test_audit_restores_outer_lookup_after_deleted_class_binding(tmp_path):
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    report = build_helper_authority_report(
        _fixture_root_with_deleted_class_binding(tmp_path)
    )

    assert [
        (item.line, item.module, item.symbol, item.matches_required_authority)
        for item in report.adapter_calls
    ] == [(11, "trellis.models.example", "price_example", True)]
    assert report.has_adapter_authority is True


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
    assert "adapter_indirect_authority_uses=0" in rendered
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
    assert all(
        (root / item.path).is_file()
        for item in report.adapter_indirect_authority_uses
    )
    assert report.to_dict()["summary"]["route_authority_reference_count"] == len(
        report.route_authority
    )


def test_current_repository_has_zero_admitted_adapter_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    authority_calls = [
        item for item in report.adapter_calls if item.matches_required_authority
    ]

    assert authority_calls == []
    assert report.adapter_indirect_authority_uses == ()
    assert report.has_adapter_authority is False
    assert report.summary["adapter_authority_call_file_count"] == 0
    assert report.summary["adapter_authority_call_count"] == 0
    assert report.summary["adapter_indirect_authority_use_file_count"] == 0
    assert report.summary["adapter_indirect_authority_use_count"] == 0


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


def test_current_repository_retires_single_name_cds_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {
        "build_cds_schedule",
        "price_cds_analytical",
        "price_cds_monte_carlo",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/cds.py"
        and item.symbol in retired_symbols
    ]
    summary = report.to_dict()["summary"]
    assert summary["route_authority_reference_count"] <= 39
    assert summary["binding_authority_reference_count"] <= 43
    assert summary["route_only_reference_count"] <= 2
    assert summary["binding_only_reference_count"] <= 6


def test_current_repository_classifies_scalar_barrier_formula_as_pricing_kernel():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    kernel_symbol = "barrier_option_price"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == kernel_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/barrieroption.py"
        and item.symbol == kernel_symbol
        and item.matches_authority
    ]


def test_current_repository_retires_european_swaption_wrapper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == "price_swaption_black76"
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol == "price_swaption_black76"
    ]


def test_current_repository_retires_european_swaption_tree_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {"price_swaption_tree", "build_swaption_tree_spec"}

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol in retired_symbols
    ]


def test_current_repository_retires_bermudan_swaption_lower_bound_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_bermudan_swaption_black76_lower_bound"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/bermudanswaption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_bermudan_swaption_tree_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_bermudan_swaption_tree"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/bermudanswaption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_european_swaption_monte_carlo_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    retired_symbols = {
        "price_swaption_monte_carlo",
        "resolve_swaption_monte_carlo_problem",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in retired_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/swaption.py"
        and item.symbol in retired_symbols
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


def test_current_repository_retires_analytical_compound_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_compound_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/compoundoption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_lookback_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbol = "price_equity_fixed_lookback_option_analytical"

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol == helper_symbol
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/lookbackoption.py"
        and item.symbol == helper_symbol
    ]


def test_current_repository_retires_analytical_variance_swap_helper_authority():
    from trellis.agent.helper_authority_audit import build_helper_authority_report

    root = Path(__file__).resolve().parents[2]
    report = build_helper_authority_report(root)
    helper_symbols = {
        "price_equity_variance_swap_analytical",
        "equity_variance_swap_outputs_analytical",
    }

    assert not [
        item
        for item in (*report.route_authority, *report.binding_authority)
        if item.symbol in helper_symbols
    ]
    assert not [
        item
        for item in report.adapter_calls
        if item.path == "trellis/instruments/_agent/varianceswap.py"
        and item.symbol in helper_symbols
    ]
