"""Architecture contracts for the internal FpML normalization boundary."""

from __future__ import annotations

import ast
from datetime import date
import importlib
import inspect
from pathlib import Path


FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "fpml"
    / "confirmation_5_13_fixed_float_swap.xml"
)

FORBIDDEN_SHARED_IMPORT_PREFIXES = (
    "trellis.agent.executor",
    "trellis.agent.knowledge",
    "trellis.agent.static_leg_admission",
    "trellis.agent.task_runtime",
    "trellis.instruments",
    "trellis.models",
)


def _imported_modules(module) -> tuple[str, ...]:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def test_shared_normalization_module_owns_product_neutral_helpers():
    shared = importlib.import_module("trellis.io.fpml._normalization_common")
    normalizer = importlib.import_module("trellis.io.fpml.normalizer")

    for helper_name in (
        "_adjustable_date",
        "_blocked_from",
        "_blocker",
        "_normalize_stream",
        "_provenance",
        "_validate_document_metadata",
    ):
        shared_helper = getattr(shared, helper_name)
        assert shared_helper.__module__ == shared.__name__
        assert getattr(normalizer, helper_name) is shared_helper

    for product_mapper_name in (
        "_normalize_cap_floor",
        "_normalize_european_swaption",
        "_normalize_fixed_float_swap",
    ):
        assert not hasattr(shared, product_mapper_name)


def test_shared_normalization_module_has_no_pricing_or_route_authority_imports():
    shared = importlib.import_module("trellis.io.fpml._normalization_common")

    violations = tuple(
        imported
        for imported in _imported_modules(shared)
        if imported.startswith(FORBIDDEN_SHARED_IMPORT_PREFIXES)
    )

    assert violations == ()


def test_internal_normalization_module_is_not_codegen_import_authority():
    from trellis.agent.knowledge.import_registry import get_import_registry

    assert "trellis.io.fpml._normalization_common" not in get_import_registry()


def test_public_and_inspected_document_facades_remain_field_compatible():
    from trellis.io.fpml import inspect_fpml_document, normalize_fpml_document
    from trellis.io.fpml.normalizer import _normalize_inspected_fpml_document

    xml = FIXTURE.read_bytes()
    inspected = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )

    direct = normalize_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id="PARTY-A",
        valuation_date=date(2025, 1, 15),
    )
    staged = _normalize_inspected_fpml_document(
        xml,
        inspected=inspected,
        valuation_party_id="PARTY-A",
        valuation_date=date(2025, 1, 15),
    )

    assert direct == staged
