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
CAP_FLOOR_FIXTURE = (
    Path(__file__).with_name("fixtures") / "fpml" / "confirmation_5_13_cap_floor.xml"
)
SWAPTION_FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "fpml"
    / "confirmation_5_13_european_swaption.xml"
)

FORBIDDEN_SHARED_IMPORT_PREFIXES = (
    "trellis.agent.executor",
    "trellis.agent.knowledge",
    "trellis.agent.static_leg_admission",
    "trellis.agent.task_runtime",
    "trellis.instruments",
    "trellis.models",
)
NORMALIZATION_MODULES = {
    "common": "trellis.io.fpml._normalization_common",
    "swap": "trellis.io.fpml._normalization_swap",
    "cap_floor": "trellis.io.fpml._normalization_cap_floor",
    "swaption": "trellis.io.fpml._normalization_swaption",
    "facade": "trellis.io.fpml.normalizer",
}


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


def test_normalizer_facade_defines_only_bounded_orchestration():
    normalizer = importlib.import_module(NORMALIZATION_MODULES["facade"])
    tree = ast.parse(inspect.getsource(normalizer))

    functions = tuple(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    normalization_imports = {
        imported
        for imported in _imported_modules(normalizer)
        if imported in set(NORMALIZATION_MODULES.values())
    }

    assert functions == (
        "normalize_fpml_document",
        "_normalize_inspected_fpml_document",
    )
    assert normalization_imports == {
        NORMALIZATION_MODULES["common"],
        NORMALIZATION_MODULES["swap"],
        NORMALIZATION_MODULES["cap_floor"],
        NORMALIZATION_MODULES["swaption"],
    }


def test_internal_normalization_dependency_graph_is_explicit_and_acyclic():
    expected_dependencies = {
        "common": set(),
        "swap": {NORMALIZATION_MODULES["common"]},
        "cap_floor": {NORMALIZATION_MODULES["common"]},
        "swaption": {
            NORMALIZATION_MODULES["common"],
            NORMALIZATION_MODULES["swap"],
        },
    }
    internal_module_names = set(NORMALIZATION_MODULES.values())

    for module_name, expected in expected_dependencies.items():
        module = importlib.import_module(NORMALIZATION_MODULES[module_name])
        dependencies = {
            imported
            for imported in _imported_modules(module)
            if imported in internal_module_names
        }
        assert dependencies == expected


def test_public_fpml_exports_keep_the_stable_normalization_facade():
    fpml = importlib.import_module("trellis.io.fpml")
    normalizer = importlib.import_module(NORMALIZATION_MODULES["facade"])

    assert fpml.__all__ == [
        "DEFAULT_FPML_INSPECTION_LIMITS",
        "FPML_5_13_CONFIRMATION",
        "SUPPORTED_FPML_PROFILES",
        "FpMLClarification",
        "FpMLDocumentIdentity",
        "FpMLFieldProvenance",
        "FpMLImportBlocker",
        "FpMLImportReport",
        "FpMLInspectionLimits",
        "FpMLPremiumMetadata",
        "FpMLProfile",
        "FpMLTradeIdentity",
        "fpml_import_report_summary",
        "inspect_fpml_document",
        "normalize_fpml_document",
    ]
    assert fpml.normalize_fpml_document is normalizer.normalize_fpml_document


def test_shared_normalization_module_owns_product_neutral_helpers():
    shared = importlib.import_module("trellis.io.fpml._normalization_common")
    normalizer = importlib.import_module("trellis.io.fpml.normalizer")

    for helper_name in (
        "_adjustable_date",
        "_blocked_from",
        "_blocker",
        "_normalize_option_premiums",
        "_normalize_stream",
        "_provenance",
        "_validate_document_metadata",
    ):
        shared_helper = getattr(shared, helper_name)
        assert shared_helper.__module__ == shared.__name__

    for facade_helper_name in (
        "_blocked_from",
        "_blocker",
        "_validate_document_metadata",
    ):
        shared_helper = getattr(shared, facade_helper_name)
        assert getattr(normalizer, facade_helper_name) is shared_helper

    for product_mapper_name in (
        "_normalize_cap_floor",
        "_normalize_european_swaption",
        "_normalize_fixed_float_swap",
    ):
        assert not hasattr(shared, product_mapper_name)


def test_swap_normalization_module_owns_fixed_float_mapping():
    normalizer = importlib.import_module("trellis.io.fpml.normalizer")
    swap_mapper = importlib.import_module("trellis.io.fpml._normalization_swap")

    for helper_name in (
        "_normalize_fixed_float_swap",
        "_reject_unresolved_swap_historical_fixings",
    ):
        mapper_helper = getattr(swap_mapper, helper_name)
        assert mapper_helper.__module__ == swap_mapper.__name__
        assert getattr(normalizer, helper_name) is mapper_helper

    for other_product_mapper_name in (
        "_normalize_cap_floor",
        "_normalize_european_swaption",
    ):
        assert not hasattr(swap_mapper, other_product_mapper_name)


def test_cap_floor_normalization_module_owns_strip_mapping():
    cap_floor_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_cap_floor"
    )
    normalizer = importlib.import_module("trellis.io.fpml.normalizer")

    mapper = cap_floor_mapper._normalize_cap_floor
    assert mapper.__module__ == cap_floor_mapper.__name__
    assert normalizer._normalize_cap_floor is mapper
    assert (
        cap_floor_mapper._normalize_cap_floor_strike_schedule.__module__
        == cap_floor_mapper.__name__
    )

    shared_premium_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_common"
    )._normalize_option_premiums
    assert (
        cap_floor_mapper._normalize_cap_floor.__globals__["_normalize_option_premiums"]
        is shared_premium_mapper
    )
    assert (
        normalizer._normalize_european_swaption.__globals__[
            "_normalize_option_premiums"
        ]
        is shared_premium_mapper
    )

    for other_product_mapper_name in (
        "_normalize_european_swaption",
        "_normalize_fixed_float_swap",
    ):
        assert not hasattr(cap_floor_mapper, other_product_mapper_name)


def test_swaption_normalization_module_owns_european_mapping():
    normalizer = importlib.import_module("trellis.io.fpml.normalizer")
    swaption_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_swaption"
    )

    mapper = swaption_mapper._normalize_european_swaption
    assert mapper.__module__ == swaption_mapper.__name__
    assert normalizer._normalize_european_swaption is mapper

    shared_premium_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_common"
    )._normalize_option_premiums
    swap_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_swap"
    )._normalize_fixed_float_swap
    assert mapper.__globals__["_normalize_option_premiums"] is shared_premium_mapper
    assert mapper.__globals__["_normalize_fixed_float_swap"] is swap_mapper

    assert not hasattr(swaption_mapper, "_normalize_cap_floor")


def test_shared_normalization_module_has_no_pricing_or_route_authority_imports():
    shared = importlib.import_module("trellis.io.fpml._normalization_common")

    violations = tuple(
        imported
        for imported in _imported_modules(shared)
        if imported.startswith(FORBIDDEN_SHARED_IMPORT_PREFIXES)
    )

    assert violations == ()


def test_swap_normalization_module_has_no_pricing_or_route_authority_imports():
    swap_mapper = importlib.import_module("trellis.io.fpml._normalization_swap")

    violations = tuple(
        imported
        for imported in _imported_modules(swap_mapper)
        if imported.startswith(FORBIDDEN_SHARED_IMPORT_PREFIXES)
    )

    assert violations == ()


def test_cap_floor_module_has_no_pricing_or_route_authority_imports():
    cap_floor_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_cap_floor"
    )

    violations = tuple(
        imported
        for imported in _imported_modules(cap_floor_mapper)
        if imported.startswith(FORBIDDEN_SHARED_IMPORT_PREFIXES)
    )

    assert violations == ()


def test_swaption_module_has_no_pricing_or_route_authority_imports():
    swaption_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_swaption"
    )

    violations = tuple(
        imported
        for imported in _imported_modules(swaption_mapper)
        if imported.startswith(FORBIDDEN_SHARED_IMPORT_PREFIXES)
    )

    assert violations == ()


def test_internal_normalization_module_is_not_codegen_import_authority():
    from trellis.agent.knowledge.import_registry import get_import_registry

    registry = get_import_registry()

    assert "trellis.io.fpml._normalization_common" not in registry
    assert "trellis.io.fpml._normalization_cap_floor" not in registry
    assert "trellis.io.fpml._normalization_swap" not in registry
    assert "trellis.io.fpml._normalization_swaption" not in registry


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


def test_direct_swap_mapper_matches_public_facade():
    from trellis.io.fpml import inspect_fpml_document, normalize_fpml_document
    from trellis.io.fpml.contracts import DEFAULT_FPML_INSPECTION_LIMITS
    from trellis.io.fpml.importer import (
        _bounded_parse,
        _content_bytes,
        _direct_children,
        _first_direct_child,
    )

    swap_mapper = importlib.import_module("trellis.io.fpml._normalization_swap")
    xml = FIXTURE.read_bytes()
    inspected = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )
    root = _bounded_parse(
        _content_bytes(xml),
        limits=DEFAULT_FPML_INSPECTION_LIMITS,
    )
    namespace = inspected.document.namespace
    trade = _direct_children(root, "trade", namespace=namespace)[0]
    swap = _first_direct_child(trade, "swap", namespace=namespace)

    contract, provenance = swap_mapper._normalize_fixed_float_swap(
        swap,
        namespace=namespace,
        valuation_party_id="PARTY-A",
        known_party_ids=inspected.trade.party_ids,
    )
    report = normalize_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id="PARTY-A",
    )

    assert contract == report.normalized_contract
    assert provenance == report.mapping_provenance
    assert (
        importlib.import_module(
            "trellis.io.fpml.normalizer"
        )._normalize_european_swaption.__globals__["_normalize_fixed_float_swap"]
        is swap_mapper._normalize_fixed_float_swap
    )


def test_direct_cap_floor_mapper_matches_source_neutral_public_facade():
    from trellis.io.fpml import inspect_fpml_document, normalize_fpml_document
    from trellis.io.fpml.contracts import DEFAULT_FPML_INSPECTION_LIMITS
    from trellis.io.fpml.importer import (
        _bounded_parse,
        _content_bytes,
        _direct_children,
        _first_direct_child,
    )

    cap_floor_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_cap_floor"
    )
    xml = CAP_FLOOR_FIXTURE.read_bytes()
    inspected = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )
    root = _bounded_parse(
        _content_bytes(xml),
        limits=DEFAULT_FPML_INSPECTION_LIMITS,
    )
    namespace = inspected.document.namespace
    trade = _direct_children(root, "trade", namespace=namespace)[0]
    cap_floor = _first_direct_child(trade, "capFloor", namespace=namespace)

    contract, provenance, premium_metadata = cap_floor_mapper._normalize_cap_floor(
        cap_floor,
        namespace=namespace,
        valuation_party_id="PARTY-A",
        valuation_date=date(2025, 1, 15),
        known_party_ids=inspected.trade.party_ids,
    )
    report = normalize_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id="PARTY-A",
        valuation_date=date(2025, 1, 15),
    )

    assert contract == report.normalized_contract
    assert provenance == report.mapping_provenance
    assert premium_metadata == report.premium_metadata
    assert contract.metadata["semantic_family"] == "period_rate_option_strip"


def test_direct_swaption_mapper_matches_public_facade_and_preserves_underlying():
    from trellis.io.fpml import inspect_fpml_document, normalize_fpml_document
    from trellis.io.fpml.contracts import DEFAULT_FPML_INSPECTION_LIMITS
    from trellis.io.fpml.importer import (
        _bounded_parse,
        _content_bytes,
        _direct_children,
        _first_direct_child,
    )

    swaption_mapper = importlib.import_module(
        "trellis.io.fpml._normalization_swaption"
    )
    xml = SWAPTION_FIXTURE.read_bytes()
    inspected = inspect_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
    )
    root = _bounded_parse(
        _content_bytes(xml),
        limits=DEFAULT_FPML_INSPECTION_LIMITS,
    )
    namespace = inspected.document.namespace
    trade = _direct_children(root, "trade", namespace=namespace)[0]
    swaption = _first_direct_child(trade, "swaption", namespace=namespace)

    contract, provenance, premium_metadata = (
        swaption_mapper._normalize_european_swaption(
            swaption,
            namespace=namespace,
            valuation_party_id="PARTY-A",
            valuation_date=date(2025, 1, 15),
            known_party_ids=inspected.trade.party_ids,
        )
    )
    report = normalize_fpml_document(
        xml,
        declared_view="confirmation",
        declared_version="5-13",
        valuation_party_id="PARTY-A",
        valuation_date=date(2025, 1, 15),
    )

    assert contract == report.normalized_contract
    assert provenance == report.mapping_provenance
    assert premium_metadata == report.premium_metadata
    assert contract.underlying_contract is not None
    assert contract.underlying_contract == report.normalized_contract.underlying_contract
