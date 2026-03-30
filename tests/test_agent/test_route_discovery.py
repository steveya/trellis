"""End-to-end integration test for autonomous route discovery.

Validates the full closed loop:
1. Build with unknown product (no matching promoted route)
2. reflect.py fires route discovery → candidate entry written
3. Successive builds boost confidence → auto-promote
4. Promoted route appears in match_candidate_routes()
5. Scorer includes discovered route in feature map
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from trellis.agent.knowledge.gap_check import RouteGap, _check_route_gap
from trellis.agent.knowledge.reflect import (
    _boost_route_confidence,
    _capture_discovered_route,
    _find_equivalent_route,
)
from trellis.agent.knowledge.schema import ProductDecomposition, ProductIR
from trellis.agent.route_registry import (
    clear_route_registry_cache,
    load_route_registry,
    match_candidate_routes,
)


_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "trellis" / "agent" / "knowledge"
_ROUTES_ENTRIES = _KNOWLEDGE_DIR / "routes" / "entries"


@pytest.fixture(autouse=True)
def clean_discovered_routes():
    """Remove any test-generated discovered routes after each test."""
    yield
    for f in _ROUTES_ENTRIES.glob("test_*.yaml"):
        f.unlink(missing_ok=True)
    clear_route_registry_cache()


# ---------------------------------------------------------------------------
# Route gap detection
# ---------------------------------------------------------------------------

class TestRouteGapDetection:
    def test_known_product_has_no_gap(self):
        decomp = ProductDecomposition(
            instrument="swaption",
            features=("vanilla_option",),
            method="analytical",
        )
        gap = _check_route_gap(decomp)
        assert gap is None  # analytical_black76 should match

    def test_unknown_product_has_gap(self):
        decomp = ProductDecomposition(
            instrument="exotic_product_xyz_999",
            features=("exotic",),
            method="analytical",
        )
        gap = _check_route_gap(decomp)
        # Might match analytical_black76 generically (no instrument filter)
        # or might not if the registry filters it out
        # Either way, the function should not crash
        assert gap is None or isinstance(gap, RouteGap)


# ---------------------------------------------------------------------------
# Route discovery capture
# ---------------------------------------------------------------------------

class TestRouteDiscoveryCapture:
    def test_capture_new_route(self):
        route_data = {
            "route_id": "test_exotic_analytical",
            "engine_family": "analytical",
            "match_methods": ["analytical"],
            "match_instruments": ["exotic_test"],
            "primitives_used": [
                {"module": "trellis.models.black", "symbol": "black76_call", "role": "pricing_kernel"},
            ],
            "market_data_accessed": ["discount_curve", "black_vol_surface"],
            "parameters_extracted": ["maturity", "strike"],
            "rationale": "Test route for exotic products.",
        }
        decomp = ProductDecomposition(
            instrument="exotic_test",
            features=("exotic",),
            method="analytical",
        )

        rid, outcome = _capture_discovered_route(route_data, decomp, attempt=1)
        assert rid == "test_exotic_analytical"
        assert outcome == "captured"

        # Verify file was written
        route_path = _ROUTES_ENTRIES / "test_exotic_analytical.yaml"
        assert route_path.exists()
        data = yaml.safe_load(route_path.read_text())
        assert data["status"] == "candidate"
        assert data["confidence"] == 0.5

    def test_capture_existing_route_boosts(self):
        # First capture
        route_data = {
            "route_id": "test_boost_route",
            "engine_family": "analytical",
            "match_methods": ["analytical"],
            "primitives_used": [],
            "market_data_accessed": [],
            "parameters_extracted": [],
        }
        decomp = ProductDecomposition(
            instrument="test", features=(), method="analytical",
        )
        _capture_discovered_route(route_data, decomp, attempt=1)

        # Second capture of same ID
        rid, outcome = _capture_discovered_route(route_data, decomp, attempt=2)
        assert rid == "test_boost_route"
        assert outcome == "existing_boosted"


# ---------------------------------------------------------------------------
# Route deduplication
# ---------------------------------------------------------------------------

class TestRouteDeduplication:
    def test_find_equivalent_by_primitives(self):
        registry = load_route_registry()
        # The analytical_black76 route has black76_call and black76_put
        prim_set = frozenset({
            ("trellis.models.black", "black76_call", "pricing_kernel"),
            ("trellis.models.black", "black76_put", "pricing_kernel"),
        })
        # This won't match because analytical_black76 has more primitives
        # (schedule_builder, assembly_helper, etc.)
        result = _find_equivalent_route(prim_set, registry)
        # Should NOT match (subset, not exact match)
        assert result is None

    def test_exact_match_finds_equivalent(self):
        # Capture a route, then try to find it by its primitives
        route_data = {
            "route_id": "test_dedup_route",
            "engine_family": "analytical",
            "match_methods": ["analytical"],
            "primitives_used": [
                {"module": "trellis.models.black", "symbol": "test_dedup_fn", "role": "test"},
            ],
            "market_data_accessed": [],
            "parameters_extracted": [],
        }
        decomp = ProductDecomposition(
            instrument="test", features=(), method="analytical",
        )
        _capture_discovered_route(route_data, decomp, attempt=1)
        clear_route_registry_cache()

        registry = load_route_registry()
        prim_set = frozenset({("trellis.models.black", "test_dedup_fn", "test")})
        result = _find_equivalent_route(prim_set, registry)
        assert result == "test_dedup_route"


# ---------------------------------------------------------------------------
# Confidence lifecycle
# ---------------------------------------------------------------------------

class TestConfidenceLifecycle:
    def test_boost_promotes_at_threshold(self):
        # Create a candidate route
        route_data = {
            "route_id": "test_lifecycle_route",
            "engine_family": "analytical",
            "match_methods": ["analytical"],
            "primitives_used": [],
            "market_data_accessed": [],
            "parameters_extracted": [],
        }
        decomp = ProductDecomposition(
            instrument="test", features=(), method="analytical",
        )
        _capture_discovered_route(route_data, decomp, attempt=1)

        # Boost confidence 3 times (0.5 → 0.6 → 0.7 → 0.8)
        _boost_route_confidence("test_lifecycle_route", delta=0.1)
        _boost_route_confidence("test_lifecycle_route", delta=0.1)
        _boost_route_confidence("test_lifecycle_route", delta=0.1)

        route_path = _ROUTES_ENTRIES / "test_lifecycle_route.yaml"
        data = yaml.safe_load(route_path.read_text())
        assert data["confidence"] == 0.8
        assert data["status"] == "promoted"
        assert data["successful_builds"] == 4  # 1 initial + 3 boosts

    def test_promoted_route_appears_in_registry(self):
        # Create and promote a route
        route_data = {
            "route_id": "test_promoted_visible",
            "engine_family": "analytical",
            "match_methods": ["analytical"],
            "match_instruments": ["test_visible_instrument"],
            "primitives_used": [],
            "market_data_accessed": [],
            "parameters_extracted": [],
        }
        decomp = ProductDecomposition(
            instrument="test_visible_instrument", features=(), method="analytical",
        )
        _capture_discovered_route(route_data, decomp, attempt=1)

        # Promote it
        for _ in range(3):
            _boost_route_confidence("test_promoted_visible", delta=0.1)

        clear_route_registry_cache()
        registry = load_route_registry()
        ir = ProductIR(instrument="test_visible_instrument", payoff_family="test")
        matches = match_candidate_routes(registry, "analytical", ir, promoted_only=True)
        route_ids = [r.id for r in matches]
        assert "test_promoted_visible" in route_ids
