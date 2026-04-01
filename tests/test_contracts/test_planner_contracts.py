"""Tier 2 contract tests: planner spec schemas for canary tasks (QUA-427).

These tests verify that the planner produces correct spec schemas for known
instrument types.  All use static specs — no LLM, no cassettes.
"""

from __future__ import annotations

import pytest

from trellis.agent.planner import STATIC_SPECS, SPECIALIZED_SPECS, SpecSchema, plan_build


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_static_spec(instrument_type: str) -> SpecSchema | None:
    """Look up a static spec schema by instrument type."""
    key = instrument_type.lower().replace(" ", "_")
    return STATIC_SPECS.get(key) or SPECIALIZED_SPECS.get(key)


# ---------------------------------------------------------------------------
# T38 — CDS spec schema
# ---------------------------------------------------------------------------

class TestT38PlannerSpec:
    """T38: CDS must have a complete spec schema with credit fields."""

    @pytest.mark.tier2
    def test_cds_spec_exists(self):
        spec = _get_static_spec("cds")
        assert spec is not None, "CDS must have a static spec schema"

    @pytest.mark.tier2
    def test_cds_class_name(self):
        spec = _get_static_spec("cds")
        assert spec is not None
        assert "CDS" in spec.class_name or "Cds" in spec.class_name, (
            f"CDS class name should contain 'CDS', got {spec.class_name}"
        )

    @pytest.mark.tier2
    def test_cds_has_required_fields(self):
        spec = _get_static_spec("cds")
        assert spec is not None
        field_names = {f.name for f in spec.fields}
        # CDS must have credit-related fields
        assert any("hazard" in name or "spread" in name or "recovery" in name
                    for name in field_names), (
            f"CDS spec should have credit-related fields, got: {field_names}"
        )

    @pytest.mark.tier2
    def test_cds_has_notional(self):
        spec = _get_static_spec("cds")
        assert spec is not None
        field_names = {f.name for f in spec.fields}
        assert "notional" in field_names, (
            f"CDS spec should have notional field, got: {field_names}"
        )

    @pytest.mark.tier2
    def test_cds_spread_field_calls_out_decimal_units(self):
        spec = _get_static_spec("cds")
        assert spec is not None
        spread_field = next(f for f in spec.fields if f.name == "spread")
        assert "0.015" in spread_field.description
        assert "150bps" in spread_field.description


# ---------------------------------------------------------------------------
# Callable bond spec schema
# ---------------------------------------------------------------------------

class TestCallableBondPlannerSpec:
    """Callable bond (T02, T17) must have rate and callable fields."""

    @pytest.mark.tier2
    def test_callable_bond_spec_exists(self):
        spec = _get_static_spec("callable_bond")
        assert spec is not None, "Callable bond must have a static spec schema"

    @pytest.mark.tier2
    def test_callable_bond_class_name(self):
        spec = _get_static_spec("callable_bond")
        assert spec is not None
        assert "Callable" in spec.class_name, (
            f"Class name should contain 'Callable', got {spec.class_name}"
        )

    @pytest.mark.tier2
    def test_callable_bond_has_call_schedule_or_dates(self):
        spec = _get_static_spec("callable_bond")
        assert spec is not None
        field_names = {f.name for f in spec.fields}
        assert any("call" in name for name in field_names), (
            f"Callable bond should have call-related fields, got: {field_names}"
        )


# ---------------------------------------------------------------------------
# CDO tranche spec schema (T49)
# ---------------------------------------------------------------------------

class TestT49PlannerSpec:
    """T49: CDO tranche must have correlation and tranche fields."""

    @pytest.mark.tier2
    def test_cdo_spec_exists(self):
        spec = _get_static_spec("cdo")
        assert spec is not None, "CDO must have a static spec schema"

    @pytest.mark.tier2
    def test_cdo_has_tranche_fields(self):
        spec = _get_static_spec("cdo")
        assert spec is not None
        field_names = {f.name for f in spec.fields}
        assert any("attach" in name or "detach" in name or "tranche" in name
                    for name in field_names), (
            f"CDO should have tranche fields, got: {field_names}"
        )


# ---------------------------------------------------------------------------
# Swaption spec schema (T73)
# ---------------------------------------------------------------------------

class TestT73PlannerSpec:
    """T73: Swaption must have vol and expiry fields."""

    @pytest.mark.tier2
    def test_swaption_spec_exists(self):
        spec = _get_static_spec("swaption")
        assert spec is not None, "Swaption must have a static spec schema"

    @pytest.mark.tier2
    def test_swaption_has_expiry(self):
        spec = _get_static_spec("swaption")
        assert spec is not None
        field_names = {f.name for f in spec.fields}
        assert any("expiry" in name or "maturity" in name or "exercise" in name
                    for name in field_names), (
            f"Swaption should have expiry-related fields, got: {field_names}"
        )


# ---------------------------------------------------------------------------
# Cross-cutting spec contracts
# ---------------------------------------------------------------------------

class TestSpecSchemaInvariants:
    """All static specs must satisfy structural invariants."""

    @pytest.mark.tier2
    def test_all_specs_have_class_name(self):
        for key, spec in STATIC_SPECS.items():
            assert spec.class_name, f"STATIC_SPECS[{key}] has empty class_name"

    @pytest.mark.tier2
    def test_all_specs_have_spec_name(self):
        for key, spec in STATIC_SPECS.items():
            assert spec.spec_name, f"STATIC_SPECS[{key}] has empty spec_name"

    @pytest.mark.tier2
    def test_all_specs_have_at_least_one_field(self):
        for key, spec in STATIC_SPECS.items():
            assert len(spec.fields) >= 1, (
                f"STATIC_SPECS[{key}] has no fields"
            )

    @pytest.mark.tier2
    def test_all_field_names_are_valid_python_identifiers(self):
        for key, spec in STATIC_SPECS.items():
            for f in spec.fields:
                assert f.name.isidentifier(), (
                    f"STATIC_SPECS[{key}] field '{f.name}' is not a valid identifier"
                )

    @pytest.mark.tier2
    def test_all_field_types_are_non_empty(self):
        for key, spec in STATIC_SPECS.items():
            for f in spec.fields:
                assert f.type, (
                    f"STATIC_SPECS[{key}] field '{f.name}' has empty type"
                )

    @pytest.mark.tier2
    def test_canary_instrument_types_have_specs(self):
        """At least the key instrument types used by canary tasks have specs."""
        expected = {"cds", "callable_bond", "swaption", "cdo"}
        for inst_type in expected:
            assert _get_static_spec(inst_type) is not None, (
                f"Canary instrument type '{inst_type}' has no static spec"
            )
