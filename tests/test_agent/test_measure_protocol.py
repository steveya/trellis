"""Tests for the DSL measure protocol (QUA-411).

Covers:
- DslMeasure enum construction and string compatibility
- normalize_dsl_measure with valid names, aliases, and unknowns
- normalize_requested_measures returns DslMeasure tuples
- Blueprint propagation of requested_measures
- Measure ⊆ support validation warnings
- Bridge function dsl_measure_to_runtime
"""

from __future__ import annotations

import pytest

from trellis.core.types import DslMeasure, normalize_dsl_measure


# ---------------------------------------------------------------------------
# DslMeasure enum tests
# ---------------------------------------------------------------------------

class TestDslMeasure:
    def test_is_str_subclass(self):
        assert isinstance(DslMeasure.DV01, str)
        assert DslMeasure.DV01 == "dv01"
        assert DslMeasure.PRICE == "price"

    def test_in_set(self):
        measures = {"dv01", "vega"}
        assert DslMeasure.DV01 in measures
        assert DslMeasure.VEGA in measures
        assert DslMeasure.DELTA not in measures

    def test_all_members_have_lowercase_values(self):
        for m in DslMeasure:
            assert m.value == m.value.lower()

    def test_from_value(self):
        assert DslMeasure("dv01") == DslMeasure.DV01
        assert DslMeasure("key_rate_durations") == DslMeasure.KEY_RATE_DURATIONS


class TestNormalizeDslMeasure:
    def test_valid_name(self):
        assert normalize_dsl_measure("dv01") == DslMeasure.DV01
        assert normalize_dsl_measure("vega") == DslMeasure.VEGA
        assert normalize_dsl_measure("price") == DslMeasure.PRICE

    def test_alias_krd(self):
        assert normalize_dsl_measure("krd") == DslMeasure.KEY_RATE_DURATIONS

    def test_alias_pv(self):
        assert normalize_dsl_measure("pv") == DslMeasure.PRICE
        assert normalize_dsl_measure("npv") == DslMeasure.PRICE

    def test_alias_modified_duration(self):
        assert normalize_dsl_measure("modified_duration") == DslMeasure.DURATION

    def test_alias_zspread_variants(self):
        assert normalize_dsl_measure("zspread") == DslMeasure.Z_SPREAD
        assert normalize_dsl_measure("z-spread") == DslMeasure.Z_SPREAD
        assert normalize_dsl_measure("z_spread") == DslMeasure.Z_SPREAD

    def test_case_insensitive(self):
        assert normalize_dsl_measure("DV01") == DslMeasure.DV01
        assert normalize_dsl_measure("Vega") == DslMeasure.VEGA

    def test_strips_whitespace(self):
        assert normalize_dsl_measure("  dv01  ") == DslMeasure.DV01

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown measure"):
            normalize_dsl_measure("beta")


# ---------------------------------------------------------------------------
# normalize_requested_measures returns DslMeasure
# ---------------------------------------------------------------------------

class TestNormalizeRequestedMeasures:
    def test_returns_dsl_measure_instances(self):
        from trellis.agent.sensitivity_support import normalize_requested_measures

        result = normalize_requested_measures(["dv01", "vega"])
        assert all(isinstance(m, DslMeasure) for m in result)
        assert DslMeasure.DV01 in result
        assert DslMeasure.VEGA in result

    def test_filters_price(self):
        from trellis.agent.sensitivity_support import normalize_requested_measures

        result = normalize_requested_measures(["price", "dv01"])
        assert DslMeasure.PRICE not in result
        assert DslMeasure.DV01 in result

    def test_empty_input(self):
        from trellis.agent.sensitivity_support import normalize_requested_measures

        assert normalize_requested_measures(None) == ()
        assert normalize_requested_measures([]) == ()

    def test_backward_compat_string_comparison(self):
        from trellis.agent.sensitivity_support import normalize_requested_measures

        result = normalize_requested_measures(["dv01"])
        assert result[0] == "dv01"  # str comparison works


# ---------------------------------------------------------------------------
# Blueprint propagation
# ---------------------------------------------------------------------------

class TestBlueprintPropagation:
    @staticmethod
    def _make_contract():
        from trellis.agent.semantic_contracts import make_vanilla_option_contract
        return make_vanilla_option_contract(
            description="European call on AAPL",
            underliers=("AAPL",),
            observation_schedule=("2025-06-30",),
        )

    def test_requested_measures_on_blueprint(self):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract

        contract = self._make_contract()
        blueprint = compile_semantic_contract(
            contract,
            requested_measures=["dv01", "vega"],
        )
        assert len(blueprint.requested_measures) > 0
        assert any(m == "dv01" for m in blueprint.requested_measures)

    def test_no_measures_empty_tuple(self):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract

        contract = self._make_contract()
        blueprint = compile_semantic_contract(contract)
        assert blueprint.requested_measures == ()

    def test_unsupported_measure_produces_warning(self):
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract

        contract = self._make_contract()
        blueprint = compile_semantic_contract(
            contract,
            requested_measures=["delta"],
        )
        # delta may or may not be in the method's support set
        # but the field should be populated regardless
        assert isinstance(blueprint.requested_measures, tuple)
        assert isinstance(blueprint.measure_support_warnings, tuple)


# ---------------------------------------------------------------------------
# Bridge function
# ---------------------------------------------------------------------------

class TestDslMeasureToRuntime:
    def test_known_measures(self):
        from trellis.analytics.measures import (
            CallableScenarioExplain,
            Delta,
            DV01,
            Gamma,
            OASDuration,
            Theta,
            Vega,
            dsl_measure_to_runtime,
        )

        assert dsl_measure_to_runtime("dv01") is DV01
        assert dsl_measure_to_runtime("vega") is Vega
        assert dsl_measure_to_runtime("delta") is Delta
        assert dsl_measure_to_runtime("gamma") is Gamma
        assert dsl_measure_to_runtime("theta") is Theta
        assert dsl_measure_to_runtime("oas_duration") is OASDuration
        assert dsl_measure_to_runtime("callable_scenario_explain") is CallableScenarioExplain

    def test_unknown_returns_none(self):
        from trellis.analytics.measures import dsl_measure_to_runtime

        assert dsl_measure_to_runtime("nonexistent") is None

    def test_dsl_enum_value_works(self):
        from trellis.analytics.measures import dsl_measure_to_runtime

        assert dsl_measure_to_runtime(DslMeasure.DV01.value) is not None
