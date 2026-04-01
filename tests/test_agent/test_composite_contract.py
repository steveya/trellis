"""Tests for the composition algebra DSL (QUA-439).

Covers:
- CompositeSemanticContract construction and validation
- DAG acyclicity detection
- Topological sort
- Market data unioning
- Compilation to CompositeBlueprint with data_flow
- Factory function (callable_bond_composite)
"""

from __future__ import annotations

import pytest

from trellis.agent.composite_contract import (
    CompositeBlueprint,
    CompositeSemanticContract,
    CompositeStep,
    ContractEdge,
    SubContractRef,
    callable_bond_composite,
    compile_composite_contract,
    topological_sort,
    union_market_data_requirements,
    validate_composite_contract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_composite() -> CompositeSemanticContract:
    """A → B → C linear chain."""
    return CompositeSemanticContract(
        composite_id="test_linear",
        description="A → B → C",
        sub_contracts=(
            SubContractRef(contract_id="a", contract=None, proven=True, primitive_ref="mod_a"),
            SubContractRef(contract_id="b", contract=None, proven=False),
            SubContractRef(contract_id="c", contract=None, proven=False),
        ),
        edges=(
            ContractEdge(from_contract="a", to_contract="b", data_flow=("x",)),
            ContractEdge(from_contract="b", to_contract="c", data_flow=("y",)),
        ),
        root_contract="a",
        terminal_contracts=("c",),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_linear_chain(self):
        errors = validate_composite_contract(_linear_composite())
        assert errors == ()

    def test_missing_root(self):
        c = CompositeSemanticContract(
            composite_id="bad",
            description="",
            sub_contracts=(SubContractRef(contract_id="a", contract=None),),
            edges=(),
            root_contract="nonexistent",
            terminal_contracts=("a",),
        )
        errors = validate_composite_contract(c)
        assert any("nonexistent" in e for e in errors)

    def test_missing_terminal(self):
        c = CompositeSemanticContract(
            composite_id="bad",
            description="",
            sub_contracts=(SubContractRef(contract_id="a", contract=None),),
            edges=(),
            root_contract="a",
            terminal_contracts=("nonexistent",),
        )
        errors = validate_composite_contract(c)
        assert any("nonexistent" in e for e in errors)

    def test_no_terminals(self):
        c = CompositeSemanticContract(
            composite_id="bad",
            description="",
            sub_contracts=(SubContractRef(contract_id="a", contract=None),),
            edges=(),
            root_contract="a",
            terminal_contracts=(),
        )
        errors = validate_composite_contract(c)
        assert any("No terminal" in e for e in errors)

    def test_undefined_edge_endpoint(self):
        c = CompositeSemanticContract(
            composite_id="bad",
            description="",
            sub_contracts=(
                SubContractRef(contract_id="a", contract=None),
                SubContractRef(contract_id="b", contract=None),
            ),
            edges=(ContractEdge(from_contract="a", to_contract="ghost"),),
            root_contract="a",
            terminal_contracts=("b",),
        )
        errors = validate_composite_contract(c)
        assert any("ghost" in e for e in errors)

    def test_cycle_detected(self):
        c = CompositeSemanticContract(
            composite_id="cyclic",
            description="",
            sub_contracts=(
                SubContractRef(contract_id="a", contract=None),
                SubContractRef(contract_id="b", contract=None),
            ),
            edges=(
                ContractEdge(from_contract="a", to_contract="b"),
                ContractEdge(from_contract="b", to_contract="a"),
            ),
            root_contract="a",
            terminal_contracts=("b",),
        )
        errors = validate_composite_contract(c)
        assert any("cycle" in e.lower() for e in errors)

    def test_unreachable_terminal(self):
        c = CompositeSemanticContract(
            composite_id="unreachable",
            description="",
            sub_contracts=(
                SubContractRef(contract_id="a", contract=None),
                SubContractRef(contract_id="b", contract=None),
                SubContractRef(contract_id="c", contract=None),
            ),
            edges=(ContractEdge(from_contract="a", to_contract="b"),),
            root_contract="a",
            terminal_contracts=("c",),  # c is unreachable
        )
        errors = validate_composite_contract(c)
        assert any("reachable" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_linear_order(self):
        order = topological_sort(_linear_composite())
        assert order.index("a") < order.index("b") < order.index("c")

    def test_single_node(self):
        c = CompositeSemanticContract(
            composite_id="single",
            description="",
            sub_contracts=(SubContractRef(contract_id="only", contract=None),),
            edges=(),
            root_contract="only",
            terminal_contracts=("only",),
        )
        assert topological_sort(c) == ("only",)

    def test_cycle_raises(self):
        c = CompositeSemanticContract(
            composite_id="cyclic",
            description="",
            sub_contracts=(
                SubContractRef(contract_id="a", contract=None),
                SubContractRef(contract_id="b", contract=None),
            ),
            edges=(
                ContractEdge(from_contract="a", to_contract="b"),
                ContractEdge(from_contract="b", to_contract="a"),
            ),
            root_contract="a",
            terminal_contracts=("b",),
        )
        with pytest.raises(ValueError, match="Cycle"):
            topological_sort(c)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

class TestCompilation:
    def test_linear_compiles(self):
        bp = compile_composite_contract(_linear_composite())
        assert isinstance(bp, CompositeBlueprint)
        assert len(bp.steps) == 3
        # First step is "a" (root, proven)
        assert bp.steps[0].contract_id == "a"
        assert bp.steps[0].proven is True
        assert bp.steps[0].primitive_ref == "mod_a"

    def test_data_flow_propagated(self):
        bp = compile_composite_contract(_linear_composite())
        step_b = next(s for s in bp.steps if s.contract_id == "b")
        step_c = next(s for s in bp.steps if s.contract_id == "c")
        assert "x" in step_b.consumes  # from a→b edge
        assert "y" in step_b.produces  # from b→c edge
        assert "y" in step_c.consumes  # from b→c edge

    def test_invalid_composite_raises(self):
        c = CompositeSemanticContract(
            composite_id="bad",
            description="",
            sub_contracts=(),
            edges=(),
            root_contract="nonexistent",
            terminal_contracts=(),
        )
        with pytest.raises(ValueError, match="Cannot compile"):
            compile_composite_contract(c)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCallableBondComposite:
    def test_is_valid(self):
        c = callable_bond_composite()
        errors = validate_composite_contract(c)
        assert errors == (), errors
        assert c.root_contract == "hw_calibration"
        assert "backward_induction" in c.terminal_contracts

    def test_compiles(self):
        c = callable_bond_composite()
        bp = compile_composite_contract(c)
        assert len(bp.steps) == 3
        # First step is calibration (proven)
        assert bp.steps[0].contract_id == "hw_calibration"
        assert bp.steps[0].proven is True

    def test_calibration_data_flows_to_cashflows(self):
        c = callable_bond_composite()
        bp = compile_composite_contract(c)
        cashflow_step = next(s for s in bp.steps if s.contract_id == "bond_cashflows")
        assert "calibrated_lattice" in cashflow_step.consumes

    def test_has_three_sub_contracts(self):
        c = callable_bond_composite()
        assert len(c.sub_contracts) == 3
        ids = {sc.contract_id for sc in c.sub_contracts}
        assert ids == {"hw_calibration", "bond_cashflows", "backward_induction"}


# ---------------------------------------------------------------------------
# Market data union
# ---------------------------------------------------------------------------

class TestMarketDataUnion:
    def test_callable_bond_has_data(self):
        c = callable_bond_composite()
        data = union_market_data_requirements(c)
        # The HW calibration contract has fitting instrument type "swaption"
        assert "swaption" in data

    def test_empty_composite(self):
        c = CompositeSemanticContract(
            composite_id="empty",
            description="",
            sub_contracts=(SubContractRef(contract_id="a", contract=None),),
            edges=(),
            root_contract="a",
            terminal_contracts=("a",),
        )
        data = union_market_data_requirements(c)
        assert isinstance(data, frozenset)
