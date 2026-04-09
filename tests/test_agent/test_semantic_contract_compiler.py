"""Tests for shared semantic-contract specialization in the compiler."""

from __future__ import annotations


def test_compiler_method_specialization_uses_shared_semantic_authority():
    from trellis.agent.semantic_contract_compiler import _specialize_contract_for_preferred_method
    from trellis.agent.semantic_contracts import (
        make_vanilla_option_contract,
        specialize_semantic_contract_for_method,
    )

    contract = make_vanilla_option_contract(
        description="European call on AAPL",
        underliers=("AAPL",),
        observation_schedule=("2025-11-15",),
        preferred_method="analytical",
    )

    specialized = _specialize_contract_for_preferred_method(contract, "rate_tree")
    expected = specialize_semantic_contract_for_method(
        contract,
        preferred_method="rate_tree",
    )

    assert specialized == expected
