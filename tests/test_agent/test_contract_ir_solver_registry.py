"""Tests for the Phase 3 structural solver declaration substrate (QUA-925)."""

from __future__ import annotations

import pytest

from trellis.agent.contract_ir_solver_registry import (
    ContractIRSolverDeclaration,
    ContractIRSolverMaterialization,
    ContractIRSolverOutputSupport,
    ContractIRSolverOverlapError,
    ContractIRSolverProvenance,
    ContractIRSolverRegistryError,
    ContractIRSolverSelectionAuthority,
    build_contract_ir_solver_registry,
)
from trellis.agent.contract_pattern import parse_contract_pattern


def _declaration(
    declaration_id: str,
    *,
    payoff_kind: str,
    precedence: int = 0,
    methods: tuple[str, ...] = ("analytical",),
    exercise_style: str | None = None,
    subordinates_to: tuple[str, ...] = (),
    payload: dict[str, object] | None = None,
) -> ContractIRSolverDeclaration:
    payload = dict(payload or {"payoff": {"kind": payoff_kind}})
    if exercise_style is not None:
        payload["exercise"] = {"style": exercise_style}
    return ContractIRSolverDeclaration(
        authority=ContractIRSolverSelectionAuthority(
            contract_pattern=parse_contract_pattern(payload),
            admissible_methods=methods,
        ),
        outputs=ContractIRSolverOutputSupport(supported_outputs=("price",)),
        materialization=ContractIRSolverMaterialization(
            callable_ref=f"trellis.models.synthetic.{declaration_id}",
            call_style="helper_call",
        ),
        provenance=ContractIRSolverProvenance(declaration_id=declaration_id),
        precedence=precedence,
        subordinates_to=subordinates_to,
    )


class TestContractIRSolverRegistry:
    def test_selection_order_uses_precedence_then_registration_order(self):
        low_a = _declaration("low_a", payoff_kind="vanilla_payoff", precedence=10)
        low_b = _declaration("low_b", payoff_kind="digital_payoff", precedence=10)
        high = _declaration("high", payoff_kind="swaption_payoff", precedence=20)

        registry = build_contract_ir_solver_registry((low_a, low_b, high))

        assert tuple(item.declaration_id for item in registry.selection_order()) == (
            "high",
            "low_a",
            "low_b",
        )
        assert tuple(item.registration_index for item in registry.declarations) == (0, 1, 2)

    def test_equal_precedence_overlapping_declarations_fail_closed(self):
        left = _declaration("left", payoff_kind="vanilla_payoff", precedence=10)
        right = _declaration("right", payoff_kind="vanilla_payoff", precedence=10)

        with pytest.raises(ContractIRSolverOverlapError, match="left.*right|right.*left"):
            build_contract_ir_solver_registry((left, right))

    def test_instrument_tag_and_structural_vanilla_overlap_fail_closed(self):
        tag_decl = _declaration("tag", payoff_kind="vanilla_payoff", precedence=10)
        structural_decl = _declaration(
            "structural",
            payoff_kind="unused",
            precedence=10,
            payload={
                "payoff": {
                    "kind": "max",
                    "args": [
                        {
                            "kind": "sub",
                            "args": [
                                {"kind": "spot", "underlier": "_u"},
                                {"kind": "strike", "value": "_k"},
                            ],
                        },
                        {"kind": "constant", "value": 0.0},
                    ],
                }
            },
        )

        with pytest.raises(ContractIRSolverOverlapError, match="tag.*structural|structural.*tag"):
            build_contract_ir_solver_registry((tag_decl, structural_decl))

    def test_subordinate_overlap_is_allowed_when_precedence_is_lower(self):
        specific = _declaration("specific", payoff_kind="vanilla_payoff", precedence=20)
        general = _declaration(
            "general",
            payoff_kind="vanilla_payoff",
            precedence=10,
            subordinates_to=("specific",),
        )

        registry = build_contract_ir_solver_registry((general, specific))

        assert tuple(item.declaration_id for item in registry.selection_order()) == (
            "specific",
            "general",
        )

    def test_subordinate_overlap_requires_strictly_lower_precedence(self):
        specific = _declaration("specific", payoff_kind="vanilla_payoff", precedence=20)
        general = _declaration(
            "general",
            payoff_kind="vanilla_payoff",
            precedence=20,
            subordinates_to=("specific",),
        )

        with pytest.raises(ContractIRSolverRegistryError, match="general.*specific.*precedence"):
            build_contract_ir_solver_registry((general, specific))

    def test_unknown_subordinate_target_is_rejected(self):
        declaration = _declaration(
            "orphan",
            payoff_kind="vanilla_payoff",
            subordinates_to=("missing",),
        )

        with pytest.raises(ContractIRSolverRegistryError, match="missing"):
            build_contract_ir_solver_registry((declaration,))

    def test_distinct_literal_exercise_styles_are_not_treated_as_overlapping(self):
        european = _declaration(
            "european_vanilla",
            payoff_kind="vanilla_payoff",
            precedence=10,
            exercise_style="european",
        )
        american = _declaration(
            "american_vanilla",
            payoff_kind="vanilla_payoff",
            precedence=10,
            exercise_style="american",
        )

        registry = build_contract_ir_solver_registry((european, american))

        assert tuple(item.declaration_id for item in registry.selection_order()) == (
            "european_vanilla",
            "american_vanilla",
        )
