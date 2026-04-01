from __future__ import annotations

import pytest

from trellis.agent.dsl_algebra import (
    AddExpr,
    ChoiceExpr,
    ContractAtom,
    ContractSignature,
    ContractUnit,
    ContractZero,
    ControlStyle,
    ScaleExpr,
    ThenExpr,
    choose_holder,
    choose_issuer,
    collect_primitive_refs,
    collect_control_styles,
    contract_signature,
    is_control_free,
    normalize_contract_expr,
    validate_contract_expr,
)
from trellis.core.types import TimelineRole


def _sig(
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    *,
    roles: tuple[TimelineRole, ...] = (),
    market: tuple[str, ...] = (),
) -> ContractSignature:
    return ContractSignature(
        inputs=inputs,
        outputs=outputs,
        timeline_roles=frozenset(roles),
        market_data_requirements=frozenset(market),
    )


def _atom(
    atom_id: str,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    *,
    roles: tuple[TimelineRole, ...] = (),
    market: tuple[str, ...] = (),
    primitive_ref: str | None = None,
) -> ContractAtom:
    return ContractAtom(
        atom_id=atom_id,
        signature=_sig(inputs, outputs, roles=roles, market=market),
        primitive_ref=primitive_ref,
    )


class TestSignatureRules:
    def test_additive_compatibility_requires_same_interface(self):
        left = _sig(("state",), ("cashflow",))
        right = _sig(("state",), ("cashflow",))
        wrong = _sig(("path",), ("cashflow",))

        assert left.additive_compatible(right)
        assert not left.additive_compatible(wrong)

    def test_sequential_compatibility_matches_ports(self):
        left = _sig(("state",), ("cashflow",))
        right = _sig(("cashflow",), ("pv",))
        wrong = _sig(("state",), ("pv",))

        assert left.sequential_compatible(right)
        assert not left.sequential_compatible(wrong)


class TestNormalization:
    def test_add_normalization_is_canonical(self):
        a = _atom("a", ("state",), ("cashflow",))
        b = _atom("b", ("state",), ("cashflow",))
        expr = AddExpr(
            terms=(
                b,
                AddExpr(terms=(a, ContractZero(signature=contract_signature(a)))),
            )
        )

        normalized = normalize_contract_expr(expr)
        assert normalized == AddExpr(terms=(a, b))

    def test_then_normalization_flattens_and_drops_unit(self):
        a = _atom("a", ("state",), ("cashflow",))
        b = _atom("b", ("cashflow",), ("pv",))
        expr = ThenExpr(
            terms=(
                ThenExpr(terms=(a, ContractUnit(ports=("cashflow",)))),
                b,
            )
        )

        normalized = normalize_contract_expr(expr)
        assert normalized == ThenExpr(terms=(a, b))

    def test_scale_normalization_handles_zero_and_one(self):
        a = _atom("a", ("state",), ("cashflow",))

        assert normalize_contract_expr(ScaleExpr(scalar=1, expr=a)) == a
        assert normalize_contract_expr(ScaleExpr(scalar=0, expr=a)) == ContractZero(
            signature=contract_signature(a)
        )

    def test_choice_normalization_preserves_control(self):
        a = _atom("a", ("state",), ("pv",))
        b = _atom("b", ("state",), ("pv",))
        expr = ChoiceExpr(
            style=ControlStyle.HOLDER_MAX,
            branches=(AddExpr(terms=(b, a)),),
            label="exercise",
        )

        normalized = normalize_contract_expr(expr)
        assert normalized == AddExpr(terms=(a, b))

    def test_multi_branch_choice_is_canonical_and_control_preserving(self):
        a = _atom("a", ("state",), ("pv",))
        b = _atom("b", ("state",), ("pv",))
        expr = ChoiceExpr(
            style=ControlStyle.ISSUER_MIN,
            branches=(b, a, b),
            label="call",
        )

        normalized = normalize_contract_expr(expr)
        assert normalized == ChoiceExpr(
            style=ControlStyle.ISSUER_MIN,
            branches=(a, b),
            label="call",
        )


class TestValidation:
    def test_add_rejects_signature_mismatch(self):
        expr = AddExpr(
            terms=(
                _atom("a", ("state",), ("pv",)),
                _atom("b", ("path",), ("pv",)),
            )
        )

        errors = validate_contract_expr(expr)
        assert len(errors) == 1
        assert "AddExpr signature mismatch" in errors[0]

    def test_then_rejects_sequential_mismatch(self):
        expr = ThenExpr(
            terms=(
                _atom("a", ("state",), ("cashflow",)),
                _atom("b", ("state",), ("pv",)),
            )
        )

        errors = validate_contract_expr(expr)
        assert len(errors) == 1
        assert "ThenExpr signature mismatch" in errors[0]

    def test_choice_rejects_branch_mismatch(self):
        expr = ChoiceExpr(
            style=ControlStyle.ISSUER_MIN,
            branches=(
                _atom("a", ("state",), ("pv",)),
                _atom("b", ("state",), ("cashflow",)),
            ),
        )

        errors = validate_contract_expr(expr)
        assert len(errors) == 1
        assert "ChoiceExpr branch mismatch" in errors[0]

    def test_choice_is_not_control_free(self):
        expr = ChoiceExpr(
            style=ControlStyle.HOLDER_MAX,
            branches=(
                _atom("continuation", ("state",), ("pv",)),
                _atom("exercise", ("state",), ("pv",)),
            ),
        )

        assert not is_control_free(expr)

    def test_choice_constructors_and_control_collection(self):
        expr = ThenExpr(
            terms=(
                choose_holder(
                    _atom("continuation", ("state",), ("pv",)),
                    _atom("exercise", ("state",), ("pv",)),
                    label="holder",
                ),
                choose_issuer(
                    _atom("issuer_cont", ("pv",), ("pv",)),
                    _atom("issuer_call", ("pv",), ("pv",)),
                    label="issuer",
                ),
            )
        )

        assert collect_control_styles(expr) == (
            ControlStyle.HOLDER_MAX,
            ControlStyle.ISSUER_MIN,
        )

    def test_linear_expression_is_control_free(self):
        expr = ThenExpr(
            terms=(
                _atom("coupon", ("state",), ("cashflow",)),
                ScaleExpr(
                    scalar=-1.0,
                    expr=_atom("discount", ("cashflow",), ("pv",)),
                ),
            )
        )

        assert is_control_free(expr)


class TestSignatureAggregation:
    def test_additive_signature_unions_metadata(self):
        left = _atom(
            "premium_leg",
            ("state",),
            ("pv",),
            roles=(TimelineRole.PAYMENT,),
            market=("discount_curve",),
        )
        right = _atom(
            "protection_leg",
            ("state",),
            ("pv",),
            roles=(TimelineRole.SETTLEMENT,),
            market=("credit_curve",),
        )

        signature = contract_signature(AddExpr(terms=(left, right)))
        assert signature.inputs == ("state",)
        assert signature.outputs == ("pv",)
        assert signature.timeline_roles == {
            TimelineRole.PAYMENT,
            TimelineRole.SETTLEMENT,
        }
        assert signature.market_data_requirements == {"discount_curve", "credit_curve"}

    def test_collect_primitive_refs_preserves_atoms(self):
        expr = ThenExpr(
            terms=(
                _atom(
                    "schedule",
                    (),
                    ("schedule",),
                    primitive_ref="trellis.core.date_utils.build_payment_timeline",
                ),
                ChoiceExpr(
                    style=ControlStyle.HOLDER_MAX,
                    branches=(
                        _atom(
                            "tree",
                            ("schedule",),
                            ("pv",),
                            primitive_ref="trellis.models.equity_option_tree.price_vanilla_equity_option_tree",
                        ),
                        _atom(
                            "pde",
                            ("schedule",),
                            ("pv",),
                            primitive_ref="trellis.models.equity_option_pde.price_vanilla_equity_option_pde",
                        ),
                    ),
                ),
            )
        )

        refs = collect_primitive_refs(expr)
        assert refs == (
            "trellis.core.date_utils.build_payment_timeline",
            "trellis.models.equity_option_tree.price_vanilla_equity_option_tree",
            "trellis.models.equity_option_pde.price_vanilla_equity_option_pde",
        )


def test_choice_normalization_rejects_incompatible_branches():
    expr = ChoiceExpr(
        style=ControlStyle.HOLDER_MAX,
        branches=(
            _atom("a", ("state",), ("pv",)),
            _atom("b", ("path",), ("pv",)),
        ),
    )

    with pytest.raises(ValueError, match="ChoiceExpr branch mismatch"):
        normalize_contract_expr(expr)
