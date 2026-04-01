"""Typed DSL algebra for contract composition.

This module gives the internal semantic DSL an executable algebraic core:

- a typed semiring-like fragment for control-free composition
- an explicit choice layer for holder/issuer optionality

It is intentionally small. The first slice provides normalization, basic
typing, and structure-preserving validation without yet replacing the full
semantic compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trellis.core.types import TimelineRole


class ControlStyle(str, Enum):
    """Explicit control operator used by the DSL."""

    HOLDER_MAX = "holder_max"
    ISSUER_MIN = "issuer_min"


@dataclass(frozen=True)
class ContractSignature:
    """Typed interface and capability summary for one contract fragment."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    timeline_roles: frozenset[TimelineRole] = frozenset()
    market_data_requirements: frozenset[str] = frozenset()

    def additive_compatible(self, other: "ContractSignature") -> bool:
        """Return whether two signatures can be superposed."""
        return self.inputs == other.inputs and self.outputs == other.outputs

    def sequential_compatible(self, other: "ContractSignature") -> bool:
        """Return whether the left output can feed the right input."""
        return self.outputs == other.inputs

    def merge_add(self, other: "ContractSignature") -> "ContractSignature":
        """Merge two additively compatible signatures."""
        if not self.additive_compatible(other):
            raise ValueError(
                "Additive composition requires matching inputs and outputs: "
                f"{self.inputs}->{self.outputs} vs {other.inputs}->{other.outputs}"
            )
        return ContractSignature(
            inputs=self.inputs,
            outputs=self.outputs,
            timeline_roles=self.timeline_roles | other.timeline_roles,
            market_data_requirements=(
                self.market_data_requirements | other.market_data_requirements
            ),
        )

    def compose(self, other: "ContractSignature") -> "ContractSignature":
        """Compose two signatures sequentially."""
        if not self.sequential_compatible(other):
            raise ValueError(
                "Sequential composition requires left outputs to match right inputs: "
                f"{self.outputs} vs {other.inputs}"
            )
        return ContractSignature(
            inputs=self.inputs,
            outputs=other.outputs,
            timeline_roles=self.timeline_roles | other.timeline_roles,
            market_data_requirements=(
                self.market_data_requirements | other.market_data_requirements
            ),
        )


@dataclass(frozen=True)
class ContractAtom:
    """Atomic DSL term that can later lower to a helper or primitive."""

    atom_id: str
    signature: ContractSignature = field(default_factory=ContractSignature)
    primitive_ref: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ContractZero:
    """Additive identity for a given interface."""

    signature: ContractSignature = field(default_factory=ContractSignature)


@dataclass(frozen=True)
class ContractUnit:
    """Sequential identity on one interface."""

    ports: tuple[str, ...] = ()

    @property
    def signature(self) -> ContractSignature:
        return ContractSignature(inputs=self.ports, outputs=self.ports)


@dataclass(frozen=True)
class ScaleExpr:
    """Scaled linear expression."""

    scalar: float
    expr: "ContractExpr"


@dataclass(frozen=True)
class AddExpr:
    """Portfolio superposition of compatible fragments."""

    terms: tuple["ContractExpr", ...]


@dataclass(frozen=True)
class ThenExpr:
    """Sequential composition of compatible fragments."""

    terms: tuple["ContractExpr", ...]


@dataclass(frozen=True)
class ChoiceExpr:
    """Bellman-style control operator over compatible branches."""

    style: ControlStyle
    branches: tuple["ContractExpr", ...]
    label: str = ""


ContractExpr = ContractAtom | ContractZero | ContractUnit | ScaleExpr | AddExpr | ThenExpr | ChoiceExpr


def choose_holder(*branches: ContractExpr, label: str = "") -> ChoiceExpr:
    """Construct a holder-maximizing control node."""
    return ChoiceExpr(style=ControlStyle.HOLDER_MAX, branches=tuple(branches), label=label)


def choose_issuer(*branches: ContractExpr, label: str = "") -> ChoiceExpr:
    """Construct an issuer-minimizing control node."""
    return ChoiceExpr(style=ControlStyle.ISSUER_MIN, branches=tuple(branches), label=label)


def contract_signature(expr: ContractExpr) -> ContractSignature:
    """Return the signature induced by *expr*."""
    if isinstance(expr, ContractAtom):
        return expr.signature
    if isinstance(expr, ContractZero):
        return expr.signature
    if isinstance(expr, ContractUnit):
        return expr.signature
    if isinstance(expr, ScaleExpr):
        return contract_signature(expr.expr)
    if isinstance(expr, AddExpr):
        if not expr.terms:
            raise ValueError("AddExpr requires at least one term to infer a signature")
        signature = contract_signature(expr.terms[0])
        for term in expr.terms[1:]:
            signature = signature.merge_add(contract_signature(term))
        return signature
    if isinstance(expr, ThenExpr):
        if not expr.terms:
            raise ValueError("ThenExpr requires at least one term to infer a signature")
        signature = contract_signature(expr.terms[0])
        for term in expr.terms[1:]:
            signature = signature.compose(contract_signature(term))
        return signature
    if isinstance(expr, ChoiceExpr):
        if not expr.branches:
            raise ValueError("ChoiceExpr requires at least one branch")
        signature = contract_signature(expr.branches[0])
        for branch in expr.branches[1:]:
            signature = signature.merge_add(contract_signature(branch))
        return signature
    raise TypeError(f"Unsupported contract expression: {type(expr)!r}")


def is_control_free(expr: ContractExpr) -> bool:
    """Return whether *expr* stays inside the linear semiring fragment."""
    if isinstance(expr, ChoiceExpr):
        return False
    if isinstance(expr, ScaleExpr):
        return is_control_free(expr.expr)
    if isinstance(expr, AddExpr):
        return all(is_control_free(term) for term in expr.terms)
    if isinstance(expr, ThenExpr):
        return all(is_control_free(term) for term in expr.terms)
    return True


def collect_control_styles(expr: ContractExpr) -> tuple[ControlStyle, ...]:
    """Return the distinct control styles present in *expr* in traversal order."""
    styles: list[ControlStyle] = []
    seen: set[ControlStyle] = set()

    def _visit(node: ContractExpr) -> None:
        if isinstance(node, ChoiceExpr):
            if node.style not in seen:
                styles.append(node.style)
                seen.add(node.style)
            for branch in node.branches:
                _visit(branch)
            return
        if isinstance(node, ScaleExpr):
            _visit(node.expr)
            return
        if isinstance(node, AddExpr):
            for term in node.terms:
                _visit(term)
            return
        if isinstance(node, ThenExpr):
            for term in node.terms:
                _visit(term)

    _visit(expr)
    return tuple(styles)


def collect_primitive_refs(expr: ContractExpr) -> tuple[str, ...]:
    """Return stable primitive references mentioned by *expr*."""
    refs: list[str] = []

    def _visit(node: ContractExpr) -> None:
        if isinstance(node, ContractAtom):
            if node.primitive_ref:
                refs.append(node.primitive_ref)
            return
        if isinstance(node, ScaleExpr):
            _visit(node.expr)
            return
        if isinstance(node, AddExpr):
            for term in node.terms:
                _visit(term)
            return
        if isinstance(node, ThenExpr):
            for term in node.terms:
                _visit(term)
            return
        if isinstance(node, ChoiceExpr):
            for branch in node.branches:
                _visit(branch)

    _visit(expr)
    return tuple(refs)


def validate_contract_expr(expr: ContractExpr) -> tuple[str, ...]:
    """Return structural typing errors for *expr*."""
    errors: list[str] = []

    def _walk(node: ContractExpr) -> ContractSignature | None:
        if isinstance(node, ContractAtom):
            return node.signature
        if isinstance(node, ContractZero):
            return node.signature
        if isinstance(node, ContractUnit):
            return node.signature
        if isinstance(node, ScaleExpr):
            return _walk(node.expr)
        if isinstance(node, AddExpr):
            if not node.terms:
                errors.append("AddExpr must contain at least one term")
                return None
            first = _walk(node.terms[0])
            if first is None:
                return None
            merged = first
            for term in node.terms[1:]:
                next_signature = _walk(term)
                if next_signature is None:
                    continue
                if not merged.additive_compatible(next_signature):
                    errors.append(
                        "AddExpr signature mismatch: "
                        f"{merged.inputs}->{merged.outputs} vs "
                        f"{next_signature.inputs}->{next_signature.outputs}"
                    )
                    continue
                merged = merged.merge_add(next_signature)
            return merged
        if isinstance(node, ThenExpr):
            if not node.terms:
                errors.append("ThenExpr must contain at least one term")
                return None
            first = _walk(node.terms[0])
            if first is None:
                return None
            composed = first
            for term in node.terms[1:]:
                next_signature = _walk(term)
                if next_signature is None:
                    continue
                if not composed.sequential_compatible(next_signature):
                    errors.append(
                        "ThenExpr signature mismatch: "
                        f"{composed.outputs} does not feed {next_signature.inputs}"
                    )
                    continue
                composed = composed.compose(next_signature)
            return composed
        if isinstance(node, ChoiceExpr):
            if not node.branches:
                errors.append("ChoiceExpr must contain at least one branch")
                return None
            first = _walk(node.branches[0])
            if first is None:
                return None
            merged = first
            for branch in node.branches[1:]:
                next_signature = _walk(branch)
                if next_signature is None:
                    continue
                if not merged.additive_compatible(next_signature):
                    errors.append(
                        "ChoiceExpr branch mismatch: "
                        f"{merged.inputs}->{merged.outputs} vs "
                        f"{next_signature.inputs}->{next_signature.outputs}"
                    )
                    continue
                merged = merged.merge_add(next_signature)
            return merged
        errors.append(f"Unsupported expression node: {type(node)!r}")
        return None

    _walk(expr)
    return tuple(errors)


def normalize_contract_expr(expr: ContractExpr) -> ContractExpr:
    """Normalize a contract expression without erasing control structure."""
    if isinstance(expr, (ContractAtom, ContractZero, ContractUnit)):
        return expr

    if isinstance(expr, ScaleExpr):
        normalized = normalize_contract_expr(expr.expr)
        if expr.scalar == 0:
            return ContractZero(signature=contract_signature(normalized))
        if expr.scalar == 1:
            return normalized
        if isinstance(normalized, ScaleExpr):
            return normalize_contract_expr(
                ScaleExpr(scalar=expr.scalar * normalized.scalar, expr=normalized.expr)
            )
        return ScaleExpr(scalar=expr.scalar, expr=normalized)

    if isinstance(expr, AddExpr):
        normalized_terms: list[ContractExpr] = []
        for term in expr.terms:
            normalized = normalize_contract_expr(term)
            if isinstance(normalized, AddExpr):
                normalized_terms.extend(normalized.terms)
                continue
            if isinstance(normalized, ContractZero):
                continue
            normalized_terms.append(normalized)
        if not normalized_terms:
            raise ValueError("Cannot normalize AddExpr to an untyped zero term")
        if len(normalized_terms) == 1:
            return normalized_terms[0]
        normalized_terms.sort(key=_normal_form_key)
        result = AddExpr(terms=tuple(normalized_terms))
        errors = validate_contract_expr(result)
        if errors:
            raise ValueError("; ".join(errors))
        return result

    if isinstance(expr, ThenExpr):
        normalized_terms: list[ContractExpr] = []
        for term in expr.terms:
            normalized = normalize_contract_expr(term)
            if isinstance(normalized, ThenExpr):
                normalized_terms.extend(normalized.terms)
                continue
            if isinstance(normalized, ContractUnit):
                continue
            normalized_terms.append(normalized)
        if not normalized_terms:
            raise ValueError("Cannot normalize ThenExpr to an untyped unit term")
        if len(normalized_terms) == 1:
            return normalized_terms[0]
        result = ThenExpr(terms=tuple(normalized_terms))
        errors = validate_contract_expr(result)
        if errors:
            raise ValueError("; ".join(errors))
        return result

    if isinstance(expr, ChoiceExpr):
        normalized_branches = tuple(
            normalize_contract_expr(branch) for branch in expr.branches
        )
        deduped: list[ContractExpr] = []
        seen_keys: set[tuple[str, ...]] = set()
        for branch in normalized_branches:
            key = _normal_form_key(branch)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(branch)
        deduped.sort(key=_normal_form_key)
        normalized_branches = tuple(deduped)
        if len(normalized_branches) == 1:
            return normalized_branches[0]
        result = ChoiceExpr(style=expr.style, branches=normalized_branches, label=expr.label)
        errors = validate_contract_expr(result)
        if errors:
            raise ValueError("; ".join(errors))
        return result

    raise TypeError(f"Unsupported contract expression: {type(expr)!r}")


def _normal_form_key(expr: ContractExpr) -> tuple[str, ...]:
    """Return a stable key for additive canonicalization."""
    if isinstance(expr, ContractAtom):
        return ("atom", expr.atom_id)
    if isinstance(expr, ContractZero):
        return ("zero", repr(expr.signature))
    if isinstance(expr, ContractUnit):
        return ("unit", repr(expr.ports))
    if isinstance(expr, ScaleExpr):
        return ("scale", repr(expr.scalar), *_normal_form_key(expr.expr))
    if isinstance(expr, AddExpr):
        key: list[str] = ["add"]
        for term in expr.terms:
            key.extend(_normal_form_key(term))
        return tuple(key)
    if isinstance(expr, ThenExpr):
        key = ["then"]
        for term in expr.terms:
            key.extend(_normal_form_key(term))
        return tuple(key)
    if isinstance(expr, ChoiceExpr):
        key = [f"choice:{expr.style.value}"]
        for branch in expr.branches:
            key.extend(_normal_form_key(branch))
        return tuple(key)
    return (repr(type(expr)),)
