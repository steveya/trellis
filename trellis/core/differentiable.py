"""Differentiable backend boundary.

The public pricing stack should depend on this module rather than importing an
automatic-differentiation library directly.  The current backend is
``autograd``; this module also publishes the operators Trellis expects so a
future backend or portfolio-AAD implementation has a stable compatibility
surface to target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import autograd
import autograd.numpy as anp


_SUPPORTED_OPERATOR_NAMES = frozenset(
    {
        "grad",
        "jacobian",
        "hessian",
        "jvp",
        "vjp",
        "hessian_vector_product",
        "portfolio_aad",
    }
)


@dataclass(frozen=True)
class DifferentiableBackendCapabilities:
    """Capability declaration for the active differentiable backend."""

    backend_id: str
    array_namespace: str
    operators: Mapping[str, bool]
    notes: tuple[str, ...] = ()
    unsupported_reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.operators) - _SUPPORTED_OPERATOR_NAMES
        missing = _SUPPORTED_OPERATOR_NAMES - set(self.operators)
        if unknown:
            raise ValueError(
                f"unknown differentiable backend capabilities: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(f"missing differentiable backend capabilities: {sorted(missing)}")
        reason_keys = set(self.unsupported_reasons)
        unknown_reasons = reason_keys - _SUPPORTED_OPERATOR_NAMES
        if unknown_reasons:
            raise ValueError(
                f"unknown differentiable backend unsupported reasons: {sorted(unknown_reasons)}"
            )
        object.__setattr__(self, "operators", MappingProxyType(dict(self.operators)))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(
            self,
            "unsupported_reasons",
            MappingProxyType(
                {
                    str(operator): str(reason)
                    for operator, reason in self.unsupported_reasons.items()
                    if str(reason)
                }
            ),
        )

    def supports(self, capability: str) -> bool:
        """Return whether the backend supports a named differentiable operator."""
        if capability not in _SUPPORTED_OPERATOR_NAMES:
            raise ValueError(f"unknown differentiable backend capability: {capability}")
        return bool(self.operators[capability])

    def operator_support(self, capability: str) -> dict[str, Any]:
        """Return a JSON-friendly support record for one backend operator."""
        if capability not in _SUPPORTED_OPERATOR_NAMES:
            raise ValueError(f"unknown differentiable backend capability: {capability}")
        supported = bool(self.operators[capability])
        return {
            "operator": capability,
            "supported": supported,
            "backend_id": self.backend_id,
            "array_namespace": self.array_namespace,
            "unsupported_reason": (
                None if supported else self.unsupported_reasons.get(capability)
            ),
        }

    @property
    def support_matrix(self) -> dict[str, dict[str, Any]]:
        """Return stable operator support records keyed by operator name."""
        return {
            name: self.operator_support(name)
            for name in sorted(_SUPPORTED_OPERATOR_NAMES)
        }

    @property
    def supported_operators(self) -> tuple[str, ...]:
        """Return supported operator names in stable sorted order."""
        return tuple(sorted(name for name, supported in self.operators.items() if supported))

    @property
    def unsupported_operators(self) -> tuple[str, ...]:
        """Return unsupported operator names in stable sorted order."""
        return tuple(sorted(name for name, supported in self.operators.items() if not supported))

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-friendly capability payload."""
        return {
            "backend_id": self.backend_id,
            "array_namespace": self.array_namespace,
            "operators": dict(self.operators),
            "supported_operators": list(self.supported_operators),
            "unsupported_operators": list(self.unsupported_operators),
            "unsupported_reasons": dict(self.unsupported_reasons),
            "support_matrix": self.support_matrix,
            "notes": list(self.notes),
        }


_AUTOGRAD_CAPABILITIES = DifferentiableBackendCapabilities(
    backend_id="autograd",
    array_namespace="autograd.numpy",
    operators={
        "grad": True,
        "jacobian": True,
        "hessian": True,
        "jvp": False,
        "vjp": True,
        "hessian_vector_product": True,
        "portfolio_aad": False,
    },
    notes=(
        "Current backend supports scalar gradients, dense Jacobians, dense Hessians, VJP, and HVP.",
        "HVP support is for scalar objectives on smooth interior regions.",
        "JVP and portfolio AAD are explicit future extension hooks.",
        "JVP remains disabled because stock autograd lacks JVP coverage for pricing primitives such as norm.cdf.",
    ),
    unsupported_reasons={
        "jvp": (
            "stock autograd make_jvp lacks checked JVP coverage for pricing "
            "primitives such as norm.cdf"
        ),
        "portfolio_aad": (
            "portfolio AAD is still exposed through bounded book-specific "
            "lanes, not a universal backend operator"
        ),
    },
)


def get_backend_capabilities() -> DifferentiableBackendCapabilities:
    """Return the active differentiable backend capability declaration."""
    return _AUTOGRAD_CAPABILITIES


def supports_capability(capability: str) -> bool:
    """Return whether the active backend supports *capability*."""
    return get_backend_capabilities().supports(capability)


def require_capability(capability: str) -> None:
    """Raise if the active differentiable backend does not support *capability*."""
    if not supports_capability(capability):
        backend_id = get_backend_capabilities().backend_id
        raise NotImplementedError(
            f"differentiable backend {backend_id!r} does not support {capability!r}"
        )


def get_numpy():
    """Return the autograd-wrapped numpy module."""
    return anp


def gradient(fn, argnum: int = 0):
    """Return a function that computes the gradient of *fn* w.r.t. ``argnum``."""
    require_capability("grad")
    return autograd.grad(fn, argnum)


def hessian(fn, argnum: int = 0):
    """Return a function that computes the Hessian of *fn* w.r.t. ``argnum``."""
    require_capability("hessian")
    return autograd.hessian(fn, argnum)


def jacobian(fn, argnum: int = 0):
    """Return a function that computes the Jacobian of *fn* w.r.t. ``argnum``."""
    require_capability("jacobian")
    return autograd.jacobian(fn, argnum)


def _as_call_args(primals, *, unpack_primals: bool):
    """Normalize AD call arguments without unpacking unary tuple inputs by default."""
    if not unpack_primals:
        return (primals,)
    if not isinstance(primals, tuple):
        raise TypeError("unpack_primals=True requires primals to be a tuple")
    return primals


def jvp(fn, primals, tangents):
    """Future JVP hook.

    The current ``autograd`` backend exposes ``make_jvp`` but lacks forward-mode
    rules for pricing primitives Trellis depends on, including
    ``autograd.scipy.stats.norm.cdf``. This explicit placeholder prevents call
    sites from silently assuming forward-mode support before Trellis owns the
    missing primitive rules or adopts a backend with complete coverage.
    """
    if not supports_capability("jvp"):
        capabilities = get_backend_capabilities()
        backend_id = capabilities.backend_id
        support = capabilities.operator_support("jvp")
        reason = support.get("unsupported_reason") or (
            "the active backend does not provide checked JVP coverage"
        )
        raise NotImplementedError(
            f"differentiable backend {backend_id!r} does not support 'jvp'; "
            f"{reason}, so Trellis "
            "keeps JVP fail-closed until it owns the required pricing primitive rules"
        )
    raise NotImplementedError("jvp is declared supported but no implementation is wired")


def vjp(fn, primals, argnum: int = 0, *, unpack_primals: bool = False):
    """Return ``(value, pullback)`` for the VJP of *fn* at *primals*.

    Tuple-valued unary primals are preserved by default. Set
    ``unpack_primals=True`` when *primals* is the complete positional argument
    tuple for an n-ary callable.
    """
    require_capability("vjp")
    pullback, value = autograd.make_vjp(fn, argnum)(
        *_as_call_args(primals, unpack_primals=unpack_primals)
    )
    return value, pullback


def hessian_vector_product(
    fn,
    primals,
    vector,
    argnum: int = 0,
    *,
    unpack_primals: bool = False,
):
    """Return ``H @ vector`` for a scalar-objective Hessian at *primals*.

    This is a reverse-over-reverse ``autograd`` HVP wrapper. It is intended for
    scalar objectives on smooth interior regions; discontinuities, branch
    singularities, or vector-valued objectives should fail rather than imply a
    broader second-order support contract. Tuple-valued unary primals are
    preserved by default; set ``unpack_primals=True`` for n-ary callables.
    """
    require_capability("hessian_vector_product")
    return autograd.hessian_vector_product(fn, argnum)(
        *_as_call_args(primals, unpack_primals=unpack_primals),
        vector,
    )
