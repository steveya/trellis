"""Canonical pricing-method names and compatibility aliases.

The knowledge system uses a small set of stable method-family identifiers.
Legacy labels and conceptual synonyms are normalized to these canonical names
so prompts, retrieval, and tests can tolerate gradual architectural cleanup
without breaking older traces or configs.
"""

from __future__ import annotations

CANONICAL_METHODS = frozenset({
    "analytical",
    "rate_tree",
    "monte_carlo",
    "qmc",
    "pde_solver",
    "fft_pricing",
    "copula",
    "waterfall",
})

_METHOD_ALIASES = {
    "analytic": "analytical",
    "analytics": "analytical",
    "tree": "rate_tree",
    "trees": "rate_tree",
    "lattice": "rate_tree",
    "lattices": "rate_tree",
    "lattice_tree": "rate_tree",
    "lattice_methods": "rate_tree",
    "short_rate_tree": "rate_tree",
    "montecarlo": "monte_carlo",
    "mc": "monte_carlo",
    "qmc": "qmc",
    "quasi_monte_carlo": "qmc",
    "quasi_random": "qmc",
    "low_discrepancy": "qmc",
    "sobol": "qmc",
    "sobol_mc": "qmc",
    "pde": "pde_solver",
    "pide": "pde_solver",
    "finite_difference": "pde_solver",
    "finite_differences": "pde_solver",
    "fd": "pde_solver",
    "transform": "fft_pricing",
    "transforms": "fft_pricing",
    "fft": "fft_pricing",
    "fourier": "fft_pricing",
    "characteristic_function": "fft_pricing",
    "cashflow": "waterfall",
    "cashflow_engine": "waterfall",
    "structured_cashflow": "waterfall",
}


def normalize_method(method: str | None) -> str:
    """Normalize a method label to the canonical family name.

    Unknown inputs are normalized for formatting only and returned unchanged.
    This keeps the normalization layer non-destructive for experimental labels.
    """
    if method is None:
        return ""

    key = method.strip().lower().replace(" ", "_").replace("-", "_")
    return _METHOD_ALIASES.get(key, key)


def aliases_for_method(method: str | None) -> tuple[str, ...]:
    """Return all known aliases for *method*, including the canonical label."""
    canonical = normalize_method(method)
    aliases = {canonical}
    for alias, target in _METHOD_ALIASES.items():
        if target == canonical:
            aliases.add(alias)
    return tuple(sorted(a for a in aliases if a))


def is_known_method(method: str | None) -> bool:
    """Return whether *method* resolves to a known canonical method family."""
    return normalize_method(method) in CANONICAL_METHODS
