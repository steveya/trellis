"""Operator-facing metadata for backend bindings.

This catalog separates display/diagnostic wording from route YAML so operator
surfaces can resolve stable binding-first labels without reading route-card
prose. Explicit entries cover the current checked bindings that appear in
operator traces and diagnostics; unknown bindings fall back to a generic,
binding-id-derived label until a dedicated entry is added.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BindingOperatorMetadata:
    """Operator-facing metadata for one backend binding."""

    display_name: str
    short_description: str
    diagnostic_label: str


_PDE_SOLVER_FALLBACK_BINDING_ID = "pde_solver:pde_solver:fallback"


_CANONICAL_BINDING_OPERATOR_METADATA: dict[str, BindingOperatorMetadata] = {
    "trellis.models.quanto_option.price_quanto_option_analytical_from_market_state": BindingOperatorMetadata(
        display_name="Quanto option analytical binding",
        short_description="Exact analytical backend binding for semantic quanto option pricing.",
        diagnostic_label="quanto_analytical_binding",
    ),
    "trellis.models.fx_vanilla.price_fx_vanilla_analytical": BindingOperatorMetadata(
        display_name="FX vanilla analytical binding",
        short_description="Exact Garman-Kohlhagen analytical helper binding for FX vanilla pricing.",
        diagnostic_label="fx_vanilla_analytical_binding",
    ),
    "trellis.models.credit_default_swap.price_cds_analytical": BindingOperatorMetadata(
        display_name="CDS analytical binding",
        short_description="Exact analytical backend binding for single-name CDS pricing.",
        diagnostic_label="credit_default_swap_analytical_binding",
    ),
    "trellis.models.zcb_option.price_zcb_option_jamshidian": BindingOperatorMetadata(
        display_name="ZCB option analytical binding",
        short_description="Exact Jamshidian analytical binding for zero-coupon bond option pricing.",
        diagnostic_label="zcb_option_analytical_binding",
    ),
    "trellis.models.zcb_option_tree.price_zcb_option_tree": BindingOperatorMetadata(
        display_name="ZCB option tree binding",
        short_description="Exact rate-tree backend binding for zero-coupon bond option pricing.",
        diagnostic_label="zcb_option_tree_binding",
    ),
    "trellis.models.black.black76_call": BindingOperatorMetadata(
        display_name="Black-76 analytical binding",
        short_description="Exact Black-76 helper binding for analytical vanilla and swaption pricing.",
        diagnostic_label="black76_analytical_binding",
    ),
    "trellis.models.black.black76_put": BindingOperatorMetadata(
        display_name="Black-76 analytical put binding",
        short_description="Exact Black-76 put helper binding for analytical vanilla and swaption pricing.",
        diagnostic_label="black76_put_analytical_binding",
    ),
    # Fallback binding ids intentionally use the shared `engine:route:fallback`
    # shape from backend_bindings.py / route_registry.py rather than module paths.
    _PDE_SOLVER_FALLBACK_BINDING_ID: BindingOperatorMetadata(
        display_name="PDE solver fallback binding",
        short_description="Binding-first fallback label for the generic theta-method PDE solver surface.",
        diagnostic_label="pde_solver_fallback_binding",
    ),
}


def resolve_binding_operator_metadata(
    *,
    binding_id: str,
    engine_family: str = "",
    route_family: str = "",
    route_id: str = "",
) -> BindingOperatorMetadata | None:
    """Resolve operator-facing metadata for a backend binding."""
    normalized_binding_id = str(binding_id or "").strip()
    if not normalized_binding_id:
        return None
    explicit = _CANONICAL_BINDING_OPERATOR_METADATA.get(normalized_binding_id)
    if explicit is not None:
        return explicit
    return _fallback_binding_operator_metadata(
        binding_id=normalized_binding_id,
        engine_family=engine_family,
        route_family=route_family,
        route_id=route_id,
    )


def _fallback_binding_operator_metadata(
    *,
    binding_id: str,
    engine_family: str = "",
    route_family: str = "",
    route_id: str = "",
) -> BindingOperatorMetadata:
    """Return a generic metadata record derived from binding identity."""
    symbol = str(binding_id.rsplit(".", 1)[-1] or binding_id).strip()
    display_root = _humanize_symbol(symbol)
    family_bits = [str(engine_family or "").strip(), str(route_family or "").strip()]
    family_bits = [item for item in family_bits if item]
    display_name = display_root
    if family_bits:
        display_name = f"{display_root} ({' / '.join(dict.fromkeys(family_bits))})"
    route_fragment = str(route_id or "").strip()
    return BindingOperatorMetadata(
        display_name=display_name,
        short_description=(
            f"Fallback operator metadata derived from backend binding `{binding_id}`."
        ),
        diagnostic_label=_slugify(route_fragment) if route_fragment else _slugify(symbol),
    )


def _humanize_symbol(symbol: str) -> str:
    """Project a code symbol onto a readable operator label."""
    text = str(symbol or "").strip()
    for prefix in ("price_", "build_", "resolve_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("_from_market_state", "_helper", "_kernel"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    words = [part for part in text.replace(":", "_").split("_") if part]
    if not words:
        return "Backend binding"
    acronyms = {
        "fx": "FX",
        "cds": "CDS",
        "zcb": "ZCB",
        "pde": "PDE",
        "qmc": "QMC",
        "fft": "FFT",
        "mc": "MC",
    }
    rendered: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in acronyms:
            rendered.append(acronyms[lower])
        elif lower == "black76":
            rendered.append("Black-76")
        else:
            rendered.append(word.capitalize())
    return " ".join(rendered)


def _slugify(text: str) -> str:
    """Return a compact snake-case diagnostic label."""
    normalized = [
        char.lower() if char.isalnum() else "_"
        for char in str(text or "").strip()
    ]
    slug = "".join(normalized)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "binding_fallback"
