"""Shared lower-layer instrument identity helpers.

These helpers keep instrument-family normalization and ingress-only text
fallbacks aligned across planner, executor, and task-runtime paths. The rule
is simple: explicit family identity beats generic text heuristics, and only
true ingress boundaries should still use pattern-based inference.
"""

from __future__ import annotations

from dataclasses import dataclass

_INSTRUMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("zero-coupon bond option", "zcb_option"),
    ("zero coupon bond option", "zcb_option"),
    ("zero-coupon bond", "zcb_option"),
    ("zero coupon bond", "zcb_option"),
    ("bond option", "zcb_option"),
    ("zcb option", "zcb_option"),
    ("himalaya", "basket_option"),
    ("ranked observation", "basket_option"),
    ("remaining constituents", "basket_option"),
    ("american put", "american_put"),
    ("american option", "american_option"),
    ("worst-of", "basket_option"),
    ("worst of", "basket_option"),
    ("best-of", "basket_option"),
    ("best of", "basket_option"),
    ("rainbow", "basket_option"),
    ("spread option", "basket_option"),
    ("basket", "basket_option"),
    ("european equity call", "european_option"),
    ("european equity put", "european_option"),
    ("european call", "european_option"),
    ("european put", "european_option"),
    ("european option", "european_option"),
    ("callable bond", "callable_bond"),
    ("puttable bond", "puttable_bond"),
    ("bermudan swaption", "bermudan_swaption"),
    ("barrier", "barrier_option"),
    ("asian option", "asian_option"),
    ("asian", "asian_option"),
    ("lookback", "barrier_option"),
    ("autocallable", "autocallable"),
    ("variance swap", "variance_swap"),
    ("heston", "heston_option"),
    ("cev", "european_option"),
    ("cdo", "cdo"),
    ("cds", "credit_default_swap"),
    ("nth-to-default", "nth_to_default"),
    ("swaption", "swaption"),
    ("cap", "cap"),
    ("floor", "floor"),
    ("convertible", "callable_bond"),
    ("mbs", "mbs"),
    ("range accrual", "range_accrual"),
    ("digital", "european_option"),
    ("compound option", "european_option"),
    ("chooser", "european_option"),
    ("cliquet", "autocallable"),
    ("double barrier", "barrier_option"),
    ("quanto", "quanto_option"),
    ("forward start", "european_option"),
    ("vanilla option", "european_option"),
    ("vanilla", "european_option"),
    ("european", "european_option"),
    ("fx", "european_option"),
)

_REFINABLE_INSTRUMENT_FAMILIES: dict[str, frozenset[str]] = {
    "generic": frozenset(),
    "european_option": frozenset(
        {
            "asian_option",
            "barrier_option",
            "basket_option",
            "quanto_option",
            "zcb_option",
        }
    ),
    "american_option": frozenset({"american_put"}),
    "bond": frozenset({"callable_bond", "puttable_bond", "range_accrual", "zcb_option"}),
    "swaption": frozenset({"bermudan_swaption"}),
}


@dataclass(frozen=True)
class InstrumentIdentityResolution:
    """Resolved instrument identity plus the provenance of that resolution."""

    instrument_type: str | None
    source: str = "missing"


def normalize_instrument_type(value: str | None) -> str:
    """Return a canonical lower_snake_case instrument identifier."""
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def infer_instrument_type_from_text(
    text: str | None,
    *,
    explicit_instrument_type: str | None = None,
) -> str | None:
    """Infer the most likely instrument family from text.

    If an explicit instrument type is already known, return it directly.

    This function is intentionally conservative. It is reserved for ingress
    paths that genuinely start from raw text, and it avoids the broadest
    catch-all patterns that previously widened products like generic bonds or
    swaps before semantic compilation had a chance to narrow them correctly.
    """
    explicit = normalize_instrument_type(explicit_instrument_type)
    if explicit:
        return explicit
    lower_text = str(text or "").lower()
    for pattern, instrument_type in _INSTRUMENT_PATTERNS:
        if pattern in lower_text:
            return instrument_type
    return None


def resolve_instrument_identity(
    text: str | None,
    *,
    explicit_instrument_type: str | None = None,
    explicit_source: str = "explicit",
    inferred_source: str = "text_fallback",
) -> InstrumentIdentityResolution:
    """Resolve instrument identity once and record how that resolution happened."""
    explicit = normalize_instrument_type(explicit_instrument_type)
    if explicit:
        return InstrumentIdentityResolution(instrument_type=explicit, source=explicit_source)
    inferred = infer_instrument_type_from_text(text)
    if inferred:
        return InstrumentIdentityResolution(instrument_type=inferred, source=inferred_source)
    return InstrumentIdentityResolution(instrument_type=None, source="missing")


def resolve_authoritative_instrument_type(*candidates: str | None) -> str | None:
    """Return the most authoritative instrument family among candidate sources.

    The first non-empty family wins unless a later candidate is a declared
    refinement of that earlier family. This lets lower layers upgrade from a
    generic request family like `european_option` to a compiled family like
    `zcb_option` without letting unrelated local heuristics override explicit
    specific families.
    """
    authoritative: str | None = None
    for candidate in candidates:
        normalized = normalize_instrument_type(candidate)
        if not normalized or normalized == "unknown":
            continue
        if authoritative is None:
            authoritative = normalized
            continue
        if _is_family_refinement(authoritative, normalized):
            authoritative = normalized
    return authoritative


def _is_family_refinement(current: str, candidate: str) -> bool:
    """Return whether `candidate` is an allowed refinement of `current`."""
    if not current or not candidate or current == candidate:
        return False
    if current == "generic":
        return True
    return candidate in _REFINABLE_INSTRUMENT_FAMILIES.get(current, frozenset())


__all__ = [
    "InstrumentIdentityResolution",
    "infer_instrument_type_from_text",
    "normalize_instrument_type",
    "resolve_authoritative_instrument_type",
    "resolve_instrument_identity",
]
