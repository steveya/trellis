"""Product → feature decomposition.

For known instruments, returns a static decomposition from
canonical/decompositions.yaml.  For novel/composite products,
uses LLM to decompose into known features from the taxonomy.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any
from typing import TYPE_CHECKING

from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import ProductDecomposition, ProductIR, RetrievalSpec
from trellis.agent.semantic_tokens import (
    EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY,
)
from trellis.core.capabilities import normalize_market_data_requirements

if TYPE_CHECKING:
    from trellis.agent.knowledge.store import KnowledgeStore


_DECOMPOSITION_CACHE: dict[tuple[Any, ...], ProductDecomposition] = {}
_DECOMPOSITION_CACHE_HITS = 0
_DECOMPOSITION_CACHE_MISSES = 0


def decompose(
    description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    store: KnowledgeStore | None = None,
) -> ProductDecomposition:
    """Decompose a product into features.

    1. Normalise instrument_type and check static decompositions.
    2. Try fuzzy matching against known decomposition keys.
    3. Fall back to LLM decomposition using the feature taxonomy.

    Parameters
    ----------
    description
        Natural-language product description (e.g., "callable range accrual").
    instrument_type
        Optional explicit type key (e.g., "callable_bond").
    model
        LLM model to use for fallback decomposition.
    store
        KnowledgeStore instance (default: global singleton).
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    cache_key = _decomposition_cache_key(
        description=description,
        instrument_type=instrument_type,
        model=model,
        store=store,
    )
    cached = _DECOMPOSITION_CACHE.get(cache_key)
    if cached is not None:
        global _DECOMPOSITION_CACHE_HITS
        _DECOMPOSITION_CACHE_HITS += 1
        return cached
    global _DECOMPOSITION_CACHE_MISSES
    _DECOMPOSITION_CACHE_MISSES += 1

    # Step 1: exact match on normalised key
    key = _normalise(instrument_type or description)
    matched = _match_static_decomposition(key, store)
    if matched is not None:
        _DECOMPOSITION_CACHE[cache_key] = matched
        return matched

    # Step 4: LLM decomposition
    result = _decompose_via_llm(description, key, store, model)
    _DECOMPOSITION_CACHE[cache_key] = result
    return result


def decomposition_cache_stats() -> dict[str, int]:
    """Return lightweight runtime decomposition-cache statistics."""
    return {
        "hits": _DECOMPOSITION_CACHE_HITS,
        "misses": _DECOMPOSITION_CACHE_MISSES,
        "size": len(_DECOMPOSITION_CACHE),
    }


def clear_decomposition_cache() -> None:
    """Clear the warm/runtime decomposition cache."""
    global _DECOMPOSITION_CACHE_HITS, _DECOMPOSITION_CACHE_MISSES
    _DECOMPOSITION_CACHE.clear()
    _DECOMPOSITION_CACHE_HITS = 0
    _DECOMPOSITION_CACHE_MISSES = 0


def _decomposition_cache_key(
    *,
    description: str,
    instrument_type: str | None,
    model: str | None,
    store: KnowledgeStore,
) -> tuple[Any, ...]:
    """Build a stable cache key for runtime decomposition reuse."""
    return (
        id(store),
        description.strip(),
        _normalise(instrument_type) if instrument_type else None,
        model,
    )


def decompose_to_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    store: KnowledgeStore | None = None,
) -> ProductIR:
    """Decompose a product description into a structured ``ProductIR`` without calling an LLM.

    For known instruments (e.g. "callable bond"), returns the canonical
    static decomposition from YAML.  For novel or composite products that
    have no static entry, falls back to keyword-based trait extraction
    that avoids guessing -- it only assigns traits it can identify with
    certainty from the text, leaving unknowns as unresolved primitives.
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    inferred_instrument = _infer_instrument(description, instrument_type)
    matched = None
    if inferred_instrument and not _looks_composite(description):
        matched = store._decompositions.get(inferred_instrument)
    if matched is None:
        matched = _match_static_decomposition(_normalise(description), store)
        if matched is not None and _looks_composite(description):
            matched = None

    if matched is not None:
        instrument = inferred_instrument or matched.instrument
        if instrument != matched.instrument and instrument in store._decompositions:
            matched = store._decompositions[instrument]
        return _product_ir_from_decomposition(
            instrument=instrument,
            decomposition=matched,
            description=description,
            store=store,
        )

    return _infer_composite_ir(description, inferred_instrument, store)


def build_product_ir(
    *,
    description: str,
    instrument: str | None = None,
    payoff_family: str | None = None,
    payoff_traits: tuple[str, ...] | list[str] = (),
    exercise_style: str | None = None,
    state_dependence: str | None = None,
    schedule_dependence: bool | None = None,
    model_family: str | None = None,
    candidate_engine_families: tuple[str, ...] | list[str] | None = None,
    required_market_data: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
    reusable_primitives: tuple[str, ...] | list[str] = (),
    unresolved_primitives: tuple[str, ...] | list[str] | None = None,
    supported: bool | None = None,
    preferred_method: str | None = None,
    store: KnowledgeStore | None = None,
    event_machine: object | None = None,
) -> ProductIR:
    """Build a ``ProductIR`` from explicit structured fields.

    This is the deterministic bridge for user-defined product specifications.
    It reuses the same normalization and inference helpers as the text-based IR
    path, but starts from explicit semantic fields instead of a free-form
    product description.
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    normalized_instrument = _normalise(instrument or description)
    normalized_traits = tuple(sorted(set(payoff_traits)))
    resolved_schedule_dependence = (
        _schedule_dependence_for(normalized_instrument, normalized_traits)
        if schedule_dependence is None
        else schedule_dependence
    )
    resolved_state_dependence = (
        _state_dependence_for(normalized_instrument, normalized_traits, resolved_schedule_dependence)
        if state_dependence is None
        else state_dependence
    )
    resolved_exercise_style = (
        _exercise_style_for(normalized_instrument, normalized_traits, description)
        if exercise_style is None
        else exercise_style
    )
    resolved_model_family = (
        _model_family_for(normalized_instrument, normalized_traits, preferred_method or "", description)
        if model_family is None
        else model_family
    )
    resolved_payoff_family = (
        _payoff_family_for(normalized_instrument, normalized_traits, description)
        if not payoff_family
        else payoff_family
    )
    resolved_route_families = _route_families_for(
        normalized_instrument,
        resolved_payoff_family,
        resolved_exercise_style,
        resolved_model_family,
    )
    resolved_engine_families = tuple(candidate_engine_families or _candidate_engine_families_for(
        preferred_method or "",
        resolved_exercise_style,
        normalized_traits,
        resolved_model_family,
    ))
    resolved_unresolved_primitives = tuple(
        unresolved_primitives
        if unresolved_primitives is not None
        else _unresolved_primitives_for(
            normalized_traits,
            resolved_exercise_style,
            resolved_model_family,
        )
    )
    resolved_required_market_data = frozenset(
        normalize_market_data_requirements(
            required_market_data or _market_data_for_traits(normalized_traits, store)
        )
    )
    resolved_reusable_primitives = tuple(
        reusable_primitives or _reusable_primitives_for(
            normalized_traits,
            resolved_model_family,
        )
    )

    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=normalized_instrument,
        payoff_family=payoff_family or _payoff_family_for(
            normalized_instrument,
            normalized_traits,
            description,
        ),
        payoff_traits=normalized_traits,
        exercise_style=resolved_exercise_style,
        state_dependence=resolved_state_dependence,
        schedule_dependence=resolved_schedule_dependence,
        model_family=resolved_model_family,
        candidate_engine_families=resolved_engine_families,
        route_families=resolved_route_families,
        required_market_data=resolved_required_market_data,
        reusable_primitives=resolved_reusable_primitives,
        unresolved_primitives=resolved_unresolved_primitives,
        supported=len(resolved_unresolved_primitives) == 0 if supported is None else supported,
        event_machine=event_machine,
    ), description))


def retrieval_spec_from_ir(
    ir: ProductIR,
    *,
    preferred_method: str | None = None,
) -> RetrievalSpec:
    """Build a retrieval spec from ProductIR.

    This is the Phase 3 bridge that lets retrieval and prompt guidance use the
    same typed product representation that semantic validation already consumes.
    """
    features = set(ir.payoff_traits)
    features.update(_retrieval_features_from_exercise(ir.exercise_style))
    features.update(_retrieval_features_from_state(ir.state_dependence))
    features.update(_retrieval_features_from_model(ir.model_family))
    features.update(_retrieval_features_from_market_data(ir.required_market_data))
    if normalize_method(preferred_method or "") == "pde_solver":
        features.add("pde_grid")

    return RetrievalSpec(
        method=normalize_method(preferred_method) if preferred_method else None,
        features=sorted(features),
        instrument=ir.instrument,
        exercise_style=ir.exercise_style,
        state_dependence=ir.state_dependence,
        schedule_dependence=ir.schedule_dependence,
        model_family=ir.model_family,
        candidate_engine_families=tuple(ir.candidate_engine_families),
        semantic_text_markers=_semantic_text_markers_from_ir(ir),
        reusable_primitives=tuple(ir.reusable_primitives),
        unresolved_primitives=tuple(ir.unresolved_primitives),
    )


def _normalise(text: str) -> str:
    """Normalise to a decomposition key: lowercase, underscores."""
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def _match_static_decomposition(
    key: str,
    store: KnowledgeStore,
) -> ProductDecomposition | None:
    """Match a static decomposition by exact or fuzzy key."""
    if key in store._decompositions:
        return store._decompositions[key]

    candidates: list[tuple[int, str, ProductDecomposition]] = []
    for known_key, decomp in store._decompositions.items():
        if known_key in key or key in known_key:
            candidates.append((len(known_key), known_key, decomp))
    for known_key, decomp in store._decompositions.items():
        if all(word in key for word in known_key.split("_")):
            candidates.append((len(known_key), known_key, decomp))
    if not candidates:
        return None

    candidates.sort(key=lambda item: -item[0])
    return candidates[0][2]


def _infer_instrument(description: str, instrument_type: str | None) -> str | None:
    """Infer the most specific supported instrument key from text."""
    desc = _normalise(description)
    if instrument_type:
        normalized = _normalise(instrument_type)
        if normalized in {"credit_default_swap"}:
            normalized = "cds"
        if normalized in {"basket_option", "basket_path_payoff"}:
            if any(
                cue in desc
                for cue in (
                    "nth_to_default",
                    "nth to default",
                    "nth-default",
                    "first_to_default",
                    "first to default",
                    "default correlation",
                    "basket cds",
                )
            ):
                return "nth_to_default"
            if any(
                cue in desc
                for cue in (
                    "cdo tranche",
                    "collateralized debt obligation",
                    "attachment",
                    "detachment",
                )
            ):
                return "cdo"
        return normalized

    patterns = [
        ("bermudan_swaption", ("bermudan_swaption", "bermudan swaption")),
        ("callable_bond", ("callable_bond", "callable bond")),
        ("puttable_bond", ("puttable_bond", "puttable bond")),
        ("zcb_option", ("zcb_option", "zcb option", "zero_coupon_bond_option", "zero-coupon bond option")),
        ("american_put", ("american_put", "american put")),
        ("american_option", ("american_option", "american option")),
        ("barrier_option", ("barrier_option", "barrier option")),
        ("asian_option", ("asian_option", "asian option")),
        ("heston_option", ("heston_option", "heston option", "heston")),
        ("variance_swap", ("variance_swap", "variance swap")),
        (
            "credit_loss_distribution",
            (
                "credit_loss_distribution",
                "portfolio_loss_distribution",
                "portfolio loss distribution",
                "multi-name portfolio loss distribution",
                "recursive loss distribution",
            ),
        ),
        ("cds", ("cds", "credit default swap", "credit_default_swap")),
        ("nth_to_default", ("nth_to_default", "nth-to-default", "nth to default")),
        ("swaption", ("swaption",)),
        ("cap", ("cap",)),
        ("floor", ("floor",)),
        ("swap", ("swap",)),
        ("bond", ("bond",)),
    ]
    for instrument, aliases in patterns:
        if any(alias.replace(" ", "_") in desc for alias in aliases):
            return instrument
    if "european_call" in desc or "european_put" in desc or "european_option" in desc:
        return "european_option"
    return None


def _looks_composite(description: str) -> bool:
    """Return whether the description combines multiple primary product traits."""
    desc = _normalise(description)
    composite_markers = [
        "asian",
        "barrier",
        "lookback",
        "american",
        "bermudan",
        "heston",
        "jump",
        "callable",
    ]
    hits = sum(1 for marker in composite_markers if marker in desc)
    return hits >= 3


def _product_ir_from_decomposition(
    *,
    instrument: str,
    decomposition: ProductDecomposition,
    description: str,
    store: KnowledgeStore,
) -> ProductIR:
    """Convert a canonical product decomposition into ``ProductIR``."""
    payoff_traits = tuple(sorted(set(decomposition.features)))
    exercise_style = _exercise_style_for(instrument, payoff_traits, description)
    schedule_dependence = _schedule_dependence_for(instrument, payoff_traits)
    state_dependence = _state_dependence_for(instrument, payoff_traits, schedule_dependence)
    model_family = _model_family_for(instrument, payoff_traits, decomposition.method, description)
    route_families = _route_families_for(
        instrument,
        _payoff_family_for(instrument, payoff_traits, description),
        exercise_style,
        model_family,
    )
    candidate_engine_families = _candidate_engine_families_for(
        decomposition.method,
        exercise_style,
        payoff_traits,
        model_family,
    )
    normalized_desc = _normalise(description)
    if instrument == "bermudan_swaption" and (
        "analytical_lower_bound" in normalized_desc
        or ("analytical" in normalized_desc and "lower_bound" in normalized_desc)
    ):
        route_families = tuple(dict.fromkeys((*route_families, "analytical")))
        candidate_engine_families = tuple(dict.fromkeys((*candidate_engine_families, "analytical")))
    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=instrument,
        payoff_family=_payoff_family_for(instrument, payoff_traits, description),
        payoff_traits=payoff_traits,
        exercise_style=exercise_style,
        state_dependence=state_dependence,
        schedule_dependence=schedule_dependence,
        model_family=model_family,
        candidate_engine_families=candidate_engine_families,
        route_families=route_families,
        required_market_data=frozenset(
            normalize_market_data_requirements(decomposition.required_market_data)
        ),
        reusable_primitives=decomposition.method_modules,
        unresolved_primitives=(),
        supported=True,
    ), description))


def _infer_composite_ir(
    description: str,
    instrument: str | None,
    store: KnowledgeStore,
) -> ProductIR:
    """Rule-based IR for unsupported or composite products."""
    desc = _normalise(description)
    payoff_traits = _traits_from_text(desc)
    schedule_dependence = _schedule_dependence_for(instrument or "", payoff_traits)
    state_dependence = _state_dependence_for(instrument or "", payoff_traits, schedule_dependence)
    model_family = _model_family_for(instrument or "", payoff_traits, "", description)
    exercise_style = _exercise_style_for(instrument or "", payoff_traits, description)
    route_families = _route_families_for(
        instrument or "",
        _payoff_family_for(instrument or "", payoff_traits, description),
        exercise_style,
        model_family,
    )
    candidate_engine_families = _candidate_engine_families_for(
        "",
        exercise_style,
        payoff_traits,
        model_family,
    )
    required_market_data = frozenset(
        normalize_market_data_requirements(_market_data_for_traits(payoff_traits, store))
    )
    unresolved_primitives = _unresolved_primitives_for(
        payoff_traits,
        exercise_style,
        model_family,
    )
    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=instrument or _normalise(description),
        payoff_family=_payoff_family_for(instrument or "", payoff_traits, description),
        payoff_traits=payoff_traits,
        exercise_style=exercise_style,
        state_dependence=state_dependence,
        schedule_dependence=schedule_dependence,
        model_family=model_family,
        candidate_engine_families=candidate_engine_families,
        route_families=route_families,
        required_market_data=required_market_data,
        reusable_primitives=_reusable_primitives_for(payoff_traits, model_family),
        unresolved_primitives=unresolved_primitives,
        supported=len(unresolved_primitives) == 0,
    ), description))


def _traits_from_text(desc: str) -> tuple[str, ...]:
    """Infer feature-like payoff traits from free text."""
    trait_aliases = {
        "asian": ("asian",),
        "barrier": ("barrier",),
        "lookback": ("lookback",),
        "callable": ("callable",),
        "puttable": ("puttable",),
        "bermudan": ("bermudan",),
        "american": ("american", "early_exercise"),
        "early_exercise": ("early exercise",),
        "stochastic_vol": ("heston", "stochastic vol", "stochastic_vol"),
        "jump_diffusion": ("jump", "jump_diffusion", "merton"),
        "mean_reversion": ("mean_reversion", "mean reversion", "short rate"),
    }
    traits: set[str] = set()
    for trait, aliases in trait_aliases.items():
        if any(alias.replace(" ", "_") in desc for alias in aliases):
            traits.add(trait)
    if any(
        marker in desc
        for marker in (
            "best_of_two",
            "best_of",
            "rainbow_option",
            "spread_option",
            "kirk_approximation",
            "kirk_spread",
        )
    ):
        traits.add("two_asset_terminal_basket")
    if "option" in desc and "asian" not in traits and "barrier" not in traits:
        traits.add("vanilla_option")
    return tuple(sorted(traits))


def _retrieval_features_from_exercise(exercise_style: str) -> set[str]:
    """Map exercise-style labels onto retrieval features used by the knowledge store."""
    if exercise_style in {"american", "bermudan", "issuer_call", "holder_put"}:
        features = {"early_exercise"}
        if exercise_style == "issuer_call":
            features.add("callable")
        elif exercise_style == "holder_put":
            features.add("puttable")
        return features
    return set()


def _retrieval_features_from_state(state_dependence: str) -> set[str]:
    """Map state-dependence labels onto retrieval features."""
    if state_dependence == "path_dependent":
        return {"path_dependent"}
    if state_dependence == "schedule_dependent":
        return {"backward_induction"}
    return set()


def _retrieval_features_from_model(model_family: str) -> set[str]:
    """Map model-family labels onto retrieval features."""
    if model_family == "interest_rate":
        return {"mean_reversion"}
    if model_family == "stochastic_volatility":
        return {"stochastic_vol"}
    return set()


def _retrieval_features_from_market_data(
    required_market_data: frozenset[str] | set[str] | tuple[str, ...],
) -> set[str]:
    """Map required market-data capabilities onto retrieval features.

    ProductIR carries precise market-data requirements, but the retrieval bridge
    previously dropped them and kept only payoff traits. That made knowledge
    ranking too generic for routes that depend on specific foreign-carry /
    forward-rate contracts, such as the FX vanilla analytical lane.
    """
    mapping = {
        "forward_curve": {"forward_rate"},
        "forecast_curve": {"forward_rate"},
        "fx_rates": {"fx"},
    }
    features: set[str] = set()
    for capability in required_market_data:
        features.update(mapping.get(str(capability).strip(), set()))
    return features


def _semantic_text_markers_from_ir(ir: ProductIR) -> tuple[str, ...]:
    """Build generic high-signal text markers for lesson reranking.

    Keep these markers focused on helper and primitive identity. Broader
    product-family labels are already represented in the indexed retrieval
    features and instrument fields; repeating them here perturbs unrelated
    canary prompt surfaces without adding much disambiguation value.
    """
    raw_markers: list[str] = []
    raw_markers.extend(ir.reusable_primitives)
    raw_markers.extend(ir.unresolved_primitives)
    if ir.model_family == "fx":
        raw_markers.extend(
            value
            for value in (
                ir.instrument,
                ir.payoff_family,
                ir.model_family,
            )
            if value and value != "generic"
        )
        raw_markers.extend(ir.payoff_traits)

    markers: list[str] = []
    seen: set[str] = set()
    for raw in raw_markers:
        text = str(raw).strip()
        if not text:
            continue
        variants = (
            text.lower(),
            text.replace("_", " ").lower(),
            re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip().lower(),
        )
        for variant in variants:
            normalized = " ".join(variant.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            markers.append(normalized)
    return tuple(markers)


def _payoff_family_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    description: str,
) -> str:
    """Map an instrument/trait set onto a stable payoff-family label."""
    product_traits = {"asian", "barrier", "lookback", "callable", "puttable"}
    if instrument == "ranked_observation_basket":
        return "basket_path_payoff"
    if instrument == "basket_option":
        if "ranked_observation" in payoff_traits:
            return "basket_path_payoff"
        return "basket_option"
    if len(product_traits.intersection(payoff_traits)) >= 2:
        return "composite_option"
    if instrument in {"swaption", "bermudan_swaption"}:
        return "swaption"
    if instrument == "cds":
        return EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY
    if instrument == "zcb_option":
        return "zcb_option"
    if instrument == "callable_bond":
        return "callable_fixed_income"
    if instrument == "puttable_bond":
        return "puttable_fixed_income"
    if instrument in {"asian_option"}:
        return "asian_option"
    if instrument in {"barrier_option"}:
        return "barrier_option"
    if instrument in {"american_put", "american_option", "european_option", "heston_option"}:
        return "vanilla_option"
    if instrument == "credit_loss_distribution":
        return "credit_loss_distribution"
    if instrument in {"cdo", "nth_to_default"}:
        return "credit_basket"
    if instrument == "mbs":
        return "mortgage_pool"
    if "option" in _normalise(description):
        return "composite_option"
    return instrument or "generic_product"


def _exercise_style_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    description: str,
) -> str:
    """Infer exercise style from instrument and traits."""
    desc = _normalise(description)
    if instrument == "callable_bond" or "callable" in payoff_traits:
        return "issuer_call"
    if instrument == "puttable_bond":
        return "holder_put"
    if instrument == "bermudan_swaption" or "bermudan" in payoff_traits:
        return "bermudan"
    if instrument in {"american_put", "american_option"} or "american" in payoff_traits or "american" in desc:
        return "american"
    if "option" in desc or instrument in {
        "swaption",
        "barrier_option",
        "asian_option",
        "european_option",
        "heston_option",
    }:
        return "european"
    return "none"


def _schedule_dependence_for(instrument: str, payoff_traits: tuple[str, ...]) -> bool:
    """Return whether the product is schedule dependent."""
    if instrument in {"american_put", "american_option", "european_option", "heston_option", "asian_option", "barrier_option"}:
        return False
    if any(
        trait in payoff_traits
        for trait in ("callable", "puttable", "bermudan", "fixed_coupons", "floating_coupons", "amortization", "prepayment")
    ):
        return True
    return instrument in {
        "bond",
        "swap",
        "cap",
        "floor",
        "swaption",
        "bermudan_swaption",
        "callable_bond",
        "puttable_bond",
        "mbs",
    }


def _state_dependence_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    schedule_dependence: bool,
) -> str:
    """Infer state dependence from traits."""
    if instrument == "barrier_option" and not any(
        trait in payoff_traits for trait in ("asian", "lookback", "path_dependent")
    ):
        return "terminal_markov"
    if any(trait in payoff_traits for trait in ("barrier", "asian", "lookback", "path_dependent", "prepayment", "range_condition")):
        return "path_dependent"
    if schedule_dependence:
        return "schedule_dependent"
    return "terminal_markov"


def _model_family_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    method: str,
    description: str,
) -> str:
    """Map traits and method to a broad model family."""
    desc = _normalise(description)
    if "stochastic_vol" in payoff_traits or "heston" in desc or instrument == "heston_option":
        return "stochastic_volatility"
    if "jump_diffusion" in payoff_traits:
        return "jump_diffusion"
    if instrument in {"swap", "swaption", "bermudan_swaption", "callable_bond", "puttable_bond", "bond", "cap", "floor", "zcb_option"}:
        return "interest_rate"
    if method == "copula" or instrument in {"cdo", "nth_to_default", "credit_loss_distribution"}:
        return "credit_copula"
    if method == "waterfall" or instrument == "mbs":
        return "cashflow_structured"
    if "option" in desc or instrument in {"barrier_option", "asian_option", "american_put", "american_option", "european_option"}:
        return "equity_diffusion"
    return "generic"


def _candidate_engine_families_for(
    method: str,
    exercise_style: str,
    payoff_traits: tuple[str, ...],
    model_family: str,
) -> tuple[str, ...]:
    """Map canonical methods and traits to conceptual engine families."""
    method_map = {
        "analytical": ("analytical",),
        "rate_tree": ("lattice",),
        "monte_carlo": ("monte_carlo",),
        "qmc": ("qmc",),
        "pde_solver": ("pde",),
        "fft_pricing": ("transforms",),
        "copula": ("copula",),
        "waterfall": ("cashflow",),
    }
    families = list(method_map.get(method, ()))
    if exercise_style not in {"none", "european"} and "exercise" not in families:
        families.append("exercise")
    if any(trait in payoff_traits for trait in ("asian", "barrier", "lookback", "path_dependent")) and "monte_carlo" not in families:
        families.append("monte_carlo")
    if "barrier" in payoff_traits and model_family == "equity_diffusion" and "pde" not in families:
        families.append("pde")
    if model_family == "stochastic_volatility" and "monte_carlo" not in families:
        families.append("monte_carlo")
    return tuple(families)


def _augment_ir_with_contextual_support(ir: ProductIR, description: str) -> ProductIR:
    """Augment ProductIR with high-signal request context missing from static decompositions."""
    if ir.instrument == "quanto_option":
        candidate_engine_families = list(ir.candidate_engine_families)
        for family in ("analytical", "monte_carlo"):
            if family not in candidate_engine_families:
                candidate_engine_families.append(family)

        required_market_data = set(ir.required_market_data)
        required_market_data.update(
            {"discount_curve", "forward_curve", "spot", "black_vol_surface", "fx_rates", "model_parameters"}
        )

        payoff_traits = list(ir.payoff_traits)
        for trait in ("discounting", "fx_translation", "vol_surface_dependence"):
            if trait not in payoff_traits:
                payoff_traits.append(trait)

        return replace(
            ir,
            payoff_family="vanilla_option",
            payoff_traits=tuple(payoff_traits),
            model_family="fx_cross_currency",
            candidate_engine_families=tuple(candidate_engine_families),
            required_market_data=frozenset(
                normalize_market_data_requirements(required_market_data)
            ),
        )

    if not _looks_like_fx_option_context(description, instrument_type=ir.instrument):
        return ir

    candidate_engine_families = list(ir.candidate_engine_families)
    for family in ("analytical", "monte_carlo"):
        if family not in candidate_engine_families:
            candidate_engine_families.append(family)

    required_market_data = set(ir.required_market_data)
    required_market_data.update({"fx_rates", "forward_curve", "spot"})

    return replace(
        ir,
        model_family="fx",
        candidate_engine_families=tuple(candidate_engine_families),
        required_market_data=frozenset(
            normalize_market_data_requirements(required_market_data)
        ),
    )


def _augment_ir_with_promoted_route_support(ir: ProductIR) -> ProductIR:
    """Augment ProductIR with compatible promoted route families from the live registry."""
    from trellis.agent.route_registry import load_route_registry

    route_families = list(ir.route_families)
    candidate_engine_families = list(ir.candidate_engine_families)
    exercise_style = str(getattr(ir, "exercise_style", "") or "").strip().lower()

    for route in load_route_registry().routes:
        if route.status != "promoted":
            continue
        if not _route_matches_product_ir(route, ir):
            continue
        # QUA-909: a route whose scorer declares ``non_european_penalty`` is
        # signaling that it is a lower-bound / fallback approximation for
        # non-European exercise styles and must not be advertised as a
        # first-class candidate engine family against rate-tree / PDE /
        # Monte-Carlo routes that are the true method for those products
        # (e.g. Bermudan swaption selects rate_tree, not the Black76
        # lower-bound helper). Skip the augmentation contribution for such
        # routes; their direct ``match_candidate_routes`` dispatch via the
        # scorer still works.
        if exercise_style and exercise_style != "european":
            score_hints = getattr(route, "score_hints", None) or {}
            non_european_penalty = float(
                score_hints.get("non_european_penalty", 0.0) or 0.0
            )
            if non_european_penalty < 0:
                continue

        route_family = str(getattr(route, "route_family", "") or "").strip()
        if (
            route_family
            and getattr(route, "match_instruments", None) is not None
            and ir.instrument in route.match_instruments
            and route_family not in route_families
        ):
            route_families.append(route_family)

        for engine_family in _candidate_engine_families_from_route(route.engine_family):
            if engine_family and engine_family not in candidate_engine_families:
                candidate_engine_families.append(engine_family)

    if tuple(route_families) == tuple(ir.route_families) and tuple(candidate_engine_families) == tuple(ir.candidate_engine_families):
        return ir
    return replace(
        ir,
        route_families=tuple(route_families),
        candidate_engine_families=tuple(candidate_engine_families),
    )


def _route_matches_product_ir(route, ir: ProductIR) -> bool:
    """Return whether one promoted route is structurally compatible with ProductIR."""
    instrument = getattr(ir, "instrument", None)
    exercise = getattr(ir, "exercise_style", "none")
    payoff_family = getattr(ir, "payoff_family", "")
    payoff_traits = set(getattr(ir, "payoff_traits", ()) or ())
    required_market_data = set(getattr(ir, "required_market_data", ()) or ())

    if route.exclude_instruments and instrument in route.exclude_instruments:
        return False
    if route.match_exercise is not None and exercise not in route.match_exercise:
        return False
    if route.exclude_exercise and exercise in route.exclude_exercise:
        return False
    if route.match_required_market_data is not None and not all(
        item in required_market_data for item in route.match_required_market_data
    ):
        return False
    if route.exclude_required_market_data is not None and any(
        item in required_market_data for item in route.exclude_required_market_data
    ):
        return False

    instrument_ok = route.match_instruments is not None and instrument in route.match_instruments
    payoff_family_ok = route.match_payoff_family is not None and payoff_family in route.match_payoff_family
    payoff_traits_ok = route.match_payoff_traits is not None and bool(
        payoff_traits.intersection(route.match_payoff_traits)
    )
    has_positive_filter = (
        route.match_instruments is not None
        or route.match_payoff_family is not None
        or route.match_payoff_traits is not None
    )
    if has_positive_filter and not (instrument_ok or payoff_family_ok or payoff_traits_ok):
        return False
    return has_positive_filter


def _candidate_engine_families_from_route(engine_family: str) -> tuple[str, ...]:
    """Map one route engine-family label onto ProductIR engine-family hints."""
    normalized = str(engine_family or "").strip().lower()
    mapping = {
        "analytical": ("analytical",),
        "monte_carlo": ("monte_carlo",),
        "qmc": ("qmc",),
        "pde_solver": ("pde",),
        "pde": ("pde",),
        "rate_tree": ("lattice",),
        "tree": ("lattice",),
        "lattice": ("lattice",),
        "copula": ("copula",),
        "waterfall": ("cashflow",),
        "cashflow": ("cashflow",),
    }
    return mapping.get(normalized, (normalized,) if normalized else ())


def _looks_like_fx_option_context(
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> bool:
    """Detect a vanilla FX-option context from free-form request text."""
    if instrument_type == "fx_option":
        return True
    if not description:
        return False
    lower = description.lower()
    if any(
        token in lower
        for token in ("fx option", "fx vanilla", "forex option", "garman-kohlhagen", "gk analytical")
    ):
        return True
    return re.search(r"\b[A-Z]{6}\b", description) is not None


def _route_families_for(
    instrument: str,
    payoff_family: str,
    exercise_style: str,
    model_family: str,
) -> tuple[str, ...]:
    """Return the exact route-family labels that remain semantically valid."""
    families: list[str] = []
    if payoff_family == EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY:
        families.append(EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY)
    if instrument == "nth_to_default":
        families.append("nth_to_default")
    if (
        payoff_family == "vanilla_option"
        and exercise_style in {"american", "bermudan"}
        and model_family == "equity_diffusion"
    ):
        families.append("exercise")
        families.append("equity_tree")
    if instrument == "barrier_option" and model_family == "equity_diffusion":
        families.append("pde_solver")
    if (
        instrument in {"callable_bond", "puttable_bond", "bermudan_swaption"}
        or (
            exercise_style in {"issuer_call", "holder_put", "bermudan"}
            and model_family == "interest_rate"
        )
    ):
        families.append("rate_lattice")
    return tuple(dict.fromkeys(families))


def _market_data_for_traits(
    payoff_traits: tuple[str, ...],
    store: KnowledgeStore,
) -> list[str]:
    """Infer market-data needs from known features."""
    market_data: set[str] = set()
    for trait in payoff_traits:
        feature = store._features.get(trait)
        if feature is not None:
            market_data.update(feature.market_data)
    if "barrier" in payoff_traits or "asian" in payoff_traits or "vanilla_option" in payoff_traits:
        market_data.update({"discount_curve", "black_vol_surface"})
    return sorted(market_data)


def _reusable_primitives_for(
    payoff_traits: tuple[str, ...],
    model_family: str,
) -> tuple[str, ...]:
    """Return likely reusable primitives for unsupported composites."""
    primitives: list[str] = []
    if any(trait in payoff_traits for trait in ("barrier", "asian", "lookback")):
        primitives.extend([
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ])
    if model_family == "stochastic_volatility":
        primitives.append("trellis.models.processes.heston")
    if "american" in payoff_traits:
        primitives.extend([
            "trellis.models.monte_carlo.lsm",
            "trellis.models.monte_carlo.schemes",
        ])
    return tuple(dict.fromkeys(primitives))


def _unresolved_primitives_for(
    payoff_traits: tuple[str, ...],
    exercise_style: str,
    model_family: str,
) -> tuple[str, ...]:
    """Identify explicit blockers for unsupported composite products."""
    unresolved: list[str] = []
    path_dependent = any(trait in payoff_traits for trait in ("barrier", "asian", "lookback", "path_dependent"))
    if exercise_style == "american" and path_dependent and model_family == "stochastic_volatility":
        unresolved.append("path_dependent_early_exercise_under_stochastic_vol")
    elif exercise_style == "american" and path_dependent:
        unresolved.append("path_dependent_early_exercise")
    elif exercise_style == "american" and model_family == "stochastic_volatility":
        unresolved.append("exercise_under_stochastic_vol")
    return tuple(unresolved)


def _decompose_via_llm(
    description: str,
    key: str,
    store: KnowledgeStore,
    model: str | None,
) -> ProductDecomposition:
    """Use LLM to decompose a novel product into known features."""
    from trellis.agent.config import llm_generate_json, load_env
    from trellis.agent.knowledge.retrieval import (
        format_decomposition_knowledge_for_prompt,
    )
    load_env()

    # Build feature taxonomy context
    feature_list = "\n".join(
        f"- {f.id}: {f.description}"
        + (f" (implies: {', '.join(f.implies)})" if f.implies else "")
        + (f" (method_hint: {f.method_hint})" if f.method_hint else "")
        for f in store._features.values()
    )

    # Available methods
    methods = sorted({d.method for d in store._decompositions.values()})
    heuristic_features = list(_traits_from_text(_normalise(description)))
    instrument_hint = key if key in store._decompositions else _infer_instrument(description, None)
    prior_knowledge = store.retrieve_for_task(
        RetrievalSpec(
            method=None,
            features=heuristic_features,
            instrument=instrument_hint,
            max_lessons=5,
        )
    )
    knowledge_text = format_decomposition_knowledge_for_prompt(prior_knowledge)
    knowledge_section = ""
    if knowledge_text:
        knowledge_section = f"\n\n## Shared Knowledge\n{knowledge_text}"

    prompt = f"""You are a quantitative finance expert decomposing a financial instrument
into its constituent features for a pricing library.

## Available Features
{feature_list}

## Available Pricing Methods
{', '.join(methods)}
{knowledge_section}

## Instrument to Decompose
"{description}"

## Your Task
Decompose this instrument into a list of features from the taxonomy above.
You may also propose NEW feature IDs if needed (use snake_case).
Select the most appropriate pricing method.

Return JSON:
{{
    "features": ["feature1", "feature2", ...],
    "method": "pricing_method",
    "method_modules": ["trellis.models.module1", ...],
    "required_market_data": ["discount_curve", "black_vol_surface", ...],
    "reasoning": "Brief explanation of why this decomposition and method",
    "notes": "Any known complexities or edge cases"
}}"""

    try:
        data = llm_generate_json(prompt, model)
    except Exception:
        # If LLM fails, return a minimal decomposition
        return ProductDecomposition(
            instrument=key,
            features=("discounting",),
            method=normalize_method("analytical"),
            reasoning="LLM decomposition failed — falling back to analytical.",
            learned=True,
        )

    return ProductDecomposition(
        instrument=key,
        features=tuple(data.get("features", ["discounting"])),
        method=normalize_method(data.get("method", "analytical")),
        method_modules=tuple(data.get("method_modules", [])),
        required_market_data=frozenset(data.get("required_market_data", ["discount_curve"])),
        reasoning=data.get("reasoning", ""),
        notes=data.get("notes", ""),
        learned=True,
    )
