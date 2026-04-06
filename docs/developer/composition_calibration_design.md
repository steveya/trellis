# Composition Algebra and Calibration Contract Design

**QUA-413 design spike.** This document defines how Trellis expresses
structured products as compositions of typed sub-contracts, and how
calibration steps are declared as typed contracts.

## 1. Design Principles

1. **Components are atoms; products are molecules.** Every derivative
   decomposes into a small set of typed payoff components. The LLM
   assembles molecules from atoms rather than writing monolithic
   `evaluate()` functions.

2. **Composition is a DAG, not a flat list.** Components have typed edges
   (sequential, conditional, parallel, override) that express temporal and
   logical relationships, and explicit control boundaries capture Bellman-style
   issuer/holder choice without overloading those edges.

3. **Method resolution is compositional.** Each component declares
   compatible methods. The compiler intersects them, and a dominance rule
   resolves conflicts.

4. **The existing contract model is extended, not replaced.**
   `CompositeSemanticContract` wraps multiple `SemanticContract` nodes.
   Simple products continue to use flat contracts.

5. **The basket proving case (QUA-284) is the first instantiation.**
   Every abstraction must be validated against the ranked-observation
   basket before it is accepted.

## 2. PayoffComponent: The Atom

```python
@dataclass(frozen=True)
class PayoffComponent:
    """A single typed payoff building block."""

    component_id: str
    component_type: str
    # Typed interface
    inputs: tuple[ComponentPort, ...]
    outputs: tuple[ComponentPort, ...]
    # Constraints
    compatible_methods: tuple[str, ...]
    market_data_requirements: frozenset[str]
    # Validation
    semantic_validators: tuple[str, ...]
    financial_invariants: tuple[str, ...]
    # Metadata
    description: str = ""
    proven_primitive: str | None = None  # module.symbol if reusable
```

### Component Types (Initial Catalog)

| Type | Description | Example | Compatible Methods |
|------|------------|---------|-------------------|
| `barrier` | Knock-in/out condition on underlier path | Up-and-out at 120 | MC, PDE |
| `coupon_stream` | Scheduled fixed or floating payments | Semi-annual 5% fixed | Analytical, Tree |
| `exercise_policy` | Early exercise decision rule | Bermudan call at par | Tree, MC (LSM) |
| `knock_condition` | Discrete observation trigger | Autocall at 105% | MC |
| `aggregation` | Terminal payoff assembly from state | Average locked returns | Any |
| `observation_schedule` | Ordered dates for path sampling | Monthly for 3 years | MC |
| `selection_rule` | Ranked/best/worst constituent pick | Best-of remaining | MC |
| `lock_remove` | State mutation: lock return, remove name | Lock + remove selected | MC |
| `maturity_settlement` | Terminal value computation + discounting | Average + discount | Any |
| `discount_leg` | Present-value discounting of cashflows | OIS discounting | Any |
| `credit_leg` | Default-contingent payment stream | CDS protection leg | Analytical, MC |
| `correlation_structure` | Multi-asset dependency model | Gaussian copula | MC |

### ComponentPort (Typed Interface)

```python
@dataclass(frozen=True)
class ComponentPort:
    """A named, typed input or output of a component."""

    name: str
    port_type: str  # "scalar", "array", "schedule", "state", "mask"
    description: str = ""
    optional: bool = False
```

## 3. CompositeSemanticContract: The Molecule

```python
@dataclass(frozen=True)
class CompositionEdge:
    """A typed relationship between two components."""

    source: str        # component_id
    target: str        # component_id
    edge_type: str     # "sequential", "conditional", "parallel", "override"
    condition: str = ""  # for conditional edges

@dataclass(frozen=True)
class ControlBoundary:
    """Explicit Bellman-style choice boundary over component branches."""

    boundary_id: str
    controller_component: str
    style: ControlStyle
    branches: tuple[str, ...]
    label: str = ""

@dataclass(frozen=True)
class CompositeSemanticContract:
    """A structured product expressed as a DAG of components."""

    composite_id: str
    description: str
    components: tuple[PayoffComponent, ...]
    edges: tuple[CompositionEdge, ...]
    control_boundaries: tuple[ControlBoundary, ...] = ()
    # Derived
    market_data_union: frozenset[str]  # union of all component requirements
    method_intersection: tuple[str, ...]  # intersection of compatible methods
    dominant_component: str | None  # component_id that wins method conflicts
    # Validation
    composite_validators: tuple[str, ...] = ()
    # Link to base contract model
    base_contract: object | None = None  # SemanticContract for compatibility

    def validate_dag(self) -> tuple[str, ...]:
        """Check DAG is acyclic, connected, ports are wired, and control
        boundaries refer to compatible priced branches."""
        ...
```

### Explicit Control Boundaries

Conditional and parallel edges are useful for dependency flow, but they are too
weak to carry exercise semantics by themselves. The Bellman layer now treats
issuer and holder choice as explicit control boundaries:

- the controller is usually an ``exercise_policy`` component
- the branches are the priced continuation or exercise alternatives
- the boundary style records whether rollback applies ``max`` or ``min``

This is the bridge from the composition proof-of-concept into the DSL algebra
in ``trellis.agent.dsl_algebra.ChoiceExpr``.

### Method Conflict Resolution

```python
@dataclass(frozen=True)
class MethodResolution:
    """Result of resolving method conflicts across components."""

    resolved_method: str
    resolution_kind: str  # "intersection", "dominance", "conflict"
    dominant_component: str | None
    overridden_components: tuple[str, ...] = ()
    reason: str = ""

def resolve_method_conflicts(
    components: tuple[PayoffComponent, ...],
) -> MethodResolution:
    """
    1. Intersect compatible_methods across all components.
    2. If intersection is non-empty, use it (pick preferred from intersection).
    3. If empty, find the component with exercise_policy type — it dominates.
    4. If no exercise component, flag as unresolvable conflict.
    """
```

**Dominance rule:** The component owning the exercise feature wins because
exercise determines the pricing method's numerical backbone (backward
induction for trees, LSM for MC). Other components are evaluated within
that backbone's framework.

## 4. Compiler Behavior

```python
def compile_composite(
    composite: CompositeSemanticContract,
) -> CompositeGenerationPlan:
    """Flatten a composite contract into a generation plan.

    1. Resolve method conflicts → single method
    2. Union market data requirements
    3. For each component:
       a. If proven_primitive exists → mark as reusable
       b. Else → mark as needs_generation
    4. Match route from route_registry using component features
    5. Emit skeleton with typed hooks at component boundaries
    """
```

### Skeleton emission with component hooks

The compiler emits sub-contract boundaries as named methods in the
generated class:

```python
class AutocallablePayoff:
    def evaluate(self, market_state):
        # Component: observation_schedule (proven)
        dates = self._component_observation_schedule(market_state)
        # Component: knock_condition (needs generation)
        triggered = self._component_knock_condition(dates, market_state)
        # Component: coupon_stream (proven)
        coupons = self._component_coupon_stream(dates, triggered, market_state)
        # Component: maturity_settlement (proven)
        return self._component_maturity_settlement(coupons, triggered, market_state)

    def _component_observation_schedule(self, market_state):
        # PROVEN: delegates to generate_schedule()
        return generate_schedule(self.spec.start, self.spec.end, self.spec.frequency)

    def _component_knock_condition(self, dates, market_state):
        # NEEDS GENERATION: LLM fills this in
        raise NotImplementedError

    def _component_coupon_stream(self, dates, triggered, market_state):
        # PROVEN: delegates to existing coupon primitives
        ...

    def _component_maturity_settlement(self, coupons, triggered, market_state):
        # PROVEN: discount and aggregate
        ...
```

The LLM's task is reduced: generate only the `_component_knock_condition`
method body, not the entire `evaluate()`.

### Integration with route_registry

```python
def match_composite_route(
    composite: CompositeSemanticContract,
    registry: RouteRegistry,
) -> tuple[RouteSpec, ...]:
    """Match routes using component features, not product name."""
    composite_ir = ProductIR(
        instrument=composite.composite_id,
        payoff_family=_dominant_payoff_family(composite),
        payoff_components=composite.components,
        exercise_style=_extract_exercise_style(composite),
        required_market_data=composite.market_data_union,
    )
    return match_candidate_routes(registry, composite.method_intersection[0], composite_ir)
```

## 5. ProductIR Extension

```python
@dataclass(frozen=True)
class ProductIR:
    # Existing fields (unchanged)
    instrument: str
    payoff_family: str
    payoff_traits: tuple[str, ...] = ()
    exercise_style: str = "none"
    state_dependence: str = "terminal_markov"
    schedule_dependence: bool = False
    model_family: str = "generic"
    candidate_engine_families: tuple[str, ...] = ()
    route_families: tuple[str, ...] = ()
    required_market_data: frozenset[str] = frozenset()
    reusable_primitives: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    supported: bool = True
    event_machine: object | None = None
    # NEW: composition support
    payoff_components: tuple[PayoffComponent, ...] = ()
    composition_edges: tuple[CompositionEdge, ...] = ()
    dominant_method: str | None = None
```

**Backward compatible:** Simple products have `payoff_components = ()`.
The route matcher checks `payoff_components` only when non-empty;
otherwise uses flat `payoff_family` matching (existing behavior).

## 6. CalibrationContract

```python
@dataclass(frozen=True)
class CalibrationTarget:
    """What parameter to calibrate."""
    parameter: str        # "hw_mean_reversion", "sabr_alpha", "local_vol_surface"
    output_capability: str  # MarketState capability name for the result
    quote_map: QuoteMapSpec | None = None  # "Price", "ImpliedVol(Black)", "Spread", ...

@dataclass(frozen=True)
class CalibrationContract:
    """Typed calibration step executed before pricing."""

    calibration_id: str
    target: CalibrationTarget
    fitting_instruments: tuple[str, ...]    # "atm_swaptions_1y_10y", "cap_vols"
    optimizer: str                          # "analytical", "least_squares", "differential_evolution"
    acceptance_criteria: CalibrationAcceptanceCriteria
    output_binding: str                     # MarketState capability: "hw_short_rate_params"
    proven_primitive: str | None = None     # existing calibration module.symbol

@dataclass(frozen=True)
class CalibrationAcceptanceCriteria:
    """When is calibration good enough?"""
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    stability_check: bool = True
    max_fitting_error_bps: float = 5.0
```

### Quote maps as first-class calibration semantics

The calibration contract now assumes that quote conventions are explicit rather
than buried inside each workflow. Trellis uses a bounded quote-map surface in
``trellis.models.calibration.quote_maps`` to describe how a calibration target
is assembled and how repriced residuals are reported:

- ``Price``
- ``ImpliedVol(Black)``
- ``ImpliedVol(Normal)``
- ``ParRate``
- ``Spread``
- ``Hazard``

Each quote map is two-sided where applicable:

- quote-to-price for objective-target assembly
- price-to-quote for residual reporting back in market quote units

Transform failures are explicit and become calibration provenance instead of
route-local exceptions. That matters especially for implied-vol inversion and
for rates workflows where the quote contract must preserve explicit
discount-curve and forecast-curve roles under multi-curve pricing.

Reduced-form credit now uses the same surface. In practice that means the
``Spread`` and ``Hazard`` quote maps also carry the potential-binding metadata
needed by CDS-style workflows: recovery, discount-curve role, default-curve
role, and the risky-discount contract that combines discounting with survival.

### Output binding as MarketState capabilities

Calibrated parameters are materialized as `MarketState` capabilities:

```python
# Before pricing:
calibrated = run_calibration(calibration_contract, market_state)
enriched_market_state = market_state.with_capability(
    calibration_contract.output_binding,
    calibrated.result,
)
# Pricing code accesses:
hw_params = market_state.hw_short_rate_params
```

This is consistent with how `market_state.discount`, `market_state.vol_surface`,
etc. already work. No new infrastructure needed.

For the migrated workflow surface, that handoff is now explicit in
`trellis.models.calibration.materialization`. Those helpers still populate the
compatibility fields that existing pricing code expects, but they also record a
typed binding record under `market_provenance["calibrated_objects"]` and the
current selection under `market_provenance["selected_calibrated_objects"]`.

That gives downstream review and replay tooling one authoritative place to
inspect:

- what calibrated object family was materialized
- which named runtime object is selected
- the source kind and source ref for the binding
- any explicit multi-curve discount / forecast role selections

The supported lookup surface is
`MarketState.materialized_calibrated_object(object_kind=..., object_name=...)`.

### Canonical model-grammar registry

The planner and shared retrieval surface now also read a canonical calibration
registry from `trellis/agent/knowledge/canonical/model_grammar.yaml`.

That registry is intentionally descriptive:

- it records the supported model family, quote families, calibration workflow,
  runtime materialization kind, and deferred scope
- it gives the planner and prompt surfaces a stable lookup table for supported
  calibration workflows
- it does not approve unsupported routes by itself and it does not replace the
  route registry, import registry, or code/docs authority

In practice, the code-level engine-model spec, quote-map surface, and runtime
materialization helpers remain authoritative. The canonical registry mirrors
that shipped boundary so the planner can look it up instead of reconstructing
it from scratch.

### Validation hardening at the model-grammar boundary

The migrated calibration boundary is now defended by a fixed replay/benchmark
pack plus targeted negative canaries:

- replay and benchmark coverage includes Hull-White, SABR, Heston, local vol,
  and single-name credit
- rates replay and benchmark artifacts now carry explicit multi-curve
  discount/forecast role metadata so contract drift is detectable
- the benchmark fixtures now consume the same seeded mock-snapshot contracts
  used by proving runs: SABR, Heston, and local-vol canaries read the
  ``synthetic_generation_contract`` surface, while the single-name credit
  canary reads the derived ``model_consistency_contract`` compatibility packet
- negative tests explicitly defend missing calibration binding, unsupported
  quote-map families, and invalid calibrated-object materialization kinds

### Integration with GenerationPlan

```python
@dataclass(frozen=True)
class GenerationPlan:
    # Existing fields...
    # NEW: calibration pre-steps
    calibration_steps: tuple[CalibrationContract, ...] = ()
```

The executor runs calibration steps before code generation:

```python
for cal_step in generation_plan.calibration_steps:
    if cal_step.proven_primitive:
        # Reuse existing calibration
        result = run_proven_calibration(cal_step)
    else:
        # Generate calibration code (rare)
        result = generate_calibration(cal_step)
    market_state = market_state.with_capability(cal_step.output_binding, result)
```

## 7. Worked Examples

### Example A: Callable Range Accrual

```
callable_range_accrual = CompositeSemanticContract(
    components=(
        PayoffComponent("coupon", "coupon_stream",
            compatible_methods=("analytical", "rate_tree", "monte_carlo")),
        PayoffComponent("range", "knock_condition",
            compatible_methods=("monte_carlo",)),
        PayoffComponent("call", "exercise_policy",
            compatible_methods=("rate_tree",)),
        PayoffComponent("discount", "discount_leg",
            compatible_methods=("analytical", "rate_tree", "monte_carlo")),
    ),
    edges=(
        CompositionEdge("coupon", "range", "conditional"),  # coupon accrues only in range
        CompositionEdge("range", "call", "parallel"),        # call overlays range
        CompositionEdge("call", "discount", "sequential"),   # discount after exercise
    ),
)
```

**Method resolution:** range requires MC, call requires tree. Intersection
is empty. Call has exercise_policy → dominance wins → **rate_tree**. Range
condition is evaluated within the tree's backward induction as a node-value
modifier.

### Example B: Ranked-Observation Basket (QUA-284 Proving Case)

```
himalaya_basket = CompositeSemanticContract(
    components=(
        PayoffComponent("obs", "observation_schedule",
            compatible_methods=("monte_carlo",)),
        PayoffComponent("select", "selection_rule",
            compatible_methods=("monte_carlo",)),
        PayoffComponent("lock", "lock_remove",
            compatible_methods=("monte_carlo",)),
        PayoffComponent("settle", "maturity_settlement",
            compatible_methods=("monte_carlo", "analytical")),
    ),
    edges=(
        CompositionEdge("obs", "select", "sequential"),
        CompositionEdge("select", "lock", "sequential"),
        CompositionEdge("lock", "settle", "sequential"),
    ),
)
```

**Method resolution:** All components compatible with MC → **monte_carlo**.
No conflict.

### Example C: Barrier Option on Callable Bond (Method Conflict)

```
barrier_callable = CompositeSemanticContract(
    components=(
        PayoffComponent("bond", "coupon_stream",
            compatible_methods=("analytical", "rate_tree")),
        PayoffComponent("call", "exercise_policy",
            compatible_methods=("rate_tree",)),
        PayoffComponent("barrier", "barrier",
            compatible_methods=("monte_carlo", "pde_solver")),
        PayoffComponent("discount", "discount_leg",
            compatible_methods=("analytical", "rate_tree", "monte_carlo")),
    ),
    edges=(
        CompositionEdge("bond", "call", "parallel"),
        CompositionEdge("call", "barrier", "conditional"),
        CompositionEdge("barrier", "discount", "sequential"),
    ),
)
```

**Method resolution:** call requires tree, barrier requires MC/PDE.
Intersection is empty. Call has exercise_policy → dominance → **rate_tree**.
Barrier is evaluated as a node-value test within backward induction.

### Example D: Hull-White Calibration for Callable Bond

```
hw_calibration = CalibrationContract(
    calibration_id="hw_callable_bond",
    target=CalibrationTarget(
        parameter="hw_mean_reversion",
        output_capability="hw_short_rate_params",
    ),
    fitting_instruments=("atm_swaptions_1y_10y",),
    optimizer="analytical",
    acceptance_criteria=CalibrationAcceptanceCriteria(
        max_fitting_error_bps=5.0,
    ),
    output_binding="hw_short_rate_params",
    proven_primitive="trellis.models.calibration.rates.calibrate_hull_white",
)
```

**Pipeline:** calibrate HW → enrich MarketState with `hw_short_rate_params`
→ build rate lattice using calibrated params → price callable bond.

## 8. Integration Notes

### Route Registry

Composite products generate a `ProductIR` with `payoff_components`.
`match_candidate_routes()` uses component features for matching when
`payoff_components` is non-empty. Each component's features are expanded
via the feature taxonomy's `implies` chains and unioned for retrieval.

### Semantic Validators

Each `PayoffComponent` declares which semantic validators apply. The
composite validator runs each component's validators, then composite-level
checks (DAG validity, market data union consistency, method resolution
soundness).

### Knowledge Retrieval

Each component contributes features to knowledge retrieval. For a callable
range accrual, the feature set is the union of:
- coupon_stream features: `[fixed_coupons, discounting]`
- knock_condition features: `[range_condition, path_dependent]`
- exercise_policy features: `[callable, early_exercise, backward_induction]`
- discount_leg features: `[discounting]`

Lessons matching any component feature are retrieved.

### QUA-286/287 Implementability

- **QUA-286 (validator rules):** Implement `validate_observation_schedule`,
  `validate_ranked_selection`, `validate_lock_remove`,
  `validate_maturity_aggregation` as component-level validators. Run via
  `CompositeSemanticContract.validate_dag()` + per-component dispatch.

- **QUA-287 (compiler routing):** Implement `compile_composite()` for the
  basket case. All 4 components are MC-compatible → no conflict resolution
  needed. Route matches `correlated_basket_monte_carlo`. Skeleton emits
  4 component hooks; all 4 have proven primitives in semantic_basket.py.
