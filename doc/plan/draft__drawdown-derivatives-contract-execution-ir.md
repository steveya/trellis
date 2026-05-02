# Drawdown Derivatives Contract/Execution IR Plan

## Status

Draft. Parked research and implementation plan. No executable support is
claimed by this document.

## Linked Context

- `doc/plan/active__contract-execution-ir-and-visitor-framework.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__automatic-event-state-lowering-plan.md`
- `doc/plan/draft__contract-ir-arithmetic-asian-solver-follow-on.md`
- `docs/quant/contract_ir.rst`
- `docs/quant/dynamic_contract_ir.rst`
- `docs/quant/differentiable_pricing.rst`
- Existing repo surfaces:
  - `trellis/agent/contract_ir.py`
  - `trellis/agent/dynamic_contract_ir.py`
  - `trellis/agent/contract_ir_solver_compiler.py`
  - `trellis/execution/compiler.py`
  - `trellis/execution/admission.py`
  - `trellis/execution/visitors/simulation_bridge.py`
  - `trellis/models/monte_carlo/path_state.py`
  - `trellis/models/monte_carlo/engine.py`

## Purpose

Keep drawdown derivatives visible as a possible future path-dependent
ContractIR / ExecutionIR family without bypassing semantic authority through a
route-local generated exotic helper.

Drawdown products depend on the gap between an asset and its running high-water
mark. Typical path state is:

```text
M_t = max_{u <= t} S_u
D_t = M_t - S_t
relative_drawdown_t = 1 - S_t / M_t
MDD_T = max_{t <= T} D_t
```

The intended architectural stance is:

- terminal max-drawdown payoffs belong on the payoff-expression `ContractIR`
  path-state surface
- first-hit drawdown insurance belongs on `DynamicContractIR`
- executable support should start with a narrow discretely monitored Monte
  Carlo lane
- continuous monitoring, Brownian-bridge correction, PDE methods, cancellation,
  American/control variants, and desk-grade Greeks are follow-ons

## Literature Anchor

The family should be researched and documented against:

- Carr, Zhang, and Hadjiliadis on maximum drawdown insurance
- Zhang, Leung, and Hadjiliadis on fair valuation and cancellation
- Pospisil and Vecer on PDE methods for maximum drawdown
- Brownian maximum-drawdown distribution literature

The literature should inform benchmark fixtures and limitations, but the first
implementation should not claim closed-form or continuous-monitoring support.

## Current Repository Boundary

Current `ContractIR` can mark observations as `path_dependent`, but its payoff
leaves do not yet include running extrema or drawdown observables. Current
decomposition also keeps lookbacks, barriers, and broad path-dependent shapes
outside the admitted ContractIR bridge.

Current `DynamicContractIR` can represent event/state/termination structure,
but the executable dynamic lane is still bounded to existing automatic,
discrete-control, and continuous-control cohorts. Drawdown first-hit insurance
therefore needs structural admission before executable pricing.

Current `ExecutionIR` is mostly an authority seam plus bounded P001 proof. A
drawdown execution lane should be added only when it has explicit capability
admission and a checked visitor.

## Target Semantics

### Terminal payoff-expression lane

Add a path-state observable, with final naming still open:

```text
DrawdownObservable(
    underlier_id,
    schedule,
    statistic = "max_drawdown" | "current_drawdown",
    basis = "absolute" | "relative",
    initial_high_water_mark = "spot" | explicit value
)
```

Admitted first payoff shapes:

- `MDD_T`
- `(MDD_T - K)+`
- `1{MDD_T >= K}`
- notional-scaled versions of the same shapes

The node should be semantic. It should not encode Monte Carlo, PDE, or any
route-local helper identity.

### Dynamic first-hit insurance lane

Represent first-hit drawdown insurance as `DynamicContractIR` with:

- state fields such as `running_max`, `current_drawdown`, and `max_drawdown`
- observation events over the monitoring schedule
- state update events for high-water mark and drawdown state
- automatic termination when drawdown crosses the contractual threshold
- contingent settlement on trigger or maturity, as applicable

This lane is structural until a checked dynamic compiler and visitor exist.

## Minimal Executable MVP

The first executable slice should be:

- single-underlier GBM
- discretely monitored terminal maximum drawdown
- Monte Carlo only
- reduced-state path accumulation, without storing full paths
- price output only unless an explicit finite-difference risk policy is added

The natural runtime primitive is the existing Monte Carlo `PathReducer`
mechanism. A drawdown reducer can stream:

```text
running_max = max(running_max, S_i)
drawdown = running_max - S_i
max_drawdown = max(max_drawdown, drawdown)
```

For relative drawdown:

```text
drawdown = 1 - S_i / running_max
```

## Proposed Queue

| Queue ID | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- |
| `DD.1` | Backlog | ContractIR path-state drawdown observable and well-formedness tests | none |
| `DD.2` | Backlog | Monte Carlo drawdown reducer and terminal payoff helper | none |
| `DD.3` | Backlog | ContractIR solver declaration and compiler binding for terminal max drawdown | `DD.1`, `DD.2` |
| `DD.4` | Backlog | ExecutionIR lowering, capability admission, and MC visitor for terminal drawdown | `DD.3`, active XIR visitor framework |
| `DD.5` | Backlog | DynamicContractIR structural admission for first-hit drawdown insurance | dynamic semantic foundation |
| `DD.6` | Backlog | Continuous monitoring, Brownian-bridge/PDE research, cancellation, and Greeks plan | executable discrete MVP evidence |

## Validation

- `tests/test_agent/test_contract_ir_types.py`
- `tests/test_agent/test_contract_ir_solver_registry.py`
- `tests/test_agent/test_contract_ir_solver_compiler.py`
- `tests/test_execution/test_execution_ir.py`
- `tests/test_models/test_monte_carlo/test_mc.py`
- drawdown-specific numerical tests comparing reduced-state payoff evaluation
  to full-path replay under the same seed
- docs and limitation updates whenever a slice changes the support contract

## Risks And Limits

- Continuous monitoring is out of scope for the MVP. Discrete monitoring should
  be stated explicitly in docs and limitations.
- Digitals and first-hit contracts are discontinuous; pathwise autodiff should
  fail closed unless a later smoothing/custom-adjoint policy is implemented.
- Max drawdown itself is nonsmooth; Greeks need explicit support metadata and
  probably a finite-difference or unsupported policy first.
- Dynamic first-hit insurance should not be priced through a route-local helper
  before the dynamic execution lane exists.
- Literature methods are model-specific. PDE or closed-form work should be
  admitted only through separate validated tickets.

## Non-goals

- Generic lookback support
- Continuous-monitoring support in the first executable slice
- Cancellable drawdown insurance
- American/control variants
- Claiming route-free dynamic execution before the dynamic compiler exists
- Adding generated `_agent` drawdown helpers as the semantic authority

## Next Steps

1. Keep this as a draft backlog plan while the current XIR and route-retirement
   queues continue.
2. Create Linear issues only when the team is ready to promote drawdown work
   into the active implementation ledger.
3. Start with `DD.2` only if a standalone runtime primitive is desired before
   the semantic observable lands; otherwise start with `DD.1`.
