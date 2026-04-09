# General Helper Layer Extraction Plan

## Purpose

This plan turns the current product-shaped wrapper fixes into a reusable helper
layer program.

The immediate objective is to preserve and then improve the current canary
recovery surface while moving away from helpers whose public contract is
effectively:

- "price vanilla equity option with Monte Carlo"
- "price vanilla equity option with transforms"

Those wrappers were useful because they removed low-level glue generation, but
they are not the final abstraction boundary.

## Target Outcome

Trellis should expose general family helper layers that cover semantic-to-
runtime assembly for bounded problem classes, so the agent can price a wider
range of products without inventing product-local glue.

The first acceptance set for this plan is:

- `T25`
- `T26`
- `T39`
- `T40`
- `T49`

The goal is:

- keep `T25`, `T26`, `T39`, and `T40` green throughout the refactor
- recover `T49` by replacing the current raw copula/nth-to-default helper
  boundary with a semantic-facing basket-credit helper surface

## Why This Plan Exists

The recent wrapper work proved something important:

- tasks fail when the public helper surface is too low-level
- tasks pass when the semantic-to-runtime bridge is owned by checked-in code

But the current wrappers still bundle too much product-specific logic. They are
best understood as transitional compatibility layers.

We now want to extract the reusable layers underneath them:

1. market / convention / quote resolvers
2. model / process / characteristic-function builders
3. event/control-aware family problem assemblers
4. payoff / reducer components
5. thin semantic-facing wrappers that compose the above

## Current Repo-Grounded State

The current repo already has two successful transitional wrappers:

- `trellis/models/equity_option_monte_carlo.py`
- `trellis/models/equity_option_transforms.py`

These helped recover:

- `T25`
- `T26`
- `T39`
- `T40`

The remaining red canary in this target set is `T49`, which still exposes a
different weakness:

- the current copula path is using the raw nth-to-default helper boundary
- the task is a CDO tranche comparison, not an nth-to-default payoff
- the semantic/product shape and the public helper contract are misaligned

So the next stage is not "add more product wrappers." It is "extract reusable
helper layers and then rebuild the wrappers and basket-credit path on top of
them."

## Design Principles

### 1. Helper layers should be family-shaped, not product-shaped

Good:

- single-state diffusion market/model resolver
- event-aware Monte Carlo problem assembler
- transform pricing helper kit
- basket-credit dependence and tranche-loss helper kit

Bad:

- one helper per product x method pair
- raw kernel exposed as the public route helper

### 2. Public helper surfaces should begin from semantic inputs

The public helper should usually accept:

- `market_state`
- `spec`
- bounded family controls

and own:

- convention hydration
- market binding
- model input resolution
- family problem assembly
- execution and result shaping

Lower-level kernels should remain available, but not as the primary route
helper.

### 3. Wrappers should become thin compositions

Existing wrapper modules are allowed to remain for compatibility, but they
should shrink over time into thin compositions of shared helper layers.

### 4. Canary recovery is the acceptance surface

Every slice in this plan must preserve or improve the current status of:

- `T25`
- `T26`
- `T39`
- `T40`
- `T49`

## Proposed Helper Layer Stack

### A. Shared single-state diffusion helper kit

Reusable responsibilities:

- settlement / maturity normalization
- option-style normalization
- spot / discount / dividend / vol resolution
- Black/GBM-compatible characteristic function construction
- common transform and Monte Carlo parameter defaults

Initial consumers:

- `equity_option_monte_carlo.py`
- `equity_option_transforms.py`

### B. Event-aware Monte Carlo assembly helpers

Reusable responsibilities:

- map resolved process inputs into event-aware MC problem specs
- choose scheme / path requirements / reducers
- execute the bounded event-aware MC runtime

Initial consumers:

- vanilla European equity MC
- later rate/event-aware MC slices

### C. Transform assembly helpers

Reusable responsibilities:

- map resolved process inputs into FFT/COS-compatible characteristic functions
- choose method-family controls
- apply put/call parity where needed

Initial consumers:

- vanilla equity FFT/COS
- later Heston smile / transform surfaces

### D. Basket-credit dependence and tranche helper kit

Reusable responsibilities:

- normalize basket/tranche semantics
- bind representative credit curve and discount curve inputs
- map tranche attachment/detachment and recovery conventions
- assemble Gaussian / Student-t basket-loss engines
- expose semantic-facing tranche and nth-to-default wrappers

Initial consumers:

- `T49` CDO tranche comparison
- later `T50` / `T53`

## Ordered Delivery Queue

### `QUA-741` Shared helper layers: single-state diffusion resolver kit

Objective:

Extract reusable spot/discount/dividend/vol/maturity resolution and
characteristic-function support from the current vanilla MC/transform wrappers.

Scope:

- new shared helper module(s) under `trellis/models/`
- refactor `equity_option_monte_carlo.py`
- refactor `equity_option_transforms.py`

Acceptance:

- wrappers become thin compositions over the shared layer
- targeted unit tests pass
- `T25`, `T26`, `T39`, `T40` still pass

### `QUA-742` Monte Carlo helper layers: event-aware MC family composition

Objective:

Refactor the vanilla-equity MC wrapper to use reusable event-aware Monte Carlo
assembly helpers instead of directly owning all family wiring.

Acceptance:

- `equity_option_monte_carlo.py` shrinks materially
- family-level MC assembly helpers become reusable by other products
- `T25` and `T26` remain green

### `QUA-743` Transform helper layers: reusable transform assembly surface

Objective:

Refactor the vanilla-equity transform wrapper to use reusable transform helper
layers rather than directly owning characteristic-function and dispatch logic.

Acceptance:

- `equity_option_transforms.py` shrinks materially
- public transform helper surface is family-shaped
- `T39` and `T40` remain green

### `QUA-744` Credit basket helper layers: tranche and nth-to-default semantic helpers

Objective:

Introduce a semantic-facing basket-credit helper kit that handles both
nth-to-default and tranche-style basket-loss problems without exposing a raw
scalar kernel as the route helper.

Acceptance:

- `T49` passes
- current nth-to-default helper path stays compatible
- basket-credit helper contracts are explicit about tranche attachment,
  detachment, dependence family, and quote/report semantics

Status:

- `Done`

Implemented:

- added ``trellis.models.credit_basket_copula`` as the semantic-facing
  basket-credit helper layer for both nth-to-default compatibility and
  tranche-style basket-loss pricing
- taught semantic drafting and validation about the
  ``credit_basket_tranche`` contract instead of collapsing tranche requests
  into ``nth_to_default``
- updated semantic validation so a helper-backed basket-credit route can treat
  the public helper as the assembly boundary rather than demanding direct
  loss-distribution primitive calls in generated adapters

Validation:

- targeted unit and compiler slices pass
- live ``T49`` canary passes

Acceptance set status after `QUA-744`:

- `T25` pass
- `T26` pass
- `T39` pass
- `T40` pass
- `T49` pass

### `QUA-745` Helper layers: docs, observability, canary hardening

Objective:

Document the new helper layer stack, update traces/observability to reflect the
new family helper surfaces, and refresh the canary mirror.

Acceptance:

- official docs updated
- canary mirror updated
- acceptance reruns for `T25`, `T26`, `T39`, `T40`, `T49`

## Acceptance Matrix

| Ticket | `T25` | `T26` | `T39` | `T40` | `T49` |
| --- | --- | --- | --- | --- | --- |
| `QUA-741` | preserve | preserve | preserve | preserve | no regression |
| `QUA-742` | green | green | preserve | preserve | no regression |
| `QUA-743` | preserve | preserve | green | green | no regression |
| `QUA-744` | preserve | preserve | preserve | preserve | green |
| `QUA-745` | verify | verify | verify | verify | verify |

## Current Status

- `QUA-741` is complete.
- The shared single-state diffusion resolver layer now exists under
  `trellis.models.resolution`.
- `T25`, `T26`, `T39`, and `T40` are green on the refactored helper surface.
- `QUA-744` is complete.
- `T49` is now green on the semantic-facing basket-credit helper layer.
- The full target acceptance set (`T25`, `T26`, `T39`, `T40`, `T49`) is green.
- The fresh canary rerun on `2026-04-09` confirmed all five acceptance tasks in
  a clean `14/14` curated batch, recorded in
  `canary_results_20260409_full_rerun_budget10.json`.

## Immediate Next Step

Proceed to `QUA-745`.

The remaining work in this plan is closeout and hardening:

- sync the doc/observability tranche to the new helper boundaries
- keep the acceptance set stable in subsequent full canary reruns
- use the shared helper-layer patterns here as the template for the next family
  extractions rather than adding new product-shaped wrappers
