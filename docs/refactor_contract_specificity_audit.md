# Contract Specificity Audit

This note records where Trellis contracts were more specific than the correct
abstraction level, and what remains to migrate.

## Fixed in this tranche

### Monte Carlo early exercise

Old contract:
- Monte Carlo early exercise effectively meant `longstaff_schwartz`

New contract:
- Monte Carlo early exercise means an approved optimal-stopping policy class
- current approved classes:
  - `longstaff_schwartz`
  - `tsitsiklis_van_roy`
  - `primal_dual_mc`
  - `stochastic_mesh`

Implementation honesty:
- only `longstaff_schwartz` is implemented
- semantic validation recognizes the broader family
- import validation still blocks nonexistent planned imports

### Regression basis inside MC early exercise

Old contract:
- `LaguerreBasis` effectively appeared as a required primitive of the route

New contract:
- basis choice is a continuation-estimator detail
- route planning now treats it as an adapter/configuration choice, not a
  mandatory primitive

## Remaining overly-specific surfaces

### Non-canonical knowledge still contains older Longstaff-Schwartz wording

Canonical files updated in this tranche:
- `trellis/agent/knowledge/canonical/method_requirements.yaml`
- `trellis/agent/knowledge/canonical/api_map.yaml`
- `trellis/agent/knowledge/canonical/cookbooks.yaml`
- `trellis/agent/knowledge/canonical/failure_signatures.yaml`

They now distinguish:
- policy-family abstraction
- current concrete implementation
- optional basis / estimator choice

### Non-canonical lessons and experience still reference LSM more narrowly

Files:
- `trellis/agent/experience.yaml`
- `trellis/agent/knowledge/lessons/entries/*.yaml`

Issue:
- some entries still describe American/Bermudan MC as if
  `longstaff_schwartz` were the only valid policy class

Recommendation:
- migrate them opportunistically when those entries are next touched or promoted

### `analytical_black76` route naming is implementation-first

Issue:
- the route name mixes abstraction and concrete kernel choice
- for vanilla equity options, the deeper abstraction is "closed-form forward /
  discount / volatility vanilla pricing"

Why I did not change it here:
- the current implementation really is built around `black76_call` /
  `black76_put`
- renaming the route would ripple through planning, tests, and knowledge

Recommendation:
- revisit later if we add multiple closed-form vanilla kernels

### Transform route planning still centers `fft_price`

Issue:
- `transform_fft` currently carries `fft_price` as required and `cos_price` as
  optional in one route plan

Why this is acceptable for now:
- comparison-task planning already preserves `fft` vs `cos` as distinct targets
- the remaining issue is route-family naming, not correctness

Recommendation:
- revisit when transform-family planning becomes more granular

## Principle going forward

When designing contracts, prefer:

- route family / mathematical construct

over:

- one current algorithm or one current import path

Then keep a separate layer for:

- which concrete primitive is currently implemented
- which cookbook/example is the default
- which imports are currently valid
