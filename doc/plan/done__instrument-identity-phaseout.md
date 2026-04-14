# Instrument Identity Phaseout Plan

## Purpose

This plan shrinks `trellis/agent/instrument_identity.py` over time until only
the smallest unavoidable ingress fallback remains, or the file can be removed
entirely.

The target is not "delete the file immediately." The target is:

- infer family identity once at ingress when necessary
- carry explicit family identity structurally below that point
- forbid lower layers from re-inferring family from raw text once a family is
  already known

## Why This Plan Exists

`instrument_identity.py` was introduced as a lower-layer safety rail after
`T01` exposed an architectural leak:

- one layer knew the task was a `zcb_option`
- another lower layer still treated it as a generic `european_option`

The shared fallback mapping fixed that inconsistency quickly, but it is still a
transitional heuristic layer. It should not become permanent compiler
authority.

## End State

The desired steady state is:

```text
raw request/task text
  -> ingress identity resolution (only if explicit family missing)
  -> explicit instrument family on the request / semantic contract / ProductIR
  -> lower layers consume explicit family only
```

At that point:

- planner does not widen an explicit family into a generic spec
- task runtime does not rediscover the family from titles/descriptions
- executor does not rediscover the family from descriptions
- cached-module selection, validation, and helper binding use explicit family
  identity rather than text heuristics

## Conditional Rule

Every removal in this plan is conditional:

- remove a text fallback from a lower layer only after that boundary already
  receives explicit family identity and tests defend that fact
- if a boundary still legitimately receives raw text with no explicit family,
  keep the fallback there for now

This means the file may shrink gradually:

1. shared normalization helper remains
2. ingress-only text inference remains
3. downstream text inference disappears
4. remaining pattern table is reviewed for real ingress need

## Current Repo-Grounded State

Current direct uses are:

- `trellis/agent/task_runtime.py`
- `trellis/agent/executor.py`

The planner-side `EuropeanOptionSpec` leak has already been fixed by enforcing
"explicit family beats generic heuristic" at schema specialization time.

The remaining work is lower-layer cleanup:

- provenance for where family identity came from
- removal of downstream re-inference where explicit family is already present
- quarantine of the residual heuristic table to ingress-only behavior

## Ordered Queue

### `QUA-753` Semantic lower layers: phase out `instrument_identity` fallbacks

Umbrella objective:

Reduce `instrument_identity.py` to the smallest ingress-only responsibility the
platform still needs, and remove lower-layer text re-inference wherever
explicit family identity is already available.

### `QUA-754` Semantic ingress: materialize instrument identity and source once

Objective:

Compile instrument identity once at ingress and carry both:

- `instrument_type`
- `instrument_identity_source`

through task/runtime request metadata so downstream layers can decide
deterministically whether fallback inference is still allowed.

Acceptance:

- task/runtime metadata records whether the family came from an explicit field
  or a text fallback
- lower layers can read that provenance without re-parsing raw text

### `QUA-755` Task runtime and executor: remove downstream family re-inference

Objective:

Remove text-based family inference from task-runtime and executor paths that
already have explicit family identity from the request, compiled request, or
`ProductIR`.

Acceptance:

- executor no longer falls back to description text when explicit family is
  already present
- task-runtime helper binding and validation paths prefer threaded family
  identity over heuristic rediscovery

### `QUA-757` Cached selection and validation: require explicit family authority

Objective:

Remove remaining cached-module, validation-bundle, and reference-oracle seams
that still behave as though family identity must be rediscovered locally.

Acceptance:

- cached schema inference rejects family widening by default
- validation/oracle selection stays on explicit family authority once present

### `QUA-756` Instrument identity cleanup: quarantine residual heuristics

Objective:

Reduce `instrument_identity.py` to either:

- a tiny ingress-only utility, or
- a removable shim if ingress parsing no longer needs it

Acceptance:

- only the true ingress layer may still use text-pattern inference
- the remaining pattern table is reviewed and trimmed to actual ingress cases
- docs and traces describe the residual fallback honestly

## Validation Posture

Each slice should validate at three levels:

### Local

- task-runtime / executor tests for family identity provenance
- planner/runtime tests proving explicit family beats generic heuristics

### Regional

- cached module selection tests
- validation-bundle / reference-oracle selection tests

### Global

- rerun canaries that historically exposed family drift:
  - `T01`
  - `T25`
  - `T39`
  - `T49`
  - `T73`

## Relationship to `QUA-733`

This plan is a supporting tranche under:

- `QUA-737` route/binding decentering
- `QUA-739` adapter architecture minimization
- the broader `QUA-733` semantic platform hardening umbrella

It does not replace those tickets. It narrows one specific transitional seam so
the broader architecture can keep moving meaning out of heuristics and into
typed semantic/compiler authority.

## Linear Mirror

Status mirror last synced: `2026-04-09`

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-753` | Phase out `instrument_identity` lower-layer fallbacks | Done |
| `QUA-754` | Materialize instrument identity and source at ingress | Done |
| `QUA-755` | Remove downstream family re-inference in task runtime and executor | Done |
| `QUA-757` | Require explicit family authority in cached selection and validation | Done |
| `QUA-756` | Quarantine or remove residual heuristics | Done |
