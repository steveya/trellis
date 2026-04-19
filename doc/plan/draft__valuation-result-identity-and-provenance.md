# Valuation Result Identity And Provenance

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__market-coordinate-overlay-and-shock-model.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__valuation-session-and-request-surface.md`
- `doc/plan/draft__portfolio-path-and-result-set-surface.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- existing repo surfaces:
  - `docs/quant/contract_algebra.rst`
  - `trellis.book.ScenarioResultCube`
  - `trellis.curves.bootstrap.SolveProvenance`

## Purpose

Define the route-free identity and provenance packet that priced results
should carry once route ids stop being the primary operator-facing key.

The design is inspired by the useful parts of `gs-quant`'s `RiskKey`
and result wrappers, but it must reflect Trellis' semantic-compiler
architecture:

- structural declaration is the authority
- valuation policy and market identity are first-class
- route ids, if retained, are compatibility metadata only

## Why This Is Needed

Route retirement removes an old form of identity:

- route id
- route family
- instrument string

If nothing replaces that cleanly, operators lose the ability to answer:

- what was priced
- under which valuation policy
- against which market snapshot
- with which overlay or scenario
- through which exact helper or kernel

So the route-free program needs a positive replacement, not just a
negative deletion rule.

## Core Distinctions

### 1. Identity vs provenance

These should stay separate:

- **identity**: the stable key for "what valuation result is this?"
- **provenance**: the supporting record for "how was it produced?"

Identity should stay small and stable. Provenance may be richer and more
verbose.

### 2. Authority vs compatibility

Once a family is migrated:

- structural declaration id, valuation identity, and market identity are
  authority surfaces
- route aliases are compatibility-only metadata

### 3. Market identity vs scenario identity

A valuation result may depend on:

- a base market snapshot
- an overlay or scenario applied to that base

Those must be distinguishable. "scenario P&L" is not the same thing as
"same result but with a different label."

### 4. Contract identity vs trade / position identity

This note is about valuation-result identity for the priced semantic
contract.

Trade-envelope or position metadata may still matter around that value,
but the primary valuation identity should not silently collapse those
surfaces into `contract_identity`.

That boundary is tracked separately in
`doc/plan/draft__semantic-contract-target-and-trade-envelope.md`.

## Candidate Surface

Exact names may change, but the first useful shape should be close to:

```text
ValuationIdentityKey =
    { contract_identity: ContractIdentity
    ; declaration_id: str
    ; requested_method: str
    ; valuation_policy_identity: str | dict
    ; market_identity: str | dict
    ; market_overlay_identity: str | dict | None
    ; requested_output_identity: tuple[str, ...] | dict
    }

ValuationProvenance =
    { helper_refs: tuple[str, ...]
    ; kernel_refs: tuple[str, ...]
    ; binding_ids: tuple[str, ...]
    ; validation_bundle_ids: tuple[str, ...]
    ; generic_term_groups: tuple[str, ...]
    ; resolved_market_coordinates: tuple[MarketCoordinate, ...] | tuple
    ; replay_artifacts: dict
    ; compatibility_aliases: dict
    }
```

`ContractIdentity` here should be structural, not route-based. For the
current program that likely means some combination of:

- structural declaration id
- contract family / pattern identity
- optional canonical contract hash or structural summary

## Immediate Relevance To Phase 3 And 4

### Phase 3

Bound solver calls and compiler decisions should already attach the
minimal route-free valuation identity:

- declaration id
- valuation policy identity
- market identity
- overlay/scenario identity when present

They should also attach provenance sufficient for:

- helper/kernel refs
- validation bundle refs
- generic term groups consumed
- optional resolved market-coordinate references

### Phase 4

Traces, scorecards, and replay summaries should switch to this surface
as the primary identity before route ids are demoted or removed from
operator-facing meaning.

## Relationship To Existing Trellis Surfaces

### `construction_identity`

`docs/quant/contract_algebra.rst` already distinguishes construction
identity from route aliases. This note extends that idea down to the
valuation-result level.

### `ScenarioResultCube`

`ScenarioResultCube` already stores scenario provenance, but the current
surface is scenario-centric. This note defines the valuation-identity
key that one cube cell or one direct valuation result should carry.

### Valuation-session and result-set follow-ons

This note deliberately stays at the identity/provenance layer.

Two adjacent surfaces are tracked separately:

- `doc/plan/draft__valuation-session-and-request-surface.md` for
  session-scoped valuation controls and typed requested-output specs
- `doc/plan/draft__portfolio-path-and-result-set-surface.md` for
  path-aware nested result containers and projections

### `SolveProvenance`

Calibration and bootstrap provenance already exist for some lower
layers. This note does not replace them. It says how those lower-layer
provenance packets should attach to a priced result.

## Identity Invariants

For migrated fresh-build surface:

- changing route aliases alone must not change the valuation identity
- changing trade-envelope or booking metadata alone should not change
  the valuation identity unless that metadata is explicitly modeled as
  semantic or valuation authority
- changing valuation policy may change the valuation identity
- changing market snapshot or overlay may change the valuation identity
- changing requested outputs may change the valuation identity

This is the route-free analogue of the Phase 4 metadata-masking rule.

## Non-Goals

- Do not make the provenance packet so large that it becomes a second
  raw trace dump.
- Do not require every result to expose exact point-level market
  coordinates immediately.
- Do not make route aliases disappear from replay metadata before
  replay consumers are migrated.
- Do not collapse scenario provenance, calibration provenance, and
  helper provenance into one opaque string.

## Ordered Follow-On Queue

### V1 — Minimal valuation identity for structural compiler results

Objective:

Land the smallest stable identity packet on Phase 3 compiler decisions.

Acceptance:

- structural declaration id is primary
- market identity and requested-output identity are attached
- route aliases are absent from the primary key

### V2 — Rich provenance packet for helper-bound values

Objective:

Attach helper/kernel refs, validation bundle refs, and optional market
coordinate refs to the priced result provenance.

Acceptance:

- at least one migrated family emits the richer provenance packet
- route ids are compatibility metadata, not primary identity

### V3 — Trace and scorecard migration

Objective:

Use valuation identity and provenance as the primary operator-facing
surface in traces and scorecards.

Acceptance:

- operator tooling can identify values without route ids
- replay packages still retain compatibility aliases where required

### V4 — Scenario and cube alignment

Objective:

Align direct valuation identity with `ScenarioResultCube` and later
future-value cube semantics.

Acceptance:

- cube cells and direct values can share the same valuation identity
  vocabulary
- scenario/overlay identity stays explicit and separate from the base
  market identity

## Risks To Avoid

- **Identity collapse back to route.** If `route_id` remains the easiest
  operator key, the migration is incomplete.
- **Identity explosion.** If every trace invents its own ad hoc packet,
  downstream consumers will not trust the new surface.
- **Scenario ambiguity.** Base-vs-overlaid valuations must not share the
  same identity unintentionally.
- **Compatibility confusion.** Replay aliases are useful, but they must
  not become selector authority again.
