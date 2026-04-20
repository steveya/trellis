# Insurance Contract Overlay Foundation

## Status

`INS.1` / `QUA-936` is now active.

The current bounded financial-control slice remains overlay-free, and
this note records the first executable boundary between that core and
later insurance-style semantics.

## Linked Context

- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__continuous-singular-control-lowering-plan.md`

## Purpose

Keep insurance-specific overlays explicit and separate enough that
financial-control semantics remain reusable outside insurance.

This note matters only if the eventual GMxB roadmap intends to include
more than the pure financial-control core, for example:

- mortality state
- lapse state
- rider-fee logic
- policyholder-behavior overlays beyond the contractual withdrawal
  action itself

## Core Boundary

The financial-control core should answer:

- what the contractual financial state is
- what the admissible withdrawal/surrender actions are
- what the payoff and continuation semantics are under the admitted
  financial model

The insurance overlay should answer:

- whether additional policy-status state such as alive/dead/lapsed is
  present
- whether mortality or lapse intensities affect continuation or cashflow
  semantics
- whether fees, benefits, or policyholder behavior introduce additional
  non-financial state transitions

Those are related layers, not the same layer.

### Current Bounded Boundary

| Semantic element | Current owner | Current bounded handling |
| --- | --- | --- |
| `account_value`, `guarantee_base` | financial-control core | admitted continuous/singular control state |
| withdrawal / surrender action magnitude | financial-control core | admitted control action semantics |
| withdrawal dates and withdrawal cashflows | financial-control core | admitted event-program semantics |
| policy status such as alive / dead / lapsed | insurance overlay | deferred; current admission must fail closed |
| mortality / lapse intensities or transition rules | insurance overlay | deferred; current decomposition must fail closed |
| rider-fee accrual or policy-level benefit adjustments | insurance overlay | deferred; current decomposition must fail closed |

The practical rule for the current slice is:

- if a GMWB-style request can be described entirely in terms of
  financial account state, guarantee state, withdrawal control, and the
  resulting cashflows, it belongs to the bounded financial-control core
- if the request adds policy-status state, mortality or lapse behavior,
  or fee/benefit overlays, it belongs to the deferred insurance overlay
  track and must not be silently reinterpreted as plain financial
  control

## Why This Separation Matters

Without a separate overlay note, the first bounded GMWB/GMxB plan may
sprawl in one of two bad directions:

1. the financial-control core is never made reusable because it is
   immediately entangled with biometric overlays
2. mortality/lapse semantics are pushed down into route-local adapters
   because the control lane is too narrow to talk about them honestly

## Minimal Overlay Surface

This note does not lock an ADT, but the later bounded overlay should be
able to express:

```text
InsuranceOverlay =
    { policy_state_schema: PolicyStateSchema
    ; overlay_events: tuple[OverlayEvent, ...]
    ; overlay_parameters: OverlayParameterSet
    ; composition_rule: OverlayCompositionRule
    }
```

with bounded examples such as:

- mortality transition events
- lapse transition events
- rider-fee accrual events
- benefit-adjustment rules that act on top of the financial-control
  state

## Ordered Follow-On Queue

| Queue ID | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- |
| `INS.1` | In Progress (`QUA-936`) | document the overlay boundary against the financial-control core | bounded financial-control lane exists |
| `INS.2` | Backlog | define the minimal policy-state overlay surface | `INS.1` |
| `INS.3` | Backlog | choose one bounded proving family if insurance overlays are actually in scope | `INS.2` |

## Admission Rule

Do not start `INS.*` work unless at least one of the following is true:

- the roadmap explicitly wants mortality-aware or lapse-aware GMxB
- a requested family cannot be represented honestly without policy-state
  overlays

Otherwise, keep the financial-control core separate and complete it
first.

## Risks To Avoid

- **Premature overlay coupling.** The first continuous/singular control
  slice should not become an insurance-platform epic by accident.
- **Semantic drift.** Mortality or lapse effects should be explicit
  overlay semantics, not hidden assumptions in helper selection.
- **Overclaim.** Landing an overlay note or one rider slice does not
  imply broad actuarial-product support.

## Next Steps

1. Finish `INS.1` / `QUA-936` by locking the boundary into fixtures,
   admission blockers, and the closure execution mirror.
2. Promote `INS.2` only if a real requested family needs policy-state
   semantics beyond the bounded financial-control core.
3. Treat any later overlay work as a sibling bounded track above the
   financial-control core rather than reopening the core design.
