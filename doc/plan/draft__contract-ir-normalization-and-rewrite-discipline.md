# Contract IR Normalization And Rewrite Discipline

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-904 — Phase 2 umbrella for payoff-expression Contract IR
- QUA-905 — Phase 3 structural solver compiler
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`

## Purpose

Turn "rewrite the payoff tree into canonical form" into a reviewable
design contract rather than an implementation afterthought.

The main inspiration is SymPy's explicit strategy vocabulary:

- local rules
- traversal order
- first-success selection
- fixed-point iteration

Trellis does not need the full generality of a symbolic algebra system.
It does need the same discipline.

## Why This Is Its Own Note

The Phase 2 AST document defines *what* canonical form should mean.

This note defines *how* Trellis should organize the normalization
machinery so that:

- canonicalization is deterministic
- rule interactions are reviewable
- property tests can target the right invariants

## Design Objectives

The rewrite layer should be:

- deterministic
- semantics-preserving
- explicitly stratified
- idempotent at fixed point
- narrow in scope to the admitted contract algebra

## Rule Taxonomy

Trellis should classify rewrite rules into four buckets.

### 1. Local canonicalization rules

These act on one node and its immediate children.

Examples:

- flatten nested `Add`
- sort commutative arguments
- remove additive zero or multiplicative one

These rules should not need global reasoning.

### 2. Local guarded rewrites

These act on one node, but only when a side condition is locally
provable from node form.

Examples:

- distribute positive scalar across `Max`
- fuse constant factors

If the side condition is not syntactically provable, the rule should not
fire.

### 3. Traversal strategies

These determine where a local rule is applied:

- bottom-up
- top-down
- first-success
- exhaustive fixed-point

The traversal strategy is part of the algorithm contract. It should not
be implicit in hand-ordered recursive code.

### 4. Normal-form postconditions

These are not rules themselves. They are the invariants the final
normalizer promises:

- deterministic operand ordering
- no reducible identities remain
- canonical ramp orientation
- fixed-point under reapplication

## Candidate Strategy Vocabulary

The implementation does not need to copy SymPy's exact names, but it
should be conceptually close to:

```text
RewriteRule = Expr -> Expr

bottom_up(rule)
top_down(rule)
do_one(rule_1, ..., rule_n)
chain(rule_1, ..., rule_n)
exhaust(rule)
guard(predicate, rule)
```

The important architectural point is that rewrite *strategy* is a
first-class reviewed surface, not just buried in recursive helper code.

## Recommended Normalization Shape

For the current payoff-expression AST, the safest default stack is:

1. bottom-up local simplification
2. local guarded distribution or orientation rewrites
3. canonical ordering / operand sorting
4. exhaustive fixed-point until no change

This sequence should be explicit in the implementation and in tests.

## Termination Discipline

Every admitted rewrite family should have an informal descent argument.

Examples:

- flattening reduces tree depth
- identity elimination reduces constructor count
- constant fusion reduces the number of constant leaves
- canonical sorting leaves a node unchanged after one pass

If a rule family does not have a plausible descent or fixed-point
argument, it should be treated as suspicious until tested heavily.

## Testing Contract

The minimal rewrite test suite should prove more than handpicked
examples.

### Required invariants

1. **Semantic preservation.**
   Each rule family preserves denotation on the admitted finite test
   domain.
2. **Idempotence.**
   `normalize(normalize(e)) = normalize(e)`.
3. **Order independence on canonical commutative inputs.**
   Different argument orderings normalize to the same form.
4. **No immediate redexes remain.**
   The normal form does not still match the same reducible patterns.

### Desirable invariants

1. **Constructor budget monotonicity.**
   The normalizer should usually not blow up tree size on already
   near-canonical inputs.
2. **Orientation stability.**
   Call and put templates remain distinguishable under normalization.

## Non-Goals

- Do not build a general-purpose theorem prover.
- Do not add symbolic inequality solving just to trigger more rewrites.
- Do not optimize for maximum simplification at the expense of
  predictability.
- Do not let rewrites leak pricing-method assumptions into the contract
  layer.

## Relationship To Phase 3

Phase 3 structural declarations depend on canonical forms being stable.

If the normalizer is unstable, selector ambiguity and family drift will
show up downstream as apparent compiler problems that are really Phase 2
normalization problems.

So this note is technically Phase-2-adjacent but operationally a Phase-3
dependency.

## Ordered Follow-On Queue

### N1 — Bounded rewrite engine for `PayoffExpr`

Objective:

Implement the minimal rewrite strategy layer needed for Contract IR
canonicalization.

Acceptance:

- rewrite strategy is explicit in code
- the bounded Phase 2 rule set runs to fixed point deterministically

### N2 — Property-based normalizer tests

Objective:

Add randomized tests for idempotence, ordering invariance, and canonical
forms.

Acceptance:

- randomized AST generation exists for the admitted Phase 2 node set
- normalization fixed-point and canonical-order properties are checked

### N3 — Selector-facing normal form snapshot tests

Objective:

Freeze representative normalized forms that Phase 3 declarations depend
on.

Acceptance:

- declaration templates and normalizer outputs can be reviewed together
- canonical-family drift is detectable in CI

## Risks To Avoid

- **Ad hoc recursion.** Hidden traversal order makes review and
  debugging much harder.
- **Rule oscillation.** Two individually sensible rules can loop if the
  strategy contract is vague.
- **False cleverness.** A rewrite that is algebraically attractive but
  semantically under-justified is a bug source.
- **Compiler blame shifting.** If canonicalization is unstable, Phase 3
  ambiguity will be misdiagnosed as a declaration problem.
