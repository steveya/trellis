# QUA-908 Audit: route_scorer.py route-id-keyed scoring bonuses

Ticket: QUA-908 (Contract IR P1.2 — audit `route_scorer.py` for bonuses keyed
on specific route id strings that would silently break under QUA-887 / QUA-903
Phase 1 route rewrites and collapses).

Scope audited:
- `trellis/agent/route_scorer.py` (full file, 488 lines)
- Cross-referenced against 30 route ids + aliases in
  `trellis/agent/knowledge/canonical/routes.yaml`
- Cross-referenced against the Phase 1 deletion / rename list in
  `doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`.

## Summary

- Total route.id string literals found in `route_scorer.py`: **0**
- Conditionals keyed on `route.id` / `spec.id`: **0 that branch on a
  specific id** (the only read of `spec.id` is at line 230/238, used as a
  provenance field on `RouteScore` — not for scoring).
- Phase 1 breakage risk from `route_scorer.py` itself: **none**.

Classification counts:
- Safe / unaffected: 0 (no id literals to classify)
- Migrate to family/trait key: 0
- Requires follow-on ticket: 0
- Unused / stale: 0

## Inventory

### Safe / unaffected (0)
None.

### Migrate to family/trait key (0)
None.

### Requires follow-on ticket (0)
None.

### Unused / stale (0)
None.

## How the file scores routes (for reviewer context)

`route_scorer.py` is deliberately family/trait-keyed. Two scoring paths:

1. **Linear / learned path** (`extract_scoring_features`, lines 123-182).
   All feature keys are derived from family- or trait-level fields on
   `RouteSpec` and `ProductIR`, not from `spec.id`. The id-shaped feature
   emissions are:
   - `f"engine_family:{spec.engine_family}"` (line 176)
   - `f"status:{spec.status}"` (line 177)
   - `f"exercise:{ir.exercise_style}"` (line 167)
   - `f"state:{ir.state_dependence}"` (line 168)
   - `f"model:{ir.model_family}"` (line 169)
   - `f"payoff:{ir.payoff_family}"` (line 170)
   - `f"binding_role:{role}"` (line 148)
   - `f"model_support_role:{model_family}:{role}"` (line 155)
   - `f"capability:{predicate}"` / `f"capability_failure:{failure}"`
     (lines 172, 174)
   - `f"blocker:{blocker}"` (line 180)

   None of these are keyed on `spec.id`. The feature names embed families,
   payoff traits, exercise styles, blockers, and binding roles — all of
   which survive Phase 1 unchanged.

2. **Heuristic fallback path** (`_route_score` in
   `trellis/agent/codegen_guardrails.py`, lines 1753-1869, delegated from
   `RouteScorer.score_route`). Bonuses are applied from
   `spec.score_hints` (YAML) and from the spec's family/trait columns
   (`spec.match_instruments`, `spec.match_payoff_family`,
   `spec.match_exercise`, `spec.engine_family`, `spec.route_family`).
   The hint keys consumed are: `exercise_match_bonus`,
   `exercise_match_styles`, `vanilla_exercise_bonus`,
   `vanilla_exercise_payoff`, `vanilla_exercise_styles`,
   `schedule_dependence_bonus`, `payoff_family_bonus`,
   `non_european_penalty`, `bonus_when_market_data`,
   `penalize_when_market_data`. All are keyed by trait / market-data
   names, not route ids. Source: `codegen_guardrails.py:1815-1863`.

3. **LLM rerank path** (`_llm_rerank`, lines 274-333). Embeds
   `a.route_id` / `b.route_id` as *free-text prompt content only* — not as
   a lookup key. Even when Phase 1 renames a route, the LLM simply sees
   the new name. No behavior is keyed off specific id strings.

4. **Training row extraction** (`_extract_from_run`, lines 395-444).
   Uses `r.id == route_name or route_name in r.aliases` to match
   historical task-run entries against the live registry (line 416). This
   is a registry lookup, not a hardcoded id branch. If Phase 1 renames a
   route, historical rows referencing the old name will simply fail to
   match and be dropped — a minor, already-expected staleness issue, not
   a breakage.

## Verification procedure

1. Collected every string literal in `route_scorer.py` via `ast.walk` on
   the parsed module. Yielded ~85 literals.
2. Loaded the full set of 30 route ids + aliases from
   `routes.yaml`.
3. Computed the intersection: **0 literals match any route id or alias**.
4. Substring scan (to catch `f"prefix_{id}"` style compositions): **0
   literals contain any known route id as a substring**.
5. `grep` scan for tokens typical of Phase 1-affected ids
   (`analytical`, `monte_carlo`, `rate_tree`, `black76`, `quanto`,
   `digital`, `barrier`, `lookback`, `cliquet`, `chooser`, `compound`,
   `variance_swap`, `default_swap`, `nth_to_default`, `zcb_option`,
   `correlated_gbm`): **no matches in `route_scorer.py`**.

## Proposed follow-on tickets

**None.** `route_scorer.py` is cleanly family/trait-keyed and safe for
Phase 1 as currently scoped. The Phase 1 plan's "Failure modes to watch"
entry about route_scorer bonuses tied to route id
(`doc/plan/draft__contract-ir-compiler-retiring-route-registry.md:164-166`)
can be marked audited-clean; the scorer was already refactored for this
(docstring at `codegen_guardrails.py:1761-1767` explicitly states
"Route-specific bonuses and penalties are declared in `score_hints`
within routes.yaml rather than hard-coded per route name").

## Secondary observations (out of QUA-908 scope, but noted)

These are surfaces QUA-908 did not target but which Phase 1 authors may
want to spot-check separately — they are *not* in `route_scorer.py`:

- `routes.yaml` itself declares per-route `score_hints`. When Phase 1
  deletes a route (e.g. `equity_digital_analytical`), the hint block
  disappears with it, which is fine. When Phase 1 collapses two routes
  into one (e.g. the `credit_default_swap_analytical` +
  `credit_default_swap_monte_carlo` → `credit_default_swap` collapse in
  P1.7), the merged route needs `score_hints` that correctly cover both
  old payoff-family / exercise coverage cases. This is a Phase 1 authoring
  concern, not a `route_scorer.py` bug.
- `codegen_guardrails._route_score` uses `spec.match_instruments` as a
  direct-instrument bonus (`+1.25`, line 1804-1805). Once pattern-keyed
  dispatch replaces instrument-name matching in Phase 1, this branch will
  read whatever `match_instruments` becomes post-collapse. If Phase 1
  drops `match_instruments` entirely from `RouteSpec`, that conditional
  becomes dead code and the `+1.25` bonus is silently removed. Whether
  that matters depends on whether the learned linear model has
  compensating weight for family/trait matches, and on whether the
  heuristic fallback path is still live by the time instrument-name
  matching is retired. This is a Phase 1 design consideration worth
  flagging but not a QUA-908 finding.

## Disposition

**Phase 1 may proceed without additional blockers from
`route_scorer.py`.** No follow-on tickets are needed. The scorer is
already family/trait-keyed by design (per the module docstring and the
`_route_score` docstring), and this audit confirms that invariant has
been maintained across the current codebase.

Reviewer: If you want to strengthen the invariant, add a one-line
grep-based regression test that asserts no route id from `routes.yaml`
appears as a literal in `route_scorer.py` or `codegen_guardrails.py`.
That could be a small P3 follow-on but is not required by Phase 1.
