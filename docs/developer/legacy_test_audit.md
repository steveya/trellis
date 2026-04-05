# Strict Legacy-Test Audit

This note records the strict legacy-only test set identified in the April 2026
audit. The boundary is intentional:

- include tests whose primary target is a deprecated compiler/shim or an
  explicit compatibility-only execution path
- exclude tests that merely mention legacy mirrors while still validating the
  current semantic/compiler spine

The tagged tests use the `legacy_compat` pytest marker so they can be reviewed,
ported, or filtered without deleting them immediately.

That marker is now registered at the root test layer alongside the broader
non-integration strata used during review:

- `crossval`
- `verification`
- `global_workflow`
- `legacy_compat`

Useful selector examples:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "legacy_compat and not integration"
/Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "global_workflow and not integration"
```

## Legacy-Only Set

- `tests/test_models/test_trees/test_lattice_performance_contract.py::test_legacy_tree_entry_points_emit_deprecation_warnings`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_exercise_policy_matches_legacy_kwargs`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_lattice_backward_induction_accepts_legacy_terminal_value_and_exercise_value_fn`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_lattice_backward_induction_accepts_legacy_callable_signatures`
- `tests/test_models/test_monte_carlo/test_early_exercise.py::test_longstaff_schwartz_legacy_function_matches_policy_result`

The tree-wrapper boundary is now defended by
`tests/test_models/test_trees/test_lattice_wrapper_audit.py`. Ordinary lattice
and verification suites should build through the shared unified helpers in
`tests/lattice_builders.py`, not through deprecated wrapper entry points.

## First Rewrite-Or-Drop Candidates

The semantic-path successor coverage for the removed family-blueprint bridge now
lives in:

- `tests/test_agent/test_family_contracts.py`
- `tests/test_agent/test_primitive_planning.py`
- `tests/test_agent/test_platform_requests.py`
- `tests/test_agent/test_validation_bundles.py`

## Keep List

Keep these as current-path coverage even when they mention legacy mirrors or
compatibility wrappers:

- `tests/test_agent/test_family_lowering_ir.py`
- `tests/test_agent/test_semantic_contracts.py`
- `tests/test_agent/test_market_binding.py`
- `tests/test_agent/test_valuation_context.py`
- `tests/test_agent/test_checkpoints.py`
- `tests/test_agent/test_platform_traces.py`

Keep these as parity-oracle coverage, not legacy-track cleanup targets:

- `tests/test_models/test_trees/test_lattice_algebra.py`
- `tests/test_models/test_monte_carlo/test_early_exercise.py` result tests for
  `primal_dual_mc`, `tsitsiklis_van_roy`, and `stochastic_mesh`

Do not add new ordinary math or verification coverage directly against
`build_spot_lattice(...)` or `build_rate_lattice(...)`. Those suites should use
the unified helpers in `tests/lattice_builders.py`, and direct wrapper calls
should remain limited to explicit deprecation checks, compatibility tests, or
parity oracles.
