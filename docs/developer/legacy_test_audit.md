# Strict Legacy-Test Audit

This note records the strict legacy-only test set identified in the April 2026
audit. The boundary is intentional:

- include tests whose primary target is a deprecated compiler/shim or an
  explicit compatibility-only execution path
- exclude tests that merely mention legacy mirrors while still validating the
  current semantic/compiler spine

The tagged tests use the `legacy_compat` pytest marker so they can be reviewed,
ported, or filtered without deleting them immediately.

## Legacy-Only Set

- `tests/test_agent/test_family_contracts.py::test_quanto_contract_compiles_to_expected_blueprint`
- `tests/test_agent/test_quant.py::TestPricingPlan::test_family_blueprint_quant_plan_preserves_quanto_routes`
- `tests/test_agent/test_validation_bundles.py::test_select_validation_bundle_for_quanto_family_includes_family_checks`
- `tests/test_agent/test_primitive_planning.py::test_builds_quanto_analytical_plan_with_shared_resolution_and_black76`
- `tests/test_agent/test_executor.py::test_record_lesson_maps_why_to_legacy_explanation`
- `tests/test_agent/test_critic.py::TestRunCriticTests::test_legacy_test_code_still_supported`
- `tests/test_agent/test_critic.py::TestRunCriticTests::test_broken_legacy_test_code_skipped`
- `tests/test_agent/test_critic.py::test_critique_filters_legacy_test_code_payload_by_default`
- `tests/test_agent/test_critic.py::test_critique_can_opt_in_legacy_test_code_payload`
- `tests/test_agent/test_critic.py::test_run_critic_tests_respects_allowed_check_ids`
- `tests/test_models/test_trees/test_lattice_performance_contract.py::test_legacy_tree_entry_points_emit_deprecation_warnings`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_exercise_policy_matches_legacy_kwargs`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_lattice_backward_induction_accepts_legacy_terminal_value_and_exercise_value_fn`
- `tests/test_models/test_trees/test_lattice.py::TestRateLattice::test_lattice_backward_induction_accepts_legacy_callable_signatures`
- `tests/test_models/test_monte_carlo/test_early_exercise.py::test_longstaff_schwartz_legacy_function_matches_policy_result`

## First Rewrite-Or-Drop Candidates

These tests sit directly on the deprecated family-contract / family-blueprint
bridge and should be the first ones ported or removed once the semantic path is
the only supported route:

- `tests/test_agent/test_family_contracts.py::test_quanto_contract_compiles_to_expected_blueprint`
- `tests/test_agent/test_quant.py::TestPricingPlan::test_family_blueprint_quant_plan_preserves_quanto_routes`
- `tests/test_agent/test_validation_bundles.py::test_select_validation_bundle_for_quanto_family_includes_family_checks`
- `tests/test_agent/test_primitive_planning.py::test_builds_quanto_analytical_plan_with_shared_resolution_and_black76`

Semantic-path successor coverage already exists in:

- `tests/test_agent/test_platform_requests.py`
- `tests/test_agent/test_dsl_integration.py`

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

Keep math/verification tests that happen to call `build_spot_lattice(...)` or
`build_rate_lattice(...)` unless they are only asserting deprecation or old
calling conventions.
