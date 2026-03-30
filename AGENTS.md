# Trellis — Multi-Agent Collaboration Guide

## Repository Overview

Trellis is an AI-augmented quantitative finance library for derivative pricing. It has a self-maintaining knowledge system that learns from build failures and prevents repeated mistakes.

**Language:** Python 3.10+
**Dependencies:** numpy, autograd, scipy (core); openai/anthropic (agent); pytest (test)
**Python:** `/Users/steveyang/miniforge3/bin/python3`
**Tests:** `pytest tests/ -x -q -m "not integration"` (~980 tests)

Always run Python through the conda interpreter above. Do not rely on the
system `python3` on `PATH` for tests, replays, or task execution.
When in doubt, invoke `/Users/steveyang/miniforge3/bin/python3` directly or
activate the conda environment first and confirm `which python3` resolves to
the miniforge interpreter.

## Agent Roles

### Role: Library Developer
**Scope:** `trellis/models/`, `trellis/core/`, `trellis/curves/`, `trellis/instruments/`
**Task:** Build, refactor, or extend pricing engines, stochastic processes, numerical methods.
**Rules:**
- Read `CLAUDE.md` for full architecture context
- Check `LIMITATIONS.md` before starting — some features are known-incomplete
- Use `from trellis.core.differentiable import get_numpy` instead of `import numpy` for autograd compatibility
- All value types should be frozen dataclasses
- All market interfaces should be protocols (not base classes)
- Run `pytest tests/ -x -q -m "not integration"` after every change
- Do NOT modify `trellis/agent/knowledge/` files — that's the Knowledge Agent's domain

### Role: Knowledge Agent
**Scope:** `trellis/agent/knowledge/`, `trellis/agent/experience.py`, `trellis/agent/cookbooks.py`
**Task:** Maintain the knowledge system — lessons, features, decompositions, cookbooks, import registry.
**Rules:**
- After Library Developer adds/moves modules, update `import_registry.py` (run `get_import_registry()` to regenerate)
- Treat validated `_fresh` adapter snapshots as the upgrade signal for checked-in adapter code. If a fresh-build adapter differs from the checked-in route, surface it as `stale` in retrieval/prompt text and keep the first pass warning-only until the replacement is validated.
- After new instruments are added, add decompositions to `canonical/decompositions.yaml`
- After new pricing methods are added, add cookbooks to `canonical/cookbooks.yaml`, contracts to `data_contracts.yaml`, requirements to `method_requirements.yaml`
- After new features emerge, add to `canonical/features.yaml` with `implies` chains
- Run `pytest tests/test_agent/test_knowledge_store.py -x -q` after changes
- Run `scripts/remediate.py --analyze-only` to check for knowledge gaps

### Role: Task Runner
**Scope:** `scripts/`, `TASKS.yaml`, `FRAMEWORK_TASKS.yaml`, `task_results_*.json`
**Task:** Execute pricing tasks via `build_with_knowledge()`, analyze failures, trigger remediation.
**Rules:**
- Use `python scripts/run_tasks.py T13 T24` to run task blocks
- Use `python scripts/remediate.py` to analyze failures + fix knowledge + re-run
- Use `python scripts/rerun_ids.py T54 T62` for specific re-runs
- Results saved to `task_results_*.json` (not committed)
- After each batch, check if `scripts/remediate.py --analyze-only` shows fixable patterns

### Role: Test & Validation Agent
**Scope:** `tests/`, `LIMITATIONS.md`
**Task:** Write tests, verify cross-validation, update limitations.
**Rules:**
- Test files follow pattern: `tests/test_{area}/test_{topic}.py`
- Task challenge tests: `tests/test_tasks/test_t{nn}_{name}.py`
- Cross-validation: `tests/test_crossval/test_xv_{topic}.py`
- Verification: `tests/test_verification/test_{topic}.py`
- Use `pytest.importorskip("QuantLib")` for optional external library tests
- Update `LIMITATIONS.md` when resolving or discovering limitations

## Module Ownership

| Path | Owner | Notes |
|------|-------|-------|
| `trellis/core/` | Library Developer | Protocols, types, market state |
| `trellis/models/` | Library Developer | All pricing engines and processes |
| `trellis/curves/` | Library Developer | Yield, forward, credit curves |
| `trellis/instruments/` | Library Developer | Reference implementations |
| `trellis/instruments/_agent/` | Task Runner | Auto-generated, can be deleted |
| `trellis/agent/knowledge/` | Knowledge Agent | Knowledge system |
| `trellis/agent/knowledge/canonical/` | Knowledge Agent | Feature taxonomy, cookbooks, etc. |
| `trellis/agent/knowledge/lessons/` | Knowledge Agent | Auto-maintained lessons |
| `trellis/agent/executor.py` | Library Developer | Build pipeline |
| `trellis/agent/prompts.py` | Knowledge Agent | Prompt templates |
| `scripts/` | Task Runner | Run/remediate scripts |
| `tests/` | Test Agent | All test files |
| `TASKS.yaml` | Task Runner | Priceable task definitions |
| `FRAMEWORK_TASKS.yaml` | Task Runner | Framework/meta task inventory |
| `LIMITATIONS.md` | Test Agent | Known issues |

## Coordination Protocol

### When Library Developer moves or renames modules:
1. Make the code change
2. Run tests to verify
3. **Notify Knowledge Agent** to update:
   - `import_registry.py` (regenerate)
   - `canonical/decompositions.yaml` (update method_modules)
   - `canonical/cookbooks.yaml` (update import paths in templates)
   - `canonical/features.yaml` (if new features added)

### When Knowledge Agent adds new knowledge:
1. Update YAML files
2. Run `pytest tests/test_agent/test_knowledge_store.py`
3. **Notify Task Runner** to re-run previously failed tasks that might benefit

### When Task Runner finds failures:
1. Run `scripts/remediate.py --analyze-only`
2. Categorize: knowledge gap vs implementation gap vs infrastructure
3. **Knowledge gaps** → notify Knowledge Agent
4. **Implementation gaps** → notify Library Developer
5. **Infrastructure** (missing market data, rate limits) → fix in `scripts/run_tasks.py`

### When Test Agent resolves limitations:
1. Fix the code
2. Write/update tests
3. Update `LIMITATIONS.md` (move to Resolved section)
4. **Notify Knowledge Agent** if the fix changes module structure

## Linear Issue Writing Spec

Linear is the shared work ledger for cross-agent work. Issues should be
specific, searchable, dependency-aware, and tied to an observable outcome.
Treat an issue as a short engineering spec, not as a note or a chat summary.

### When to create or update an issue

- Create a Linear issue when work must survive beyond the current chat, spans
  more than one file or agent, introduces a blocker, or needs durable tracking.
- Update an existing issue instead of creating a duplicate when the scope is
  the same.
- Split a ticket when it contains more than one independent reviewable outcome.
  Use an epic for the umbrella objective and child issues for delivery slices.
- Do not mirror every local task row into Linear. Use Linear for coordination,
  blockers, durable planning, or user-visible work that benefits from tracking.

### Title and naming

- Use outcome-first titles of the form `<Area>: <specific result>`.
- Keep titles short, concrete, and searchable.
- Put the domain noun in the title, not just the implementation verb.
- If the work concerns product semantics, contract synthesis, semantic
  validation, compiler planning, route selection, or blocker taxonomy, include
  `Semantic` in the title.
- If the work is reusable math, name the reusable kernel or primitive.
- If the work is route-local, name the route and the concrete artifact.
- Avoid vague titles such as `Cleanup`, `Refactor`, or `Improve pricing`
  unless paired with the exact target.
- Good examples:
  - `Semantic contract: generic derivative IR`
  - `Analytical support: reusable barrier kernel`
  - `Barrier option pricing: raw kernel assembly`
  - `Monte Carlo: pathwise gradients for smooth payoffs`

### Issue body shape

Use a compact spec structure so any agent can recover the plan quickly:

- Objective: what should exist when the issue is done.
- Why now: why this matters now.
- Scope: the exact modules, routes, or docs in scope.
- Non-goals: what is explicitly out of scope.
- Dependencies: hard blockers with issue IDs.
- Acceptance criteria: observable conditions that define success.
- Validation: tests, benchmarks, docs, or review gates.
- Follow-on work: separate issues for future slices, if needed.

If the issue is exploratory or research-oriented, add the decision it informs
and the concrete artifact it should produce.

### Dependency rules

- Use parent/child relationships for program structure and epics.
- Use `blockedBy` and `blocks` only for hard prerequisites.
- Use `relatedTo` for adjacent work that does not prevent completion.
- Keep dependency chains shallow. Do not create a blocker unless the work
  cannot be completed correctly without it.
- If the blocker does not yet exist, create the blocker issue first or state
  the missing prerequisite explicitly.
- Do not block on anticipated future reuse alone; keep that as a scoped note
  unless the reuse is already real.
- For semantic work, block on the smallest upstream semantic contract or
  primitive that is truly required, not on the entire future architecture.

### Priority rules

- Priority 1 / Urgent: active outage, broken build, release blocker, or a hard
  blocker for an approved near-term task.
- Priority 2 / High: user-visible work, foundational platform work, or an item
  that unlocks multiple other issues.
- Priority 3 / Normal: planned implementation slices and most feature work.
- Priority 4 / Low: docs, cleanup, exploratory refactors, and deferred follow-ups.
- Priority 0 / None: parking-lot ideas that are not ready to schedule.
- Default to Priority 3 unless there is a concrete reason to raise it.
- Do not use Urgent for roadmap ambition alone.

### Blocker handling

- If a task is blocked, say so explicitly in the issue and when communicating
  with the user.
- State the blocking issue ID(s), the missing prerequisite, and the next unblock
  step.
- Never present a blocked issue as complete.
- If the user asks to complete work but a Linear blocker remains, surface the
  blocker before claiming success.
- Convert vague blockers into concrete missing primitives, routes, data, or
  validation steps whenever possible.

### Done criteria

- Mark an issue Done only when the implementation slice is landed, the agreed
  tests or validation pass, and the acceptance criteria are satisfied.
- For design or docs work, Done means the artifact exists, is reviewed, and the
  user-visible contract has been updated.
- If an issue changes behavior, APIs, workflows, validation, or knowledge that
  users or developers rely on, update the relevant official documentation as
  part of the closeout:
  - `docs/quant/` for mathematical and pricing-stack behavior
  - `docs/developer/` for runtime, agents, observability, and operational flow
  - `docs/user_guide/` for user-facing usage and workflow changes
  If the issue is intentionally doc-free, record that explicitly in the issue
  closeout note.
- Close the issue with a short note summarizing the result, the validation, and
  any follow-on issue IDs.
- If useful work remains, split it into follow-on issues instead of leaving the
  original issue ambiguous.
- Do not leave an issue in Done if it still has an unresolved hard blocker.

### Epic closeout

- Treat an epic as a maintenance checkpoint, not just a delivery milestone.
- When an epic/umbrella issue is completed, do a cleanup and refactoring pass
  before closing the umbrella work.
- After that pass, do a substantial documentation maintenance update across the
  official docs that the work touches:
  - `docs/quant/` for mathematical and pricing-stack behavior
  - `docs/developer/` for runtime, agents, observability, and operational flow
  - `docs/user_guide/` for user-facing usage and workflow changes
- If an epic is intentionally doc-free, record that explicitly in the issue
  closeout note and explain why no docs changed.
- Do not treat epic closeout as optional polish; it is part of completing the
  umbrella issue.

### Semantic issue conventions

- Use `Semantic` for work on product semantics, contract synthesis, semantic
  validation, compiler planning, route selection, and blocker taxonomy.
- Include `semantic`, `contract`, `IR`, `validation`, `compiler`, `route`,
  `primitive`, and `blocker` keywords in the body when they help discovery.
- Prefer phase-based semantic issues:
  - semantic contract
  - bounded compiler or blueprint
  - thin route adapters
  - docs, knowledge, and roadmap hardening
- When semantic work feeds another strand, keep the semantic epic separate and
  link the downstream implementation issue(s).

### Engineering plan quality

- Every implementation issue should include a concrete plan with small phases.
  Each phase should produce a checked-in artifact or a validation gate.
- Prefer stable scaffolds plus surgical deltas over whole-module rewrites.
- If the plan cannot be explained as a few reviewable phases, the issue is too
  large and should be split.

## Current State (March 2026)

- **131 priceable tasks** in `TASKS.yaml`
- **23 framework/meta tasks** in `FRAMEWORK_TASKS.yaml`
- **~110 attempted**, ~99 succeeded (90%)
- **46 lessons** in knowledge system (21 promoted, 17 archived, 8 candidate)
- **7 cookbooks** (analytical, rate_tree, monte_carlo, copula, waterfall, pde_solver, fft_pricing)
- **34 features** in taxonomy with transitive implies chains
- **Import registry** eliminates most import hallucination
- **4 limitations resolved** (L1, L4, L15, L16), 13 open

## Anti-Hallucination Rules

1. **Never invent import paths.** The import registry (`knowledge/import_registry.py`) has every valid import. If it's not there, it doesn't exist.
2. **Never invent formulas.** If you need a pricing formula, check `trellis/models/` first. If it's not implemented, say so.
3. **Never overpromise in docstrings.** Document what IS implemented, not what COULD be.
4. **Check before claiming.** Before saying "trellis supports X", verify with `grep` or `import`.
5. **Use the API map before the registry.** When the module family is unclear, inspect `canonical/api_map.yaml` or call `inspect_api_map` first, then confirm exact symbols with the import registry, `find_symbol`, or `list_exports`.
6. **Check file targets before patching.** When editing or referencing files, resolve the path from the repo root and confirm it exists with `git status`, `rg --files`, or `realpath` before calling `apply_patch`; do not rely on a hand-typed absolute path.

## Key Files to Read First

1. `CLAUDE.md` — Full architecture guide
2. `LIMITATIONS.md` — Known issues and resolved items
3. `TASKS.yaml` — Priceable task inventory with status
4. `FRAMEWORK_TASKS.yaml` — Framework/meta task inventory
5. `trellis/agent/knowledge/canonical/features.yaml` — Feature taxonomy
6. `trellis/agent/knowledge/canonical/decompositions.yaml` — Product → feature mappings
7. `trellis/agent/knowledge/canonical/api_map.yaml` — Small family-level API navigation map
8. `trellis/agent/knowledge/import_registry.py` — All valid imports
