# Trellis — Multi-Agent Collaboration Guide

## Repository Overview

Trellis is an AI-augmented quantitative finance library for derivative pricing. It has a self-maintaining knowledge system that learns from build failures and prevents repeated mistakes.

**Language:** Python 3.10+
**Dependencies:** numpy, autograd, scipy (core); openai/anthropic (agent); pytest (test)
**Python:** `/Users/steveyang/miniforge3/bin/python3`
**Tests:** `pytest tests/ -x -q -m "not integration"` (~980 tests)

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

## Key Files to Read First

1. `CLAUDE.md` — Full architecture guide
2. `LIMITATIONS.md` — Known issues and resolved items
3. `TASKS.yaml` — Priceable task inventory with status
4. `FRAMEWORK_TASKS.yaml` — Framework/meta task inventory
4. `trellis/agent/knowledge/canonical/features.yaml` — Feature taxonomy
5. `trellis/agent/knowledge/canonical/decompositions.yaml` — Product → feature mappings
6. `trellis/agent/knowledge/import_registry.py` — All valid imports
