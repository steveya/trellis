# Trellis — Claude Code Project Guide

## What is Trellis

AI-augmented quantitative finance library for pricing fixed-income securities and derivatives. ~9,000 lines of Python. The library has a multi-agent pipeline that builds pricing models via LLM code generation, validates them, and learns from failures.

## Quick Start

```bash
cd /Users/steveyang/Projects/steveya/trellis
/Users/steveyang/miniforge3/bin/python3 -m pytest tests/ -x -q -m "not integration"
```

Python environment: `/Users/steveyang/miniforge3/bin/python3` (3.10, has all deps including autograd, numpy, scipy).

Always use the conda Python above for tests, replays, and build tasks.
Do not depend on the system `python3` from `PATH`; if a shell is launched,
activate the conda environment or call the miniforge interpreter directly.

When editing files, resolve the target path from the repo root and confirm it
exists with `git status`, `rg --files`, or `realpath` before applying a patch;
avoid hand-typing long absolute paths when a discovered path is available.

API keys are in `.env` (not committed). LLM_PROVIDER can be `openai` (o3-mini) or `anthropic` (claude-sonnet-4-6).

## Architecture

### Core Layer (`trellis/core/`)
- `payoff.py` — Payoff protocol (requirements + evaluate → float)
- `market_state.py` — Immutable bag of market data (discount, vol_surface, credit_curve, etc.)
- `capabilities.py` — Market data + method capability registry
- `date_utils.py` — Schedule generation, year fractions, day count conventions
- `types.py` — DayCountConvention, Frequency, etc.

### Instruments (`trellis/instruments/`)
- `bond.py`, `swap.py`, `cap.py` — Hand-written reference implementations
- `callable_bond.py`, `barrier_option.py` — Reference implementations using trees/MC
- `_agent/` — Agent-generated payoffs (can be deleted for clean-slate runs)

### Models (`trellis/models/`)
- `black.py` — Black76 analytical
- `analytical/` — Jamshidian, barrier closed-forms
- `trees/` — Binomial, trinomial, calibrated lattice, backward induction
- `monte_carlo/` — Engine, LSM, discretization schemes, variance reduction
- `pde/` — Theta method (CN/implicit), PSOR, operators (BS, CEV, HW)
- `transforms/` — FFT pricer, COS method
- `processes/` — GBM, Hull-White, Heston, Vasicek, CIR, SABR, Merton jump-diffusion, local vol
- `copulas/` — Gaussian, Student-t, factor copulas
- `calibration/` — Implied vol, SABR fit, local vol (Dupire)
- `cashflow_engine/` — Prepayment (PSA), amortization, waterfall
- `vol_surface.py` — FlatVol, VolSurface protocol

### Agent System (`trellis/agent/`)
Multi-agent pipeline: quant → planner → builder → critic → arbiter → validator.

- `executor.py` — Main pipeline (`build_payoff()`)
- `quant.py` — Method selection (STATIC_PLANS + LLM fallback)
- `planner.py` — Spec schema design (STATIC_SPECS + LLM)
- `builder.py` — Code generation + dynamic import
- `prompts.py` — LLM prompt templates
- `cookbooks.py` — 5 evaluate() templates (analytical, rate_tree, monte_carlo, copula, waterfall)
- `data_contract.py` — Vol unit conversion rules
- `experience.py` — Legacy experience system (delegates to knowledge/)
- `test_resolution.py` — Failure diagnosis + lesson recording

### Knowledge System (`trellis/agent/knowledge/`)
**This is the self-maintaining knowledge infrastructure built in March 2026.**

- `__init__.py` — Public API: `retrieve_for_task()`, `build_with_knowledge()`
- `schema.py` — All dataclasses (Feature, Lesson, ProductDecomposition, etc.)
- `store.py` — KnowledgeStore singleton, 3-tier retrieval (hot/warm/cold)
- `retrieval.py` — `format_knowledge_for_prompt()` — formats knowledge as markdown for LLM
- `import_registry.py` — **Authoritative import paths** — injected into every prompt to prevent hallucination
- `decompose.py` — Product → feature decomposition (static + LLM fallback)
- `autonomous.py` — `build_with_knowledge()` — self-maintaining build wrapper
- `gap_check.py` — Pre-flight knowledge audit (confidence score 0-1)
- `reflect.py` — Post-build reflection (attribute success, capture lessons, enrich cookbooks)
- `promotion.py` — Gated learning: capture → validate → promote → archive → distill
- `signatures.py` — YAML-driven failure pattern matching
- `canonical/` — Feature taxonomy (34 features), decompositions (19+ products), principles (8), cookbooks (7 methods), data contracts, method requirements, failure signatures, API map
- `lessons/` — 46 lessons (21 promoted, 17 archived, 8 candidate), individual YAML per lesson
- `benchmarks/` — Placeholder for extracted test reference data
- `traces/` — Cold storage for run traces (374 files, .gitignored)

### Key Design Decisions

1. **Feature-based retrieval, not instrument-name lookup.** Products decompose into features (atoms). `callable_bond` = `[callable, fixed_coupons, mean_reversion]`. Features expand transitively via `implies` chains. Retrieval unions lessons matching any feature.

2. **Import registry eliminates hallucination.** `import_registry.py` introspects the package tree and injects all valid import paths at the top of every code generation prompt. This was the single biggest fix — took success rate from 58% to 90%+.

3. **API map first, then registry.** `canonical/api_map.yaml` is the small family-level navigation layer. Use it to orient to the right module family before falling back to the full import registry and package walk.

4. **Autonomous learning loop.** Every build: gap_check → build → reflect. Lessons auto-captured with feature tags from decomposition, auto-validated/promoted based on confidence, periodically distilled.

5. **Backward compatibility.** `experience.py` and `cookbooks.py` are shims that delegate to KnowledgeStore. All old function signatures work.

## Task System

`TASKS.yaml` now holds the 131 priceable tasks, while `FRAMEWORK_TASKS.yaml` holds 23 framework/meta tasks that should not run through the pricing-task runner. Historically, the combined inventory was 146 tasks before the later stress-task expansion.

Run tasks: `python scripts/run_tasks.py T13 T24`
Remediate failures: `python scripts/remediate.py`
Re-run specific: `python scripts/rerun_ids.py T54 T62 T71`

## Epic Implementation Workflow

When implementing an epic with multiple sub-issues, follow this workflow:

### 1. Dependency Analysis and Wave Planning

Group issues into waves by dependency order. Issues with no blockers go in
Wave 1. Issues blocked by Wave 1 go in Wave 2, etc. Independent issues
within a wave can run in parallel.

### 2. Per-Issue Implementation Cycle

For each issue, follow this cycle:

1. **Read** the relevant code before editing — understand the current state.
2. **Implement** the changes described in the issue scope.
3. **Add tests** — every behavioral change needs a test. Use `tmp_path` or
   monkeypatch to avoid mutating real data files.
4. **Run tests** — run the new tests, then the full agent test suite:
   ```bash
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/ -x -q \
     --ignore=tests/test_agent/test_swaption_demo.py \
     --ignore=tests/test_agent/test_build_loop.py \
     --ignore=tests/test_agent/test_callable_bond.py \
     -m "not integration"
   ```
5. **Update the Linear ticket** — add a Result section describing what was
   done, then mark the issue Done.
6. **Move to the next issue** in dependency order.

### 3. Final Cleanup Pass

After all issues in the epic are complete:

1. Run the **full** test suite: `pytest tests/ -q -m "not integration"`
2. **Refactor** — check for unused imports, dead code, duplicated logic
   introduced across multiple issues.
3. **Optimize** — consolidate any patterns that emerged from multiple issues
   touching the same module.
4. **Clean up tests** — verify no duplicate test names, all assertions are
   meaningful, no leftover debug prints.
5. Run the full test suite again to confirm the cleanup introduced no
   regressions.

### 4. Epic Closeout

When an epic or umbrella ticket is completed, treat the closeout as a
maintenance checkpoint:

1. Run a cleanup and refactoring pass around the affected code paths.
2. Perform a substantial documentation maintenance update in the official
   docs that the work touches:
   - `docs/quant/` for mathematical and pricing-stack behavior
   - `docs/developer/` for runtime, agents, observability, and operational flow
   - `docs/user_guide/` for user-facing usage and workflow changes
3. Record in the issue closeout note if the epic is intentionally doc-free.

Epic completion should leave the codebase cleaner and the docs current, not
just the Linear ticket marked done.

## Limitations

See `LIMITATIONS.md`. Resolved: L1 (trinomial), L4 (old PDE), L15 (silent validation), L16 (silent recording). Open: L2, L3, L5-L14, L17-L22.

## Testing

```bash
# All non-integration tests (~1620 pass)
/Users/steveyang/miniforge3/bin/python3 -m pytest tests/ -x -q -m "not integration"

# Agent tests only (~610 pass)
/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/ -x -q --ignore=tests/test_agent/test_swaption_demo.py --ignore=tests/test_agent/test_build_loop.py --ignore=tests/test_agent/test_callable_bond.py -m "not integration"

# Knowledge system tests (37 pass)
/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_knowledge_store.py -x -q

# Architecture consolidation tests (QUA-398 epic, ~124 tests)
/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_route_registry.py tests/test_agent/test_recalibration.py tests/test_agent/test_semantic_contracts.py tests/test_agent/test_lesson_supersedes.py tests/test_agent/test_trace_compaction.py tests/test_agent/test_principle_extraction.py -x -q
```

## Style

- Frozen dataclasses for all value types
- Protocol-based design (Payoff, DiscountCurve, VolSurface)
- No registries — capability-based discovery
- Immutable MarketState
- numpy via autograd wrapper (`from trellis.core.differentiable import get_numpy`)
