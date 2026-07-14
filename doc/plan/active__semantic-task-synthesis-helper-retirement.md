# Semantic Task Synthesis: Helper Authority Retirement

## Purpose

This plan mirrors the active `QUA-1166` Linear program. The objective is to
keep previously green pricing tasks green while removing product-, method-, and
task-shaped helpers as construction authority. Fresh agents should assemble
pricing functions from semantic contracts, exact market bindings, reusable
numerical primitives, and bounded validation evidence.

## Guardrails

- Do not replace one product helper with another renamed helper.
- Do not hand-author cookbook promotions for task failures.
- Prefer API, signature, navigation, and documentation repairs when agents
  repeatedly misassemble existing primitives.
- Preserve deterministic, fail-closed production behavior.
- Require method-specific artifact coherence; a numerically plausible fallback
  is not a green result when it uses the wrong method or market object.
- Merge small milestones so the local worktree never becomes the durable
  program ledger.

## Linear Ticket Mirror

Status mirror last synced: `2026-07-14`

| Ticket | Outcome | Status |
| --- | --- | --- |
| `QUA-1166` | Semantic task synthesis: retire helper authority | In Progress |
| `QUA-1167` | American LSM: primitive-composed task lane | Done |
| `QUA-1168` | Equity tree: primitive-composed early exercise lane | Done |
| `QUA-1169` | Vanilla Monte Carlo: primitive-composed terminal claim lane | Done |
| `QUA-1170` | Vanilla PDE: primitive-composed theta-method lane | Done |
| `QUA-1171` | FX barrier: primitive-composed analytical and MC lanes | Done |
| `QUA-1172` | FX vanilla: primitive-composed analytical and MC lanes | Done |
| `QUA-1174` | Hybrid market binding: named underlier and volatility surfaces | Done |
| `QUA-1173` | Quanto option: primitive-composed analytical and MC lanes | Done |
| `QUA-1175` | Agent orientation: runtime quant and model-validator navigation contracts | Done |
| `QUA-1176` | Observation returns: reusable bounded accumulation primitives | Done |
| `QUA-1177` | Cliquet pricing: primitive-composed analytical and Monte Carlo lanes | Done |
| `QUA-1178` | Semantic route audit: machine-readable helper authority inventory | Done |
| `QUA-1179` | Monte Carlo: reusable scheduled observation aggregation | Done |
| `QUA-1180` | Analytical support: weighted lognormal-sum moments | In Progress |
| `QUA-1181` | Arithmetic Asian pricing: retire helper authority | Todo (blocked by `QUA-1179`, `QUA-1180`) |

## Current Sequence

1. Implement QUA-1179 product-neutral scheduled observation aggregation for
   reduced-state Monte Carlo.
2. Implement QUA-1180 product-neutral weighted lognormal-sum moments for
   analytical composition.
3. Complete QUA-1181 by removing arithmetic-Asian helper authority only after
   both reusable primitive tickets land; keep geometric and multi-asset Asian
   contracts outside that admitted cohort.

## Completion Evidence

Each migrated lane must provide:

- fresh generated source with no retired product/helper call
- exact semantic, method, market, and validation coherence evidence
- targeted tests plus `make gate-pr`
- a strict offline replay, and a live fresh-generation replay when agent
  synthesis behavior is the claim being proved
- official quant, developer, and user documentation updates
- a merged milestone PR and synchronized Linear/plan status
