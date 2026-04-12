# Trellis Paper Workspace

This workspace is the starting point for a paper series on Trellis.

The repo uses `docs/`, not `doc/`, so the paper workspace lives under
`docs/paper/`.

## Recommended shape

The strongest framing today is a three-part series:

1. request-to-outcome compiler and semantic pipeline
2. mathematical and computational substrate
3. governed knowledge loop and the path to real learning

If later you want only two papers, merge Parts II and III.

## Why three parts is better right now

- Part I is already a strong systems/compiler story.
- Part II is already a strong quantitative-methods story.
- Part III is important, but the repo is more honest as "governed reflection,
  retrieval, promotion, and future closed-loop learning" than as "the system
  already learns robustly in production."

## Workspace layout

- `series-plan.md`: repo-grounded paper strategy and writing order
- `common/preamble.tex`: shared LaTeX packages and macros
- `artifacts/`: figures, tables, exported traces, benchmark snapshots
- `part1-request-to-outcome/main.tex`: Part I draft entrypoint
- `part2-computational-lanes/main.tex`: Part II draft entrypoint
- `part3-learning-loop/main.tex`: Part III draft entrypoint

## Authoring rules

- Keep the core claim consistent across all papers: the LLM is not in the
  deterministic pricing hot path.
- Distinguish clearly between shipped architecture, active transition work,
  and future roadmap.
- Use `LIMITATIONS.md` as a guardrail against overclaiming numerical or
  learning maturity.
- Prefer repo-grounded figures and traces over aspirational diagrams.

## Suggested next move

Write Part I first, because it defines the vocabulary the other papers reuse:
`PlatformRequest`, `SemanticContract`, `ValuationContext`, `ProductIR`,
family IRs, admissibility, validation, and traces.
