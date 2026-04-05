# Benchmark Artifacts

This folder contains checked benchmark baselines that are intentionally part of
the repo contract.

What belongs here:

- curated JSON and Markdown benchmark reports that are referenced by docs
- benchmark baselines that verification tests read directly from the repo
- artifacts whose workflow coverage, tolerance contract, or report shape should
  be reviewable in git

What does not belong here:

- ad hoc local timing dumps
- machine-specific scratch output
- task-run byproducts that are not part of a checked benchmark contract

Portability rules:

- checked JSON artifacts must stay portable and must not embed machine-local
  absolute paths such as `json_path` or `text_path`
- save helpers may return filesystem paths to callers, but the persisted report
  payload itself should remain repo-portable

When updating these files:

- regenerate or edit them only when the benchmark contract, workflow coverage,
  or intentionally checked baseline changes
- keep the paired `.json` and `.md` files aligned
- mention the update in the relevant user, quant, or developer docs when the
  benchmark meaning changes
