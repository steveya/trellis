# Numba Cache Note

This note documents a validation-specific issue that can appear in restricted
execution environments, including sandboxed Codex runs.

## Symptom

FinancePy-backed tests fail during import with an error like:

```text
RuntimeError: cannot cache function 'date_index': no locator available
```

This happens inside `financepy.utils.date` when Numba tries to initialize
`@njit(cache=True)` functions.

## Root Cause

The failure is not a Trellis pricing bug and it is not a FinancePy logic bug.
It is a cache-path permissions problem.

Numba tries its cache locators in this order:

1. in-tree `__pycache__` near the installed package
2. user-wide cache under `~/Library/Caches/numba` on macOS

In restricted runners, neither location may be writable. When all locators
fail, FinancePy import fails.

## Workaround for Sandboxed Runs

Point Numba's cache to a writable directory, for example `/tmp`:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache /Users/steveyang/miniforge3/bin/python3 -m pytest -x -q tests/test_crossval/test_xv_bonds.py
```

The same pattern works for FinancePy-backed task tests:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache /Users/steveyang/miniforge3/bin/python3 -m pytest -x -q tests/test_tasks/test_t01_zcb_option.py
```

## Scope

- Outside the sandbox, FinancePy cross-checks may work without any override.
- Inside the sandbox, set `NUMBA_CACHE_DIR` if a test imports FinancePy.
- This is primarily a validation-environment note, not a library limitation.
