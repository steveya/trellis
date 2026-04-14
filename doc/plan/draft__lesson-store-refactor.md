# Lesson Store Refactor: Generated Index from Canonical Entry Files

## Problem

The lesson store maintains two sources of truth — `index.yaml` and per-lesson
`entries/{id}.yaml` files — synchronized by writing both on every mutation.
This creates three classes of bugs:

1. **Consistency drift.** If an entry-file write succeeds but the index write
   fails (or vice versa), the two disagree silently. No reconciliation
   mechanism exists.
2. **Compounding I/O on writes.** A single `record_semantic_extension_trace()`
   call can trigger `capture_lesson()` → `validate_lesson()` →
   `promote_lesson()`, each doing a full read-modify-write of the 1,894-line
   index file. That is six YAML parse/serialize cycles for one logical
   mutation.
3. **Stale agent state.** Dedup and ID generation read the index, not entry
   files. The retrieval cache is not invalidated after writes. The
   `supersedes` field is parsed but never consulted in retrieval. Together
   these mean the agent's own learning is partially invisible to itself within
   a single pipeline run.

## Design Principles

- Entry files are the single source of truth.
- `index.yaml` is a generated cache artifact, not a hand-edited file.
- One rebuild function, called eagerly after writes, keeps the read path fast.
- `supersedes` actively suppresses older lessons from retrieval.
- The retrieval cache is invalidated after any lesson mutation.
- All existing tests pass without behavioral change (except for the
  `TestReflect` tests that manually clean up the index — those simplify).

## Scope

### In scope (Phase 1)

- `rebuild_index()` function that materializes `index.yaml` from entry files.
- Write-path simplification: all mutation functions write the entry file, then
  call `rebuild_index()`.
- Dedup and ID generation scan entry files, not the index.
- `supersedes` filtering in `_query_lessons()`.
- Retrieval cache invalidation after lesson mutations.
- Migration: diff current index against rebuilt index, fix any existing
  inconsistencies.
- Test updates.

### Out of scope (Phase 2, later)

- Append-only event log.
- Richer history / audit trail.
- Index versioning or checksumming.

---

## Phase 1 Implementation Plan

### Step 1: Add `rebuild_index()` to `promotion.py`

Create a function that globs `entries/*.yaml`, reads metadata from each file,
and writes a fresh `index.yaml`.

```
def rebuild_index() -> dict:
    """Rebuild index.yaml from canonical entry files.

    Globs entries/*.yaml, reads id/title/severity/category/status/applies_when
    from each, sorts by id, writes index.yaml with settings header.
    Returns the index dict for immediate use.
    """
```

Design decisions:

- **Sort order:** Alphabetical by lesson ID. This makes the generated file
  deterministic and diff-friendly.
- **Settings preservation:** Keep `settings.max_prompt_entries: 7` as a
  hardcoded default in the rebuild function. This value is not per-lesson
  metadata; it belongs in the generator.
- **Error handling:** If an entry file fails to parse, log a warning and skip
  it. Do not let one corrupt file block the entire index rebuild.
- **Performance:** At 126 files averaging ~30 lines each, the glob + parse
  takes ~50-80ms. This runs only on writes (rare), not on reads.

### Step 2: Simplify write paths in `promotion.py`

Replace the dual-write pattern in each mutation function:

| Function | Current pattern | New pattern |
|----------|----------------|-------------|
| `capture_lesson()` | Write entry + `_append_to_index()` | Write entry + `rebuild_index()` |
| `validate_lesson()` | Write entry + `_update_index_status()` | Write entry + `rebuild_index()` |
| `promote_lesson()` | Write entry + `_update_index_status()` | Write entry + `rebuild_index()` |
| `archive_lesson()` | Write entry + `_update_index_status()` | Write entry + `rebuild_index()` |
| `boost_confidence()` | Write entry only (no index update) | Write entry only (unchanged — confidence is not in the index) |
| `distill()` | N transitions × `_update_index_status()` | N entry writes + one `rebuild_index()` at the end |
| `record_semantic_extension_trace()` | Up to 3 chained index writes | Same chain, but each now calls `rebuild_index()` internally. Consider batching: mark a `_dirty` flag and rebuild once at the end of the function. |

Remove `_append_to_index()` and `_update_index_status()` entirely. Keep
`_load_index()` as-is — it is still used by `_load_lesson_index()` in
`store.py` for the hot-tier read path.

**Batching optimization for `record_semantic_extension_trace()`:** This
function can call `capture_lesson()` → `validate_lesson()` →
`promote_lesson()` in sequence. To avoid three rebuilds, introduce an
internal context manager or a `_suppress_rebuild` flag:

```python
_suppress_rebuild = False

def _maybe_rebuild():
    if not _suppress_rebuild:
        rebuild_index()
```

Then `record_semantic_extension_trace()` sets `_suppress_rebuild = True`
before the chain and calls `rebuild_index()` once at the end.

### Step 3: Move dedup and ID generation to scan entry files

Currently `capture_lesson()` calls `_load_index()` for both dedup (title
word-overlap) and `_generate_id()` (next sequential ID). Both should scan
entry files instead:

```python
def _scan_entry_metadata() -> list[dict]:
    """Read id and title from all entry files for dedup and ID generation."""
    entries_dir = _LESSONS_DIR / "entries"
    results = []
    for path in sorted(entries_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
            if data and "id" in data:
                results.append({"id": data["id"], "title": data.get("title", "")})
        except Exception:
            continue
    return results
```

This replaces the index as the authority for dedup and ID generation. The cost
is reading ~126 small files (~50ms) instead of one large file (~10ms), but
`capture_lesson()` is called infrequently and correctness matters more than
speed here.

**Optimization:** If this becomes a hot path later, cache the scan result and
invalidate it when `rebuild_index()` runs.

### Step 4: Add `supersedes` filtering to `_query_lessons()` in `store.py`

After scoring and before hydrating, build the suppression set:

```python
def _query_lessons(self, expanded_features, spec, max_n):
    scored = [...]  # existing scoring logic

    # Hydrate top candidates
    candidates = []
    for _, idx_entry in scored[:max_n * 2]:  # over-fetch to allow filtering
        lesson = self._load_lesson(idx_entry.id)
        if lesson is not None:
            candidates.append(lesson)

    # Build superseded set from all hydrated lessons
    superseded_ids = set()
    for lesson in candidates:
        superseded_ids.update(lesson.supersedes)

    # Filter and trim
    filtered = [l for l in candidates if l.id not in superseded_ids]
    return filtered[:max_n]
```

Key design choice: we over-fetch by 2x before filtering to ensure we still
return `max_n` lessons after removing superseded ones. The `supersedes` field
is already populated in entry files and parsed by `_load_lesson()` (line 415
of `store.py`).

### Step 5: Invalidate retrieval cache after lesson mutations

Add a module-level hook that `KnowledgeStore` can register with, so promotion
functions can signal that the store's caches are stale:

**Option A (simple, recommended):** Add `_invalidate_store_caches()` to
`promotion.py` that calls the singleton store's `clear_runtime_caches()` +
reloads the lesson index:

```python
def _invalidate_store_caches():
    """Signal the KnowledgeStore singleton that lesson data has changed."""
    try:
        from trellis.agent.knowledge import get_store
        store = get_store()
        store._lesson_index.clear()
        store._load_lesson_index()
        store._retrieval_cache.clear()
        store._lessons_cache.clear()
    except Exception:
        pass  # Store may not be initialized yet
```

Call this at the end of `rebuild_index()`, so every write that rebuilds the
index also refreshes the in-memory state.

**Option B (decoupled):** Use a version counter. `rebuild_index()` increments
a module-level `_index_generation` counter. `KnowledgeStore.retrieve_for_task()`
checks whether its `_last_seen_generation` matches, and reloads if not. This
avoids the import cycle but adds complexity. Recommend Option A for now.

### Step 6: Migration script

A one-time script to validate the transition:

```bash
# 1. Rebuild index from entry files
python -c "from trellis.agent.knowledge.promotion import rebuild_index; rebuild_index()"

# 2. Diff against current index
diff <(python -c "...print current index...") <(python -c "...print rebuilt index...")

# 3. Inspect differences — these are existing inconsistencies to fix
```

If the diff is non-empty, the inconsistencies should be resolved by trusting
the entry files (the richer source). Commit the rebuilt index as the
migration commit.

### Step 7: Test updates

**Tests that simplify:**

- `TestReflect.test_attribute_success_boosts_confidence` (lines 777-811):
  Currently does manual cleanup of both the entry file AND the index. After
  the refactor, cleanup of the entry file is sufficient — the index rebuilds
  from entries.
- `TestReflect.test_auto_validate_and_promote` (lines 813-838): Same manual
  cleanup pattern. Simplifies to just removing the entry file.

**Tests that need updating:**

- `TestPromotion.isolated_store` fixture (lines 567-586): Currently
  monkeypatches `_LESSONS_DIR` and `_INDEX_PATH`. Should also monkeypatch
  the `_scan_entry_metadata` function (or its underlying directory path) and
  the `rebuild_index` function to use the isolated directory.

**New tests to add:**

- `test_rebuild_index_matches_entries`: Create N entry files with known
  metadata, call `rebuild_index()`, verify the generated index has exactly N
  entries with correct fields.
- `test_rebuild_index_skips_corrupt_files`: Create one valid and one corrupt
  entry file, verify `rebuild_index()` succeeds with one entry.
- `test_rebuild_index_deterministic`: Call `rebuild_index()` twice, verify
  identical output.
- `test_supersedes_excludes_from_retrieval`: Create lesson B that supersedes
  lesson A. Verify `_query_lessons()` returns B but not A.
- `test_capture_invalidates_retrieval_cache`: Capture a lesson, verify the
  retrieval cache was cleared.
- `test_semantic_extension_trace_rebuilds_index_once`: Monkeypatch
  `rebuild_index` to count calls. Verify that
  `record_semantic_extension_trace()` with capture+validate+promote calls
  `rebuild_index` exactly once (not three times).
- `test_distill_rebuilds_index_once`: Same pattern for `distill()`.

---

## Files Changed

| File | Change |
|------|--------|
| `trellis/agent/knowledge/promotion.py` | Add `rebuild_index()`, `_scan_entry_metadata()`, `_maybe_rebuild()`. Remove `_append_to_index()`, `_update_index_status()`. Simplify all write functions. Add `_invalidate_store_caches()`. Add batch suppression for `record_semantic_extension_trace()` and `distill()`. |
| `trellis/agent/knowledge/store.py` | Update `_query_lessons()` to filter by `supersedes`. No change to `_load_lesson_index()` — it still reads `index.yaml`. |
| `tests/test_agent/test_knowledge_store.py` | Update `TestPromotion` fixture. Simplify `TestReflect` cleanup. Add new tests per Step 7. |

**Files NOT changed:**

- `store.py._load_lesson_index()` — unchanged, still reads `index.yaml`
- `retrieval.py` — unchanged, operates on hydrated `Lesson` objects
- `autonomous.py` — unchanged, calls `retrieve_for_task()` which is unaffected
- `experience.py` — legacy shim, separate index in `experience/`, unrelated
- `schema.py` — all dataclasses unchanged
- `reflect.py` — unchanged. Note: `_should_distill()` (line 522) reads
  `index.yaml` directly to count candidates. This is a read-only path that
  continues to work because the generated index contains the same status
  fields. No code change needed.

---

## Risk Assessment

**Low risk:** The read path (`_load_lesson_index()` → `_query_lessons()` →
`_load_lesson()`) is unchanged. The only read-path addition is `supersedes`
filtering, which is additive and cannot break existing behavior (existing
lessons have empty `supersedes` lists).

**Medium risk:** The write path is refactored, but the externally visible
behavior (entry file + index file both updated) is preserved. The main risk
is that `rebuild_index()` introduces a subtle ordering or field-name
difference compared to the current `_append_to_index()` output. The migration
diff (Step 6) catches this before any code change is committed.

**Mitigation:** Run the full test suite (`pytest tests/ -x -q -m "not
integration"`) after each step. The 37 knowledge system tests and 109 agent
tests provide good coverage of the read and write paths.

---

## Sequence

1. Write `rebuild_index()` and `_scan_entry_metadata()` — no callers yet, purely additive.
2. Run migration diff (Step 6) to surface any existing inconsistencies. Fix them.
3. Wire `rebuild_index()` into write paths, remove `_append_to_index()` / `_update_index_status()`.
4. Add `_invalidate_store_caches()`, wire it into `rebuild_index()`.
5. Add `supersedes` filtering to `_query_lessons()`.
6. Update and add tests.
7. Run full test suite. Commit.
