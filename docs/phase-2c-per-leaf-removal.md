# Phase 2c — Remove the per-leaf build path

> Status: **plan, not yet executed**.
> Prerequisite: Phase 2b (chunked build) is complete, merged or on `feat/chunked-build`, and has passed production validation on a 42.8B-point dataset. See [phase-2b-chunked-build.md](phase-2b-chunked-build.md) for context.
> Expected scope: ~700 line deletion from `src/octree.rs` plus small public-API trimming. ~1 day of focused work including validation.

## 1. Goal & motivation

Delete the per-leaf build path entirely. The chunked path has been validated on:
- All 15 integration tests (point conservation, hierarchy structure, determinism, cross-strategy)
- The 42.8B-point production dataset (exact point conservation, matching max depth and compression ratios to per-leaf)
- Tile-level comparison against per-leaf output (identical per-level breakdowns)
- The writer-phase memory fix (see Phase 2b final commit)

Per the Phase 2b design doc §9.5, the per-leaf path was kept as "an escape hatch during a stability period" that has now elapsed. Maintaining two code paths for the same operation is a long-term maintenance cost with no remaining benefit.

**What removal buys us:**
- ~700 lines of deletion from `src/octree.rs` (the largest file)
- Elimination of the strategy-dispatch layer in `distribute` and `build_node_map`
- Elimination of `BuildStrategy` from the public API (simpler `PipelineConfig`)
- Removal of the conditional progress-step count logic in `main.rs`
- One fewer CLI flag (`--build-strategy`)
- One fewer thing to explain in the README
- The pipeline becomes "how the tool works," not "one of two ways the tool works"

**What removal does NOT touch:**
- `src/writer.rs` — unchanged, it doesn't know about strategies
- `src/chunking.rs` — unchanged, no per-leaf coupling
- The writer's memory budget + mini-batching (the Phase 2b final commit stays)
- `bottom_up_levels` — used by both paths via the chunked `build_chunk_in_memory`; **stays**
- `grid_sample` — core LOD primitive used by both paths; **stays**
- `read_node`, `write_node_to_temp`, `node_path`, `all_node_keys` — shared primitives; **stay**

## 2. Pre-flight checklist

Before starting the removal, verify the current branch state is healthy. All of these should pass:

```sh
cargo build
cargo fmt --check
cargo clippy --all-targets --workspace -- -D warnings
cargo test  # expect 15 passing integration tests + 38 lib unit tests + 1 doctest
```

Confirm the default is **still** `PerLeaf` in the code right now (this plan is the commit that flips it to chunked-only — so we start from the current state). Check:

```sh
grep 'impl Default for BuildStrategy\|default_value_t = BuildStrategyArg' src/lib.rs src/main.rs
```

If those lines say `PerLeaf`, you're at the right starting point.

## 3. Inventory of what gets deleted

### 3.1 `src/octree.rs` — the bulk of the deletion

These are the per-leaf-only functions and helpers, with their exact line ranges as of commit `97863b4` on `feat/chunked-build`. **Line numbers will shift as you delete; treat these as "delete in this order from bottom to top" to preserve the earlier ranges.**

Order the deletes from the **bottom** of the file upward so earlier line numbers stay valid. The full delete list:

| Lines | Function | Notes |
|---|---|---|
| 1561-1652 | `fn grid_sample_streaming` | **See §3.1a** — this has one remaining chunked caller that must be addressed first |
| 1378-1550 | `fn bottom_up_on_disk` | 173 lines. Level-by-level disk-backed build loop. |
| 1223-1254 | `fn bottom_up_in_memory` | 32 lines. Thin wrapper: reads leaves from disk and delegates to `bottom_up_levels`. The delegate stays. |
| 1131-1217 | `fn build_node_map_per_leaf` | 87 lines. The top-level per-leaf build routine. |
| 1115-1120 | **Dispatch in `pub fn build_node_map`** | Flatten from 6 lines to a direct call — see §3.1b |
| 982-1110 | `fn normalize_leaves` | 129 lines. Splits oversized leaf files on disk. Not needed in chunked path (adaptive in-memory subdivision replaces it). |
| 903-925 | `fn flush_buffers` | 23 lines. Helper for per-leaf distribute only. |
| 815-901 | `fn distribute_per_leaf` | 87 lines. The per-leaf distribute loop. |
| 803-808 | **Dispatch in `pub fn distribute`** | Flatten from 6 lines to a direct call — see §3.1c |
| 779-795 | `fn merge_into_buffers` | 17 lines. Helper for per-leaf distribute only. |

**Total deletion: ~640 lines** (plus associated comments, doc strings).

### 3.1a — Decision point: `grid_sample_streaming`

`grid_sample_streaming` is called from **two** places:

1. Per-leaf `bottom_up_on_disk` (will be deleted)
2. **Chunked `merge_chunk_tops` as a fallback for oversized parents** ([octree.rs:2254](../src/octree.rs#L2254))

Look at the relevant block in `merge_chunk_tops`:

```rust
if !large_parents.is_empty() {
    // Should not happen with a well-formed chunk plan, but if it
    // does, fall back to grid_sample_streaming which is bounded
    // (one child at a time, ~80 MB per parent for the grid set
    // and accumulator).
    for (parent, children, est_mem) in &large_parents {
        debug!(...);
        self.grid_sample_streaming(parent, children)?;
        all_new_parents.push(*parent);
        keys_by_level.entry(d).or_default().insert(*parent);
    }
}
```

In production this branch never fired on the 42.8B-point dataset. But deleting it entirely removes our safety net against pathological chunk plans.

**Two options — pick one before starting the delete:**

**Option A (recommended, safer): keep `grid_sample_streaming`.**
- Don't delete lines 1561-1652.
- Move it higher in the file alongside `grid_sample` (just before `impl Drop for OctreeBuilder`) for locality with its chunked caller.
- Update its doc comment to remove references to `bottom_up_on_disk` and explain it's now only used as the chunked merge fallback.
- **Keeps ~92 lines alive.**

**Option B (fully clean): delete `grid_sample_streaming` and the fallback.**
- Delete lines 1561-1652.
- In `merge_chunk_tops`, replace the fallback with `anyhow::bail!("merge_chunk_tops: parent {parent:?} has {est_mem} MB of children, exceeds memory_budget; this indicates a pathological chunk plan — please file a bug")`.
- Any future run that hits this edge case gets a clear error instead of silent slow-path fallback.
- **Removes all ~92 lines.**

**Recommendation:** **Option A.** Trivial cost in lines retained (we keep ~640 of the ~700 total deletion). Zero risk of OOMing a future edge case. `grid_sample_streaming` is well-tested via the per-leaf path's many runs and can be trusted. If we later find it's truly never hit, a subsequent cleanup commit can remove it.

If you pick Option A, **remove it from the "delete" list** above (skip lines 1561-1652) and instead move it to section §4.5 as "relocate and rename" — see below.

### 3.1b — Flatten the `build_node_map` dispatch

Current code (roughly lines 1115-1120):

```rust
pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
    match config.build_strategy {
        crate::BuildStrategy::PerLeaf => self.build_node_map_per_leaf(config),
        crate::BuildStrategy::Chunked => self.build_node_map_chunked(config),
    }
}
```

**Becomes:**

```rust
pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
    self.build_node_map_chunked(config)
}
```

Then **rename** `build_node_map_chunked` back to something friendlier (since it's the only one):
- Option 1: Keep the name `build_node_map_chunked` for now and let a follow-up rename it. Keeps this commit focused.
- Option 2: Rename to `build_node_map_impl` or just inline it into `pub fn build_node_map`.

**Recommendation:** Rename `build_node_map_chunked` → delete the wrapper entirely and make the chunked impl be the `pub fn build_node_map`. Cleanest. One less indirection.

### 3.1c — Flatten the `distribute` dispatch

Current code (roughly lines 803-808):

```rust
pub fn distribute(&mut self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
    match config.build_strategy {
        crate::BuildStrategy::PerLeaf => self.distribute_per_leaf(input_files, config),
        crate::BuildStrategy::Chunked => self.distribute_chunked(input_files, config),
    }
}
```

**Becomes** (after deleting the dispatch and renaming):

```rust
pub fn distribute(&mut self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
    // (the body of the current `distribute_chunked` function)
}
```

Same treatment — inline `distribute_chunked` into `pub fn distribute`. Rename or inline.

### 3.2 `src/lib.rs`

Delete the entire `BuildStrategy` enum definition and its documentation. Currently at lines 159-183 (approx):

```rust
/// Strategy for the distribute + build phases.
/// ...
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    ...
}
```

Delete the `build_strategy` field from `PipelineConfig` (around line 199):

```rust
pub struct PipelineConfig {
    // ...
    pub build_strategy: BuildStrategy,  // <-- DELETE THIS LINE
    // ...
}
```

Update the doctest at the top of `lib.rs` to remove the import and field:

```rust
//! use copc_converter::{BuildStrategy, Pipeline, PipelineConfig};  // <-- remove BuildStrategy
//!     build_strategy: BuildStrategy::PerLeaf,  // <-- delete this line
```

### 3.3 `src/main.rs`

Delete:
- Import of `BuildStrategy` from the `use copc_converter::...` line (line 4)
- The `#[arg(long, value_enum, default_value_t = BuildStrategyArg::PerLeaf)] build_strategy` field on `Args` (lines 54-55)
- The entire `BuildStrategyArg` enum and its `From<BuildStrategyArg> for BuildStrategy` impl (lines 58-78)
- The `total_steps_for(BuildStrategy)` function (lines 184-194) — replace with a `const TOTAL_STEPS: u32 = 4;` (chunked path has 4 stages)
- The `args.build_strategy.into()` reference in the `total_steps_for` call site and the `PipelineConfig` construction

**Replace `total_steps_for(...)` call** with direct `TOTAL_STEPS` reference:

```rust
// BEFORE:
let total_steps = total_steps_for(args.build_strategy.into());

// AFTER:
const TOTAL_STEPS: u32 = 4;
```

Constructing `PipelineConfig`:

```rust
// BEFORE:
let config = PipelineConfig {
    memory_budget,
    temp_dir: args.temp_dir,
    temporal_index: args.temporal_index,
    temporal_stride: args.temporal_stride,
    progress: Some(progress),
    build_strategy: args.build_strategy.into(),   // <-- DELETE
    chunk_target_override: args.chunk_target,
};
```

### 3.4 `src/bin/copc_analyze.rs`

Delete:

```rust
// Around line 418:
build_strategy: copc_converter::BuildStrategy::PerLeaf,
```

### 3.5 `README.md`

Update the Library usage example (lines 75-95):

```rust
// BEFORE:
use copc_converter::{BuildStrategy, Pipeline, PipelineConfig, collect_input_files};
let config = PipelineConfig {
    ...
    build_strategy: BuildStrategy::PerLeaf,
    ...
};

// AFTER:
use copc_converter::{Pipeline, PipelineConfig, collect_input_files};
let config = PipelineConfig {
    ...  // no build_strategy field
    ...
};
```

### 3.6 `tests/integration.rs`

**Delete**:
- `chunked_and_perleaf_have_same_point_count` (lines 1082-1132) — cross-strategy comparison, no longer meaningful.

**Rename and simplify**:
- `chunked_strategy_produces_valid_copc` → `produces_valid_copc`
- `chunked_strategy_preserves_all_points` → `preserves_all_points`
- `chunked_strategy_deterministic_within_strategy` → merge into the existing `deterministic_output` test or delete (redundant now).
- `chunked_multi_chunk_preserves_all_points` → **keep as-is with its name**; it's a regression test for the Phase 2b point-loss bug and the `--chunk-target` flag is still relevant for forcing multi-chunk scenarios on small inputs. Just remove the `--build-strategy chunked` arg (chunked is the default).

For each remaining test that passes `--build-strategy chunked`, **remove that argument** — it's redundant. Example:

```rust
// BEFORE:
run_converter_with_args(input, output, &["--build-strategy", "chunked"]);

// AFTER:
run_converter(input, output);
// or
run_converter_with_args(input, output, &[]);  // if there are other args
```

Update `low_memory_produces_valid_output`'s comment:

```rust
// BEFORE:
// 1 MB budget forces the out-of-core build path and streaming grid_sample
// for parents whose children exceed this tiny limit.

// AFTER:
// 1 MB budget forces the chunked path to produce many small chunks,
// exercising the merge step under tight memory pressure. Verify the
// chunked path still produces a valid COPC file when memory-constrained.
```

The test body doesn't need changes — the assertions are strategy-agnostic.

**Expected final integration test count:** 15 → 13 (if we delete both `chunked_and_perleaf_have_same_point_count` and `chunked_strategy_deterministic_within_strategy`) or 14 (if we keep the latter as an alias for the general determinism test).

### 3.7 `docs/phase-2b-chunked-build.md`

Leave the Phase 2b doc alone — it's a historical design artifact. Add a note at the very top of the doc:

```markdown
> **Phase 2b status (as of Phase 2c completion): the per-leaf path referenced throughout this doc has been deleted. Chunked is now the only path. See `phase-2c-per-leaf-removal.md` for the removal commit.**
```

## 4. Execution order

Execute in this order to keep the build passing at every step. Each step should compile and all tests should pass before moving to the next.

### Step 4.1 — Flip the default first (defensive baseline)

Before deleting anything, make chunked the default. This is a single-line change:

```rust
// In src/lib.rs BuildStrategy definition:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    PerLeaf,
    #[default]
    Chunked,  // <-- add #[default] here, remove it from PerLeaf
}
```

```rust
// In src/main.rs BuildStrategyArg:
#[arg(long, value_enum, default_value_t = BuildStrategyArg::Chunked)]  // <-- change PerLeaf → Chunked
build_strategy: BuildStrategyArg,
```

```rust
// In src/bin/copc_analyze.rs (if it matters — analyze doesn't use distribute/build anyway):
build_strategy: copc_converter::BuildStrategy::Chunked,
```

Build, run the full test suite. **All 15 integration tests should still pass.** Per-leaf path tests pass because they explicitly pass `--build-strategy per-leaf`; chunked tests pass because chunked is now the default.

**Commit 1:** "Flip build strategy default to chunked"

### Step 4.2 — Delete per-leaf from `octree.rs` in bottom-up order

Now start the actual deletion. **Delete from the bottom of `octree.rs` upward** so earlier line numbers don't shift as you go.

Concretely, open `src/octree.rs` in the editor and delete in this order:

1. **`grid_sample_streaming` (lines 1561-1652)** — OR if you chose Option A above (§3.1a), leave it alone and see §4.5 for the relocation step.
2. **`bottom_up_on_disk` (lines 1378-1550)**.
3. **`bottom_up_in_memory` (lines 1223-1254)**.
4. **`build_node_map_per_leaf` (lines 1131-1217)**.
5. **Flatten `build_node_map` dispatch (lines 1115-1120)** — replace with direct call to `build_node_map_chunked`, or inline `build_node_map_chunked`'s body.
6. **`normalize_leaves` (lines 982-1110)**.
7. **`flush_buffers` (lines 903-925)**.
8. **`distribute_per_leaf` (lines 815-901)**.
9. **Flatten `distribute` dispatch (lines 803-808)**.
10. **`merge_into_buffers` (lines 779-795)**.

After each deletion, run:

```sh
cargo build
```

Expect errors about unused imports or helpers. Chase them to completion — any function that becomes `#[allow(dead_code)]` or is only called by deleted code should also be deleted.

Things that are likely to become dead code after this pass:
- Imports of `HashSet` (only `bottom_up_on_disk` uses it outside of tests) — verify, delete if unused
- `SampleTask` / `SampleResult` type aliases if they're only used in the deleted `bottom_up_in_memory`
- Possibly some debug-only imports

Run `cargo clippy --all-targets --workspace -- -D warnings` after deletion to catch any straggling warnings.

**Commit 2:** "Delete per-leaf build path implementation"

### Step 4.3 — Remove strategy dispatch plumbing

At this point `build_strategy` still exists on `PipelineConfig` but has only one valid value. Delete:

- `BuildStrategy` enum from `src/lib.rs`
- `build_strategy` field from `PipelineConfig`
- `BuildStrategy` from the doctest in `src/lib.rs`
- `BuildStrategyArg` and its impl from `src/main.rs`
- `total_steps_for` in `src/main.rs` → replace with `const TOTAL_STEPS: u32 = 4;`
- `build_strategy` field from the `PipelineConfig` literal construction in `src/main.rs`
- `build_strategy` field from `src/bin/copc_analyze.rs`
- `build_strategy` field from `README.md` library example
- `BuildStrategy` import from `README.md`

Build. Clippy. Test.

**Commit 3:** "Remove BuildStrategy enum and dispatch plumbing"

### Step 4.4 — Trim tests and CLI

- Delete `chunked_and_perleaf_have_same_point_count` integration test.
- Optionally delete `chunked_strategy_deterministic_within_strategy` (redundant with `deterministic_output`).
- Rename `chunked_strategy_*` tests (drop the prefix, since there's only one strategy now).
- Remove `--build-strategy chunked` from all test invocations.
- Update `low_memory_produces_valid_output` comment.
- In `src/main.rs`, delete the `#[arg(long, value_enum, ...)] build_strategy` field entirely from `Args`.
- Keep the `--chunk-target` hidden flag — it's still useful for regression tests and debugging.

Build. Clippy. Test. Expect 13 or 14 passing integration tests.

**Commit 4:** "Rename chunked tests and drop --build-strategy flag"

### Step 4.5 — (Option A only) Relocate `grid_sample_streaming`

If you kept `grid_sample_streaming` for the merge fallback:

- Move the function from its current location (around line 1561 originally) to just before `impl Drop for OctreeBuilder`, alongside `grid_sample`. They're logical siblings.
- Update its doc comment:

```rust
/// Streaming grid-sample for a single oversized parent.
///
/// Used as a fallback by `merge_chunk_tops` when a parent group's combined
/// child points exceed the memory budget. Processes one child at a time so
/// only one child's points are ever in memory. The grid occupancy set
/// (max 128³ entries ≈ 48 MB) and the parent accumulator (max 128³ points
/// ≈ 80 MB) stay resident; everything else is streamed from/to disk.
///
/// In practice this path is rarely hit for well-formed chunk plans
/// (merge-sparse-cells sizes chunks to fit the budget by construction),
/// but it's kept as a safety net for degenerate inputs.
///
/// Tradeoff: points are not Morton-sorted across children, so spatial
/// coherence of the parent is slightly worse than in-memory grid_sample.
fn grid_sample_streaming(&self, parent: &VoxelKey, children: &[VoxelKey]) -> Result<()> {
```

**Commit 5:** "Relocate grid_sample_streaming and update docs"

### Step 4.6 — Final validation

```sh
cargo fmt
cargo clippy --all-targets --workspace -- -D warnings
cargo test
```

Expect: **13-14 integration tests, 38 lib unit tests, 1 doctest**, all passing.

Smoke test end-to-end on the test input:

```sh
cargo build --bin copc_converter --bin inspect_copc --features tools
./target/debug/copc_converter tests/data/input.laz /tmp/test.copc.laz --progress plain
./target/debug/inspect_copc /tmp/test.copc.laz
```

Expect: 829,570 points, 35 nodes, max depth 4, 3.84 MB file size. Same as the Phase 2b baseline.

## 5. Wall-time budget

Rough estimate for a focused session:

| Step | Time |
|---|---|
| Pre-flight + baseline sanity check | 10 min |
| Step 4.1 (flip default) | 10 min |
| Step 4.2 (delete from octree.rs) | 90 min — the biggest step, lots of mechanical editing and compile-loop chasing |
| Step 4.3 (remove strategy plumbing) | 30 min |
| Step 4.4 (trim tests and CLI) | 30 min |
| Step 4.5 (relocate grid_sample_streaming, if Option A) | 15 min |
| Step 4.6 (final validation + smoke test) | 20 min |
| **Total** | **~3-4 hours** |

With ~500 lines deleted and net LOC reduction of ~600+, expect `cargo build` warnings at each intermediate state that need to be chased. Budget time for that; don't rush it.

## 6. Risks & rollback

**Risk 1: A test in `low_memory_produces_valid_output` uncovers a chunked-path bug at 1 MB budget that was never exercised before.**
- The chunked path's `chunk_target_override` interacts with `compute_chunk_target`'s MIN_CHUNK_POINTS clamp at 1M. A 1 MB budget yields a raw target below 1M → clamp kicks in → chunk target = 1M → 830K points → 1 chunk → trivially passes.
- **Mitigation:** actually run the test during 4.2 after each delete step. If it fails, investigate before proceeding.

**Risk 2: grid_sample_streaming is deleted (Option B) and a production run hits a pathological chunk plan.**
- **Mitigation:** go with Option A (keep it). If you picked Option B and production fails, revert Commit 2 via `git revert` and add a streaming fallback.

**Risk 3: The CLI `--build-strategy` flag being removed breaks external scripts that pass it.**
- `copc-converter` is version 0.0.0-dev with no known external users passing this flag.
- **Mitigation:** leave a deprecated stub if concerned: `#[arg(long, value_enum, hide = true)] build_strategy: Option<BuildStrategyArg>` that warns but is ignored. I'd skip this — it's a 0.x crate and internal-only so far.

**Rollback strategy:** each step is a commit. If a step breaks something unfixable, `git reset --hard HEAD~1` to undo and regroup. If the whole removal turns out to be premature, `git reset --hard <pre-removal-commit>` restores everything. The chunked path's correctness has no dependency on the per-leaf path existing.

## 7. Success criteria

After all steps complete, the following must be true:

- ✅ `cargo build` clean
- ✅ `cargo fmt --check` clean
- ✅ `cargo clippy --all-targets --workspace -- -D warnings` clean
- ✅ `cargo test` passes (≥ 13 integration tests + 38 unit tests + 1 doctest)
- ✅ `tests/data/input.laz` round-trips through the pipeline to a valid COPC file
- ✅ `inspect_copc` on the output reports correct hierarchy structure
- ✅ No occurrences of `BuildStrategy`, `per_leaf`, `PerLeaf`, `distribute_per_leaf`, `build_node_map_per_leaf`, `normalize_leaves`, `bottom_up_on_disk`, `bottom_up_in_memory`, `merge_into_buffers`, `flush_buffers` in `src/*` or `tests/*` (except `grid_sample_streaming` if Option A)
- ✅ `src/octree.rs` has dropped from ~2700 lines to ~2050 lines
- ✅ `src/lib.rs` line count shrinks by ~30 lines
- ✅ `src/main.rs` line count shrinks by ~30 lines

## 8. What's NOT in this phase (deferred)

- **Renaming `distribute_chunked`/`build_node_map_chunked` to drop the `_chunked` suffix.** Recommended but cosmetic. Can be a follow-up commit or part of step 4.2 if you inline them into the public methods.
- **Renaming `MAX_LEAF_POINTS`**, `GRID_CELLS_PER_AXIS`, etc. They're still used by the chunked path unchanged.
- **`chunked_plan` field on `OctreeBuilder`.** It's the one piece of state shared between distribute and build in the chunked path. Stays.
- **Writer-phase memory improvements** from the Phase 2b final commit. Those are orthogonal and already landed.
- **The `--chunk-target` hidden CLI flag.** Still useful for testing; keep it.
- **Documentation updates beyond the README.** The inline code comments that reference "the per-leaf path" should be updated but that's a low-priority cleanup pass after this.

## 9. Questions for the person executing this plan

Before starting, decide:

1. **Option A or Option B for `grid_sample_streaming`?** Recommendation: **Option A** (keep it, relocate). Safer, small cost.
2. **Should `chunked_strategy_deterministic_within_strategy` be deleted or kept as an alias for the general `deterministic_output` test?** Recommendation: delete. `deterministic_output` (the original pre-Phase-2b test) already covers this.
3. **Should the commit sequence be 5 commits or squashed into 1?** Recommendation: keep as 5 commits — makes the `git log` history of the removal easier to bisect if something regresses.
4. **Should we also rename `distribute_chunked` → `distribute` (inline) in the same pass?** Recommendation: yes, do it at step 4.2 when flattening the dispatch. Less code to explain.

## 10. Summary in one paragraph

The chunked build has proven itself on the production 42.8B-point dataset with exact point conservation, equivalent LOD quality, and memory that fits within budget. The per-leaf path is now dead code maintained only as insurance. This phase deletes the per-leaf path wholesale: ~640 lines from `src/octree.rs`, the `BuildStrategy` enum, the CLI flag, and the strategy-conditional tests. The chunked path becomes the only path and the public API gets smaller. Five focused commits, ~3-4 hours of work, and meaningful reduction in long-term maintenance surface area.
