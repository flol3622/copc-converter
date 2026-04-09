# Phase 2b — Chunked Build Design

> **Phase 2b status (as of Phase 2c completion): the per-leaf path referenced throughout this doc has been deleted. Chunked is now the only path. See `phase-2c-per-leaf-removal.md` for the removal commit.**

> Status: **draft for review**, not yet implemented.
> Author: design pass on `feat/chunked-build` branch, after Phase 2a counting prototype validated chunk distribution on a real 42.8B-point dataset.
> Reviewers: please scrutinize §5 (the merge step) — it is the only part with real algorithmic risk.

## 1. Goals & non-goals

**Goal:** replace the per-leaf-temp-file distribute + multi-pass build pipeline with the Schütz et al. 2020 chunked architecture, while keeping the existing public `Pipeline` API and writer phase unchanged.

**Goals:**
- Convert distribute from "per-point HashMap insert + per-leaf file" to "per-point LUT lookup + per-chunk file" so the temp directory contains thousands of medium files instead of hundreds of thousands of tiny ones. Critical for EFS / network-FS clients.
- Replace `bottom_up_on_disk` (multi-pass disk I/O) with per-chunk in-memory build, eliminating the level-by-level disk thrashing that dominates current build wall time on large datasets.
- Use the memory budget aggressively: bigger budget → bigger chunks → more in-memory work, less disk I/O.
- Preserve point conservation (every input point ends up in exactly one node).
- Preserve the writer contract: `build_node_map()` returns `Vec<(VoxelKey, usize)>` and the writer reads each node via `builder.read_node(key)`. Chunked-build is a drop-in replacement that satisfies the same contract.

**Non-goals:**
- Custom LAZ decoder. Phase 2a confirmed `laz-rs` is competitive (~1.8× per-core perf vs Schütz on equivalent silicon). Decoder optimization is a separate, larger conversation.
- Recursive re-chunking for oversized chunks. Phase 2a measured 0 chunks above 2× target on real data. Park this until evidence demands it.
- Approximate Poisson-disk sampling (Schütz §5). Our existing Morton-ordered grid sampling is equivalent in quality and proven correct.
- Changes to the typestate `Pipeline<S>` API. The new path lives behind a `BuildStrategy::Chunked` enum value on `PipelineConfig`.

## 2. Inputs from Phase 2a (data we trust)

From the analyzer run on the 42.8B-point dataset (8-core EPYC 9354P, 64 GB pod):

| Metric | Value |
|---|---|
| Total points | 42.84B |
| Grid resolution chosen | 512³ |
| Dynamic chunk target | 36.4M points |
| Chunks generated | 3,422 |
| Max chunk size | 46.4M (1.27× target) |
| Chunks above 2× target | **0** |
| P50 chunk size | 11.27M |
| P99 chunk size | 35.49M |
| Counting pass throughput | 16.45 Mpts/s on 8 cores |
| Memory during count | flat ~950 MB |

Per-level distribution:
| Level | Chunks | Total points | Avg/chunk |
|---|---|---|---|
| 2 | 4 | 15.77M | 3.94M |
| 3 | 17 | 241.46M | 14.20M |
| 4 | 55 | 619.15M | 11.26M |
| 5 | 259 | 3.87B | 14.94M |
| 6 | 518 | 7.07B | 13.64M |
| 7 | 1336 | 18.36B | 13.74M |
| 8 | 1081 | 10.99B | 10.16M |
| 9 | 152 | 1.68B | 11.04M |

**Implications for Phase 2b:**
- No oversized chunks → no recursive re-chunking needed.
- Chunks span levels 2 through 9 → the merge step at coarse levels has to handle multi-level chunk roots, not just one fixed chunk depth.
- Most points (85%) live at levels 6-8 → that's where parallelism scales best.
- 882 chunks (~26%) are below 10% of target → tail of small chunks, not pathological. Handle naturally; optimize only if measurement shows they hurt.

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Pipeline                                                                 │
│                                                                          │
│  Scanned ──► Validated ──► Distributed ──► Built ──► Written             │
│                                                                          │
│                              ▲                ▲                          │
│                              │                │                          │
│                  if BuildStrategy::Chunked:   │                          │
│                  ┌───────────┴───────┐    ┌───┴────────────────┐         │
│                  │ count_points      │    │ build_chunked_     │         │
│                  │ merge_sparse_cells│    │   node_map         │         │
│                  │   = ChunkPlan     │    │                    │         │
│                  │                   │    │ ┌───────────────┐  │         │
│                  │ distribute_to_    │    │ │ per-chunk:    │  │         │
│                  │   chunks          │    │ │ load + grid_  │  │         │
│                  │   = chunk files   │    │ │ sample bottom-│  │         │
│                  │                   │    │ │ up to chunk   │  │         │
│                  └───────────────────┘    │ │ root level    │  │         │
│                                           │ └───────────────┘  │         │
│                                           │                    │         │
│                                           │ merge_chunk_tops:  │         │
│                                           │   bottom-up from   │         │
│                                           │   chunk roots to   │         │
│                                           │   global root      │         │
│                                           │   = canonical      │         │
│                                           │     per-node files │         │
│                                           └────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────┘
```

The chunked path replaces the entire current `distribute()` and `build_node_map()` implementation. The output it produces (canonical per-node temp files + `Vec<(VoxelKey, usize)>`) is exactly what the existing writer consumes — **no writer changes**.

## 4. Pipeline stages in detail

### 4.1 Counting + chunk planning

Already implemented as `chunking::analyze_chunking()`. Returns a `ChunkPlan { grid_size, grid_depth, chunk_target, total_points, chunks: Vec<PlannedChunk> }`.

For Phase 2b, this is invoked at the start of `Pipeline<Validated>::distribute()` when `BuildStrategy::Chunked` is selected, before the actual point distribution starts.

**Output:** `ChunkPlan` plus a fast **lookup table** that maps any fine-grid cell `(gx, gy, gz)` at `grid_depth` to the index of the chunk it belongs to. Same `Vec<u32>` shape as the counting grid: `grid_size³` entries, each holding a chunk index. Built once, read many times during distribute.

**Memory:** `512³ × 4 B = 537 MB` for the LUT, same as the counting grid. Lives for the duration of the distribute pass; freed after.

### 4.2 Distribute to chunks

**Replaces:** the existing per-leaf distribute path.

**What it does:** stream every input file in parallel, classify each point into its chunk via the LUT (one lookup per point), and append it to that chunk's temp file. Writes go through a bounded LRU writer cache exactly like the per-leaf distribute path uses (the cache code is reusable as-is — only the keys change from `VoxelKey` to chunk index).

**Key code:**
```rust
fn distribute_to_chunks(
    &self,
    input_files: &[PathBuf],
    plan: &ChunkPlan,
    config: &PipelineConfig,
) -> Result<()> {
    // Build the cell → chunk_index LUT.
    let lut = build_chunk_lut(plan); // Vec<u32>, len = grid_size³

    // Set up shard subdirs (same pattern as the per-leaf parallel distribute).
    let num_workers = distribute_worker_count(input_files.len());
    let chunks_root = self.tmp_dir.join("chunks");
    fs::create_dir_all(&chunks_root)?;

    // Each worker writes into its own shards/{worker_id}/ subdir.
    // Files are named "chunk_{idx:06}" — flat, no nesting.
    (0..num_workers).into_par_iter().try_for_each(|worker_id| -> Result<()> {
        let shard_dir = chunks_root.join(format!("shards/{worker_id}"));
        fs::create_dir_all(&shard_dir)?;
        let mut cache = ChunkWriterCache::new(per_worker_cap);

        for path in &input_files[start..end] {
            let mut reader = las::Reader::from_path(path)?;
            loop {
                points.clear();
                let n = reader.read_points_into(chunk_size as u64, &mut points)?;
                if n == 0 { break; }

                // For each point: LUT lookup → chunk index → append to that chunk's file.
                for p in &points {
                    let raw = self.convert_point(p);
                    // Use round-tripped coordinates to match the converter exactly.
                    let rx = raw.x as f64 * self.scale_x + self.offset_x;
                    let ry = raw.y as f64 * self.scale_y + self.offset_y;
                    let rz = raw.z as f64 * self.scale_z + self.offset_z;
                    let key = point_to_key(rx, ry, rz, self.cx, self.cy, self.cz, self.halfsize, plan.grid_depth);
                    let cell_idx = key.x as usize + key.y as usize * g + key.z as usize * g * g;
                    let chunk_idx = lut[cell_idx];
                    cache.append(chunk_idx, &shard_dir, &raw)?;
                }
                // ... progress reporting ...
            }
        }
        cache.flush_all()
    })?;

    // Merge per-worker shards into canonical chunk files.
    self.merge_chunk_shards(&chunks_root, num_workers, plan.chunks.len())?;
    Ok(())
}
```

**Memory bound:** same shape as the per-leaf parallel distribute. Per-worker chunk size is computed from the budget; per-worker LRU cap on open chunk files is `DISTRIBUTE_OPEN_FILES_CAP / num_workers`. With ~3,422 chunks total and a per-worker cap of ~100, each worker thrashes its LRU but writes are amortized via 64 KB BufWriters. Far less thrashing than the per-leaf path which had hundreds of thousands of files.

**On-disk layout:**
```
{tmp_dir}/
  chunks/
    shards/
      0/
        chunk_000000
        chunk_000001
        ...
      1/
        ...
    chunk_000000   ← merged result
    chunk_000001
    ...
```

**Critical detail — temp file format inside a chunk file:** the same `RawPoint::write_bulk` format used by the per-leaf path. **No extra headers, no per-point metadata.** A chunk file is just `RawPoint::BYTE_SIZE × N` bytes. The chunk's *identity* (level, x, y, z) lives in the `ChunkPlan` indexed by chunk index, not in the file itself.

**Why this works:** the per-chunk in-memory build (§4.3) gets its chunk metadata from the `ChunkPlan`, not from the file. The file is just a point soup waiting to be octree-d.

### 4.3 Per-chunk in-memory build

**Replaces:** `bottom_up_in_memory` and `bottom_up_on_disk` for the chunk's interior levels.

**What it does:** for each chunk in parallel:
1. Load the entire chunk's point soup from `chunks/chunk_{idx:06}` into RAM as `Vec<RawPoint>`. Worst case: 46.4M points × 38 B = ~1.8 GB.
2. Compute the chunk's *local* leaf depth: enough subdivisions below the chunk's root level so the average leaf holds ≤ `MAX_LEAF_POINTS` points. For a 36M-point chunk, that's `ceil(log8(360))` ≈ 3 extra levels below the chunk root.
3. Classify each point into its leaf voxel (same `point_to_key` machinery as today).
4. Run **the existing `bottom_up_in_memory` machinery** on just this chunk's nodes, walking from the leaf level up to the chunk root level (NOT all the way to global root).
5. As nodes are produced, write each one to its canonical per-node temp file via `write_node_to_temp(key, &points)` — same path layout as the per-leaf build path uses today.

**Key insight: `grid_sample` is reusable as-is.** It takes a `parent: &VoxelKey`, a `Vec<(child_index, RawPoint)>`, and returns `(parent_pts, per_child_remaining)`. It's a pure function over the builder's immutable geometry (`cx`, `cy`, `cz`, `halfsize`, `scale`). It doesn't care whether the parent is the global root or a deep chunk-internal node. **Phase 2b reuses it without modification.**

**Required modification to `bottom_up_in_memory`:** add a `min_level: u32` parameter so the loop becomes:
```rust
for d in (min_level..actual_max_depth).rev() { ... }
```
For a chunk with root at level 5, we pass `min_level = 5`, and the loop walks from chunk's leaf depth (e.g., 8) up to level 5 inclusive on the descent (`d = 7, 6, 5`). The parent at level 5 is the chunk root and is the final node produced by the per-chunk build. Levels 0-4 are deferred to the merge step.

**Code shape:**
```rust
fn build_chunk_in_memory(
    &self,
    chunk: &PlannedChunk,
    chunk_file: &Path,
) -> Result<Vec<(VoxelKey, usize)>> {
    // 1. Load points.
    let points = read_chunk_file(chunk_file)?; // Vec<RawPoint>

    // 2. Compute chunk-local leaf depth.
    let leaf_depth = chunk_local_leaf_depth(chunk.point_count, MAX_LEAF_POINTS);
    let global_leaf_level = chunk.level + leaf_depth;

    // 3. Classify into leaf voxels.
    let mut leaves: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
    for p in points {
        let key = point_to_voxel_key_at_level(&p, global_leaf_level);
        leaves.entry(key).or_default().push(p);
    }

    // 4. Bottom-up build, stopping at chunk.level.
    let nodes = self.bottom_up_in_memory_ranged(
        leaves,
        global_leaf_level,
        chunk.level, // min_level — stop here
    )?;

    // 5. Write each node to canonical temp file.
    for (k, pts) in &nodes {
        self.write_node_to_temp(k, pts)?;
    }

    Ok(nodes.into_iter().map(|(k, pts)| (k, pts.len())).collect())
}
```

**Parallelism:** chunks are processed in parallel via rayon. Each chunk is independent — no shared state, no coordination, embarrassingly parallel. Number of concurrent chunks limited by **memory budget**:

```
max_concurrent_chunks = memory_budget × 0.6 / max_chunk_bytes
```

For 64 GB budget, max chunk = 46.4M × ~120 B/pt working set = ~5.6 GB → ~6-7 concurrent chunks. We can use rayon's `par_iter().with_max_len(1)` or just wrap in a semaphore. This is the same constraint pattern as `bottom_up_on_disk`'s small-parent batching — we can lift that batching code or simplify.

**Output of step 4.3:** the union over all chunks of `Vec<(VoxelKey, usize)>` for nodes at levels ≥ chunk.level (which varies by chunk). Plus per-node temp files written to `tmp_dir/{level}_{x}_{y}_{z}` in the canonical format.

### 4.4 Merge chunk tops — **the only algorithmically interesting step**

**This is where the design needs the most scrutiny.** Read carefully.

**The problem.** After step 4.3, we have nodes at levels ≥ chunk.level for each chunk independently. But we also need nodes at levels **above** the chunk root, going all the way up to the global root at level 0. These coarse-level nodes are formed by combining points from *multiple chunks* whose ancestors meet at the same coarse voxel.

Concretely: if two chunks have roots at level 7 and they share a level-6 ancestor (i.e., their level-7 keys differ only by their last bit and have the same level-6 parent), then to build the level-6 node, we need a `grid_sample` over the union of their level-7 root points. And so on up the tree.

**The tricky bit: chunks live at variable levels.** If chunk A has root at level 7 and chunk B has root at level 5, they don't share a common parent at level 5 — chunk B *is* a level-5 node. So the merge has to handle:
- Multiple chunks meeting at a common ancestor (the common case)
- A coarser chunk being a node at a level where finer chunks are still children (the variable-level case)

**Conceptual algorithm:**

```
1. Initialize a "current set" of (VoxelKey → Vec<RawPoint>) from the chunk roots
   produced by step 4.3. Each chunk contributes ONE node at its chunk.level —
   the chunk root. This is the "parent points" returned by the bottom-up build
   inside that chunk.

2. For d in (0..max_chunk_level).rev():
       Find all nodes in current_set at level (d+1) whose parent is at level d.
       Group them by parent.
       For each parent group:
         If the group has ≥ 1 child node at level (d+1):
           Run grid_sample(parent, all_child_points, n_children).
           parent_pts → write as a new node at level d.
           per_child remaining → write back to update child node files.

       Now also pull in any chunk that has its root EXACTLY at level d.
       Those chunks are already nodes at level d — their points contribute to
       further grid_samples at coarser levels (handled in the next iteration).

3. After d = 0, we have the global root.
```

**The variable-level wrinkle, restated more carefully.** A chunk root at level 5 contributes points to:
- Itself as a level-5 node (already built in step 4.3)
- Its level-4 parent's grid_sample input (along with sibling nodes at level 5)
- Its level-3 grandparent's grid_sample input (transitively, via the level-4 parent's sampled points)
- ... up to level 0

The points that "flow up" are not the chunk's *original* points — they're the *sampled* parent_pts that the chunk's bottom-up build produced at the chunk's root. So step 4.3 needs to **return the chunk root's parent_pts** as the unit of work for the merge step.

**Refined algorithm:**

```
After step 4.3 we have, for each chunk i:
  - All interior nodes at levels [chunk_i.level, leaf_level_i] written to disk.
  - chunk_root_pts[i] = the points sampled into the chunk's root at chunk_i.level.
    (These are also written to disk as the level-chunk_i.level node, but we keep
     them in memory to feed the merge step.)

Merge step:

current_nodes: HashMap<VoxelKey, Vec<RawPoint>> = empty
for chunk_i in chunks:
    current_nodes.insert(chunk_i.root_key, chunk_root_pts[i].clone())

for d in (0..max_chunk_level).rev():
    # Find all nodes currently at level d+1.
    children_at_d_plus_1: Vec<VoxelKey> =
        current_nodes.keys().filter(|k| k.level == d+1).collect()

    # Group by parent at level d.
    parent_to_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new()
    for c in children_at_d_plus_1:
        parent_to_children[c.parent()].push(c)

    # Process each parent.
    for (parent, child_keys) in parent_to_children:
        # Build the (child_index, RawPoint) input list.
        all_pts: Vec<(usize, RawPoint)> = Vec::new()
        for (ci, ck) in child_keys.iter().enumerate():
            for p in current_nodes.remove(&ck).unwrap_or_default():
                all_pts.push((ci, p));

        if all_pts.is_empty(): continue

        let (parent_pts, remaining) = self.grid_sample(&parent, all_pts, child_keys.len());

        # Update child nodes on disk: each child's node file is now reduced
        # (some points moved up to parent). Write the remaining points back.
        for (ci, ck) in child_keys.iter().enumerate():
            self.write_node_to_temp(&ck, &remaining[ci])?;

        # Insert parent into current_nodes for the next iteration.
        current_nodes.insert(parent, parent_pts);

        # Also write parent to disk so the writer can read it.
        self.write_node_to_temp(&parent, &current_nodes[&parent])?;
```

**Memory for the merge step.** `current_nodes` holds the points that "flow up" through the tree. At any iteration, that's the union of:
- Chunk root parent_pts for chunks at the *current* level
- Promoted parent_pts from prior iterations

Each chunk root contributes ≤ `GRID_CELLS_PER_AXIS³ = 128³ ≈ 2.1M points` (the cap that `grid_sample` enforces). So with 3,422 chunks, the *upper bound* on memory is `3,422 × 2.1M × 38 B ≈ 273 GB`. That's catastrophic.

**In practice it's much smaller** because:
- Most chunks are at levels 7-8 (your dataset). Their parent_pts get sampled down to ~2.1M (or less) at each step.
- Only chunks at level d+1 contribute to current_nodes when processing level d.
- Promoted parent_pts at level d are themselves bounded by 2.1M each.

**Realistic worst-case for your dataset:** at level 8 → level 7, there are 1,081 chunks at level 8 plus all the level-7+ chunks promoted from finer levels. Say 1,500 nodes total at level 8 going into the level-7 grid_samples. Each holds ≤ 2.1M points × 38 B = ~80 MB. Total: **~120 GB**. Still too much.

**Mitigation: process the merge step level-by-level with batching, just like `bottom_up_on_disk` does today.** Inside one level, group parents into batches that fit in budget × 0.6, process each batch in parallel, free memory between batches.

This re-introduces some of the complexity from `bottom_up_on_disk` that I was hoping to delete. But it's only for the *coarse* levels (0 to max_chunk_level, typically 0-8), not for every level all the way down. The majority of the build work has already happened inside chunks where memory is naturally bounded.

**Better mitigation: don't keep `current_nodes` in memory at all. Read it back from disk per iteration.** Each promoted node was just written to disk via `write_node_to_temp`. The next iteration reads them back via `read_node`. This is exactly `bottom_up_on_disk`'s pattern, but only for the small set of coarse-level nodes.

**My recommendation:** for the merge step, **adapt `bottom_up_on_disk`'s batched parallel pattern** rather than trying to keep everything in memory. The number of nodes at coarse levels is small (single-digit thousands at most for a 42.8B-point dataset) and they're sampled to ≤ 2.1M points each, so disk I/O at this level is cheap. The complexity I was hoping to eliminate is mostly in *normalize_leaves* + the *fine-level* on-disk path; the coarse-level merge is fundamentally similar to what we have today, just operating on far fewer nodes.

**Refined merge step:**

```rust
fn merge_chunk_tops(
    &self,
    chunk_results: Vec<(VoxelKey, Vec<RawPoint>)>, // chunk root keys + their points
    config: &PipelineConfig,
) -> Result<Vec<(VoxelKey, usize)>> {
    let max_chunk_level = chunk_results.iter().map(|(k, _)| k.level).max().unwrap_or(0);

    // Write chunk roots to disk so the merge can re-read them.
    for (k, pts) in &chunk_results {
        self.write_node_to_temp(k, pts)?;
    }

    // Track keys by level, just like bottom_up_on_disk does today.
    let mut keys_by_level: HashMap<i32, HashSet<VoxelKey>> = HashMap::new();
    for (k, _) in &chunk_results {
        keys_by_level.entry(k.level as i32).or_default().insert(*k);
    }

    // Bottom-up from max_chunk_level to 0.
    for d in (0..max_chunk_level).rev() {
        let child_keys = keys_by_level.get(&(d as i32 + 1)).cloned().unwrap_or_default();
        if child_keys.is_empty() { continue; }

        // Group children by parent at level d.
        let mut parent_to_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
        for ck in &child_keys {
            if let Some(parent) = ck.parent() {
                parent_to_children.entry(parent).or_default().push(*ck);
            }
        }

        // Batch parents by estimated memory cost (same heuristic as bottom_up_on_disk).
        let parents: Vec<_> = parent_to_children.into_iter().collect();
        let batches = batch_by_memory(&parents, config.memory_budget);

        let mut new_parent_keys = HashSet::new();
        for batch in batches {
            batch.par_iter().try_for_each(|(parent, children)| -> Result<()> {
                let mut all_pts: Vec<(usize, RawPoint)> = Vec::new();
                for (ci, ck) in children.iter().enumerate() {
                    self.read_node_into_indexed(ck, ci, &mut all_pts)?;
                }
                if all_pts.is_empty() { return Ok(()); }

                let (parent_pts, remaining) = self.grid_sample(parent, all_pts, children.len());

                // Write remaining points back to children.
                for (ci, ck) in children.iter().enumerate() {
                    self.write_node_to_temp(ck, &remaining[ci])?;
                }

                // Write parent.
                if !parent_pts.is_empty() {
                    self.write_node_to_temp(parent, &parent_pts)?;
                }
                Ok(())
            })?;

            for (parent, _) in batch {
                new_parent_keys.insert(*parent);
            }
        }

        keys_by_level.entry(d as i32).or_default().extend(new_parent_keys);
    }

    // Enumerate result.
    let mut result = Vec::new();
    for level_keys in keys_by_level.values() {
        for key in level_keys {
            let n = self.read_node(key)?.len();
            if n > 0 {
                result.push((*key, n));
            }
        }
    }
    Ok(result)
}
```

This pattern is **almost identical to `bottom_up_on_disk`** ([octree.rs:1077-1248](src/octree.rs#L1077)). I should be able to refactor `bottom_up_on_disk` into a generic "merge bottom-up across levels" routine that both the current path and the new path call, with just a different starting set of keys. Or I can copy-paste-tweak it for now and unify later.

**Why this is correct:**
- At every level, we run `grid_sample` over all the children of each parent.
- `grid_sample` was already proven to handle the sampling correctly — it doesn't matter whether the children are leaves, intermediate nodes, or chunk roots. They're all just `Vec<RawPoint>` from its perspective.
- Point conservation holds because `grid_sample` preserves it (parent_pts + remaining = original input).
- The COPC zero-point-node guarantee is preserved by `grid_sample`'s "every child that contributed must keep at least one" hack ([octree.rs:1428-1439](src/octree.rs#L1428-L1439)).

**Why this might still be wrong:**
- **The variable-level chunk wrinkle.** When a chunk root is at level 5 and the merge is processing level 4 → 3, the level-5 chunk root contributes to the level-4 grid_sample as a *sibling* of any level-5 nodes coming from chunks at level 6+. Specifically, the level-4 parent's children might be a mix of "real" level-5 nodes (from chunks rooted at level 5) and "promoted" level-5 nodes (from coarse-level merges of level-6+ chunk roots). The algorithm above handles this naturally because it just looks at "what nodes exist at level d+1" without caring about provenance. **I think this is correct, but I want a reviewer's eyes on it.**

- **`grid_sample`'s child_index tracking.** The `child_index` parameter is used by `grid_sample` to guarantee "every contributing child keeps at least one point" ([octree.rs:1428-1439](src/octree.rs#L1428-L1439)). When merging across chunks, the child_index is a synthetic 0..n_children index over the children we found at level d+1, not a fixed octant index. That's fine — the guarantee still holds: any child node that contributed points keeps at least one. **Verify in implementation.**

### 4.5 Output to writer

After step 4.4 returns, we have:
- `Vec<(VoxelKey, usize)>` listing every node in the hierarchy at every level
- Per-node temp files at canonical paths `{tmp_dir}/{level}_{x}_{y}_{z}` containing `RawPoint`s

This is **exactly the contract** the writer expects. `writer::write_copc(builder, node_keys, config)` works unchanged.

## 5. Edge cases and known wrinkles

**Empty chunks.** If a chunk file ends up with 0 points (because the counting estimate was off, or because of a degenerate input), skip it. The merge step naturally handles this — its parent will simply have one fewer contributing child.

**Chunks at the global root level (level 0).** For tiny datasets, the entire input might fit in one chunk at level 0. In that case there are no merge levels (max_chunk_level = 0 = global root), and step 4.4 is a no-op. The chunk's bottom-up build already produces the global root.

**Chunks rooted at level d when another chunk is rooted at level < d covering the same region.** This **cannot happen** by construction: the merge-sparse-cells algorithm produces chunks at *most one level per region*. Each fine cell maps to exactly one chunk in the LUT.

**Chunks whose root level is below the global octree leaf depth.** This is the common case. A chunk at level 5 needs to subdivide internally to level 5+leaf_depth so its leaves hold ≤ MAX_LEAF_POINTS. The chunk-local leaf depth is `ceil(log8(chunk_point_count / MAX_LEAF_POINTS))`. For a 36M-point chunk: `log8(360) ≈ 2.84` → 3 extra levels → leaves at level 8.

**`MAX_LEAF_POINTS` consistency.** The current code has `MAX_LEAF_POINTS = 100_000`. Chunked-build should use the same constant so leaves are sized identically to the per-leaf path. This means the *total* number of leaves in the chunked build is approximately the same as today (~430K leaves for 42.8B points), but they're built per-chunk instead of all at once.

**Coordinate consistency.** Phase 2a uses raw `p.x/p.y/p.z` for the counting grid. Phase 2b's distribute step should use **round-tripped** coordinates (`raw.x * scale + offset`) to match the converter's voxel assignment exactly, otherwise points classified into one chunk by the LUT might land in a different leaf voxel during the per-chunk build. This is a one-line change inside `distribute_to_chunks` (use `raw.x * scale + offset` instead of `p.x` when calling `point_to_key`).

**Storage layout interaction with the LUT.** The LUT maps grid cell → chunk index. For Phase 2b's distribute, we need to convert a point into a grid cell index quickly. Two paths:
1. Call `point_to_key(p, ..., grid_depth)` and flatten — same as Phase 2a.
2. Inline the math: `(p - origin) / cell_size` per axis. ~3× faster than `point_to_key` because it skips the depth-9 loop. Worth doing.

I'll inline. The benchmark cost is `~30 ns/point × 42.8B points = ~21 minutes` of pure CPU saved over the dataset by avoiding the loop. Not negligible.

**Drop semantics.** `OctreeBuilder` already implements `Drop` to remove `tmp_dir`. The chunked path uses subdirectories of `tmp_dir` (`tmp_dir/chunks/`, `tmp_dir/chunks/shards/...`) so the existing Drop cleans them up automatically. No new Drop logic needed.

## 6. Dispatch and compatibility

Add to `lib.rs`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    /// Original per-leaf-temp-file build with multi-pass disk I/O.
    /// Lower per-chunk memory pressure, slower on network filesystems.
    #[default]
    PerLeaf,
    /// Chunked build (Schütz et al. 2020): counting-sort into medium chunks,
    /// independent per-chunk in-memory build, merge at coarse levels.
    /// Faster overall, far faster on network filesystems.
    Chunked,
}

pub struct PipelineConfig {
    pub memory_budget: u64,
    pub temp_dir: Option<PathBuf>,
    pub temporal_index: bool,
    pub temporal_stride: u32,
    pub progress: Option<Arc<dyn ProgressObserver>>,
    pub build_strategy: BuildStrategy, // NEW
}
```

In `OctreeBuilder::distribute()`:
```rust
pub fn distribute(&self, files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
    match config.build_strategy {
        BuildStrategy::PerLeaf => self.distribute_per_leaf(files, config),
        BuildStrategy::Chunked => self.distribute_chunked(files, config),
    }
}
```

In `OctreeBuilder::build_node_map()`:
```rust
pub fn build_node_map(&self, config: &PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
    match config.build_strategy {
        BuildStrategy::PerLeaf => self.build_per_leaf(config),
        BuildStrategy::Chunked => self.build_chunked(config),
    }
}
```

The chunked path needs state shared between distribute and build (the `ChunkPlan`). Store it on `OctreeBuilder` as an `Option<ChunkPlan>` field, populated by `distribute_chunked`, consumed by `build_chunked`. This mirrors how the per-leaf path implicitly shares state via the temp file layout.

CLI flag in `main.rs`:
```rust
#[arg(long, value_enum, default_value_t = BuildStrategy::PerLeaf)]
build_strategy: BuildStrategy,
```

Default stays `PerLeaf` until chunked has been validated against real datasets and the integration tests. Flip the default in a follow-up commit.

## 7. Testing strategy

**Existing integration tests.** Run all 10 with `--build-strategy chunked`. The most important ones:

- `low_memory_produces_valid_output` — verifies bounded memory with the new path.
- `deterministic_output` — verifies that the chunked path produces bit-identical output to itself across runs.
- `point_count_conservation` — verifies that no points are lost or duplicated.
- Cross-strategy comparison: it is **NOT** required that per-leaf and chunked produce bit-identical output. They may produce *different but equally valid* COPC files because grid_sample tie-breaking can differ when point ordering differs at the leaf level. The cross-strategy comparison should verify "both files are valid COPC, contain the same total point count, and have the same set of voxel keys" — not byte equality.

**New unit tests for chunking-specific code:**
- `merge_chunk_tops` correctness on synthetic inputs (already prototyped in chunking::tests).
- `chunk_local_leaf_depth` calculation.
- LUT construction from `ChunkPlan`.
- Empty chunk handling.

**End-to-end smoke test on the analyzer dataset.** Run the full chunked pipeline on the same 42.8B-point dataset that Phase 2a measured. Verify:
- Total wall time vs the previous fix-bounded-memory run (target: faster)
- Memory curve vs the previous run (target: similar or lower except writer phase)
- Output COPC opens cleanly in untwine, lasinfo, and the Potree viewer
- Point count matches input headers exactly

## 8. Implementation order

Suggested order, with rough effort estimates:

1. **Add `BuildStrategy` enum + dispatch** (½ day). Pure plumbing. Default to `PerLeaf`. All existing tests still pass without behavior change.

2. **`distribute_to_chunks` with shard merge** (1 day). Reuses `LeafWriterCache` pattern, just keyed by chunk index instead of `VoxelKey`. Reuses the existing parallel-distribute structure.

3. **`build_chunk_in_memory` for a single chunk** (1 day). Refactors `bottom_up_in_memory` to take a `min_level` parameter. Adds the chunk-local leaf depth calculation. Tests in isolation on synthetic inputs.

4. **`merge_chunk_tops`** (1 day). Adapts `bottom_up_on_disk`'s batching pattern. **This is the highest-risk step** — review §4.4 carefully before coding.

5. **Wire it all together in `build_chunked()`** (½ day). Calls `chunking::analyze_chunking()` for the plan, then dispatches steps 2-4.

6. **Run integration tests with `--build-strategy chunked`** (½ day). Fix issues. Add cross-strategy comparison test.

7. **End-to-end run on the 42.8B-point dataset** (1 day, mostly waiting). Compare wall time and memory curve against the fix-bounded-memory baseline. Document results.

8. **Decide whether to flip the default** (½ day). If chunked is reliably faster and passes all tests, flip the default to `Chunked`. Keep `PerLeaf` as escape hatch.

**Total effort: ~5-6 days of focused work.** Plus a calendar day or two for the end-to-end run on the pod.

## 9. Things I'm asking the reviewer to verify

This section is the **explicit ask** — please push back on any of these.

1. **§4.4 merge step correctness, especially the variable-level chunk handling.** I described the algorithm in two passes (conceptual and refined). Does the refined version handle the case where chunks at levels 5 and 7 both have points contributing to a level-3 ancestor? My claim is yes, because the merge naturally promotes level-7 → level-6 → level-5 first, and by the time we process level-5 → level-4, the chunk root at level 5 is just another node at level 5 alongside the promoted nodes.

2. **`grid_sample`'s child_index protection during merge.** The "every contributing child keeps at least one point" rule was added for COPC's no-zero-node requirement. When merging across chunks, the child_index is synthetic. Does that break the protection logic? My claim is no, because the protection is *about* child indices (whichever ones contributed), not about specific octant identities.

3. **Per-chunk leaf depth calculation.** A chunk at level 5 holding 36M points needs 3 extra levels of subdivision to hit `MAX_LEAF_POINTS = 100K` per leaf on average. But subdivision is not uniform — dense regions in the chunk will have more points per leaf. Should the per-chunk leaf depth be computed *adaptively* (like `from_scan` does for the global builder), or is uniform `+3 levels` good enough? My current plan: compute `chunk_local_leaf_depth = ceil(log8(point_count / MAX_LEAF_POINTS))` and rely on the chunk's internal `bottom_up_in_memory` to handle imbalance via `grid_sample`'s natural cascading. If a leaf overflows MAX_LEAF_POINTS, that's fine — `MAX_LEAF_POINTS` is the *target*, not a hard limit. The current per-leaf path enforces it via `normalize_leaves` after distribute, but in chunked-build the chunks have already been sized to fit in RAM, so a slightly oversized leaf is acceptable. Worth confirming.

4. **Memory accounting for the per-chunk build phase.** Each chunk loaded for build costs `point_count × ~120 B/pt` working set (Vec<RawPoint> + grid_sample input + per-child remaining + HashSet scratch). For the 46M-point worst-case chunk, that's ~5.5 GB. With 64 GB budget × 0.6 safety = 38.4 GB available, we can run ~7 concurrent chunks. With ~3,422 chunks total and 7 concurrent, the build phase takes `3422 / 7 × per_chunk_time`. If each chunk takes 5-10 seconds: `3422 / 7 × 7.5s = ~60 minutes` for the build phase, fully parallel. This matches my Phase 2b time estimate. Sanity check: is the 120 B/pt working set realistic? The current per-leaf path uses `MEM_PER_POINT = 128` for the same purpose, derived empirically. Reusable.

5. **Whether to delete the per-leaf path eventually.** After chunked-build is the default and battle-tested for, say, a month of production runs, should we delete the per-leaf path entirely? It would simplify `octree.rs` significantly (drop `bottom_up_on_disk`, `bottom_up_in_memory`, `normalize_leaves`, `grid_sample_streaming`, the LRU writer cache for per-leaf, ~800 lines total). Or do we keep it as an escape hatch indefinitely for users who hit edge cases the chunked path can't handle? My recommendation: **delete after one month of stability**, on the principle that maintaining two code paths is itself a long-term cost and the chunked path covers 100% of use cases by design. But this is a judgment call worth getting input on.

## 10. Open questions for follow-up

Not blocking Phase 2b implementation, but worth noting:

- **`MAX_LEAF_POINTS` tuning.** Current value 100K is hardcoded. Untwine uses 100K too. Schütz uses 10K. For COPC viewers, larger leaves mean fewer LAZ chunks → smaller hierarchy file → faster initial load, but more data per chunk → higher latency for individual chunk fetches. Worth empirically benchmarking on a few datasets after Phase 2b lands.
- **Storage of intermediate chunk files.** Each chunk file is uncompressed `RawPoint` data. For a 42.8B-point dataset that's ~1.6 TB of temp space. We could optionally LZ4-compress the chunk files at write time and decompress at read time. LZ4 typically gives 2-3× compression on point data and is fast enough to be a net win on slow storage. **Skip for Phase 2b**, revisit if temp storage becomes a complaint.
- **Per-chunk early writer kickoff.** After a chunk finishes its in-memory build, its node files are *complete and final* — no other chunk will modify them. We could start the LAZ encoding for those nodes in parallel with the rest of the build. This would overlap the build and writer phases. Significant complexity, modest savings (writer is ~25% of wall time). **Skip for Phase 2b**, revisit later.

---

**Sign off needed before coding begins.** Once §9 questions are resolved, I'll start with step 8.1 (the dispatch plumbing) and work through the list.
