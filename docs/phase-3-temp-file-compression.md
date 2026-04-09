# Phase 3 — Compressed temp files

> Status: **plan, not yet executed**.
> Goal: cut the on-disk footprint of the chunked-build temp directory by
> compressing the per-chunk and per-node `RawPoint` temp files with LZ4.
> Expected effort: half a day of focused work including benchmarking.

## 1. Motivation

On a large run (42.8B points) the temp directory peaks at roughly the
full raw-point footprint: `38 bytes × N_points`. For 42.8B points that's
~1.6 TB of scratch space. On hosts with small NVMe scratch disks (K8s
workers, bare-metal boxes with a single SSD) that can be the limiting
factor for input size, not memory or wall time.

`RawPoint` records are highly compressible:
- `x`/`y`/`z` scaled ints share leading bits across spatially adjacent points
- `classification`, `return_number`, `number_of_returns`, `scan_angle`,
  `user_data`, `point_source_id` are usually near-constant across a file
- RGB/NIR are all-zero for many datasets
- `gps_time` has tight locality within a flight line

Empirically, LZ4 on comparable lidar buffers gets **~3-5×** shrink at
>1 GB/s per core, and **zstd -1** gets **~5-8×** at ~400 MB/s. Both are
more than fast enough to stay off the critical path for any reasonable
storage tier.

**What this buys us:**
- ~3-5× smaller scratch directory → ~1.6 TB → ~350-500 GB for the
  production run
- Enables larger inputs on space-constrained workers
- Reduces I/O bytes hitting network filesystems (EFS/NFS/FUSE-backed
  disks), which is usually the real bottleneck on those substrates

**What it does NOT buy us:**
- Wall-time improvement on fast local NVMe (compute becomes the
  bottleneck; compression is roughly a wash but not a big win)
- Smaller final COPC output — the output path is untouched; this is
  purely a scratch-file change
- Lower memory usage — `ChunkWriterCache`'s `BufWriter` buffers stay,
  and per-node reads still materialize the full `Vec<RawPoint>`

## 2. Codec choice

**Recommendation: LZ4 frame format (lz4_flex crate), opt-in via
`--temp-compression=lz4`.**

| Codec  | Ratio   | Compress speed | Decompress speed | Notes |
|--------|---------|----------------|------------------|-------|
| none   | 1×      | ∞              | ∞                | Current behavior |
| lz4    | ~3-4×   | >1 GB/s/core   | >2 GB/s/core     | Frame format supports streaming |
| zstd-1 | ~5-7×   | ~400 MB/s/core | ~800 MB/s/core   | Better ratio, more CPU |
| zstd-3 | ~6-8×   | ~200 MB/s/core | ~800 MB/s/core   | Default zstd level; too slow for a default |

**Why LZ4 by name in the flag even though we may support zstd later:**
start with a single codec. If zstd pays off in benchmarking (§8) we add
it as a second value without reworking the plumbing.

**Crate: `lz4_flex` (pure Rust, zero external deps, `frame` module
implements the official LZ4 frame format).** Alternative `lz4` crate
wraps the C library — avoid the C dep. `lz4_flex` is already
battle-tested in the Rust data ecosystem (used by datafusion, polars,
etc.).

Add to `Cargo.toml`:
```toml
lz4_flex = { version = "0.11", default-features = false, features = ["frame"] }
```

## 3. Scope — what gets compressed

Two kinds of temp files exist in the chunked path:

1. **Per-chunk shard files** written by distribute, and the canonical
   merged chunk files consumed at the start of build.
   Paths: `tmp_dir/chunks/shards/{worker}/chunk_N` → merged to
   `tmp_dir/chunks/chunk_N`.
   I/O pattern: many small appends (shards) → one sequential read
   (build). Writer lifetime is bounded by the LRU cache
   (`CHUNKED_OPEN_FILES_CAP = 512` writers max).

2. **Per-node files** written by the bottom-up merge step when it
   rewrites children with remaining points and parents with promoted
   points (`write_node_to_temp` / `read_node`), and by the grid-sample
   fallback path.
   Paths: `tmp_dir/{level}_{x}_{y}_{z}`.
   I/O pattern: full overwrite-and-flush per call; read-once later.

Both should be compressed. Category (1) dominates footprint during the
distribute phase; category (2) dominates during the build phase for
chunky inputs where the merge step rewrites lots of nodes.

**Out of scope:** temp files written by `copc_analyze` and the writer
phase. The writer-phase temp files (inside `src/writer.rs`) are already
gone by the time peak footprint matters — if they turn out to be
significant, a follow-up can reuse the same compression abstraction.

## 4. Design

### 4.1 The CLI flag

```rust
// src/main.rs
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum TempCompressionArg {
    /// No compression (default, fastest on local NVMe).
    #[default]
    None,
    /// LZ4 frame format (~3-4× smaller, ~1 GB/s/core).
    Lz4,
}

#[arg(long, value_enum, default_value_t = TempCompressionArg::None)]
temp_compression: TempCompressionArg,
```

Mirror in `src/lib.rs`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TempCompression {
    #[default]
    None,
    Lz4,
}
```

Add to `PipelineConfig`:
```rust
/// Compression for intermediate temp files. Defaults to none.
/// LZ4 cuts scratch-disk usage ~3-4× for a small CPU cost.
pub temp_compression: TempCompression,
```

Wire through `src/main.rs` and `src/bin/copc_analyze.rs` (the latter
just passes `TempCompression::None` for now — analyze doesn't hit the
distribute/build temp path heavily).

### 4.2 I/O abstraction

Two new helper functions on `OctreeBuilder` that encapsulate the codec
choice. Put them next to `read_node` / `write_node_to_temp` so all
reads/writes go through one pair:

```rust
impl OctreeBuilder {
    /// Open a writer for a temp file, applying compression if configured.
    /// Caller provides the already-opened File (possibly in append mode).
    fn temp_writer(&self, f: File) -> Box<dyn Write + Send> {
        match self.config_temp_compression {
            TempCompression::None => Box::new(BufWriter::new(f)),
            TempCompression::Lz4 => Box::new(
                lz4_flex::frame::FrameEncoder::new(BufWriter::new(f))
            ),
        }
    }

    /// Open a reader for a temp file, applying decompression if configured.
    fn temp_reader(&self, f: File) -> Box<dyn Read> {
        match self.config_temp_compression {
            TempCompression::None => Box::new(BufReader::new(f)),
            TempCompression::Lz4 => Box::new(
                lz4_flex::frame::FrameDecoder::new(BufReader::new(f))
            ),
        }
    }
}
```

`OctreeBuilder` gains one field:
```rust
pub struct OctreeBuilder {
    // ...
    temp_compression: TempCompression,
}
```
set from `PipelineConfig::temp_compression` in `OctreeBuilder::new`.

### 4.3 Counting points without seeking to file length

This is the gotcha that drives the rest of the design.

The current code computes the number of points in a temp file via
`file_len / RawPoint::BYTE_SIZE`. That works because the uncompressed
file is exactly `BYTE_SIZE × N` bytes. Under compression this
invariant breaks — `file_len` is now the **compressed** length.

Call sites that rely on this (searched for `RawPoint::BYTE_SIZE` and
`metadata`):

1. `read_node` (`src/octree.rs:~747`) — pre-sizes the `Vec` via
   `file_len / BYTE_SIZE`.
2. `read_chunk_file` (~1280) — same thing.
3. `bottom_up_on_disk` fallback in `merge_chunk_tops` (~1553) —
   estimates per-parent memory from summed child file sizes.
4. Hierarchy enumeration at the end of `build_node_map_chunked`
   (~1788) — computes final point counts per node from file length.
5. `grid_sample_streaming` (~975) — inspects child file length to
   decide whether to give that child its last point back.

**Options, in preference order:**

**Option A (recommended): length-prefixed frames.**
Prepend a 4-byte little-endian `u32 point_count` header to every
temp file, then the compressed (or uncompressed) payload. `read_node`
/ `read_chunk_file` read the header first, allocate the right-sized
`Vec`, then stream the decoder. For call sites 3-5, replace
`file_len / BYTE_SIZE` with a new helper `self.point_count(path)`
that reads just the 4-byte header. Cheap — one `pread` per file.

The 4-byte overhead is negligible (we write max ~100K temp files on a
42.8B-point run: 400 KB total).

**Option B: sidecar `.count` files.** Write a tiny `.count` file next
to each temp file. More files, more inodes, no measurable benefit.
Skip.

**Option C: scan the whole file on every read to learn the count.**
Forces a two-pass read (once to count, once to populate). Doubles
decompression cost. Skip.

Go with **Option A**. The header applies uniformly to compressed AND
uncompressed files, so we don't branch in the reader on
`TempCompression` — the format is always "4-byte count + payload",
and only the payload codec changes. This also removes the implicit
`file_len / BYTE_SIZE` coupling that's a footgun today.

### 4.4 Impact on `ChunkWriterCache::append`

`append` currently does `RawPoint::write_bulk(points, w)` directly into
a `BufWriter<File>` opened in append mode. Two changes needed:

1. **The writer type in the cache becomes `Box<dyn Write + Send>`
   (or an enum holding one of two concrete writer types).** Prefer
   an enum to avoid the dyn-dispatch allocation per write:

    ```rust
    enum TempWriter {
        Raw(BufWriter<File>),
        Lz4(lz4_flex::frame::FrameEncoder<BufWriter<File>>),
    }
    impl Write for TempWriter { /* delegate */ }
    ```

2. **The length-prefix header must be written once per file**, not on
    every `append` call. The cache currently opens in append mode and
    rewrites the same file across many `append` calls. For **compressed**
    shards this doesn't work at all — you cannot resume an LZ4 frame
    encoder across process-separated append calls. Each new append would
    start a new independent frame.

    Two sub-options:
    - **A1: multi-frame files.** LZ4 frame format supports concatenated
      frames in one file, and `FrameDecoder` reads them transparently.
      Every time we re-open a chunk's shard file for append we start a
      fresh encoder frame and finalize it before the next eviction. Each
      frame carries its own header; the **file-level** length prefix
      needs special handling (see A1-fix below).
    - **A2: no append.** Close and re-write the whole file on each
      "append" batch. Catastrophic — defeats the whole point of the
      cache, which exists to coalesce small writes across many source
      files.

    **A1 is the only viable option.** The wrinkle: our file-level
    `u32 point_count` header is written once when the file is first
    created, but we don't know the final count at that moment. Fix:
    **instead of a file-level count header, make the per-frame count the
    source of truth.** Each encoder frame prepends its own 4-byte count
    header *before* the LZ4 frame bytes. The reader loop becomes:
    ```
    while !eof:
        count = read_u32_le()
        decode_next_frame_into(buf, count * BYTE_SIZE)
        append to points vec
    ```
    Summing counts gives the total. The file-level overhead is
    `4 × num_appends_to_this_file` bytes — bounded by the LRU eviction
    frequency, so at most a few hundred bytes per chunk. Negligible.

    **For uncompressed files with append semantics, use the same
    per-frame scheme** (`u32 count` header followed by `count × 38` raw
    bytes). Symmetric codec dispatch, one file format.

**Canonical chunk files (post-`merge_chunk_shards`) inherit the same
multi-frame layout automatically**, since `merge_chunk_shards` uses
`std::io::copy` to concatenate shard bytes — each shard's frames just
land sequentially in the canonical file.

### 4.5 Per-node files (`write_node_to_temp`)

Per-node files are always full-rewrite, not append. Simpler:
- **Single frame per file.** Write one `u32 count` header, then one
  encoder-framed (or raw) payload, close.
- `read_node` reads the header, decodes one frame.

No multi-frame handling needed for this path. Keep the single-frame
helper function separate from the multi-frame chunked path helper to
keep each call site simple.

### 4.6 Progress reporting

Nothing changes. Progress events are per-stage, not per-byte.
Compression adds CPU time but does not affect the "done" counter.

## 5. Execution order

Land in three commits. Each keeps all tests passing.

### Step 5.1 — Introduce the length-prefix format for uncompressed files

No codec yet; just add the `u32 count` header to every temp file and
rewrite the five call sites (§4.3) to use a new
`OctreeBuilder::read_temp_file(path) -> Result<Vec<RawPoint>>` helper
and a `write_temp_file(path, points)` helper. Both go through the
length prefix. For `ChunkWriterCache::append`, each re-open (on LRU
eviction + subsequent write) starts a new framed segment with its own
count header.

After this step, the on-disk format is the new multi-frame format
(uncompressed) and all reads/writes route through the two helpers.
`file_len / BYTE_SIZE` is gone.

Run the full test suite. All 13 integration tests + 38 unit tests
must pass. The binary output must be byte-identical to the pre-change
baseline (the COPC file is unchanged — only the scratch format moved).

**Commit 1:** "Introduce length-prefixed temp file format"

### Step 5.2 — Add the `TempCompression` enum and LZ4 backend

Add the `lz4_flex` dependency. Introduce `TempCompression` in
`src/lib.rs`, `PipelineConfig::temp_compression`, and the
`--temp-compression` CLI flag. Default is `None`.

Plumb the flag into `OctreeBuilder::new`, store it on the struct, and
branch inside `read_temp_file` / `write_temp_file` /
`ChunkWriterCache::append` on the stored value to pick the encoder/
decoder. Implement `TempWriter` as an enum over the two concrete
writer types and derive `Write` via delegation.

Run the suite with the default. Then run the suite explicitly with
`--temp-compression=lz4`:

```sh
cargo test
# Add one new test: run_converter_with_args(input, output,
#   &["--temp-compression", "lz4"]) and assert the output is valid.
```

Verify the chunked multi-chunk regression test also passes with
`--temp-compression=lz4` — this is the one that forces many chunks
and catches the cross-chunk classification boundary.

**Commit 2:** "Add --temp-compression=lz4 for chunked-build scratch files"

### Step 5.3 — Docs and integration tests

- Add a test `temp_compression_lz4_preserves_all_points` modeled on
  `preserves_all_points` but with the flag set.
- Add a test that runs the existing `chunked_multi_chunk_preserves_all_points`
  scenario with `--temp-compression=lz4` to exercise multi-frame
  append under compression.
- Update `README.md`: mention the flag under a new "Temp file
  compression" subsection. Note that it trades CPU for disk and is
  most useful on space-constrained workers and network filesystems.
- Update `CLAUDE.md` if the new flag deserves a mention in the
  architecture section (it's a small configuration knob — probably
  not).

**Commit 3:** "Test and document --temp-compression"

## 6. Benchmarking

Before declaring the feature done, run on the 42.8B-point production
dataset with both `--temp-compression=none` and `--temp-compression=lz4`
and record:

| Metric | Baseline | LZ4 | Delta |
|---|---|---|---|
| Peak scratch disk | ? | ? | ? |
| Wall time | ? | ? | ? |
| Peak CPU | ? | ? | ? |
| Writer-phase memory | ? | ? | — (should be identical) |
| Output file bytes | ? | ? | — (must be identical) |
| Output hierarchy node count | ? | ? | — (must be identical) |

Fill in the results and append them to `docs/phase-3-temp-file-compression.md`
before merging.

If the compression ratio comes in below ~2× or the wall-time
regression exceeds 20% on fast NVMe, stop and re-evaluate. Either
zstd-1 will do better (→ add a second codec) or the RawPoint layout
isn't as compressible as expected (→ consider delta-coding x/y/z
before compression, or skip the feature entirely).

## 7. Risks & mitigations

**R1: LZ4 decompressor streaming mismatch.**
`lz4_flex::frame::FrameDecoder` expects a `Read` source and decodes
frame-by-frame. If we write multi-frame files and the decoder doesn't
transparently advance across frames, the reader loop has to detect
EOF vs end-of-frame and restart. **Mitigation:** prototype the
multi-frame round-trip in a unit test (`tests/` or a new
`src/octree.rs` unit test) before wiring it into the pipeline. If the
library misbehaves, switch to manually sequencing
`read_u32_le → FrameDecoder::new(Take(BufReader, frame_byte_len))`.

**R2: Cache eviction overhead.**
Under LZ4, every eviction now finalizes a frame (one small trailer
write) and every re-open starts a new frame (one small header write).
With `CHUNKED_OPEN_FILES_CAP = 512` and tens of thousands of evictions
over a run, that's tens of MB of overhead — negligible but noted.

**R3: Compression CPU steals from the rayon pool.**
The distribute phase is already CPU-bound on the LAZ parallel decoder.
Adding compression work to the same pool may slow decode throughput.
**Mitigation:** if benchmarks show contention, consider a dedicated
small thread pool for compression, or drop to LZ4 `block` mode
(faster, no framing overhead) on the hot path.

**R4: Mixed-format temp dirs between crashed runs.**
Step 5.1 changes the on-disk format. If a previous-run crash left
old-format files in the shared temp dir, the new code reads garbage.
**Mitigation:** `OctreeBuilder::new` already blows away any stale
`copc_<pid>` directory (see src/octree.rs ~690). This mitigation is
already in place — verify it still runs before the new reader logic.

**R5: File extension / magic number.**
We're not adding a magic number, just the length prefix. If a future
version changes the format again, old temp files from in-flight runs
could misparse. **Mitigation:** the `copc_<pid>` temp dir naming
already scopes files to a single process — cross-version collisions
don't happen in practice. A magic number is overkill here.

## 8. Open questions to resolve before starting

1. **LZ4 block mode vs frame mode?** Frame mode handles streaming and
   multi-frame cleanly; block mode is ~10% faster and simpler but
   requires knowing each block's decompressed length up front. Frame
   is the right pick unless benchmarking shows block makes a
   meaningful difference. **Default answer: frame.**
2. **Should we also add zstd as a second codec in this phase?**
   Only if LZ4 benchmarks fall short of the ~3× target. Otherwise
   defer — a follow-up phase can add it against real measurements.
   **Default answer: LZ4 only for phase 3.**
3. **Make LZ4 the default instead of opt-in?** No. Local-NVMe users
   are the common case and they don't want the CPU tax. Opt-in is
   correct.
4. **Feature flag the dependency (`--features compression`) so users
   who don't want it avoid the extra ~50 KB of compiled code?**
   `lz4_flex` is small. Not worth the feature-flag complexity.
   **Default answer: unconditional dep.**

## 9. Success criteria

- ✅ `cargo build`, `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test` all clean
- ✅ New integration test
  `temp_compression_lz4_preserves_all_points` passes
- ✅ Existing tests pass unchanged on the default path
- ✅ Smoke run on `tests/data/input.laz` produces a byte-identical
  output file with `--temp-compression=lz4` vs `--temp-compression=none`
- ✅ Benchmark numbers recorded in §6 show ≥2× scratch-disk reduction
  on the production dataset with ≤20% wall-time regression on fast NVMe
- ✅ README mentions the new flag
