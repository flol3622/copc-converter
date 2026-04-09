# copc_converter

[![Crates.io](https://img.shields.io/crates/v/copc_converter)](https://crates.io/crates/copc_converter)
[![docs.rs](https://docs.rs/copc_converter/badge.svg)](https://docs.rs/copc_converter)

A fast, memory-efficient converter that turns LAS/LAZ point cloud files into [COPC](https://copc.io/) (Cloud-Optimized Point Cloud) files.

## Features

- Produces spec-compliant COPC 1.0 files (LAS 1.4, point format 6, 7, or 8 — automatically chosen from input)
- Merges multiple input files into a single COPC output
- Out-of-core processing with a configurable memory budget — handles datasets larger than RAM
- Parallel reading, octree construction, and LAZ compression via rayon
- Preserves WKT CRS from input files
- Optional temporal index for GPS-time-based filtering ([spec](https://github.com/360-geo/copc/blob/master/copc-temporal/docs/temporal-index-spec.md))

## Installation

Requires Rust 1.85+.

### From crates.io

```sh
cargo install copc_converter
```

### From source

```sh
git clone https://github.com/360-geo/copc-converter.git
cd copc-converter
cargo install --path .
```

This installs the `copc_converter` binary to `~/.cargo/bin/`, which should be on your `PATH`.

### Pre-built binaries

Download pre-built binaries from the [GitHub releases](https://github.com/360-geo/copc-converter/releases) page. These are built for broad compatibility and run on any machine.

For best performance, prefer installing from source via `cargo install` — this automatically compiles with `target-cpu=native`, optimizing for your specific CPU's instruction set (AVX2, NEON, etc.).

## Usage

```sh
# Single file
copc_converter input.laz output.copc.laz

# Directory of LAZ/LAS files
copc_converter ./tiles/ merged.copc.laz
```

### Options

| Flag | Description | Default |
|---|---|---|
| `--memory-limit` | Max memory budget (`16G`, `4096M`, etc.) | auto-detected |
| `--threads` | Max parallel threads | all cores |
| `--temp-dir` | Directory for intermediate files | system temp |
| `--temporal-index` | Write a temporal index EVLR for time-based queries | off |
| `--temporal-stride` | Sampling stride for the temporal index (every n-th point) | `1000` |
| `--progress` | Progress output format: `bar`, `plain`, or `json` | `bar` |
| `--temp-compression` | Compress scratch temp files: `none` or `lz4` | `none` |

#### Temp file compression

Chunked-build scratch files hold `RawPoint` records (38 bytes each) and are
highly compressible. On a large run (tens of billions of points) the temp
directory can approach the full raw-point footprint, which becomes the
limiting resource on space-constrained workers.

`--temp-compression=lz4` wraps each temp-file write in a self-contained LZ4
frame. Expect roughly a 3-4× reduction in scratch-disk usage at a modest CPU
cost (LZ4 compresses at >1 GB/s per core). On fast local NVMe this trades CPU
for disk without a clear wall-time win; on network filesystems (EFS/NFS) it
typically also reduces wall time because the bottleneck shifts from I/O to
compute.

### Examples

```sh
copc_converter ./my_survey/ survey.copc.laz --memory-limit 8G

# With temporal index (useful for multi-pass mobile mapping data)
copc_converter ./my_survey/ survey.copc.laz --temporal-index
```

## Library usage

The crate exposes a typestate pipeline API that enforces correct step ordering at compile time:

```rust
use copc_converter::{Pipeline, PipelineConfig, collect_input_files};

let files = collect_input_files("./tiles/".into())?;
let config = PipelineConfig {
    memory_budget: 12_884_901_888,
    temp_dir: None,
    temporal_index: false,
    temporal_stride: 1000,
    progress: None, // or Some(Arc::new(your_observer))
    chunk_target_override: None,
};

Pipeline::scan(&files, config)?
    .validate()?
    .distribute()?
    .build()?
    .write("output.copc.laz")?;
```

## Tools

Optional analysis tools are available behind the `tools` feature:

```sh
cargo build --release --features tools
```

### inspect_copc

Inspect a COPC file's structure, or compare two files side-by-side. Works with local files and HTTP URLs.

```sh
# Inspect a single file
inspect_copc pointcloud.copc.laz

# Compare two files
inspect_copc pointcloud.copc.laz --compare other.copc.laz
```

Prints node counts, point distribution, compressed sizes, and compression ratios per octree level.

### inspect_temporal

Inspect the temporal index EVLR of a COPC file:

```sh
inspect_temporal <url>
```

Prints GPS time range, per-level temporal coverage, a time histogram showing node overlap across time windows, and sample density stats.

## How it works

1. **Scan** — reads headers from all input files in parallel to determine bounds, CRS, point format, and point count.
2. **Validate** — checks that all input files share the same CRS and point format, and selects the appropriate COPC output format (6, 7, or 8).
3. **Count** — first full pass over the input: populates an occupancy grid used by the chunk planner to carve the dataset into thousands of roughly equal-sized chunks via counting sort.
4. **Distribute** — second full pass over the input: streams every point into its chunk's scratch file on disk, bounded by the configured memory budget.
5. **Build** — each chunk's sub-octree is built independently in memory in parallel, then merged at coarse levels up to a single global root, thinning points at each level to produce multi-resolution LODs.
6. **Write** — encodes and compresses nodes in parallel into a single COPC file with a hierarchy EVLR for spatial indexing.

## Acknowledgments

The chunked octree build is based on the counting-sort approach described in:

> Markus Schütz, Stefan Ohrhallinger, and Michael Wimmer. "Fast Out-of-Core Octree Generation for Massive Point Clouds." *Computer Graphics Forum*, 2020. [doi:10.1111/cgf.14134](https://doi.org/10.1111/cgf.14134)

## License

MIT
