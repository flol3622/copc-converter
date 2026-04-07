/// Out-of-core octree builder.
///
/// Strategy
/// --------
/// 1. Pass 1 – scan all input files in parallel, collect bounding box + point count.
/// 2. Determine octree depth so that leaf nodes contain ≤ MAX_LEAF_POINTS on average.
/// 3. Pass 2 – read every point, assign it to the leaf voxel key, and
///    accumulate into per-key temporary files on disk.
///    Point classification (key + coordinate conversion) is parallelized via rayon.
///    Memory-aware: fast path (full file) or batched path depending on budget.
/// 4. Normalize leaves: any leaf with > MAX_LEAF_POINTS is split into children on disk.
/// 5. Build the tree bottom-up in parallel: each parent node gets a thinned sample
///    of its children's points written back to disk.
/// 6. Produce the list of (VoxelKey, point_count) for the writer, which reads from disk.
///
/// Memory usage is bounded by the configurable memory budget.
use crate::PipelineConfig;
use crate::copc_types::VoxelKey;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Chunked-build constants and helper types
// ---------------------------------------------------------------------------

/// Maximum number of leaf chunk files kept open simultaneously across all
/// distribute workers in the chunked path. Divided evenly between workers.
///
/// Each entry holds a `BufWriter<File>` (~8 KiB internal buffer) plus an OS
/// file descriptor. Capping prevents FD exhaustion on hosts with low ulimits
/// (default 256 on macOS, 1024 on most Linux containers).
const CHUNKED_OPEN_FILES_CAP: usize = 512;

/// Minimum per-worker LRU capacity for the chunk writer cache.
const MIN_PER_WORKER_CHUNK_FILES: usize = 32;

/// Bounded LRU cache of append-mode `BufWriter`s for chunk temp files.
///
/// Keys are chunk indices (`u32`). On eviction the writer is flushed and
/// dropped, releasing both its buffer and its file descriptor. Subsequent
/// writes to the same chunk index reopen the file in append mode.
///
/// This is the chunk-keyed analog of the per-leaf path's writer cache: the
/// design is identical, only the key type changes.
struct ChunkWriterCache {
    writers: HashMap<u32, BufWriter<File>>,
    /// Insertion / access order. Front = least recently used.
    order: VecDeque<u32>,
    capacity: usize,
}

impl ChunkWriterCache {
    fn new(capacity: usize) -> Self {
        ChunkWriterCache {
            writers: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    /// Append the given points to the temp file for `chunk_idx`, opening (and
    /// possibly evicting another entry) as needed.
    fn append(&mut self, chunk_idx: u32, shard_dir: &Path, points: &[RawPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        if self.writers.contains_key(&chunk_idx) {
            // Move to back (most-recently-used).
            if let Some(pos) = self.order.iter().position(|k| *k == chunk_idx) {
                self.order.remove(pos);
            }
            self.order.push_back(chunk_idx);
        } else {
            // Evict if at capacity.
            while self.writers.len() >= self.capacity {
                if let Some(victim) = self.order.pop_front() {
                    if let Some(mut w) = self.writers.remove(&victim) {
                        w.flush().context("flush evicted chunk writer")?;
                    }
                } else {
                    break;
                }
            }
            let path = chunk_shard_path(shard_dir, chunk_idx);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("Cannot open chunk file {:?}", path))?;
            self.writers.insert(chunk_idx, BufWriter::new(f));
            self.order.push_back(chunk_idx);
        }

        let w = self.writers.get_mut(&chunk_idx).expect("just inserted");
        RawPoint::write_bulk(points, w)?;
        Ok(())
    }

    /// Flush and drop every cached writer.
    fn flush_all(&mut self) -> Result<()> {
        for (_, mut w) in self.writers.drain() {
            w.flush().context("flush chunk writer")?;
        }
        self.order.clear();
        Ok(())
    }
}

/// Path for a chunk's per-worker shard file.
fn chunk_shard_path(shard_dir: &Path, chunk_idx: u32) -> PathBuf {
    shard_dir.join(format!("chunk_{chunk_idx}"))
}

/// Path for a chunk's canonical (merged) file under `tmp_dir/chunks/`.
fn chunk_canonical_path(chunks_dir: &Path, chunk_idx: u32) -> PathBuf {
    chunks_dir.join(format!("chunk_{chunk_idx}"))
}

/// Same heuristic as the per-leaf parallel distribute path: ~2/3 of cores,
/// leaving headroom for the LAZ parallel decoder. Capped by input file count.
fn chunked_distribute_worker_count(input_file_count: usize) -> usize {
    let cores = rayon::current_num_threads();
    let target = ((cores * 2) / 3).max(2);
    target.min(input_file_count).max(1)
}

/// Build the cell → chunk_index lookup table from a chunk plan.
///
/// The grid has `grid_size³` fine cells; for each chunk we fill in every
/// fine cell it covers. A chunk at level L covers a `2^(grid_depth - L)`
/// cube per axis, since the grid corresponds to octree cells at `grid_depth`.
///
/// Returns a flat `Vec<u32>` indexed by `gx + gy*G + gz*G²`. Each entry
/// holds the chunk index this cell belongs to. Cells not covered by any
/// chunk (which should not happen for a well-formed plan) get `u32::MAX`
/// as a sentinel; we treat any point landing in such a cell as a bug.
fn build_chunk_lut(plan: &crate::chunking::ChunkPlan) -> Vec<u32> {
    let g = plan.grid_size as usize;
    let n_cells = g * g * g;
    let mut lut = vec![u32::MAX; n_cells];

    for (chunk_idx, chunk) in plan.chunks.iter().enumerate() {
        // The chunk lives at chunk.level. The fine grid is at plan.grid_depth.
        // Each chunk cell covers a stride of 2^(grid_depth - chunk.level)
        // along each axis in the fine grid.
        let level_diff = plan.grid_depth as i32 - chunk.level as i32;
        debug_assert!(
            level_diff >= 0,
            "chunk level {} above grid depth {}",
            chunk.level,
            plan.grid_depth
        );
        let stride: usize = 1usize << level_diff as u32;

        // Top-left corner of this chunk in fine-grid coordinates, clamped
        // defensively against malformed chunks. (Should never trigger for
        // a plan produced by `merge_sparse_cells`.)
        let base_x = (chunk.gx as usize * stride).min(g);
        let base_y = (chunk.gy as usize * stride).min(g);
        let base_z = (chunk.gz as usize * stride).min(g);
        let end_x = (base_x + stride).min(g);
        let end_y = (base_y + stride).min(g);
        let end_z = (base_z + stride).min(g);

        for z in base_z..end_z {
            for y in base_y..end_y {
                let row_start = base_x + y * g + z * g * g;
                let row_end = end_x + y * g + z * g * g;
                for cell in &mut lut[row_start..row_end] {
                    *cell = chunk_idx as u32;
                }
            }
        }
    }

    lut
}

/// Concatenate per-worker shard files into the canonical
/// `tmp_dir/chunks/chunk_N` location. Runs in parallel over chunk indices.
///
/// After this returns successfully, the `shards/` subdirectory is removed
/// and only the canonical chunk files remain under `chunks/`.
fn merge_chunk_shards(shards_root: &Path, chunks_root: &Path, n_chunks: u32) -> Result<()> {
    let num_workers = match std::fs::read_dir(shards_root) {
        Ok(it) => it
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .count(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(e.into()),
    };
    debug!(
        "Merging {} chunks across {} worker shards",
        n_chunks, num_workers
    );

    (0..n_chunks)
        .into_par_iter()
        .try_for_each(|chunk_idx| -> Result<()> {
            let canonical = chunk_canonical_path(chunks_root, chunk_idx);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&canonical)
                .with_context(|| format!("opening canonical chunk file {:?}", canonical))?;
            let mut out = BufWriter::new(f);
            for w in 0..num_workers {
                let shard_path = shards_root
                    .join(w.to_string())
                    .join(format!("chunk_{chunk_idx}"));
                match File::open(&shard_path) {
                    Ok(f) => {
                        let mut reader = BufReader::new(f);
                        std::io::copy(&mut reader, &mut out).with_context(|| {
                            format!("copying shard {:?} into {:?}", shard_path, canonical)
                        })?;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e.into()),
                }
            }
            out.flush().context("flush merged chunk file")?;
            Ok(())
        })?;

    std::fs::remove_dir_all(shards_root)
        .with_context(|| format!("removing shards root {:?}", shards_root))?;
    Ok(())
}

/// Task fed into the parallel grid-sampling step: parent key, child keys, and
/// indexed points (child-index, point).
type SampleTask = (VoxelKey, Vec<VoxelKey>, Vec<(usize, RawPoint)>);

/// Result coming back from parallel grid-sampling: parent key, child keys,
/// promoted points, and per-child remaining points.
type SampleResult = (VoxelKey, Vec<VoxelKey>, Vec<RawPoint>, Vec<Vec<RawPoint>>);

// ---------------------------------------------------------------------------
// Morton code helper (used for spatially coherent traversal order)
// ---------------------------------------------------------------------------

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    #[inline]
    fn spread(mut v: u64) -> u64 {
        v &= 0x1F_FFFF;
        v = (v | (v << 32)) & 0x1F00000000FFFF;
        v = (v | (v << 16)) & 0x1F0000FF0000FF;
        v = (v | (v << 8)) & 0x100F00F00F00F00F;
        v = (v | (v << 4)) & 0x10C30C30C30C30C3;
        v = (v | (v << 2)) & 0x1249249249249249;
        v
    }
    spread(x as u64) | (spread(y as u64) << 1) | (spread(z as u64) << 2)
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum points per leaf voxel before we subdivide further.
const MAX_LEAF_POINTS: u64 = 100_000;

/// Grid cells per axis for LOD thinning. Matches untwine's CellCount = 128.
/// Higher values keep more points at coarse LOD levels (better progressive rendering).
const GRID_CELLS_PER_AXIS: i64 = 128;

// ---------------------------------------------------------------------------
// Raw point storage
// ---------------------------------------------------------------------------

/// A raw point stored as scaled integer coordinates plus attributes.
/// Scaled ints allow exact LAS round-trip without floating-point loss.
#[derive(Debug, Clone)]
pub struct RawPoint {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification: u8,
    pub scan_angle: i16,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub nir: u16,
}

impl RawPoint {
    pub const BYTE_SIZE: usize = 4 + 4 + 4 + 2 + 1 + 1 + 1 + 2 + 1 + 2 + 8 + 2 + 2 + 2 + 2; // 38

    #[allow(unused)]
    pub fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        let mut buf = [0u8; Self::BYTE_SIZE];
        {
            use std::io::Cursor;
            let mut c = Cursor::new(&mut buf[..]);
            c.write_i32::<LittleEndian>(self.x)?;
            c.write_i32::<LittleEndian>(self.y)?;
            c.write_i32::<LittleEndian>(self.z)?;
            c.write_u16::<LittleEndian>(self.intensity)?;
            c.write_u8(self.return_number)?;
            c.write_u8(self.number_of_returns)?;
            c.write_u8(self.classification)?;
            c.write_i16::<LittleEndian>(self.scan_angle)?;
            c.write_u8(self.user_data)?;
            c.write_u16::<LittleEndian>(self.point_source_id)?;
            c.write_f64::<LittleEndian>(self.gps_time)?;
            c.write_u16::<LittleEndian>(self.red)?;
            c.write_u16::<LittleEndian>(self.green)?;
            c.write_u16::<LittleEndian>(self.blue)?;
            c.write_u16::<LittleEndian>(self.nir)?;
        }
        w.write_all(&buf)?;
        Ok(())
    }

    pub fn read<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut buf = [0u8; Self::BYTE_SIZE];
        r.read_exact(&mut buf)?;
        let mut c = std::io::Cursor::new(&buf[..]);
        Ok(RawPoint {
            x: c.read_i32::<LittleEndian>()?,
            y: c.read_i32::<LittleEndian>()?,
            z: c.read_i32::<LittleEndian>()?,
            intensity: c.read_u16::<LittleEndian>()?,
            return_number: c.read_u8()?,
            number_of_returns: c.read_u8()?,
            classification: c.read_u8()?,
            scan_angle: c.read_i16::<LittleEndian>()?,
            user_data: c.read_u8()?,
            point_source_id: c.read_u16::<LittleEndian>()?,
            gps_time: c.read_f64::<LittleEndian>()?,
            red: c.read_u16::<LittleEndian>()?,
            green: c.read_u16::<LittleEndian>()?,
            blue: c.read_u16::<LittleEndian>()?,
            nir: c.read_u16::<LittleEndian>()?,
        })
    }

    /// Write multiple points to a writer in a single bulk operation.
    pub fn write_bulk<W: std::io::Write>(points: &[RawPoint], w: &mut W) -> Result<()> {
        let mut buf = vec![0u8; Self::BYTE_SIZE * points.len()];
        let mut c = std::io::Cursor::new(&mut buf[..]);
        for p in points {
            c.write_i32::<LittleEndian>(p.x)?;
            c.write_i32::<LittleEndian>(p.y)?;
            c.write_i32::<LittleEndian>(p.z)?;
            c.write_u16::<LittleEndian>(p.intensity)?;
            c.write_u8(p.return_number)?;
            c.write_u8(p.number_of_returns)?;
            c.write_u8(p.classification)?;
            c.write_i16::<LittleEndian>(p.scan_angle)?;
            c.write_u8(p.user_data)?;
            c.write_u16::<LittleEndian>(p.point_source_id)?;
            c.write_f64::<LittleEndian>(p.gps_time)?;
            c.write_u16::<LittleEndian>(p.red)?;
            c.write_u16::<LittleEndian>(p.green)?;
            c.write_u16::<LittleEndian>(p.blue)?;
            c.write_u16::<LittleEndian>(p.nir)?;
        }
        w.write_all(&buf)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Bounds {
    pub fn empty() -> Self {
        Bounds {
            min_x: f64::MAX,
            min_y: f64::MAX,
            min_z: f64::MAX,
            max_x: f64::MIN,
            max_y: f64::MIN,
            max_z: f64::MIN,
        }
    }

    pub fn expand_with(&mut self, x: f64, y: f64, z: f64) {
        if x < self.min_x {
            self.min_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if z < self.min_z {
            self.min_z = z;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y > self.max_y {
            self.max_y = y;
        }
        if z > self.max_z {
            self.max_z = z;
        }
    }

    pub fn merge(&mut self, other: &Bounds) {
        if other.min_x < self.min_x {
            self.min_x = other.min_x;
        }
        if other.min_y < self.min_y {
            self.min_y = other.min_y;
        }
        if other.min_z < self.min_z {
            self.min_z = other.min_z;
        }
        if other.max_x > self.max_x {
            self.max_x = other.max_x;
        }
        if other.max_y > self.max_y {
            self.max_y = other.max_y;
        }
        if other.max_z > self.max_z {
            self.max_z = other.max_z;
        }
    }

    /// Cube that contains this AABB.
    pub fn to_cube(&self) -> (f64, f64, f64, f64) {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        let cz = (self.min_z + self.max_z) / 2.0;
        let half = ((self.max_x - self.min_x)
            .max(self.max_y - self.min_y)
            .max(self.max_z - self.min_z))
            / 2.0
            * 1.0001; // tiny epsilon
        (cx, cy, cz, half)
    }
}

// ---------------------------------------------------------------------------
// VoxelKey assignment
// ---------------------------------------------------------------------------

/// Assign a point to the voxel at the given tree depth.
#[allow(clippy::too_many_arguments)]
pub fn point_to_key(
    x: f64,
    y: f64,
    z: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    halfsize: f64,
    depth: u32,
) -> VoxelKey {
    let mut vx = 0i32;
    let mut vy = 0i32;
    let mut vz = 0i32;
    let mut half = halfsize;
    let mut ox = cx;
    let mut oy = cy;
    let mut oz = cz;

    for _ in 0..depth {
        half /= 2.0;
        let bx = if x >= ox {
            vx = vx * 2 + 1;
            ox + half
        } else {
            vx *= 2;
            ox - half
        };
        let by = if y >= oy {
            vy = vy * 2 + 1;
            oy + half
        } else {
            vy *= 2;
            oy - half
        };
        let bz = if z >= oz {
            vz = vz * 2 + 1;
            oz + half
        } else {
            vz *= 2;
            oz - half
        };
        ox = bx;
        oy = by;
        oz = bz;
    }

    VoxelKey {
        level: depth as i32,
        x: vx,
        y: vy,
        z: vz,
    }
}

// ---------------------------------------------------------------------------
// OctreeBuilder
// ---------------------------------------------------------------------------

/// Map an input LAS point format ID (0–10) to a COPC-compatible output format (6, 7, or 8).
pub fn input_to_copc_format(id: u8) -> u8 {
    match id {
        2 | 3 | 5 | 7 => 7,
        8 | 10 => 8,
        _ => 6, // 0, 1, 4, 6, 9
    }
}

/// Per-file results from the scan phase, used by validation.
pub struct ScanResult {
    pub bounds: Bounds,
    pub point_count: u64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    pub wkt_crs: Option<Vec<u8>>,
    pub point_format_id: u8,
}

/// Builds a COPC octree from scanned input files.
pub struct OctreeBuilder {
    /// Spatial bounds of all input points.
    pub bounds: Bounds,
    /// Total number of points across all input files.
    pub total_points: u64,
    /// Octree root center X.
    pub cx: f64,
    /// Octree root center Y.
    pub cy: f64,
    /// Octree root center Z.
    pub cz: f64,
    /// Half-size of the root voxel.
    pub halfsize: f64,
    /// Initial octree depth before normalization.
    pub depth: u32,
    /// X scale factor from the first input file.
    pub scale_x: f64,
    /// Y scale factor.
    pub scale_y: f64,
    /// Z scale factor.
    pub scale_z: f64,
    /// X offset.
    pub offset_x: f64,
    /// Y offset.
    pub offset_y: f64,
    /// Z offset.
    pub offset_z: f64,
    /// Temp directory where node files are written.
    pub tmp_dir: PathBuf,
    /// WKT CRS payload from the first input file (if present).
    pub wkt_crs: Option<Vec<u8>>,
    /// COPC output point format (6, 7, or 8), derived from input files.
    pub point_format: u8,
    /// Chunk plan computed by `distribute_chunked` and consumed by
    /// `build_node_map_chunked`. `None` for the per-leaf path.
    #[allow(dead_code)] // populated and read in incremental Phase 2b commits
    pub(crate) chunked_plan: Option<crate::chunking::ChunkPlan>,
}

impl OctreeBuilder {
    /// Pass 1: scan all files in parallel to get bounds and total point count.
    pub fn scan(input_files: &[PathBuf], config: &PipelineConfig) -> Result<Vec<ScanResult>> {
        let done = std::sync::atomic::AtomicU64::new(0);
        let results: Result<Vec<ScanResult>> = input_files
            .par_iter()
            .map(|path| -> Result<ScanResult> {
                debug!("Scanning {:?}", path);
                let reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                let hdr = reader.header();
                let b = hdr.bounds();
                let mut bounds = Bounds::empty();
                bounds.expand_with(b.min.x, b.min.y, b.min.z);
                bounds.expand_with(b.max.x, b.max.y, b.max.z);
                let t = hdr.transforms();
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                config.report(crate::ProgressEvent::StageProgress { done: n });
                Ok(ScanResult {
                    bounds,
                    point_count: hdr.number_of_points(),
                    scale_x: t.x.scale,
                    scale_y: t.y.scale,
                    scale_z: t.z.scale,
                    offset_x: t.x.offset,
                    offset_y: t.y.offset,
                    offset_z: t.z.offset,
                    wkt_crs: hdr
                        .all_vlrs()
                        .find(|v| v.is_wkt_crs())
                        .map(|v| v.data.clone()),
                    point_format_id: hdr.point_format().to_u8().unwrap_or(0),
                })
            })
            .collect();
        results
    }

    /// Build an OctreeBuilder from scan results and validated inputs.
    pub fn from_scan(
        scan_results: &[ScanResult],
        validated: &crate::validate::ValidatedInputs,
        config: &PipelineConfig,
    ) -> Result<Self> {
        let mut bounds = Bounds::empty();
        let mut total_points = 0u64;
        for r in scan_results {
            bounds.merge(&r.bounds);
            total_points += r.point_count;
        }

        let first = &scan_results[0];
        let (scale_x, scale_y, scale_z) = (first.scale_x, first.scale_y, first.scale_z);
        let (offset_x, offset_y, offset_z) = (first.offset_x, first.offset_y, first.offset_z);

        let (cx, cy, cz, halfsize) = bounds.to_cube();

        // Choose depth so that leaf voxels hold ≤ MAX_LEAF_POINTS on average.
        let depth = {
            let mut d = 0u32;
            while (total_points as f64) / (8u64.pow(d) as f64) > MAX_LEAF_POINTS as f64 {
                d += 1;
                if d > 16 {
                    break;
                }
            }
            d.max(1)
        };
        debug!("Octree depth = {depth}, total points = {total_points}");

        let sys_tmp = std::env::temp_dir();
        let base_tmp = config.temp_dir.as_deref().unwrap_or(&sys_tmp);
        let tmp_dir = base_tmp.join(format!("copc_{}", std::process::id()));
        // Remove any leftover temp dir from a previous run (e.g. crashed pod
        // that never ran Drop cleanup). Without this, append-mode writes in
        // distribute/normalize would double-count points from the old run.
        if tmp_dir.exists() {
            info!("Removing stale temp dir {:?}", tmp_dir);
            std::fs::remove_dir_all(&tmp_dir)?;
        }
        std::fs::create_dir_all(&tmp_dir)?;

        Ok(OctreeBuilder {
            bounds,
            total_points,
            cx,
            cy,
            cz,
            halfsize,
            depth,
            scale_x,
            scale_y,
            scale_z,
            offset_x,
            offset_y,
            offset_z,
            tmp_dir,
            wkt_crs: validated.wkt_crs.clone(),
            point_format: validated.point_format,
            chunked_plan: None,
        })
    }

    /// Path for a node's temp file.
    fn node_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }

    /// Convert a `las::Point` to a `RawPoint` using the builder's scale/offset.
    fn convert_point(&self, p: &las::Point) -> RawPoint {
        let ix = ((p.x - self.offset_x) / self.scale_x).round() as i32;
        let iy = ((p.y - self.offset_y) / self.scale_y).round() as i32;
        let iz = ((p.z - self.offset_z) / self.scale_z).round() as i32;
        RawPoint {
            x: ix,
            y: iy,
            z: iz,
            intensity: p.intensity,
            return_number: p.return_number,
            number_of_returns: p.number_of_returns,
            classification: p.classification.into(),
            scan_angle: (p.scan_angle / 0.006).round() as i16,
            user_data: p.user_data,
            point_source_id: p.point_source_id,
            gps_time: p.gps_time.unwrap_or(0.0),
            red: p.color.as_ref().map(|c| c.red).unwrap_or(0),
            green: p.color.as_ref().map(|c| c.green).unwrap_or(0),
            blue: p.color.as_ref().map(|c| c.blue).unwrap_or(0),
            nir: p.nir.unwrap_or(0),
        }
    }

    /// Parallel key assignment + coordinate conversion for a batch of points.
    ///
    /// Key assignment uses the *reconstructed* world coordinates (integer → world)
    /// rather than the original floating-point coordinates.  This guarantees that
    /// what the validator computes from the stored integers always falls inside the
    /// assigned voxel, even when input files use different scales/offsets.
    fn classify_points_parallel(&self, points: &[las::Point]) -> Vec<(VoxelKey, RawPoint)> {
        points
            .par_iter()
            .map(|p| {
                let raw = self.convert_point(p);
                let rx = raw.x as f64 * self.scale_x + self.offset_x;
                let ry = raw.y as f64 * self.scale_y + self.offset_y;
                let rz = raw.z as f64 * self.scale_z + self.offset_z;
                let key = point_to_key(
                    rx,
                    ry,
                    rz,
                    self.cx,
                    self.cy,
                    self.cz,
                    self.halfsize,
                    self.depth,
                );
                (key, raw)
            })
            .collect()
    }

    /// Merge classified points into per-key buffers and flush periodically.
    fn merge_into_buffers(
        classified: Vec<(VoxelKey, RawPoint)>,
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
        point_idx: &mut u64,
        flush_every: usize,
    ) -> Result<()> {
        for (key, raw) in classified {
            buffers.entry(key).or_default().push(raw);
            *point_idx += 1;
            if (*point_idx).is_multiple_of(flush_every as u64) {
                Self::flush_buffers(buffers, writers, tmp_dir)?;
            }
        }
        Ok(())
    }

    /// Pass 2: dispatch to the configured build strategy.
    ///
    /// Per-leaf assigns each point to a leaf voxel temp file and runs a
    /// multi-pass disk-backed bottom-up build later. Chunked partitions
    /// points into medium chunks via counting sort and runs a per-chunk
    /// in-memory build later. See `BuildStrategy`.
    pub fn distribute(&mut self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        match config.build_strategy {
            crate::BuildStrategy::PerLeaf => self.distribute_per_leaf(input_files, config),
            crate::BuildStrategy::Chunked => self.distribute_chunked(input_files, config),
        }
    }

    /// Per-leaf distribute: assign all points to leaf temp files.
    ///
    /// Uses `read_all_points_into` (fast parallel decompression) when the file
    /// fits within half the memory budget; otherwise falls back to batched reads.
    /// Key assignment and coordinate conversion are always parallelized via rayon.
    fn distribute_per_leaf(&self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        let flush_every =
            ((config.memory_budget / 4) as usize / RawPoint::BYTE_SIZE).clamp(10_000, 500_000);
        debug!(
            "Distribute: budget={} MB, flush_every={} points",
            config.memory_budget / 1_048_576,
            flush_every
        );

        let mut buffers: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        let mut writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();
        let mut point_idx = 0u64;

        let half_budget = config.memory_budget / 2;

        for path in input_files {
            let mut reader =
                las::Reader::from_path(path).with_context(|| format!("Cannot open {:?}", path))?;

            let file_point_count = reader.header().number_of_points();
            // Estimated memory per las::Point (~120 bytes)
            let estimated_mem = file_point_count * 120;

            debug!(
                "Distribute file {:?}: {} points, est {} MB, half_budget={} MB → {}",
                path.file_name().unwrap_or_default(),
                file_point_count,
                estimated_mem / 1_048_576,
                half_budget / 1_048_576,
                if estimated_mem <= half_budget {
                    "fast path"
                } else {
                    "batched path"
                }
            );

            if estimated_mem <= half_budget {
                // Fast path: load entire file with parallel decompression
                let mut points: Vec<las::Point> = Vec::new();
                reader.read_all_points_into(&mut points)?;
                let classified = self.classify_points_parallel(&points);
                drop(points);
                Self::merge_into_buffers(
                    classified,
                    &mut buffers,
                    &mut writers,
                    &self.tmp_dir,
                    &mut point_idx,
                    flush_every,
                )?;
                config.report(crate::ProgressEvent::StageProgress { done: point_idx });
            } else {
                // Batched path: read in chunks to stay within budget
                let batch_size = (half_budget / 120).max(10_000) as usize;
                debug!(
                    "File too large (~{} MB), using batched reads of {} points",
                    estimated_mem / (1024 * 1024),
                    batch_size
                );
                let mut points: Vec<las::Point> = Vec::new();
                loop {
                    points.clear();
                    let n = reader.read_points_into(batch_size as u64, &mut points)?;
                    if n == 0 {
                        break;
                    }
                    let classified = self.classify_points_parallel(&points);
                    Self::merge_into_buffers(
                        classified,
                        &mut buffers,
                        &mut writers,
                        &self.tmp_dir,
                        &mut point_idx,
                        flush_every,
                    )?;
                    config.report(crate::ProgressEvent::StageProgress { done: point_idx });
                }
            }
        }

        Self::flush_buffers(&mut buffers, &mut writers, &self.tmp_dir)?;
        // Explicitly flush all BufWriters so no data is lost on drop.
        for (_, w) in writers.iter_mut() {
            std::io::Write::flush(w).context("flush distribute writer")?;
        }
        Ok(())
    }

    fn flush_buffers(
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
    ) -> Result<()> {
        for (key, pts) in buffers.iter_mut() {
            if pts.is_empty() {
                continue;
            }
            let w = writers.entry(*key).or_insert_with(|| {
                let path = tmp_dir.join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z));
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .expect("Cannot open leaf file");
                BufWriter::new(f)
            });
            RawPoint::write_bulk(pts, w)?;
            pts.clear();
        }
        Ok(())
    }

    /// Read all raw points for a given node key from disk.
    pub fn read_node(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.node_path(key);
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        let file_len = f.metadata()?.len();
        let count = file_len as usize / RawPoint::BYTE_SIZE;
        let mut r = BufReader::new(f);
        let mut pts = Vec::with_capacity(count);
        for _ in 0..count {
            pts.push(RawPoint::read(&mut r)?);
        }
        Ok(pts)
    }

    /// Write points to a temp file for the given node key (overwrites if exists).
    pub fn write_node_to_temp(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        use std::io::Write;
        let path = self.node_path(key);
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        RawPoint::write_bulk(points, &mut w)?;
        w.flush().context("flush node temp file")?;
        Ok(())
    }

    /// Enumerate all node keys that have a non-empty temp file.
    fn all_node_keys(&self) -> Result<Vec<VoxelKey>> {
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(&self.tmp_dir)? {
            let entry = entry?;
            let name = entry.file_name().into_string().unwrap_or_default();
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() == 4
                && let (Ok(l), Ok(x), Ok(y), Ok(z)) = (
                    parts[0].parse::<i32>(),
                    parts[1].parse::<i32>(),
                    parts[2].parse::<i32>(),
                    parts[3].parse::<i32>(),
                )
            {
                keys.push(VoxelKey { level: l, x, y, z });
            }
        }
        Ok(keys)
    }

    /// Phase 0 of build_node_map: split any leaf file that exceeds MAX_LEAF_POINTS.
    ///
    /// All nodes being split in one round occupy disjoint voxels, so their child
    /// files never conflict — the entire round is processed in parallel via rayon.
    /// Rounds repeat until no oversized nodes remain.
    fn normalize_leaves(&self, memory_budget: u64, config: &crate::PipelineConfig) -> Result<()> {
        let oversized = |k: &VoxelKey| -> bool {
            self.node_path(k)
                .metadata()
                .is_ok_and(|m| m.len() / RawPoint::BYTE_SIZE as u64 > MAX_LEAF_POINTS)
        };

        let mut to_split: Vec<VoxelKey> = self
            .all_node_keys()?
            .into_iter()
            .filter(|k| oversized(k))
            .collect();

        if to_split.is_empty() {
            return Ok(());
        }

        // Report normalize as its own stage. We don't know the total upfront
        // (each round can produce new oversized children), so use 0 = spinner.
        config.report(crate::ProgressEvent::StageStart {
            name: "Normalizing",
            total: 0,
        });

        // Points per chunk when streaming a split. Each rayon thread holds one
        // chunk (~40 bytes/point) plus a HashMap of classified output (~40 bytes/
        // point + hash overhead ≈ 1.3x), so peak per thread ≈ 2.5x chunk size.
        // Divide the budget by (threads × 2.5) to size chunks so all threads
        // can be active simultaneously without exceeding the budget.
        let num_threads = rayon::current_num_threads() as u64;
        let per_thread_budget = memory_budget / num_threads.max(1);
        // 2.5x overhead → divide by 3 for safety margin
        let chunk_points = ((per_thread_budget / 3) as usize / RawPoint::BYTE_SIZE).max(100_000);

        let mut round = 0u32;
        while !to_split.is_empty() {
            round += 1;
            let split_count = to_split.len();
            let total_split_bytes: u64 = to_split
                .iter()
                .map(|k| self.node_path(k).metadata().map_or(0, |m| m.len()))
                .sum();
            info!(
                "normalize round {round}: splitting {} nodes ({} MB on disk)",
                split_count,
                total_split_bytes / 1_048_576,
            );

            // All nodes in a round occupy disjoint voxels → their children
            // are also disjoint → safe to split in parallel. Each node is
            // streamed in chunks so memory stays bounded regardless of node size.
            let new_children: Vec<VoxelKey> = to_split
                .par_iter()
                .map(|key| -> Result<Vec<VoxelKey>> {
                    let path = self.node_path(key);
                    let f = File::open(&path)?;
                    let file_len = f.metadata()?.len();
                    let total_count = file_len as usize / RawPoint::BYTE_SIZE;
                    let mut reader = BufReader::new(f);

                    // Open child writers in append mode so successive chunks
                    // accumulate into the same files.
                    let child_level = key.level + 1;
                    let mut child_writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();
                    let mut child_keys_set: HashSet<VoxelKey> = HashSet::new();

                    let mut read_so_far = 0usize;
                    while read_so_far < total_count {
                        let n = chunk_points.min(total_count - read_so_far);
                        let mut chunk = Vec::with_capacity(n);
                        for _ in 0..n {
                            chunk.push(RawPoint::read(&mut reader)?);
                        }
                        read_so_far += n;

                        // Classify and append to child files.
                        let mut per_child: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
                        for p in chunk {
                            let wx = p.x as f64 * self.scale_x + self.offset_x;
                            let wy = p.y as f64 * self.scale_y + self.offset_y;
                            let wz = p.z as f64 * self.scale_z + self.offset_z;
                            let ck = point_to_key(
                                wx,
                                wy,
                                wz,
                                self.cx,
                                self.cy,
                                self.cz,
                                self.halfsize,
                                child_level as u32,
                            );
                            per_child.entry(ck).or_default().push(p);
                        }

                        for (ck, pts) in per_child {
                            child_keys_set.insert(ck);
                            let w = child_writers.entry(ck).or_insert_with(|| {
                                let child_path = self.node_path(&ck);
                                let f = OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(&child_path)
                                    .expect("Cannot open child temp file");
                                BufWriter::new(f)
                            });
                            RawPoint::write_bulk(&pts, w)?;
                        }
                    }

                    // Flush all child writers.
                    for (_, mut w) in child_writers {
                        std::io::Write::flush(&mut w).context("flush child temp file")?;
                    }

                    // Remove the parent file now that all data has been written to children.
                    std::fs::remove_file(&path)?;

                    Ok(child_keys_set.into_iter().collect())
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            to_split = new_children.into_iter().filter(|k| oversized(k)).collect();
        }
        config.report(crate::ProgressEvent::StageDone);
        Ok(())
    }

    /// Build the full node hierarchy. Dispatches to the configured strategy.
    ///
    /// Returns (VoxelKey, point_count) for every node in the hierarchy.
    pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
        match config.build_strategy {
            crate::BuildStrategy::PerLeaf => self.build_node_map_per_leaf(config),
            crate::BuildStrategy::Chunked => self.build_node_map_chunked(config),
        }
    }

    /// Per-leaf build: normalize leaves, then bottom-up ancestor building.
    ///
    /// Phase 0: normalize leaves in parallel.
    /// Phase 1: determine actual max depth.
    /// Phase 2: bottom-up ancestor building.
    ///   Fast path  – if all leaf data fits in the memory budget, everything is
    ///                processed in RAM (one disk read + one disk write total).
    ///   Slow path  – level-by-level with disk I/O, but without redundant directory
    ///                scans (keys are tracked in a per-level map).
    fn build_node_map_per_leaf(
        &self,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Phase 0
        self.normalize_leaves(config.memory_budget, config)?;

        // Phase 1
        let leaf_keys = self.all_node_keys()?;
        let actual_max_depth = leaf_keys.iter().map(|k| k.level as u32).max().unwrap_or(0);
        info!(
            "Leaf nodes after normalization: {}, max depth: {actual_max_depth}",
            leaf_keys.len()
        );

        // Estimate memory required to hold all leaf data. The in-memory path
        // keeps all points in a HashMap<VoxelKey, Vec<RawPoint>> and during
        // grid_sample temporarily holds both input and output, so the peak is
        // roughly 2x the raw data size.
        let total_bytes: u64 = leaf_keys
            .iter()
            .map(|k| self.node_path(k).metadata().map_or(0, |m| m.len()))
            .sum();
        let estimated_peak = total_bytes * 2;

        config.report(crate::ProgressEvent::StageStart {
            name: "Building",
            total: actual_max_depth as u64,
        });

        // Phase 2
        let result = if estimated_peak <= config.memory_budget {
            info!(
                "Build path: IN-MEMORY (leaf data={} MB, est peak={} MB, budget={} MB)",
                total_bytes / 1_048_576,
                estimated_peak / 1_048_576,
                config.memory_budget / 1_048_576,
            );
            self.bottom_up_in_memory(&leaf_keys, actual_max_depth, config)?
        } else {
            info!(
                "Build path: OUT-OF-CORE (leaf data={} MB, est peak={} MB > budget={} MB)",
                total_bytes / 1_048_576,
                estimated_peak / 1_048_576,
                config.memory_budget / 1_048_576,
            );
            self.bottom_up_on_disk(leaf_keys, actual_max_depth, config.memory_budget, config)?
        };
        config.report(crate::ProgressEvent::StageDone);

        let total_pts: usize = result.iter().map(|(_, c)| *c).sum();
        info!(
            "Total octree nodes: {}, total points: {} (original: {})",
            result.len(),
            total_pts,
            self.total_points
        );
        if total_pts as u64 != self.total_points {
            debug!(
                "COPC contains {} points vs {} from input headers (diff {}). \
                 Input LAZ headers sometimes report inaccurate point counts.",
                total_pts,
                self.total_points,
                self.total_points as i64 - total_pts as i64
            );
        }

        // Ensure every ancestor of every data node is present in the hierarchy
        // (empty ancestors allow validators to traverse the tree top-down).
        let mut result = result;
        let mut present: HashSet<VoxelKey> = result.iter().map(|(k, _)| *k).collect();
        let mut extra: Vec<VoxelKey> = Vec::new();
        for (key, _) in &result {
            let mut k = *key;
            while let Some(parent) = k.parent() {
                if present.insert(parent) {
                    extra.push(parent);
                }
                k = parent;
            }
        }
        for k in extra {
            result.push((k, 0));
        }
        result.sort_by_key(|(k, _)| k.level);
        Ok(result)
    }

    /// Bottom-up pass — in-memory fast path.
    ///
    /// Loads all leaf data into a HashMap, runs grid_sample at every level
    /// without any intermediate disk I/O, then writes the final node files once.
    fn bottom_up_in_memory(
        &self,
        leaf_keys: &[VoxelKey],
        actual_max_depth: u32,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        debug!(
            "In-memory build: loading {} leaf nodes in parallel",
            leaf_keys.len()
        );
        // Load all leaf nodes in parallel.
        let mut nodes: HashMap<VoxelKey, Vec<RawPoint>> = leaf_keys
            .par_iter()
            .map(|k| -> Result<(VoxelKey, Vec<RawPoint>)> { Ok((*k, self.read_node(k)?)) })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, pts)| !pts.is_empty())
            .collect();

        let total_in_mem: usize = nodes.values().map(|v| v.len()).sum();
        debug!(
            "In-memory build: loaded {} nodes, {} points ({} MB)",
            nodes.len(),
            total_in_mem,
            (total_in_mem * std::mem::size_of::<RawPoint>()) / 1_048_576,
        );

        for d in (0..actual_max_depth).rev() {
            config.report(crate::ProgressEvent::StageProgress {
                done: (actual_max_depth - d) as u64,
            });
            let level_points: usize = nodes
                .iter()
                .filter(|(k, _)| k.level as u32 == d + 1)
                .map(|(_, v)| v.len())
                .sum();
            debug!(
                "In-memory level {d}: {} total points at child level, {} nodes in map ({} MB est)",
                level_points,
                nodes.len(),
                (nodes.values().map(|v| v.len()).sum::<usize>() * std::mem::size_of::<RawPoint>())
                    / 1_048_576,
            );

            // Group children at level d+1 by parent (iterate keys, no disk I/O).
            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for k in nodes.keys() {
                if k.level as u32 == d + 1
                    && let Some(p) = k.parent()
                {
                    parent_children.entry(p).or_default().push(*k);
                }
            }
            if parent_children.is_empty() {
                continue;
            }

            // Remove children data from `nodes` and build owned tasks (no cloning).
            let tasks: Vec<SampleTask> = parent_children
                .into_iter()
                .map(|(parent, children)| {
                    let all_pts: Vec<(usize, RawPoint)> = children
                        .iter()
                        .enumerate()
                        .flat_map(|(ci, ck)| {
                            nodes
                                .remove(ck)
                                .unwrap_or_default()
                                .into_iter()
                                .map(move |p| (ci, p))
                        })
                        .collect();
                    (parent, children, all_pts)
                })
                .collect();

            // Grid-sample in parallel.
            let results: Vec<SampleResult> = tasks
                .into_par_iter()
                .map(|(parent, children, all_pts)| -> Result<_> {
                    if all_pts.is_empty() {
                        let n = children.len();
                        return Ok((parent, children, vec![], vec![vec![]; n]));
                    }
                    let n = children.len();
                    let (parent_pts, remaining) = self.grid_sample(&parent, all_pts, n);
                    Ok((parent, children, parent_pts, remaining))
                })
                .collect::<Result<_>>()?;

            // Apply updates to `nodes` (sequential, needs &mut).
            for (parent, children, parent_pts, remaining) in results {
                for (ck, rem) in children.into_iter().zip(remaining.into_iter()) {
                    if !rem.is_empty() {
                        nodes.insert(ck, rem);
                    }
                }
                if !parent_pts.is_empty() {
                    nodes.insert(parent, parent_pts);
                }
            }
        }

        // Write final nodes to disk for the writer.
        nodes
            .par_iter()
            .map(|(k, pts)| -> Result<()> { self.write_node_to_temp(k, pts) })
            .collect::<Result<Vec<_>>>()?;

        Ok(nodes
            .iter()
            .filter(|(_, pts)| !pts.is_empty())
            .map(|(k, pts)| (*k, pts.len()))
            .collect())
    }

    /// Bottom-up pass — disk-based slow path.
    ///
    /// Processes one level at a time with disk I/O.  Avoids redundant directory
    /// scans by maintaining a per-level key map updated after each level.
    /// Parents are batched so that the estimated peak memory of all concurrently
    /// loaded points stays within the configured memory budget.
    fn bottom_up_on_disk(
        &self,
        leaf_keys: Vec<VoxelKey>,
        actual_max_depth: u32,
        memory_budget: u64,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Organise known keys by level — updated as parent nodes are created.
        // Use a set per level to prevent duplicate keys (a leaf and a parent
        // created by the bottom-up pass can share the same key when
        // normalize_leaves produces leaves at mixed depths).
        let mut keys_by_level: HashMap<i32, HashSet<VoxelKey>> = HashMap::new();
        for k in leaf_keys {
            keys_by_level.entry(k.level).or_default().insert(k);
        }

        // In-memory cost per point during grid_sample: the (usize, RawPoint) input
        // vec plus the output vecs (parent + per-child remaining) — roughly 2x the
        // input since points move from input to output.
        const MEM_PER_POINT: u64 =
            (std::mem::size_of::<(usize, RawPoint)>() + std::mem::size_of::<RawPoint>()) as u64;

        for d in (0..actual_max_depth).rev() {
            config.report(crate::ProgressEvent::StageProgress {
                done: (actual_max_depth - d) as u64,
            });
            debug!("Building ancestor level {d}");
            let child_keys: Vec<VoxelKey> = match keys_by_level.get(&(d as i32 + 1)) {
                Some(v) => v.iter().copied().collect(),
                None => continue,
            };

            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for ck in &child_keys {
                if let Some(parent) = ck.parent() {
                    parent_children.entry(parent).or_default().push(*ck);
                }
            }
            if parent_children.is_empty() {
                continue;
            }

            // Estimate each parent's memory cost from its children's file sizes,
            // then batch parents so the total stays within the memory budget.
            // Parents that exceed the budget on their own use a streaming path.
            let mut small_parents: Vec<(VoxelKey, Vec<VoxelKey>, u64)> = Vec::new();
            let mut large_parents: Vec<(VoxelKey, Vec<VoxelKey>)> = Vec::new();

            for (parent, children) in parent_children {
                let child_bytes: u64 = children
                    .iter()
                    .map(|ck| self.node_path(ck).metadata().map_or(0, |m| m.len()))
                    .sum();
                let est_points = child_bytes / RawPoint::BYTE_SIZE as u64;
                let est_mem = est_points * MEM_PER_POINT;
                if est_mem > memory_budget {
                    large_parents.push((parent, children));
                } else {
                    small_parents.push((parent, children, est_mem));
                }
            }

            debug!(
                "Level {d}: {} parents ({} small, {} large/streaming), budget={} MB",
                small_parents.len() + large_parents.len(),
                small_parents.len(),
                large_parents.len(),
                memory_budget / 1_048_576,
            );

            let mut new_parent_keys: Vec<VoxelKey> = Vec::new();

            // Process oversized parents sequentially with streaming grid_sample.
            // Only one child is in memory at a time.
            for (parent, children) in &large_parents {
                let child_bytes: u64 = children
                    .iter()
                    .map(|ck| self.node_path(ck).metadata().map_or(0, |m| m.len()))
                    .sum();
                debug!(
                    "  STREAMING parent {:?}: {} children, {} MB on disk",
                    parent,
                    children.len(),
                    child_bytes / 1_048_576,
                );
                self.grid_sample_streaming(parent, children)?;
                new_parent_keys.push(*parent);
            }

            // Batch small parents for parallel processing.
            small_parents.sort_by(|a, b| b.2.cmp(&a.2));

            let mut batch_start = 0;
            while batch_start < small_parents.len() {
                let mut batch_mem: u64 = 0;
                let mut batch_end = batch_start;
                while batch_end < small_parents.len() {
                    if batch_end > batch_start
                        && batch_mem + small_parents[batch_end].2 > memory_budget
                    {
                        break;
                    }
                    batch_mem += small_parents[batch_end].2;
                    batch_end += 1;
                }

                let batch = &small_parents[batch_start..batch_end];
                debug!(
                    "  Batch: {} parents, est {} MB",
                    batch.len(),
                    batch_mem / 1_048_576,
                );
                batch
                    .par_iter()
                    .map(|(parent, children, _)| -> Result<()> {
                        let mut all_pts: Vec<(usize, RawPoint)> = Vec::new();
                        for (ci, ck) in children.iter().enumerate() {
                            for p in self.read_node(ck)? {
                                all_pts.push((ci, p));
                            }
                        }
                        if all_pts.is_empty() {
                            return Ok(());
                        }
                        let (parent_pts, per_child) =
                            self.grid_sample(parent, all_pts, children.len());
                        for (ci, ck) in children.iter().enumerate() {
                            self.write_node_to_temp(ck, &per_child[ci])?;
                        }
                        if !parent_pts.is_empty() {
                            self.write_node_to_temp(parent, &parent_pts)?;
                        }
                        Ok(())
                    })
                    .collect::<Result<Vec<_>>>()?;

                for (parent, _, _) in &small_parents[batch_start..batch_end] {
                    new_parent_keys.push(*parent);
                }
                batch_start = batch_end;
            }

            keys_by_level
                .entry(d as i32)
                .or_default()
                .extend(new_parent_keys.into_iter());
        }

        // Enumerate result from the in-memory key map (no directory scan needed).
        let mut result = Vec::new();
        for level_keys in keys_by_level.values() {
            for key in level_keys {
                let file_len = self.node_path(key).metadata().map_or(0, |m| m.len());
                let count = file_len as usize / RawPoint::BYTE_SIZE;
                if count > 0 {
                    result.push((*key, count));
                }
            }
        }

        // Verify point conservation — flag if total differs from input.
        let total: u64 = result.iter().map(|(_, c)| *c as u64).sum();
        if total != self.total_points {
            info!(
                "Point conservation check: octree has {} points vs {} input (diff {})",
                total,
                self.total_points,
                total as i64 - self.total_points as i64
            );
        }

        Ok(result)
    }

    /// Streaming grid-sample for a single oversized parent.
    ///
    /// Processes one child at a time so that only one child's points are ever in
    /// memory. The grid occupancy set (max 128³ entries ≈ 48 MB) and the parent
    /// accumulator (max 128³ points ≈ 80 MB) stay resident; everything else is
    /// streamed from/to disk.
    ///
    /// Tradeoff: points are not Morton-sorted across children, so spatial
    /// coherence of the parent is slightly worse than the in-memory path.
    fn grid_sample_streaming(&self, parent: &VoxelKey, children: &[VoxelKey]) -> Result<()> {
        let voxel_size_world = 2.0 * self.halfsize / (1u64 << parent.level) as f64;
        let origin_x = ((self.cx - self.halfsize + parent.x as f64 * voxel_size_world
            - self.offset_x)
            / self.scale_x)
            .round() as i64;
        let origin_y = ((self.cy - self.halfsize + parent.y as f64 * voxel_size_world
            - self.offset_y)
            / self.scale_y)
            .round() as i64;
        let origin_z = ((self.cz - self.halfsize + parent.z as f64 * voxel_size_world
            - self.offset_z)
            / self.scale_z)
            .round() as i64;
        let int_size =
            (voxel_size_world / self.scale_x.min(self.scale_y).min(self.scale_z)).round() as i64;
        let cell = (int_size / GRID_CELLS_PER_AXIS).max(1);

        let grid_key = |p: &RawPoint| -> (i32, i32, i32) {
            (
                ((p.x as i64 - origin_x) / cell) as i32,
                ((p.y as i64 - origin_y) / cell) as i32,
                ((p.z as i64 - origin_z) / cell) as i32,
            )
        };

        let max_accepted =
            (GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS) as usize;
        let mut occupied: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut parent_pts: Vec<RawPoint> = Vec::new();

        // Track which children had points, to guarantee at least one per child.
        let mut child_donated: Vec<bool> = vec![false; children.len()];

        // Pass 1: iterate children one at a time, accept points into the parent
        // grid, write remaining back to the child's temp file.
        for (ci, ck) in children.iter().enumerate() {
            let pts = self.read_node(ck)?;
            if pts.is_empty() {
                continue;
            }
            debug!(
                "    streaming child {}/{}: {:?} → {} points ({} MB), grid {}/{} full",
                ci + 1,
                children.len(),
                ck,
                pts.len(),
                (pts.len() * std::mem::size_of::<RawPoint>()) / 1_048_576,
                occupied.len(),
                max_accepted,
            );
            let mut remaining: Vec<RawPoint> = Vec::new();
            for p in pts {
                if parent_pts.len() < max_accepted && occupied.insert(grid_key(&p)) {
                    child_donated[ci] = true;
                    parent_pts.push(p);
                } else {
                    remaining.push(p);
                }
            }
            // If this child had points but none remain (all promoted), we need to
            // return at least one. Done in pass 2 below.
            if remaining.is_empty() && child_donated[ci] {
                // Will be fixed in pass 2.
                child_donated[ci] = false;
            }
            self.write_node_to_temp(ck, &remaining)?;
        }

        // Pass 2: guarantee every child that contributed keeps at least one point.
        for (ci, ck) in children.iter().enumerate() {
            if !child_donated[ci] {
                // Check if this child originally had points (file might now be
                // empty because all were promoted).
                let file_len = self.node_path(ck).metadata().map_or(0, |m| m.len());
                if file_len == 0 && !parent_pts.is_empty() {
                    // Return one point from the parent back to this child.
                    // Find the last parent point that came from any child — we
                    // can't track origin in the streaming path, so just pop the last.
                    if let Some(p) = parent_pts.pop() {
                        self.write_node_to_temp(ck, &[p])?;
                    }
                }
            }
        }

        if !parent_pts.is_empty() {
            self.write_node_to_temp(parent, &parent_pts)?;
        }

        Ok(())
    }

    /// Grid-based spatial sampling for one parent node.
    ///
    /// Divides the parent voxel into a uniform grid of GRID_CELLS_PER_AXIS³ cells.
    /// Points are sorted by Morton code and iterated in that order; the first point
    /// that falls into each unoccupied cell is accepted for the parent.  All others
    /// are returned to their originating child so every point lands in exactly one node.
    ///
    /// This mirrors untwine's approach and produces spatially homogeneous LOD levels.
    fn grid_sample(
        &self,
        parent: &VoxelKey,
        mut pts: Vec<(usize, RawPoint)>, // takes ownership — no cloning
        n_children: usize,
    ) -> (Vec<RawPoint>, Vec<Vec<RawPoint>>) {
        if pts.is_empty() {
            return (vec![], vec![vec![]; n_children]);
        }

        // Parent voxel geometry in integer coordinate space.
        let voxel_size_world = 2.0 * self.halfsize / (1u64 << parent.level) as f64;
        let origin_x = ((self.cx - self.halfsize + parent.x as f64 * voxel_size_world
            - self.offset_x)
            / self.scale_x)
            .round() as i64;
        let origin_y = ((self.cy - self.halfsize + parent.y as f64 * voxel_size_world
            - self.offset_y)
            / self.scale_y)
            .round() as i64;
        let origin_z = ((self.cz - self.halfsize + parent.z as f64 * voxel_size_world
            - self.offset_z)
            / self.scale_z)
            .round() as i64;
        let int_size =
            (voxel_size_world / self.scale_x.min(self.scale_y).min(self.scale_z)).round() as i64;

        // Grid resolution: fixed cells per axis, matching untwine's CellCount.
        let cell = (int_size / GRID_CELLS_PER_AXIS).max(1);

        // Sort by Morton code within the parent voxel for spatially coherent traversal.
        pts.sort_unstable_by_key(|(_, p)| {
            let dx = (p.x as i64 - origin_x).max(0) as u32;
            let dy = (p.y as i64 - origin_y).max(0) as u32;
            let dz = (p.z as i64 - origin_z).max(0) as u32;
            morton3(dx, dy, dz)
        });

        let grid_key = |p: &RawPoint| -> (i32, i32, i32) {
            (
                ((p.x as i64 - origin_x) / cell) as i32,
                ((p.y as i64 - origin_y) / cell) as i32,
                ((p.z as i64 - origin_z) / cell) as i32,
            )
        };

        // Track which children actually have points so we can protect them.
        let mut child_has_pts = vec![false; n_children];
        for (ci, _) in &pts {
            child_has_pts[*ci] = true;
        }

        // Partition: accepted for parent vs remaining for children. No cloning.
        let mut occupied: HashSet<(i32, i32, i32)> = HashSet::new();
        let max_accepted =
            (GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS) as usize;
        let mut parent_pts: Vec<(usize, RawPoint)> = Vec::with_capacity(max_accepted);
        let mut remaining: Vec<Vec<RawPoint>> = vec![Vec::new(); n_children];

        for (ci, p) in pts {
            if parent_pts.len() < max_accepted && occupied.insert(grid_key(&p)) {
                parent_pts.push((ci, p));
            } else {
                remaining[ci].push(p);
            }
        }

        // Guarantee every child that contributed points keeps at least one.
        // This prevents zero-point intermediate nodes in the COPC hierarchy
        // (which confuse validators that check point_count > 0 for all entries).
        for ci in 0..n_children {
            if child_has_pts[ci]
                && remaining[ci].is_empty()
                && let Some(pos) = parent_pts.iter().rposition(|(c, _)| *c == ci)
            {
                let (_, p) = parent_pts.remove(pos);
                remaining[ci].push(p);
            }
        }

        let parent_pts = parent_pts.into_iter().map(|(_, p)| p).collect();
        (parent_pts, remaining)
    }

    // -----------------------------------------------------------------------
    // Chunked-build path (Schütz et al. 2020).
    //
    // See docs/phase-2b-chunked-build.md for the full design. The chunked
    // path replaces both `distribute_per_leaf` and `build_node_map_per_leaf`,
    // routing intermediate state via `self.chunked_plan`.
    // -----------------------------------------------------------------------

    /// Chunked distribute. Computes the chunk plan via the counting-sort
    /// analyzer, builds a fast cell→chunk lookup table, then streams each
    /// input file in parallel and appends each point to its chunk's temp
    /// file via a bounded LRU writer cache.
    ///
    /// See `docs/phase-2b-chunked-build.md` §4.1-4.2 for the design.
    fn distribute_chunked(
        &mut self,
        input_files: &[PathBuf],
        config: &PipelineConfig,
    ) -> Result<()> {
        // 1. Compute the chunk plan (counting + merge-sparse-cells).
        let plan = crate::chunking::compute_chunk_plan(self, input_files, config, None)?;
        info!(
            "Chunked distribute: {} chunks, target {} points each, grid {}³",
            plan.chunks.len(),
            plan.chunk_target,
            plan.grid_size
        );

        // 2. Build the cell → chunk_index lookup table. The grid has
        //    grid_size³ fine cells; for each chunk we fill in every fine cell
        //    it covers (a 2^(grid_depth - chunk.level) cube per axis).
        let lut = build_chunk_lut(&plan);

        // 3. Set up shard subdirectories under tmp_dir/chunks/shards/{worker}.
        let chunks_root = self.tmp_dir.join("chunks");
        if chunks_root.exists() {
            std::fs::remove_dir_all(&chunks_root)
                .with_context(|| format!("removing stale chunks dir {:?}", chunks_root))?;
        }
        std::fs::create_dir_all(&chunks_root)?;
        let shards_root = chunks_root.join("shards");
        std::fs::create_dir_all(&shards_root)?;

        let num_workers = chunked_distribute_worker_count(input_files.len());
        let shard_dirs: Vec<PathBuf> = (0..num_workers)
            .map(|w| {
                let d = shards_root.join(w.to_string());
                std::fs::create_dir_all(&d)?;
                Ok(d)
            })
            .collect::<Result<Vec<_>>>()?;

        // 4. Compute per-worker resources.
        let per_worker_cap = (CHUNKED_OPEN_FILES_CAP / num_workers).max(MIN_PER_WORKER_CHUNK_FILES);

        // Per-worker memory budget for the input chunk size. The transient
        // peak is `Vec<las::Point>` (~120 B/pt) plus the cache's BufWriter
        // overhead (negligible). Size for 1/8 of the per-worker budget so
        // there's headroom for the LAZ decoder + grouping overhead.
        const BYTES_PER_POINT_TRANSIENT: u64 = 120;
        let per_worker_budget = config.memory_budget / num_workers as u64;
        let read_chunk_size =
            ((per_worker_budget / 8) / BYTES_PER_POINT_TRANSIENT).clamp(50_000, 2_000_000) as usize;

        debug!(
            "Chunked distribute: budget={} MB, workers={}, read_chunk_size={}, open-file cap={} (per worker)",
            config.memory_budget / 1_048_576,
            num_workers,
            read_chunk_size,
            per_worker_cap
        );

        // Pre-compute geometry for the inlined grid-cell math (per design
        // §5: ~3× faster than calling point_to_key with a depth-9 loop).
        let g = plan.grid_size;
        let cube_size = self.halfsize * 2.0;
        let cube_min_x = self.cx - self.halfsize;
        let cube_min_y = self.cy - self.halfsize;
        let cube_min_z = self.cz - self.halfsize;
        let g_per_cube = g as f64 / cube_size;
        let g_max = g as i64 - 1;

        // Capture the LUT and per-worker file ranges for the parallel block.
        let files_per_worker = input_files.len().div_ceil(num_workers);
        let progress = AtomicU64::new(0);

        // 5. Parallel workers stream input files and append points to chunks.
        shard_dirs
            .par_iter()
            .enumerate()
            .try_for_each(|(worker_id, shard_dir)| -> Result<()> {
                let start = worker_id * files_per_worker;
                let end = (start + files_per_worker).min(input_files.len());
                if start >= end {
                    return Ok(());
                }

                let mut cache = ChunkWriterCache::new(per_worker_cap);
                let mut points: Vec<las::Point> = Vec::with_capacity(read_chunk_size);
                // Group buffer reused across batches so capacity amortizes.
                // Drained at the end of each batch so retained Vec capacity
                // stays bounded by the working set, not by all-time peak.
                let mut groups: HashMap<u32, Vec<RawPoint>> = HashMap::new();

                for path in &input_files[start..end] {
                    let mut reader = las::Reader::from_path(path)
                        .with_context(|| format!("Cannot open {:?}", path))?;
                    let file_pts = reader.header().number_of_points();
                    debug!(
                        "Chunked distribute worker {} file {:?}: {} points",
                        worker_id,
                        path.file_name().unwrap_or_default(),
                        file_pts
                    );

                    loop {
                        points.clear();
                        let n = reader.read_points_into(read_chunk_size as u64, &mut points)?;
                        if n == 0 {
                            break;
                        }

                        // Classify points into chunks via the LUT and group
                        // them in memory before writing. Grouping batches
                        // many points per writer call → fewer LRU touches
                        // and better amortization than per-point appends.
                        for p in &points {
                            // Convert via the same RawPoint conversion the
                            // per-leaf path uses, so the chunked path stores
                            // the exact same int-coordinate representation.
                            let raw = self.convert_point(p);

                            // Inline grid-cell computation: ~3× faster than
                            // point_to_key. Use round-tripped world coords
                            // (raw.x * scale + offset) so the cell assignment
                            // matches what the per-chunk build will see when
                            // it reads RawPoints back from the chunk file.
                            let wx = raw.x as f64 * self.scale_x + self.offset_x;
                            let wy = raw.y as f64 * self.scale_y + self.offset_y;
                            let wz = raw.z as f64 * self.scale_z + self.offset_z;

                            let mut gx = ((wx - cube_min_x) * g_per_cube).floor() as i64;
                            let mut gy = ((wy - cube_min_y) * g_per_cube).floor() as i64;
                            let mut gz = ((wz - cube_min_z) * g_per_cube).floor() as i64;
                            // Defensive clamp for points exactly on the
                            // upper boundary that round out by one.
                            gx = gx.clamp(0, g_max);
                            gy = gy.clamp(0, g_max);
                            gz = gz.clamp(0, g_max);

                            let cell_idx = gx as usize
                                + gy as usize * g as usize
                                + gz as usize * g as usize * g as usize;
                            let chunk_idx = lut[cell_idx];
                            groups.entry(chunk_idx).or_default().push(raw);
                        }

                        // Free the las::Point buffer before flushing groups.
                        points.clear();

                        // Flush each group to the cache. Drain so vec capacity
                        // doesn't accumulate across batches.
                        for (chunk_idx, pts) in groups.drain() {
                            cache.append(chunk_idx, shard_dir, &pts)?;
                        }

                        let done = progress.fetch_add(n, Ordering::Relaxed) + n;
                        config.report(crate::ProgressEvent::StageProgress { done });
                    }
                }

                cache.flush_all()
            })?;

        // 6. Merge per-worker shards into canonical chunk files.
        merge_chunk_shards(&shards_root, &chunks_root, plan.chunks.len() as u32)?;

        // 7. Stash the chunk plan for the build phase to consume.
        self.chunked_plan = Some(plan);
        Ok(())
    }

    /// Chunked build. Per-chunk in-memory build, then merge across chunks
    /// at coarse levels up to the global root.
    fn build_node_map_chunked(
        &self,
        _config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Implemented incrementally — see docs/phase-2b-chunked-build.md §4.3-4.4.
        anyhow::bail!("chunked build not yet implemented")
    }
}

impl Drop for OctreeBuilder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.tmp_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point() -> RawPoint {
        RawPoint {
            x: -123456,
            y: 789012,
            z: -1,
            intensity: 65535,
            return_number: 3,
            number_of_returns: 5,
            classification: 6,
            scan_angle: -15000,
            user_data: 42,
            point_source_id: 1001,
            gps_time: 123456789.987654,
            red: 255,
            green: 0,
            blue: 65535,
            nir: 32768,
        }
    }

    #[test]
    fn rawpoint_roundtrip_single() {
        let p = sample_point();
        let mut buf = Vec::new();
        p.write(&mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::BYTE_SIZE);
        let p2 = RawPoint::read(&mut &buf[..]).unwrap();
        assert_eq!(p.x, p2.x);
        assert_eq!(p.y, p2.y);
        assert_eq!(p.z, p2.z);
        assert_eq!(p.intensity, p2.intensity);
        assert_eq!(p.return_number, p2.return_number);
        assert_eq!(p.number_of_returns, p2.number_of_returns);
        assert_eq!(p.classification, p2.classification);
        assert_eq!(p.scan_angle, p2.scan_angle);
        assert_eq!(p.user_data, p2.user_data);
        assert_eq!(p.point_source_id, p2.point_source_id);
        assert_eq!(p.gps_time, p2.gps_time);
        assert_eq!(p.red, p2.red);
        assert_eq!(p.green, p2.green);
        assert_eq!(p.blue, p2.blue);
        assert_eq!(p.nir, p2.nir);
    }

    #[test]
    fn rawpoint_roundtrip_bulk() {
        let points = vec![
            sample_point(),
            RawPoint {
                x: 0,
                y: 0,
                z: 0,
                intensity: 0,
                return_number: 0,
                number_of_returns: 0,
                classification: 0,
                scan_angle: 0,
                user_data: 0,
                point_source_id: 0,
                gps_time: 0.0,
                red: 0,
                green: 0,
                blue: 0,
                nir: 0,
            },
            sample_point(),
        ];

        let mut buf = Vec::new();
        RawPoint::write_bulk(&points, &mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::BYTE_SIZE * 3);

        // Read them back one at a time
        let mut cursor = std::io::Cursor::new(&buf[..]);
        for orig in &points {
            let p = RawPoint::read(&mut cursor).unwrap();
            assert_eq!(orig.x, p.x);
            assert_eq!(orig.gps_time, p.gps_time);
            assert_eq!(orig.nir, p.nir);
        }
    }

    #[test]
    fn rawpoint_bulk_matches_single() {
        let p = sample_point();
        let mut single_buf = Vec::new();
        p.write(&mut single_buf).unwrap();

        let mut bulk_buf = Vec::new();
        RawPoint::write_bulk(std::slice::from_ref(&p), &mut bulk_buf).unwrap();

        assert_eq!(
            single_buf, bulk_buf,
            "bulk write must produce identical bytes to single write"
        );
    }

    #[test]
    fn input_to_copc_format_mapping() {
        // No color, no NIR → format 6
        assert_eq!(input_to_copc_format(0), 6);
        assert_eq!(input_to_copc_format(1), 6);
        assert_eq!(input_to_copc_format(4), 6);
        assert_eq!(input_to_copc_format(6), 6);
        assert_eq!(input_to_copc_format(9), 6);

        // Has color, no NIR → format 7
        assert_eq!(input_to_copc_format(2), 7);
        assert_eq!(input_to_copc_format(3), 7);
        assert_eq!(input_to_copc_format(5), 7);
        assert_eq!(input_to_copc_format(7), 7);

        // Has color + NIR → format 8
        assert_eq!(input_to_copc_format(8), 8);
        assert_eq!(input_to_copc_format(10), 8);
    }

    // ----- Chunked-build helpers -----

    use crate::chunking::{ChunkPlan, PlannedChunk};

    fn make_plan(grid_size: u32, chunks: Vec<PlannedChunk>) -> ChunkPlan {
        let total_points = chunks.iter().map(|c| c.point_count).sum();
        ChunkPlan {
            grid_size,
            grid_depth: grid_size.trailing_zeros(),
            chunk_target: 1_000_000,
            total_points,
            chunks,
        }
    }

    #[test]
    fn build_chunk_lut_single_root_chunk() {
        // 4³ grid, one chunk at level 0 (the root) covering everything.
        let plan = make_plan(
            4,
            vec![PlannedChunk {
                level: 0,
                gx: 0,
                gy: 0,
                gz: 0,
                point_count: 100,
            }],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 64);
        assert!(lut.iter().all(|&c| c == 0));
    }

    #[test]
    fn build_chunk_lut_eight_finest_chunks() {
        // 2³ grid (depth 1). Eight chunks at level 1, one per fine cell.
        let chunks: Vec<PlannedChunk> = (0i32..8)
            .map(|i| PlannedChunk {
                level: 1,
                gx: i % 2,
                gy: (i / 2) % 2,
                gz: i / 4,
                point_count: 10,
            })
            .collect();
        let plan = make_plan(2, chunks);
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 8);
        // Each cell should map to a distinct chunk index 0..8.
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &c in &lut {
            assert!(c < 8, "got out-of-range chunk idx {c}");
            seen.insert(c);
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn build_chunk_lut_mixed_levels() {
        // 4³ grid (depth 2). Two chunks:
        //   - chunk 0 at level 1, position (0,0,0) — covers the 2³ subcube at the origin (8 cells)
        //   - chunk 1 at level 2, position (3,3,3) — covers exactly 1 fine cell at (3,3,3)
        // The remaining 64 - 8 - 1 = 55 cells should be u32::MAX (uncovered).
        let plan = make_plan(
            4,
            vec![
                PlannedChunk {
                    level: 1,
                    gx: 0,
                    gy: 0,
                    gz: 0,
                    point_count: 50,
                },
                PlannedChunk {
                    level: 2,
                    gx: 3,
                    gy: 3,
                    gz: 3,
                    point_count: 5,
                },
            ],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 64);

        // Cells (gx, gy, gz) with all coords in 0..2 should map to chunk 0.
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let idx = x + y * 4 + z * 16;
                    assert_eq!(lut[idx], 0, "cell ({x},{y},{z}) should map to chunk 0");
                }
            }
        }

        // Cell (3, 3, 3) → chunk 1.
        assert_eq!(lut[3 + 3 * 4 + 3 * 16], 1);

        // Count uncovered cells: should be 64 - 8 - 1 = 55.
        let uncovered = lut.iter().filter(|&&c| c == u32::MAX).count();
        assert_eq!(uncovered, 55);
    }

    #[test]
    fn build_chunk_lut_chunk_at_corner() {
        // 8³ grid (depth 3). One chunk at level 2 position (3, 3, 3) — covers
        // 2³ cells at the far corner of the grid: (6..8, 6..8, 6..8).
        let plan = make_plan(
            8,
            vec![PlannedChunk {
                level: 2,
                gx: 3,
                gy: 3,
                gz: 3,
                point_count: 10,
            }],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 512);
        let g = 8;
        for z in 6..8 {
            for y in 6..8 {
                for x in 6..8 {
                    let idx = x + y * g + z * g * g;
                    assert_eq!(lut[idx], 0, "cell ({x},{y},{z}) should be in the chunk");
                }
            }
        }
        // No other cell should be in the chunk.
        let in_chunk = lut.iter().filter(|&&c| c == 0).count();
        assert_eq!(in_chunk, 8);
    }
}
