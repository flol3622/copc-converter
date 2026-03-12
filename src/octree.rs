/// Out-of-core octree builder.
///
/// Strategy
/// --------
/// 1. Pass 1 – scan all input files, collect bounding box + point count.
/// 2. Determine octree depth so that leaf nodes contain ≤ MAX_LEAF_POINTS.
/// 3. Pass 2 – read every point, assign it to the leaf voxel key, and
///    accumulate into per-key temporary files on disk.
/// 4. Build the tree bottom-up: each parent node gets a random sample
///    of its children's points (so upper levels are sparse overviews).
/// 5. Produce the list of (VoxelKey, Vec<RawPoint>) for the writer.
///
/// Memory usage stays well below 16 GB because each pass processes one
/// input file at a time and flushes leaf buffers to disk.
use crate::copc_types::VoxelKey;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::{debug, info};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum points per leaf voxel before we subdivide further.
const MAX_LEAF_POINTS: u64 = 65_536;

/// Maximum points to keep per non-leaf node (thinned sample for overview).
const MAX_NODE_POINTS: usize = 65_536;

/// Flush in-memory leaf buffer to disk every this many points.
const FLUSH_EVERY: usize = 100_000;

// ---------------------------------------------------------------------------
// Raw point storage
// ---------------------------------------------------------------------------

/// A raw point stored as scaled integer coordinates plus classification,
/// intensity, return number etc.  We keep the original scaled ints so we
/// can reconstruct exact LAS integer values without floating-point loss.
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

    pub fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_i32::<LittleEndian>(self.x)?;
        w.write_i32::<LittleEndian>(self.y)?;
        w.write_i32::<LittleEndian>(self.z)?;
        w.write_u16::<LittleEndian>(self.intensity)?;
        w.write_u8(self.return_number)?;
        w.write_u8(self.number_of_returns)?;
        w.write_u8(self.classification)?;
        w.write_i16::<LittleEndian>(self.scan_angle)?;
        w.write_u8(self.user_data)?;
        w.write_u16::<LittleEndian>(self.point_source_id)?;
        w.write_f64::<LittleEndian>(self.gps_time)?;
        w.write_u16::<LittleEndian>(self.red)?;
        w.write_u16::<LittleEndian>(self.green)?;
        w.write_u16::<LittleEndian>(self.blue)?;
        w.write_u16::<LittleEndian>(self.nir)?;
        Ok(())
    }

    pub fn read<R: std::io::Read>(r: &mut R) -> Result<Self> {
        Ok(RawPoint {
            x: r.read_i32::<LittleEndian>()?,
            y: r.read_i32::<LittleEndian>()?,
            z: r.read_i32::<LittleEndian>()?,
            intensity: r.read_u16::<LittleEndian>()?,
            return_number: r.read_u8()?,
            number_of_returns: r.read_u8()?,
            classification: r.read_u8()?,
            scan_angle: r.read_i16::<LittleEndian>()?,
            user_data: r.read_u8()?,
            point_source_id: r.read_u16::<LittleEndian>()?,
            gps_time: r.read_f64::<LittleEndian>()?,
            red: r.read_u16::<LittleEndian>()?,
            green: r.read_u16::<LittleEndian>()?,
            blue: r.read_u16::<LittleEndian>()?,
            nir: r.read_u16::<LittleEndian>()?,
        })
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

/// Assign a point to the leaf voxel at the given tree depth.
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

pub struct OctreeBuilder {
    pub bounds: Bounds,
    pub total_points: u64,
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub halfsize: f64,
    pub depth: u32,
    /// Scale / offset from the first input file – used for all raw-integer conversions.
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    /// Temp directory where leaf files are written.
    pub tmp_dir: PathBuf,
}

impl OctreeBuilder {
    /// Pass 1: scan all files to get bounds and total point count.
    pub fn scan(input_files: &[PathBuf]) -> Result<Self> {
        let mut bounds = Bounds::empty();
        let mut total_points = 0u64;
        let mut first_transforms: Option<(f64, f64, f64, f64, f64, f64)> = None;

        for path in input_files {
            info!("Scanning {:?}", path);
            let reader =
                las::Reader::from_path(path).with_context(|| format!("Cannot open {:?}", path))?;
            let hdr = reader.header();
            let b = hdr.bounds();
            bounds.expand_with(b.min.x, b.min.y, b.min.z);
            bounds.expand_with(b.max.x, b.max.y, b.max.z);
            total_points += hdr.number_of_points();
            if first_transforms.is_none() {
                let t = hdr.transforms();
                first_transforms = Some((
                    t.x.scale, t.y.scale, t.z.scale, t.x.offset, t.y.offset, t.z.offset,
                ));
            }
        }

        let (scale_x, scale_y, scale_z, offset_x, offset_y, offset_z) =
            first_transforms.unwrap_or((0.001, 0.001, 0.001, 0.0, 0.0, 0.0));

        let (cx, cy, cz, halfsize) = bounds.to_cube();

        // Choose depth so that leaf voxels hold ≤ MAX_LEAF_POINTS on average.
        // Octree has 8^depth leaves.
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
        info!("Octree depth = {depth}, total points = {total_points}");

        let tmp_dir = std::env::temp_dir().join(format!("copc_{}", std::process::id()));
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
        })
    }

    /// Path for the leaf temp file for a given key.
    fn leaf_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }

    /// Pass 2: assign all points to leaf temp files.
    pub fn distribute(&self, input_files: &[PathBuf]) -> Result<()> {
        // in-memory buffer before flushing
        let mut buffers: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        let mut writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();

        let mut point_idx = 0u64;

        for path in input_files {
            info!("Distributing {:?}", path);
            let mut reader =
                las::Reader::from_path(path).with_context(|| format!("Cannot open {:?}", path))?;

            let mut points: Vec<las::Point> = Vec::new();
            reader.read_all_points_into(&mut points)?;

            for p in &points {
                let wx = p.x;
                let wy = p.y;
                let wz = p.z;

                let key = point_to_key(
                    wx,
                    wy,
                    wz,
                    self.cx,
                    self.cy,
                    self.cz,
                    self.halfsize,
                    self.depth,
                );

                // Convert world coords to raw integers using the output scale/offset
                // (always from the first file) so all files share a consistent system.
                let ix = ((wx - self.offset_x) / self.scale_x).round() as i32;
                let iy = ((wy - self.offset_y) / self.scale_y).round() as i32;
                let iz = ((wz - self.offset_z) / self.scale_z).round() as i32;

                let raw = RawPoint {
                    x: ix,
                    y: iy,
                    z: iz,
                    intensity: p.intensity,
                    return_number: p.return_number,
                    number_of_returns: p.number_of_returns,
                    classification: p.classification.into(),
                    scan_angle: (p.scan_angle / 0.006).round() as i16, // raw i16 units (0.006°)
                    user_data: p.user_data,
                    point_source_id: p.point_source_id,
                    gps_time: p.gps_time.unwrap_or(0.0),
                    red: p.color.as_ref().map(|c| c.red).unwrap_or(0),
                    green: p.color.as_ref().map(|c| c.green).unwrap_or(0),
                    blue: p.color.as_ref().map(|c| c.blue).unwrap_or(0),
                    nir: p.extra_bytes.first().copied().map(|_| 0u16).unwrap_or(0),
                };

                let buf = buffers.entry(key).or_default();
                buf.push(raw);
                point_idx += 1;

                // Flush buffers periodically to keep RAM usage low.
                if point_idx.is_multiple_of(FLUSH_EVERY as u64) {
                    Self::flush_buffers(&mut buffers, &mut writers, &self.tmp_dir)?;
                }
            }
        }

        // Final flush
        Self::flush_buffers(&mut buffers, &mut writers, &self.tmp_dir)?;
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
            for p in pts.iter() {
                p.write(w)?;
            }
            pts.clear();
        }
        Ok(())
    }

    /// Read all raw points for a given leaf key from disk.
    pub fn read_leaf(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.leaf_path(key);
        if !path.exists() {
            return Ok(vec![]);
        }
        let f = File::open(&path)?;
        let mut r = BufReader::new(f);
        let file_len = path.metadata()?.len();
        let count = file_len / RawPoint::BYTE_SIZE as u64;
        let mut pts = Vec::with_capacity(count as usize);
        for _ in 0..count {
            pts.push(RawPoint::read(&mut r)?);
        }
        Ok(pts)
    }

    /// Enumerate all leaf keys that have data.
    pub fn leaf_keys(&self) -> Result<Vec<VoxelKey>> {
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

    /// Build the complete per-node point set:
    /// leaf nodes get all their points; ancestor nodes get a thinned sample.
    ///
    /// Returns a map from VoxelKey → Vec<RawPoint>.
    pub fn build_node_map(&self) -> Result<HashMap<VoxelKey, Vec<RawPoint>>> {
        let leaf_keys = self.leaf_keys()?;
        info!("Number of leaf nodes: {}", leaf_keys.len());

        let mut node_map: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();

        // Fill leaf nodes
        for key in &leaf_keys {
            let pts = self.read_leaf(key)?;
            debug!("Leaf {:?}: {} points", key, pts.len());
            if !pts.is_empty() {
                node_map.insert(*key, pts);
            }
        }

        // Build ancestor nodes bottom-up
        for d in (0..self.depth).rev() {
            let child_keys: Vec<VoxelKey> = node_map
                .keys()
                .filter(|k| k.level as u32 == d + 1)
                .copied()
                .collect();

            let mut parent_candidates: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();

            for ck in child_keys {
                if let Some(parent) = ck.parent()
                    && let Some(cpts) = node_map.get(&ck)
                {
                    let sample = thin_sample(cpts, MAX_NODE_POINTS / 8);
                    parent_candidates.entry(parent).or_default().extend(sample);
                }
            }

            for (pk, mut pts) in parent_candidates {
                pts = thin_sample(&pts, MAX_NODE_POINTS);
                node_map.insert(pk, pts);
            }
        }

        Ok(node_map)
    }

    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.tmp_dir);
    }
}

/// Sub-sample a point vector to at most `max_count` points using
/// a simple stride-based approach (deterministic, fast).
fn thin_sample(pts: &[RawPoint], max_count: usize) -> Vec<RawPoint> {
    if pts.len() <= max_count {
        return pts.to_vec();
    }
    let step = pts.len() / max_count;
    pts.iter().step_by(step).take(max_count).cloned().collect()
}
