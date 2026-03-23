//! Fast, memory-efficient converter from LAS/LAZ to COPC (Cloud-Optimized Point Cloud).

pub(crate) mod copc_types;
pub(crate) mod octree;
pub(crate) mod validate;
pub(crate) mod writer;

use anyhow::{Context, Result};
use std::path::PathBuf;

// Re-export the public API.
pub use copc_types::VoxelKey;
pub use octree::OctreeBuilder;
pub use validate::{ValidatedInputs, validate};
pub use writer::write_copc;

/// Configuration threaded through the pipeline.
pub struct PipelineConfig {
    /// Effective memory budget in bytes (after safety factor).
    pub memory_budget: u64,
    /// Optional custom temp directory.
    pub temp_dir: Option<PathBuf>,
    /// Whether to write a temporal index EVLR.
    pub temporal_index: bool,
    /// Sampling stride for temporal index.
    pub temporal_stride: u32,
}

/// Parse a human-readable size string into bytes.
/// Supports suffixes: G/g (GiB), M/m (MiB), K/k (KiB), or plain bytes.
pub fn parse_memory_limit(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num_part, multiplier) = if let Some(n) = s.strip_suffix(['G', 'g']) {
        (n.trim(), 1024u64 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix(['M', 'm']) {
        (n.trim(), 1024u64 * 1024)
    } else if let Some(n) = s.strip_suffix(['K', 'k']) {
        (n.trim(), 1024u64)
    } else {
        (s, 1u64)
    };
    let value: f64 = num_part
        .parse()
        .with_context(|| format!("Invalid memory limit: {s:?}"))?;
    Ok((value * multiplier as f64) as u64)
}

/// Expand a single input path into a list of LAZ/LAS files.
/// If `raw` is a directory, all `.laz`/`.las` files in it are returned (sorted).
/// If `raw` is a file, it is returned as-is.
pub fn collect_input_files(raw: PathBuf) -> Result<Vec<PathBuf>> {
    if raw.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&raw)
            .with_context(|| format!("Cannot read directory {:?}", raw))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && matches!(
                        p.extension().and_then(|s| s.to_str()),
                        Some("laz") | Some("las") | Some("LAZ") | Some("LAS")
                    )
            })
            .collect();
        files.sort();
        anyhow::ensure!(!files.is_empty(), "No LAZ/LAS files found in {:?}", raw);
        Ok(files)
    } else {
        Ok(vec![raw])
    }
}
