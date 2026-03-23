//! Fast, memory-efficient converter from LAS/LAZ to COPC (Cloud-Optimized Point Cloud).
//!
//! # Pipeline
//!
//! The conversion pipeline is enforced at compile time via typestate:
//!
//! ```no_run
//! use copc_converter::{Pipeline, PipelineConfig};
//!
//! let config = PipelineConfig {
//!     memory_budget: 12_884_901_888,
//!     temp_dir: None,
//!     temporal_index: false,
//!     temporal_stride: 1000,
//! };
//! let files = copc_converter::collect_input_files("input.laz".into()).unwrap();
//!
//! Pipeline::scan(&files, &config).unwrap()
//!     .validate().unwrap()
//!     .distribute().unwrap()
//!     .build().unwrap()
//!     .write("output.copc.laz").unwrap();
//! ```

pub(crate) mod copc_types;
pub(crate) mod octree;
pub(crate) mod validate;
pub(crate) mod writer;

use log::info;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use copc_types::VoxelKey;
use octree::OctreeBuilder;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors returned by the conversion pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Input files have mismatched CRS.
    #[error("CRS mismatch: {file_a:?} has a different WKT CRS than {file_b:?}")]
    CrsMismatch {
        /// Path of the first file.
        file_a: PathBuf,
        /// Path of the differing file.
        file_b: PathBuf,
    },

    /// Input files have mismatched point formats.
    #[error(
        "Point format mismatch: {file_a:?} has format {format_a} but {file_b:?} has format {format_b}"
    )]
    PointFormatMismatch {
        /// Path of the first file.
        file_a: PathBuf,
        /// Format of the first file.
        format_a: u8,
        /// Path of the differing file.
        file_b: PathBuf,
        /// Format of the differing file.
        format_b: u8,
    },

    /// Temporal index requested but point format lacks GPS time.
    #[error("Temporal index requested but input point format {format} does not include GPS time")]
    NoGpsTime {
        /// The incompatible point format.
        format: u8,
    },

    /// No LAZ/LAS files found in a directory.
    #[error("No LAZ/LAS files found in {path:?}")]
    NoInputFiles {
        /// The directory that was searched.
        path: PathBuf,
    },

    /// I/O or other internal error.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

/// Result type for the conversion pipeline.
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Pipeline typestate stages
// ---------------------------------------------------------------------------

/// Pipeline stage: input files have been scanned.
pub struct Scanned(());
/// Pipeline stage: inputs have been validated.
pub struct Validated(());
/// Pipeline stage: points have been distributed to leaf voxels.
pub struct Distributed(());
/// Pipeline stage: octree has been built with LOD thinning.
pub struct Built(());

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the conversion pipeline.
pub struct PipelineConfig {
    /// Effective memory budget in bytes.
    pub memory_budget: u64,
    /// Optional custom temp directory.
    pub temp_dir: Option<PathBuf>,
    /// Whether to write a temporal index EVLR.
    pub temporal_index: bool,
    /// Sampling stride for temporal index.
    pub temporal_stride: u32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Type-safe conversion pipeline. Each stage exposes only the next step.
pub struct Pipeline<S> {
    inner: PipelineInner,
    _stage: PhantomData<S>,
}

struct PipelineInner {
    input_files: Vec<PathBuf>,
    config: PipelineConfig,
    scan_results: Vec<octree::ScanResult>,
    validated: Option<validate::ValidatedInputs>,
    builder: Option<OctreeBuilder>,
    node_keys: Option<Vec<(VoxelKey, usize)>>,
}

impl Pipeline<Scanned> {
    /// Scan input files to read headers, bounds, CRS, and point format.
    pub fn scan(input_files: &[PathBuf], config: PipelineConfig) -> Result<Self> {
        info!(
            "=== Pass 1: scanning {} input file(s) ===",
            input_files.len()
        );
        let scan_results = OctreeBuilder::scan(input_files)?;
        Ok(Pipeline {
            inner: PipelineInner {
                input_files: input_files.to_vec(),
                config,
                scan_results,
                validated: None,
                builder: None,
                node_keys: None,
            },
            _stage: PhantomData,
        })
    }

    /// Validate that all input files are consistent (CRS, point format).
    pub fn validate(mut self) -> Result<Pipeline<Validated>> {
        info!("=== Validating inputs ===");
        let validated = validate::validate(
            &self.inner.input_files,
            &self.inner.scan_results,
            self.inner.config.temporal_index,
        )?;
        self.inner.validated = Some(validated);
        Ok(Pipeline {
            inner: self.inner,
            _stage: PhantomData,
        })
    }
}

impl Pipeline<Validated> {
    /// Distribute all points to leaf voxels on disk.
    pub fn distribute(mut self) -> Result<Pipeline<Distributed>> {
        let validated = self.inner.validated.as_ref().unwrap();
        let builder =
            OctreeBuilder::from_scan(&self.inner.scan_results, validated, &self.inner.config)?;

        info!("=== Pass 2: distributing points to leaf voxels ===");
        builder.distribute(&self.inner.input_files, &self.inner.config)?;
        self.inner.builder = Some(builder);
        Ok(Pipeline {
            inner: self.inner,
            _stage: PhantomData,
        })
    }
}

impl Pipeline<Distributed> {
    /// Build the octree node map with LOD thinning.
    pub fn build(mut self) -> Result<Pipeline<Built>> {
        info!("=== Building octree node map ===");
        let builder = self.inner.builder.as_ref().unwrap();
        let node_keys = builder.build_node_map(&self.inner.config)?;
        self.inner.node_keys = Some(node_keys);
        Ok(Pipeline {
            inner: self.inner,
            _stage: PhantomData,
        })
    }
}

impl Pipeline<Built> {
    /// Write the COPC file.
    pub fn write(self, output_path: impl AsRef<Path>) -> Result<()> {
        let output_path = output_path.as_ref();
        info!("=== Writing COPC file: {:?} ===", output_path);
        let builder = self.inner.builder.as_ref().unwrap();
        let node_keys = self.inner.node_keys.as_ref().unwrap();
        writer::write_copc(output_path, builder, node_keys, &self.inner.config)?;
        info!("Done.");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Expand a single input path into a list of LAZ/LAS files.
/// If `raw` is a directory, all `.laz`/`.las` files in it are returned (sorted).
/// If `raw` is a file, it is returned as-is.
pub fn collect_input_files(raw: PathBuf) -> Result<Vec<PathBuf>> {
    if raw.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&raw)
            .map_err(|e| anyhow::anyhow!("Cannot read directory {:?}: {}", raw, e))?
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
        if files.is_empty() {
            return Err(Error::NoInputFiles { path: raw });
        }
        Ok(files)
    } else {
        Ok(vec![raw])
    }
}
