use anyhow::Result;
use clap::Parser;
use copc_converter::{PipelineConfig, collect_input_files, parse_memory_limit};
use log::info;
use std::path::PathBuf;

/// Maximum fraction of the stated memory limit to actually use.
const MEMORY_SAFETY_FACTOR: f64 = 0.75;

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert LAZ files to a COPC file")]
struct Args {
    /// Input LAZ/LAS file, or a directory containing them
    input: PathBuf,

    /// Output COPC file path
    output: PathBuf,

    /// Maximum memory budget (e.g. "16G", "8G", "4096M", "512M"). Default: "16G"
    #[arg(long, default_value = "16G")]
    memory_limit: String,

    /// Temp directory for intermediate files. Default: system temp
    #[arg(long)]
    temp_dir: Option<PathBuf>,

    /// Enable temporal index EVLR for GPS-time-based queries
    #[arg(long)]
    temporal_index: bool,

    /// Sampling stride for temporal index (every n-th point). Default: 1000
    #[arg(long, default_value_t = 1000)]
    temporal_stride: u32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let input_files = collect_input_files(args.input)?;

    let raw_limit = parse_memory_limit(&args.memory_limit)?;
    let memory_budget = (raw_limit as f64 * MEMORY_SAFETY_FACTOR) as u64;
    info!(
        "Memory limit: {} bytes (effective budget: {} bytes)",
        raw_limit, memory_budget
    );

    let config = PipelineConfig {
        memory_budget,
        temp_dir: args.temp_dir,
        temporal_index: args.temporal_index,
        temporal_stride: args.temporal_stride,
    };

    info!(
        "=== Pass 1: scanning {} input file(s) ===",
        input_files.len()
    );
    let scan_results = copc_converter::octree::OctreeBuilder::scan(&input_files)?;

    info!("=== Validating inputs ===");
    let validated =
        copc_converter::validate::validate(&input_files, &scan_results, config.temporal_index)?;

    let builder =
        copc_converter::octree::OctreeBuilder::from_scan(&scan_results, &validated, &config)?;

    info!("=== Pass 2: distributing points to leaf voxels ===");
    builder.distribute(&input_files, &config)?;

    info!("=== Building octree node map ===");
    let node_keys = builder.build_node_map(&config)?;

    info!("=== Writing COPC file: {:?} ===", args.output);
    copc_converter::writer::write_copc(&args.output, &builder, &node_keys, &config)?;

    drop(builder);
    info!("Done.");
    Ok(())
}
