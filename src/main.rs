use anyhow::Result;
use clap::Parser;
use copc_converter::{Pipeline, PipelineConfig, collect_input_files, parse_memory_limit};
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

    Pipeline::scan(&input_files, config)?
        .validate()?
        .distribute()?
        .build()?
        .write(&args.output)?;

    Ok(())
}
