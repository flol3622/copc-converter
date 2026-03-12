mod copc_types;
mod octree;
mod writer;

use anyhow::{Context, Result};
use clap::Parser;
use log::info;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert LAZ files to a COPC file")]
struct Args {
    /// One or more input LAZ/LAS files, or a single directory containing them
    #[arg(required = true, num_args = 1..)]
    input: Vec<PathBuf>,

    /// Output COPC file path
    #[arg(short, long)]
    output: PathBuf,
}

fn collect_input_files(raw: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if raw.len() == 1 && raw[0].is_dir() {
        let dir = &raw[0];
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read directory {:?}", dir))?
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
        anyhow::ensure!(!files.is_empty(), "No LAZ/LAS files found in {:?}", dir);
        Ok(files)
    } else {
        Ok(raw)
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let input_files = collect_input_files(args.input)?;

    info!(
        "=== Pass 1: scanning {} input file(s) ===",
        input_files.len()
    );
    let builder = octree::OctreeBuilder::scan(&input_files)?;

    info!("=== Pass 2: distributing points to leaf voxels ===");
    builder.distribute(&input_files)?;

    info!("=== Building octree node map ===");
    let node_map = builder.build_node_map()?;

    info!("=== Writing COPC file: {:?} ===", args.output);
    writer::write_copc(&args.output, &builder, &node_map)?;

    builder.cleanup();
    info!("Done.");
    Ok(())
}
