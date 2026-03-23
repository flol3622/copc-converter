use anyhow::{Context, Result};
use clap::Parser;
use copc_converter::{
    Pipeline, PipelineConfig, ProgressEvent, ProgressObserver, collect_input_files,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Mutex;

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

/// Parse a human-readable size string into bytes.
fn parse_memory_limit(s: &str) -> Result<u64> {
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

struct CliProgress {
    bar: Mutex<Option<ProgressBar>>,
}

impl CliProgress {
    fn new() -> Self {
        Self {
            bar: Mutex::new(None),
        }
    }
}

impl ProgressObserver for CliProgress {
    fn on_progress(&self, event: ProgressEvent) {
        let mut bar = self.bar.lock().unwrap();
        match event {
            ProgressEvent::StageStart { name, total } => {
                let pb = if total > 0 {
                    let pb = ProgressBar::new(total);
                    pb.set_style(
                        ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len} ({eta})")
                            .unwrap()
                            .progress_chars("=> "),
                    );
                    pb
                } else {
                    let pb = ProgressBar::new_spinner();
                    pb.set_style(ProgressStyle::with_template("{msg} {spinner}").unwrap());
                    pb
                };
                pb.set_message(name.to_string());
                *bar = Some(pb);
            }
            ProgressEvent::StageProgress { done } => {
                if let Some(ref pb) = *bar {
                    pb.set_position(done);
                }
            }
            ProgressEvent::StageDone => {
                if let Some(pb) = bar.take() {
                    pb.finish_and_clear();
                }
            }
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let input_files = collect_input_files(args.input)?;

    let raw_limit = parse_memory_limit(&args.memory_limit)?;
    let memory_budget = (raw_limit as f64 * MEMORY_SAFETY_FACTOR) as u64;

    let progress = std::sync::Arc::new(CliProgress::new());

    let config = PipelineConfig {
        memory_budget,
        temp_dir: args.temp_dir,
        temporal_index: args.temporal_index,
        temporal_stride: args.temporal_stride,
        progress: Some(progress),
    };

    Pipeline::scan(&input_files, config)?
        .validate()?
        .distribute()?
        .build()?
        .write(&args.output)?;

    Ok(())
}
