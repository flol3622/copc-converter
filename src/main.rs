use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
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

    /// Progress output format: "bar" (default, interactive), "plain" (log lines),
    /// or "json" (NDJSON, one JSON object per line)
    #[arg(long, value_enum, default_value_t = ProgressMode::Bar)]
    progress: ProgressMode,
}

#[derive(Debug, Clone, ValueEnum)]
enum ProgressMode {
    /// Interactive progress bar (default)
    Bar,
    /// Plain text log lines
    Plain,
    /// NDJSON — one JSON object per line
    Json,
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

const TOTAL_STEPS: u32 = 5;

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------------------
// Bar progress (interactive terminal)
// ---------------------------------------------------------------------------

struct BarProgress {
    bar: Mutex<Option<ProgressBar>>,
    step: std::sync::atomic::AtomicU32,
    stage_prefix: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
}

impl BarProgress {
    fn new() -> Self {
        Self {
            bar: Mutex::new(None),
            step: std::sync::atomic::AtomicU32::new(0),
            stage_prefix: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl ProgressObserver for BarProgress {
    fn on_progress(&self, event: ProgressEvent) {
        let mut bar = self.bar.lock().unwrap();
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let prefix = format!("[{step}/{TOTAL_STEPS}] {name}");
                *self.stage_prefix.lock().unwrap() = prefix.clone();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                let pb = if total > 0 {
                    let pb = ProgressBar::new(total);
                    pb.set_style(
                        ProgressStyle::with_template("{msg} [{bar:40}] ({eta})")
                            .unwrap()
                            .progress_chars("=> "),
                    );
                    pb.set_message(format!("{prefix} 0/{}", human_count(total)));
                    pb
                } else {
                    let pb = ProgressBar::new_spinner();
                    pb.set_style(ProgressStyle::with_template("{msg} {spinner}").unwrap());
                    pb.set_message(prefix);
                    pb
                };
                *bar = Some(pb);
            }
            ProgressEvent::StageProgress { done } => {
                if let Some(ref pb) = *bar {
                    pb.set_position(done);
                    let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                    let prefix = self.stage_prefix.lock().unwrap().clone();
                    pb.set_message(format!(
                        "{prefix} {}/{}",
                        human_count(done),
                        human_count(total)
                    ));
                }
            }
            ProgressEvent::StageDone => {
                if let Some(pb) = bar.take() {
                    let prefix = self.stage_prefix.lock().unwrap().clone();
                    pb.finish_and_clear();
                    eprintln!("{prefix} done");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plain progress (log-friendly text lines on stderr)
// ---------------------------------------------------------------------------

struct PlainProgress {
    step: std::sync::atomic::AtomicU32,
    stage_name: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    last_percent: std::sync::atomic::AtomicU32,
}

impl PlainProgress {
    fn new() -> Self {
        Self {
            step: std::sync::atomic::AtomicU32::new(0),
            stage_name: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            last_percent: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl ProgressObserver for PlainProgress {
    fn on_progress(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                *self.stage_name.lock().unwrap() = name.to_string();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                self.last_percent
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                if total > 0 {
                    eprintln!("[{step}/{TOTAL_STEPS}] {name} started ({} units)", human_count(total));
                } else {
                    eprintln!("[{step}/{TOTAL_STEPS}] {name} started");
                }
            }
            ProgressEvent::StageProgress { done } => {
                let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                if total == 0 {
                    return;
                }
                let pct = (done as f64 / total as f64 * 100.0) as u32;
                // Log every 10%
                let bucket = pct / 10 * 10;
                let prev = self.last_percent.load(std::sync::atomic::Ordering::Relaxed);
                if bucket > prev
                    && self
                        .last_percent
                        .compare_exchange(
                            prev,
                            bucket,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                    let name = self.stage_name.lock().unwrap().clone();
                    eprintln!(
                        "[{step}/{TOTAL_STEPS}] {name} {bucket}% ({}/{})",
                        human_count(done),
                        human_count(total),
                    );
                }
            }
            ProgressEvent::StageDone => {
                let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                let name = self.stage_name.lock().unwrap().clone();
                eprintln!("[{step}/{TOTAL_STEPS}] {name} done");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON progress (NDJSON on stdout)
// ---------------------------------------------------------------------------

struct JsonProgress {
    step: std::sync::atomic::AtomicU32,
    stage_name: Mutex<String>,
    stage_total: std::sync::atomic::AtomicU64,
    last_percent: std::sync::atomic::AtomicU32,
}

impl JsonProgress {
    fn new() -> Self {
        Self {
            step: std::sync::atomic::AtomicU32::new(0),
            stage_name: Mutex::new(String::new()),
            stage_total: std::sync::atomic::AtomicU64::new(0),
            last_percent: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn emit(&self, value: &serde_json::Value) {
        // Write to stderr so it doesn't interfere with stdout data piping.
        // unwrap: serialization of our values cannot fail.
        eprintln!("{}", serde_json::to_string(value).unwrap());
    }
}

impl ProgressObserver for JsonProgress {
    fn on_progress(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::StageStart { name, total } => {
                let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                *self.stage_name.lock().unwrap() = name.to_string();
                self.stage_total
                    .store(total, std::sync::atomic::Ordering::Relaxed);
                self.last_percent
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                self.emit(&serde_json::json!({
                    "event": "stage_start",
                    "stage": name,
                    "step": step,
                    "total_steps": TOTAL_STEPS,
                    "total_units": total,
                }));
            }
            ProgressEvent::StageProgress { done } => {
                let total = self.stage_total.load(std::sync::atomic::Ordering::Relaxed);
                if total == 0 {
                    return;
                }
                let pct = (done as f64 / total as f64 * 100.0) as u32;
                let bucket = pct / 10 * 10;
                let prev = self.last_percent.load(std::sync::atomic::Ordering::Relaxed);
                if bucket > prev
                    && self
                        .last_percent
                        .compare_exchange(
                            prev,
                            bucket,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                    let name = self.stage_name.lock().unwrap().clone();
                    let percent = done as f64 / total as f64 * 100.0;
                    self.emit(&serde_json::json!({
                        "event": "stage_progress",
                        "stage": name,
                        "step": step,
                        "total_steps": TOTAL_STEPS,
                        "done": done,
                        "total": total,
                        "percent": (percent * 10.0).round() / 10.0,
                    }));
                }
            }
            ProgressEvent::StageDone => {
                let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
                let name = self.stage_name.lock().unwrap().clone();
                self.emit(&serde_json::json!({
                    "event": "stage_done",
                    "stage": name,
                    "step": step,
                    "total_steps": TOTAL_STEPS,
                }));
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

    let progress: std::sync::Arc<dyn ProgressObserver> = match args.progress {
        ProgressMode::Bar => std::sync::Arc::new(BarProgress::new()),
        ProgressMode::Plain => std::sync::Arc::new(PlainProgress::new()),
        ProgressMode::Json => std::sync::Arc::new(JsonProgress::new()),
    };

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
