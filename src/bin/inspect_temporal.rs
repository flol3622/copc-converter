//! Inspect the temporal index EVLR of a COPC file.
//!
//! Prints time range, per-level stats, and a GPS time histogram.
//!
//! Usage:
//!   cargo run --release --features tools --bin inspect_temporal -- <url>

use copc_converter::tools::Source;
use copc_streaming::CopcStreamingReader;
use copc_temporal::{GpsTime, TemporalCache};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GPS time formatting
// ---------------------------------------------------------------------------

fn format_gps_time(t: f64) -> String {
    format!("{:.2}", t)
}

fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1}m", seconds / 60.0)
    } else if seconds < 86400.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else {
        format!("{:.1}d", seconds / 86400.0)
    }
}

fn print_histogram(title: &str, buckets: &[(String, u64)], max_width: usize) {
    let max_val = buckets.iter().map(|(_, v)| *v).max().unwrap_or(1);
    println!("{title}");
    for (label, val) in buckets {
        let bar_len = if max_val > 0 {
            (*val as f64 / max_val as f64 * max_width as f64) as usize
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        println!("  {label:>20} | {bar:<max_width$} {val}");
    }
    println!();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: inspect_temporal <url>");
        std::process::exit(1);
    }

    let source = Source::from_arg(&args[1])?;
    eprintln!("Opening COPC file...");
    let mut reader = CopcStreamingReader::open(source).await?;

    let copc_info = reader.header().copc_info();
    println!(
        "COPC GPS time range: [{}, {}]",
        format_gps_time(copc_info.gpstime_minimum),
        format_gps_time(copc_info.gpstime_maximum),
    );
    println!(
        "  Duration: {}",
        format_duration(copc_info.gpstime_maximum - copc_info.gpstime_minimum)
    );
    println!();

    eprintln!("Loading temporal index...");
    let temporal = TemporalCache::from_reader(&reader).await?;
    let mut temporal = match temporal {
        Some(t) => t,
        None => {
            eprintln!("No temporal index found in this COPC file.");
            std::process::exit(1);
        }
    };

    let header = temporal.header().unwrap();
    println!("Temporal Index Header:");
    println!("  Version:    {}", header.version);
    println!(
        "  Stride:     {} (every {}th point sampled)",
        header.stride, header.stride
    );
    println!("  Node count: {}", header.node_count);
    println!("  Page count: {}", header.page_count);
    println!();

    eprintln!("Loading all temporal pages...");
    let source = Source::from_arg(&args[1])?;
    temporal.load_all_pages(&source).await?;

    eprintln!("Loading hierarchy...");
    reader.load_all_hierarchy().await?;

    // Per-level stats
    let mut level_stats: HashMap<i32, (usize, f64, f64, usize)> = HashMap::new();
    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;

    for (key, entry) in temporal.iter() {
        let samples = entry.samples();
        let (t_min, t_max) = entry.time_range();

        if t_min.0 < global_min {
            global_min = t_min.0;
        }
        if t_max.0 > global_max {
            global_max = t_max.0;
        }

        let s = level_stats
            .entry(key.level)
            .or_insert((0, f64::MAX, f64::MIN, 0));
        s.0 += 1;
        if t_min.0 < s.1 {
            s.1 = t_min.0;
        }
        if t_max.0 > s.2 {
            s.2 = t_max.0;
        }
        s.3 += samples.len();
    }

    let mut levels: Vec<i32> = level_stats.keys().copied().collect();
    levels.sort();

    println!("Per-level temporal coverage:");
    println!(
        "  {:>5}  {:>8}  {:>10}  {:>16}  {:>16}  {:>10}",
        "Level", "Nodes", "Samples", "Time min", "Time max", "Span"
    );
    println!("  {}", "-".repeat(75));
    for level in &levels {
        let (nodes, t_min, t_max, samples) = level_stats[level];
        println!(
            "  {:>5}  {:>8}  {:>10}  {:>16}  {:>16}  {:>10}",
            level,
            nodes,
            samples,
            format_gps_time(t_min),
            format_gps_time(t_max),
            format_duration(t_max - t_min),
        );
    }
    println!();

    // Time histogram
    let num_buckets = 20;
    let range = global_max - global_min;
    if range > 0.0 {
        let bucket_width = range / num_buckets as f64;
        let mut buckets: Vec<(String, u64)> = Vec::new();

        for i in 0..num_buckets {
            let b_start = global_min + i as f64 * bucket_width;
            let b_end = b_start + bucket_width;
            let count = temporal
                .nodes_in_range(GpsTime(b_start), GpsTime(b_end))
                .len() as u64;
            buckets.push((
                format!("{} - {}", format_gps_time(b_start), format_gps_time(b_end),),
                count,
            ));
        }

        print_histogram(
            "Node overlap by time bucket (nodes active in each time window):",
            &buckets,
            40,
        );
    }

    // Sample density
    let mut sample_counts: Vec<usize> = temporal.iter().map(|(_, e)| e.samples().len()).collect();
    sample_counts.sort();
    if !sample_counts.is_empty() {
        let min_s = sample_counts[0];
        let max_s = *sample_counts.last().unwrap();
        let median_s = sample_counts[sample_counts.len() / 2];
        let avg_s = sample_counts.iter().sum::<usize>() as f64 / sample_counts.len() as f64;
        println!("Sample density per node:");
        println!("  Min: {min_s}, Max: {max_s}, Median: {median_s}, Avg: {avg_s:.1}");
        println!("  Total nodes with temporal data: {}", sample_counts.len());

        let hier_data_nodes = reader.entries().filter(|(_, e)| e.point_count > 0).count();
        let temporal_nodes = sample_counts.len();
        if hier_data_nodes != temporal_nodes {
            println!(
                "  Warning: {} hierarchy data nodes vs {} temporal entries (diff: {})",
                hier_data_nodes,
                temporal_nodes,
                (hier_data_nodes as i64 - temporal_nodes as i64).abs(),
            );
        }
        println!();
    }

    Ok(())
}
