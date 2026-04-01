//! Inspect the temporal index EVLR of a COPC file.
//!
//! Prints time range, per-level stats, and a GPS time histogram.
//!
//! Usage:
//!   cargo run --release --features tools --bin inspect_temporal -- <url>

use copc_converter::tools::Source;
use copc_streaming::{ByteSource, CopcStreamingReader};
use copc_temporal::TemporalCache;
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
        println!("  {label:>20} | {bar}");
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

    // Per the spec, all temporal pages reside within the temporal EVLR payload.
    // Scan EVLR headers sequentially to find the temporal one, then fetch its
    // entire payload in a single request.
    let evlr_offset = reader.evlr_offset();
    let evlr_count = reader.evlr_count();
    let source = Source::from_arg(&args[1])?;

    let mut temporal_data_offset: Option<u64> = None;
    let mut temporal_data_length: Option<u64> = None;
    let mut pos = evlr_offset;
    for _ in 0..evlr_count {
        // Fetch just this EVLR's 60-byte header
        let hdr = source.read_range(pos, 60).await?;
        let user_id = std::str::from_utf8(&hdr[2..18])
            .unwrap_or("")
            .trim_end_matches('\0');
        let record_id = u16::from_le_bytes([hdr[18], hdr[19]]);
        let data_length = u64::from_le_bytes(hdr[20..28].try_into().unwrap());

        if user_id == "copc_temporal" && record_id == 1000 {
            temporal_data_offset = Some(pos + 60);
            temporal_data_length = Some(data_length);
            break;
        }
        pos += 60 + data_length;
    }

    let temporal_data_offset =
        temporal_data_offset.ok_or_else(|| anyhow::anyhow!("Temporal EVLR not found"))?;
    let temporal_data_length = temporal_data_length.unwrap();

    eprintln!(
        "Fetching temporal EVLR payload ({:.2} MB)...",
        temporal_data_length as f64 / 1_048_576.0,
    );
    let evlr_payload = source
        .read_range(temporal_data_offset, temporal_data_length)
        .await?;

    // Wrap as an in-memory ByteSource with correct absolute file offsets.
    struct OffsetSource {
        data: Vec<u8>,
        base_offset: u64,
    }
    impl ByteSource for OffsetSource {
        async fn read_range(
            &self,
            offset: u64,
            length: u64,
        ) -> Result<Vec<u8>, copc_streaming::CopcError> {
            let start = (offset - self.base_offset) as usize;
            let end = start + length as usize;
            if end > self.data.len() {
                return Err(copc_streaming::CopcError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("offset {offset} + len {length} exceeds EVLR buffer"),
                )));
            }
            Ok(self.data[start..end].to_vec())
        }
        async fn size(&self) -> Result<Option<u64>, copc_streaming::CopcError> {
            Ok(Some(self.base_offset + self.data.len() as u64))
        }
    }
    let mem_source = OffsetSource {
        data: evlr_payload,
        base_offset: temporal_data_offset,
    };

    eprintln!("Parsing temporal pages from memory...");
    temporal
        .load_all_pages(&mem_source)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load temporal pages: {e}"))?;

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
        "  {:>5}  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}",
        "Level", "Nodes", "Samples", "From start", "To", "Span"
    );
    println!("  {}", "-".repeat(65));
    for level in &levels {
        let (nodes, t_min, t_max, samples) = level_stats[level];
        println!(
            "  {:>5}  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}",
            level,
            nodes,
            samples,
            format_duration(t_min - global_min),
            format_duration(t_max - global_min),
            format_duration(t_max - t_min),
        );
    }
    println!();

    // Time histogram — bin actual sample timestamps, not node overlap.
    // This shows when data was actually captured (e.g. reveals gaps at night).
    let num_buckets = 20;
    let range = global_max - global_min;
    if range > 0.0 {
        let bucket_width = range / num_buckets as f64;
        let mut counts = vec![0u64; num_buckets];

        for (_key, entry) in temporal.iter() {
            for sample in entry.samples() {
                let idx = ((sample.0 - global_min) / bucket_width) as usize;
                let idx = idx.min(num_buckets - 1);
                counts[idx] += 1;
            }
        }

        let buckets: Vec<(String, u64)> = (0..num_buckets)
            .map(|i| {
                let rel_start = i as f64 * bucket_width;
                let rel_end = rel_start + bucket_width;
                (
                    format!(
                        "{} - {}",
                        format_duration(rel_start),
                        format_duration(rel_end)
                    ),
                    counts[i],
                )
            })
            .collect();

        print_histogram(
            "Sample distribution over time (when data was captured):",
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
