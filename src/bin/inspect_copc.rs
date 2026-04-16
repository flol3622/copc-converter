//! Inspect a COPC file's structure, or compare two COPC files side-by-side.
//! When the file has a temporal index EVLR, its stats are appended.
//!
//! Usage:
//!   inspect_copc <file_or_url>
//!   inspect_copc <file_or_url> --compare <other_file_or_url>

use copc_converter::tools::Source;
use copc_streaming::{ByteSource, CopcStreamingReader};
use copc_temporal::TemporalCache;
use std::collections::HashMap;

fn human_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.2} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.2} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

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

struct CopcStats {
    label: String,
    file_size: u64,
    total_points: u64,
    point_format: u8,
    point_record_len: u16,
    node_count: usize,
    max_depth: i32,
    nodes_per_level: Vec<(i32, usize, u64, u64)>,
    copc_info_str: String,
}

async fn gather_stats(path: &str, label: &str) -> anyhow::Result<CopcStats> {
    eprintln!("[{label}] Opening...");
    let source = Source::from_arg(path)?;
    let file_size = copc_streaming::ByteSource::size(&source)
        .await?
        .unwrap_or(0);
    let mut reader = CopcStreamingReader::open(source).await?;

    let header = reader.header();
    let las_h = header.las_header();
    let copc_info = header.copc_info();
    let point_format = las_h.point_format().to_u8().unwrap_or(0);
    let point_record_len = las_h.point_format().len();
    let total_points = las_h.number_of_points();

    let info_str = format!(
        "center=({:.2}, {:.2}, {:.2}), halfsize={:.2}, spacing={:.6}, gps=[{:.2}, {:.2}]",
        copc_info.center[0],
        copc_info.center[1],
        copc_info.center[2],
        copc_info.halfsize,
        copc_info.spacing,
        copc_info.gpstime_minimum,
        copc_info.gpstime_maximum,
    );

    eprintln!("[{label}] Loading hierarchy...");
    reader.load_all_hierarchy().await?;

    let mut level_map: HashMap<i32, (usize, u64, u64)> = HashMap::new();
    let mut max_depth: i32 = 0;
    let mut node_count = 0;

    for (key, entry) in reader.entries() {
        node_count += 1;
        if key.level > max_depth {
            max_depth = key.level;
        }
        let e = level_map.entry(key.level).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += entry.point_count as u64;
        e.2 += entry.byte_size as u64;
    }

    let mut nodes_per_level: Vec<(i32, usize, u64, u64)> = level_map
        .into_iter()
        .map(|(l, (c, p, b))| (l, c, p, b))
        .collect();
    nodes_per_level.sort_by_key(|(l, _, _, _)| *l);

    eprintln!("[{label}] Done — {node_count} nodes, {total_points} points");

    Ok(CopcStats {
        label: label.to_string(),
        file_size,
        total_points,
        point_format,
        point_record_len,
        node_count,
        max_depth,
        nodes_per_level,
        copc_info_str: info_str,
    })
}

fn print_stats(s: &CopcStats) {
    let total_compressed: u64 = s.nodes_per_level.iter().map(|(_, _, _, c)| *c).sum();
    println!("=== {} ===", s.label);
    if s.file_size > 0 {
        println!("  File size:        {}", human_bytes(s.file_size));
    } else {
        println!(
            "  Compressed total: {} (point data only)",
            human_bytes(total_compressed)
        );
    }
    println!("  Total points:     {}", s.total_points);
    println!("  Point format:     {}", s.point_format);
    println!("  Point record len: {}", s.point_record_len);
    println!("  Nodes:            {}", s.node_count);
    println!("  Max depth:        {}", s.max_depth);
    println!("  COPC info:        {}", s.copc_info_str);
    println!();
    println!(
        "  {:>5}  {:>8}  {:>14}  {:>14}  {:>10}  {:>12}",
        "Level", "Nodes", "Points", "Compressed", "Ratio", "Avg pts/node"
    );
    println!("  {}", "-".repeat(72));
    for (level, count, points, compressed) in &s.nodes_per_level {
        let uncompressed = *points * s.point_record_len as u64;
        let ratio = if *compressed > 0 {
            uncompressed as f64 / *compressed as f64
        } else {
            0.0
        };
        let avg = if *count > 0 {
            *points / *count as u64
        } else {
            0
        };
        println!(
            "  {:>5}  {:>8}  {:>14}  {:>14}  {:>9.1}x  {:>12}",
            level,
            count,
            points,
            human_bytes(*compressed),
            ratio,
            avg
        );
    }
    println!();
}

fn print_comparison(a: &CopcStats, b: &CopcStats) {
    println!("=== Comparison ===");
    println!(
        "  File size:    {} vs {} ({:.1}x)",
        human_bytes(a.file_size),
        human_bytes(b.file_size),
        b.file_size as f64 / a.file_size.max(1) as f64,
    );
    println!("  Total points: {} vs {}", a.total_points, b.total_points);
    println!("  Nodes:        {} vs {}", a.node_count, b.node_count);
    println!("  Max depth:    {} vs {}", a.max_depth, b.max_depth);

    let max_level = a.max_depth.max(b.max_depth);
    println!();
    println!(
        "  {:>5}  {:>10} {:>10}  {:>12} {:>12}  {:>8} {:>8}  {:>10} {:>10}",
        "Level",
        "Nodes-A",
        "Nodes-B",
        "Compress-A",
        "Compress-B",
        "Ratio-A",
        "Ratio-B",
        "Pts-A",
        "Pts-B"
    );
    println!("  {}", "-".repeat(105));
    for level in 0..=max_level {
        let a_entry = a.nodes_per_level.iter().find(|(l, _, _, _)| *l == level);
        let b_entry = b.nodes_per_level.iter().find(|(l, _, _, _)| *l == level);
        let (a_nodes, a_pts, a_comp) = a_entry.map_or((0, 0, 0), |(_, c, p, b)| (*c, *p, *b));
        let (b_nodes, b_pts, b_comp) = b_entry.map_or((0, 0, 0), |(_, c, p, b)| (*c, *p, *b));
        let a_uncomp = a_pts * a.point_record_len as u64;
        let b_uncomp = b_pts * b.point_record_len as u64;
        let a_ratio = if a_comp > 0 {
            format!("{:.1}x", a_uncomp as f64 / a_comp as f64)
        } else {
            "-".into()
        };
        let b_ratio = if b_comp > 0 {
            format!("{:.1}x", b_uncomp as f64 / b_comp as f64)
        } else {
            "-".into()
        };
        println!(
            "  {:>5}  {:>10} {:>10}  {:>12} {:>12}  {:>8} {:>8}  {:>10} {:>10}",
            level,
            a_nodes,
            b_nodes,
            human_bytes(a_comp),
            human_bytes(b_comp),
            a_ratio,
            b_ratio,
            a_pts,
            b_pts,
        );
    }
    println!();
}

/// Print temporal-index stats for a COPC file when the EVLR is present.
/// No-op when the file lacks a temporal index.
async fn print_temporal_stats(path: &str, label: &str) -> anyhow::Result<()> {
    let source = Source::from_arg(path)?;
    let mut reader = CopcStreamingReader::open(source).await?;

    let temporal = TemporalCache::from_reader(&reader).await?;
    let mut temporal = match temporal {
        Some(t) => t,
        None => return Ok(()),
    };

    println!("=== {label} — Temporal Index ===");
    let copc_info = reader.header().copc_info();
    println!(
        "  GPS time range: [{}, {}]  (duration: {})",
        format_gps_time(copc_info.gpstime_minimum),
        format_gps_time(copc_info.gpstime_maximum),
        format_duration(copc_info.gpstime_maximum - copc_info.gpstime_minimum),
    );

    let header = temporal.header().unwrap();
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
    let source = Source::from_arg(path)?;

    let mut temporal_data_offset: Option<u64> = None;
    let mut temporal_data_length: Option<u64> = None;
    let mut pos = evlr_offset;
    for _ in 0..evlr_count {
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
        "[{label}] Fetching temporal EVLR payload ({:.2} MB)...",
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

    temporal
        .load_all_pages(&mem_source)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load temporal pages: {e}"))?;

    reader.load_all_hierarchy().await?;

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

    println!("  Per-level temporal coverage:");
    println!(
        "    {:>5}  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}",
        "Level", "Nodes", "Samples", "From start", "To", "Span"
    );
    println!("    {}", "-".repeat(65));
    for level in &levels {
        let (nodes, t_min, t_max, samples) = level_stats[level];
        println!(
            "    {:>5}  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}",
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
            "  Sample distribution over time (when data was captured):",
            &buckets,
            40,
        );
    }

    let mut sample_counts: Vec<usize> = temporal.iter().map(|(_, e)| e.samples().len()).collect();
    sample_counts.sort();
    if !sample_counts.is_empty() {
        let min_s = sample_counts[0];
        let max_s = *sample_counts.last().unwrap();
        let median_s = sample_counts[sample_counts.len() / 2];
        let avg_s = sample_counts.iter().sum::<usize>() as f64 / sample_counts.len() as f64;
        println!("  Sample density per node:");
        println!("    Min: {min_s}, Max: {max_s}, Median: {median_s}, Avg: {avg_s:.1}");
        println!(
            "    Total nodes with temporal data: {}",
            sample_counts.len()
        );

        let hier_data_nodes = reader.entries().filter(|(_, e)| e.point_count > 0).count();
        let temporal_nodes = sample_counts.len();
        if hier_data_nodes != temporal_nodes {
            println!(
                "    Warning: {} hierarchy data nodes vs {} temporal entries (diff: {})",
                hier_data_nodes,
                temporal_nodes,
                (hier_data_nodes as i64 - temporal_nodes as i64).abs(),
            );
        }
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let (input, compare_with) = match args.len() {
        2 => (args[1].as_str(), None),
        4 if args[2] == "--compare" => (args[1].as_str(), Some(args[3].as_str())),
        _ => {
            eprintln!("Usage: inspect_copc <file_or_url> [--compare <other>]");
            std::process::exit(1);
        }
    };

    if let Some(other) = compare_with {
        let (stats_a, stats_b) = tokio::join!(gather_stats(input, "A"), gather_stats(other, "B"));
        let stats_a = stats_a?;
        let stats_b = stats_b?;
        print_stats(&stats_a);
        print_stats(&stats_b);
        print_comparison(&stats_a, &stats_b);
        print_temporal_stats(input, "A").await?;
        print_temporal_stats(other, "B").await?;
    } else {
        let stats = gather_stats(input, input).await?;
        print_stats(&stats);
        print_temporal_stats(input, input).await?;
    }

    Ok(())
}
