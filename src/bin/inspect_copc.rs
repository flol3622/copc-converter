//! Inspect a COPC file's structure, or compare two COPC files side-by-side.
//!
//! Usage:
//!   inspect_copc <file_or_url>
//!   inspect_copc <file_or_url> --compare <other_file_or_url>

use copc_converter::tools::Source;
use copc_streaming::CopcStreamingReader;
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
    println!("=== {} ===", s.label);
    println!("  File size:        {}", human_bytes(s.file_size));
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
    } else {
        let stats = gather_stats(input, input).await?;
        print_stats(&stats);
    }

    Ok(())
}
