//! Compare two COPC files side-by-side: hierarchy depth, node counts,
//! chunk sizes, point counts, and compression ratios.
//!
//! Parses headers and hierarchy directly with minimal HTTP requests
//! (no page-by-page loading).
//!
//! Usage:
//!   cargo run --release --bin compare_copc -- <url_a> <url_b>

use byteorder::{LittleEndian, ReadBytesExt};
use reqwest::Client;
use std::collections::HashMap;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

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

async fn fetch_range(
    client: &Client,
    url: &str,
    offset: u64,
    length: u64,
) -> anyhow::Result<Vec<u8>> {
    let end = offset + length - 1;
    let resp = client
        .get(url)
        .header("Range", format!("bytes={offset}-{end}"))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} fetching range {offset}-{end}", resp.status());
    }
    Ok(resp.bytes().await?.to_vec())
}

// ---------------------------------------------------------------------------
// Minimal COPC parsing — just enough to read headers + hierarchy
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct CopcHeader {
    point_format: u8,
    point_record_len: u16,
    total_points: u64,
    scale: [f64; 3],
    offset: [f64; 3],
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    evlr_offset: u64,
    num_vlrs: u32,
    offset_to_point_data: u32,
}

struct CopcInfoVlr {
    center: [f64; 3],
    halfsize: f64,
    spacing: f64,
    root_hier_offset: u64,
    root_hier_size: u64,
    gpstime_min: f64,
    gpstime_max: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HierarchyEntry {
    level: i32,
    x: i32,
    y: i32,
    z: i32,
    offset: u64,
    byte_size: i32,
    point_count: i32,
}

fn parse_las_header(data: &[u8]) -> anyhow::Result<CopcHeader> {
    let mut r = Cursor::new(data);

    // Offset 96: offset to point data
    r.set_position(96);
    let offset_to_point_data = r.read_u32::<LittleEndian>()?;
    // Offset 100: number of VLRs
    let num_vlrs = r.read_u32::<LittleEndian>()?;
    // Offset 104: point format (bit 7 = compression flag)
    let raw_format = r.read_u8()?;
    let point_format = raw_format & 0x3F;
    // Offset 105: point record length
    let point_record_len = r.read_u16::<LittleEndian>()?;

    // Offset 131: scale X/Y/Z
    r.set_position(131);
    let scale_x = r.read_f64::<LittleEndian>()?;
    let scale_y = r.read_f64::<LittleEndian>()?;
    let scale_z = r.read_f64::<LittleEndian>()?;
    let offset_x = r.read_f64::<LittleEndian>()?;
    let offset_y = r.read_f64::<LittleEndian>()?;
    let offset_z = r.read_f64::<LittleEndian>()?;
    // Offset 179: max_x, min_x, max_y, min_y, max_z, min_z
    let max_x = r.read_f64::<LittleEndian>()?;
    let min_x = r.read_f64::<LittleEndian>()?;
    let max_y = r.read_f64::<LittleEndian>()?;
    let min_y = r.read_f64::<LittleEndian>()?;
    let max_z = r.read_f64::<LittleEndian>()?;
    let min_z = r.read_f64::<LittleEndian>()?;

    // Offset 227: waveform data (skip)
    // Offset 235: EVLR offset
    r.set_position(235);
    let evlr_offset = r.read_u64::<LittleEndian>()?;
    // Offset 243: num EVLRs (skip)
    r.read_u32::<LittleEndian>()?;
    // Offset 247: number of point records (u64, LAS 1.4)
    let total_points = r.read_u64::<LittleEndian>()?;

    Ok(CopcHeader {
        point_format,
        point_record_len,
        total_points,
        scale: [scale_x, scale_y, scale_z],
        offset: [offset_x, offset_y, offset_z],
        bounds_min: [min_x, min_y, min_z],
        bounds_max: [max_x, max_y, max_z],
        evlr_offset,
        num_vlrs,
        offset_to_point_data,
    })
}

fn parse_copc_info(data: &[u8]) -> anyhow::Result<CopcInfoVlr> {
    let mut r = Cursor::new(data);
    let cx = r.read_f64::<LittleEndian>()?;
    let cy = r.read_f64::<LittleEndian>()?;
    let cz = r.read_f64::<LittleEndian>()?;
    let halfsize = r.read_f64::<LittleEndian>()?;
    let spacing = r.read_f64::<LittleEndian>()?;
    let root_hier_offset = r.read_u64::<LittleEndian>()?;
    let root_hier_size = r.read_u64::<LittleEndian>()?;
    let gpstime_min = r.read_f64::<LittleEndian>()?;
    let gpstime_max = r.read_f64::<LittleEndian>()?;
    Ok(CopcInfoVlr {
        center: [cx, cy, cz],
        halfsize,
        spacing,
        root_hier_offset,
        root_hier_size,
        gpstime_min,
        gpstime_max,
    })
}

fn parse_hierarchy(data: &[u8]) -> Vec<HierarchyEntry> {
    let mut entries = Vec::new();
    let mut r = Cursor::new(data);
    while (r.position() as usize + 32) <= data.len() {
        let level = r.read_i32::<LittleEndian>().unwrap();
        let x = r.read_i32::<LittleEndian>().unwrap();
        let y = r.read_i32::<LittleEndian>().unwrap();
        let z = r.read_i32::<LittleEndian>().unwrap();
        let offset = r.read_u64::<LittleEndian>().unwrap();
        let byte_size = r.read_i32::<LittleEndian>().unwrap();
        let point_count = r.read_i32::<LittleEndian>().unwrap();
        entries.push(HierarchyEntry {
            level,
            x,
            y,
            z,
            offset,
            byte_size,
            point_count,
        });
    }
    entries
}

/// Fetch the entire hierarchy by fetching the root page, then any sub-pages
/// referenced by entries with point_count == -1 (page pointers).
/// Groups sub-page fetches into a single contiguous range request when possible.
async fn fetch_all_hierarchy(
    client: &Client,
    url: &str,
    root_offset: u64,
    root_size: u64,
    label: &str,
) -> anyhow::Result<Vec<HierarchyEntry>> {
    let mut all_entries: Vec<HierarchyEntry> = Vec::new();
    let mut pages_to_fetch: Vec<(u64, u64)> = vec![(root_offset, root_size)];

    let mut round = 0u32;
    while !pages_to_fetch.is_empty() {
        round += 1;

        // Sort by offset so we can try to merge into contiguous ranges
        pages_to_fetch.sort_by_key(|(o, _)| *o);

        // Merge adjacent/overlapping ranges
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (offset, size) in &pages_to_fetch {
            if let Some(last) = merged.last_mut() {
                let last_end = last.0 + last.1;
                if *offset <= last_end + 1024 {
                    // merge if gap <= 1KB
                    let new_end = (*offset + *size).max(last_end);
                    last.1 = new_end - last.0;
                    continue;
                }
            }
            merged.push((*offset, *size));
        }

        let total_bytes: u64 = merged.iter().map(|(_, s)| *s).sum();
        eprintln!(
            "[{label}]   hierarchy round {round}: fetching {} range(s), {} total",
            merged.len(),
            human_bytes(total_bytes),
        );

        // Fetch all merged ranges
        let mut page_data: Vec<(u64, Vec<u8>)> = Vec::new();
        for (offset, size) in &merged {
            let data = fetch_range(client, url, *offset, *size).await?;
            page_data.push((*offset, data));
        }

        // Parse all pages, extracting entries from the original (non-merged) ranges
        let mut next_pages: Vec<(u64, u64)> = Vec::new();
        for (page_offset, page_size) in &pages_to_fetch {
            // Find the merged range that contains this page
            let (range_offset, range_data) = page_data
                .iter()
                .find(|(o, d)| {
                    *o <= *page_offset && *page_offset + *page_size <= *o + d.len() as u64
                })
                .expect("page not found in fetched data");
            let start = (*page_offset - *range_offset) as usize;
            let end = start + *page_size as usize;
            let slice = &range_data[start..end];
            let entries = parse_hierarchy(slice);
            for e in entries {
                if e.point_count == -1 {
                    // Page pointer — need to fetch this sub-page
                    next_pages.push((e.offset, e.byte_size as u64));
                } else {
                    all_entries.push(e);
                }
            }
        }

        eprintln!(
            "[{label}]     parsed {} data entries, {} sub-pages to follow",
            all_entries.len(),
            next_pages.len(),
        );

        pages_to_fetch = next_pages;
    }

    Ok(all_entries)
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct CopcStats {
    label: String,
    file_size: u64,
    total_points: u64,
    point_format: u8,
    point_record_len: u16,
    node_count: usize,
    max_depth: i32,
    nodes_per_level: Vec<(i32, usize, u64, u64)>, // (level, count, total_points, total_compressed_bytes)
    copc_info_str: String,
}

async fn gather_stats(url: &str, label: &str) -> anyhow::Result<CopcStats> {
    let client = Client::new();

    // 1. Fetch header (first 64KB covers LAS header + VLRs in most cases)
    eprintln!("[{label}] Fetching header...");
    let header_data = fetch_range(&client, url, 0, 65536).await?;
    let header = parse_las_header(&header_data)?;

    // Get file size from a HEAD (may fail on some R2 configs, that's OK)
    let file_size = client
        .head(url)
        .send()
        .await
        .ok()
        .and_then(|r| {
            r.headers()
                .get("content-length")?
                .to_str()
                .ok()?
                .parse()
                .ok()
        })
        .unwrap_or(0u64);

    // 2. Parse COPC info VLR — it's the first VLR right after the 375-byte header
    // VLR header: 2 (reserved) + 16 (user_id) + 2 (record_id) + 2 (length) + 32 (description) = 54
    let copc_vlr_payload_start = 375 + 54;
    let copc_info =
        parse_copc_info(&header_data[copc_vlr_payload_start..copc_vlr_payload_start + 160])?;

    let info_str = format!(
        "center=({:.2}, {:.2}, {:.2}), halfsize={:.2}, spacing={:.6}, gps=[{:.2}, {:.2}]",
        copc_info.center[0],
        copc_info.center[1],
        copc_info.center[2],
        copc_info.halfsize,
        copc_info.spacing,
        copc_info.gpstime_min,
        copc_info.gpstime_max,
    );

    eprintln!(
        "[{label}] Header: format={}, record_len={}, points={}, evlr_offset={}",
        header.point_format, header.point_record_len, header.total_points, header.evlr_offset,
    );
    eprintln!(
        "[{label}] COPC info: hier_offset={}, hier_size={}",
        copc_info.root_hier_offset,
        human_bytes(copc_info.root_hier_size),
    );

    // 3. Fetch and parse the entire hierarchy
    eprintln!("[{label}] Loading hierarchy...");
    let entries = fetch_all_hierarchy(
        &client,
        url,
        copc_info.root_hier_offset,
        copc_info.root_hier_size,
        label,
    )
    .await?;

    // 4. Aggregate stats
    let mut level_map: HashMap<i32, (usize, u64, u64)> = HashMap::new();
    let mut max_depth: i32 = 0;

    for e in &entries {
        if e.level > max_depth {
            max_depth = e.level;
        }
        let s = level_map.entry(e.level).or_insert((0, 0, 0));
        s.0 += 1;
        s.1 += e.point_count.max(0) as u64;
        s.2 += e.byte_size.max(0) as u64;
    }

    let mut nodes_per_level: Vec<(i32, usize, u64, u64)> = level_map
        .into_iter()
        .map(|(l, (c, p, b))| (l, c, p, b))
        .collect();
    nodes_per_level.sort_by_key(|(l, _, _, _)| *l);

    eprintln!(
        "[{label}] Done — {} nodes, {} points",
        entries.len(),
        header.total_points
    );

    Ok(CopcStats {
        label: label.to_string(),
        file_size,
        total_points: header.total_points,
        point_format: header.point_format,
        point_record_len: header.point_record_len,
        node_count: entries.len(),
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
    if args.len() != 3 {
        eprintln!("Usage: compare_copc <url_a> <url_b>");
        std::process::exit(1);
    }

    let (stats_a, stats_b) = tokio::join!(
        gather_stats(&args[1], "A (untwine)"),
        gather_stats(&args[2], "B (rust)"),
    );
    let stats_a = stats_a?;
    let stats_b = stats_b?;

    print_stats(&stats_a);
    print_stats(&stats_b);
    print_comparison(&stats_a, &stats_b);

    Ok(())
}
