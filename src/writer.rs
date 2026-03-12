/// Write a COPC 1.0 file.
///
/// Layout
/// ------
///  [LAS 1.4 header]           375 bytes
///  [copc info VLR]            54 + 160 = 214 bytes
///  [laszip VLR]               54 + 46  = 100 bytes   (user_id "laszip encoded", record_id 22204)
///  --- offset_to_point_data = 689 ---
///  [i64 chunk-table offset]   8 bytes  (points to chunk table after all data)
///  [compressed chunk 0]       variable
///  [compressed chunk 1]       variable
///  ...
///  [LAZ chunk table]          variable (appended after data, referenced by the i64 above)
///  [copc hierarchy EVLR]      60 + n*32 bytes
///
/// The laz::LasZipCompressor is used for real LAZ 1.4 compression.
/// Chunks are written into an in-memory Cursor<Vec<u8>> so we can track
/// per-chunk cursor positions without buffering issues, then the buffer is
/// written to the file. The 8-byte chunk-table offset pointer that laz
/// writes as a cursor-relative value must be patched to an absolute file
/// position before writing to disk.
use crate::copc_types::{
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, VoxelKey, write_evlr, write_vlr,
};
use crate::octree::{OctreeBuilder, RawPoint};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use laz::{LasZipCompressor, LazVlrBuilder};
use log::info;
use std::collections::HashMap;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// LAS 1.4 format 7 raw byte size = 30 (base format 6) + 6 (RGB) = 36
// ---------------------------------------------------------------------------
const POINT_RECORD_LENGTH: u16 = 36;

/// Encode one point as LAS 1.4 format 7 raw bytes (36 bytes, little-endian).
fn encode_point_fmt7(rp: &RawPoint, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&rp.x.to_le_bytes()); // 4  X (scaled integer)
    buf.extend_from_slice(&rp.y.to_le_bytes()); // 4  Y
    buf.extend_from_slice(&rp.z.to_le_bytes()); // 4  Z
    buf.extend_from_slice(&rp.intensity.to_le_bytes()); // 2  Intensity
    // Return bit fields: bits 0-3 = return number, bits 4-7 = number of returns
    let return_byte = (rp.return_number & 0x0F) | ((rp.number_of_returns & 0x0F) << 4);
    buf.push(return_byte); // 1
    buf.push(0u8); // classification flags / scanner channel / scan dir / edge
    buf.push(rp.classification); // 1  Classification
    buf.push(rp.user_data); // 1  User data
    buf.extend_from_slice(&rp.scan_angle.to_le_bytes()); // 2  Scan angle (0.006° units, i16)
    buf.extend_from_slice(&rp.point_source_id.to_le_bytes()); // 2  Point source ID
    buf.extend_from_slice(&rp.gps_time.to_le_bytes()); // 8  GPS time
    buf.extend_from_slice(&rp.red.to_le_bytes()); // 2  Red
    buf.extend_from_slice(&rp.green.to_le_bytes()); // 2  Green
    buf.extend_from_slice(&rp.blue.to_le_bytes()); // 2  Blue
    // Total = 4+4+4+2+1+1+1+1+2+2+8+2+2+2 = 36 bytes ✓
}

/// Write a complete COPC file to `output_path`.
pub fn write_copc(
    output_path: &Path,
    builder: &OctreeBuilder,
    node_map: &HashMap<VoxelKey, Vec<RawPoint>>,
) -> Result<()> {
    // -----------------------------------------------------------------------
    // Scale / offset – taken from the builder (captured from the first input file)
    // -----------------------------------------------------------------------
    let scale_x = builder.scale_x;
    let scale_y = builder.scale_y;
    let scale_z = builder.scale_z;
    let offset_x = builder.offset_x;
    let offset_y = builder.offset_y;
    let offset_z = builder.offset_z;

    // -----------------------------------------------------------------------
    // Build the LAZ VLR (variable-size chunks, format 7 items)
    // -----------------------------------------------------------------------
    let laz_vlr = LazVlrBuilder::default()
        .with_point_format(7, 0)
        .context("LazVlrBuilder for format 7")?
        .with_variable_chunk_size()
        .build();

    let mut laz_vlr_payload: Vec<u8> = Vec::new();
    laz_vlr.write_to(&mut laz_vlr_payload)?;
    // Should be 46 bytes for format 7 (2 items: Point14 + RGB14)

    // -----------------------------------------------------------------------
    // File layout constants
    // -----------------------------------------------------------------------
    let copc_info_vlr_size: u32 = 54 + 160; // 214
    let laz_vlr_size: u32 = 54 + laz_vlr_payload.len() as u32; // 100
    let offset_to_point_data: u32 = 375 + copc_info_vlr_size + laz_vlr_size;
    // = 375 + 214 + 100 = 689

    let copc_info_payload_pos: u64 = 375 + 54; // byte position of copc info payload in file

    let b = &builder.bounds;

    // -----------------------------------------------------------------------
    // Write LAS 1.4 header manually (375 bytes)
    // -----------------------------------------------------------------------
    let file = std::fs::File::create(output_path)
        .with_context(|| format!("Cannot create {:?}", output_path))?;
    let mut w = std::io::BufWriter::new(file);

    w.write_all(b"LASF")?;
    w.write_u16::<LittleEndian>(0)?; // file source ID
    w.write_u16::<LittleEndian>(0x0001 | 0x0010)?; // global encoding: GPS standard + WKT
    w.write_all(&[0u8; 16])?; // project ID (GUID)
    w.write_u8(1)?; // version major
    w.write_u8(4)?; // version minor
    let mut sysid = [0u8; 32];
    b"copc_converter"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| sysid[i] = c);
    w.write_all(&sysid)?; // system identifier
    let mut gensoft = [0u8; 32];
    b"copc_converter 0.1"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| gensoft[i] = c);
    w.write_all(&gensoft)?; // generating software
    w.write_u16::<LittleEndian>(1)?; // file creation day
    w.write_u16::<LittleEndian>(2024)?; // file creation year
    w.write_u16::<LittleEndian>(375)?; // header size
    w.write_u32::<LittleEndian>(offset_to_point_data)?; // offset to point data
    w.write_u32::<LittleEndian>(2)?; // number of VLRs (copc info + laszip)
    w.write_u8(128 | 7)?; // point data format: 135 = LAZ-compressed format 7
    w.write_u16::<LittleEndian>(POINT_RECORD_LENGTH)?; // point data record length = 36
    w.write_u32::<LittleEndian>(0)?; // legacy point count
    for _ in 0..5 {
        w.write_u32::<LittleEndian>(0)?;
    } // legacy point counts by return
    w.write_f64::<LittleEndian>(scale_x)?;
    w.write_f64::<LittleEndian>(scale_y)?;
    w.write_f64::<LittleEndian>(scale_z)?;
    w.write_f64::<LittleEndian>(offset_x)?;
    w.write_f64::<LittleEndian>(offset_y)?;
    w.write_f64::<LittleEndian>(offset_z)?;
    w.write_f64::<LittleEndian>(b.max_x)?;
    w.write_f64::<LittleEndian>(b.min_x)?;
    w.write_f64::<LittleEndian>(b.max_y)?;
    w.write_f64::<LittleEndian>(b.min_y)?;
    w.write_f64::<LittleEndian>(b.max_z)?;
    w.write_f64::<LittleEndian>(b.min_z)?;
    w.write_u64::<LittleEndian>(0)?; // start of waveform data
    // start_of_first_EVLR (u64) at byte offset 235 – placeholder, patched below
    w.write_u64::<LittleEndian>(0)?;
    w.write_u32::<LittleEndian>(1)?; // number of EVLRs
    w.write_u64::<LittleEndian>(builder.total_points)?; // point count (u64)
    for _ in 0..15 {
        w.write_u64::<LittleEndian>(0)?;
    } // point counts by return

    // -----------------------------------------------------------------------
    // VLR 1: copc info (placeholder – patched at the end)
    // -----------------------------------------------------------------------
    let copc_info_placeholder = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
        root_hier_offset: 0,
        root_hier_size: 0,
        gpstime_minimum: 0.0,
        gpstime_maximum: 0.0,
    };
    let mut copc_info_buf = Vec::with_capacity(160);
    copc_info_placeholder.write(&mut copc_info_buf)?;
    write_vlr(&mut w, "copc", 1, "copc info", &copc_info_buf)?;

    // -----------------------------------------------------------------------
    // VLR 2: laszip VLR
    // -----------------------------------------------------------------------
    write_vlr(
        &mut w,
        "laszip encoded",
        22204,
        "laz variable chunks",
        &laz_vlr_payload,
    )?;

    // We are now at byte offset_to_point_data. Flush so the file is complete up to here.
    w.flush()?;

    // -----------------------------------------------------------------------
    // Compress point data into an in-memory buffer via LasZipCompressor
    // -----------------------------------------------------------------------
    // We use Cursor<Vec<u8>> so stream_position() is always accurate (no
    // buffering lag).  After done(), we patch the 8-byte chunk-table-offset
    // pointer from cursor-relative to file-absolute before writing to disk.

    // Collect nodes in BFS order.
    let ordered_keys: Vec<VoxelKey> = {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(VoxelKey::root());
        while let Some(k) = queue.pop_front() {
            if node_map.contains_key(&k) {
                result.push(k);
                for child in k.children() {
                    queue.push_back(child);
                }
            }
        }
        result
    };

    // Build the laz_vlr again (consumed by the compressor).
    let laz_vlr_for_compressor = LazVlrBuilder::default()
        .with_point_format(7, 0)
        .context("LazVlrBuilder (compressor)")?
        .with_variable_chunk_size()
        .build();

    let cursor = Cursor::new(Vec::<u8>::new());
    let mut compressor = LasZipCompressor::new(cursor, laz_vlr_for_compressor)
        .map_err(|e| anyhow::anyhow!("LasZipCompressor::new: {e}"))?;

    // Reserve the chunk-table offset slot now so that cursor positions we
    // query below are accurate (the slot is 8 bytes at cursor position 0).
    compressor
        .reserve_offset_to_chunk_table()
        .context("reserve_offset_to_chunk_table")?;

    let mut chunk_cursor_starts: Vec<u64> = Vec::with_capacity(ordered_keys.len());

    for (i, key) in ordered_keys.iter().enumerate() {
        let pts = node_map.get(key).unwrap();

        // Record the cursor position at the start of this chunk.
        let chunk_cursor_start = compressor.get_mut().stream_position()?;
        chunk_cursor_starts.push(chunk_cursor_start);

        // Encode all points for this node as raw format-7 bytes.
        let mut raw_bytes: Vec<u8> = Vec::with_capacity(POINT_RECORD_LENGTH as usize * pts.len());
        for rp in pts {
            encode_point_fmt7(rp, &mut raw_bytes);
        }

        compressor
            .compress_many(&raw_bytes)
            .context("compress_many")?;

        // For all chunks except the last, explicitly close the chunk so the
        // next node starts a fresh chunk.
        if i < ordered_keys.len() - 1 {
            compressor
                .finish_current_chunk()
                .context("finish_current_chunk")?;
        }
    }

    // Finalize the last chunk and write the chunk table to the cursor.
    compressor.done().context("compressor done")?;

    let mut cursor = compressor.into_inner();

    // Read the cursor-relative chunk-table offset written by laz at cursor[0..8].
    let cursor_bytes = cursor.get_mut();
    let cursor_chunk_table_pos = i64::from_le_bytes(cursor_bytes[0..8].try_into().unwrap());

    // Patch it to the absolute FILE position of the chunk table.
    let file_chunk_table_pos = offset_to_point_data as i64 + cursor_chunk_table_pos;
    cursor_bytes[0..8].copy_from_slice(&file_chunk_table_pos.to_le_bytes());

    // Write the entire point data buffer to the output file.
    let point_data = cursor.into_inner();
    w.write_all(&point_data)?;

    // -----------------------------------------------------------------------
    // Build chunk_info for the hierarchy EVLR
    // -----------------------------------------------------------------------
    // Absolute file position of chunk i = offset_to_point_data + chunk_cursor_starts[i]
    // Byte size of chunk i = next chunk start (or chunk table start) − this chunk start
    let mut chunk_info: Vec<(VoxelKey, u64, i32, i32)> = Vec::new();

    for (i, key) in ordered_keys.iter().enumerate() {
        let pts = node_map.get(key).unwrap();
        let cursor_start = chunk_cursor_starts[i];
        let cursor_end = if i + 1 < chunk_cursor_starts.len() {
            chunk_cursor_starts[i + 1]
        } else {
            // Last chunk ends where the chunk table starts.
            cursor_chunk_table_pos as u64
        };
        let file_offset = offset_to_point_data as u64 + cursor_start;
        let byte_size = (cursor_end - cursor_start) as i32;
        chunk_info.push((*key, file_offset, byte_size, pts.len() as i32));
        info!(
            "Chunk {:?}: offset={}, size={}, pts={}",
            key,
            file_offset,
            byte_size,
            pts.len()
        );
    }

    // -----------------------------------------------------------------------
    // EVLR: copc hierarchy
    // -----------------------------------------------------------------------
    let evlr_start = offset_to_point_data as u64 + point_data.len() as u64;

    let mut hier_payload: Vec<u8> = Vec::with_capacity(chunk_info.len() * 32);
    for (key, offset, byte_size, point_count) in &chunk_info {
        HierarchyEntry {
            key: *key,
            offset: *offset,
            byte_size: *byte_size,
            point_count: *point_count,
        }
        .write(&mut hier_payload)?;
    }
    write_evlr(&mut w, "copc", 1000, "copc hierarchy", &hier_payload)?;

    w.flush()?;
    drop(w);

    // -----------------------------------------------------------------------
    // Patch the file: copc info VLR + EVLR offset
    // -----------------------------------------------------------------------
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(output_path)?;

    // Patch copc info: root_hier_offset / root_hier_size
    let patched_info = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
        root_hier_offset: evlr_start + EVLR_HEADER_SIZE as u64,
        root_hier_size: hier_payload.len() as u64,
        gpstime_minimum: 0.0,
        gpstime_maximum: 0.0,
    };
    let mut pinfo_buf = Vec::with_capacity(160);
    patched_info.write(&mut pinfo_buf)?;
    f.seek(SeekFrom::Start(copc_info_payload_pos))?;
    f.write_all(&pinfo_buf)?;

    // Patch start_of_first_EVLR at byte 235
    f.seek(SeekFrom::Start(235))?;
    f.write_all(&evlr_start.to_le_bytes())?;

    info!("COPC file written: {:?}", output_path);
    Ok(())
}
