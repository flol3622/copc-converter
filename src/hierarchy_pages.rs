//! Build a paged COPC hierarchy EVLR following the reference untwine algorithm.
//!
//! A flat hierarchy is a single contiguous block of `HierarchyEntry`s (32 B each).
//! Potree's COPC support expects the *root* page to be small so it can be
//! fetched and parsed eagerly; descendant information should live in separate
//! child pages that can be lazy-loaded on demand. With a single 29 MB / 917k-
//! entry root page the browser receives the entire octree up front and runs out
//! of WebGL context (see issue #12).
//!
//! The layout that matches the COPC 1.0 spec and untwine's implementation:
//!
//! - Pages are concatenated back-to-back inside one `copc` EVLR (record id 1000).
//! - Each page is a flat array of 32-byte [`HierarchyEntry`] records.
//! - An entry with `point_count == -1` is a pointer to a child page:
//!   `offset` is the absolute file byte offset of the child page and
//!   `byte_size` is the child page's size.
//! - An entry with `point_count == 0` is a known-empty node present for
//!   tree-traversal (it has data-bearing descendants).
//! - An entry with `point_count > 0` is a data node with `offset`/`byte_size`
//!   pointing at the LAZ chunk.
//!
//! The split follows untwine's `CopcSupport::emitRoot` / `emitChildren`:
//! every `LEVEL_BREAK` octree levels we open a new child page, unless the
//! subtree would contain fewer than `MIN_HIERARCHY_SIZE` descendants in which
//! case we inline it in the parent page.

use crate::copc_types::{HierarchyEntry, VoxelKey};
use std::collections::HashMap;

/// Number of octree levels per page, matching untwine's `LevelBreak`.
pub(crate) const LEVEL_BREAK: i32 = 4;

/// Minimum subtree size required to split a child page off, matching
/// untwine's `MinHierarchySize`.
pub(crate) const MIN_HIERARCHY_SIZE: u32 = 50;

/// Per-node hierarchy record produced by the writer. `offset` / `byte_size`
/// follow the COPC spec: positive `point_count` → LAZ chunk reference, zero
/// → empty-ancestor placeholder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChunkEntry {
    pub key: VoxelKey,
    pub offset: u64,
    pub byte_size: i32,
    pub point_count: i32,
}

/// One octree page in the final EVLR payload.
struct BuiltPage {
    /// Serialized entries; child-page pointer entries are written with
    /// placeholder offsets to be patched in a second pass.
    data: Vec<u8>,
    /// Byte offsets (relative to `data`) of the `offset: u64` field of each
    /// child-page pointer, paired with the index of the referenced page in
    /// the flat page list.
    pointer_patches: Vec<(usize, usize)>,
}

/// Result of building the paged hierarchy.
pub(crate) struct PagedHierarchy {
    /// Concatenated bytes of every page, in depth-first order, root first.
    pub payload: Vec<u8>,
    /// Size of the root page in bytes. The root page always starts at byte 0.
    pub root_page_size: u64,
}

/// Produce the paged EVLR payload for the hierarchy entries.
///
/// `evlr_data_start` is the absolute file offset where the EVLR *data* payload
/// begins (i.e. just past the 60-byte EVLR record header). It is needed to
/// write absolute child-page offsets back into parent pages.
pub(crate) fn build_paged_hierarchy(
    entries: &[ChunkEntry],
    evlr_data_start: u64,
) -> anyhow::Result<PagedHierarchy> {
    if entries.is_empty() {
        return Ok(PagedHierarchy {
            payload: Vec::new(),
            root_page_size: 0,
        });
    }

    // Index by key for O(1) lookups during recursion.
    let by_key: HashMap<VoxelKey, ChunkEntry> = entries.iter().map(|e| (e.key, *e)).collect();

    // Count descendants per key (excluding the key itself), matching untwine's
    // `calcCounts`. This guides the inline-vs-subpage decision.
    let descendants = compute_descendant_counts(&by_key);

    // The root of the octree is always (0,0,0,0) in a COPC file.
    let root_key = VoxelKey {
        level: 0,
        x: 0,
        y: 0,
        z: 0,
    };
    if !by_key.contains_key(&root_key) {
        anyhow::bail!("hierarchy is missing the root node (0,0,0,0)");
    }

    let mut pages: Vec<BuiltPage> = Vec::new();
    let root_idx = emit_page(&root_key, &by_key, &descendants, &mut pages)?;
    debug_assert_eq!(root_idx, 0, "root page must be emitted first");

    // Compute absolute file offsets for every page. Pages are laid out
    // sequentially starting at `evlr_data_start`.
    let mut page_abs_offsets: Vec<u64> = Vec::with_capacity(pages.len());
    let mut page_sizes: Vec<u64> = Vec::with_capacity(pages.len());
    let mut cursor = evlr_data_start;
    for page in &pages {
        page_abs_offsets.push(cursor);
        page_sizes.push(page.data.len() as u64);
        cursor += page.data.len() as u64;
    }

    // Patch child_page_offset / child_page_size in parent pages.
    for (i, page) in pages.iter_mut().enumerate() {
        for &(local_off, child_idx) in &page.pointer_patches {
            let child_abs = page_abs_offsets[child_idx];
            let child_size = page_sizes[child_idx];
            // Entry layout: [key 16B][offset u64 8B][byte_size i32 4B][point_count i32 4B].
            // `local_off` points at the start of the `offset` field.
            page.data[local_off..local_off + 8].copy_from_slice(&child_abs.to_le_bytes());
            page.data[local_off + 8..local_off + 12]
                .copy_from_slice(&(child_size as i32).to_le_bytes());
            // point_count at local_off+12 was already written as -1.
            let _ = i;
        }
    }

    // Concatenate all pages into a single payload.
    let total: usize = pages.iter().map(|p| p.data.len()).sum();
    let mut payload = Vec::with_capacity(total);
    for page in &pages {
        payload.extend_from_slice(&page.data);
    }

    Ok(PagedHierarchy {
        payload,
        root_page_size: page_sizes[0],
    })
}

/// Compute, for every key in `by_key`, the number of descendants
/// (excluding the key itself). Matches untwine's `createHierarchy`.
fn compute_descendant_counts(by_key: &HashMap<VoxelKey, ChunkEntry>) -> HashMap<VoxelKey, u32> {
    let mut counts: HashMap<VoxelKey, u32> = HashMap::with_capacity(by_key.len());

    // Each node contributes one count to itself and, recursively, to every
    // ancestor up to the root. Walking ancestors iteratively avoids recursion
    // depth issues on very deep trees.
    for key in by_key.keys() {
        let mut cur = *key;
        while let Some(parent) = cur.parent() {
            *counts.entry(parent).or_insert(0) += 1;
            cur = parent;
        }
    }

    counts
}

/// Recursively emit a page rooted at `root` into `pages`, returning the
/// newly-pushed page's index.
///
/// Mirrors untwine's `emitRoot` + `emitChildren`. The page contains `root`
/// itself plus every descendant up to (but not past) `stop_level`.
/// Any subtree at exactly `stop_level` whose descendant count exceeds
/// [`MIN_HIERARCHY_SIZE`] is split off into a new child page (with a pointer
/// entry written in place in the parent).
fn emit_page(
    root: &VoxelKey,
    by_key: &HashMap<VoxelKey, ChunkEntry>,
    descendants: &HashMap<VoxelKey, u32>,
    pages: &mut Vec<BuiltPage>,
) -> anyhow::Result<usize> {
    // Reserve a slot first so the page index is stable even as children
    // recursively append more pages after us.
    let idx = pages.len();
    pages.push(BuiltPage {
        data: Vec::new(),
        pointer_patches: Vec::new(),
    });

    let stop_level = root.level + LEVEL_BREAK;
    let mut data: Vec<u8> = Vec::new();
    let mut pointer_patches: Vec<(usize, usize)> = Vec::new();

    // Always write the root entry first.
    write_chunk_entry(&mut data, by_key[root])?;

    // Depth-first traversal to preserve untwine's inline ordering.
    emit_children(
        root,
        stop_level,
        by_key,
        descendants,
        &mut data,
        &mut pointer_patches,
        pages,
    )?;

    pages[idx] = BuiltPage {
        data,
        pointer_patches,
    };
    Ok(idx)
}

fn emit_children(
    parent: &VoxelKey,
    stop_level: i32,
    by_key: &HashMap<VoxelKey, ChunkEntry>,
    descendants: &HashMap<VoxelKey, u32>,
    data: &mut Vec<u8>,
    pointer_patches: &mut Vec<(usize, usize)>,
    pages: &mut Vec<BuiltPage>,
) -> anyhow::Result<()> {
    // Iterate children in a deterministic (x, y, z) order so repeated runs
    // produce byte-identical output.
    for i in 0..8 {
        let child = VoxelKey {
            level: parent.level + 1,
            x: parent.x * 2 + (i & 1),
            y: parent.y * 2 + ((i >> 1) & 1),
            z: parent.z * 2 + ((i >> 2) & 1),
        };
        if !by_key.contains_key(&child) {
            continue;
        }

        let child_descendants = descendants.get(&child).copied().unwrap_or(0);

        if child.level != stop_level || child_descendants <= MIN_HIERARCHY_SIZE {
            // Inline this child's subtree into the current page.
            write_chunk_entry(data, by_key[&child])?;
            emit_children(
                &child,
                stop_level,
                by_key,
                descendants,
                data,
                pointer_patches,
                pages,
            )?;
        } else {
            // Split: write a subpage-pointer entry in the parent page and
            // recurse to build the actual child page.
            let pointer_off = data.len() + 16; // skip VoxelKey (16B) → offset field
            write_page_pointer_placeholder(data, child)?;
            let sub_idx = emit_page(&child, by_key, descendants, pages)?;
            pointer_patches.push((pointer_off, sub_idx));
        }
    }
    Ok(())
}

fn write_chunk_entry(data: &mut Vec<u8>, entry: ChunkEntry) -> anyhow::Result<()> {
    HierarchyEntry {
        key: entry.key,
        offset: entry.offset,
        byte_size: entry.byte_size,
        point_count: entry.point_count,
    }
    .write(data)?;
    Ok(())
}

/// Write a hierarchy entry that marks the given key as living in a child
/// page. `offset` / `byte_size` are zeroed; the caller patches them once
/// absolute page offsets are known.
fn write_page_pointer_placeholder(data: &mut Vec<u8>, key: VoxelKey) -> anyhow::Result<()> {
    HierarchyEntry {
        key,
        offset: 0,    // patched later
        byte_size: 0, // patched later
        point_count: -1,
    }
    .write(data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::{Cursor, Read};

    fn entry(level: i32, x: i32, y: i32, z: i32, points: i32) -> ChunkEntry {
        ChunkEntry {
            key: VoxelKey { level, x, y, z },
            offset: if points > 0 { 1000 } else { 0 },
            byte_size: if points > 0 { 100 } else { 0 },
            point_count: points,
        }
    }

    /// Parse every page referenced from the root, returning all leaf (data /
    /// empty-ancestor) entries keyed by `VoxelKey` so we can compare content.
    fn collect_all_entries(
        payload: &[u8],
        evlr_data_start: u64,
        root_size: u64,
    ) -> HashMap<VoxelKey, (u64, i32, i32)> {
        let mut found: HashMap<VoxelKey, (u64, i32, i32)> = HashMap::new();
        // queue of (page_start_in_payload, page_size)
        let mut queue: Vec<(u64, u64)> = vec![(0, root_size)];
        while let Some((start, size)) = queue.pop() {
            let slice = &payload[start as usize..(start + size) as usize];
            let mut r = Cursor::new(slice);
            let end = slice.len() as u64;
            while r.position() < end {
                let level = r.read_i32::<LittleEndian>().unwrap();
                let x = r.read_i32::<LittleEndian>().unwrap();
                let y = r.read_i32::<LittleEndian>().unwrap();
                let z = r.read_i32::<LittleEndian>().unwrap();
                let off = r.read_u64::<LittleEndian>().unwrap();
                let bsize = r.read_i32::<LittleEndian>().unwrap();
                let pcount = r.read_i32::<LittleEndian>().unwrap();
                let key = VoxelKey { level, x, y, z };
                if pcount == -1 {
                    // Child page — compute relative offset into `payload`.
                    let rel = off - evlr_data_start;
                    queue.push((rel, bsize as u64));
                } else {
                    found.insert(key, (off, bsize, pcount));
                }
            }
        }
        found
    }

    #[test]
    fn single_page_for_small_tree() {
        // Root + 8 level-1 children = 9 entries. Well below MIN_HIERARCHY_SIZE
        // so nothing should be split out.
        let mut entries = vec![entry(0, 0, 0, 0, 10)];
        for i in 0..8 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = (i >> 2) & 1;
            entries.push(entry(1, x, y, z, 5));
        }

        let paged = build_paged_hierarchy(&entries, 1_000).unwrap();
        assert_eq!(
            paged.payload.len() as u64,
            paged.root_page_size,
            "small tree should fit in a single page"
        );
        // 9 entries × 32 B = 288 B.
        assert_eq!(paged.root_page_size, 9 * 32);
    }

    #[test]
    fn splits_deep_tree_into_child_pages() {
        // Build a proper octree where (4,0,0,0) has >50 descendants to
        // trigger a page split at stop_level = 4.
        let mut entries = vec![entry(0, 0, 0, 0, 1)];
        // Single branch from root to level 4.
        for lvl in 1..=4 {
            entries.push(entry(lvl, 0, 0, 0, 1));
        }
        // All 8 children of (4,0,0,0) at level 5.
        for i in 0..8u32 {
            let x = (i & 1) as i32;
            let y = ((i >> 1) & 1) as i32;
            let z = ((i >> 2) & 1) as i32;
            entries.push(entry(5, x, y, z, 1));
            // All 8 children of each level-5 node at level 6.
            for j in 0..8u32 {
                let cx = x * 2 + (j & 1) as i32;
                let cy = y * 2 + ((j >> 1) & 1) as i32;
                let cz = z * 2 + ((j >> 2) & 1) as i32;
                entries.push(entry(6, cx, cy, cz, 1));
            }
        }
        // 8 + 64 = 72 descendants of (4,0,0,0), which is > MIN_HIERARCHY_SIZE.

        let evlr_start = 10_000u64;
        let paged = build_paged_hierarchy(&entries, evlr_start).unwrap();
        assert!(
            paged.payload.len() as u64 > paged.root_page_size,
            "deep tree should produce more than one page"
        );

        // Every original entry is reachable via page walk.
        let found = collect_all_entries(&paged.payload, evlr_start, paged.root_page_size);
        for e in &entries {
            assert!(found.contains_key(&e.key), "missing {:?}", e.key);
        }
    }

    #[test]
    fn pointer_offsets_are_absolute() {
        let mut entries = vec![entry(0, 0, 0, 0, 1)];
        for lvl in 1..=4 {
            entries.push(entry(lvl, 0, 0, 0, 1));
        }
        // All 8 children of (4,0,0,0) at level 5 + all grandchildren at
        // level 6 = 72 descendants, exceeding MIN_HIERARCHY_SIZE.
        for i in 0..8u32 {
            let x = (i & 1) as i32;
            let y = ((i >> 1) & 1) as i32;
            let z = ((i >> 2) & 1) as i32;
            entries.push(entry(5, x, y, z, 1));
            for j in 0..8u32 {
                let cx = x * 2 + (j & 1) as i32;
                let cy = y * 2 + ((j >> 1) & 1) as i32;
                let cz = z * 2 + ((j >> 2) & 1) as i32;
                entries.push(entry(6, cx, cy, cz, 1));
            }
        }

        let evlr_start = 4_096u64;
        let paged = build_paged_hierarchy(&entries, evlr_start).unwrap();

        // Find the subpage pointer in the root page and verify its offset
        // lies within the EVLR payload.
        let mut r = Cursor::new(&paged.payload[..paged.root_page_size as usize]);
        let end = paged.root_page_size;
        let mut saw_pointer = false;
        while r.position() < end {
            let mut key_bytes = [0u8; 16];
            r.read_exact(&mut key_bytes).unwrap();
            let off = r.read_u64::<LittleEndian>().unwrap();
            let bsize = r.read_i32::<LittleEndian>().unwrap();
            let pcount = r.read_i32::<LittleEndian>().unwrap();
            if pcount == -1 {
                saw_pointer = true;
                assert!(off >= evlr_start, "subpage offset must be absolute");
                assert!(
                    off + bsize as u64 <= evlr_start + paged.payload.len() as u64,
                    "subpage must lie within the EVLR payload"
                );
            }
        }
        assert!(saw_pointer, "expected at least one child-page pointer");
    }
}
