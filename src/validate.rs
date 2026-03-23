/// Validate consistency of scanned input files before building the octree.
use crate::Error;
use crate::octree::{ScanResult, input_to_copc_format};
use std::path::PathBuf;
use tracing::debug;

/// Validated output: consistent properties across all input files.
#[derive(Debug)]
pub struct ValidatedInputs {
    /// WKT CRS string from input files (if present).
    pub wkt_crs: Option<Vec<u8>>,
    /// COPC output point format (6, 7, or 8).
    pub point_format: u8,
}

/// Returns true if the LAS point format includes GPS time.
fn format_has_gps_time(fmt: u8) -> bool {
    // LAS formats 0 and 2 lack GPS time; all others (1, 3–10) include it.
    !matches!(fmt, 0 | 2)
}

/// Check that all scanned files agree on CRS and point format,
/// and derive the COPC output point format.
pub fn validate(
    input_files: &[PathBuf],
    results: &[ScanResult],
    temporal_index: bool,
) -> crate::Result<ValidatedInputs> {
    let wkt_crs = results[0].wkt_crs.clone();
    let first_format = results[0].point_format_id;

    for (i, r) in results.iter().enumerate().skip(1) {
        if r.wkt_crs != wkt_crs {
            return Err(Error::CrsMismatch {
                file_a: input_files[0].clone(),
                file_b: input_files[i].clone(),
            });
        }
        if r.point_format_id != first_format {
            return Err(Error::PointFormatMismatch {
                file_a: input_files[0].clone(),
                format_a: first_format,
                file_b: input_files[i].clone(),
                format_b: r.point_format_id,
            });
        }
    }

    if temporal_index && !format_has_gps_time(first_format) {
        return Err(Error::NoGpsTime {
            format: first_format,
        });
    }

    let point_format = input_to_copc_format(first_format);
    debug!("Input point format: {first_format}, output COPC point format: {point_format}");

    Ok(ValidatedInputs {
        wkt_crs,
        point_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Bounds, ScanResult};

    fn make_result(wkt: Option<Vec<u8>>, fmt: u8) -> ScanResult {
        ScanResult {
            bounds: Bounds::empty(),
            point_count: 100,
            scale_x: 0.001,
            scale_y: 0.001,
            scale_z: 0.001,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            wkt_crs: wkt,
            point_format_id: fmt,
        }
    }

    #[test]
    fn validate_single_file() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 3)];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.point_format, 7);
        assert!(v.wkt_crs.is_none());
    }

    #[test]
    fn validate_matching_files() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let wkt = Some(b"WKT".to_vec());
        let results = vec![make_result(wkt.clone(), 8), make_result(wkt, 8)];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.point_format, 8);
    }

    #[test]
    fn validate_crs_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(b"WKT_A".to_vec()), 7),
            make_result(Some(b"WKT_B".to_vec()), 7),
        ];
        let err = validate(&files, &results, false).unwrap_err();
        assert!(matches!(err, Error::CrsMismatch { .. }));
    }

    #[test]
    fn validate_format_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(None, 3), make_result(None, 7)];
        let err = validate(&files, &results, false).unwrap_err();
        assert!(matches!(err, Error::PointFormatMismatch { .. }));
    }

    #[test]
    fn validate_temporal_index_requires_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 0)];
        let err = validate(&files, &results, true).unwrap_err();
        assert!(matches!(err, Error::NoGpsTime { .. }));
    }

    #[test]
    fn validate_temporal_index_with_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 1)];
        let v = validate(&files, &results, true).unwrap();
        assert_eq!(v.point_format, 6);
    }
}
