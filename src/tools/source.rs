use copc_streaming::{ByteSource, CopcError, FileSource};

use super::HttpSource;

/// Unified source that handles both HTTP URLs and local file paths.
pub enum Source {
    Http(HttpSource),
    File(FileSource),
}

impl Source {
    pub fn from_arg(arg: &str) -> Result<Self, CopcError> {
        if arg.starts_with("http://") || arg.starts_with("https://") {
            Ok(Source::Http(HttpSource::new(arg)))
        } else {
            Ok(Source::File(FileSource::open(arg)?))
        }
    }
}

impl ByteSource for Source {
    async fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CopcError> {
        match self {
            Source::Http(s) => s.read_range(offset, length).await,
            Source::File(s) => s.read_range(offset, length).await,
        }
    }

    async fn size(&self) -> Result<Option<u64>, CopcError> {
        match self {
            Source::Http(s) => s.size().await,
            Source::File(s) => s.size().await,
        }
    }

    async fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, CopcError> {
        match self {
            Source::Http(s) => s.read_ranges(ranges).await,
            Source::File(s) => s.read_ranges(ranges).await,
        }
    }
}
