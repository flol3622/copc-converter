use copc_streaming::{ByteSource, CopcError};
use reqwest::Client;

pub struct HttpSource {
    client: Client,
    url: String,
}

impl HttpSource {
    pub fn new(url: &str) -> Self {
        Self {
            client: Client::new(),
            url: url.to_string(),
        }
    }
}

impl ByteSource for HttpSource {
    async fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CopcError> {
        let end = offset + length - 1;
        let resp = self
            .client
            .get(&self.url)
            .header("Range", format!("bytes={offset}-{end}"))
            .send()
            .await
            .map_err(|e| CopcError::Io(std::io::Error::other(e)))?;
        if !resp.status().is_success() {
            return Err(CopcError::Io(std::io::Error::other(format!(
                "HTTP {}: range {offset}-{end}",
                resp.status()
            ))));
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| CopcError::Io(std::io::Error::other(e)))
    }

    async fn size(&self) -> Result<Option<u64>, CopcError> {
        // Use a 1-byte range GET instead of HEAD, since presigned URLs
        // are method-specific and HEAD often returns 403.
        // Parse total file size from the Content-Range header:
        //   Content-Range: bytes 0-0/123456789
        let resp = self
            .client
            .get(&self.url)
            .header("Range", "bytes=0-0")
            .send()
            .await
            .map_err(|e| CopcError::Io(std::io::Error::other(e)))?;
        Ok(resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit_once('/'))
            .and_then(|(_, total)| total.parse().ok()))
    }

    async fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, CopcError> {
        let mut indexed: Vec<(usize, u64, u64)> = ranges
            .iter()
            .enumerate()
            .map(|(i, (o, l))| (i, *o, *l))
            .collect();
        indexed.sort_by_key(|(_, o, _)| *o);

        struct Merged {
            offset: u64,
            length: u64,
            parts: Vec<(usize, u64, u64)>,
        }
        let mut merged: Vec<Merged> = Vec::new();
        for (i, offset, length) in &indexed {
            if let Some(last) = merged.last_mut() {
                let last_end = last.offset + last.length;
                if *offset <= last_end + 1024 {
                    let new_end = (*offset + *length).max(last_end);
                    last.length = new_end - last.offset;
                    last.parts.push((*i, *offset, *length));
                    continue;
                }
            }
            merged.push(Merged {
                offset: *offset,
                length: *length,
                parts: vec![(*i, *offset, *length)],
            });
        }

        let fetches: Vec<_> = merged
            .iter()
            .map(|m| self.read_range(m.offset, m.length))
            .collect();
        let fetched: Vec<Vec<u8>> = futures::future::join_all(fetches)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut results = vec![Vec::new(); ranges.len()];
        for (m, data) in merged.iter().zip(fetched.iter()) {
            for (orig_idx, offset, length) in &m.parts {
                let start = (*offset - m.offset) as usize;
                let end = start + *length as usize;
                results[*orig_idx] = data[start..end].to_vec();
            }
        }
        Ok(results)
    }
}
