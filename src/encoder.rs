//! Streaming encoder with incremental repair-symbol computation.
//!
//! The encoder is designed for the media-streaming metric
//! **data-avail-to-first-send latency**:
//!
//! - Timer starts when the source data is first available.
//! - Timer stops when the first symbol is sent on the wire.
//!
//! For a systematic code, the first `k` symbols are the data symbols
//! themselves, so the latency for those is **zero**: as soon as a data
//! symbol is available it can be transmitted. Repair symbols are
//! computed incrementally in the background; when the `k`-th data
//! symbol arrives, all repair symbols are already finished and the
//! first repair can be sent immediately.
//!
//! # Usage
//!
//! ```
//! use scrs::encoder::StreamingEncoder;
//!
//! let mut enc = StreamingEncoder::new(4, 2, 1400).unwrap();
//!
//! // Feed data symbols one at a time (e.g. as they arrive from a
//! // network source or a capture device).
//! for i in 0..enc.k() {
//!     let data = vec![i as u8; enc.symbol_len()];
//!     enc.feed_data_symbol(i, &data).unwrap();
//!     // The data symbol can be sent on the wire right now.
//! }
//!
//! // All repair symbols are ready immediately.
//! for j in 0..enc.m() {
//!     let repair = enc.repair_symbol(j).unwrap();
//!     // Send repair on the wire.
//! }
//! ```

use crate::gf256::GfElem;
use crate::good_cauchy::GoodCauchyView;

/// Error type for streaming encoder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// Invalid symbol index (out of range for data or repair).
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },
    /// Payload length does not match `symbol_len`.
    WrongPayloadLen {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// Data symbol has already been fed.
    DuplicateData {
        /// The duplicate index.
        index: usize,
    },
}

/// A streaming encoder that computes repair symbols incrementally.
///
/// The encoder owns `m` repair-symbol buffers. Each call to
/// [`feed_data_symbol`](StreamingEncoder::feed_data_symbol) updates all
/// `m` buffers in place, so when the `k`-th data symbol arrives the
/// repairs are fully computed and ready for transmission.
///
/// Coefficients are stored as compact GF(256) bytes (one per matrix entry),
/// built via Good-Cauchy diagonal factorization. Nibble backends resolve the
/// process-wide shared scale-table bank; GFNI uses the coefficient bytes
/// directly.
pub struct StreamingEncoder {
    k: usize,
    m: usize,
    symbol_len: usize,
    /// Coefficients in data-major order: `coeffs[i * m + j] = C[i][j]`.
    coeffs: Vec<GfElem>,
    /// Repair-symbol buffers, each `symbol_len` bytes.
    repairs: Vec<Vec<u8>>,
    /// Bitmask of which data symbols have been fed.
    fed: Vec<bool>,
    /// Number of distinct data symbols fed so far.
    fed_count: usize,
}

impl StreamingEncoder {
    /// Create a new streaming encoder for `(k, m)` with `symbol_len`-byte
    /// symbols.
    ///
    /// Returns `None` for invalid dimensions or `k + m > 255`.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let cauchy = GoodCauchyView::new(k, m)?;
        if symbol_len == 0 {
            return None;
        }
        Some(Self {
            k,
            m,
            symbol_len,
            coeffs: cauchy.coefficient_matrix(),
            repairs: vec![vec![0u8; symbol_len]; m],
            fed: vec![false; k],
            fed_count: 0,
        })
    }

    /// Number of data symbols `k`.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols `m`.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Total codeword length `n = k + m`.
    pub const fn n(&self) -> usize {
        self.k + self.m
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of distinct data symbols fed so far.
    pub const fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Feed one data symbol into the encoder.
    ///
    /// `idx` must be in `0..k`. The payload must have length
    /// `symbol_len`. This method updates all `m` repair-symbol buffers
    /// in place.
    ///
    /// # Latency note
    ///
    /// The data symbol itself can be transmitted immediately (zero
    /// latency). The repair-symbol buffers are updated in the
    /// background; they are ready as soon as the `k`-th distinct data
    /// symbol has been fed.
    pub fn feed_data_symbol(&mut self, idx: usize, payload: &[u8]) -> Result<(), EncodeError> {
        if idx >= self.k {
            return Err(EncodeError::IndexOutOfRange {
                index: idx,
                max: self.k - 1,
            });
        }
        if payload.len() != self.symbol_len {
            return Err(EncodeError::WrongPayloadLen {
                expected: self.symbol_len,
                got: payload.len(),
            });
        }
        if self.fed[idx] {
            return Err(EncodeError::DuplicateData { index: idx });
        }
        self.fed[idx] = true;
        self.fed_count += 1;

        let row_start = idx * self.m;
        crate::simd::xor_scaled_bytes_many(
            &mut self.repairs,
            &self.coeffs[row_start..row_start + self.m],
            payload,
        );

        Ok(())
    }

    /// Return a reference to repair symbol `j` (`0..m`).
    ///
    /// Returns an error if `j` is out of range. The repair symbol is
    /// valid even before all data symbols have been fed (it simply
    /// reflects the contributions of the data symbols seen so far).
    pub fn repair_symbol(&self, j: usize) -> Result<&[u8], EncodeError> {
        if j >= self.m {
            return Err(EncodeError::IndexOutOfRange {
                index: j,
                max: self.m - 1,
            });
        }
        Ok(&self.repairs[j])
    }

    /// Consume the encoder and return all repair symbols.
    ///
    /// This is primarily a convenience for testing and batch use. In a
    /// streaming context the caller typically sends data symbols
    /// immediately and collects repair symbols via
    /// [`repair_symbol`](StreamingEncoder::repair_symbol).
    pub fn into_repairs(self) -> Vec<Vec<u8>> {
        self.repairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_matrix_matches_view_get() {
        let k = 16;
        let m = 8;
        let view = GoodCauchyView::new(k, m).unwrap();
        let matrix = view.coefficient_matrix();
        for i in 0..k {
            for j in 0..m {
                assert_eq!(matrix[i * m + j], view.get(i, j));
            }
        }
    }

    #[test]
    fn streaming_matches_batch_repairs() {
        use crate::batch::BatchCodec;
        use crate::good_cauchy::GoodCauchyView;

        let k = 8;
        let m = 4;
        let slen = 64;
        let data: Vec<u8> = (0..k * slen).map(|i| i as u8).collect();

        let batch = BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap();
        let symbols = batch.encode(&data).unwrap();

        let mut enc = StreamingEncoder::new(k, m, slen).unwrap();
        for i in 0..k {
            enc.feed_data_symbol(i, &data[i * slen..(i + 1) * slen])
                .unwrap();
        }
        for j in 0..m {
            assert_eq!(enc.repair_symbol(j).unwrap(), symbols[k + j].as_slice());
        }
    }
}
