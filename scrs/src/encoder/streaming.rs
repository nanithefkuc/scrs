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
//! # Matrix and capacity
//!
//! `StreamingEncoder` always uses [`GoodCauchyView`], not standard Cauchy.
//! It accepts nonzero dimensions and symbol lengths only when
//! `k + m <= 255`, the Good-Cauchy capacity.
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
//!
//! # Reuse
//!
//! Construct the encoder once for a fixed `(k, m, symbol_len)` and call
//! [`reset`](StreamingEncoder::reset) between blocks. Coefficient tables and
//! repair buffers are retained; only the fed-state and repair payload are
//! cleared. This matches production multi-block streaming and avoids paying
//! construction cost on every block.

use crate::gf256::GfElem;
use crate::good_cauchy::GoodCauchyView;

use super::EncodeError;

/// A Good-Cauchy streaming encoder that computes repair symbols incrementally.
///
/// This type intentionally uses [`GoodCauchyView`] and supports
/// `n = k + m <= 255`.
///
/// The encoder owns a flat `m × symbol_len` repair buffer. Each call to
/// [`feed_data_symbol`](StreamingEncoder::feed_data_symbol) updates all
/// `m` repair rows in place. With the `simd` feature enabled this uses a
/// multi-destination SIMD kernel when the host supports one; otherwise it
/// uses portable scalar AXPY operations. When the `k`-th data symbol arrives,
/// the repairs are fully computed and ready for transmission.
///
/// Coefficients are stored as compact GF(256) bytes (one per matrix entry),
/// built via Good-Cauchy diagonal factorization.
pub struct StreamingEncoder {
    k: usize,
    m: usize,
    symbol_len: usize,
    /// Coefficients in data-major order: `coeffs[i * m + j] = C[i][j]`.
    coeffs: Vec<GfElem>,
    /// Flat repair payload: row `j` is `repairs[j * symbol_len .. (j+1) * symbol_len]`.
    repairs: Vec<u8>,
    /// Bitmask of which data symbols have been fed.
    fed: Vec<bool>,
    /// Number of distinct data symbols fed so far.
    fed_count: usize,
}

impl StreamingEncoder {
    /// Create a new streaming encoder for `(k, m)` with `symbol_len`-byte
    /// symbols.
    ///
    /// Returns `None` if `k` or `m` is zero, `symbol_len` is zero, or
    /// `k + m > 255`. These are exactly the dimensions rejected by
    /// [`GoodCauchyView::new`], plus a zero symbol length.
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
            repairs: vec![0u8; m * symbol_len],
            fed: vec![false; k],
            fed_count: 0,
        })
    }

    /// Clear fed-state and zero repair buffers for the next block.
    ///
    /// Coefficient tables and buffer capacity are retained. Prefer this over
    /// constructing a fresh encoder for every block of the same configuration.
    pub fn reset(&mut self) {
        self.repairs.fill(0);
        self.fed.fill(false);
        self.fed_count = 0;
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
        crate::payload::xor_scaled_bytes_rows(
            &mut self.repairs,
            self.symbol_len,
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
        let start = j * self.symbol_len;
        Ok(&self.repairs[start..start + self.symbol_len])
    }

    /// Consume the encoder and return all repair symbols as separate buffers.
    ///
    /// This is primarily a convenience for testing and batch use. In a
    /// streaming context the caller typically sends data symbols
    /// immediately and collects repair symbols via
    /// [`repair_symbol`](StreamingEncoder::repair_symbol).
    pub fn into_repairs(self) -> Vec<Vec<u8>> {
        self.repairs
            .chunks_exact(self.symbol_len)
            .map(|row| row.to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exact_good_cauchy_capacity() {
        assert!(StreamingEncoder::new(254, 1, 1).is_some());
        assert!(StreamingEncoder::new(255, 1, 1).is_none());
    }

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

    #[test]
    fn reset_reuses_encoder_for_second_block() {
        use crate::batch::BatchCodec;
        use crate::good_cauchy::GoodCauchyView;

        let k = 5;
        let m = 3;
        let slen = 32;
        let batch = BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap();

        let data_a: Vec<u8> = (0..k * slen).map(|i| (i as u8).wrapping_mul(3)).collect();
        let data_b: Vec<u8> = (0..k * slen)
            .map(|i| (i as u8).wrapping_mul(7) ^ 0x5A)
            .collect();
        let symbols_a = batch.encode(&data_a).unwrap();
        let symbols_b = batch.encode(&data_b).unwrap();

        let mut enc = StreamingEncoder::new(k, m, slen).unwrap();
        for i in 0..k {
            enc.feed_data_symbol(i, &data_a[i * slen..(i + 1) * slen])
                .unwrap();
        }
        for j in 0..m {
            assert_eq!(enc.repair_symbol(j).unwrap(), symbols_a[k + j].as_slice());
        }

        enc.reset();
        assert_eq!(enc.fed_count(), 0);
        for j in 0..m {
            assert!(enc.repair_symbol(j).unwrap().iter().all(|&b| b == 0));
        }

        for i in 0..k {
            enc.feed_data_symbol(i, &data_b[i * slen..(i + 1) * slen])
                .unwrap();
        }
        for j in 0..m {
            assert_eq!(enc.repair_symbol(j).unwrap(), symbols_b[k + j].as_slice());
        }
    }

    #[test]
    fn into_repairs_splits_flat_buffer() {
        let k = 4;
        let m = 3;
        let slen = 8;
        let data: Vec<u8> = (0..k * slen).map(|i| i as u8).collect();
        let mut enc = StreamingEncoder::new(k, m, slen).unwrap();
        for i in 0..k {
            enc.feed_data_symbol(i, &data[i * slen..(i + 1) * slen])
                .unwrap();
        }
        let expected: Vec<Vec<u8>> = (0..m)
            .map(|j| enc.repair_symbol(j).unwrap().to_vec())
            .collect();
        assert_eq!(enc.into_repairs(), expected);
    }
}
