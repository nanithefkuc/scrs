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
pub struct StreamingEncoder {
    k: usize,
    m: usize,
    symbol_len: usize,
    cauchy: GoodCauchyView,
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
            cauchy,
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

        let mut repair_slices: Vec<&mut [u8]> =
            self.repairs.iter_mut().map(|r| r.as_mut_slice()).collect();
        self.cauchy
            .add_data_symbol(idx, payload, &mut repair_slices);

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

    /// Consume the encoder and return all `n` symbols.
    ///
    /// The returned vector has length `n`; entries `0..k` are the data
    /// symbols (which the caller must have kept) and entries `k..n`
    /// are the repair symbols computed by this encoder.
    ///
    /// This is primarily a convenience for testing and batch use. In a
    /// streaming context the caller typically sends data symbols
    /// immediately and collects repair symbols via
    /// [`repair_symbol`](StreamingEncoder::repair_symbol).
    pub fn into_repairs(self) -> Vec<Vec<u8>> {
        self.repairs
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_params() {
        assert!(StreamingEncoder::new(0, 5, 10).is_none());
        assert!(StreamingEncoder::new(5, 0, 10).is_none());
        assert!(StreamingEncoder::new(5, 5, 0).is_none());
        assert!(StreamingEncoder::new(200, 100, 10).is_none());
    }

    #[test]
    fn accessors() {
        let enc = StreamingEncoder::new(4, 3, 16).unwrap();
        assert_eq!(enc.k(), 4);
        assert_eq!(enc.m(), 3);
        assert_eq!(enc.n(), 7);
        assert_eq!(enc.symbol_len(), 16);
    }

    /// Compute expected repair symbols using GoodCauchyView directly (batch).
    fn good_cauchy_repairs(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
        let view = GoodCauchyView::new(k, m).unwrap();
        let mut repairs = vec![vec![0u8; symbol_len]; m];
        for j in 0..m {
            for i in 0..k {
                let coeff = view.get(i, j);
                if coeff == crate::gf256::GfElem::ZERO {
                    continue;
                }
                let data_slice = &data[i * symbol_len..(i + 1) * symbol_len];
                if coeff == crate::gf256::GfElem::ONE {
                    for (out, &b) in repairs[j].iter_mut().zip(data_slice.iter()) {
                        *out ^= b;
                    }
                } else {
                    for (out, &b) in repairs[j].iter_mut().zip(data_slice.iter()) {
                        *out ^= crate::gf256::GfElem(b).mul(coeff).0;
                    }
                }
            }
        }
        repairs
    }

    #[test]
    fn incremental_matches_batch() {
        let k = 5;
        let m = 3;
        let symbol_len = 8;
        let mut enc = StreamingEncoder::new(k, m, symbol_len).unwrap();

        let data: Vec<u8> = (0..k * symbol_len).map(|i| i as u8).collect();
        let expected_repairs = good_cauchy_repairs(k, m, symbol_len, &data);

        // Feed data symbols incrementally.
        for i in 0..k {
            let start = i * symbol_len;
            enc.feed_data_symbol(i, &data[start..start + symbol_len])
                .unwrap();
        }

        // Repair symbols must match the Good-Cauchy batch computation exactly.
        for j in 0..m {
            let repair = enc.repair_symbol(j).unwrap();
            assert_eq!(repair, &expected_repairs[j], "repair symbol {} mismatch", j);
        }
    }

    #[test]
    fn out_of_order_feed() {
        let k = 4;
        let m = 2;
        let symbol_len = 4;
        let mut enc = StreamingEncoder::new(k, m, symbol_len).unwrap();

        let data: Vec<u8> = (0..k * symbol_len).map(|i| i as u8).collect();
        let expected_repairs = good_cauchy_repairs(k, m, symbol_len, &data);

        // Feed in reverse order.
        for i in (0..k).rev() {
            let start = i * symbol_len;
            enc.feed_data_symbol(i, &data[start..start + symbol_len])
                .unwrap();
        }

        for j in 0..m {
            let repair = enc.repair_symbol(j).unwrap();
            assert_eq!(repair, &expected_repairs[j]);
        }
    }

    #[test]
    fn duplicate_rejected() {
        let mut enc = StreamingEncoder::new(3, 2, 4).unwrap();
        let payload = vec![0x42; 4];
        enc.feed_data_symbol(0, &payload).unwrap();
        assert_eq!(
            enc.feed_data_symbol(0, &payload),
            Err(EncodeError::DuplicateData { index: 0 })
        );
    }

    #[test]
    fn wrong_payload_len_rejected() {
        let mut enc = StreamingEncoder::new(3, 2, 4).unwrap();
        assert_eq!(
            enc.feed_data_symbol(0, &vec![0x42; 3]),
            Err(EncodeError::WrongPayloadLen {
                expected: 4,
                got: 3,
            })
        );
    }

    #[test]
    fn index_out_of_range_rejected() {
        let mut enc = StreamingEncoder::new(3, 2, 4).unwrap();
        assert_eq!(
            enc.feed_data_symbol(3, &vec![0x42; 4]),
            Err(EncodeError::IndexOutOfRange { index: 3, max: 2 })
        );
        assert_eq!(
            enc.repair_symbol(2),
            Err(EncodeError::IndexOutOfRange { index: 2, max: 1 })
        );
    }
}
