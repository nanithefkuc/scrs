//! Batch (non-streaming) Cauchy Reed-Solomon encode and decode.
//!
//! This is the correctness reference: a straightforward implementation that
//! uses [`crate::matrix`] operations directly. It does not attempt to minimize
//! latency and allocates freely. The streaming decoder in [`crate::v2`] is
//! validated against the round-trip property tests defined here.
//!
//! # Coding scheme
//!
//! Systematic Cauchy-RS with generator `G = [I_k | A]` where `A` is the
//! `k x m` Cauchy matrix produced by [`crate::cauchy::CauchyView`]. The
//! codeword is `n = k + m` symbols of `symbol_len` bytes each:
//!
//! - Symbols `0..k` are **data**: symbol `i` equals `data[i]` verbatim.
//! - Symbols `k..n` are **repair**: symbol `k + j` is the GF(256) linear
//!   combination `sum_i A[i][j] * data[i]`, applied independently to each
//!   byte position.
//!
//! Any `k` of the `n` symbols suffice to recover the original `k` data
//! symbols, by MDS-ness of the Cauchy construction.

use crate::coding_matrix::CodingMatrix;
use crate::gf256::GfElem;
use crate::matrix::{self, MatrixViewMut};

/// Error returned when constructing a [`BatchCodec`] with invalid parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `k` or `m` is zero.
    ZeroDimension,
    /// `k + m` exceeds 256, the v0.1 cap (see [crate::cauchy]).
    TooManySymbols,
    /// `symbol_len` is zero.
    ZeroSymbolLen,
}

/// Error returned by [`BatchCodec::decode`] when the inputs are malformed or
/// insufficient to recover the data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Wrong number of symbols provided: expected exactly `k`.
    WrongCount {
        /// The expected count (`k`).
        expected: usize,
        /// The actual count received.
        got: usize,
    },
    /// A symbol index is outside the valid range `0..n`.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The codeword length `n`.
        n: usize,
    },
    /// The same symbol index appeared more than once.
    DuplicateIndex {
        /// The duplicated index.
        index: usize,
    },
    /// A payload has the wrong length.
    WrongPayloadLen {
        /// The expected length (`symbol_len`).
        expected: usize,
        /// The actual length.
        got: usize,
    },
    /// The provided symbols are linearly dependent (rank < `k`). For a valid
    /// MDS code with distinct in-range indices this never occurs; seeing it
    /// indicates either a bug in SCRS or a corrupted codeword.
    InsufficientRank {
        /// The rank achieved during elimination.
        rank: usize,
        /// The required rank (`k`).
        k: usize,
    },
}

/// A batch (non-streaming) Cauchy Reed-Solomon codec configuration.
///
/// Construct once via [`BatchCodec::new`], then use [`encode`][BatchCodec::encode]
/// and [`decode`][BatchCodec::decode] repeatedly. The codec stores only the
/// parameters `(k, m, symbol_len)` and a coding matrix view; it holds no per-call
/// state.
#[derive(Clone, Copy, Debug)]
pub struct BatchCodec<C: CodingMatrix> {
    k: usize,
    m: usize,
    symbol_len: usize,
    cauchy: C,
}

impl<C: CodingMatrix> BatchCodec<C> {
    /// Create a codec for `(k, m)` with symbols of `symbol_len` bytes.
    ///
    /// Returns [`ConfigError::ZeroDimension`] if `k` or `m` is zero,
    /// [`ConfigError::TooManySymbols`] if `k + m > 256`, and
    /// [`ConfigError::ZeroSymbolLen`] if `symbol_len` is zero.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        let cauchy = C::new(k, m).ok_or(ConfigError::TooManySymbols)?;
        Ok(Self {
            k,
            m,
            symbol_len,
            cauchy,
        })
    }

    /// Number of data symbols.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols.
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

    /// Encode `k * symbol_len` bytes of data into `n` symbols.
    ///
    /// The returned vector has length `n`; entry `i` is `symbol_len` bytes.
    /// Entries `0..k` are the data copied verbatim (systematic); entries
    /// `k..n` are repair symbols computed as Cauchy-weighted combinations of
    /// the data.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, ConfigError> {
        let expected = self.k * self.symbol_len;
        if data.len() != expected {
            // Wrong input length: surface as ZeroSymbolLen (the only
            // ConfigError variant that does not imply the parameters
            // themselves are invalid). The streaming encoder (Phase 3+) will
            // introduce a dedicated EncodeError type.
            return Err(ConfigError::ZeroSymbolLen);
        }
        let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(self.n());
        // Data symbols: copy.
        for i in 0..self.k {
            let start = i * self.symbol_len;
            symbols.push(data[start..start + self.symbol_len].to_vec());
        }
        // Repair symbols: for each repair j, payload[p] = sum_i A[i][j] * data[i][p].
        for j in 0..self.m {
            let mut repair = vec![0u8; self.symbol_len];
            for i in 0..self.k {
                let coeff = self.cauchy.get(i, j);
                if coeff == GfElem::ZERO {
                    continue;
                }
                let data_start = i * self.symbol_len;
                let data_slice = &data[data_start..data_start + self.symbol_len];
                if coeff == GfElem::ONE {
                    for (out, &b) in repair.iter_mut().zip(data_slice.iter()) {
                        *out ^= b;
                    }
                } else {
                    for (out, &b) in repair.iter_mut().zip(data_slice.iter()) {
                        *out ^= GfElem(b).mul(coeff).0;
                    }
                }
            }
            symbols.push(repair);
        }
        Ok(symbols)
    }

    /// Decode any `k` of the `n` symbols back into the original
    /// `k * symbol_len` bytes.
    ///
    /// `symbols` is a slice of `(index, payload)` pairs where `index` is in
    /// `0..n` identifying which codeword symbol this is, and `payload` is
    /// `symbol_len` bytes. Exactly `k` pairs with distinct, in-range indices
    /// must be provided.
    ///
    /// The decoder builds the `k x k` coefficient submatrix `M` of the
    /// systematic generator corresponding to the received indices, augments
    /// it with the `k x symbol_len` payload block `[M | P]`, and reduces to
    /// RREF via [`crate::matrix::rref`]. After elimination the payload block
    /// directly holds the recovered data symbols.
    pub fn decode(&self, symbols: &[(usize, &[u8])]) -> Result<Vec<u8>, DecodeError> {
        let k = self.k;
        let n = self.n();
        let symbol_len = self.symbol_len;

        if symbols.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: symbols.len(),
            });
        }

        // Validate indices and payload lengths, check for duplicates.
        // Use a heap-allocated bool vector sized to n (not a fixed 256-byte
        // array) so the decoder generalizes to k + m > 256 in a future
        // GF(2^16) backend.
        let mut seen = vec![false; n];
        for &(idx, payload) in symbols {
            if idx >= n {
                return Err(DecodeError::IndexOutOfRange { index: idx, n });
            }
            if seen[idx] {
                return Err(DecodeError::DuplicateIndex { index: idx });
            }
            seen[idx] = true;
            if payload.len() != symbol_len {
                return Err(DecodeError::WrongPayloadLen {
                    expected: symbol_len,
                    got: payload.len(),
                });
            }
        }

        // Build the augmented matrix [M | P] as a flat row-major buffer.
        // Stride = k + symbol_len. Row r: [M[r][0..k] | P[r][0..symbol_len]].
        let stride = k + symbol_len;
        let mut buf = vec![GfElem::ZERO; k * stride];
        for (row, &(idx, payload)) in symbols.iter().enumerate() {
            // Coefficients: M[row][col] = G[col][idx], the entry in column
            // `idx` of the systematic generator G = [I_k | A].
            if idx < k {
                // Identity column: M[row][idx] = 1.
                buf[row * stride + idx] = GfElem::ONE;
            } else {
                // Cauchy column: M[row][col] = A[col][idx - k].
                let repair = idx - k;
                for col in 0..k {
                    buf[row * stride + col] = self.cauchy.get(col, repair);
                }
            }
            // Payload bytes as GF(256) elements.
            for (p, &byte) in payload.iter().enumerate() {
                buf[row * stride + k + p] = GfElem(byte);
            }
        }

        // Reduce to RREF. For a valid MDS code with k distinct indices, the
        // k x k coefficient block is non-singular and rref yields rank k,
        // leaving the left block as I_k and the payload block holding the
        // decoded data.
        let mut view = MatrixViewMut::new(&mut buf, k, stride)
            .expect("augmented matrix dimensions are internally consistent");
        let rank = matrix::rref(&mut view);
        if rank != k {
            return Err(DecodeError::InsufficientRank { rank, k });
        }

        // Read out the data: after RREF with M -> I_k, row r's payload
        // block holds data[r].
        let mut out = vec![0u8; k * symbol_len];
        for r in 0..k {
            for p in 0..symbol_len {
                out[r * symbol_len + p] = buf[r * stride + k + p].0;
            }
        }
        Ok(out)
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate all k-subsets of 0..n in lexicographic order (duplicates
    /// included in the count for test brevity).
    fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
        if k > n {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut state: Vec<usize> = (0..k).collect();
        loop {
            result.push(state.clone());
            // Advance.
            let mut i = k - 1;
            loop {
                if state[i] < n - k + i {
                    state[i] += 1;
                    for j in (i + 1)..k {
                        state[j] = state[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return result;
                }
                i -= 1;
            }
        }
    }

    // ---- Construction ----

    #[test]
    fn rejects_zero_dimensions() {
        assert!(matches!(
            BatchCodec::<crate::cauchy::CauchyView>::new(0, 5, 10),
            Err(ConfigError::ZeroDimension)
        ));
        assert!(matches!(
            BatchCodec::<crate::cauchy::CauchyView>::new(5, 0, 10),
            Err(ConfigError::ZeroDimension)
        ));
        assert!(matches!(
            BatchCodec::<crate::cauchy::CauchyView>::new(5, 5, 0),
            Err(ConfigError::ZeroSymbolLen)
        ));
    }

    #[test]
    fn rejects_oversized() {
        assert!(matches!(
            BatchCodec::<crate::cauchy::CauchyView>::new(200, 100, 10),
            Err(ConfigError::TooManySymbols)
        ));
    }

    #[test]
    fn accessors() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(4, 3, 16).unwrap();
        assert_eq!(c.k(), 4);
        assert_eq!(c.m(), 3);
        assert_eq!(c.n(), 7);
        assert_eq!(c.symbol_len(), 16);
    }

    // ---- Encode ----

    #[test]
    fn encode_data_symbols_are_systematic() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data: Vec<u8> = vec![
            0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0,
        ];
        let symbols = c.encode(&data).unwrap();
        assert_eq!(symbols.len(), 5);
        // First k symbols are the data copies.
        for i in 0..3 {
            assert_eq!(symbols[i], &data[i * 4..(i + 1) * 4]);
        }
    }

    #[test]
    fn encode_repair_symbol_is_cauchy_combination() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(2, 1, 2).unwrap();
        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        let symbols = c.encode(&data).unwrap();
        // Repair symbol 0, byte p = A[0][0]*data[0*2+p] + A[1][0]*data[1*2+p]
        let a00 = c.cauchy.get(0, 0);
        let a10 = c.cauchy.get(1, 0);
        let expected_0 = GfElem(data[0]).mul(a00).0 ^ GfElem(data[2]).mul(a10).0;
        let expected_1 = GfElem(data[1]).mul(a00).0 ^ GfElem(data[3]).mul(a10).0;
        assert_eq!(symbols[2], vec![expected_0, expected_1]);
    }

    // ---- Decode: round-trip over all k-of-n subsets ----

    #[test]
    fn roundtrip_all_subsets_small() {
        let (k, m, symbol_len) = (3, 2, 4);
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, symbol_len).unwrap();
        let n = c.n();
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        let symbols = c.encode(&data).unwrap();

        for subset in k_subsets(n, k) {
            let received: Vec<(usize, &[u8])> = subset
                .iter()
                .map(|&idx| (idx, symbols[idx].as_slice()))
                .collect();
            let recovered = c
                .decode(&received)
                .unwrap_or_else(|e| panic!("decode failed for subset {:?}: {:?}", subset, e));
            assert_eq!(
                recovered, data,
                "round-trip mismatch for subset {:?}",
                subset
            );
        }
    }

    #[test]
    fn roundtrip_k4_m3() {
        let (k, m, symbol_len) = (4, 3, 8);
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, symbol_len).unwrap();
        let n = c.n();
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|i| (i as u8).wrapping_mul(11) ^ 0xA5)
            .collect();
        let symbols = c.encode(&data).unwrap();

        // Check a representative subset of k-of-n combinations (not all, since
        // C(7,4)=35 is manageable but we also run the proptest below).
        for subset in k_subsets(n, k) {
            let received: Vec<(usize, &[u8])> = subset
                .iter()
                .map(|&idx| (idx, symbols[idx].as_slice()))
                .collect();
            let recovered = c.decode(&received).unwrap();
            assert_eq!(recovered, data, "subset {:?}", subset);
        }
    }

    #[test]
    fn roundtrip_repair_only_recovery() {
        // Recover using only repair symbols (no data symbols at all).
        let (k, m, symbol_len) = (3, 3, 4);
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, symbol_len).unwrap();
        let data: Vec<u8> = vec![
            0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
        ];
        let symbols = c.encode(&data).unwrap();
        // Use repair symbols only: indices 3, 4, 5.
        let received: Vec<(usize, &[u8])> = vec![
            (3, symbols[3].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        let recovered = c.decode(&received).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn roundtrip_all_subsets_small_good_cauchy() {
        let (k, m, symbol_len) = (3, 2, 4);
        let c = BatchCodec::<crate::good_cauchy::GoodCauchyView>::new(k, m, symbol_len).unwrap();
        let n = c.n();
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        let symbols = c.encode(&data).unwrap();

        for subset in k_subsets(n, k) {
            let received: Vec<(usize, &[u8])> = subset
                .iter()
                .map(|&idx| (idx, symbols[idx].as_slice()))
                .collect();
            let recovered = c
                .decode(&received)
                .unwrap_or_else(|e| panic!("decode failed for subset {:?}: {:?}", subset, e));
            assert_eq!(
                recovered, data,
                "round-trip mismatch for subset {:?}",
                subset
            );
        }
    }

    // ---- Decode: error cases ----

    #[test]
    fn decode_wrong_count() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        // Only 2 symbols (need 3).
        let received: Vec<(usize, &[u8])> =
            vec![(0, symbols[0].as_slice()), (1, symbols[1].as_slice())];
        assert_eq!(
            c.decode(&received),
            Err(DecodeError::WrongCount {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn decode_index_out_of_range() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let bad = vec![0u8; 4];
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (1, symbols[1].as_slice()),
            (5, bad.as_slice()), // n=5, index 5 is out of range
        ];
        assert_eq!(
            c.decode(&received),
            Err(DecodeError::IndexOutOfRange { index: 5, n: 5 })
        );
    }

    #[test]
    fn decode_duplicate_index() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (1, symbols[1].as_slice()),
            (0, symbols[0].as_slice()),
        ];
        assert_eq!(
            c.decode(&received),
            Err(DecodeError::DuplicateIndex { index: 0 })
        );
    }

    #[test]
    fn decode_wrong_payload_len() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let bad = vec![0u8; 3];
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (1, symbols[1].as_slice()),
            (2, bad.as_slice()),
        ];
        assert_eq!(
            c.decode(&received),
            Err(DecodeError::WrongPayloadLen {
                expected: 4,
                got: 3
            })
        );
    }

    // ---- Property tests ----

    fn any_bytes(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(0u8..=255, len)
    }

    /// Resize an arbitrary byte vector to exactly `len`, truncating or
    /// zero-padding as needed.
    fn fit_bytes(mut v: Vec<u8>, len: usize) -> Vec<u8> {
        v.truncate(len);
        while v.len() < len {
            v.push(0);
        }
        v
    }

    proptest! {
        #[test]
        fn prop_roundtrip_all_subsets(
            k in 1usize..=4,
            m in 1usize..=4,
            symbol_len in 1usize..=8,
            data in any_bytes(32),
        ) {
            let c = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, symbol_len).unwrap();
            let n = c.n();
            let data = fit_bytes(data, k * symbol_len);
            let symbols = c.encode(&data).unwrap();

            for subset in k_subsets(n, k) {
                let received: Vec<(usize, &[u8])> = subset
                    .iter()
                    .map(|&idx| (idx, symbols[idx].as_slice()))
                    .collect();
                let recovered = c.decode(&received).unwrap();
                prop_assert_eq!(recovered, data.clone(), "subset {:?}", subset);
            }
        }

        #[test]
        fn prop_encode_repair_correctness(
            k in 1usize..=5,
            m in 1usize..=5,
            symbol_len in 1usize..=8,
            data in any_bytes(40),
        ) {
            let c = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, symbol_len).unwrap();
            let data = fit_bytes(data, k * symbol_len);
            let symbols = c.encode(&data).unwrap();
            // Verify each repair symbol independently.
            for j in 0..m {
                for p in 0..symbol_len {
                    let mut acc = GfElem::ZERO;
                    for i in 0..k {
                        let coeff = c.cauchy.get(i, j);
                        acc = acc.add(coeff.mul(GfElem(data[i * symbol_len + p])));
                    }
                    prop_assert_eq!(symbols[k + j][p], acc.0, "repair {} byte {}", j, p);
                }
            }
        }
    }
}
