//! Batch (non-streaming) Cauchy Reed-Solomon encode and decode.
//!
//! Convenience APIs allocate their returned buffers or temporary workspace.
//! Latency-sensitive callers can reuse [`DecodeScratch`] with
//! [`BatchCodec::decode_into_with`] for allocation-free steady-state decode.
//! Standard Cauchy supports `n <= 256`; Good Cauchy supports `n <= 255`.
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
#[cfg(test)]
use crate::matrix::{self, MatrixViewMut};

use crate::codec::{BatchDecoder, BatchEncoder, Coded};
use crate::error::{ConfigError, DecodeError, EncodeError};

/// A batch (non-streaming) Cauchy Reed-Solomon codec configuration.
///
/// Construct once via [`BatchCodec::new`], then use [`encode`][BatchCodec::encode]
/// and [`decode`][BatchCodec::decode] repeatedly. The codec stores only the
/// parameters `(k, m, symbol_len)` and a coding matrix view; it holds no per-call
/// state.
#[derive(Clone, Debug)]
pub struct BatchCodec<C: CodingMatrix> {
    k: usize,
    m: usize,
    symbol_len: usize,
    /// Kept for the test-suite reference paths; production encode/decode go
    /// through `coeffs` exclusively.
    #[cfg_attr(not(test), allow(dead_code))]
    cauchy: C,
    /// Precomputed source-major coefficient table (`i * m + j` = `C[i][j]`),
    /// built once at construction so encode avoids per-`(i, j)` lookups.
    coeffs: Vec<GfElem>,
}

/// Caller-owned workspace for allocation-free steady-state batch decode.
///
/// Construct with [`BatchCodec::decode_scratch`] and reuse it with
/// [`BatchCodec::decode_into_with`]. A scratch is tied to one `(k, m,
/// symbol_len)` geometry, but may be shared sequentially by codecs with the
/// same geometry.
#[derive(Debug)]
pub struct DecodeScratch {
    k: usize,
    m: usize,
    symbol_len: usize,
    repair_cols: Vec<usize>,
    work: Vec<u8>,
    missing: Vec<usize>,
    b: Vec<GfElem>,
    b_inv: Vec<GfElem>,
    flat_coeffs: Vec<GfElem>,
}

impl<C: CodingMatrix> BatchCodec<C> {
    /// Create a codec for `(k, m)` with symbols of `symbol_len` bytes.
    ///
    /// Returns [`ConfigError::ZeroDimension`] if `k` or `m` is zero,
    /// [`ConfigError::TooManySymbols`] if the selected `C` rejects `(k, m)`,
    /// and [`ConfigError::ZeroSymbolLen`] if `symbol_len` is zero.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        let cauchy = C::new(k, m).ok_or(ConfigError::TooManySymbols { cap: C::CAPACITY })?;
        let coeffs = cauchy.coefficient_matrix();
        Ok(Self {
            k,
            m,
            symbol_len,
            cauchy,
            coeffs,
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
    /// Returns [`EncodeError::WrongInputLen`] when `data` does not have
    /// exactly `k * symbol_len` bytes.
    ///
    /// The returned vector has length `n`; entry `i` is `symbol_len` bytes.
    /// Entries `0..k` are the data copied verbatim (systematic); entries
    /// `k..n` are repair symbols computed as Cauchy-weighted combinations of
    /// the data.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncodeError> {
        let expected = self.k * self.symbol_len;
        if data.len() != expected {
            return Err(EncodeError::WrongInputLen {
                expected,
                got: data.len(),
            });
        }
        let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(self.n());
        // Data symbols: copy verbatim (systematic).
        for i in 0..self.k {
            let start = i * self.symbol_len;
            symbols.push(data[start..start + self.symbol_len].to_vec());
        }
        // Repair symbols into a flat scratch, then split into per-symbol vecs.
        let mut repairs = vec![0u8; self.m * self.symbol_len];
        self.encode_into(data, &mut repairs)?;
        for j in 0..self.m {
            let start = j * self.symbol_len;
            symbols.push(repairs[start..start + self.symbol_len].to_vec());
        }
        Ok(symbols)
    }

    /// Encode the `m` repair symbols for `data` into `repairs`, allocation-free.
    ///
    /// `data` is `k * symbol_len` bytes; `repairs` is `m * symbol_len` bytes and
    /// is fully overwritten (`repairs[j * symbol_len ..]` = repair `j`).
    ///
    /// Repairs are computed **source-major, multi-destination**: each source
    /// symbol is read once and scattered into all `m` repair rows via the SIMD
    /// row kernel (the streaming encoder's hot path), rather than re-streaming
    /// every source per repair with a single-destination AXPY. Bit-identical to
    /// the repair-major reference (`encode_into_reference`).
    pub fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        let din = self.k * self.symbol_len;
        if data.len() != din {
            return Err(EncodeError::WrongInputLen {
                expected: din,
                got: data.len(),
            });
        }
        let dout = self.m * self.symbol_len;
        if repairs.len() != dout {
            return Err(EncodeError::WrongOutputLen {
                expected: dout,
                got: repairs.len(),
            });
        }
        repairs.fill(0);
        // One batched pass: every term is a (coefficient row, source symbol)
        // pair, so the SIMD backend is resolved once and destination tiles
        // are register-blocked across all `k` sources instead of being
        // re-streamed from memory per source.
        let mut term_storage: [core::mem::MaybeUninit<(&[GfElem], &[u8])>; 256] =
            [const { core::mem::MaybeUninit::uninit() }; 256];
        for (i, term) in term_storage.iter_mut().enumerate().take(self.k) {
            term.write((
                &self.coeffs[i * self.m..(i + 1) * self.m],
                &data[i * self.symbol_len..(i + 1) * self.symbol_len],
            ));
        }
        // SAFETY: construction guarantees `k <= 255`; entries `0..k` were
        // initialized exactly once above and the slice cannot outlive its
        // stack storage or borrowed codec/input data.
        let terms = unsafe {
            core::slice::from_raw_parts(term_storage.as_ptr().cast::<(&[GfElem], &[u8])>(), self.k)
        };
        crate::payload::xor_scaled_bytes_rows_terms(repairs, self.symbol_len, self.m, terms);
        Ok(())
    }

    /// Naive repair-major, single-destination reference encode into `repairs`.
    /// Correctness oracle for [`encode_into`](Self::encode_into): the fast path
    /// must produce bit-identical output for every configuration.
    #[cfg(test)]
    fn encode_into_reference(&self, data: &[u8], repairs: &mut [u8]) {
        for j in 0..self.m {
            let rstart = j * self.symbol_len;
            let repair = &mut repairs[rstart..rstart + self.symbol_len];
            repair.fill(0);
            for i in 0..self.k {
                let coefficient = self.cauchy.get(i, j);
                let data_start = i * self.symbol_len;
                crate::payload::xor_scaled_bytes(
                    repair,
                    coefficient,
                    &data[data_start..data_start + self.symbol_len],
                );
            }
        }
    }

    /// Decode any `k` of the `n` symbols back into the original
    /// `k * symbol_len` bytes.
    ///
    /// `symbols` is a slice of `(index, payload)` pairs where `index` is in
    /// `0..n` identifying which codeword symbol this is, and `payload` is
    /// `symbol_len` bytes. Exactly `k` pairs with distinct, in-range indices
    /// must be provided.
    ///
    /// The decoder solves only the erasure sub-system rather than reducing a
    /// full `k x k` augmented matrix. Let `E` be the missing data symbols
    /// (`e = |E|`), and let `R` be the `e` received repair symbols (exactly
    /// `e` repairs are present among any valid `k` distinct symbols). Then:
    ///
    /// 1. `e = 0`: all data present; the decode is a straight copy.
    /// 2. Otherwise, subtract every present data symbol's contribution from
    ///    the received repair payloads (`e x (k - e) x symbol_len` of SIMD
    ///    mul-add work), invert the `e x e` Cauchy submatrix
    ///    `B[t][s] = A[E[s]][R[t]]`, and apply it to the residual repairs
    ///    (`e x e x symbol_len` more).
    ///
    /// Payload cost is `O(e * k * symbol_len)` — proportional to encoding
    /// `e` repair symbols — instead of the `O(k^2 * symbol_len)` of a full
    /// RREF decode.
    pub fn decode(&self, symbols: &[(usize, &[u8])]) -> Result<Vec<u8>, DecodeError> {
        let mut out = vec![0u8; self.k * self.symbol_len];
        self.decode_into(symbols, &mut out)?;
        Ok(out)
    }

    /// Allocate reusable workspace for this codec's decode geometry.
    pub fn decode_scratch(&self) -> DecodeScratch {
        let max_e = self.k.min(self.m);
        DecodeScratch {
            k: self.k,
            m: self.m,
            symbol_len: self.symbol_len,
            repair_cols: Vec::with_capacity(max_e),
            work: vec![0u8; max_e * self.symbol_len],
            missing: Vec::with_capacity(max_e),
            b: vec![GfElem::ZERO; max_e * max_e],
            b_inv: vec![GfElem::ZERO; max_e * max_e],
            flat_coeffs: vec![GfElem::ZERO; self.k * max_e],
        }
    }

    /// Decode into `out`, allocating a temporary workspace.
    ///
    /// Call [`decode_into_with`](Self::decode_into_with) with a reused
    /// [`DecodeScratch`] when allocation must be excluded from the hot path.
    pub fn decode_into(
        &self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let mut scratch = self.decode_scratch();
        self.decode_into_with(symbols, out, &mut scratch)
    }

    /// Decode any `k` of the `n` symbols into `out` without heap allocation.
    ///
    /// `scratch` must come from a codec with the same geometry. After
    /// [`decode_scratch`](Self::decode_scratch), repeated successful calls to
    /// this method allocate nothing.
    pub fn decode_into_with(
        &self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
        scratch: &mut DecodeScratch,
    ) -> Result<(), DecodeError> {
        let k = self.k;
        let m = self.m;
        let n = self.n();
        let symbol_len = self.symbol_len;

        if (scratch.k, scratch.m, scratch.symbol_len) != (k, m, symbol_len) {
            return Err(DecodeError::ScratchMismatch);
        }
        let expected = k * symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        if symbols.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: symbols.len(),
            });
        }

        debug_assert!(n <= 256);
        let mut seen = [false; 256];
        scratch.repair_cols.clear();
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
            if idx < k {
                out[idx * symbol_len..(idx + 1) * symbol_len].copy_from_slice(payload);
            } else {
                let repair = scratch.repair_cols.len();
                scratch.repair_cols.push(idx - k);
                scratch.work[repair * symbol_len..(repair + 1) * symbol_len]
                    .copy_from_slice(payload);
            }
        }

        let e = scratch.repair_cols.len();
        if e == 0 {
            return Ok(());
        }

        scratch.missing.clear();
        for (i, &was_seen) in seen[..k].iter().enumerate() {
            if !was_seen {
                scratch.missing.push(i);
            }
        }
        debug_assert_eq!(scratch.missing.len(), e);

        let present = k - e;
        let flat_coeffs = &mut scratch.flat_coeffs[..present * e];
        let mut w = 0;
        for &(idx, _) in symbols {
            if idx >= k {
                continue;
            }
            for (t, &col) in scratch.repair_cols.iter().enumerate() {
                flat_coeffs[w * e + t] = self.coeffs[idx * m + col];
            }
            w += 1;
        }
        debug_assert_eq!(w, present);

        // Avoid both a heap allocation and zero-initializing the maximum
        // 256-entry descriptor array. Every prefix element is written before
        // the kernel borrows it, and the tuples contain borrowed slices only.
        let mut term_storage: [core::mem::MaybeUninit<(&[GfElem], &[u8])>; 256] =
            [const { core::mem::MaybeUninit::uninit() }; 256];
        let mut term = 0;
        for &(idx, payload) in symbols {
            if idx >= k {
                continue;
            }
            term_storage[term].write((&flat_coeffs[term * e..(term + 1) * e], payload));
            term += 1;
        }
        debug_assert_eq!(term, present);
        // SAFETY: entries `0..present` were initialized exactly once above;
        // the resulting slice does not outlive `term_storage` or its inputs.
        let terms = unsafe {
            core::slice::from_raw_parts(term_storage.as_ptr().cast::<(&[GfElem], &[u8])>(), present)
        };
        let work = &mut scratch.work[..e * symbol_len];
        crate::payload::xor_scaled_bytes_rows_terms(work, symbol_len, e, terms);

        let matrix_len = e * e;
        let b = &mut scratch.b[..matrix_len];
        let b_inv = &mut scratch.b_inv[..matrix_len];
        for (t, &col) in scratch.repair_cols.iter().enumerate() {
            for (s, &row) in scratch.missing.iter().enumerate() {
                b[t * e + s] = self.coeffs[row * m + col];
            }
        }
        if !invert_square_into(b, e, b_inv) {
            return Err(DecodeError::InsufficientRank { rank: e - 1, k: e });
        }

        for (s, &row) in scratch.missing.iter().enumerate() {
            let dst = &mut out[row * symbol_len..(row + 1) * symbol_len];
            dst.fill(0);
            for t in 0..e {
                crate::payload::xor_scaled_bytes(
                    dst,
                    b_inv[s * e + t],
                    &work[t * symbol_len..(t + 1) * symbol_len],
                );
            }
        }
        Ok(())
    }

    /// Naive full-RREF reference decode. Correctness oracle for
    /// [`decode_into`](Self::decode_into): the fast path must produce
    /// bit-identical output for every valid symbol selection.
    #[cfg(test)]
    fn decode_reference(&self, symbols: &[(usize, &[u8])]) -> Result<Vec<u8>, DecodeError> {
        let k = self.k;
        let symbol_len = self.symbol_len;

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
impl<C: CodingMatrix> Coded for BatchCodec<C> {
    fn k(&self) -> usize {
        self.k
    }

    fn m(&self) -> usize {
        self.m
    }

    fn symbol_len(&self) -> usize {
        self.symbol_len
    }
}

impl<C: CodingMatrix> BatchEncoder for BatchCodec<C> {
    type Scratch = ();

    fn scratch(&self) {}

    fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        BatchCodec::encode_into(self, data, repairs)
    }

    fn encode_into_with(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        _scratch: &mut Self::Scratch,
    ) -> Result<(), EncodeError> {
        BatchCodec::encode_into(self, data, repairs)
    }
}

impl<C: CodingMatrix> BatchDecoder for BatchCodec<C> {
    type Scratch = DecodeScratch;

    fn scratch(&self) -> Self::Scratch {
        self.decode_scratch()
    }

    fn decode_into(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
    ) -> Result<(), DecodeError> {
        BatchCodec::decode_into(self, symbols, out)
    }

    fn decode_into_with(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError> {
        BatchCodec::decode_into_with(self, symbols, out, scratch)
    }
}

/// Invert a small row-major `e x e` GF(256) matrix in place via Gauss-Jordan
/// with partial pivoting, writing the row-major inverse into `inv`.
/// Returns `false` when the matrix is singular.
///
/// `e` is the erasure count on the decode hot path (typically 1-4), so the
/// scalar `O(e^3)` cost is negligible next to the payload arithmetic.
fn invert_square_into(matrix: &mut [GfElem], e: usize, inv: &mut [GfElem]) -> bool {
    debug_assert_eq!(matrix.len(), e * e);
    debug_assert_eq!(inv.len(), e * e);
    inv.fill(GfElem::ZERO);
    for i in 0..e {
        inv[i * e + i] = GfElem::ONE;
    }
    for col in 0..e {
        let Some(pivot) = (col..e).find(|&r| matrix[r * e + col] != GfElem::ZERO) else {
            return false;
        };
        if pivot != col {
            for c in 0..e {
                matrix.swap(col * e + c, pivot * e + c);
                inv.swap(col * e + c, pivot * e + c);
            }
        }
        let scale = matrix[col * e + col].inv();
        for c in 0..e {
            matrix[col * e + c] = matrix[col * e + c].mul(scale);
            inv[col * e + c] = inv[col * e + c].mul(scale);
        }
        for r in 0..e {
            if r == col {
                continue;
            }
            let factor = matrix[r * e + col];
            if factor == GfElem::ZERO {
                continue;
            }
            for c in 0..e {
                matrix[r * e + c] = matrix[r * e + c].add(factor.mul(matrix[col * e + c]));
                inv[r * e + c] = inv[r * e + c].add(factor.mul(inv[col * e + c]));
            }
        }
    }
    true
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
            Err(ConfigError::TooManySymbols { cap: 256 })
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

    #[test]
    fn aliases_select_and_roundtrip_with_their_matrix() {
        use crate::batch::{GoodCauchyBatchCodec, StandardCauchyBatchCodec};
        use crate::decoder::LazyDecoderState;

        let data = vec![0x5a; 6];
        let good = GoodCauchyBatchCodec::new(3, 2, 2).unwrap();
        let good_symbols = good.encode(&data).unwrap();
        let mut good_decoder =
            LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(3, 2, 2).unwrap();
        for (index, symbol) in good_symbols.iter().take(3).enumerate() {
            good_decoder.push_symbol(index, symbol).unwrap();
        }
        assert_eq!(good_decoder.finalize_ref().unwrap(), data);

        let standard = StandardCauchyBatchCodec::new(3, 2, 2).unwrap();
        let standard_symbols = standard.encode(&data).unwrap();
        let mut standard_decoder =
            LazyDecoderState::<crate::cauchy::CauchyView>::new(3, 2, 2).unwrap();
        for (index, symbol) in standard_symbols.iter().take(3).enumerate() {
            standard_decoder.push_symbol(index, symbol).unwrap();
        }
        assert_eq!(standard_decoder.finalize_ref().unwrap(), data);
    }

    #[test]
    fn encode_rejects_wrong_input_length_with_details() {
        let codec = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        assert_eq!(
            codec.encode(&[0; 11]),
            Err(EncodeError::WrongInputLen {
                expected: 12,
                got: 11
            })
        );
    }

    #[test]
    fn batch_matrix_capacity_boundaries_are_exact() {
        assert!(crate::batch::StandardCauchyBatchCodec::new(255, 1, 1).is_ok());
        assert!(crate::batch::StandardCauchyBatchCodec::new(255, 2, 1).is_err());
        assert!(crate::batch::GoodCauchyBatchCodec::new(254, 1, 1).is_ok());
        assert!(crate::batch::GoodCauchyBatchCodec::new(255, 1, 1).is_err());
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
    #[test]
    fn optimized_decode_matches_reference() {
        use crate::cauchy::CauchyView;
        use crate::good_cauchy::GoodCauchyView;
        // (k, m, slen); subsets are sampled deterministically: e0 (all data),
        // single erasures, erasure pairs, high-erasure mixes, all-repairs.
        let cases = [
            (1usize, 1usize, 2usize),
            (4, 2, 16),
            (8, 4, 64),
            (6, 6, 33),
            (16, 8, 100),
            (32, 16, 64),
        ];
        for &(k, m, slen) in &cases {
            let n = k + m;
            let data: Vec<u8> = (0..k * slen)
                .map(|x| (x.wrapping_mul(131) + 7) as u8)
                .collect();
            // Deterministic subset sampler: every k-subset whose indices are
            // generated by rotating erasure positions, capped at 64 subsets.
            let mut subsets: Vec<Vec<usize>> = Vec::new();
            subsets.push((0..k).collect()); // e0
            for erase_count in 1..=m.min(k) {
                for shift in 0..(k.max(1)) {
                    if subsets.len() >= 64 {
                        break;
                    }
                    let mut erased: Vec<usize> =
                        (0..erase_count).map(|t| (shift + t * 3) % k).collect();
                    erased.sort_unstable();
                    erased.dedup();
                    let mut subset: Vec<usize> = (0..k).filter(|i| !erased.contains(i)).collect();
                    subset.extend((0..erased.len()).map(|t| k + (shift + t) % m));
                    subset.sort_unstable();
                    if !subsets.contains(&subset) {
                        subsets.push(subset);
                    }
                }
            }
            if m >= k {
                subsets.push((k..n).collect()); // all repairs
            }
            macro_rules! check {
                ($c:expr, $name:literal) => {
                    let symbols = $c.encode(&data).unwrap();
                    for subset in &subsets {
                        let received: Vec<(usize, &[u8])> = subset
                            .iter()
                            .map(|&idx| (idx, symbols[idx].as_slice()))
                            .collect();
                        let fast = $c.decode(&received).unwrap();
                        let refr = $c.decode_reference(&received).unwrap();
                        assert_eq!(
                            fast, refr,
                            "{} k={} m={} slen={} subset={:?}",
                            $name, k, m, slen, subset
                        );
                        assert_eq!(
                            fast, data,
                            "{} roundtrip k={} m={} subset={:?}",
                            $name, k, m, subset
                        );
                    }
                };
            }
            if n <= 255 {
                check!(
                    BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap(),
                    "good-cauchy"
                );
            }
            if n <= 256 {
                check!(
                    BatchCodec::<CauchyView>::new(k, m, slen).unwrap(),
                    "standard-cauchy"
                );
            }
        }
    }

    #[test]
    fn decode_into_rejects_wrong_output_len() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (1, symbols[1].as_slice()),
            (2, symbols[2].as_slice()),
        ];
        let mut out = vec![0u8; 11];
        assert_eq!(
            c.decode_into(&received, &mut out),
            Err(DecodeError::WrongOutputLen {
                expected: 12,
                got: 11
            })
        );
    }

    #[test]
    fn decode_into_with_rejects_scratch_from_another_geometry() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let other = BatchCodec::<crate::cauchy::CauchyView>::new(4, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let received: Vec<(usize, &[u8])> = (0..3).map(|i| (i, symbols[i].as_slice())).collect();
        let mut out = vec![0u8; 12];
        let mut scratch = other.decode_scratch();
        assert_eq!(
            c.decode_into_with(&received, &mut out, &mut scratch),
            Err(DecodeError::ScratchMismatch)
        );
    }

    #[test]
    fn decode_into_overwrites_garbage_output() {
        // decode_into must not rely on the output buffer being zeroed.
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(4, 3, 17).unwrap();
        let data: Vec<u8> = (0..4 * 17)
            .map(|x| (x as u8).wrapping_mul(29) ^ 0x5C)
            .collect();
        let symbols = c.encode(&data).unwrap();
        // Erase data symbols 1 and 3.
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (2, symbols[2].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        let mut out = vec![0xABu8; 4 * 17];
        c.decode_into(&received, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn optimized_encode_matches_reference() {
        use crate::cauchy::CauchyView;
        use crate::good_cauchy::GoodCauchyView;
        let cases = [
            (1usize, 1usize, 2usize),
            (4, 2, 16),
            (8, 4, 64),
            (16, 8, 1024),
            (32, 16, 1024),
            (128, 64, 1024),
            (200, 50, 256),
            (1, 254, 64),
            (254, 1, 64),
        ];
        for &(k, m, slen) in &cases {
            let data: Vec<u8> = (0..k * slen)
                .map(|x| (x.wrapping_mul(131) + 7) as u8)
                .collect();
            if k + m <= 255 {
                let c = BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap();
                let mut fast = vec![0u8; m * slen];
                let mut refr = vec![0u8; m * slen];
                c.encode_into(&data, &mut fast).unwrap();
                c.encode_into_reference(&data, &mut refr);
                assert_eq!(fast, refr, "good-cauchy k={k} m={m} slen={slen}");
            }
            if k + m <= 256 {
                let c = BatchCodec::<CauchyView>::new(k, m, slen).unwrap();
                let mut fast = vec![0u8; m * slen];
                let mut refr = vec![0u8; m * slen];
                c.encode_into(&data, &mut fast).unwrap();
                c.encode_into_reference(&data, &mut refr);
                assert_eq!(fast, refr, "standard-cauchy k={k} m={m} slen={slen}");
            }
        }
    }
}

#[cfg(test)]
mod _bench_tmp {
    use super::*;
    use crate::good_cauchy::GoodCauchyView;
    use std::time::Instant;
    fn best(mut f: impl FnMut(), iters: usize) -> u128 {
        for _ in 0..20 {
            f();
        }
        let mut b = u128::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            f();
            b = b.min(t.elapsed().as_nanos());
        }
        b
    }
    /// Interleaved A/B measurement: alternate short rounds of each closure so
    /// frequency scaling and cache state affect both equally; report the
    /// per-round minimum for each.
    fn best2(
        mut a: impl FnMut(),
        mut b: impl FnMut(),
        rounds: usize,
        iters: usize,
    ) -> (u128, u128) {
        for _ in 0..20 {
            a();
            b();
        }
        let (mut ba, mut bb) = (u128::MAX, u128::MAX);
        for _ in 0..rounds {
            for _ in 0..iters {
                let t = Instant::now();
                a();
                ba = ba.min(t.elapsed().as_nanos());
            }
            for _ in 0..iters {
                let t = Instant::now();
                b();
                bb = bb.min(t.elapsed().as_nanos());
            }
        }
        (ba, bb)
    }
    #[test]
    fn bench_encode_opt_vs_naive() {
        let slen = 1024;
        for &(k, m) in &[
            (8usize, 4usize),
            (16, 4),
            (32, 16),
            (64, 32),
            (128, 64),
            (221, 34),
        ] {
            let c = BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap();
            let data: Vec<u8> = (0..k * slen)
                .map(|x| (x.wrapping_mul(131) + 7) as u8)
                .collect();
            let mut fast = vec![0u8; m * slen];
            let mut refr = vec![0u8; m * slen];
            let (opt, naive) = best2(
                || {
                    c.encode_into(std::hint::black_box(&data), std::hint::black_box(&mut fast))
                        .unwrap();
                },
                || {
                    c.encode_into_reference(
                        std::hint::black_box(&data),
                        std::hint::black_box(&mut refr),
                    );
                },
                10,
                500,
            );
            println!(
                "k={k:3} m={m:3}  naive={naive:>8}ns  opt={opt:>8}ns  speedup={:.2}x",
                naive as f64 / opt as f64
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,gfni")]
    unsafe fn gfni_chain_probe(iters: u64) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let f = _mm256_set1_epi8(0x53);
        let mut a0 = _mm256_set1_epi8(1);
        let mut a1 = _mm256_set1_epi8(2);
        let mut a2 = _mm256_set1_epi8(3);
        let mut a3 = _mm256_set1_epi8(4);
        let mut a4 = _mm256_set1_epi8(5);
        let mut a5 = _mm256_set1_epi8(6);
        let mut a6 = _mm256_set1_epi8(7);
        let mut a7 = _mm256_set1_epi8(8);
        for _ in 0..iters {
            a0 = _mm256_gf2p8mul_epi8(a0, f);
            a1 = _mm256_gf2p8mul_epi8(a1, f);
            a2 = _mm256_gf2p8mul_epi8(a2, f);
            a3 = _mm256_gf2p8mul_epi8(a3, f);
            a4 = _mm256_gf2p8mul_epi8(a4, f);
            a5 = _mm256_gf2p8mul_epi8(a5, f);
            a6 = _mm256_gf2p8mul_epi8(a6, f);
            a7 = _mm256_gf2p8mul_epi8(a7, f);
        }
        _mm256_xor_si256(
            _mm256_xor_si256(a0, a1),
            _mm256_xor_si256(
                _mm256_xor_si256(a2, a3),
                _mm256_xor_si256(_mm256_xor_si256(a4, a5), _mm256_xor_si256(a6, a7)),
            ),
        )
    }

    #[test]
    fn bench_gfni_ceiling() {
        // Pure GF2P8MULB throughput probe: 8 independent accumulator chains,
        // no memory traffic. Reports muls/ns => the hardware ceiling for any
        // GFNI encode/decode kernel on this machine.
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("avx2") {
                let iters = 20_000_000u64;
                let sink = unsafe { gfni_chain_probe(100) };
                std::hint::black_box(sink);
                let t = Instant::now();
                let sink = unsafe { gfni_chain_probe(iters) };
                let ns = t.elapsed().as_nanos() as f64;
                std::hint::black_box(sink);
                let total = iters as f64 * 8.0;
                println!(
                    "gf2p8mulb ymm: {:.2} muls/ns ({:.1} GB/s dst-equivalent)",
                    total / ns,
                    total * 32.0 / ns
                );
            }
        }
    }

    #[test]
    fn bench_decode_opt_vs_naive() {
        let slen = 1024;
        for &(k, m) in &[
            (8usize, 4usize),
            (16, 4),
            (32, 16),
            (64, 32),
            (128, 64),
            (221, 34),
        ] {
            let c = BatchCodec::<GoodCauchyView>::new(k, m, slen).unwrap();
            let data: Vec<u8> = (0..k * slen)
                .map(|x| (x.wrapping_mul(131) + 7) as u8)
                .collect();
            let symbols = c.encode(&data).unwrap();
            // e0: first k symbols (all data). e2: drop data 0 and 1, take
            // repairs k and k+1 instead.
            let e0: Vec<(usize, &[u8])> = (0..k).map(|i| (i, symbols[i].as_slice())).collect();
            let e2: Vec<(usize, &[u8])> = (2..k + 2).map(|i| (i, symbols[i].as_slice())).collect();
            // Reference RREF decode is O(k^2 * slen) scalar: keep its
            // iteration count low at large k so the bench stays fast.
            let naive_iters = if k > 64 { 3 } else { 20 };
            for (name, recv) in [("e0", &e0), ("e2", &e2)] {
                let opt = best(
                    || {
                        std::hint::black_box(c.decode(std::hint::black_box(recv)).unwrap());
                    },
                    2000,
                );
                let naive = best(
                    || {
                        std::hint::black_box(
                            c.decode_reference(std::hint::black_box(recv)).unwrap(),
                        );
                    },
                    naive_iters,
                );
                println!(
                    "k={k:3} m={m:3} {name}  naive={naive:>9}ns  opt={opt:>8}ns  speedup={:.2}x",
                    naive as f64 / opt as f64
                );
            }
            // Compare the convenience API's temporary scratch with the
            // caller-owned workspace used by latency-sensitive adapters.
            let mut out = vec![0u8; k * slen];
            let mut scratch = c.decode_scratch();
            for (name, recv) in [("e0", &e0), ("e2", &e2)] {
                let allocating = best(
                    || {
                        c.decode_into(std::hint::black_box(recv), std::hint::black_box(&mut out))
                            .unwrap();
                    },
                    5000,
                );
                let reused = best(
                    || {
                        c.decode_into_with(
                            std::hint::black_box(recv),
                            std::hint::black_box(&mut out),
                            std::hint::black_box(&mut scratch),
                        )
                        .unwrap();
                    },
                    5000,
                );
                println!(
                    "k={k:3} m={m:3} {name}  decode_into={allocating:>8}ns  scratch={reused:>8}ns"
                );
            }
            if k >= 32 {
                // Breakdown at e2: survivor copy only, subtract pass only.
                let copy_t = best(
                    || {
                        for &(idx, payload) in e2.iter().take(k - 2) {
                            out[idx * slen..(idx + 1) * slen]
                                .copy_from_slice(std::hint::black_box(payload));
                        }
                        std::hint::black_box(&mut out);
                    },
                    2000,
                );
                let mut work = vec![0u8; 2 * slen];
                let coeff_pairs: Vec<[GfElem; 2]> = (2..k)
                    .map(|i| [c.coeffs[i * m], c.coeffs[i * m + 1]])
                    .collect();
                let terms: Vec<(&[GfElem], &[u8])> = coeff_pairs
                    .iter()
                    .map(|p| &p[..])
                    .zip(e2.iter().take(k - 2).map(|&(_, p)| p))
                    .collect();
                let work_ptr = work.as_mut_ptr();
                // SAFETY: each closure reborrows the same 2*slen buffer
                // sequentially inside best2; the closures never run
                // concurrently and `work` outlives both.
                let (sub_terms, sub_persrc) = best2(
                    || {
                        let w = unsafe { core::slice::from_raw_parts_mut(work_ptr, 2 * slen) };
                        crate::payload::xor_scaled_bytes_rows_terms(
                            std::hint::black_box(w),
                            slen,
                            2,
                            std::hint::black_box(&terms),
                        );
                    },
                    || {
                        let w = unsafe { core::slice::from_raw_parts_mut(work_ptr, 2 * slen) };
                        for (s, &(_, payload)) in e2.iter().take(k - 2).enumerate() {
                            let pair = &coeff_pairs[s];
                            crate::payload::xor_scaled_bytes_rows(
                                std::hint::black_box(w),
                                slen,
                                std::hint::black_box(pair),
                                std::hint::black_box(payload),
                            );
                            let _ = s;
                        }
                    },
                    10,
                    500,
                );
                println!(
                    "k={k:3} m={m:3} e2  parts: survivor_copy={copy_t:>6}ns subtract: terms={sub_terms:>6}ns persrc={sub_persrc:>6}ns"
                );
            }
        }
    }
}
