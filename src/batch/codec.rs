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
use fgf::gf8::Elem as GfElem;
#[cfg(test)]
use gfm::{Matrix, Ple, PleScratch};

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
    /// Coding matrix view. Encode uses the materialized `coeffs`; the fused
    /// decode derives its reduced-system variables (`x_var`/`y_var`) from this.
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
///
/// The scratch also memoizes the fused reconstruction coefficients of the
/// most recent receipt pattern: repeated decodes of one erasure pattern pay
/// only validation and payload arithmetic.
#[derive(Debug)]
pub struct DecodeScratch {
    k: usize,
    m: usize,
    symbol_len: usize,
    repair_cols: Vec<usize>,
    present: Vec<usize>,
    missing: Vec<usize>,
    vars: Vec<GfElem>,
    inverse: Vec<GfElem>,
    lagrange: Vec<GfElem>,
    flat_coeffs: Vec<GfElem>,
    work: Vec<u8>,
    cached_pattern: Option<crate::decoder::pattern::PatternKey>,
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl DecodeScratch {
    /// Data-symbol count this scratch was sized for; a decode with a different
    /// `k` is rejected as [`DecodeError::ScratchMismatch`].
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Repair-symbol count this scratch was sized for.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Per-symbol byte length this scratch was sized for.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Repair columns `j` (i.e. codeword index minus `k`) of the repair symbols
    /// supplied to the last decode, in the order they were presented.
    ///
    /// Its length is the erasure count `e` and indexes the rows of the reduced
    /// system; capacity is fixed at `min(k, m)`.
    #[must_use]
    pub fn repair_cols(&self) -> &[usize] {
        &self.repair_cols
    }

    /// Data indices of the received data symbols from the last decode, in the
    /// order they were presented.
    ///
    /// Its length is `k - e`; capacity is fixed at `k`.
    #[must_use]
    pub fn present(&self) -> &[usize] {
        &self.present
    }

    /// Contiguous reconstruction staging, `e` rows of `symbol_len` bytes.
    ///
    /// The fused kernel needs adjacent destination rows, so a full
    /// [`decode_into_with`](BatchCodec::decode_into_with) with `e > 1`
    /// reconstructs here and then scatters the rows into the caller output.
    /// Allocated at the maximum `min(k, m) * symbol_len`; only the first
    /// `e * symbol_len` bytes are meaningful after a decode.
    #[must_use]
    pub fn work(&self) -> &[u8] {
        &self.work
    }

    /// Ascending indices of the data symbols absent from the last decode.
    ///
    /// It has the same length `e` as [`repair_cols`](Self::repair_cols) and
    /// selects the columns of the reduced system.
    #[must_use]
    pub fn missing(&self) -> &[usize] {
        &self.missing
    }

    /// Inverse of the reduced `e x e` system, row-major as
    /// `[missing_row][repair_row]` over the first `e * e` entries.
    ///
    /// This is the rational-Lagrange closed form, not a Gauss-Jordan product.
    /// Allocated at the maximum `min(k, m)^2`.
    #[must_use]
    pub fn inverse(&self) -> &[GfElem] {
        &self.inverse
    }

    /// Fused source-major reconstruction coefficients over the first
    /// `k * e` entries: the first `e * e` are the repair terms (the reduced
    /// inverse transposed to `[repair_row][missing_row]`), the remaining
    /// `(k - e) * e` are the present-data terms in arrival order.
    ///
    /// Each length-`e` row pairs with one received source symbol, so a single
    /// fused matrix kernel call reconstructs every missing row.
    #[must_use]
    pub fn flat_coeffs(&self) -> &[GfElem] {
        &self.flat_coeffs
    }
}

/// A prepared batch decode plan for one erasure pattern.
///
/// Built by [`BatchCodec::prepare_decode`]. The plan owns the fused
/// reconstruction coefficients, the receipt-pattern membership test, and the
/// staging the full-decode path needs, so applying it costs symbol
/// validation plus one matrix kernel call — no heap allocation, no
/// coefficient construction, no partitioning.
///
/// Unlike [`DecodeScratch`]'s single-pattern memoization, a plan is an
/// explicit, clonable artifact: build one per scheduled loss pattern (or per
/// receiver feedback) and keep them all resident.
#[derive(Clone, Debug)]
pub struct DecodePlan {
    k: usize,
    m: usize,
    symbol_len: usize,
    /// Ascending missing data indices; its length is the erasure count `r`.
    missing_data: Vec<usize>,
    /// Receipt-pattern membership test.
    pattern: crate::decoder::pattern::PatternKey,
    /// Codeword index -> coefficient-row ordinal; `u16::MAX` when the symbol
    /// is not part of the prepared receipt.
    term_of: [u16; 256],
    /// Fused coefficients, `(k, r)` row-major with one row per source term
    /// (repair terms first, then present terms, arrival order). GF(2^8)
    /// kernel preparation is a static-table borrow, so the raw elements are
    /// already the backend-optimal form.
    coefficients: Box<[GfElem]>,
    /// Contiguous reconstruction staging for the scattered full-decode
    /// output, `r * symbol_len` bytes.
    staging: Vec<u8>,
}

impl DecodePlan {
    /// Data-symbol count of the geometry this plan was built for.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Repair-symbol count of the geometry this plan was built for.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Per-symbol byte length.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of erased data symbols this plan reconstructs.
    #[must_use]
    pub fn erasure_count(&self) -> usize {
        self.missing_data.len()
    }

    /// Ascending data indices reconstructed by this plan.
    #[must_use]
    pub fn missing_indices(&self) -> &[usize] {
        &self.missing_data
    }

    /// Validate one symbol against the prepared pattern and return its
    /// coefficient-row ordinal.
    fn term_ordinal(
        &self,
        idx: usize,
        payload: &[u8],
        seen: &mut [bool; 256],
    ) -> Result<usize, DecodeError> {
        let n = self.k + self.m;
        if idx >= n {
            return Err(DecodeError::IndexOutOfRange { index: idx, n });
        }
        if !self.pattern.get(idx) {
            return Err(DecodeError::UnexpectedIndex { index: idx });
        }
        if seen[idx] {
            return Err(DecodeError::DuplicateIndex { index: idx });
        }
        seen[idx] = true;
        if payload.len() != self.symbol_len {
            return Err(DecodeError::WrongPayloadLen {
                expected: self.symbol_len,
                got: payload.len(),
            });
        }
        Ok(self.term_of[idx] as usize)
    }

    /// Reconstruct only the missing data symbols into `missing_out`
    /// (`erasure_count * symbol_len` bytes, ascending data-index order).
    ///
    /// `symbols` must be exactly the `k` received symbols the plan was
    /// prepared for, in any order. Allocates nothing.
    pub fn reconstruct_missing_into(
        &self,
        symbols: &[(usize, &[u8])],
        missing_out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let r = self.erasure_count();
        if symbols.len() != self.k {
            return Err(DecodeError::WrongCount {
                expected: self.k,
                got: symbols.len(),
            });
        }
        let expected = r * self.symbol_len;
        if missing_out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: missing_out.len(),
            });
        }
        let mut seen = [false; 256];
        let mut term_storage = [(&[][..], &[][..]); 256];
        for &(idx, payload) in symbols {
            let ordinal = self.term_ordinal(idx, payload, &mut seen)?;
            term_storage[ordinal] = (&self.coefficients[ordinal * r..(ordinal + 1) * r], payload);
        }
        if r == 0 {
            return Ok(());
        }
        missing_out.fill(0);
        let terms = &term_storage[..self.k];
        crate::payload::xor_scaled_bytes_rows_terms(missing_out, self.symbol_len, r, terms);
        Ok(())
    }

    /// Decode all `k` data symbols into `out` (`k * symbol_len` bytes):
    /// surviving rows are copied, missing rows reconstructed.
    ///
    /// `symbols` must be exactly the `k` received symbols the plan was
    /// prepared for, in any order. Allocates nothing.
    pub fn decode_into(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let k = self.k;
        let symbol_len = self.symbol_len;
        let r = self.erasure_count();
        if symbols.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: symbols.len(),
            });
        }
        let expected = k * symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let mut seen = [false; 256];
        let mut term_storage = [(&[][..], &[][..]); 256];
        for &(idx, payload) in symbols {
            let ordinal = self.term_ordinal(idx, payload, &mut seen)?;
            term_storage[ordinal] = (&self.coefficients[ordinal * r..(ordinal + 1) * r], payload);
            if idx < k {
                out[idx * symbol_len..(idx + 1) * symbol_len].copy_from_slice(payload);
            }
        }
        if r == 0 {
            return Ok(());
        }
        let terms = &term_storage[..k];
        if r == 1 {
            // One missing row is trivially contiguous: reconstruct straight
            // into the output and skip the staging round trip.
            let start = self.missing_data[0] * symbol_len;
            let dst = &mut out[start..start + symbol_len];
            dst.fill(0);
            crate::payload::xor_scaled_bytes_rows_terms(dst, symbol_len, 1, terms);
            return Ok(());
        }
        {
            let dst = &mut self.staging[..r * symbol_len];
            dst.fill(0);
            crate::payload::xor_scaled_bytes_rows_terms(dst, symbol_len, r, terms);
        }
        for (row, &data_idx) in self.missing_data.iter().enumerate() {
            out[data_idx * symbol_len..(data_idx + 1) * symbol_len]
                .copy_from_slice(&self.staging[row * symbol_len..(row + 1) * symbol_len]);
        }
        Ok(())
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl DecodePlan {
    /// Fused coefficients as a `(k, r)` row-major matrix: one row per
    /// received source symbol, in repair-then-present term order.
    #[must_use]
    pub fn coefficients(&self) -> &[GfElem] {
        &self.coefficients
    }

    /// The receipt pattern this plan was prepared for.
    #[must_use]
    pub const fn pattern(&self) -> crate::decoder::pattern::PatternKey {
        self.pattern
    }

    /// Codeword index -> coefficient-row ordinal map; `u16::MAX` marks
    /// symbols outside the prepared receipt.
    #[must_use]
    pub fn term_ordinals(&self) -> &[u16; 256] {
        &self.term_of
    }
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
        let mut term_storage = [(&[][..], &[][..]); 256];
        for (i, term) in term_storage.iter_mut().enumerate().take(self.k) {
            *term = (
                &self.coeffs[i * self.m..(i + 1) * self.m],
                &data[i * self.symbol_len..(i + 1) * self.symbol_len],
            );
        }
        let terms = &term_storage[..self.k];
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
    /// 2. Otherwise, a rational-Lagrange closed form yields the `e x e`
    ///    inverse of the reduced Cauchy submatrix together with already-fused
    ///    coefficients for the surviving data symbols, and one fused
    ///    source-major matrix pass combines all `k` received symbols into the
    ///    `e` missing rows.
    ///
    /// Coefficient cost is `O(e^2 + e * k)` field operations; payload cost is
    /// one `O(e * k * symbol_len)` pass — proportional to encoding `e`
    /// repair symbols — instead of the `O(k^2 * symbol_len)` of a full RREF
    /// decode.
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
            present: Vec::with_capacity(self.k),
            missing: Vec::with_capacity(max_e),
            vars: vec![GfElem::ZERO; 2 * max_e],
            inverse: vec![GfElem::ZERO; max_e * max_e],
            lagrange: vec![GfElem::ZERO; gfm::cauchy_scratch_len(max_e)],
            flat_coeffs: vec![GfElem::ZERO; self.k * max_e],
            work: vec![0u8; max_e * self.symbol_len],
            cached_pattern: None,
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
        let symbol_len = self.symbol_len;
        self.check_decode_input(symbols, scratch)?;
        let expected = k * symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let (r, pattern) = self.partition_symbols(symbols, scratch)?;

        // Copy present data symbols directly.
        for &(idx, payload) in symbols {
            if idx < k {
                out[idx * symbol_len..(idx + 1) * symbol_len].copy_from_slice(payload);
            }
        }
        if r == 0 {
            return Ok(());
        }

        if scratch.cached_pattern != Some(pattern) {
            self.build_coefficients(scratch);
            scratch.cached_pattern = Some(pattern);
        }
        if r == 1 {
            // One missing row is trivially contiguous: reconstruct straight
            // into the output and skip the staging round trip.
            let start = scratch.missing[0] * symbol_len;
            Self::apply_coefficients(
                &scratch.flat_coeffs,
                symbols,
                r,
                &mut out[start..start + symbol_len],
            );
            return Ok(());
        }
        {
            let dst = &mut scratch.work[..r * symbol_len];
            Self::apply_coefficients(&scratch.flat_coeffs, symbols, r, dst);
        }
        for (row, &data_idx) in scratch.missing.iter().enumerate() {
            out[data_idx * symbol_len..(data_idx + 1) * symbol_len]
                .copy_from_slice(&scratch.work[row * symbol_len..(row + 1) * symbol_len]);
        }
        Ok(())
    }

    /// Reconstruct only the missing data symbols, leaving survivors borrowed.
    ///
    /// `symbols` obeys the same contract as [`decode_into_with`](Self::decode_into_with):
    /// exactly `k` distinct, in-range `(index, payload)` pairs. Instead of
    /// materializing all `k` data symbols, this writes only the `r` data
    /// symbols absent from `symbols` — one `symbol_len` row each, in
    /// ascending data-index order — into `missing_out`, which must be exactly
    /// `r * symbol_len` bytes. `r` is the number of repair symbols in
    /// `symbols` (equivalently, `k` minus the number of data symbols).
    ///
    /// Receivers that keep surviving shards in place — ring buffers, mmap
    /// regions — pay no copy for data that never moved. Allocates a temporary
    /// workspace; use [`reconstruct_missing_into_with`](Self::reconstruct_missing_into_with)
    /// to exclude allocation from the hot path.
    pub fn reconstruct_missing_into(
        &self,
        symbols: &[(usize, &[u8])],
        missing_out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let mut scratch = self.decode_scratch();
        self.reconstruct_missing_into_with(symbols, missing_out, &mut scratch)
    }

    /// Reconstruct the missing data symbols without heap allocation.
    ///
    /// Zero-alloc twin of [`reconstruct_missing_into`](Self::reconstruct_missing_into);
    /// `scratch` must come from a codec with the same geometry.
    pub fn reconstruct_missing_into_with(
        &self,
        symbols: &[(usize, &[u8])],
        missing_out: &mut [u8],
        scratch: &mut DecodeScratch,
    ) -> Result<(), DecodeError> {
        self.check_decode_input(symbols, scratch)?;
        let (r, pattern) = self.partition_symbols(symbols, scratch)?;
        let expected = r * self.symbol_len;
        if missing_out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: missing_out.len(),
            });
        }
        if r == 0 {
            return Ok(());
        }
        if scratch.cached_pattern != Some(pattern) {
            self.build_coefficients(scratch);
            scratch.cached_pattern = Some(pattern);
        }
        Self::apply_coefficients(&scratch.flat_coeffs, symbols, r, missing_out);
        Ok(())
    }

    /// Prepare a reconstruction plan for one erasure pattern.
    ///
    /// `indices` are the codeword indices of the `k` symbols decode will be
    /// given — the surviving data symbols plus exactly one repair per erased
    /// data symbol, in any order. Validation, partitioning, and coefficient
    /// construction happen here, once; applying the returned [`DecodePlan`]
    /// is payload arithmetic only and allocates nothing.
    ///
    /// For ad-hoc patterns that are not known in advance,
    /// [`decode_into_with`](Self::decode_into_with) and
    /// [`reconstruct_missing_into_with`](Self::reconstruct_missing_into_with)
    /// memoize the most recent pattern's coefficients in their scratch
    /// instead.
    pub fn prepare_decode(&self, indices: &[usize]) -> Result<DecodePlan, DecodeError> {
        let k = self.k;
        let n = self.n();
        if indices.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: indices.len(),
            });
        }
        let mut seen = [false; 256];
        let mut pattern = crate::decoder::pattern::PatternKey::empty();
        let mut present = Vec::with_capacity(k);
        let mut repair_cols = Vec::with_capacity(k.min(self.m));
        for &idx in indices {
            if idx >= n {
                return Err(DecodeError::IndexOutOfRange { index: idx, n });
            }
            if seen[idx] {
                return Err(DecodeError::DuplicateIndex { index: idx });
            }
            seen[idx] = true;
            pattern.set(idx);
            if idx < k {
                present.push(idx);
            } else {
                repair_cols.push(idx - k);
            }
        }
        let missing: Vec<usize> = (0..k).filter(|&i| !seen[i]).collect();
        let r = repair_cols.len();
        debug_assert_eq!(missing.len(), r);

        // Coefficient rows are ordered repair-terms-first, then present
        // terms, both in arrival order; map each received codeword index to
        // its row.
        let mut term_of = [u16::MAX; 256];
        for (ordinal, &repair) in repair_cols.iter().enumerate() {
            term_of[k + repair] = ordinal as u16;
        }
        for (pos, &data_idx) in present.iter().enumerate() {
            term_of[data_idx] = (r + pos) as u16;
        }

        let mut flat_coeffs = vec![GfElem::ZERO; k * r];
        let mut vars = vec![GfElem::ZERO; 2 * r];
        let mut inverse = vec![GfElem::ZERO; r * r];
        let mut lagrange = vec![GfElem::ZERO; gfm::cauchy_scratch_len(r)];
        self.fused_coefficients(
            &repair_cols,
            &present,
            &missing,
            &mut vars,
            &mut inverse,
            &mut lagrange,
            &mut flat_coeffs,
        );

        Ok(DecodePlan {
            k,
            m: self.m,
            symbol_len: self.symbol_len,
            missing_data: missing,
            pattern,
            term_of,
            coefficients: flat_coeffs.into_boxed_slice(),
            staging: vec![0u8; r * self.symbol_len],
        })
    }

    /// Shared decode prefix: scratch geometry and symbol count.
    fn check_decode_input(
        &self,
        symbols: &[(usize, &[u8])],
        scratch: &DecodeScratch,
    ) -> Result<(), DecodeError> {
        if (scratch.k, scratch.m, scratch.symbol_len) != (self.k, self.m, self.symbol_len) {
            return Err(DecodeError::ScratchMismatch);
        }
        if symbols.len() != self.k {
            return Err(DecodeError::WrongCount {
                expected: self.k,
                got: symbols.len(),
            });
        }
        Ok(())
    }

    /// Validate `symbols` and partition the receipt pattern into
    /// `scratch.present` (data indices, arrival order), `scratch.repair_cols`
    /// (repair columns, arrival order), and `scratch.missing` (data indices,
    /// ascending). Returns the erasure count `r` and the receipt pattern.
    /// Pure with respect to payload bytes: nothing is copied or combined here.
    fn partition_symbols(
        &self,
        symbols: &[(usize, &[u8])],
        scratch: &mut DecodeScratch,
    ) -> Result<(usize, crate::decoder::pattern::PatternKey), DecodeError> {
        let k = self.k;
        let n = self.n();
        let symbol_len = self.symbol_len;
        debug_assert!(n <= 256);
        let mut seen = [false; 256];
        let mut pattern = crate::decoder::pattern::PatternKey::empty();
        scratch.repair_cols.clear();
        scratch.present.clear();
        for &(idx, payload) in symbols {
            if idx >= n {
                return Err(DecodeError::IndexOutOfRange { index: idx, n });
            }
            if seen[idx] {
                return Err(DecodeError::DuplicateIndex { index: idx });
            }
            if payload.len() != symbol_len {
                return Err(DecodeError::WrongPayloadLen {
                    expected: symbol_len,
                    got: payload.len(),
                });
            }
            seen[idx] = true;
            pattern.set(idx);
            if idx < k {
                scratch.present.push(idx);
            } else {
                scratch.repair_cols.push(idx - k);
            }
        }
        scratch.missing.clear();
        for (i, &was_seen) in seen[..k].iter().enumerate() {
            if !was_seen {
                scratch.missing.push(i);
            }
        }
        debug_assert_eq!(scratch.missing.len(), scratch.repair_cols.len());
        Ok((scratch.repair_cols.len(), pattern))
    }

    /// Compute the fused reconstruction coefficients for the partitioned
    /// receipt pattern into `scratch.flat_coeffs`.
    fn build_coefficients(&self, scratch: &mut DecodeScratch) {
        let DecodeScratch {
            repair_cols,
            present,
            missing,
            vars,
            inverse,
            lagrange,
            flat_coeffs,
            ..
        } = scratch;
        self.fused_coefficients(
            repair_cols,
            present,
            missing,
            vars,
            inverse,
            lagrange,
            flat_coeffs,
        );
    }

    /// Compute the fused reconstruction coefficients for one erasure pattern
    /// into caller-provided buffers.
    ///
    /// The reduced system has rows selected by the received repairs and
    /// columns selected by the missing data symbols:
    /// `A[repair, missing] = 1 / (y_repair + x_missing)`. The rational-Lagrange
    /// closed form produces `A^-1` in `O(r^2)` (no Gauss-Jordan); each
    /// present-data symbol's fused coefficients are then the composition
    /// `A^-1 * C[data, R]` against the precomputed coefficient table, one
    /// length-`r` dot product per present symbol. `flat_coeffs` receives one
    /// source-major coefficient row per received symbol (`k * r` elements):
    /// repair terms first (the transposed inverse), then present terms in
    /// arrival order. `vars` needs `2 * r` elements, `inverse` `r * r`, and
    /// `lagrange` needs [`gfm::cauchy_scratch_len`]`(r)` elements.
    #[allow(clippy::too_many_arguments)]
    fn fused_coefficients(
        &self,
        repair_cols: &[usize],
        present: &[usize],
        missing: &[usize],
        vars: &mut [GfElem],
        inverse: &mut [GfElem],
        lagrange: &mut [GfElem],
        flat_coeffs: &mut [GfElem],
    ) {
        let m = self.m;
        let r = repair_cols.len();
        debug_assert_eq!(missing.len(), r);
        debug_assert_eq!(present.len(), self.k - r);
        let (row_vars, rest) = vars.split_at_mut(r);
        let col_vars = &mut rest[..r];
        for (i, &repair) in repair_cols.iter().enumerate() {
            row_vars[i] = self.cauchy.y_var(repair);
        }
        for (j, &data_idx) in missing.iter().enumerate() {
            col_vars[j] = self.cauchy.x_var(data_idx);
        }
        gfm::cauchy_inverse_coefficients_into::<fgf::Gf8>(
            row_vars,
            col_vars,
            &[],
            &mut inverse[..r * r],
            &mut [],
            lagrange,
        );
        let (repair_terms, present_terms) = flat_coeffs.split_at_mut(r * r);
        // Transpose the reduced inverse into source-major repair terms: repair
        // `i` then carries one coefficient for every missing output.
        for j in 0..r {
            for i in 0..r {
                repair_terms[i * r + j] = inverse[j * r + i];
            }
        }
        // Present symbol `z` contributes `A^-1 * C[z, R]` to the missing rows:
        // fusing the inverse with its cancellation coefficient replaces the
        // separate cancellation pass and the `r^2` apply tail.
        for (pos, &data_idx) in present.iter().enumerate() {
            let table_row = &self.coeffs[data_idx * m..(data_idx + 1) * m];
            for j in 0..r {
                let mut acc = GfElem::ZERO;
                for (i, &repair) in repair_cols.iter().enumerate() {
                    acc = acc.add(inverse[j * r + i].mul(table_row[repair]));
                }
                present_terms[pos * r + j] = acc;
            }
        }
    }

    /// Zero `dst` (`r * symbol_len` contiguous bytes) and accumulate every
    /// received symbol's fused contribution into it with one matrix kernel
    /// call: each source is loaded once and applied to all `r` rows, and each
    /// destination tile is loaded and stored once.
    fn apply_coefficients(
        flat_coeffs: &[GfElem],
        symbols: &[(usize, &[u8])],
        r: usize,
        dst: &mut [u8],
    ) {
        let k = symbols.len();
        let symbol_len = dst.len() / r;
        dst.fill(0);
        // Coefficients are stored source-major with the repair terms first,
        // exactly the term layout the kernel wants. The descriptor array is
        // stack-resident and bounded by the GF(256) codeword limit, so
        // reconstruction allocates nothing.
        let mut term_storage = [(&[][..], &[][..]); 256];
        let mut repair_pos = 0;
        let mut present_pos = 0;
        for &(idx, payload) in symbols {
            if idx < k {
                term_storage[r + present_pos] = (
                    &flat_coeffs[r * r + present_pos * r..r * r + (present_pos + 1) * r],
                    payload,
                );
                present_pos += 1;
            } else {
                term_storage[repair_pos] =
                    (&flat_coeffs[repair_pos * r..(repair_pos + 1) * r], payload);
                repair_pos += 1;
            }
        }
        debug_assert_eq!(repair_pos, r);
        let terms = &term_storage[..k];
        crate::payload::xor_scaled_bytes_rows_terms(dst, symbol_len, r, terms);
    }

    /// Naive full-RREF reference decode. Correctness oracle for
    /// [`decode_into`](Self::decode_into): the fast path must produce
    /// bit-identical output for every valid symbol selection.
    #[cfg(test)]
    fn decode_reference(&self, symbols: &[(usize, &[u8])]) -> Result<Vec<u8>, DecodeError> {
        let k = self.k;
        let symbol_len = self.symbol_len;

        // Build the augmented matrix [M | P].
        let stride = k + symbol_len;
        let mut augmented = Matrix::<fgf::Gf8>::zeros(k, stride)
            .expect("augmented matrix dimensions are internally consistent");
        for (row, &(idx, payload)) in symbols.iter().enumerate() {
            // M[row][col] = G[col][idx], the selected systematic-generator
            // column.
            if idx < k {
                augmented.set(row, idx, GfElem::ONE);
            } else {
                let repair = idx - k;
                for col in 0..k {
                    augmented.set(row, col, self.cauchy.get(col, repair));
                }
            }
            for (p, &byte) in payload.iter().enumerate() {
                augmented.set(row, k + p, GfElem(byte));
            }
        }

        let ple = Ple::decompose(augmented, &mut PleScratch::new());
        if ple.rank() != k {
            return Err(DecodeError::InsufficientRank {
                rank: ple.rank(),
                k,
            });
        }
        let mut reduced = Matrix::<fgf::Gf8>::zeros(k, stride)
            .expect("augmented matrix dimensions are internally consistent");
        ple.rref_into(&mut reduced);
        let mut out = vec![0u8; k * symbol_len];
        for row in 0..k {
            out[row * symbol_len..(row + 1) * symbol_len].copy_from_slice(&reduced.row(row)[k..]);
        }
        Ok(out)
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<C: CodingMatrix> BatchCodec<C> {
    /// Coding-matrix view the coefficient table was generated from.
    ///
    /// Production encode and decode never consult it; it is retained so a
    /// reference path can regenerate coefficients independently of `coeffs`.
    #[must_use]
    pub const fn cauchy(&self) -> &C {
        &self.cauchy
    }

    /// Source-major `k x m` coefficient table, `coeffs[i * m + j] == C[i][j]`.
    ///
    /// Built once in [`BatchCodec::new`] so neither encode nor decode performs
    /// a per-`(i, j)` matrix lookup.
    #[must_use]
    pub fn coeffs(&self) -> &[GfElem] {
        &self.coeffs
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
    fn reconstruct_missing_matches_full_decode() {
        use crate::cauchy::CauchyView;
        use crate::good_cauchy::GoodCauchyView;
        // Same geometry/subset spread as `optimized_decode_matches_reference`,
        // plus near-capacity erasure counts to exercise the general
        // rational-Lagrange path.
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
            let mut subsets: Vec<Vec<usize>> = Vec::new();
            subsets.push((0..k).collect()); // r0: nothing missing
            for erase_count in 1..=m.min(k) {
                for shift in 0..k.max(1) {
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
                    let mut scratch = $c.decode_scratch();
                    for subset in &subsets {
                        let received: Vec<(usize, &[u8])> = subset
                            .iter()
                            .map(|&idx| (idx, symbols[idx].as_slice()))
                            .collect();
                        let r = subset.iter().filter(|&&idx| idx >= k).count();
                        let mut full = vec![0x5Au8; k * slen];
                        $c.decode_into_with(&received, &mut full, &mut scratch)
                            .unwrap();
                        let mut missing_out = vec![0xA5u8; r * slen];
                        $c.reconstruct_missing_into_with(&received, &mut missing_out, &mut scratch)
                            .unwrap();
                        let missing: Vec<usize> = (0..k).filter(|i| !subset.contains(i)).collect();
                        assert_eq!(missing.len(), r);
                        for (row, &data_idx) in missing.iter().enumerate() {
                            assert_eq!(
                                &missing_out[row * slen..(row + 1) * slen],
                                &full[data_idx * slen..(data_idx + 1) * slen],
                                "{} k={} m={} slen={} subset={:?} row={}",
                                $name,
                                k,
                                m,
                                slen,
                                subset,
                                row
                            );
                        }
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
    fn reconstruct_missing_rejects_wrong_output_len() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(4, 2, 8).unwrap();
        let data = vec![7u8; 32];
        let symbols = c.encode(&data).unwrap();
        // Two data symbols missing, two repairs received.
        let received: Vec<(usize, &[u8])> = vec![
            (1, symbols[1].as_slice()),
            (3, symbols[3].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        let mut short = vec![0u8; 15];
        assert_eq!(
            c.reconstruct_missing_into(&received, &mut short),
            Err(DecodeError::WrongOutputLen {
                expected: 16,
                got: 15
            })
        );
        // Nothing missing: only the empty buffer is accepted.
        let all_data: Vec<(usize, &[u8])> = (0..4).map(|i| (i, symbols[i].as_slice())).collect();
        assert_eq!(
            c.reconstruct_missing_into(&all_data, &mut short[..1]),
            Err(DecodeError::WrongOutputLen {
                expected: 0,
                got: 1
            })
        );
        assert_eq!(c.reconstruct_missing_into(&all_data, &mut []), Ok(()));
    }

    #[test]
    fn reconstruct_missing_rejects_bad_symbols() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        let data = vec![0u8; 12];
        let symbols = c.encode(&data).unwrap();
        let mut out = vec![0u8; 4];
        // Wrong count.
        let two: Vec<(usize, &[u8])> = (0..2).map(|i| (i, symbols[i].as_slice())).collect();
        assert_eq!(
            c.reconstruct_missing_into(&two, &mut out),
            Err(DecodeError::WrongCount {
                expected: 3,
                got: 2
            })
        );
        // Duplicate index.
        let dup: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (0, symbols[0].as_slice()),
            (3, symbols[3].as_slice()),
        ];
        assert_eq!(
            c.reconstruct_missing_into(&dup, &mut out),
            Err(DecodeError::DuplicateIndex { index: 0 })
        );
        // Scratch from another geometry.
        let other = BatchCodec::<crate::cauchy::CauchyView>::new(4, 2, 4).unwrap();
        let received: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (2, symbols[2].as_slice()),
            (3, symbols[3].as_slice()),
        ];
        let mut scratch = other.decode_scratch();
        assert_eq!(
            c.reconstruct_missing_into_with(&received, &mut out, &mut scratch),
            Err(DecodeError::ScratchMismatch)
        );
    }

    #[test]
    fn prepare_decode_matches_full_decode() {
        use crate::cauchy::CauchyView;
        use crate::good_cauchy::GoodCauchyView;
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
            let mut subsets: Vec<Vec<usize>> = Vec::new();
            subsets.push((0..k).collect()); // r0: nothing missing
            for erase_count in 1..=m.min(k) {
                for shift in 0..k.max(1) {
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
                        // Symbols may arrive in any order; reverse every
                        // second subset to prove the plan does not care.
                        let mut ordered = subset.clone();
                        if subset.len() % 2 == 0 {
                            ordered.reverse();
                        }
                        let received: Vec<(usize, &[u8])> = ordered
                            .iter()
                            .map(|&idx| (idx, symbols[idx].as_slice()))
                            .collect();
                        let r = subset.iter().filter(|&&idx| idx >= k).count();
                        let mut plan = $c.prepare_decode(&ordered).unwrap();
                        assert_eq!(plan.erasure_count(), r);
                        let missing: Vec<usize> = (0..k).filter(|i| !subset.contains(i)).collect();
                        assert_eq!(plan.missing_indices(), &missing[..]);

                        // Full decode through the plan reproduces the data.
                        let mut full = vec![0x5Au8; k * slen];
                        plan.decode_into(&received, &mut full).unwrap();
                        assert_eq!(full, data, "{} k={} subset={:?}", $name, k, subset);

                        // Reconstruct-only rows match the full decode rows.
                        let mut missing_out = vec![0xA5u8; r * slen];
                        plan.reconstruct_missing_into(&received, &mut missing_out)
                            .unwrap();
                        for (row, &data_idx) in missing.iter().enumerate() {
                            assert_eq!(
                                &missing_out[row * slen..(row + 1) * slen],
                                &data[data_idx * slen..(data_idx + 1) * slen],
                                "{} k={} subset={:?} row={}",
                                $name,
                                k,
                                subset,
                                row
                            );
                        }
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
    fn prepare_decode_rejects_bad_indices() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(3, 2, 4).unwrap();
        assert_eq!(
            c.prepare_decode(&[0, 1]).unwrap_err(),
            DecodeError::WrongCount {
                expected: 3,
                got: 2
            }
        );
        assert_eq!(
            c.prepare_decode(&[0, 1, 5]).unwrap_err(),
            DecodeError::IndexOutOfRange { index: 5, n: 5 }
        );
        assert_eq!(
            c.prepare_decode(&[0, 1, 1]).unwrap_err(),
            DecodeError::DuplicateIndex { index: 1 }
        );
    }

    #[test]
    fn plan_apply_rejects_mismatched_symbols() {
        let c = BatchCodec::<crate::cauchy::CauchyView>::new(4, 2, 8).unwrap();
        let data = vec![7u8; 32];
        let symbols = c.encode(&data).unwrap();
        // Prepared for data symbols 0 and 2 missing.
        let mut plan = c.prepare_decode(&[1, 3, 4, 5]).unwrap();
        let received: Vec<(usize, &[u8])> = vec![
            (1, symbols[1].as_slice()),
            (3, symbols[3].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        let mut missing_out = vec![0u8; 16];
        let mut out = vec![0u8; 32];

        // A symbol outside the prepared pattern is rejected.
        let wrong: Vec<(usize, &[u8])> = vec![
            (0, symbols[0].as_slice()),
            (3, symbols[3].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        assert_eq!(
            plan.reconstruct_missing_into(&wrong, &mut missing_out),
            Err(DecodeError::UnexpectedIndex { index: 0 })
        );
        assert_eq!(
            plan.decode_into(&wrong, &mut out),
            Err(DecodeError::UnexpectedIndex { index: 0 })
        );
        // Wrong count, duplicate, payload length, output length.
        assert_eq!(
            plan.reconstruct_missing_into(&received[..3], &mut missing_out),
            Err(DecodeError::WrongCount {
                expected: 4,
                got: 3
            })
        );
        let dup: Vec<(usize, &[u8])> = vec![
            (1, symbols[1].as_slice()),
            (1, symbols[1].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        assert_eq!(
            plan.reconstruct_missing_into(&dup, &mut missing_out),
            Err(DecodeError::DuplicateIndex { index: 1 })
        );
        let short: Vec<(usize, &[u8])> = vec![
            (1, &symbols[1][..7]),
            (3, symbols[3].as_slice()),
            (4, symbols[4].as_slice()),
            (5, symbols[5].as_slice()),
        ];
        assert_eq!(
            plan.reconstruct_missing_into(&short, &mut missing_out),
            Err(DecodeError::WrongPayloadLen {
                expected: 8,
                got: 7
            })
        );
        assert_eq!(
            plan.reconstruct_missing_into(&received, &mut missing_out[..15]),
            Err(DecodeError::WrongOutputLen {
                expected: 16,
                got: 15
            })
        );
        assert_eq!(
            plan.decode_into(&received, &mut out[..31]),
            Err(DecodeError::WrongOutputLen {
                expected: 32,
                got: 31
            })
        );
        // The failed calls above must not poison the plan.
        plan.decode_into(&received, &mut out).unwrap();
        assert_eq!(out, data);
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
