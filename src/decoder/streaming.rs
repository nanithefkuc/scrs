//! Main decoder logic for SRS.
//!
//! This module provides a redesigned decoder focused on predictable receive-path
//! latency:
//!
//! - `push` only validates, records a 256-bit receipt pattern, and copies the
//!   raw payload into a codeword buffer.
//! - `finalize` solves only the reduced `r × r` system for the `r` missing data
//!   symbols, where `r` is the number of erased systematic data symbols.
//! - The coefficient solve is payload-lazy: Gaussian elimination is performed
//!   only on the tiny coefficient matrix. Payload bytes are combined once in a
//!   final reconstruction pass.

use std::sync::Arc;

use super::{cache::RecipeCache, recipe};
use crate::codec::{Coded, Decoder};
use crate::coding_matrix::CodingMatrix;
use crate::error::{ConfigError, DecodeError};
use crate::pattern_key::PatternKey;
use crate::stream::{PushOutcome, SymbolSink};
use fgf::gf8::Elem as GfElem;

/// Upper bound on the number of source symbols one reconstruction may combine.
///
/// A GF(256) codeword holds at most `n = k + m <= 256` symbols, so a recipe can
/// never name more sources than this. Because the bound is a compile-time
/// constant, `apply_recipe_into` keeps its per-source
/// `(coefficients, payload)` descriptor array in a stack `MaybeUninit` block
/// instead of a `Vec`, which is why reconstruction allocates nothing for any
/// erasure pattern.
pub const MAX_SOURCES: usize = 256;

/// Lazy, payload-deferred streaming decoder.
///
/// This decoder is MDS-aware for the systematic Cauchy generator selected by
/// `C`. The encoder and decoder must use the same matrix type: for example,
/// [`crate::good_cauchy::GoodCauchyView`] with Good-Cauchy batch encoding, or
/// [`crate::cauchy::CauchyView`] with Standard-Cauchy batch encoding. It treats
/// every distinct received symbol as independent and becomes complete after any
/// `k` distinct symbols. The decoder does not incrementally eliminate
/// payload bytes during `push` — it records a 256-bit receipt pattern and
/// defers all payload work to [`finalize_ref`](LazyDecoderState::finalize_ref).
pub struct LazyDecoderState<C: CodingMatrix> {
    k: usize,
    m: usize,
    n: usize,
    symbol_len: usize,
    cauchy: C,
    payloads: Vec<u8>,
    /// Contiguous `r`-row workspace for reconstruction, grown on demand and
    /// reused across finalizations. Missing outputs are scattered across `out`,
    /// but the fused row kernel needs them adjacent.
    staging: Vec<u8>,
    pattern: PatternKey,
    distinct: usize,
    received: usize,
}

impl<C: CodingMatrix> LazyDecoderState<C> {
    /// Create a decoder for `(k, m)` with `symbol_len`-byte symbols and
    /// coding matrix `C`.
    ///
    /// Returns [`ConfigError::ZeroDimension`] if `k` or `m` is zero,
    /// [`ConfigError::ZeroSymbolLen`] if `symbol_len` is zero, and
    /// [`ConfigError::TooManySymbols`] when `C::new(k, m)` rejects the
    /// dimensions. Consequently Standard Cauchy can use `k + m <= 256`, while
    /// Good Cauchy is limited to `k + m <= 255`.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        let cauchy = C::new(k, m).ok_or(ConfigError::TooManySymbols { cap: C::CAPACITY })?;
        Ok(Self {
            k,
            m,
            n: k + m,
            symbol_len,
            cauchy,
            payloads: vec![0u8; (k + m) * symbol_len],
            staging: Vec::new(),
            pattern: PatternKey::empty(),
            distinct: 0,
            received: 0,
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
        self.n
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of distinct symbols received, capped by `k` for completion.
    pub const fn rank(&self) -> usize {
        self.distinct
    }

    /// Total symbols received, including duplicates/dependent symbols.
    pub const fn received(&self) -> usize {
        self.received
    }

    /// The current 256-bit receipt pattern.
    pub const fn pattern_key(&self) -> PatternKey {
        self.pattern
    }
    /// Clear receipt state for another block while retaining payload storage.
    pub fn reset(&mut self) {
        self.pattern = PatternKey::empty();
        self.distinct = 0;
        self.received = 0;
    }

    /// Convenience wrapper around [`SymbolSink::push`].
    pub fn push_symbol(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, DecodeError> {
        self.push(idx, payload)
    }

    /// Non-consuming finalization returning an owned buffer.
    ///
    /// This reconstructs missing systematic data symbols using only a reduced
    /// `r × r` repair submatrix, where `r` is the number of missing data symbols.
    pub fn finalize_ref(&mut self) -> Result<Vec<u8>, DecodeError> {
        self.ensure_complete()?;
        let recipe = self.build_recipe()?;
        let mut out = vec![0u8; self.k * self.symbol_len];
        self.apply_recipe_into(&recipe, &mut out);
        Ok(out)
    }

    internals_pub! {
    /// Reuse a memoized recipe for the current receipt pattern, building and
    /// caching one on a miss.
    ///
    /// The cache key carries `(k, m, engine, pattern)`, so one cache is safe to
    /// share across decoders of different geometry or Cauchy construction.
        fn recipe_from_cache(
            &self,
            cache: &mut RecipeCache,
        ) -> Result<Arc<recipe::ReconstructionRecipe>, DecodeError> {
            let key = recipe::RecipeKey {
                k: self.k,
                m: self.m,
                engine: C::ENGINE,
                pattern: self.pattern,
            };
            if let Some(recipe) = cache.get(key) {
                Ok(recipe)
            } else {
                let recipe = Arc::new(self.build_recipe()?);
                cache.insert(key, Arc::clone(&recipe));
                Ok(recipe)
            }
        }
    }

    internals_pub! {
    /// Check that `k` distinct symbols have been recorded.
    ///
    /// Every finalization path calls this first; it is the only place that turns a
    /// short receipt count into [`DecodeError::InsufficientRank`].
        fn ensure_complete(&self) -> Result<(), DecodeError> {
            if self.distinct < self.k {
                return Err(DecodeError::InsufficientRank {
                    rank: self.distinct,
                    k: self.k,
                });
            }
            Ok(())
        }
    }

    internals_pub! {
    /// Derive the reconstruction plan for the current receipt pattern.
    ///
    /// Partitions the systematic range into present and missing indices, selects
    /// exactly `r` received repair columns for the `r` missing data symbols, and
    /// emits source-major coefficients from the factorized rational-Lagrange
    /// inverse. Pure with respect to `self`: no payload byte is read.
        fn build_recipe(&self) -> Result<recipe::ReconstructionRecipe, DecodeError> {
            let mut missing_data = Vec::new();
            let mut present_data = Vec::new();
            for data_idx in 0..self.k {
                if self.pattern.get(data_idx) {
                    present_data.push(data_idx);
                } else {
                    missing_data.push(data_idx);
                }
            }

            let r = missing_data.len();
            let mut repair_cols = Vec::with_capacity(r);
            for repair in 0..self.m {
                if self.pattern.get(self.k + repair) {
                    repair_cols.push(repair);
                    if repair_cols.len() == r {
                        break;
                    }
                }
            }
            debug_assert_eq!(
                repair_cols.len(),
                r,
                "MDS completion implies enough repairs"
            );
            if repair_cols.len() != r {
                return Err(DecodeError::InsufficientRank {
                    rank: self.distinct,
                    k: self.k,
                });
            }
            if r == 0 {
                return Ok(recipe::ReconstructionRecipe {
                    missing_data,
                    present_data,
                    source_terms: Vec::new(),
                });
            }

            // The reduced system has rows selected by repair symbols and columns
            // selected by missing data symbols:
            // A[row=repair, col=missing_data] = 1 / (y_repair + x_missing).
            // Factorized rational-Lagrange products produce both A^-1 and the fused
            // coefficients for present data in O(r² + r*(k-r)).
            let row_vars: Vec<GfElem> = repair_cols
                .iter()
                .map(|&repair| self.cauchy.y_var(repair))
                .collect();
            let col_vars: Vec<GfElem> = missing_data
                .iter()
                .map(|&data_idx| self.cauchy.x_var(data_idx))
                .collect();
            let present_vars: Vec<GfElem> = present_data
                .iter()
                .map(|&data_idx| self.cauchy.x_var(data_idx))
                .collect();
            let mut inverse = vec![GfElem::ZERO; r * r];
            let mut present_coefficients = vec![GfElem::ZERO; present_vars.len() * r];
            let mut cauchy_scratch = vec![GfElem::ZERO; gfm::cauchy_scratch_len(r)];
            gfm::cauchy_inverse_coefficients_into::<fgf::Gf8>(
                &row_vars,
                &col_vars,
                &present_vars,
                &mut inverse,
                &mut present_coefficients,
                &mut cauchy_scratch,
            );

            // Transpose the reduced inverse into source-major repair terms. Each
            // received repair carries one coefficient for every missing output.
            let mut source_terms = Vec::with_capacity(self.k);
            for (repair_pos, &repair) in repair_cols.iter().enumerate() {
                let coefficients = (0..r)
                    .map(|missing_pos| inverse[missing_pos * r + repair_pos])
                    .collect();
                source_terms.push(recipe::SourceTerm {
                    source_idx: self.k + repair,
                    coefficients,
                });
            }

            // Direct fused coefficients replace the former length-r A^-1*C dot
            // product for every (present source, missing output) pair.
            for (present_pos, &data_idx) in present_data.iter().enumerate() {
                let coefficients = (0..r)
                    .map(|missing_pos| present_coefficients[present_pos * r + missing_pos])
                    .collect();
                source_terms.push(recipe::SourceTerm {
                    source_idx: data_idx,
                    coefficients,
                });
            }

            Ok(recipe::ReconstructionRecipe {
                missing_data,
                present_data,
                source_terms,
            })
        }
    }

    internals_pub! {
        /// Apply a reconstruction recipe into `out` (`k * symbol_len` bytes).
        ///
        /// Present rows are copied straight through; missing rows are rebuilt from the
        /// recipe's source terms.
        fn apply_recipe_into(&mut self, recipe: &recipe::ReconstructionRecipe, out: &mut [u8]) {
            let slen = self.symbol_len;
            debug_assert_eq!(out.len(), self.k * slen);

            // Copy present data symbols directly.
            for &data_idx in &recipe.present_data {
                let src = data_idx * slen;
                out[src..src + slen].copy_from_slice(&self.payloads[src..src + slen]);
            }

            let rows = recipe.missing_data.len();
            if rows == 0 {
                return;
            }

            // Reconstruct through one fused multi-source, multi-row kernel call rather
            // than `rows * sources` single-AXPY calls: each source symbol is loaded once
            // and applied to every missing output, which is worth 20-37% end to end at
            // MTU-sized symbols and grows with the erasure count.
            //
            // `fgf::ops::mul_add_gather` is the other candidate shape and needs no
            // staging, but it loads every source once *per destination*, so measured
            // across this crate's geometries it is 1.2-10x slower than the matrix
            // kernel as soon as more than one symbol is missing.
            //
            // The kernel needs its destinations adjacent, and missing outputs are
            // scattered through `out` — except when exactly one is missing, where the
            // output row is trivially contiguous and staging is pure overhead. That is
            // also the most common loss pattern, so it gets the direct path.
            let Self {
                payloads, staging, ..
            } = self;
            let single = rows == 1;
            let destination = if single {
                let start = recipe.missing_data[0] * slen;
                out[start..start + slen].fill(0);
                &mut out[start..start + slen]
            } else {
                staging.clear();
                staging.resize(rows * slen, 0);
                &mut staging[..]
            };

            // Coefficients are already stored source-major, one contiguous run per
            // source over the missing outputs, which is exactly the term layout the
            // kernel wants. The descriptor array is stack-resident and bounded by the
            // GF(256) codeword limit, so reconstruction allocates nothing.
            let sources = recipe.source_terms.len();
            debug_assert!(sources <= MAX_SOURCES);
            let mut term_storage = [(&[][..], &[][..]); MAX_SOURCES];
            for (slot, term) in term_storage.iter_mut().zip(&recipe.source_terms) {
                let start = term.source_idx * slen;
                debug_assert_eq!(term.coefficients.len(), rows);
                *slot = (&term.coefficients[..], &payloads[start..start + slen]);
            }
            let terms = &term_storage[..sources];
            crate::payload::xor_scaled_bytes_rows_terms(destination, slen, rows, terms);

            if !single {
                for (row, &data_idx) in recipe.missing_data.iter().enumerate() {
                    let out_start = data_idx * slen;
                    out[out_start..out_start + slen]
                        .copy_from_slice(&staging[row * slen..(row + 1) * slen]);
                }
            }
        }
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<C: CodingMatrix> LazyDecoderState<C> {
    /// Coding-matrix view whose `x`/`y` variables define the reduced Cauchy
    /// system; the encoder must have used the same construction.
    #[must_use]
    pub const fn cauchy(&self) -> &C {
        &self.cauchy
    }

    /// Codeword buffer of `n * symbol_len` bytes indexed by codeword position.
    ///
    /// Only positions set in [`pattern_key`](Self::pattern_key) hold received
    /// bytes; the rest retain whatever the previous block left there.
    #[must_use]
    pub fn payloads(&self) -> &[u8] {
        &self.payloads
    }

    /// Contiguous `r`-row reconstruction workspace, empty until a
    /// multi-erasure finalization grows it to `r * symbol_len` bytes.
    ///
    /// The single-erasure path writes the output row in place and leaves this
    /// untouched, so a non-empty buffer reflects the last multi-erasure solve.
    #[must_use]
    pub fn staging(&self) -> &[u8] {
        &self.staging
    }
}

impl<C: CodingMatrix> SymbolSink for LazyDecoderState<C> {
    fn push(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, DecodeError> {
        if idx >= self.n {
            return Err(DecodeError::IndexOutOfRange {
                index: idx,
                n: self.n,
            });
        }
        if payload.len() != self.symbol_len {
            return Err(DecodeError::WrongPayloadLen {
                expected: self.symbol_len,
                got: payload.len(),
            });
        }
        if self.received >= self.n {
            return Err(DecodeError::TooManySymbols {
                cap: self.n,
                received: self.received,
            });
        }

        self.received += 1;

        // Once complete, extra symbols are not needed for the selected decode
        // recipe. Treat them as dependent.
        if self.distinct >= self.k || self.pattern.get(idx) {
            return Ok(PushOutcome::Dependent);
        }

        let start = idx * self.symbol_len;
        self.payloads[start..start + self.symbol_len].copy_from_slice(payload);
        self.pattern.set(idx);
        self.distinct += 1;

        if self.distinct >= self.k {
            Ok(PushOutcome::Complete)
        } else {
            Ok(PushOutcome::Advanced {
                rank: self.distinct,
                received: self.received,
            })
        }
    }

    fn is_complete(&self) -> bool {
        self.distinct >= self.k
    }

    fn finalize(mut self) -> Result<Vec<u8>, DecodeError> {
        self.finalize_ref()
    }
}

impl<C: CodingMatrix> Coded for LazyDecoderState<C> {
    fn k(&self) -> usize {
        self.k
    }

    fn m(&self) -> usize {
        self.m
    }

    fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    fn n(&self) -> usize {
        self.n
    }
}

impl<C: CodingMatrix> Decoder for LazyDecoderState<C> {
    type Scratch = RecipeCache;

    fn scratch(&self) -> RecipeCache {
        RecipeCache::new(16)
    }

    fn rank(&self) -> usize {
        self.distinct
    }

    fn received(&self) -> usize {
        self.received
    }
    fn reset(&mut self) {
        LazyDecoderState::reset(self);
    }

    fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let recipe = self.build_recipe()?;
        self.apply_recipe_into(&recipe, out);
        Ok(())
    }

    fn finalize_into_with(
        &mut self,
        out: &mut [u8],
        scratch: &mut RecipeCache,
    ) -> Result<(), DecodeError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let recipe = self.recipe_from_cache(scratch)?;
        self.apply_recipe_into(&recipe, out);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::BatchCodec;
    use crate::codec::Decoder;

    fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
        if k > n {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut state: Vec<usize> = (0..k).collect();
        loop {
            result.push(state.clone());
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

    #[test]
    fn recipe_cache_records_hits_and_misses() {
        let (k, m, slen) = (4, 3, 8);
        let codec = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        let data: Vec<u8> = (0..k * slen).map(|i| i as u8).collect();
        let symbols = codec.encode(&data).unwrap();
        let arrival: Vec<usize> = (k..k + m).chain(0..k - m).collect();
        let mut cache = RecipeCache::new(8);

        for iter in 0..2 {
            let mut dec = LazyDecoderState::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
            for &idx in &arrival {
                dec.push_symbol(idx, &symbols[idx]).unwrap();
            }
            let mut out = vec![0u8; k * slen];
            dec.finalize_into_with(&mut out, &mut cache).unwrap();
            assert_eq!(out, data);
            if iter == 0 {
                assert_eq!(cache.misses(), 1);
                assert_eq!(cache.hits(), 0);
            } else {
                assert_eq!(cache.misses(), 1);
                assert_eq!(cache.hits(), 1);
            }
        }
    }

    #[test]
    fn recipe_cache_separates_matrix_implementations() {
        let (k, m, slen) = (4, 3, 32);
        let data: Vec<u8> = (0..k * slen).map(|i| i as u8).collect();
        let good = BatchCodec::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        let standard = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        let good_symbols = good.encode(&data).unwrap();
        let standard_symbols = standard.encode(&data).unwrap();
        let arrival = [k, k + 1, 2, 3];
        let mut cache = RecipeCache::new(8);

        let mut good_decoder =
            LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        for &idx in &arrival {
            good_decoder.push_symbol(idx, &good_symbols[idx]).unwrap();
        }
        let mut good_out = vec![0u8; k * slen];
        good_decoder
            .finalize_into_with(&mut good_out, &mut cache)
            .unwrap();
        assert_eq!(good_out, data);

        let mut standard_decoder =
            LazyDecoderState::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        for &idx in &arrival {
            standard_decoder
                .push_symbol(idx, &standard_symbols[idx])
                .unwrap();
        }
        let mut standard_out = vec![0u8; k * slen];
        standard_decoder
            .finalize_into_with(&mut standard_out, &mut cache)
            .unwrap();
        assert_eq!(standard_out, data);
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 0);
    }

    #[test]
    fn constructor_uses_selected_matrix_capacity() {
        assert!(LazyDecoderState::<crate::cauchy::CauchyView>::new(255, 1, 1).is_ok());
        assert!(LazyDecoderState::<crate::cauchy::CauchyView>::new(255, 2, 1).is_err());
        assert!(LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(254, 1, 1).is_ok());
        assert!(LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(255, 1, 1).is_err());
        assert!(LazyDecoderState::<crate::cauchy::CauchyView>::new(1, 1, 0).is_err());
    }

    #[test]
    fn roundtrip_all_subsets_small() {
        let (k, m, slen) = (4, 3, 8);
        let codec = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        let n = codec.n();
        let data: Vec<u8> = (0..k * slen)
            .map(|i| (i as u8).wrapping_mul(17) ^ 0xA5)
            .collect();
        let symbols = codec.encode(&data).unwrap();

        for subset in k_subsets(n, k) {
            let mut dec = LazyDecoderState::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
            for &idx in &subset {
                dec.push_symbol(idx, &symbols[idx]).unwrap();
            }
            assert!(dec.is_complete(), "subset {subset:?}");
            assert_eq!(dec.finalize().unwrap(), data, "subset {subset:?}");
        }
    }

    #[test]
    fn roundtrip_all_subsets_small_good_cauchy() {
        let (k, m, slen) = (4, 3, 8);
        let codec = BatchCodec::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        let n = codec.n();
        let data: Vec<u8> = (0..k * slen)
            .map(|i| (i as u8).wrapping_mul(17) ^ 0xA5)
            .collect();
        let symbols = codec.encode(&data).unwrap();

        for subset in k_subsets(n, k) {
            let mut dec =
                LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
            for &idx in &subset {
                dec.push_symbol(idx, &symbols[idx]).unwrap();
            }
            assert!(dec.is_complete(), "subset {subset:?}");
            assert_eq!(dec.finalize().unwrap(), data, "subset {subset:?}");
        }
    }

    /// Reconstruction correctness at the vector-length boundaries fgf's kernels
    /// switch on (below one lane, exactly one, one plus a tail, and a large odd
    /// length). This used to be a differential test between SRS's own output-major
    /// and grouped source-major kernels; with the kernels delegated to fgf there is
    /// one path, so what remains is the boundary coverage.
    fn assert_reconstruction_at_vector_boundaries<C: CodingMatrix>() {
        let (k, m) = (4, 3);
        for slen in [1, 15, 16, 17, 31, 32, 33, 65, 1400] {
            let codec = BatchCodec::<C>::new(k, m, slen).unwrap();
            let data: Vec<u8> = (0..k * slen)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let symbols = codec.encode(&data).unwrap();

            for subset in k_subsets(k + m, k) {
                let mut decoder = LazyDecoderState::<C>::new(k, m, slen).unwrap();
                for &idx in &subset {
                    decoder.push_symbol(idx, &symbols[idx]).unwrap();
                }
                let recipe = decoder.build_recipe().unwrap();
                let mut reconstructed = vec![0u8; k * slen];
                decoder.apply_recipe_into(&recipe, &mut reconstructed);
                assert_eq!(reconstructed, data, "slen={slen}, subset={subset:?}");
            }
        }

        // Wider geometry: more missing outputs than a single vector's worth, without
        // exhaustively enumerating 8-of-14 subsets.
        let (k, m) = (8, 6);
        for slen in [1, 31, 32, 33, 1400] {
            let codec = BatchCodec::<C>::new(k, m, slen).unwrap();
            let data: Vec<u8> = (0..k * slen)
                .map(|i| (i as u8).wrapping_mul(19).wrapping_add(7))
                .collect();
            let symbols = codec.encode(&data).unwrap();
            for r in [4, 5, 6] {
                let arrival: Vec<_> = (k..k + r).chain(r..k).collect();
                let mut decoder = LazyDecoderState::<C>::new(k, m, slen).unwrap();
                for &idx in &arrival {
                    decoder.push_symbol(idx, &symbols[idx]).unwrap();
                }
                let recipe = decoder.build_recipe().unwrap();
                let mut reconstructed = vec![0u8; k * slen];
                decoder.apply_recipe_into(&recipe, &mut reconstructed);
                assert_eq!(reconstructed, data, "slen={slen}, r={r}");
            }
        }
    }

    #[test]
    fn reconstruction_holds_at_vector_boundaries() {
        assert_reconstruction_at_vector_boundaries::<crate::cauchy::CauchyView>();
        assert_reconstruction_at_vector_boundaries::<crate::good_cauchy::GoodCauchyView>();
    }

    #[test]
    fn duplicate_before_complete_is_dependent() {
        let (k, m, slen) = (3, 2, 4);
        let codec = BatchCodec::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        let data = vec![0x42; k * slen];
        let symbols = codec.encode(&data).unwrap();
        let mut dec = LazyDecoderState::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        assert!(matches!(
            dec.push_symbol(0, &symbols[0]).unwrap(),
            PushOutcome::Advanced { .. }
        ));
        assert_eq!(
            dec.push_symbol(0, &symbols[0]).unwrap(),
            PushOutcome::Dependent
        );
        assert_eq!(dec.rank(), 1);
        assert_eq!(dec.received(), 2);
    }

    #[test]
    fn finalize_into_matches_finalize_ref() {
        let (k, m, slen) = (4, 3, 64);
        let codec = BatchCodec::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        let data: Vec<u8> = (0..k * slen).map(|i| i as u8).collect();
        let symbols = codec.encode(&data).unwrap();
        // Two data missing: use repairs 0,1 and data 2,3.
        let arrival = [k, k + 1, 2, 3];

        let mut dec =
            LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        for &idx in &arrival {
            dec.push_symbol(idx, &symbols[idx]).unwrap();
        }
        let allocated = dec.finalize_ref().unwrap();

        let mut dec =
            LazyDecoderState::<crate::good_cauchy::GoodCauchyView>::new(k, m, slen).unwrap();
        for &idx in &arrival {
            dec.push_symbol(idx, &symbols[idx]).unwrap();
        }
        let mut into = vec![0xFFu8; k * slen];
        dec.finalize_into(&mut into).unwrap();
        assert_eq!(into, allocated);
        assert_eq!(into, data);

        let mut short = vec![0u8; k * slen - 1];
        let err = dec.finalize_into(&mut short).unwrap_err();
        assert!(matches!(err, DecodeError::WrongOutputLen { .. }));
    }
}
