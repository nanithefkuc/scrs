//! Main decoder logic for SCRS.
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

use crate::coding_matrix::CodingMatrix;
use crate::gf256::GfElem;
use crate::pattern_key::PatternKey;
use crate::stream::{PushOutcome, StreamError, SymbolSink};

mod cauchy_inverse;
mod recipe;

pub use cauchy_inverse::cauchy_inverse_closed_form;
pub use recipe::RecipeCache;

/// v0.2 lazy/reduced streaming decoder.
///
/// This decoder is MDS-aware for SCRS's systematic Cauchy generator. It treats
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
    pattern: PatternKey,
    distinct: usize,
    received: usize,
}

impl<C: CodingMatrix> LazyDecoderState<C> {
    /// Create a v0.2 decoder for `(k, m)` with `symbol_len`-byte symbols.
    ///
    /// Returns `None` for invalid dimensions, zero symbol length, or
    /// `k + m > 256`.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let cauchy = C::new(k, m)?;
        if symbol_len == 0 {
            return None;
        }
        Some(Self {
            k,
            m,
            n: k + m,
            symbol_len,
            cauchy,
            payloads: vec![0u8; (k + m) * symbol_len],
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

    /// Convenience wrapper around [`SymbolSink::push`].
    pub fn push_symbol(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, StreamError> {
        self.push(idx, payload)
    }

    /// Non-consuming finalization.
    ///
    /// This reconstructs missing systematic data symbols using only a reduced
    /// `r × r` repair submatrix, where `r` is the number of missing data symbols.
    pub fn finalize_ref(&mut self) -> Result<Vec<u8>, StreamError> {
        self.ensure_complete()?;
        let recipe = self.build_recipe()?;
        let mut out = vec![0u8; self.k * self.symbol_len];
        self.apply_recipe_into(&recipe, &mut out);
        Ok(out)
    }

    /// Non-consuming finalization using an external LRU recipe cache.
    ///
    /// Cache hits skip both erasure-pattern analysis and Cauchy inverse
    /// construction. Payload reconstruction still runs because it depends on the
    /// received bytes for this decode instance.
    pub fn finalize_ref_cached(&mut self, cache: &mut RecipeCache) -> Result<Vec<u8>, StreamError> {
        self.ensure_complete()?;
        let recipe = self.recipe_from_cache(cache)?;
        let mut out = vec![0u8; self.k * self.symbol_len];
        self.apply_recipe_into(&recipe, &mut out);
        Ok(out)
    }

    /// Reconstruct into a caller-provided buffer of length `k * symbol_len`.
    ///
    /// # Ownership
    ///
    /// - `out` must have length exactly `k * symbol_len`.
    /// - On success, every byte of `out` is written: present systematic symbols
    ///   are copied, missing symbols are reconstructed (from a zeroed start).
    /// - On error, `out` may be partially modified.
    /// - The decoder is not consumed and may be finalized again.
    pub fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), StreamError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if out.len() != expected {
            return Err(StreamError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let recipe = self.build_recipe()?;
        self.apply_recipe_into(&recipe, out);
        Ok(())
    }

    /// Cached variant of [`finalize_into`](Self::finalize_into).
    pub fn finalize_into_cached(
        &mut self,
        cache: &mut RecipeCache,
        out: &mut [u8],
    ) -> Result<(), StreamError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if out.len() != expected {
            return Err(StreamError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        let recipe = self.recipe_from_cache(cache)?;
        self.apply_recipe_into(&recipe, out);
        Ok(())
    }

    fn recipe_from_cache(
        &self,
        cache: &mut RecipeCache,
    ) -> Result<Arc<recipe::ReconstructionRecipe>, StreamError> {
        let key = recipe::RecipeKey {
            k: self.k,
            m: self.m,
            matrix_type: core::any::type_name::<C>(),
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

    fn ensure_complete(&self) -> Result<(), StreamError> {
        if self.distinct < self.k {
            return Err(StreamError::InsufficientRank {
                rank: self.distinct,
                k: self.k,
            });
        }
        Ok(())
    }

    fn build_recipe(&self) -> Result<recipe::ReconstructionRecipe, StreamError> {
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
            return Err(StreamError::InsufficientRank {
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
        let coefficients =
            cauchy_inverse::rational_lagrange_coefficients(&row_vars, &col_vars, &present_vars);

        // Transpose the reduced inverse into source-major repair terms. Each
        // received repair carries one coefficient for every missing output.
        let mut source_terms = Vec::with_capacity(self.k);
        for (repair_pos, &repair) in repair_cols.iter().enumerate() {
            let coefficients = (0..r)
                .map(|missing_pos| coefficients.inverse[missing_pos * r + repair_pos])
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
                .map(|missing_pos| coefficients.present[present_pos * r + missing_pos])
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

    /// Apply a reconstruction recipe into `out` (`k * symbol_len` bytes).
    ///
    /// Missing rows are zeroed before AXPY; present rows are overwritten by copy.
    fn apply_recipe_into(&self, recipe: &recipe::ReconstructionRecipe, out: &mut [u8]) {
        let slen = self.symbol_len;
        debug_assert_eq!(out.len(), self.k * slen);

        // Zero missing destinations so AXPY starts from a defined state. Present
        // rows are fully overwritten by the copy below.
        for &data_idx in &recipe.missing_data {
            let start = data_idx * slen;
            out[start..start + slen].fill(0);
        }

        // Copy present data symbols directly.
        for &data_idx in &recipe.present_data {
            let src = data_idx * slen;
            out[src..src + slen].copy_from_slice(&self.payloads[src..src + slen]);
        }

        let r = recipe.missing_data.len();
        if r == 0 {
            return;
        }

        // Select the SIMD backend once for this reconstruction when enabled.

        // A single missing output has no cross-output work to share, so retain
        // the lower-overhead single-destination path.
        if r == 1 {
            let out_start = recipe.missing_data[0] * slen;
            let out_row = &mut out[out_start..out_start + slen];
            for term in &recipe.source_terms {
                let src_start = term.source_idx * slen;
                xor_scaled_bytes(
                    out_row,
                    term.coefficients[0],
                    &self.payloads[src_start..src_start + slen],
                );
            }
            return;
        }

        // Share each source load across four outputs when a grouped kernel is
        // available (GFNI on x86, NEON nibble multi-dest on AArch64). Remainders
        // and other backends retain the lower-overhead output-major path.
        #[cfg(feature = "simd")]
        let grouped_outputs = {
            let plan = crate::simd::kernel_plan();
            if r >= 4 && plan.supports_grouped_source_major() {
                r / 4 * 4
            } else {
                0
            }
        };
        #[cfg(not(feature = "simd"))]
        let grouped_outputs = 0;

        #[cfg(feature = "simd")]
        for output_start in (0..grouped_outputs).step_by(4) {
            let mut destinations = crate::simd::IndexedDestinationRows::new(
                out,
                slen,
                &recipe.missing_data[output_start..output_start + 4],
            );
            for term in &recipe.source_terms {
                let source_row = term.source_idx * slen;
                assert!(destinations.xor_scaled_4_grouped(
                    &term.coefficients[output_start..output_start + 4],
                    &self.payloads[source_row..source_row + slen],
                ));
            }
        }

        for (missing_pos, &data_idx) in recipe.missing_data.iter().enumerate().skip(grouped_outputs)
        {
            let out_start = data_idx * slen;
            let out_row = &mut out[out_start..out_start + slen];
            for term in &recipe.source_terms {
                let src_start = term.source_idx * slen;
                xor_scaled_bytes(
                    out_row,
                    term.coefficients[missing_pos],
                    &self.payloads[src_start..src_start + slen],
                );
            }
        }
    }

    #[cfg(all(test, feature = "simd"))]
    fn apply_recipe_source_major_grouped(&self, recipe: &recipe::ReconstructionRecipe) -> Vec<u8> {
        let slen = self.symbol_len;
        let mut out = vec![0u8; self.k * slen];
        for &data_idx in &recipe.missing_data {
            let start = data_idx * slen;
            out[start..start + slen].fill(0);
        }
        for &data_idx in &recipe.present_data {
            let start = data_idx * slen;
            out[start..start + slen].copy_from_slice(&self.payloads[start..start + slen]);
        }

        const OUTPUT_GROUP_SIZE: usize = 4;
        for output_start in (0..recipe.missing_data.len()).step_by(OUTPUT_GROUP_SIZE) {
            let output_end = (output_start + OUTPUT_GROUP_SIZE).min(recipe.missing_data.len());
            let mut destinations = crate::simd::IndexedDestinationRows::new(
                &mut out,
                slen,
                &recipe.missing_data[output_start..output_end],
            );
            for term in &recipe.source_terms {
                let source_row = term.source_idx * slen;
                destinations.xor_scaled_coefficients(
                    &term.coefficients[output_start..output_end],
                    &self.payloads[source_row..source_row + slen],
                );
            }
        }
        out
    }
}

impl<C: CodingMatrix> SymbolSink for LazyDecoderState<C> {
    fn push(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, StreamError> {
        if idx >= self.n {
            return Err(StreamError::IndexOutOfRange {
                index: idx,
                n: self.n,
            });
        }
        if payload.len() != self.symbol_len {
            return Err(StreamError::WrongPayloadLen {
                expected: self.symbol_len,
                got: payload.len(),
            });
        }
        if self.received >= self.n {
            return Err(StreamError::TooManySymbols {
                cap: self.n,
                received: self.received,
            });
        }

        self.received += 1;

        // Once complete, extra symbols are not needed for the selected decode
        // recipe. Match v0.1's behavior by treating them as dependent.
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

    fn finalize(mut self) -> Result<Vec<u8>, StreamError> {
        self.finalize_ref()
    }
}

/// `dst[:] <- dst[:] + coeff * src[:]` over GF(256), with byte slices as field
/// elements.
///
/// Uses `u64`-wide chunking for the table-lookup path so LLVM can lower the
/// inner loop to wider loads/stores. The `coeff == ONE` fast path delegates to
/// [`xor_bytes`], which is already wide-chunked.
fn xor_scaled_bytes(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    crate::payload::xor_scaled_bytes(dst, coefficient, src);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::BatchCodec;

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
            assert_eq!(dec.finalize_ref_cached(&mut cache).unwrap(), data);
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
        assert_eq!(good_decoder.finalize_ref_cached(&mut cache).unwrap(), data);

        let mut standard_decoder =
            LazyDecoderState::<crate::cauchy::CauchyView>::new(k, m, slen).unwrap();
        for &idx in &arrival {
            standard_decoder
                .push_symbol(idx, &standard_symbols[idx])
                .unwrap();
        }
        assert_eq!(
            standard_decoder.finalize_ref_cached(&mut cache).unwrap(),
            data
        );
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 0);
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

    #[cfg(feature = "simd")]
    fn assert_source_major_matches_output_major<C: CodingMatrix>() {
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
                let mut output_major = vec![0u8; k * slen];
                decoder.apply_recipe_into(&recipe, &mut output_major);
                let source_major = decoder.apply_recipe_source_major_grouped(&recipe);
                assert_eq!(source_major, output_major, "slen={slen}, subset={subset:?}");
                assert_eq!(output_major, data, "slen={slen}, subset={subset:?}");
            }
        }

        // Exercise complete groups, the 4→5 boundary, and multiple groups at
        // the whole-recipe level without exhaustively enumerating 8-of-14 sets.
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
                let mut output_major = vec![0u8; k * slen];
                decoder.apply_recipe_into(&recipe, &mut output_major);
                let source_major = decoder.apply_recipe_source_major_grouped(&recipe);
                assert_eq!(source_major, output_major, "slen={slen}, r={r}");
                assert_eq!(output_major, data, "slen={slen}, r={r}");
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn source_major_matches_output_major_at_simd_boundaries() {
        assert_source_major_matches_output_major::<crate::cauchy::CauchyView>();
        assert_source_major_matches_output_major::<crate::good_cauchy::GoodCauchyView>();
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
        assert!(matches!(err, StreamError::WrongOutputLen { .. }));
    }
}
