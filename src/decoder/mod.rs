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
        Ok(self.apply_recipe(&recipe))
    }

    /// Non-consuming finalization using an external LRU recipe cache.
    ///
    /// Cache hits skip both erasure-pattern analysis and Cauchy inverse
    /// construction. Payload reconstruction still runs because it depends on the
    /// received bytes for this decode instance.
    pub fn finalize_ref_cached(&mut self, cache: &mut RecipeCache) -> Result<Vec<u8>, StreamError> {
        self.ensure_complete()?;
        let key = recipe::RecipeKey {
            k: self.k,
            m: self.m,
            pattern: self.pattern,
        };
        let recipe = if let Some(recipe) = cache.get(key) {
            recipe
        } else {
            let recipe = self.build_recipe()?;
            cache.insert(key, recipe.clone());
            recipe
        };
        Ok(self.apply_recipe(&recipe))
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
                repair_cols,
                repair_terms: Vec::new(),
                data_terms: Vec::new(),
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

        let mut repair_terms = Vec::with_capacity(r);
        for missing_pos in 0..r {
            let mut terms = Vec::with_capacity(r);
            for repair_pos in 0..r {
                let coeff = coefficients.inverse[missing_pos * r + repair_pos];
                if coeff != GfElem::ZERO {
                    terms.push(recipe::RhsTerm {
                        rhs_pos: repair_pos,
                        scale: recipe::ScaleTable::new(coeff),
                    });
                }
            }
            repair_terms.push(terms);
        }

        // Direct fused coefficients replace the former length-r A^-1*C dot
        // product for every (present source, missing output) pair.
        let mut data_terms = Vec::with_capacity(r);
        for missing_pos in 0..r {
            let mut terms = Vec::with_capacity(present_data.len());
            for (present_pos, &data_idx) in present_data.iter().enumerate() {
                let coeff = coefficients.present[present_pos * r + missing_pos];
                if coeff != GfElem::ZERO {
                    terms.push(recipe::DataTerm {
                        data_idx,
                        scale: recipe::ScaleTable::new(coeff),
                    });
                }
            }
            data_terms.push(terms);
        }

        Ok(recipe::ReconstructionRecipe {
            missing_data,
            present_data,
            repair_cols,
            repair_terms,
            data_terms,
        })
    }

    fn apply_recipe(&self, recipe: &recipe::ReconstructionRecipe) -> Vec<u8> {
        let slen = self.symbol_len;
        let mut out = vec![0u8; self.k * slen];

        // Copy present data symbols directly.
        for &data_idx in &recipe.present_data {
            let src = data_idx * slen;
            out[src..src + slen].copy_from_slice(&self.payloads[src..src + slen]);
        }

        let r = recipe.missing_data.len();
        if r == 0 {
            return out;
        }

        // Fused reconstruction: for each missing output m_j,
        //   out[m_j] = sum_{r_i} P[m_j, r_i] * repair[r_i]
        //            XOR sum_{present d} Q[m_j, d] * data[d]
        // where P and Q are precomputed in the recipe. This reads each source
        // symbol exactly once per missing output (no intermediate RHS buffer)
        // and avoids the separate RHS computation pass.
        for (missing_pos, &data_idx) in recipe.missing_data.iter().enumerate() {
            let out_start = data_idx * slen;
            let out_row = &mut out[out_start..out_start + slen];

            // Repair contributions.
            for term in &recipe.repair_terms[missing_pos] {
                let repair_symbol_idx = self.k + recipe.repair_cols[term.rhs_pos];
                let repair_start = repair_symbol_idx * slen;
                let src = &self.payloads[repair_start..repair_start + slen];
                xor_scaled_bytes(out_row, &term.scale, src);
            }

            // Present-data contributions (fused coefficients).
            for term in &recipe.data_terms[missing_pos] {
                let data_start = term.data_idx * slen;
                let src = &self.payloads[data_start..data_start + slen];
                xor_scaled_bytes(out_row, &term.scale, src);
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
fn xor_scaled_bytes(dst: &mut [u8], scale: &crate::simd::ScaleTable, src: &[u8]) {
    crate::simd::xor_scaled_bytes(dst, scale, src);
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
}
