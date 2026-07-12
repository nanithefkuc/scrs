//! Good Cauchy matrices: Toeplitz-structured Cauchy matrices for fast encoding.
//!
//! A standard Cauchy matrix uses index sets `X = {0, 1, ..., k-1}` and
//! `Y = {k, k+1, ..., k+m-1}`. A **Good Cauchy** matrix chooses `X` and `Y`
//! as geometric progressions in the multiplicative group of GF(256), which
//! makes the matrix **Toeplitz-like**: each row is a scaled version of the
//! same base sequence. This structure enables:
//!
//! - **O(1) per-coefficient access** (same as standard Cauchy)
//! - **O(k) incremental encoding** when data symbols arrive one-by-one
//! - **Better cache locality** because repair-symbol updates traverse
//!   contiguous coefficient sequences.
//!
//! # Construction
//!
//! Let `g` be a generator of the multiplicative group of GF(256). Choose:
//!
//! ```text
//! X = { g^0, g^1, ..., g^(k-1) }
//! Y = { g^k, g^(k+1), ..., g^(k+m-1) }
//! ```
//!
//! The Cauchy entry is:
//!
//! ```text
//! C[i][j] = 1 / (g^i + g^(k+j))
//!         = g^(-i) * 1 / (1 + g^(k+j-i))
//! ```
//!
//! For fixed offset `d = k+j-i`, the term `1/(1+g^d)` is constant across all
//! `(i,j)` with the same diagonal. This is the Toeplitz property: `C[i][j]`
//! depends only on `j-i` (up to the row scaling factor `g^(-i)`).
//!
//! In practice we do not materialise the full matrix; [`GoodCauchyView`]
//! computes coefficients on the fly using the log/exp tables already present
//! in [`crate::gf256`].

use crate::coding_matrix::CodingMatrix;
use crate::gf256::GfElem;

/// A Good Cauchy matrix view with geometric-progression index sets.
///
/// The matrix is defined by `X = {g^0, g^1, ..., g^(k-1)}` and
/// `Y = {g^k, g^(k+1), ..., g^(k+m-1)}` where `g = 0x03` is the AES
/// generator. Entries are computed on the fly via the log/exp tables.
#[derive(Clone, Copy, Debug)]
pub struct GoodCauchyView {
    k: usize,
    m: usize,
}

impl GoodCauchyView {
    /// Construct a view for `(k, m)`.
    ///
    /// Returns `None` if `k == 0 || m == 0 || k + m > 255`. The cap is 255
    /// (not 256) because the geometric progression `g^0 .. g^(n-1)` must
    /// contain `n` *distinct* nonzero elements; `g^255 = 1 = g^0`, so the
    /// cycle length is 255.
    pub fn new(k: usize, m: usize) -> Option<Self> {
        if k == 0 || m == 0 || k + m > 255 {
            return None;
        }
        Some(Self { k, m })
    }

    /// Number of data symbols `k`.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols `m`.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// The X index `x_i = g^i` for row `i`.
    #[inline]
    fn x_at(&self, i: usize) -> GfElem {
        debug_assert!(i < self.k);
        GfElem(exp(i))
    }

    /// The Y index `y_j = g^(k+j)` for column `j`.
    #[inline]
    fn y_at(&self, j: usize) -> GfElem {
        debug_assert!(j < self.m);
        GfElem(exp(self.k + j))
    }

    /// Read the `(i, j)` Good Cauchy coefficient on the fly.
    ///
    /// Computes `1 / (x_i + y_j)` using the log/exp tables.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> GfElem {
        debug_assert!(i < self.k && j < self.m);
        let sum = self.x_at(i).add(self.y_at(j));
        // x_i and y_j are drawn from disjoint geometric progressions;
        // they are never equal because the exponents differ by at least k>=1.
        sum.inv()
    }

    /// Iterate over row `i` as an iterator that recomputes each coefficient.
    #[inline]
    pub fn row(&self, i: usize) -> impl Iterator<Item = GfElem> + '_ {
        debug_assert!(i < self.k);
        let k = self.k;
        let x = self.x_at(i);
        (0..self.m).map(move |j| {
            let y = GfElem(exp(k + j));
            x.add(y).inv()
        })
    }

    /// Materialize the full `k × m` matrix into a flat row-major buffer.
    #[cfg(feature = "std")]
    pub fn to_vec(&self) -> Vec<GfElem> {
        self.coefficient_matrix()
    }

    /// Materialize coefficients via the diagonal factorization
    /// `C[i][j] = g^(-i) · 1/(1 + g^(k+j-i))`.
    ///
    /// This avoids a field inverse per entry: only `k + (k+m-1)` inverses are
    /// needed for the row scales and diagonal bases, then each entry is one
    /// multiply.
    #[cfg(feature = "std")]
    pub fn coefficient_matrix(&self) -> Vec<GfElem> {
        let mut row_scale = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let gi = GfElem(exp(i));
            row_scale.push(if i == 0 { GfElem::ONE } else { gi.inv() });
        }
        let max_d = self.k + self.m - 1;
        let mut base = vec![GfElem::ZERO; max_d + 1];
        for d in 1..=max_d {
            base[d] = GfElem::ONE.add(GfElem(exp(d))).inv();
        }
        let mut buf = Vec::with_capacity(self.k * self.m);
        for i in 0..self.k {
            for j in 0..self.m {
                let d = self.k + j - i;
                buf.push(row_scale[i].mul(base[d]));
            }
        }
        buf
    }

    /// Incremental update: add the contribution of data symbol `i` to all
    /// repair symbols.
    ///
    /// For each repair `j`, this computes `coeff = C[i][j]` and XORs
    /// `coeff * data_byte` into `repair[j][byte]`. This is the hot path for
    /// the streaming encoder.
    ///
    /// # Safety / correctness
    ///
    /// - `data` must have length `symbol_len`.
    /// - `repairs` must have length `m`, and each sub-slice must have length
    ///   `symbol_len`.
    pub fn add_data_symbol(&self, i: usize, data: &[u8], repairs: &mut [&mut [u8]]) {
        debug_assert!(i < self.k);
        debug_assert_eq!(repairs.len(), self.m);
        let symbol_len = data.len();
        for repair in repairs.iter_mut() {
            debug_assert_eq!(repair.len(), symbol_len);
        }

        let x = self.x_at(i);
        for (j, repair) in repairs.iter_mut().enumerate() {
            let y = GfElem(exp(self.k + j));
            let coeff = x.add(y).inv();
            if coeff == GfElem::ZERO {
                continue;
            }
            if coeff == GfElem::ONE {
                for (out, &b) in repair.iter_mut().zip(data.iter()) {
                    *out ^= b;
                }
            } else {
                for (out, &b) in repair.iter_mut().zip(data.iter()) {
                    *out ^= GfElem(b).mul(coeff).0;
                }
            }
        }
    }
}

impl CodingMatrix for GoodCauchyView {
    fn new(k: usize, m: usize) -> Option<Self> {
        Self::new(k, m)
    }

    fn k(&self) -> usize {
        self.k
    }

    fn m(&self) -> usize {
        self.m
    }

    fn get(&self, i: usize, j: usize) -> GfElem {
        self.get(i, j)
    }

    fn x_var(&self, i: usize) -> GfElem {
        GfElem(exp(i))
    }

    fn y_var(&self, j: usize) -> GfElem {
        GfElem(exp(self.k + j))
    }
}

/// Exponential `g^e` using the compile-time `EXP` table.
///
/// `e` is taken modulo 255 because `g^255 = 1`.
#[inline]
fn exp(e: usize) -> u8 {
    #[cfg(feature = "gf256-lookup")]
    {
        crate::gf256::EXP[e % 255]
    }
    #[cfg(not(feature = "gf256-lookup"))]
    {
        // Fallback: compute g^e by repeated squaring via xtime.
        // This is slow but correct for no-alloc builds.
        let mut result = GfElem::ONE;
        let mut base = GfElem(crate::gf256::GENERATOR);
        let mut exp = e % 255;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_xtime(base);
            }
            base = base.mul_xtime(base);
            exp >>= 1;
        }
        result.0
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn view_dimensions() {
        let v = GoodCauchyView::new(4, 3).unwrap();
        assert_eq!(v.k(), 4);
        assert_eq!(v.m(), 3);
    }

    #[test]
    fn view_rejects_oversized() {
        // k + m > 255 is out of scope (cycle length of multiplicative group).
        assert!(GoodCauchyView::new(200, 100).is_none());
        assert!(GoodCauchyView::new(0, 5).is_none());
        assert!(GoodCauchyView::new(5, 0).is_none());
    }

    #[test]
    fn coeff_nonzero() {
        let v = GoodCauchyView::new(5, 4).unwrap();
        for i in 0..5 {
            for j in 0..4 {
                let c = v.get(i, j);
                assert_ne!(c, GfElem::ZERO, "zero coeff at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn row_iterator_matches_get() {
        let v = GoodCauchyView::new(4, 6).unwrap();
        for i in 0..4 {
            let row: Vec<GfElem> = v.row(i).collect();
            for (j, c) in row.iter().enumerate() {
                assert_eq!(*c, v.get(i, j));
            }
        }
    }

    #[test]
    fn coefficient_matrix_matches_get() {
        let v = GoodCauchyView::new(16, 8).unwrap();
        let matrix = v.coefficient_matrix();
        for i in 0..16 {
            for j in 0..8 {
                assert_eq!(matrix[i * 8 + j], v.get(i, j));
            }
        }
    }

    #[test]
    fn incremental_update_matches_batch() {
        let k = 4;
        let m = 3;
        let symbol_len = 8;
        let view = GoodCauchyView::new(k, m).unwrap();

        let data: Vec<u8> = (0..k * symbol_len).map(|i| i as u8).collect();

        // Batch repair computation (reference).
        let mut batch_repairs: Vec<Vec<u8>> = vec![vec![0u8; symbol_len]; m];
        for j in 0..m {
            for i in 0..k {
                let coeff = view.get(i, j);
                let data_slice = &data[i * symbol_len..(i + 1) * symbol_len];
                for (out, &b) in batch_repairs[j].iter_mut().zip(data_slice.iter()) {
                    *out ^= GfElem(b).mul(coeff).0;
                }
            }
        }

        // Incremental repair computation.
        let mut inc_repairs: Vec<Vec<u8>> = vec![vec![0u8; symbol_len]; m];
        for i in 0..k {
            let data_slice = &data[i * symbol_len..(i + 1) * symbol_len];
            let mut repair_slices: Vec<&mut [u8]> =
                inc_repairs.iter_mut().map(|r| r.as_mut_slice()).collect();
            view.add_data_symbol(i, data_slice, &mut repair_slices);
        }

        assert_eq!(batch_repairs, inc_repairs);
    }

    proptest! {
        #[test]
        fn view_index_roundtrip(k in 1usize..=10, m in 1usize..=10) {
            let v = GoodCauchyView::new(k, m).unwrap();
            for i in 0..k {
                for j in 0..m {
                    let c = v.get(i, j);
                    prop_assert_ne!(c, GfElem::ZERO, "zero coeff at ({},{})", i, j);
                }
            }
        }
    }
}
