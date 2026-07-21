//! Common trait for Cauchy-style coding matrices.
//!
//! Both [`crate::cauchy::CauchyView`] (standard index sets) and
//! [`crate::good_cauchy::GoodCauchyView`] (geometric-progression index sets)
//! implement this trait, allowing [`crate::batch::BatchCodec`] and
//! [`crate::decoder::LazyDecoderState`] to work with either matrix.

use crate::gf256::GfElem;

/// Trait for a `k × m` Cauchy matrix view over GF(256).
///
/// The matrix is defined by two disjoint index sets `X = {x_0, …, x_{k-1}}`
/// and `Y = {y_0, …, y_{m-1}}` with entries `C[i,j] = 1/(x_i + y_j)`.
pub trait CodingMatrix: Clone + Copy {
    /// Construct a view for `(k, m)`. Returns `None` when the matrix rejects the
    /// dimensions; concrete matrix types document their exact capacity.
    fn new(k: usize, m: usize) -> Option<Self>;

    /// Number of data symbols `k`.
    fn k(&self) -> usize;

    /// Number of repair symbols `m`.
    fn m(&self) -> usize;

    /// Coefficient `C[i,j] = 1/(x_i + y_j)`.
    fn get(&self, i: usize, j: usize) -> GfElem;

    /// The X index `x_i` for data symbol `i`.
    fn x_var(&self, i: usize) -> GfElem;

    /// The Y index `y_j` for repair symbol `j`.
    fn y_var(&self, j: usize) -> GfElem;

    /// Materialize the full `k × m` coefficient matrix in source-major order:
    /// entry `i * m + j` equals `C[i][j] = get(i, j)`.
    ///
    /// The default builds it entry-by-entry via [`get`](CodingMatrix::get).
    /// Matrix types with a faster factorization (e.g. Good Cauchy) override
    /// this. Precomputing the table once lets the batch encoder use the
    /// source-major, multi-destination SIMD row kernel instead of a
    /// per-`(i, j)` coefficient lookup on every repair.
    fn coefficient_matrix(&self) -> Vec<GfElem> {
        let (k, m) = (self.k(), self.m());
        let mut buf = Vec::with_capacity(k * m);
        for i in 0..k {
            for j in 0..m {
                buf.push(self.get(i, j));
            }
        }
        buf
    }
}
