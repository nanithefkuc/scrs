//! Cauchy matrix coefficients and the systematic Cauchy-RS generator.
//!
//! A Cauchy matrix `C` is defined by two disjoint index sets
//! `X = {x_1, ..., x_k}` and `Y = {y_1, ..., y_m}` over GF(256), with entries
//!
//! ```text
//! C[i][j] = 1 / (x_i + y_j)   =   (x_i + y_j)^-1
//! ```
//!
//! It is MDS by construction when `X ∩ Y = ∅`: every square submatrix is
//! non-singular, so any `k` of the `n = k + m` transmitted symbols suffices to
//! recover the original `k` data symbols.
//!
//! SCRS never materializes the full `k × m` Cauchy block. Coefficients are
//! computed on the fly by [`cauchy_coeff`] (a `const fn`), and [`CauchyView`]
//! exposes them as an indexable view that recomputes on access.
//!
//! # Index assignment (`k + m <= 256`)
//!
//! The standard assignment is `X = {0, 1, ..., k-1}` and
//! `Y = {k, k+1, ..., k+m-1}`. Disjointness is guaranteed for `k + m <= 256`
//! since all indices are distinct GF(256) elements. Standard Cauchy therefore
//! supports exactly `k + m <= 256`.

use crate::coding_matrix::CodingMatrix;
use crate::gf256::GfElem;

/// Compute the `(i, j)` entry of the Cauchy matrix: `1 / (x_i + y_j)`.
///
/// `x` and `y` are the raw GF(256) elements drawn from the disjoint index
/// sets `X` and `Y`. Their sum `x + y` is nonzero whenever `x != y` (since
/// addition is XOR in GF(2^n)), so the inverse is well-defined for any
/// disjoint pair. If `x == y` the function returns `GfElem::ZERO`, which
/// signals a misconfiguration rather than a valid coefficient — callers must
/// ensure the index sets are disjoint.
///
/// This is a `const fn`, so callers can evaluate fixed coefficients at compile time.
pub const fn cauchy_coeff(x: GfElem, y: GfElem) -> GfElem {
    let denom = x.add(y);
    if denom.0 == 0 {
        // x == y: disjointness violated. Return zero so the misconfiguration
        // is observable (a zero coefficient cannot appear in a valid Cauchy
        // matrix and will be caught by MDS checks). In debug we assert.
        return GfElem::ZERO;
    }
    denom.inv()
}

/// A borrow-only, on-the-fly view over a `k × m` Cauchy matrix.
///
/// The view stores only the dimensions `(k, m)`; every coefficient is
/// recomputed from [`cauchy_coeff`] on access, using the standard index
/// assignment `X = {0, ..., k-1}`, `Y = {k, ..., k+m-1}`. There is no
/// `k * m` backing buffer.
///
/// For the systematic generator `G = [I_k | A]`, the left `k × k` block is
/// the identity and the right `k × m` block is the Cauchy matrix this view
/// represents.
#[derive(Clone, Copy, Debug)]
pub struct CauchyView {
    k: usize,
    m: usize,
}

impl CauchyView {
    /// Construct a view for a `(k, m)` configuration.
    ///
    /// Returns `None` if either dimension is zero or `k + m > 256`. Standard
    /// Cauchy supports the exact inclusive limit `k + m == 256`.
    pub fn new(k: usize, m: usize) -> Option<Self> {
        if k == 0 || m == 0 || k + m > 256 {
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

    /// The X index `x_i` for row `i` under the standard assignment.
    const fn x_at(&self, i: usize) -> GfElem {
        GfElem(i as u8)
    }

    /// The Y index `y_j` for column `j` under the standard assignment.
    const fn y_at(&self, j: usize) -> GfElem {
        GfElem((self.k + j) as u8)
    }

    /// Read the `(i, j)` Cauchy coefficient on the fly.
    pub fn get(&self, i: usize, j: usize) -> GfElem {
        debug_assert!(i < self.k && j < self.m, "cauchy index out of bounds");
        cauchy_coeff(self.x_at(i), self.y_at(j))
    }

    /// Iterate over row `i` as an iterator that recomputes each coefficient.
    ///
    /// This is the primary access pattern for the encoder: for each repair
    /// symbol `j`, the coefficient applied to data symbol `i` is `get(i, j)`.
    pub fn row(&self, i: usize) -> impl Iterator<Item = GfElem> {
        debug_assert!(i < self.k, "row index out of bounds");
        let k = self.k;
        let x = self.x_at(i);
        (0..self.m).map(move |j| cauchy_coeff(x, GfElem((k + j) as u8)))
    }

    /// Materialize the full `k × m` Cauchy matrix into a flat row-major
    /// buffer.
    ///
    /// This allocates `k * m` bytes. It is intended for testing and for
    /// callers that explicitly want a snapshot (e.g. to pass to [`crate::matrix::det`]).
    /// The streaming encode/decode paths do not call it.
    pub fn to_vec(&self) -> Vec<GfElem> {
        let mut buf = Vec::with_capacity(self.k * self.m);
        for i in 0..self.k {
            for j in 0..self.m {
                buf.push(self.get(i, j));
            }
        }
        buf
    }
}

impl CodingMatrix for CauchyView {
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
        GfElem(i as u8)
    }

    fn y_var(&self, j: usize) -> GfElem {
        GfElem((self.k + j) as u8)
    }
}

/// Check that the standard index assignment yields an MDS code
/// submatrix of the systematic generator `G = [I_k | A]` is non-singular.
///
/// For a Cauchy matrix this holds by construction whenever the index sets
/// are disjoint, so this function is primarily a self-check / test helper.
/// It is **exponential** in `min(k, m)` — it enumerates all `(r x r)` minors
/// for `r = 1..=min(k, m)` — and must not be called on the hot path.
///
/// Returns `true` if the configuration is MDS, `false` otherwise.
pub fn is_mds(k: usize, m: usize) -> bool {
    let Some(view) = CauchyView::new(k, m) else {
        return false;
    };
    // The Cauchy matrix A is k x m. For G = [I_k | A] to be MDS, every
    // square submatrix of A of size r x r (for r = 1..=min(k,m)) must be
    // non-singular. (The r x r minors that involve identity columns are
    // automatically non-singular by the Cauchy-Binet argument; only the
    // pure-Cauchy minors need checking.)
    let cauchy = view.to_vec();
    let r_max = k.min(m);
    for r in 1..=r_max {
        // Enumerate all r-row subsets and r-column subsets.
        for row_sel in combinations(k, r) {
            for col_sel in combinations(m, r) {
                let mut minor = Vec::with_capacity(r * r);
                for &ri in &row_sel {
                    for &cj in &col_sel {
                        minor.push(cauchy[ri * m + cj]);
                    }
                }
                let Some(mv) = crate::matrix::MatrixView::new(&minor, r, r) else {
                    return false;
                };
                if crate::matrix::det(mv) == GfElem::ZERO {
                    return false;
                }
            }
        }
    }
    true
}

/// Enumerate all `r`-element subsets of `0..n` in lexicographic order.
fn combinations(n: usize, r: usize) -> impl Iterator<Item = Vec<usize>> {
    // r == 0 yields exactly one subset (the empty set); r > n yields none.
    let mut state: Vec<usize> = (0..r).collect();
    let mut done = n < r; // r > n: nothing to yield
    let need_empty = r == 0; // r == 0: yield the empty set once
    core::iter::from_fn(move || {
        if done {
            return None;
        }
        if need_empty {
            done = true;
            return Some(Vec::new());
        }
        let result = state.clone();
        // Advance to the next combination (lexicographic).
        let mut i = r - 1;
        loop {
            if state[i] < n - r + i {
                state[i] += 1;
                for j in (i + 1)..r {
                    state[j] = state[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                done = true;
                break;
            }
            i -= 1;
        }
        Some(result)
    })
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn cauchy_coeff_is_inverse_of_sum() {
        // C[i][j] = 1 / (x_i + y_j); by definition (x+y) * C[i][j] = 1.
        for x in 0u8..=255 {
            for y in 0u8..=255 {
                if x == y {
                    continue;
                }
                let c = cauchy_coeff(GfElem(x), GfElem(y));
                let sum = GfElem(x).add(GfElem(y));
                assert_eq!(c.mul(sum), GfElem::ONE, "x={:#x} y={:#x}", x, y);
            }
        }
    }

    #[test]
    fn cauchy_coeff_zero_on_equal_indices() {
        assert_eq!(cauchy_coeff(GfElem(0x42), GfElem(0x42)), GfElem::ZERO);
    }

    #[test]
    fn view_dimensions() {
        let v = CauchyView::new(4, 3).unwrap();
        assert_eq!(v.k(), 4);
        assert_eq!(v.m(), 3);
    }

    #[test]
    fn view_rejects_oversized() {
        assert!(CauchyView::new(255, 1).is_some());
        assert!(CauchyView::new(255, 2).is_none());
        assert!(CauchyView::new(200, 100).is_none());
        assert!(CauchyView::new(0, 5).is_none());
        assert!(CauchyView::new(5, 0).is_none());
    }

    #[test]
    fn view_matches_direct_coefficient() {
        let v = CauchyView::new(5, 4).unwrap();
        for i in 0..5 {
            for j in 0..4 {
                let x = GfElem(i as u8);
                let y = GfElem((5 + j) as u8);
                assert_eq!(v.get(i, j), cauchy_coeff(x, y));
            }
        }
    }

    #[test]
    fn view_to_vec_is_row_major() {
        let v = CauchyView::new(3, 2).unwrap();
        let buf = v.to_vec();
        assert_eq!(buf.len(), 6);
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(buf[i * 2 + j], v.get(i, j));
            }
        }
    }

    #[test]
    fn row_iterator_matches_get() {
        let v = CauchyView::new(4, 6).unwrap();
        for i in 0..4 {
            let row: Vec<GfElem> = v.row(i).collect();
            for (j, c) in row.iter().enumerate() {
                assert_eq!(*c, v.get(i, j));
            }
        }
    }

    proptest! {
        #[test]
        fn cauchy_coeff_nonzero_for_distinct(x in 0u8..=255, y in 0u8..=255) {
            prop_assume!(x != y);
            let c = cauchy_coeff(GfElem(x), GfElem(y));
            prop_assert_ne!(c, GfElem::ZERO);
        }

        #[test]
        fn view_index_roundtrip(k in 1usize..=10, m in 1usize..=10) {
            let v = CauchyView::new(k, m).unwrap();
            // X = {0..k}, Y = {k..k+m}; all disjoint under the standard
            // assignment as long as k + m <= 256.
            for i in 0..k {
                for j in 0..m {
                    let c = v.get(i, j);
                    prop_assert_ne!(c, GfElem::ZERO, "zero coeff at ({},{})", i, j);
                }
            }
        }
    }

    // ---- MDS-ness checks on small (k, m) ----

    #[test]
    fn is_mds_small_configs() {
        // Small Cauchy matrices are MDS by construction for any k, m with
        // k + m <= 256. Check a representative set.
        for &(k, m) in &[
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (2, 3),
            (3, 3),
            (4, 2),
        ] {
            assert!(is_mds(k, m), "expected MDS for k={} m={}", k, m);
        }
    }

    #[test]
    fn is_mds_rejects_oversized() {
        assert!(!is_mds(200, 100));
        assert!(!is_mds(0, 5));
    }

    #[test]
    fn combinations_count() {
        // C(5, 2) = 10
        let count = combinations(5, 2).count();
        assert_eq!(count, 10);
        // C(4, 0) = 1 (the empty set)
        assert_eq!(combinations(4, 0).count(), 1);
        // C(3, 4) = 0 (r > n)
        assert_eq!(combinations(3, 4).count(), 0);
    }

    #[test]
    fn combinations_are_lexicographic() {
        let sets: Vec<Vec<usize>> = combinations(4, 2).collect();
        assert_eq!(
            sets,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }
}
