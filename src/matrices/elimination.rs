use super::view::{MatrixView, MatrixViewMut};
use crate::algebra::gf256::GfElem;

/// `dst[:] <- dst + s * src[:]`, elementwise over GF(256).
///
/// Both slices must have equal length. This is the workhorse of incremental
/// Gaussian elimination: it is the inner loop that zeros a column in a target
/// row by folding in a (scaled) pivot row.
///
/// Length equality is checked with `debug_assert!` only.
pub fn axpy_row(dst: &mut [GfElem], s: GfElem, src: &[GfElem]) {
    debug_assert_eq!(dst.len(), src.len(), "axpy_row length mismatch");
    if s == GfElem::ZERO {
        return;
    }
    if s == GfElem::ONE {
        for (d, &x) in dst.iter_mut().zip(src.iter()) {
            *d = d.add(x);
        }
        return;
    }
    for (d, &x) in dst.iter_mut().zip(src.iter()) {
        *d = d.add(s.mul(x));
    }
}

/// Reduce a matrix to reduced row-echelon form (RREF) via row-only pivoting,
/// using first-nonzero-column pivot selection. Returns the rank.
///
/// The pivot for column `c` is placed in row `rank` (the first available row
/// at or below `rank` with a nonzero entry in `c`). The pivot row is
/// normalized to a leading `1`, then [`MatrixViewMut::eliminate_col`] zeros
/// the column in every other row. This leaves the matrix in RREF: each pivot
/// column is a standard basis vector.
///
/// This is a reference routine. The streaming decoder (Phase 3) achieves the
/// same end state incrementally — one arriving symbol at a time — and will
/// be validated against this implementation.
pub fn rref(view: &mut MatrixViewMut<'_>) -> usize {
    let rows = view.rows();
    let cols = view.cols();
    let mut rank = 0;
    for c in 0..cols {
        if rank >= rows {
            break;
        }
        let Some(pivot) = view.first_nonzero_in_col(c, rank) else {
            continue;
        };
        if pivot != rank {
            view.swap_rows(rank, pivot);
        }
        let pv = view.get(rank, c);
        if pv != GfElem::ONE {
            view.scale_row(rank, pv.inv());
        }
        view.eliminate_col(rank, c);
        rank += 1;
    }
    rank
}

/// Compute the determinant of an `n x n` GF(256) matrix via Gaussian
/// elimination. Returns `GfElem::ZERO` for singular matrices.
///
/// This is a reference routine: it allocates a copy so it can be used on
/// immutable input. The streaming decoder does not call it on the hot path,
/// but the test suite uses it to verify MDS-ness of Cauchy matrices.
pub fn det(matrix: MatrixView<'_>) -> GfElem {
    let n = matrix.rows();
    debug_assert_eq!(n, matrix.cols(), "det requires a square matrix");
    let mut buf: Vec<GfElem> = matrix.buf.to_vec();
    let mut det_acc = GfElem::ONE;
    let mut row = 0usize;
    for col in 0..n {
        // Find a pivot.
        let pivot = (row..n).find(|&r| buf[r * n + col] != GfElem::ZERO);
        let Some(pivot) = pivot else {
            return GfElem::ZERO;
        };
        if pivot != row {
            for c in 0..n {
                let (a, b) = (buf[row * n + c], buf[pivot * n + c]);
                buf[row * n + c] = b;
                buf[pivot * n + c] = a;
            }
        }
        let pv = buf[row * n + col];
        det_acc = det_acc.mul(pv);
        let inv = pv.inv();
        // Normalize the pivot row.
        for c in col..n {
            buf[row * n + c] = buf[row * n + c].mul(inv);
        }
        // Eliminate this column from all other rows.
        for r in 0..n {
            if r == row {
                continue;
            }
            let factor = buf[r * n + col];
            if factor == GfElem::ZERO {
                continue;
            }
            for c in col..n {
                let prod = factor.mul(buf[row * n + c]);
                buf[r * n + c] = buf[r * n + c].add(prod);
            }
        }
        row += 1;
    }
    det_acc
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn any_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Vec<GfElem>> {
        proptest::collection::vec((0u8..=255).prop_map(GfElem), rows * cols)
    }

    proptest! {
        #[test]
        fn get_set_roundtrip(rows in 1usize..=6, cols in 1usize..=6) {
            let mut buf: Vec<GfElem> = (0..rows * cols).map(|i| GfElem(i as u8)).collect();
            let mut v = MatrixViewMut::new(&mut buf, rows, cols).unwrap();
            for r in 0..rows {
                for c in 0..cols {
                    let val = GfElem(((r * 7 + c * 3) & 0xff) as u8);
                    v.set(r, c, val);
                    prop_assert_eq!(v.get(r, c), val);
                }
            }
        }

        #[test]
        fn scale_row_zero_clears(rows in 1usize..=6, cols in 1usize..=6, r in 0usize..6) {
            let r = r % rows;
            let mut buf: Vec<GfElem> = (0..rows * cols).map(|i| GfElem((i as u8).wrapping_mul(3))).collect();
            let mut v = MatrixViewMut::new(&mut buf, rows, cols).unwrap();
            v.scale_row(r, GfElem::ZERO);
            for c in 0..cols {
                prop_assert_eq!(v.get(r, c), GfElem::ZERO);
            }
        }

        #[test]
        fn scale_row_one_identity(rows in 1usize..=6, cols in 1usize..=6, r in 0usize..6) {
            let r = r % rows;
            let mut buf: Vec<GfElem> = (0..rows * cols).map(|i| GfElem(i as u8)).collect();
            let before = buf.clone();
            let mut v = MatrixViewMut::new(&mut buf, rows, cols).unwrap();
            v.scale_row(r, GfElem::ONE);
            prop_assert_eq!(&buf, &before);
        }

        #[test]
        fn swap_rows_symmetric(rows in 1usize..=6, cols in 1usize..=6, a in 0usize..6, b in 0usize..6) {
            let a = a % rows;
            let b = b % rows;
            let mut buf: Vec<GfElem> = (0..rows * cols).map(|i| GfElem(i as u8)).collect();
            let before = buf.clone();
            let mut v = MatrixViewMut::new(&mut buf, rows, cols).unwrap();
            v.swap_rows(a, b);
            for c in 0..cols {
                prop_assert_eq!(v.get(a, c), before[b * cols + c]);
                prop_assert_eq!(v.get(b, c), before[a * cols + c]);
            }
        }

        #[test]
        fn axpy_zero_is_noop(buf in any_matrix(1, 8)) {
            let mut dst = buf.clone();
            let src = buf.clone();
            axpy_row(&mut dst, GfElem::ZERO, &src);
            prop_assert_eq!(dst, buf);
        }

        #[test]
        fn axpy_one_is_xor(buf_a in any_matrix(1, 8), buf_b in any_matrix(1, 8)) {
            let mut dst = buf_a.clone();
            let src = buf_b.clone();
            axpy_row(&mut dst, GfElem::ONE, &src);
            for i in 0..dst.len() {
                prop_assert_eq!(dst[i], GfElem(buf_a[i].0 ^ buf_b[i].0));
            }
        }

        #[test]
        fn axpy_distributes(buf_a in any_matrix(1, 8), buf_b in any_matrix(1, 8), s in (0u8..=255).prop_map(GfElem)) {
            let mut dst = buf_a.clone();
            let src = buf_b.clone();
            axpy_row(&mut dst, s, &src);
            // Verify element-wise: dst[i] == a[i] + s * b[i].
            for i in 0..dst.len() {
                let expected = buf_a[i].add(s.mul(buf_b[i]));
                prop_assert_eq!(dst[i], expected);
            }
        }
    }

    #[test]
    fn det_identity_is_one() {
        let buf: Vec<GfElem> = (0..3)
            .flat_map(|r| (0..3).map(move |c| GfElem(if r == c { 1 } else { 0 })))
            .collect();
        let v = MatrixView::new(&buf, 3, 3).unwrap();
        assert_eq!(det(v), GfElem::ONE);
    }

    #[test]
    fn det_singular_is_zero() {
        // Two identical rows => linearly dependent => singular.
        let buf: Vec<GfElem> = vec![
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x07),
            GfElem(0x05),
            GfElem(0x01),
        ];
        let v = MatrixView::new(&buf, 3, 3).unwrap();
        assert_eq!(det(v), GfElem::ZERO);
    }

    #[test]
    fn det_diagonal_is_product() {
        let diag = [GfElem(0x05), GfElem(0x27), GfElem(0xF3)];
        let mut buf = vec![GfElem::ZERO; 9];
        for i in 0..3 {
            buf[i * 3 + i] = diag[i];
        }
        let v = MatrixView::new(&buf, 3, 3).unwrap();
        let expected = diag[0].mul(diag[1]).mul(diag[2]);
        assert_eq!(det(v), expected);
    }

    // ---- eliminate_col + rref ----

    #[test]
    fn eliminate_col_zeros_other_rows() {
        // 3x3 matrix; pivot is row 1, col 1, already normalized to 1.
        //   [5 0 7]      [5 0 7]
        //   [3 1 2]  ->  [0 1 0]  (row 0 gets 5*0=0? no: row0[col1]=0, skip)
        //   [9 4 6]      [0 0 ?]  (row 2: subtract 4 * row1)
        // Actually row 0 has 0 in col 1, so it's untouched.
        let mut buf: Vec<GfElem> = vec![
            GfElem(0x05),
            GfElem(0x00),
            GfElem(0x07),
            GfElem(0x03),
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x09),
            GfElem(0x04),
            GfElem(0x06),
        ];
        let mut v = MatrixViewMut::new(&mut buf, 3, 3).unwrap();
        v.eliminate_col(1, 1);
        // Row 0: col 1 is 0, unchanged.
        assert_eq!(v.row(0), &[GfElem(0x05), GfElem(0x00), GfElem(0x07)]);
        // Row 1: pivot, unchanged.
        assert_eq!(v.row(1), &[GfElem(0x03), GfElem(0x01), GfElem(0x02)]);
        // Row 2: entry in col 1 was 4, eliminated to 0. Other cols updated.
        assert_eq!(v.get(2, 1), GfElem::ZERO);
        // row2 = row2 - 4 * row1 = [9,4,6] + 4*[3,1,2] (GF add = XOR)
        //   = [9^4*3, 0, 6^4*2]
        //   4*3 = 0x0C (mul), 9^0x0C = 0x09^0x0C = 0x05
        //   4*2 = 0x08, 6^0x08 = 0x06^0x08 = 0x0E
        assert_eq!(v.row(2), &[GfElem(0x05), GfElem(0x00), GfElem(0x0E)]);
    }

    #[test]
    fn rref_identity_is_identity() {
        let mut buf: Vec<GfElem> = (0..3)
            .flat_map(|r| (0..3).map(move |c| GfElem(if r == c { 1 } else { 0 })))
            .collect();
        let mut v = MatrixViewMut::new(&mut buf, 3, 3).unwrap();
        let rank = rref(&mut v);
        assert_eq!(rank, 3);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(v.get(r, c), GfElem(if r == c { 1 } else { 0 }));
            }
        }
    }

    #[test]
    fn rref_rank_deficient() {
        // Two identical rows -> rank 2.
        let mut buf: Vec<GfElem> = vec![
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x07),
            GfElem(0x05),
            GfElem(0x01),
        ];
        let mut v = MatrixViewMut::new(&mut buf, 3, 3).unwrap();
        let rank = rref(&mut v);
        assert_eq!(rank, 2);
    }

    #[test]
    fn rref_solves_system() {
        // Solve M * x = b where M is 3x3 and b is the 4th column.
        // After RREF on [M | b], the last column holds x.
        // M = [[2, 3, 1], [1, 1, 1], [1, 2, 3]], b = [4, 3, 5]
        let mut buf: Vec<GfElem> = vec![
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x01),
            GfElem(0x04),
            GfElem(0x01),
            GfElem(0x01),
            GfElem(0x01),
            GfElem(0x03),
            GfElem(0x01),
            GfElem(0x02),
            GfElem(0x03),
            GfElem(0x05),
        ];
        let mut v = MatrixViewMut::new(&mut buf, 3, 4).unwrap();
        let rank = rref(&mut v);
        assert_eq!(rank, 3);
        // The left 3x3 is now identity; the 4th column is the solution.
        // Verify by substituting back: M_orig * x = b.
        let x = [v.get(0, 3), v.get(1, 3), v.get(2, 3)];
        let m_orig: [[GfElem; 3]; 3] = [
            [GfElem(0x02), GfElem(0x03), GfElem(0x01)],
            [GfElem(0x01), GfElem(0x01), GfElem(0x01)],
            [GfElem(0x01), GfElem(0x02), GfElem(0x03)],
        ];
        let b = [GfElem(0x04), GfElem(0x03), GfElem(0x05)];
        for r in 0..3 {
            let mut acc = GfElem::ZERO;
            for c in 0..3 {
                acc = acc.add(m_orig[r][c].mul(x[c]));
            }
            assert_eq!(acc, b[r], "row {} mismatch", r);
        }
    }
}
