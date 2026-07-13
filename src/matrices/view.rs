//! GF(256) matrix views and row operations.
//!
//! The streaming decoder in [`crate::decoder`] manipulates an augmented
//! matrix `[A | b]` where `A` is `k x k` of field elements and `b` is
//! `k x symbol_len` of payload bytes (treated as a flat block of GF(256)
//! elements). The work matrix is held as one flat allocation; these views
//! give it shape without copying.
//!
//! All operations are borrow-only: no owning container is introduced on the
//! hot path. Mutability is expressed by `&self`/`&mut self` on the view, not
//! on an owned buffer.
//!
//! The primitives here are deliberately small — `axpy_row`, `scale_row`,
//! `swap_rows`, pivot selection. They are the exact instructions the
//! incremental Gaussian eliminator in [`crate::decoder::gaussian`] composes.

use super::elimination::axpy_row;
use crate::algebra::gf256::GfElem;

/// A borrow-only view over a flat GF(256) matrix stored row-major.
///
/// The backing slice has length `rows * cols`; element `(r, c)` lives at
/// index `r * cols + c`. Views are cheap to construct and re-construct as
/// the caller reshapes or repartitions the underlying buffer (e.g. when the
/// decoder treats the first `k` columns as a coding block and the rest as a
/// payload block).
#[derive(Clone, Copy)]
pub struct MatrixView<'a> {
    pub(super) buf: &'a [GfElem],
    rows: usize,
    cols: usize,
}

/// A mutable borrow-only view, the write-capable counterpart of [`MatrixView`].
pub struct MatrixViewMut<'a> {
    buf: &'a mut [GfElem],
    rows: usize,
    cols: usize,
}

impl<'a> MatrixView<'a> {
    /// Construct a view over `buf` interpreted as `rows x cols`, row-major.
    ///
    /// Returns `None` if `buf.len() != rows * cols` or if either dimension is
    /// zero. The zero-shape check keeps downstream pivot logic simple: every
    /// valid view has at least one element.
    pub fn new(buf: &'a [GfElem], rows: usize, cols: usize) -> Option<Self> {
        if rows == 0 || cols == 0 || buf.len() != rows.checked_mul(cols)? {
            return None;
        }
        Some(Self { buf, rows, cols })
    }

    /// Number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Read element `(r, c)`. Panics if out of bounds in debug.
    pub fn get(&self, r: usize, c: usize) -> GfElem {
        debug_assert!(r < self.rows && c < self.cols, "index out of bounds");
        self.buf[r * self.cols + c]
    }

    /// Read a full row as a slice. No copy.
    pub fn row(&self, r: usize) -> &'a [GfElem] {
        debug_assert!(r < self.rows, "row index out of bounds");
        &self.buf[r * self.cols..(r + 1) * self.cols]
    }

    /// Split this view vertically into `[top | bottom]` at row `at`.
    ///
    /// Used by the decoder to separate the already-pivoted top block from the
    /// not-yet-processed bottom block during incremental elimination.
    pub fn split_rows(&self, at: usize) -> Option<(MatrixView<'a>, MatrixView<'a>)> {
        if at > self.rows {
            return None;
        }
        let top = MatrixView::new(&self.buf[..at * self.cols], at, self.cols)?;
        let bot = MatrixView::new(&self.buf[at * self.cols..], self.rows - at, self.cols)?;
        Some((top, bot))
    }
}

impl<'a> MatrixViewMut<'a> {
    /// Construct a mutable view. Same shape rules as [`MatrixView::new`].
    pub fn new(buf: &'a mut [GfElem], rows: usize, cols: usize) -> Option<Self> {
        if rows == 0 || cols == 0 || buf.len() != rows.checked_mul(cols)? {
            return None;
        }
        Some(Self { buf, rows, cols })
    }

    /// Number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Read element `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> GfElem {
        debug_assert!(r < self.rows && c < self.cols, "index out of bounds");
        self.buf[r * self.cols + c]
    }

    /// Read-only access to a full row.
    pub fn row(&self, r: usize) -> &[GfElem] {
        debug_assert!(r < self.rows, "row index out of bounds");
        &self.buf[r * self.cols..(r + 1) * self.cols]
    }

    /// Mutable access to a full row. The returned slice is `&mut`, so the
    /// caller can mutate it in place (e.g. via [`axpy_row`]).
    pub fn row_mut(&mut self, r: usize) -> &mut [GfElem] {
        debug_assert!(r < self.rows, "row index out of bounds");
        &mut self.buf[r * self.cols..(r + 1) * self.cols]
    }

    /// Write element `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, v: GfElem) {
        debug_assert!(r < self.rows && c < self.cols, "index out of bounds");
        self.buf[r * self.cols + c] = v;
    }

    /// Reborrow as an immutable [`MatrixView`].
    pub fn as_view(&self) -> MatrixView<'_> {
        MatrixView {
            buf: self.buf,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Scale row `r` in place by `s`. O(cols) field multiplications.
    ///
    /// This is the only row op that touches every column of a row with a
    /// multiplication; the eliminator uses it to normalize a pivot row so its
    /// pivot entry becomes `GfElem::ONE`.
    pub fn scale_row(&mut self, r: usize, s: GfElem) {
        let row = self.row_mut(r);
        if s == GfElem::ONE {
            return;
        }
        if s == GfElem::ZERO {
            for slot in row.iter_mut() {
                *slot = GfElem::ZERO;
            }
            return;
        }
        for slot in row.iter_mut() {
            *slot = slot.mul(s);
        }
    }

    /// Swap rows `a` and `b`. O(cols) element copies, no field arithmetic.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        debug_assert!(a < self.rows && b < self.rows, "row index out of bounds");
        let cols = self.cols;
        // Order the indices so `lo < hi`; `split_at_mut` can then borrow the
        // two rows disjointly without violating aliasing.
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (head, tail) = self.buf.split_at_mut(hi * cols);
        // `head` covers rows 0..hi, `tail` starts at row `hi`.
        let row_lo = &mut head[lo * cols..(lo + 1) * cols];
        let row_hi = &mut tail[..cols];
        row_lo.swap_with_slice(row_hi);
    }

    /// Find the first row `>= row_start` whose entry in column `c` is nonzero,
    /// and return its index. Used for pivot selection.
    ///
    /// Returns `None` if no such row exists (the column is exhausted in the
    /// remaining rows).
    pub fn first_nonzero_in_col(&self, c: usize, row_start: usize) -> Option<usize> {
        debug_assert!(c < self.cols, "column index out of bounds");
        (row_start..self.rows).find(|&r| self.get(r, c) != GfElem::ZERO)
    }

    /// Reduce column `pivot_col` to zero in every row except `pivot_row`,
    /// by adding a scaled copy of `pivot_row` into each. The pivot row must
    /// already be normalized so its entry at `(pivot_row, pivot_col)` is
    /// [`GfElem::ONE`].
    ///
    /// This is the "reduce everyone else" half of installing a pivot in
    /// reduced row-echelon form. It is split out because it requires a
    /// disjoint borrow of the pivot row from all other rows — the pattern
    /// that `axpy_row` alone cannot express when both rows live in the same
    /// backing buffer.
    ///
    /// Rows whose entry in `pivot_col` is already zero are skipped (a no-op
    /// `axpy_row` with scale `ZERO`).
    pub fn eliminate_col(&mut self, pivot_row: usize, pivot_col: usize) {
        debug_assert!(
            pivot_row < self.rows && pivot_col < self.cols,
            "pivot index out of bounds",
        );
        debug_assert_eq!(
            self.get(pivot_row, pivot_col),
            GfElem::ONE,
            "pivot row must be normalized before eliminate_col",
        );
        let cols = self.cols;
        // Split the buffer into [rows 0..pivot_row | pivot_row | rows after]
        // so the pivot row can be read while the other two blocks are mutated.
        let (head, tail) = self.buf.split_at_mut(pivot_row * cols);
        let (pivot, rest) = tail.split_at_mut(cols);

        // Rows above the pivot.
        for r in 0..pivot_row {
            let f = head[r * cols + pivot_col];
            if f != GfElem::ZERO {
                let row = &mut head[r * cols..(r + 1) * cols];
                axpy_row(row, f, pivot);
            }
        }
        // Rows below the pivot.
        let remaining = self.rows - pivot_row - 1;
        for r in 0..remaining {
            let f = rest[r * cols + pivot_col];
            if f != GfElem::ZERO {
                let row = &mut rest[r * cols..(r + 1) * cols];
                axpy_row(row, f, pivot);
            }
        }
    }
}
