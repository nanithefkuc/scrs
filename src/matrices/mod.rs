//! Coding-matrix definitions and reference algorithms.

pub mod cauchy;
pub mod coding_matrix;
/// Reference row-elimination algorithms.
pub mod elimination;
pub mod good_cauchy;
pub mod view;

pub use elimination::{axpy_row, det, rref};
pub use view::{MatrixView, MatrixViewMut};
