//! Streaming Cauchy Reed-Solomon erasure coding.
//!
//! SCRS provides systematic Cauchy-RS encoding and a lazy, payload-deferred
//! streaming decoder optimized for predictable receive-path latency. The
//! decoder performs incremental Gaussian elimination on the *coefficient
//! matrix only* — payload bytes are not touched until the block reaches full
//! rank, at which point a single fused reconstruction pass recovers all
//! missing data symbols.
//!
//! # Scope
//!
//! The current crate scope is `k + m <= 256` (GF(256) index assignment). A
//! future GF(2¹⁶) backend will lift this ceiling.

#![warn(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod algebra;
pub mod matrices;

/// Compatibility facade for the former root matrix module.
pub mod matrix {
    pub use crate::matrices::{MatrixView, MatrixViewMut, axpy_row, det, rref};
}

pub use algebra::gf256;
pub use matrices::{cauchy, coding_matrix, good_cauchy};

pub mod batch;
pub mod decoder;
pub mod encoder;
pub use decoder::pattern as pattern_key;
mod payload;
#[cfg(feature = "simd")]
mod simd;
pub mod transport;
pub use transport::symbol_sink as stream;
