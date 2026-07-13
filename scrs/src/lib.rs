//! Streaming Cauchy Reed-Solomon erasure coding.
//!
//! SCRS provides systematic Cauchy-RS encoding and a lazy, payload-deferred
//! streaming decoder optimized for predictable receive-path latency. The
//! decoder records symbols as they arrive and defers payload reconstruction
//! until `k` independent symbols are available.
//!
//! # Matrix capacities
//!
//! Batch callers select a coding matrix explicitly. Standard Cauchy supports
//! `k + m <= 256`; Good Cauchy supports `k + m <= 255`.
//! [`encoder::StreamingEncoder`] uses Good Cauchy and therefore has the
//! `k + m <= 255` limit. A [`decoder::LazyDecoderState`] uses whichever
//! [`coding_matrix::CodingMatrix`] its type parameter selects.
//!
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
