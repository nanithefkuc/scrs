//! Streaming Cauchy Reed-Solomon erasure coding.
//!
//! SCRS provides systematic Cauchy-RS encoding and a lazy, payload-deferred
//! streaming decoder optimized for predictable receive-path latency. The
//! decoder records symbols as they arrive and defers payload reconstruction
//! until `k` independent symbols are available.
//!
//! # Coding profiles
//!
//! The original GF(256) profile supports `k + m <= 255` with Good Cauchy, or
//! 256 with Standard Cauchy. GF(65536) provides the incremental [`tower`]
//! profile and the block-final [`afft`] profile. Both use two-byte interleaved
//! wire elements and therefore require even-length symbols.
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

pub use algebra::{gf256, gf65536};
pub use matrices::{cauchy, coding_matrix, good_cauchy};

pub mod afft;
pub mod batch;
pub mod decoder;
pub mod encoder;
pub use decoder::pattern as pattern_key;
mod payload;
#[cfg(feature = "simd")]
mod simd;
pub mod tower;
pub mod transport;
pub use transport::symbol_sink as stream;
