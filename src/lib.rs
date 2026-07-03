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

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
pub mod batch;
pub mod cauchy;
pub mod coding_matrix;
#[cfg(feature = "std")]
pub mod decoder;
#[cfg(feature = "std")]
pub mod encoder;
pub mod gf256;
pub mod good_cauchy;
pub mod matrix;
pub mod pattern_key;
#[cfg(feature = "std")]
pub mod stream;
