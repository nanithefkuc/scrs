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

pub mod codec;
pub mod error;
pub use codec::{
    BatchDecoder, BatchEncoder, Coded, Decoder, Engine, Field, IncrementalEncoder, Profile,
};
pub use error::{ConfigError, DecodeError, EncodeError};
pub mod selector;
pub use selector::{
    AnyBatchDecodeScratch, AnyBatchDecoder, AnyBatchEncoder, AnyDecodeScratch, AnyDecoder,
    AnyEncodeScratch, AnyIncrementalEncoder, batch_decoder, batch_encoder, decoder,
    incremental_encoder,
};

/// GF(65536) coding engine selector.
///
/// The two GF(65536) profiles have **incompatible** parity: the incremental
/// [`tower`] profile and the block-final [`afft`] profile. A codec fixes the
/// engine at construction, and a sender and receiver MUST use the same one.
/// [`recommended_gf16_engine`] gives a geometry-based default both peers can
/// compute independently from `(k, m)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gf16Engine {
    /// Incremental Tower Cauchy ([`tower`]): cheap per-source repair updates and
    /// reduced `r x r` reconstruction. Best for small blocks and low erasure
    /// counts.
    Tower,
    /// Block-final additive FFT ([`afft`]): `O(n log n)` transform cost that
    /// scales to large blocks and high erasure counts.
    Afft,
}

/// Recommend a GF(65536) engine for a `(k, m)` block geometry.
///
/// Returns [`Gf16Engine::Afft`] for large blocks (`k + m > 256`) or
/// high-redundancy codes (`m >= k / 3`), where the decode-time erasure count can
/// be high enough that reduced Tower Cauchy reconstruction (`O(r * k)`) becomes
/// expensive; otherwise [`Gf16Engine::Tower`].
///
/// The recommendation depends only on block geometry — not the actual erasure
/// count, which is unknown at encode time — so both peers derive the same engine
/// from `(k, m)`. Callers that know their loss profile may override it by
/// constructing an engine directly.
#[must_use]
pub fn recommended_gf16_engine(k: usize, m: usize) -> Gf16Engine {
    let large_block = k.saturating_add(m) > 256;
    let high_redundancy = m.saturating_mul(3) >= k;
    if large_block || high_redundancy {
        Gf16Engine::Afft
    } else {
        Gf16Engine::Tower
    }
}

#[cfg(test)]
mod engine_selection_tests {
    use super::{Gf16Engine, recommended_gf16_engine};

    #[test]
    fn small_low_redundancy_blocks_pick_tower() {
        // 25% redundancy under the 1/3 threshold, small block -> tower.
        assert_eq!(recommended_gf16_engine(32, 8), Gf16Engine::Tower);
        assert_eq!(recommended_gf16_engine(200, 40), Gf16Engine::Tower);
        assert_eq!(recommended_gf16_engine(4, 1), Gf16Engine::Tower);
    }

    #[test]
    fn large_blocks_pick_afft() {
        // k + m > 256 regardless of redundancy.
        assert_eq!(recommended_gf16_engine(256, 1), Gf16Engine::Afft);
        assert_eq!(recommended_gf16_engine(1024, 8), Gf16Engine::Afft);
    }

    #[test]
    fn high_redundancy_blocks_pick_afft() {
        // m >= k / 3 (>= ~33% redundancy) even for small blocks.
        assert_eq!(recommended_gf16_engine(9, 3), Gf16Engine::Afft);
        assert_eq!(recommended_gf16_engine(10, 4), Gf16Engine::Afft);
        assert_eq!(recommended_gf16_engine(6, 2), Gf16Engine::Afft);
    }

    #[test]
    fn threshold_boundaries() {
        // n == 256 is not "large"; 257 is.
        assert_eq!(recommended_gf16_engine(255, 1), Gf16Engine::Tower);
        assert_eq!(recommended_gf16_engine(255, 2), Gf16Engine::Afft);
        // 3m == k is the redundancy boundary (inclusive).
        assert_eq!(recommended_gf16_engine(12, 4), Gf16Engine::Afft);
        assert_eq!(recommended_gf16_engine(13, 4), Gf16Engine::Tower);
    }
}
