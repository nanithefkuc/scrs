//! Streaming Cauchy Reed-Solomon erasure coding.
//!
//! SCRS provides systematic erasure coding and a lazy, payload-deferred
//! streaming decoder optimized for predictable receive-path latency. The
//! decoder records symbols as they arrive and defers payload reconstruction
//! until `k` independent symbols are available.
//!
//! Field arithmetic comes from [`fff`] and the additive-FFT engine from
//! [`cafft`]; SCRS owns the wire format, the codec shells, and the erasure
//! recipes.
//!
//! # The systematic guarantee
//!
//! **For every engine and every geometry, transmitted symbols `0..k` are the
//! input data verbatim.** A receiver that loses nothing does no arithmetic, and
//! a receiver that loses symbol `i` reconstructs only symbol `i`. This is a
//! contract, not an implementation detail: it is what makes the decoder's
//! `finalize` cost scale with the erasure count rather than with `k`, and
//! `tests/systematic.rs` asserts it across all five engines.
//!
//! # Coding profiles
//!
//! Engines are per-field, and the two peers must agree on one: their coding
//! matrices are unrelated, so a codeword is only meaningful to the engine that
//! produced it. [`Profile::recommended`] derives a default both peers reach
//! independently from `(field, k, m)`.
//!
//! | field | engine | capacity `k + m` | encode | `symbol_len` |
//! |---|---|--:|---|---|
//! | GF(256) | [`Engine::GoodCauchy`] | 255 | incremental or block-final | any |
//! | GF(256) | [`Engine::StandardCauchy`] | 256 | block-final | any |
//! | GF(256) | [`Engine::Gf8Afft`] | 256 | block-final | any |
//! | GF(65536) | [`Engine::Tower`] | 65535 | incremental or block-final | even |
//! | GF(65536) | [`Engine::Gf16Afft`] | 65536 | block-final | even |
//!
//! The even-length requirement belongs to the *field*, not to the transform:
//! GF(65536) wire elements are two interleaved bytes. The GF(256) additive FFT
//! accepts any symbol length.
//!
//! # Features
//!
//! - `simd` (default) — runtime-dispatched vector kernels in both dependencies.
//!   Disabling leaves their portable scalar backends; correctness is unchanged.
//! - `internals` — exposes implementation APIs for benchmarking and research,
//!   exempt from compatibility guarantees. See the `internals` module.
//!
//! `FFF_BACKEND` and `CAFFT_BACKEND` override the detected SIMD backend at
//! runtime, downgrade-only; `internals::backend` reports what each layer
//! resolved to.
#![warn(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Emit one item that is `pub` only under the `internals` feature.
///
/// The body is written once; exactly one `cfg` arm survives expansion. Use this
/// for private free functions, consts, statics, and inherent methods on types
/// that the *stable* API already re-exports — making those `pub` unconditionally
/// would leak them into the supported surface, and an accessor cannot stand in
/// for a method.
///
/// Items in a module that is itself private without the feature do not need
/// this: gate the `mod` declaration and write them plain `pub`.
///
/// One item per invocation. Each arm is anchored on the item's leading keyword
/// because a bare `$($item:tt)*` is ambiguous against the attribute repetition.
macro_rules! internals_pub {
    ($(#[$attr:meta])* const fn $($rest:tt)*) => {
        #[cfg(feature = "internals")] $(#[$attr])* pub const fn $($rest)*
        #[cfg(not(feature = "internals"))] $(#[$attr])* const fn $($rest)*
    };
    ($(#[$attr:meta])* const $($rest:tt)*) => {
        #[cfg(feature = "internals")] $(#[$attr])* pub const $($rest)*
        #[cfg(not(feature = "internals"))] $(#[$attr])* const $($rest)*
    };
    ($(#[$attr:meta])* static $($rest:tt)*) => {
        #[cfg(feature = "internals")] $(#[$attr])* pub static $($rest)*
        #[cfg(not(feature = "internals"))] $(#[$attr])* static $($rest)*
    };
    ($(#[$attr:meta])* fn $($rest:tt)*) => {
        #[cfg(feature = "internals")] $(#[$attr])* pub fn $($rest)*
        #[cfg(not(feature = "internals"))] $(#[$attr])* fn $($rest)*
    };
}

pub mod matrices;

pub use fff::{gf8, gf16};
pub use matrices::{cauchy, coding_matrix, good_cauchy};

pub mod afft;
pub mod batch;
pub mod decoder;
pub mod encoder;
pub use decoder::pattern as pattern_key;
#[cfg(feature = "internals")]
pub mod payload;
#[cfg(not(feature = "internals"))]
mod payload;
pub mod tower;
pub mod transport;
pub use transport::symbol_sink as stream;

pub mod codec;
pub mod error;
#[cfg(feature = "internals")]
pub mod internals;
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

/// GF(256) coding engine selector.
///
/// The three GF(256) profiles have **incompatible** parity: [`Gf8Engine::GoodCauchy`],
/// [`Gf8Engine::StandardCauchy`], and the block-final [`Gf8Engine::Afft`]. A codec
/// fixes the engine at construction, and a sender and receiver MUST use the same one.
/// [`recommended_gf8_engine`] gives a geometry-based default both peers can compute
/// independently from `(k, m)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gf8Engine {
    /// Good Cauchy: `k + m <= 255`, and the only GF(256) engine with an
    /// incremental encoder.
    GoodCauchy,
    /// Standard Cauchy: the full `k + m <= 256`, block-final.
    StandardCauchy,
    /// Block-final additive FFT ([`afft`]): `O(n log n)` transform cost instead of
    /// Cauchy's `O(r * k)` reconstruction.
    Afft,
}

/// Recommend a GF(256) engine for a `(k, m)` block geometry.
///
/// Returns [`Gf8Engine::StandardCauchy`] when the geometry needs the 256th codeword
/// position that Good Cauchy cannot address, and [`Gf8Engine::GoodCauchy`] otherwise
/// — which also keeps incremental encoding available.
///
/// **[`Gf8Engine::Afft`] is never recommended, deliberately.** It is a legitimate
/// engine and fully supported, but it is an opt-in one. Measured on GF(256) at
/// `symbol_len = 1400` (`benches/engines.rs`), the additive FFT is the better
/// *encoder* from `k >= 16` — 1.4x at `k = 32`, 2.8x at `k = 160` — but a worse
/// *decoder* at every erasure count except near-total redundancy consumption:
///
/// | geometry | `r = 1` | `r = 4` | `r = m/2` | `r = m` |
/// |---|--:|--:|--:|--:|
/// | `k=64, m=32` | 1.71x | 1.69x | 2.05x | **0.65x** |
/// | `k=160, m=80` | 1.54x | 1.74x | 2.34x | **0.24x** |
///
/// (Ratios are AFFT / Good Cauchy; below 1.0 means the AFFT wins.) Cauchy's reduced
/// reconstruction is `O(r * k)`, so it degrades with the erasure count, while the
/// transform pays a fixed `O(n log n)` whatever happens. The erasure count is
/// unknown at encode time and both peers must derive the same engine from `(k, m)`
/// alone, so a geometry-only rule cannot exploit that crossover — and SCRS optimises
/// the receive path, where losing 1.5-2x at the common small-`r` case to win at
/// `r = m` is the wrong trade by default.
///
/// Select [`Engine::Gf8Afft`] explicitly when the workload is encode-bound, or when
/// the loss profile genuinely consumes most of the redundancy.
#[must_use]
pub fn recommended_gf8_engine(k: usize, m: usize) -> Gf8Engine {
    if k.saturating_add(m) > 255 {
        Gf8Engine::StandardCauchy
    } else {
        Gf8Engine::GoodCauchy
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
