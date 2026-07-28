//! Batch codec API.

mod codec;

/// Batch codec using the Good Cauchy matrix (`n = k + m <= 255`).
///
/// Use this alias when the encoder and matching
/// [`crate::decoder::LazyDecoderState`] should use
/// [`crate::good_cauchy::GoodCauchyView`].
pub type GoodCauchyBatchCodec = BatchCodec<crate::good_cauchy::GoodCauchyView>;

/// Batch codec using the standard Cauchy matrix (`n = k + m <= 256`).
///
/// Use this alias when the encoder and matching
/// [`crate::decoder::LazyDecoderState`] should use
/// [`crate::cauchy::CauchyView`].
pub type StandardCauchyBatchCodec = BatchCodec<crate::cauchy::CauchyView>;

pub use codec::{BatchCodec, DecodeScratch};
