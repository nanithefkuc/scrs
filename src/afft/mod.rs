//! Additive-FFT Reed-Solomon coding, generic over the binary field.
//!
//! The transform engine is [`butterfly_fft`]: evaluation over nested additive subspaces
//! in the novel polynomial basis, `O(N log N)` field butterflies, truncated
//! inverse transforms for non-power-of-two message dimensions, and repair
//! evaluations immediately following the `k` systematic points. SRS supplies
//! the receipt bookkeeping, the erasure policy, and the codec API.
//!
//! Two fields are supported, and they differ only in their limits:
//!
//! | field | domain | `k + m` | `symbol_len` |
//! |---|--:|--:|---|
//! | [`fgf::Gf8`] | 256 points | `<= 256` | any |
//! | [`fgf::Gf16`] | 65536 points | `<= 65536` | even |
//!
//! The symbol-length rule is not an AFFT property but an element-width one:
//! a symbol holds whole field elements, and GF(2^8) elements are one byte.
//!
//! Encoding is block-final rather than incremental. Decoder receipt handling
//! remains payload-lazy; transform-domain reconstruction starts only after `k`
//! distinct transmitted symbols arrive.
//!
//! ```
//! use srs::afft::{Gf16Decoder, Gf16Encoder};
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let encoder = Gf16Encoder::new(3, 2, 2).unwrap();
//! let repairs = encoder.encode(&data).unwrap();
//!
//! let mut decoder = Gf16Decoder::new(3, 2, 2).unwrap();
//! decoder.push_symbol(1, &data[2..4]).unwrap();
//! decoder.push_symbol(2, &data[4..6]).unwrap();
//! decoder.push_symbol(3, &repairs[0]).unwrap();
//! assert_eq!(decoder.finalize_ref().unwrap(), data);
//! ```
//!
//! GF(2^8) is the same API with a different field, and accepts odd symbols:
//!
//! ```
//! use srs::afft::{Gf8Decoder, Gf8Encoder};
//!
//! let data = vec![1, 2, 3, 4, 5];
//! let encoder = Gf8Encoder::new(5, 3, 1).unwrap();
//! let repairs = encoder.encode(&data).unwrap();
//!
//! let mut decoder = Gf8Decoder::new(5, 3, 1).unwrap();
//! for index in [0, 1, 2, 3] {
//!     decoder.push_symbol(index, &data[index..index + 1]).unwrap();
//! }
//! decoder.push_symbol(5, &repairs[0]).unwrap();
//! assert_eq!(decoder.finalize_ref().unwrap(), data);
//! ```

#[cfg(feature = "internals")]
pub mod batch;
#[cfg(not(feature = "internals"))]
mod batch;
pub mod crossover;
#[cfg(feature = "internals")]
pub mod decoder;
#[cfg(not(feature = "internals"))]
mod decoder;
#[cfg(test)]
mod differential;
#[cfg(feature = "internals")]
pub mod encoder;
#[cfg(not(feature = "internals"))]
mod encoder;
mod generator;
#[cfg(feature = "internals")]
pub mod locator;
#[cfg(not(feature = "internals"))]
mod locator;
#[cfg(feature = "internals")]
pub mod profile;
#[cfg(not(feature = "internals"))]
pub(crate) mod profile;
#[cfg(feature = "internals")]
pub mod recovery;
#[cfg(not(feature = "internals"))]
mod recovery;
#[cfg(feature = "internals")]
pub mod strip;
#[cfg(not(feature = "internals"))]
mod strip;
mod tables;
mod targeted;

pub use batch::{BatchDecodeScratch, BatchDecoder, DecodePlan, Gf8BatchDecoder, Gf16BatchDecoder};
pub use crossover::RecoveryPath;
pub use decoder::{DecodeScratch, LazyDecoderState};
pub use encoder::{EncodeScratch, SystematicEncoder};

pub use butterfly_fft::error::TransformLengthError;

/// Fields this engine can code over.
///
/// Sealed by the internal locator-table trait, which is implemented only for
/// GF(2^8) and GF(2^16).
pub trait Field: tables::RsField<Elem: Send + Sync> {
    /// Largest evaluation domain, and so the largest `k + m`.
    ///
    /// Fixed by the field: a domain point is a distinct field element.
    const MAX_TRANSFORM_SIZE: usize;
}

impl Field for fgf::Gf8 {
    const MAX_TRANSFORM_SIZE: usize = 1 << 8;
}

impl Field for fgf::Gf16 {
    const MAX_TRANSFORM_SIZE: usize = 1 << 16;
}

/// Reusable additive-FFT plan for one power-of-two evaluation domain.
pub type TransformPlan<F> = butterfly_fft::core::transform::TransformPlan<F>;

/// GF(2^8) block-final additive-FFT encoder. `k + m <= 256`, any `symbol_len`.
pub type Gf8Encoder = SystematicEncoder<fgf::Gf8>;
/// GF(2^8) payload-lazy additive-FFT decoder.
pub type Gf8Decoder = LazyDecoderState<fgf::Gf8>;
/// GF(2^16) block-final additive-FFT encoder. `k + m <= 65536`, even `symbol_len`.
pub type Gf16Encoder = SystematicEncoder<fgf::Gf16>;
/// GF(2^16) payload-lazy additive-FFT decoder.
pub type Gf16Decoder = LazyDecoderState<fgf::Gf16>;
