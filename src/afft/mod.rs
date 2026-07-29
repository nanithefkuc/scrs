//! Additive-FFT Reed-Solomon coding over GF(65536).
//!
//! The transform engine is [`cafft`]: evaluation over nested additive subspaces
//! in the novel polynomial basis, `O(N log N)` field butterflies, truncated
//! inverse transforms for non-power-of-two message dimensions, and repair
//! evaluations immediately following the `k` systematic points. SCRS supplies
//! the receipt bookkeeping, the erasure policy, and the codec API.
//!
//! Configurations require `k + m <= 65536`, and two-byte GF(65536) wire elements
//! require an even symbol length.
//!
//! Encoding is block-final rather than incremental. Decoder receipt handling
//! remains payload-lazy; transform-domain reconstruction starts only after `k`
//! distinct transmitted symbols arrive.
//!
//! ```
//! use scrs::afft::{LazyDecoderState, SystematicEncoder};
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let encoder = SystematicEncoder::new(3, 2, 2).unwrap();
//! let repairs = encoder.encode(&data).unwrap();
//!
//! let mut decoder = LazyDecoderState::new(3, 2, 2).unwrap();
//! decoder.push_symbol(1, &data[2..4]).unwrap();
//! decoder.push_symbol(2, &data[4..6]).unwrap();
//! decoder.push_symbol(3, &repairs[0]).unwrap();
//! assert_eq!(decoder.finalize_ref().unwrap(), data);
//! ```

mod decoder;
#[cfg(test)]
mod differential;
mod encoder;
pub(crate) mod profile;

pub use decoder::{DecodeScratch, LazyDecoderState};
pub use encoder::{EncodeScratch, SystematicEncoder};

/// The field this engine codes over.
pub type Field = fff::Gf16;

/// Reusable additive-FFT plan for one power-of-two evaluation domain.
pub type TransformPlan = cafft::core::transform::TransformPlan<Field>;

pub use cafft::error::TransformLengthError;

/// Largest evaluation domain, fixed by the field: GF(65536) has 65536 points.
pub const MAX_TRANSFORM_SIZE: usize = 1 << 16;
