//! Additive-FFT Reed-Solomon coding over GF(65536).
//!
//! The codec uses normalized subspace polynomials and the novel polynomial
//! basis to evaluate over nested additive subspaces in `O(N log N)` field
//! butterflies. Non-power-of-two message dimensions use truncated inverse
//! transforms, while repair evaluations immediately follow the `k` systematic
//! points. Configurations require `k + m <= 65536`, and two-byte GF(65536)
//! wire elements require an even symbol length.
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
mod profile;
mod transform;

pub use decoder::{DecodeScratch, LazyDecoderState};
pub use encoder::{EncodeScratch, SystematicEncoder};
pub use transform::{MAX_TRANSFORM_SIZE, TransformLengthError, TransformPlan};
