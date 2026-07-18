//! Additive-FFT Reed-Solomon coding over GF(65536).
//!
//! The codec uses normalized subspace polynomials and the novel polynomial
//! basis to evaluate over nested additive subspaces in `O(N log N)` field
//! butterflies. Non-power-of-two message dimensions are shortened through
//! fixed zero evaluations. Consequently, configurations require
//! `k.next_power_of_two() + m <= 65536`, and two-byte GF(65536) wire elements
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
mod profile;
mod transform;

pub use decoder::LazyDecoderState;
pub use encoder::{EncodeError, SystematicEncoder};
pub use transform::{MAX_TRANSFORM_SIZE, TransformLengthError, TransformPlan};
