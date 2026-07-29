//! Tower Cauchy coding over GF((2^8)^2).
//!
//! This profile preserves systematic streaming encode and payload-lazy reduced
//! decode while extending the codeword capacity to 65535 symbols. Symbols use
//! the stable interleaved `[a, b]` representation from [`crate::gf16`] and
//! must have even byte length.
//!
//! ```
//! use scrs::tower::{LazyDecoderState, StreamingEncoder};
//!
//! let data = [vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
//! let mut encoder = StreamingEncoder::new(2, 1, 4).unwrap();
//! for (index, symbol) in data.iter().enumerate() {
//!     encoder.feed_data_symbol(index, symbol).unwrap();
//! }
//! let repair = encoder.repair_symbol(0).unwrap().to_vec();
//!
//! let mut decoder = LazyDecoderState::new(2, 1, 4).unwrap();
//! decoder.push_symbol(1, &data[1]).unwrap();
//! decoder.push_symbol(2, &repair).unwrap();
//! assert_eq!(decoder.finalize_ref().unwrap(), data.concat());
//! ```

#[cfg(feature = "internals")]
pub mod cauchy;
#[cfg(not(feature = "internals"))]
pub(crate) mod cauchy;
#[cfg(feature = "internals")]
pub mod decoder;
#[cfg(not(feature = "internals"))]
mod decoder;
#[cfg(feature = "internals")]
pub mod encoder;
#[cfg(not(feature = "internals"))]
mod encoder;

pub use cauchy::{MAX_SYMBOLS, TowerCauchyView};
pub use decoder::{DecodeScratch, LazyDecoderState};
pub use encoder::StreamingEncoder;
