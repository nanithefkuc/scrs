//! Incremental encoder API.

#[cfg(feature = "internals")]
pub mod streaming;
#[cfg(not(feature = "internals"))]
mod streaming;

pub use streaming::StreamingEncoder;
