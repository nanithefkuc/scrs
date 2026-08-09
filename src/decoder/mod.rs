//! Streaming decoder API.

#[cfg(feature = "internals")]
pub mod cache;
#[cfg(not(feature = "internals"))]
mod cache;
pub mod pattern;
#[cfg(feature = "internals")]
pub mod recipe;
#[cfg(not(feature = "internals"))]
pub(crate) mod recipe;
#[cfg(feature = "internals")]
pub mod streaming;
#[cfg(not(feature = "internals"))]
mod streaming;

pub use cache::RecipeCache;
pub use streaming::LazyDecoderState;
