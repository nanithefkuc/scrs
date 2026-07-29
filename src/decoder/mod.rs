//! Streaming decoder API.

#[cfg(feature = "internals")]
pub mod cache;
#[cfg(not(feature = "internals"))]
mod cache;
#[cfg(feature = "internals")]
pub mod cauchy_inverse;
#[cfg(not(feature = "internals"))]
pub(crate) mod cauchy_inverse;
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
pub use cauchy_inverse::cauchy_inverse_closed_form;
pub use streaming::LazyDecoderState;
