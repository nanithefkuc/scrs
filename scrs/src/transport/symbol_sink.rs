//! Streaming I/O traits for symbol-level transport.
//!
//! The decoder implements [`SymbolSink`] and consumes symbols one at a time.
//! This keeps the decoder pure and testable: it knows nothing about where
//! symbols come from or how they are framed.

use crate::error::DecodeError;

/// Outcome of pushing one symbol into a [`SymbolSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// The symbol advanced the decoder's rank but more independent symbols
    /// are still needed.
    Advanced {
        /// The new rank after processing this symbol.
        rank: usize,
        /// Total symbols received so far (including this one and any rejected).
        received: usize,
    },
    /// The symbol brought the decoder to full rank (`k`). The original data
    /// is now recoverable via [`SymbolSink::finalize`].
    Complete,
    /// The symbol was linearly dependent on prior symbols and was rejected.
    /// The decoder's state is unchanged.
    Dependent,
}

// Streaming decode failures use the crate-wide [`crate::error::DecodeError`].

/// A sink that consumes coded symbols one at a time and tracks decode state.
///
/// Implemented by [`crate::decoder::LazyDecoderState`]. The decoder computes
/// Cauchy coefficients internally from `idx`, so the caller only needs to
/// know which codeword position each symbol belongs to and provide its
/// payload bytes.
pub trait SymbolSink {
    /// Push one symbol: `idx` identifies its position in the codeword
    /// (`0..n`), `payload` is its `symbol_len` bytes.
    ///
    /// Returns the outcome of processing this symbol, or an error if the
    /// inputs are malformed or the decoder has exceeded its symbol cap.
    fn push(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, DecodeError>;

    /// Returns `true` once `k` independent symbols have been received and the
    /// original data is recoverable via [`finalize`](SymbolSink::finalize).
    ///
    /// This is an O(1) field read — no computation is performed.
    fn is_complete(&self) -> bool;

    /// Consume the decoder and return the recovered `k * symbol_len` bytes.
    ///
    /// Returns an error if [`is_complete`](SymbolSink::is_complete) is false.
    fn finalize(self) -> Result<Vec<u8>, DecodeError>;
}
