//! Unified error types for the SCRS v2 API.
//!
//! Every codec — GF(256) Cauchy, GF(65536) tower, GF(65536) additive-FFT —
//! reports failures through these three enums:
//!
//! - [`ConfigError`] for construction (dimensions / symbol length),
//! - [`EncodeError`] for encode-time input faults,
//! - [`DecodeError`] for streaming-decode and batch-decode faults.

/// Error returned when constructing a codec with invalid parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `k` or `m` is zero.
    ZeroDimension,
    /// The selected engine rejects `k + m` (exceeds its capacity `cap`).
    TooManySymbols {
        /// The engine's maximum `k + m`.
        cap: usize,
    },
    /// `symbol_len` is zero.
    ZeroSymbolLen,
    /// `symbol_len` is odd but the engine requires even-length symbols
    /// (GF(65536) interleaved two-byte elements).
    OddSymbolLen,
}

/// Error returned by encode operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// Input data length differed from `k * symbol_len` (batch encode).
    WrongInputLen {
        /// The expected input length.
        expected: usize,
        /// The actual input length.
        got: usize,
    },
    /// A symbol index is outside the valid range `0..n`.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The codeword length `n = k + m`.
        n: usize,
    },
    /// A payload has the wrong length.
    WrongPayloadLen {
        /// The expected length (`symbol_len`).
        expected: usize,
        /// The actual length.
        got: usize,
    },
    /// A data symbol has already been fed (incremental encode).
    DuplicateData {
        /// The duplicate index.
        index: usize,
    },
    /// A caller-provided repair buffer has the wrong length.
    WrongOutputLen {
        /// The expected length.
        expected: usize,
        /// The actual length.
        got: usize,
    },
}

/// Error returned by streaming-decode and batch-decode operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Wrong number of symbols provided: expected exactly `k` (batch decode).
    WrongCount {
        /// The expected count (`k`).
        expected: usize,
        /// The actual count received.
        got: usize,
    },
    /// A symbol index is outside the valid range `0..n`.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The codeword length `n = k + m`.
        n: usize,
    },
    /// The same symbol index appeared more than once (batch decode).
    DuplicateIndex {
        /// The duplicated index.
        index: usize,
    },
    /// A payload has the wrong length.
    WrongPayloadLen {
        /// The expected length (`symbol_len`).
        expected: usize,
        /// The actual length.
        got: usize,
    },
    /// More than `n = k + m` symbols were pushed; the decoder refuses further
    /// symbols to bound adversarial cost.
    TooManySymbols {
        /// The cap (`n = k + m`).
        cap: usize,
        /// Symbols already received.
        received: usize,
    },
    /// Finalization was attempted before the decoder reached full rank `k`.
    InsufficientRank {
        /// The current rank.
        rank: usize,
        /// The required rank (`k`).
        k: usize,
    },
    /// Caller-provided output buffer has the wrong length (`k * symbol_len`).
    WrongOutputLen {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
}
