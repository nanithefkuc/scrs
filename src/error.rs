//! Unified error types for the SRS public API.
//!
//! Every codec — GF(256) Cauchy, GF(65536) tower, GF(65536) additive-FFT —
//! reports failures through these three enums:
//!
//! - [`ConfigError`] for construction (dimensions / symbol length),
//! - [`EncodeError`] for encode-time input faults,
//! - [`DecodeError`] for streaming-decode and batch-decode faults.

use core::fmt;

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
    /// The engine does not support the requested mode (e.g. an incremental
    /// encoder was requested for a block-final engine, or vice versa).
    UnsupportedMode {
        /// The engine that was asked for the mode.
        engine: crate::codec::Engine,
    },
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
    /// Caller-owned scratch belongs to another engine or geometry.
    ScratchMismatch,
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
    /// A symbol index was not part of a prepared decoder's receipt pattern.
    UnexpectedIndex {
        /// The unexpected index.
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
    /// Caller-owned scratch belongs to another engine or geometry.
    ScratchMismatch,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("k and m must both be nonzero"),
            Self::TooManySymbols { cap } => write!(
                formatter,
                "codeword exceeds the selected engine's {cap}-symbol capacity"
            ),
            Self::ZeroSymbolLen => formatter.write_str("symbol length must be nonzero"),
            Self::OddSymbolLen => formatter.write_str("symbol length must be even for GF(65536)"),
            Self::UnsupportedMode { engine } => {
                write!(
                    formatter,
                    "{engine:?} does not support the requested codec mode"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongInputLen { expected, got } => {
                write!(
                    formatter,
                    "wrong input length: expected {expected} bytes, got {got}"
                )
            }
            Self::IndexOutOfRange { index, n } => {
                write!(
                    formatter,
                    "symbol index {index} is out of range for codeword length {n}"
                )
            }
            Self::WrongPayloadLen { expected, got } => {
                write!(
                    formatter,
                    "wrong payload length: expected {expected} bytes, got {got}"
                )
            }
            Self::DuplicateData { index } => {
                write!(formatter, "data symbol {index} was already supplied")
            }
            Self::WrongOutputLen { expected, got } => {
                write!(
                    formatter,
                    "wrong output length: expected {expected} bytes, got {got}"
                )
            }
            Self::ScratchMismatch => {
                formatter.write_str("scratch buffer belongs to a different engine or geometry")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongCount { expected, got } => {
                write!(
                    formatter,
                    "wrong symbol count: expected {expected}, got {got}"
                )
            }
            Self::IndexOutOfRange { index, n } => {
                write!(
                    formatter,
                    "symbol index {index} is out of range for codeword length {n}"
                )
            }
            Self::DuplicateIndex { index } => {
                write!(
                    formatter,
                    "symbol index {index} was supplied more than once"
                )
            }
            Self::UnexpectedIndex { index } => {
                write!(
                    formatter,
                    "symbol index {index} is not in the prepared receipt pattern"
                )
            }
            Self::WrongPayloadLen { expected, got } => {
                write!(
                    formatter,
                    "wrong payload length: expected {expected} bytes, got {got}"
                )
            }
            Self::TooManySymbols { cap, received } => write!(
                formatter,
                "decoder symbol limit exceeded: received {received}, capacity {cap}"
            ),
            Self::InsufficientRank { rank, k } => {
                write!(
                    formatter,
                    "insufficient decoder rank: have {rank}, need {k}"
                )
            }
            Self::WrongOutputLen { expected, got } => {
                write!(
                    formatter,
                    "wrong output length: expected {expected} bytes, got {got}"
                )
            }
            Self::ScratchMismatch => {
                formatter.write_str("scratch buffer belongs to a different engine or geometry")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::afft::TransformLengthError;

    fn assert_standard_error<T: std::error::Error + Send + Sync + 'static>() {}

    #[test]
    fn all_public_errors_implement_standard_error() {
        assert_standard_error::<ConfigError>();
        assert_standard_error::<EncodeError>();
        assert_standard_error::<DecodeError>();
        assert_standard_error::<TransformLengthError>();
    }

    #[test]
    fn display_reports_actionable_context() {
        assert_eq!(
            ConfigError::TooManySymbols { cap: 255 }.to_string(),
            "codeword exceeds the selected engine's 255-symbol capacity"
        );
        assert_eq!(
            EncodeError::WrongPayloadLen {
                expected: 1400,
                got: 1399
            }
            .to_string(),
            "wrong payload length: expected 1400 bytes, got 1399"
        );
        assert_eq!(
            DecodeError::InsufficientRank { rank: 7, k: 10 }.to_string(),
            "insufficient decoder rank: have 7, need 10"
        );
        // `TransformLengthError` belongs to butterfly-fft; its wording is not SRS's
        // contract, so only the trait bounds above are asserted for it.
    }
}
