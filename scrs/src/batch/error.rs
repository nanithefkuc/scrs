/// Error returned when constructing a [`crate::batch::BatchCodec`] with invalid parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `k` or `m` is zero.
    ZeroDimension,
    /// The selected coding matrix rejects `k + m`.
    TooManySymbols,
    /// `symbol_len` is zero.
    ZeroSymbolLen,
}

/// Error returned by [`crate::batch::BatchCodec::encode`] when the input data length is wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// Input data length differed from `k * symbol_len`.
    WrongInputLen {
        /// The expected input length.
        expected: usize,
        /// The actual input length.
        got: usize,
    },
}

/// Error returned by [`crate::batch::BatchCodec::decode`] when the inputs are malformed or
/// insufficient to recover the data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Wrong number of symbols provided: expected exactly `k`.
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
        /// The codeword length `n`.
        n: usize,
    },
    /// The same symbol index appeared more than once.
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
    /// The provided symbols are linearly dependent (rank < `k`). For a valid
    /// MDS code with distinct in-range indices this never occurs; seeing it
    /// indicates either a bug in SCRS or a corrupted codeword.
    InsufficientRank {
        /// The rank achieved during elimination.
        rank: usize,
        /// The required rank (`k`).
        k: usize,
    },
}
