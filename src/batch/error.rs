/// Error returned when constructing a [`BatchCodec`] with invalid parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `k` or `m` is zero.
    ZeroDimension,
    /// `k + m` exceeds 256, the v0.1 cap (see [crate::cauchy]).
    TooManySymbols,
    /// `symbol_len` is zero.
    ZeroSymbolLen,
}

/// Error returned by [`BatchCodec::decode`] when the inputs are malformed or
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
