/// Error type for streaming encoder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// Invalid symbol index (out of range for data or repair).
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },
    /// Payload length does not match `symbol_len`.
    WrongPayloadLen {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// Data symbol has already been fed.
    DuplicateData {
        /// The duplicate index.
        index: usize,
    },
}
