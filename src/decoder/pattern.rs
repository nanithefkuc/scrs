//! 256-bit receipt pattern key for the decoder.

/// Receipt-pattern key for the v0.2 decoder.
///
/// Bit `i` is set when codeword symbol `i` has been received. The fixed
/// 256-bit representation covers SCRS v0.1/v0.2's `k + m <= 256` domain and is
/// suitable as a cache key for future decode-recipe memoization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PatternKey([u64; 4]);

impl PatternKey {
    /// Create an empty key.
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    /// Set the bit for `idx`.
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < 256, "pattern index out of range");
        self.0[idx / 64] |= 1u64 << (idx % 64);
    }

    /// Return whether the bit for `idx` is set.
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < 256, "pattern index out of range");
        ((self.0[idx / 64] >> (idx % 64)) & 1) != 0
    }

    /// Expose the fixed-width representation.
    pub const fn words(&self) -> [u64; 4] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_key_covers_256_symbols() {
        let mut key = PatternKey::empty();
        for idx in [0, 1, 63, 64, 127, 128, 191, 192, 255] {
            key.set(idx);
        }
        for idx in 0..256 {
            let expected = matches!(idx, 0 | 1 | 63 | 64 | 127 | 128 | 191 | 192 | 255);
            assert_eq!(key.get(idx), expected, "idx {idx}");
        }
    }
}
