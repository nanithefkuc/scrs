#![cfg_attr(feature = "internals", allow(missing_docs))]

use fff::field::Field as _;

use super::{Field, MAX_TRANSFORM_SIZE};

/// Validated geometry for one additive-FFT configuration.
///
/// Geometry only: transform plans live in the encoder and decoder that need
/// them, so constructing a profile costs no shared-plan lookups.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub k: usize,
    pub m: usize,
    pub n: usize,
    pub symbol_len: usize,
    pub padded_k: usize,
    pub transform_size: usize,
}

impl Profile {
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let n = k.checked_add(m)?;
        if k == 0 || m == 0 || symbol_len == 0 || symbol_len % Field::BYTES != 0 {
            return None;
        }
        let padded_k = k.checked_next_power_of_two()?;
        if n > MAX_TRANSFORM_SIZE {
            return None;
        }
        let transform_size = n.checked_next_power_of_two()?;
        transform_size.checked_mul(symbol_len)?;
        n.checked_mul(symbol_len)?;
        Some(Self {
            k,
            m,
            n,
            symbol_len,
            padded_k,
            transform_size,
        })
    }

    #[inline]
    pub fn evaluation_index(&self, wire_index: usize) -> usize {
        debug_assert!(wire_index < self.n);
        wire_index
    }
}

pub fn zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).ok()?;
    bytes.resize(len, 0);
    Some(bytes)
}
