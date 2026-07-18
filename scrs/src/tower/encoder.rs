//! Incremental Tower Cauchy encoder.

use crate::encoder::EncodeError;
use crate::gf65536::GfElem;

use super::{TowerCauchyView, payload};

/// A GF(65536) Good-Cauchy encoder with incremental repair updates.
///
/// Symbols are interleaved two-byte field elements and therefore must have a
/// nonzero even byte length. The code supports `k + m <= 65535`.
pub struct StreamingEncoder {
    k: usize,
    m: usize,
    symbol_len: usize,
    coefficients: Vec<GfElem>,
    repairs: Vec<u8>,
    fed: Vec<bool>,
    fed_count: usize,
}

impl StreamingEncoder {
    /// Construct an encoder.
    ///
    /// Returns `None` for invalid dimensions, zero or odd symbol lengths, size
    /// overflow, or allocation failure.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let cauchy = TowerCauchyView::new(k, m)?;
        if symbol_len == 0 || symbol_len % 2 != 0 {
            return None;
        }
        let repair_len = m.checked_mul(symbol_len)?;
        let coefficients = cauchy.coefficient_matrix()?;
        let repairs = zeroed_bytes(repair_len)?;
        let mut fed = Vec::new();
        fed.try_reserve_exact(k).ok()?;
        fed.resize(k, false);
        Some(Self {
            k,
            m,
            symbol_len,
            coefficients,
            repairs,
            fed,
            fed_count: 0,
        })
    }

    /// Clear fed state and repair payloads while retaining allocations.
    pub fn reset(&mut self) {
        self.repairs.fill(0);
        self.fed.fill(false);
        self.fed_count = 0;
    }

    /// Number of data symbols.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Total codeword length.
    pub const fn n(&self) -> usize {
        self.k + self.m
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of distinct data symbols fed in the current block.
    pub const fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Add one systematic symbol's contribution to every repair symbol.
    pub fn feed_data_symbol(&mut self, index: usize, data: &[u8]) -> Result<(), EncodeError> {
        if index >= self.k {
            return Err(EncodeError::IndexOutOfRange {
                index,
                max: self.k - 1,
            });
        }
        if data.len() != self.symbol_len {
            return Err(EncodeError::WrongPayloadLen {
                expected: self.symbol_len,
                got: data.len(),
            });
        }
        if self.fed[index] {
            return Err(EncodeError::DuplicateData { index });
        }

        self.fed[index] = true;
        self.fed_count += 1;
        let coefficient_start = index * self.m;
        payload::xor_scaled_bytes_rows(
            &mut self.repairs,
            self.symbol_len,
            &self.coefficients[coefficient_start..coefficient_start + self.m],
            data,
        );
        Ok(())
    }

    /// Borrow repair symbol `index`.
    ///
    /// Before all data has been fed, the symbol contains the valid partial sum
    /// of contributions received so far.
    pub fn repair_symbol(&self, index: usize) -> Result<&[u8], EncodeError> {
        if index >= self.m {
            return Err(EncodeError::IndexOutOfRange {
                index,
                max: self.m - 1,
            });
        }
        let start = index * self.symbol_len;
        Ok(&self.repairs[start..start + self.symbol_len])
    }

    /// Consume the encoder and return one buffer per repair symbol.
    pub fn into_repairs(self) -> Vec<Vec<u8>> {
        self.repairs
            .chunks_exact(self.symbol_len)
            .map(<[u8]>::to_vec)
            .collect()
    }
}

fn zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).ok()?;
    bytes.resize(len, 0);
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_dimensions_and_even_wire_elements() {
        assert!(StreamingEncoder::new(256, 2, 64).is_some());
        assert!(StreamingEncoder::new(1, 1, 0).is_none());
        assert!(StreamingEncoder::new(1, 1, 3).is_none());
        assert!(StreamingEncoder::new(32_768, 32_768, 2).is_none());
    }

    #[test]
    fn incremental_repair_matches_direct_field_sum() {
        let k = 4;
        let m = 3;
        let symbol_len = 10;
        let data: Vec<Vec<u8>> = (0..k)
            .map(|row| {
                (0..symbol_len)
                    .map(|byte| (row * 31 + byte * 7) as u8)
                    .collect()
            })
            .collect();
        let matrix = TowerCauchyView::new(k, m).unwrap();
        let mut encoder = StreamingEncoder::new(k, m, symbol_len).unwrap();
        for (index, symbol) in data.iter().enumerate() {
            encoder.feed_data_symbol(index, symbol).unwrap();
        }
        for repair in 0..m {
            let mut expected = vec![0; symbol_len];
            for (index, symbol) in data.iter().enumerate() {
                payload::xor_scaled_bytes(&mut expected, matrix.get(index, repair), symbol);
            }
            assert_eq!(encoder.repair_symbol(repair).unwrap(), expected);
        }
    }

    #[test]
    fn reset_retains_configuration_and_clears_state() {
        let mut encoder = StreamingEncoder::new(3, 2, 8).unwrap();
        encoder.feed_data_symbol(0, &[7; 8]).unwrap();
        assert_ne!(encoder.repair_symbol(0).unwrap(), &[0; 8]);
        encoder.reset();
        assert_eq!(encoder.fed_count(), 0);
        assert_eq!(encoder.repair_symbol(0).unwrap(), &[0; 8]);
    }
}
