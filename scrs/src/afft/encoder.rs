//! Systematic additive-FFT encoder.

use super::profile::{Profile, zeroed_bytes};

/// Error returned when additive-FFT encoding receives the wrong data length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeError {
    /// Required length, `k * symbol_len`.
    pub expected: usize,
    /// Supplied length.
    pub got: usize,
}

/// Block-systematic Reed-Solomon encoder using the additive FFT.
///
/// Non-power-of-two `k` is shortened to `padded_k = k.next_power_of_two()` by
/// assigning zero to the omitted systematic evaluation points. Repair symbols
/// occupy evaluation points `padded_k..padded_k + m`; the omitted points are not
/// transmitted. Construction therefore requires `padded_k + m <= 65536`.
#[derive(Clone, Debug)]
pub struct SystematicEncoder {
    profile: Profile,
}

impl SystematicEncoder {
    /// Construct an encoder.
    ///
    /// Returns `None` for zero dimensions, zero or odd symbol lengths, size
    /// overflow, or when `k.next_power_of_two() + m > 65536`.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        Some(Self {
            profile: Profile::new(k, m, symbol_len)?,
        })
    }

    /// Number of systematic symbols.
    pub const fn k(&self) -> usize {
        self.profile.k
    }

    /// Number of repair symbols.
    pub const fn m(&self) -> usize {
        self.profile.m
    }

    /// Number of transmitted symbols, `k + m`.
    pub const fn n(&self) -> usize {
        self.profile.n
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Power-of-two systematic transform size after shortening.
    pub const fn padded_k(&self) -> usize {
        self.profile.padded_k
    }

    /// Power-of-two full evaluation transform size.
    pub const fn transform_size(&self) -> usize {
        self.profile.transform_size
    }

    /// Encode a flat `k * symbol_len` systematic block and return `m` repairs.
    ///
    /// The input bytes are the systematic symbols on the wire and are never
    /// modified or reserialized.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncodeError> {
        let repair_len = self.profile.m * self.profile.symbol_len;
        let mut repairs =
            zeroed_bytes(repair_len).expect("profile validated repair allocation size");
        self.encode_into(data, &mut repairs)?;
        Ok(repairs
            .chunks_exact(self.profile.symbol_len)
            .map(<[u8]>::to_vec)
            .collect())
    }

    /// Encode repairs into a caller-provided flat `m * symbol_len` buffer.
    pub fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        let expected_data = self.profile.k * self.profile.symbol_len;
        if data.len() != expected_data {
            return Err(EncodeError {
                expected: expected_data,
                got: data.len(),
            });
        }
        let expected_repairs = self.profile.m * self.profile.symbol_len;
        if repairs.len() != expected_repairs {
            return Err(EncodeError {
                expected: expected_repairs,
                got: repairs.len(),
            });
        }

        let workspace_len = self.profile.transform_size * self.profile.symbol_len;
        let mut workspace = zeroed_bytes(workspace_len)
            .expect("profile validated transform workspace allocation size");
        workspace[..data.len()].copy_from_slice(data);

        let interpolation_bytes = self.profile.padded_k * self.profile.symbol_len;
        self.profile.interpolation_plan.inverse_bytes(
            &mut workspace[..interpolation_bytes],
            self.profile.symbol_len,
        );
        self.profile
            .transform_plan
            .forward_bytes(&mut workspace, self.profile.symbol_len);

        let repair_start = self.profile.padded_k * self.profile.symbol_len;
        repairs.copy_from_slice(&workspace[repair_start..repair_start + expected_repairs]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf65536::GfElem;

    #[test]
    fn validates_shortened_transform_capacity() {
        assert!(SystematicEncoder::new(257, 128, 8).is_some());
        assert!(SystematicEncoder::new(32_769, 1, 2).is_none());
        assert!(SystematicEncoder::new(1, 1, 3).is_none());
        let encoder = SystematicEncoder::new(5, 3, 2).unwrap();
        assert_eq!(encoder.padded_k(), 8);
        assert_eq!(encoder.transform_size(), 16);
    }

    #[test]
    fn encoding_is_systematic_and_repairs_match_scalar_transform() {
        let k = 5;
        let m = 3;
        let symbol_len = 6;
        let data: Vec<_> = (0..k * symbol_len)
            .map(|index| (index * 29 ^ 0xa5) as u8)
            .collect();
        let encoder = SystematicEncoder::new(k, m, symbol_len).unwrap();
        let repairs = encoder.encode(&data).unwrap();

        for element_offset in (0..symbol_len).step_by(2) {
            let mut values = vec![GfElem::ZERO; encoder.transform_size()];
            for data_index in 0..k {
                let start = data_index * symbol_len + element_offset;
                values[data_index] = GfElem::from_bytes([data[start], data[start + 1]]);
            }
            encoder
                .profile
                .interpolation_plan
                .inverse(&mut values[..encoder.padded_k()])
                .unwrap();
            encoder.profile.transform_plan.forward(&mut values).unwrap();
            for repair in 0..m {
                let expected = values[encoder.padded_k() + repair].to_bytes();
                assert_eq!(
                    &repairs[repair][element_offset..element_offset + 2],
                    &expected
                );
            }
        }
    }
}
