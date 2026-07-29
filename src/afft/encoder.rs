//! Systematic additive-FFT encoder.

use cafft::rs::StripEncoder;

use super::Field;
use super::profile::{Profile, zeroed_bytes};
use crate::codec::{BatchEncoder, Coded};
use crate::error::{ConfigError, EncodeError};

/// Reusable scratch for allocation-free additive-FFT encoding.
///
/// Construct one with [`SystematicEncoder::encode_scratch`] and pass it to
/// [`SystematicEncoder::encode_into_with`] to run steady-state encoding without
/// heap allocation, as required by Aeron-style ring-buffer producers.
#[derive(Debug, Default)]
pub struct EncodeScratch {
    inner: cafft::rs::EncodeScratch,
}

impl EncodeScratch {
    /// Create empty scratch that grows on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Block-systematic Reed-Solomon encoder using the additive FFT.
///
/// Non-power-of-two `k` uses a truncated inverse transform over the first
/// `k.next_power_of_two()` points. Repair symbols occupy evaluation points
/// `k..k + m`, and construction therefore requires `k + m <= 65536`.
///
/// Strip blocking, the fused high-coset fast path for power-of-two `k` with
/// `m <= k`, and the transforms themselves all live in [`cafft::rs::StripEncoder`].
#[derive(Debug)]
pub struct SystematicEncoder<F: Field> {
    profile: Profile<F>,
    inner: StripEncoder<F>,
}

impl<F: Field> SystematicEncoder<F> {
    /// Construct an encoder.
    ///
    /// Fails with [`ConfigError::ZeroDimension`] for zero `k`/`m`,
    /// [`ConfigError::ZeroSymbolLen`]/[`ConfigError::OddSymbolLen`] for a zero
    /// or odd symbol length, and [`ConfigError::TooManySymbols`] when the
    /// evaluation domain would exceed `65536` points (`k + m > 65536`).
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        if symbol_len % F::BYTES != 0 {
            return Err(ConfigError::OddSymbolLen);
        }
        let cap = F::MAX_TRANSFORM_SIZE;
        let profile = Profile::new(k, m, symbol_len).ok_or(ConfigError::TooManySymbols { cap })?;
        let inner =
            StripEncoder::new(k, m, symbol_len).map_err(|_| ConfigError::TooManySymbols { cap })?;
        Ok(Self { profile, inner })
    }

    /// Number of systematic symbols.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.profile.k
    }

    /// Number of repair symbols.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.profile.m
    }

    /// Number of transmitted symbols, `k + m`.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.profile.n
    }

    /// Per-symbol byte length.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Power-of-two plan size used for truncated systematic interpolation.
    #[must_use]
    pub const fn padded_k(&self) -> usize {
        self.profile.padded_k
    }

    /// Power-of-two full evaluation transform size.
    #[must_use]
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

    /// Allocate scratch sized for one symbol-column strip of this encoder.
    ///
    /// The returned [`EncodeScratch`] can be reused across any number of
    /// [`encode_into_with`](Self::encode_into_with) calls with no further
    /// allocation.
    #[must_use]
    pub fn encode_scratch(&self) -> EncodeScratch {
        EncodeScratch {
            inner: self.inner.scratch(),
        }
    }
}

impl<F: Field> Coded for SystematicEncoder<F> {
    fn k(&self) -> usize {
        self.profile.k
    }
    fn m(&self) -> usize {
        self.profile.m
    }
    fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }
    fn n(&self) -> usize {
        self.profile.n
    }
}

impl<F: Field> BatchEncoder for SystematicEncoder<F> {
    type Scratch = EncodeScratch;

    fn scratch(&self) -> EncodeScratch {
        self.encode_scratch()
    }

    /// Encode repairs into a caller-provided flat `m * symbol_len` buffer,
    /// allocating a throwaway transform workspace per call.
    fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        let mut scratch = self.encode_scratch();
        self.encode_into_with(data, repairs, &mut scratch)
    }

    /// Encode repairs into a caller-provided buffer using reusable scratch.
    ///
    /// After the first sizing call, steady-state use performs no heap
    /// allocation.
    fn encode_into_with(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut EncodeScratch,
    ) -> Result<(), EncodeError> {
        let expected_data = self.profile.k * self.profile.symbol_len;
        if data.len() != expected_data {
            return Err(EncodeError::WrongInputLen {
                expected: expected_data,
                got: data.len(),
            });
        }
        let expected_repairs = self.profile.m * self.profile.symbol_len;
        if repairs.len() != expected_repairs {
            return Err(EncodeError::WrongOutputLen {
                expected: expected_repairs,
                got: repairs.len(),
            });
        }
        self.inner
            .encode(data, repairs, &mut scratch.inner)
            .expect("lengths validated above");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fff::gf16::Elem as GfElem;

    type Enc = SystematicEncoder<fff::Gf16>;

    #[test]
    fn validates_transform_capacity() {
        assert!(Enc::new(257, 128, 8).is_ok());
        assert!(Enc::new(32_769, 1, 2).is_ok());
        assert!(Enc::new(65_535, 2, 2).is_err());
        assert!(Enc::new(1, 1, 3).is_err());
        let encoder = Enc::new(5, 3, 2).unwrap();
        assert_eq!(encoder.padded_k(), 8);
        assert_eq!(encoder.transform_size(), 8);
    }

    /// Strip width is a cache-tuning parameter, never a correctness one: forcing
    /// the narrowest legal strip must reproduce the single-strip result. This
    /// covers cafft's gather/scatter and last-strip remainder handling at the
    /// geometries SCRS actually configures.
    #[test]
    fn strip_width_does_not_change_the_result() {
        for (k, m, l) in [(5, 3, 64), (100, 20, 64), (17, 7, 130), (512, 128, 40)] {
            let enc = Enc::new(k, m, l).unwrap();
            let data: Vec<u8> = (0..k * l).map(|i| (i * 137 + 11) as u8).collect();

            let mut tuned = vec![0u8; m * l];
            let mut s1 = enc.encode_scratch();
            enc.encode_into_with(&data, &mut tuned, &mut s1).unwrap();

            let mut narrow = vec![0u8; m * l];
            let mut s2 = cafft::rs::EncodeScratch::new();
            enc.inner
                .encode_with_width(&data, &mut narrow, &mut s2, 2)
                .unwrap();
            assert_eq!(tuned, narrow, "strip width changed the result k={k} m={m} l={l}");

            let mut wide = vec![0u8; m * l];
            let mut s3 = cafft::rs::EncodeScratch::new();
            enc.inner
                .encode_with_width(&data, &mut wide, &mut s3, l)
                .unwrap();
            assert_eq!(tuned, wide, "single-strip mismatch k={k} m={m} l={l}");
        }
    }

    /// Repairs must equal the textbook Lagrange evaluation of the systematic
    /// polynomial at points `k..k+m`. This is the ground truth for the whole
    /// engine: it shares no code with the transform.
    #[test]
    fn encoding_is_systematic_and_repairs_match_scalar_transform() {
        let k = 5;
        let m = 3;
        let symbol_len = 6;
        let data: Vec<_> = (0..k * symbol_len)
            .map(|index| ((index * 29) ^ 0xa5) as u8)
            .collect();
        let encoder = Enc::new(k, m, symbol_len).unwrap();
        let repairs = encoder.encode(&data).unwrap();

        for element_offset in (0..symbol_len).step_by(2) {
            for (repair, repair_bytes) in repairs.iter().enumerate() {
                let evaluation = GfElem((k + repair) as u16);
                let mut expected = GfElem::ZERO;
                for data_index in 0..k {
                    let mut numerator = GfElem::ONE;
                    let mut denominator = GfElem::ONE;
                    for other in 0..k {
                        if other == data_index {
                            continue;
                        }
                        numerator = numerator.mul(evaluation.add(GfElem(other as u16)));
                        denominator =
                            denominator.mul(GfElem(data_index as u16).add(GfElem(other as u16)));
                    }
                    let start = data_index * symbol_len + element_offset;
                    let value = GfElem::from_bytes([data[start], data[start + 1]]);
                    expected = expected.add(value.mul(numerator).mul(denominator.inv()));
                }
                assert_eq!(
                    &repair_bytes[element_offset..element_offset + 2],
                    &expected.to_bytes()
                );
            }
        }
    }

    #[test]
    fn encode_into_with_matches_encode_and_reuses_scratch() {
        let (k, m, symbol_len) = (100usize, 20usize, 1024usize);
        let encoder = Enc::new(k, m, symbol_len).unwrap();
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|index| (index.wrapping_mul(31) ^ 0xa5) as u8)
            .collect();
        let reference: Vec<u8> = encoder
            .encode(&data)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        let mut scratch = encoder.encode_scratch();
        let mut repairs = vec![0u8; m * symbol_len];
        encoder
            .encode_into_with(&data, &mut repairs, &mut scratch)
            .unwrap();
        assert_eq!(repairs, reference);

        for _ in 0..8 {
            encoder
                .encode_into_with(&data, &mut repairs, &mut scratch)
                .unwrap();
            assert_eq!(repairs, reference);
        }
    }
}
