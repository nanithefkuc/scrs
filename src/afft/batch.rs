//! Native block-final additive-FFT erasure decoder.

use std::sync::{Arc, OnceLock};

use cafft::core::kernel::{xor_scaled_bytes, xor_scaled_bytes_rows};
use cafft::rs::{
    ErasureLocator, LocatorScratch, generator_row, inverse_scratch_elements, invert_square_into,
};
use fff::field::Elem as _;

use super::decoder::{TARGETED_MAX_MISSING, fit, systematic_locators};
use super::profile::Profile;
use super::{Field, TransformPlan};
use crate::codec::{BatchDecoder as BatchDecoderTrait, Coded};
use crate::error::{ConfigError, DecodeError};

/// Native block-final additive-FFT decoder.
///
/// Unlike [`super::LazyDecoderState`], this type does not retain receipt state
/// or a domain-sized payload buffer. Each decode validates the borrowed input
/// rows, copies surviving systematic rows directly to the output, and builds
/// reconstruction state from those borrows. The payloads therefore cross the
/// API boundary once rather than being staged by `reset + push*k + finalize`.
#[derive(Clone, Debug)]
pub struct BatchDecoder<F: Field> {
    profile: Profile<F>,
    plan: Arc<TransformPlan<F>>,
    systematic_locator: OnceLock<Arc<ErasureLocator<F>>>,
}

/// Reusable workspace for allocation-free native AFFT batch decode.
#[derive(Debug)]
pub struct BatchDecodeScratch<F: Field> {
    k: usize,
    m: usize,
    symbol_len: usize,
    transform_size: usize,
    missing_data: Vec<usize>,
    known: Vec<bool>,
    locator: ErasureLocator<F>,
    locator_scratch: LocatorScratch,
    repair_indices: Vec<usize>,
    generator: Vec<F::Elem>,
    system: Vec<F::Elem>,
    inverse: Vec<F::Elem>,
    augmented: Vec<F::Elem>,
    coefficients: Vec<F::Elem>,
    residuals: Vec<u8>,
    recovered: Vec<u8>,
    /// Locator path's scaled `F * Λ` evaluations. This is transform input,
    /// not a staging copy of the received payload domain.
    product: Vec<u8>,
    derivative: Vec<u8>,
}

impl<F: Field> BatchDecodeScratch<F> {
    /// Create empty scratch that is sized on first use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: 0,
            m: 0,
            symbol_len: 0,
            transform_size: 0,
            missing_data: Vec::new(),
            known: Vec::new(),
            locator: ErasureLocator::for_domain(0),
            locator_scratch: LocatorScratch::new(),
            repair_indices: Vec::new(),
            generator: Vec::new(),
            system: Vec::new(),
            inverse: Vec::new(),
            augmented: Vec::new(),
            coefficients: Vec::new(),
            residuals: Vec::new(),
            recovered: Vec::new(),
            product: Vec::new(),
            derivative: Vec::new(),
        }
    }
}

impl<F: Field> Default for BatchDecodeScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<F: Field> BatchDecodeScratch<F> {
    /// Systematic dimension this scratch is sized for; `0` means never sized.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Repair count this scratch is sized for.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Per-symbol byte length this scratch is sized for.
    #[must_use]
    pub fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Evaluation-domain size this scratch is sized for.
    #[must_use]
    pub fn transform_size(&self) -> usize {
        self.transform_size
    }

    /// Missing systematic indices from the latest decode, ascending.
    #[must_use]
    pub fn missing_data(&self) -> &[usize] {
        &self.missing_data
    }

    /// Receipt map over the evaluation domain from the latest decode.
    #[must_use]
    pub fn known(&self) -> &[bool] {
        &self.known
    }

    /// Locator-path `F * Λ` transform input.
    #[must_use]
    pub fn product(&self) -> &[u8] {
        &self.product
    }

    /// Locator-path formal-derivative transform buffer.
    #[must_use]
    pub fn derivative(&self) -> &[u8] {
        &self.derivative
    }
}

impl<F: Field> BatchDecoder<F> {
    /// Construct a native batch decoder matching an AFFT encoder geometry.
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
        let plan = TransformPlan::shared(profile.transform_size)
            .map_err(|_| ConfigError::TooManySymbols { cap })?;
        Ok(Self {
            profile,
            plan,
            systematic_locator: OnceLock::new(),
        })
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

    /// Number of transmitted symbols.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.profile.n
    }

    /// Per-symbol byte length.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Power-of-two transform domain size.
    #[must_use]
    pub const fn transform_size(&self) -> usize {
        self.profile.transform_size
    }

    /// Allocate fully sized reusable batch decode scratch.
    #[must_use]
    pub fn decode_scratch(&self) -> BatchDecodeScratch<F> {
        let k = self.profile.k;
        let targeted = TARGETED_MAX_MISSING;
        let _ = self.systematic_locator();
        BatchDecodeScratch {
            k,
            m: self.profile.m,
            symbol_len: self.profile.symbol_len,
            transform_size: self.profile.transform_size,
            missing_data: Vec::with_capacity(k),
            known: vec![false; self.profile.transform_size],
            locator: ErasureLocator::for_domain(self.profile.transform_size),
            locator_scratch: LocatorScratch::for_domain(self.profile.transform_size),
            repair_indices: Vec::with_capacity(targeted),
            generator: vec![F::Elem::ZERO; targeted * k],
            system: vec![F::Elem::ZERO; targeted * targeted],
            inverse: vec![F::Elem::ZERO; targeted * targeted],
            augmented: vec![F::Elem::ZERO; inverse_scratch_elements(targeted)],
            coefficients: vec![F::Elem::ZERO; targeted],
            residuals: Vec::new(),
            recovered: Vec::new(),
            product: Vec::new(),
            derivative: Vec::new(),
        }
    }

    fn systematic_locator(&self) -> &Arc<ErasureLocator<F>> {
        self.systematic_locator.get_or_init(|| {
            systematic_locators::<F>()
                .get(&self.plan, self.profile.k)
                .expect("systematic locator domain matches the profile transform")
        })
    }

    fn ensure_scratch(&self, scratch: &mut BatchDecodeScratch<F>) -> Result<(), DecodeError> {
        if scratch.k == 0 {
            *scratch = self.decode_scratch();
            return Ok(());
        }
        if (
            scratch.k,
            scratch.m,
            scratch.symbol_len,
            scratch.transform_size,
        ) != (
            self.profile.k,
            self.profile.m,
            self.profile.symbol_len,
            self.profile.transform_size,
        ) {
            return Err(DecodeError::ScratchMismatch);
        }
        Ok(())
    }

    fn validate_symbols(
        &self,
        symbols: &[(usize, &[u8])],
        scratch: &mut BatchDecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        if symbols.len() != self.profile.k {
            return Err(DecodeError::WrongCount {
                expected: self.profile.k,
                got: symbols.len(),
            });
        }
        scratch.known.fill(false);
        scratch.repair_indices.clear();
        for &(index, payload) in symbols {
            if index >= self.profile.n {
                return Err(DecodeError::IndexOutOfRange {
                    index,
                    n: self.profile.n,
                });
            }
            if payload.len() != self.profile.symbol_len {
                return Err(DecodeError::WrongPayloadLen {
                    expected: self.profile.symbol_len,
                    got: payload.len(),
                });
            }
            if scratch.known[index] {
                return Err(DecodeError::DuplicateIndex { index });
            }
            scratch.known[index] = true;
            if index >= self.profile.k {
                scratch.repair_indices.push(index);
            }
        }
        scratch.missing_data.clear();
        scratch.missing_data.extend(
            scratch.known[..self.profile.k]
                .iter()
                .enumerate()
                .filter_map(|(index, &known)| (!known).then_some(index)),
        );
        debug_assert_eq!(scratch.missing_data.len(), scratch.repair_indices.len());
        Ok(())
    }

    fn decode_complete_into(
        &self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
        scratch: &mut BatchDecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        self.ensure_scratch(scratch)?;
        self.validate_symbols(symbols, scratch)?;
        let expected = self.profile.k * self.profile.symbol_len;
        if output.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }

        let symbol_len = self.profile.symbol_len;
        for &(index, payload) in symbols {
            if index < self.profile.k {
                let start = index * symbol_len;
                output[start..start + symbol_len].copy_from_slice(payload);
            }
        }
        if scratch.missing_data.is_empty() {
            return Ok(());
        }
        if scratch.missing_data.len() <= TARGETED_MAX_MISSING {
            self.decode_targeted_into(symbols, output, scratch);
        } else {
            self.decode_locator_into(symbols, output, scratch);
        }
        Ok(())
    }

    fn decode_targeted_into(
        &self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
        scratch: &mut BatchDecodeScratch<F>,
    ) {
        let k = self.profile.k;
        let symbol_len = self.profile.symbol_len;
        let missing_count = scratch.missing_data.len();
        let locator = self.systematic_locator();

        let generator = &mut scratch.generator[..missing_count * k];
        for (row, &wire_index) in scratch.repair_indices.iter().enumerate() {
            generator_row(
                &self.plan,
                locator,
                self.profile.evaluation_index(wire_index),
                &mut generator[row * k..(row + 1) * k],
            );
        }

        let system = &mut scratch.system[..missing_count * missing_count];
        for row in 0..missing_count {
            for (column, &data_index) in scratch.missing_data.iter().enumerate() {
                system[row * missing_count + column] = generator[row * k + data_index];
            }
        }
        let inverse = &mut scratch.inverse[..missing_count * missing_count];
        assert!(
            invert_square_into(system, missing_count, &mut scratch.augmented, inverse),
            "every supported AFFT erasure pattern is invertible"
        );

        let residuals = fit(&mut scratch.residuals, missing_count * symbol_len);
        for (row, &repair_index) in scratch.repair_indices.iter().enumerate() {
            let payload = symbols
                .iter()
                .find_map(|&(index, payload)| (index == repair_index).then_some(payload))
                .expect("validated repair index has a borrowed payload");
            residuals[row * symbol_len..(row + 1) * symbol_len].copy_from_slice(payload);
        }
        let coefficients = &mut scratch.coefficients[..missing_count];
        for &(data_index, payload) in symbols {
            if data_index >= k {
                continue;
            }
            for row in 0..missing_count {
                coefficients[row] = generator[row * k + data_index];
            }
            xor_scaled_bytes_rows::<F>(residuals, symbol_len, coefficients, payload);
        }

        let recovered = fit(&mut scratch.recovered, missing_count * symbol_len);
        for residual_row in 0..missing_count {
            for output_row in 0..missing_count {
                coefficients[output_row] = inverse[output_row * missing_count + residual_row];
            }
            let source = residual_row * symbol_len;
            xor_scaled_bytes_rows::<F>(
                recovered,
                symbol_len,
                coefficients,
                &residuals[source..source + symbol_len],
            );
        }
        for (row, &data_index) in scratch.missing_data.iter().enumerate() {
            let destination = data_index * symbol_len;
            output[destination..destination + symbol_len]
                .copy_from_slice(&recovered[row * symbol_len..(row + 1) * symbol_len]);
        }
    }

    fn decode_locator_into(
        &self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
        scratch: &mut BatchDecodeScratch<F>,
    ) {
        let symbol_len = self.profile.symbol_len;
        let transform_bytes = self.profile.transform_size * symbol_len;
        scratch
            .locator
            .recompute(&self.plan, &scratch.known, &mut scratch.locator_scratch)
            .expect("locator domain matches the profile transform");

        let product = fit(&mut scratch.product, transform_bytes);
        for &(index, payload) in symbols {
            let point = self.profile.evaluation_index(index);
            let value = scratch.locator.values()[point];
            debug_assert!(!value.is_zero());
            let start = point * symbol_len;
            xor_scaled_bytes::<F>(&mut product[start..start + symbol_len], value, payload);
        }
        self.plan
            .inverse_bytes(product, symbol_len)
            .expect("product geometry matches the transform");
        let derivative = fit(&mut scratch.derivative, transform_bytes);
        self.plan
            .derivative_bytes(product, symbol_len, derivative)
            .expect("derivative geometry matches the transform");
        self.plan
            .forward_bytes_selected(derivative, symbol_len, &scratch.missing_data)
            .expect("missing points match the transform");

        let recovered = fit(
            &mut scratch.recovered,
            scratch.missing_data.len() * symbol_len,
        );
        for (row, &point) in scratch.missing_data.iter().enumerate() {
            let source = point * symbol_len;
            let destination = row * symbol_len;
            xor_scaled_bytes::<F>(
                &mut recovered[destination..destination + symbol_len],
                scratch.locator.derivatives()[point].inv(),
                &derivative[source..source + symbol_len],
            );
        }
        for (row, &data_index) in scratch.missing_data.iter().enumerate() {
            let destination = data_index * symbol_len;
            output[destination..destination + symbol_len]
                .copy_from_slice(&recovered[row * symbol_len..(row + 1) * symbol_len]);
        }
    }
}

impl<F: Field> Coded for BatchDecoder<F> {
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

impl<F: Field> BatchDecoderTrait for BatchDecoder<F> {
    type Scratch = BatchDecodeScratch<F>;

    fn scratch(&self) -> Self::Scratch {
        self.decode_scratch()
    }

    fn decode_into(
        &mut self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
    ) -> Result<(), DecodeError> {
        let mut scratch = BatchDecodeScratch::new();
        self.decode_complete_into(symbols, output, &mut scratch)
    }

    fn decode_into_with(
        &mut self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError> {
        self.decode_complete_into(symbols, output, scratch)
    }
}

/// GF(2^8) native additive-FFT batch decoder.
pub type Gf8BatchDecoder = BatchDecoder<fff::Gf8>;
/// GF(2^16) native additive-FFT batch decoder.
pub type Gf16BatchDecoder = BatchDecoder<fff::Gf16>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::BatchEncoder as _;

    fn round_trip<F: Field>(k: usize, m: usize, symbol_len: usize, missing: usize) {
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|index| (index.wrapping_mul(131) + 7) as u8)
            .collect();
        let encoder = super::super::SystematicEncoder::<F>::new(k, m, symbol_len).unwrap();
        let mut repairs = vec![0u8; m * symbol_len];
        encoder.encode_into(&data, &mut repairs).unwrap();
        let mut word: Vec<&[u8]> = data.chunks_exact(symbol_len).collect();
        word.extend(repairs.chunks_exact(symbol_len));
        let mut indices: Vec<usize> = (missing..k).chain(k..k + missing).collect();
        indices.reverse();
        let received: Vec<(usize, &[u8])> =
            indices.iter().map(|&index| (index, word[index])).collect();

        let mut decoder = BatchDecoder::<F>::new(k, m, symbol_len).unwrap();
        let mut scratch = decoder.decode_scratch();
        let mut output = vec![0xA5; k * symbol_len];
        decoder
            .decode_into_with(&received, &mut output, &mut scratch)
            .unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn native_batch_decode_covers_both_fields_and_paths() {
        for missing in [0, 1, 4, 5, 6, 8] {
            round_trip::<fff::Gf8>(16, 8, 63, missing);
            round_trip::<fff::Gf16>(16, 8, 64, missing);
        }
        round_trip::<fff::Gf8>(64, 32, 257, 16);
        round_trip::<fff::Gf16>(64, 32, 258, 16);
    }

    #[test]
    fn native_batch_decode_rejects_invalid_inputs() {
        let mut decoder = Gf8BatchDecoder::new(4, 2, 8).unwrap();
        let mut scratch = decoder.decode_scratch();
        let payload = [0u8; 8];
        let short = [0u8; 7];
        let mut output = [0u8; 32];

        assert_eq!(
            decoder.decode_into_with(&[(0, &payload)], &mut output, &mut scratch),
            Err(DecodeError::WrongCount {
                expected: 4,
                got: 1
            })
        );
        assert_eq!(
            decoder.decode_into_with(
                &[(0, &payload), (1, &payload), (2, &payload), (6, &payload)],
                &mut output,
                &mut scratch,
            ),
            Err(DecodeError::IndexOutOfRange { index: 6, n: 6 })
        );
        assert_eq!(
            decoder.decode_into_with(
                &[(0, &payload), (1, &payload), (1, &payload), (4, &payload)],
                &mut output,
                &mut scratch,
            ),
            Err(DecodeError::DuplicateIndex { index: 1 })
        );
        assert_eq!(
            decoder.decode_into_with(
                &[(0, &payload), (1, &short), (2, &payload), (4, &payload)],
                &mut output,
                &mut scratch,
            ),
            Err(DecodeError::WrongPayloadLen {
                expected: 8,
                got: 7
            })
        );
        assert_eq!(
            decoder.decode_into_with(
                &[(0, &payload), (1, &payload), (2, &payload), (4, &payload)],
                &mut output[..31],
                &mut scratch,
            ),
            Err(DecodeError::WrongOutputLen {
                expected: 32,
                got: 31
            })
        );

        let other = Gf8BatchDecoder::new(5, 2, 8).unwrap();
        let mut wrong_scratch = other.decode_scratch();
        assert_eq!(
            decoder.decode_into_with(
                &[(0, &payload), (1, &payload), (2, &payload), (4, &payload)],
                &mut output,
                &mut wrong_scratch,
            ),
            Err(DecodeError::ScratchMismatch)
        );
    }
}
