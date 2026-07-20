//! Payload-lazy additive-FFT erasure decoder.

use std::sync::{LazyLock, OnceLock};

use crate::gf65536::GfElem;
use crate::stream::{PushOutcome, StreamError, SymbolSink};

use super::profile::{Profile, zeroed_bytes};

/// Lazy erasure decoder for [`super::SystematicEncoder`].
///
/// Receipt processing only validates, copies, and marks a dynamic bitmap. At
/// finalization, the decoder builds an erasure-locator evaluation map, obtains
/// the novel-basis coefficients of `F * locator` through an inverse additive
/// FFT, differentiates in the novel basis, and applies a forward additive FFT.
/// Missing evaluations follow from `(F * locator)' / locator'` at each erased
/// point.
#[derive(Clone, Debug)]
pub struct LazyDecoderState {
    profile: Profile,
    payloads: Vec<u8>,
    received_bits: Vec<u64>,
    distinct: usize,
    received: usize,
    /// Locator evaluations for the fixed systematic point set, computed once on
    /// first targeted-path finalize and reused by every later low-`r` decode.
    systematic_locator: OnceLock<SystematicLocator>,
}

/// Cached locator evaluations over the systematic point set `{0..k}`.
///
/// These depend only on `(k, transform_size)`, not the erasure pattern, so the
/// targeted decode path computes them at most once per decoder rather than
/// repeating an `O(N log N)` Walsh-Hadamard transform on every finalize.
#[derive(Clone, Debug)]
struct SystematicLocator {
    products: Vec<GfElem>,
    derivatives: Vec<GfElem>,
}

/// Reusable scratch for allocation-light additive-FFT decoding.
///
/// Holds the payload-proportional transform workspaces so steady-state
/// [`LazyDecoderState::finalize_into_with`] reuse never reallocates them, as
/// required by Aeron-style ring-buffer consumers. Small per-pattern metadata
/// (index lists, coefficient vectors) is still allocated per finalize.
/// Construct with [`LazyDecoderState::decode_scratch`].
#[derive(Clone, Debug, Default)]
pub struct DecodeScratch {
    work0: Vec<u8>,
    work1: Vec<u8>,
}

impl DecodeScratch {
    /// Create empty scratch that grows on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LazyDecoderState {
    /// Construct a decoder matching an additive-FFT encoder configuration.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let profile = Profile::new(k, m, symbol_len)?;
        let payloads = zeroed_bytes(profile.n * symbol_len)?;
        let bit_words = profile.n.checked_add(63)? / 64;
        let mut received_bits = Vec::new();
        received_bits.try_reserve_exact(bit_words).ok()?;
        received_bits.resize(bit_words, 0);
        Some(Self {
            profile,
            payloads,
            received_bits,
            distinct: 0,
            received: 0,
            systematic_locator: OnceLock::new(),
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

    /// Number of transmitted symbols.
    pub const fn n(&self) -> usize {
        self.profile.n
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Power-of-two plan size used for truncated systematic interpolation.
    pub const fn padded_k(&self) -> usize {
        self.profile.padded_k
    }

    /// Full power-of-two transform domain size.
    pub const fn transform_size(&self) -> usize {
        self.profile.transform_size
    }

    /// Number of distinct accepted symbols, capped at `k`.
    pub const fn rank(&self) -> usize {
        self.distinct
    }

    /// Total pushes, including duplicates.
    pub const fn received(&self) -> usize {
        self.received
    }

    /// Whether a transmitted codeword position has been accepted.
    pub fn has_symbol(&self, index: usize) -> bool {
        index < self.profile.n && self.bit(index)
    }

    /// Validate and record one systematic or repair symbol.
    pub fn push_symbol(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, StreamError> {
        self.push(index, symbol)
    }

    /// Reconstruct every systematic symbol without consuming the decoder.
    pub fn finalize_ref(&self) -> Result<Vec<u8>, StreamError> {
        self.ensure_complete()?;
        let mut output = zeroed_bytes(self.profile.k * self.profile.symbol_len)
            .expect("profile validated output allocation size");
        let mut scratch = DecodeScratch::new();
        self.finalize_complete_into(&mut output, &mut scratch);
        Ok(output)
    }

    /// Reconstruct into a caller-provided `k * symbol_len` buffer.
    ///
    /// Allocates transform scratch per call; use
    /// [`finalize_into_with`](Self::finalize_into_with) with reusable scratch
    /// for an allocation-free payload path.
    pub fn finalize_into(&self, output: &mut [u8]) -> Result<(), StreamError> {
        let mut scratch = DecodeScratch::new();
        self.finalize_into_with(output, &mut scratch)
    }

    /// Allocate decode scratch sized for this decoder's transform workspaces.
    ///
    /// Reuse the returned [`DecodeScratch`] across
    /// [`finalize_into_with`](Self::finalize_into_with) calls to avoid
    /// reallocating the payload-proportional workspaces.
    pub fn decode_scratch(&self) -> DecodeScratch {
        let workspace_len = self.profile.transform_size * self.profile.symbol_len;
        let mut work0 = Vec::new();
        work0.reserve_exact(workspace_len);
        let mut work1 = Vec::new();
        work1.reserve_exact(workspace_len);
        DecodeScratch { work0, work1 }
    }

    /// Reconstruct into a caller-provided buffer using reusable scratch.
    ///
    /// After the first sizing call the transform workspaces are reused without
    /// reallocation.
    pub fn finalize_into_with(
        &self,
        output: &mut [u8],
        scratch: &mut DecodeScratch,
    ) -> Result<(), StreamError> {
        self.ensure_complete()?;
        let expected = self.profile.k * self.profile.symbol_len;
        if output.len() != expected {
            return Err(StreamError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }
        self.finalize_complete_into(output, scratch);
        Ok(())
    }

    fn finalize_complete_into(&self, output: &mut [u8], scratch: &mut DecodeScratch) {
        let symbol_len = self.profile.symbol_len;
        let mut missing_data = Vec::new();
        for data in 0..self.profile.k {
            let start = data * symbol_len;
            if self.bit(data) {
                output[start..start + symbol_len]
                    .copy_from_slice(&self.payloads[start..start + symbol_len]);
            } else {
                output[start..start + symbol_len].fill(0);
                missing_data.push(data);
            }
        }
        if missing_data.is_empty() {
            return;
        }
        if missing_data.len() <= TARGETED_MAX_MISSING {
            self.finalize_targeted_into(output, &missing_data, scratch);
            return;
        }

        let transform_size = self.profile.transform_size;
        let mut known = vec![false; transform_size];
        for wire_index in 0..self.profile.n {
            if self.bit(wire_index) {
                known[self.profile.evaluation_index(wire_index)] = true;
            }
        }
        let erased: Vec<_> = known
            .iter()
            .enumerate()
            .filter_map(|(index, &is_known)| (!is_known).then_some(index))
            .collect();
        debug_assert_eq!(erased.len(), transform_size - self.profile.k);

        let (locator_values, locator_derivatives) = locator_evaluations(&known, &erased);
        let workspace_len = transform_size * symbol_len;
        let product_evaluations = resize_zeroed(&mut scratch.work0, workspace_len);
        for wire_index in 0..self.profile.n {
            if !self.bit(wire_index) {
                continue;
            }
            let evaluation_index = self.profile.evaluation_index(wire_index);
            let source_start = wire_index * symbol_len;
            let destination_start = evaluation_index * symbol_len;
            crate::tower::payload::xor_scaled_bytes(
                &mut product_evaluations[destination_start..destination_start + symbol_len],
                locator_values[evaluation_index],
                &self.payloads[source_start..source_start + symbol_len],
            );
        }

        self.profile
            .transform_plan
            .inverse_bytes(product_evaluations, symbol_len);
        let derivative_evaluations = resize_zeroed(&mut scratch.work1, workspace_len);
        self.profile.transform_plan.derivative_bytes(
            product_evaluations,
            symbol_len,
            derivative_evaluations,
        );
        self.profile
            .transform_plan
            .forward_bytes(derivative_evaluations, symbol_len);

        for data in missing_data {
            let output_start = data * symbol_len;
            let evaluation_start = data * symbol_len;
            crate::tower::payload::xor_scaled_bytes(
                &mut output[output_start..output_start + symbol_len],
                locator_derivatives[data].inv(),
                &derivative_evaluations[evaluation_start..evaluation_start + symbol_len],
            );
        }
    }
    fn finalize_targeted_into(
        &self,
        output: &mut [u8],
        missing_data: &[usize],
        scratch: &mut DecodeScratch,
    ) {
        let symbol_len = self.profile.symbol_len;
        let missing_count = missing_data.len();
        let repair_indices: Vec<_> = (self.profile.k..self.profile.n)
            .filter(|&wire_index| self.bit(wire_index))
            .collect();
        debug_assert_eq!(repair_indices.len(), missing_count);

        let locator = self.systematic_locator.get_or_init(|| {
            let mut outside_systematic = vec![true; self.profile.transform_size];
            outside_systematic[..self.profile.k].fill(false);
            let systematic: Vec<_> = (0..self.profile.k).collect();
            let (products, derivatives) = locator_evaluations(&outside_systematic, &systematic);
            SystematicLocator {
                products,
                derivatives,
            }
        });
        let systematic_products = &locator.products;
        let systematic_derivatives = &locator.derivatives;

        let mut generator = vec![GfElem::ZERO; missing_count * self.profile.k];
        for (row, &wire_index) in repair_indices.iter().enumerate() {
            let evaluation = self.profile.evaluation_index(wire_index);
            generator_row(
                evaluation,
                systematic_products,
                &systematic_derivatives[..self.profile.k],
                &mut generator[row * self.profile.k..(row + 1) * self.profile.k],
            );
        }

        let mut system = vec![GfElem::ZERO; missing_count * missing_count];
        for row in 0..missing_count {
            for (column, &data_index) in missing_data.iter().enumerate() {
                system[row * missing_count + column] = generator[row * self.profile.k + data_index];
            }
        }
        let inverse = invert_square(&system, missing_count)
            .expect("every supported AFFT erasure pattern is invertible");

        let residuals = resize_zeroed(&mut scratch.work0, missing_count * symbol_len);
        for (row, &wire_index) in repair_indices.iter().enumerate() {
            let source_start = wire_index * symbol_len;
            let destination_start = row * symbol_len;
            residuals[destination_start..destination_start + symbol_len]
                .copy_from_slice(&self.payloads[source_start..source_start + symbol_len]);
        }
        let mut coefficients = vec![GfElem::ZERO; missing_count];
        for data_index in 0..self.profile.k {
            if !self.bit(data_index) {
                continue;
            }
            for row in 0..missing_count {
                coefficients[row] = generator[row * self.profile.k + data_index];
            }
            let source_start = data_index * symbol_len;
            crate::tower::payload::xor_scaled_bytes_rows(
                residuals,
                symbol_len,
                &coefficients,
                &self.payloads[source_start..source_start + symbol_len],
            );
        }

        let recovered = resize_zeroed(&mut scratch.work1, missing_count * symbol_len);
        for residual_row in 0..missing_count {
            for output_row in 0..missing_count {
                coefficients[output_row] = inverse[output_row * missing_count + residual_row];
            }
            let source_start = residual_row * symbol_len;
            crate::tower::payload::xor_scaled_bytes_rows(
                recovered,
                symbol_len,
                &coefficients,
                &residuals[source_start..source_start + symbol_len],
            );
        }
        for (row, &data_index) in missing_data.iter().enumerate() {
            let source_start = row * symbol_len;
            let destination_start = data_index * symbol_len;
            output[destination_start..destination_start + symbol_len]
                .copy_from_slice(&recovered[source_start..source_start + symbol_len]);
        }
    }

    fn ensure_complete(&self) -> Result<(), StreamError> {
        if self.distinct < self.profile.k {
            Err(StreamError::InsufficientRank {
                rank: self.distinct,
                k: self.profile.k,
            })
        } else {
            Ok(())
        }
    }

    fn bit(&self, index: usize) -> bool {
        self.received_bits[index / 64] & (1u64 << (index % 64)) != 0
    }

    fn set_bit(&mut self, index: usize) {
        self.received_bits[index / 64] |= 1u64 << (index % 64);
    }
}

impl SymbolSink for LazyDecoderState {
    fn push(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, StreamError> {
        if index >= self.profile.n {
            return Err(StreamError::IndexOutOfRange {
                index,
                n: self.profile.n,
            });
        }
        if symbol.len() != self.profile.symbol_len {
            return Err(StreamError::WrongPayloadLen {
                expected: self.profile.symbol_len,
                got: symbol.len(),
            });
        }
        if self.received >= self.profile.n {
            return Err(StreamError::TooManySymbols {
                cap: self.profile.n,
                received: self.received,
            });
        }
        self.received += 1;
        if self.distinct >= self.profile.k || self.bit(index) {
            return Ok(PushOutcome::Dependent);
        }

        let start = index * self.profile.symbol_len;
        self.payloads[start..start + self.profile.symbol_len].copy_from_slice(symbol);
        self.set_bit(index);
        self.distinct += 1;
        if self.distinct == self.profile.k {
            Ok(PushOutcome::Complete)
        } else {
            Ok(PushOutcome::Advanced {
                rank: self.distinct,
                received: self.received,
            })
        }
    }

    fn is_complete(&self) -> bool {
        self.distinct >= self.profile.k
    }

    fn finalize(self) -> Result<Vec<u8>, StreamError> {
        self.finalize_ref()
    }
}

fn resize_zeroed(buffer: &mut Vec<u8>, len: usize) -> &mut [u8] {
    buffer.clear();
    buffer.resize(len, 0);
    &mut buffer[..]
}

const TARGETED_MAX_MISSING: usize = 5;

const MULTIPLICATIVE_ORDER: u32 = 65_535;

struct LogExpTables {
    log: Vec<u16>,
    exp: Vec<GfElem>,
}

static LOG_EXP_TABLES: LazyLock<LogExpTables> = LazyLock::new(|| {
    let mut log = vec![0; 65_536];
    let mut exp = Vec::with_capacity(MULTIPLICATIVE_ORDER as usize);
    let mut value = GfElem::ONE;
    for exponent in 0..MULTIPLICATIVE_ORDER {
        exp.push(value);
        log[value.to_u16() as usize] = exponent as u16;
        value = value.mul(crate::gf65536::GENERATOR);
    }
    debug_assert_eq!(value, GfElem::ONE);
    LogExpTables { log, exp }
});

fn generator_row(
    evaluation: usize,
    products: &[GfElem],
    derivatives: &[GfElem],
    row: &mut [GfElem],
) {
    debug_assert_eq!(row.len(), derivatives.len());
    debug_assert!(evaluation >= row.len());
    let tables = &*LOG_EXP_TABLES;
    let numerator_log = tables.log[products[evaluation].to_u16() as usize] as u32;
    for (data_index, coefficient) in row.iter_mut().enumerate() {
        let difference_log = tables.log[evaluation ^ data_index] as u32;
        let derivative_log = tables.log[derivatives[data_index].to_u16() as usize] as u32;
        let exponent = (numerator_log + MULTIPLICATIVE_ORDER * 2 - difference_log - derivative_log)
            % MULTIPLICATIVE_ORDER;
        *coefficient = tables.exp[exponent as usize];
    }
}

fn invert_square(matrix: &[GfElem], size: usize) -> Option<Vec<GfElem>> {
    let stride = size.checked_mul(2)?;
    let mut augmented = vec![GfElem::ZERO; size.checked_mul(stride)?];
    for row in 0..size {
        augmented[row * stride..row * stride + size]
            .copy_from_slice(&matrix[row * size..(row + 1) * size]);
        augmented[row * stride + size + row] = GfElem::ONE;
    }
    for column in 0..size {
        let pivot = (column..size).find(|&row| augmented[row * stride + column] != GfElem::ZERO)?;
        if pivot != column {
            for entry in 0..stride {
                augmented.swap(column * stride + entry, pivot * stride + entry);
            }
        }
        let inverse = augmented[column * stride + column].inv();
        for entry in column..stride {
            augmented[column * stride + entry] = augmented[column * stride + entry].mul(inverse);
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row * stride + column];
            if factor == GfElem::ZERO {
                continue;
            }
            for entry in column..stride {
                let pivot_value = augmented[column * stride + entry];
                augmented[row * stride + entry] =
                    augmented[row * stride + entry].add(factor.mul(pivot_value));
            }
        }
    }
    let mut inverse = vec![GfElem::ZERO; size * size];
    for row in 0..size {
        inverse[row * size..(row + 1) * size]
            .copy_from_slice(&augmented[row * stride + size..(row + 1) * stride]);
    }
    Some(inverse)
}

fn locator_evaluations(known: &[bool], erased: &[usize]) -> (Vec<GfElem>, Vec<GfElem>) {
    // Product evaluations become sums of discrete logarithms:
    // log Π(p) = Σ_e log(p + e). Since additive points use the raw binary
    // basis, p + e is p XOR e. This is an XOR convolution, computed with a
    // Walsh-Hadamard transform over Z/65535Z in O(N log N). Setting log(0) to
    // zero also gives log Π'(e) at erased roots by omitting the self-factor.
    let tables = &*LOG_EXP_TABLES;
    let mut indicator = vec![0u32; known.len()];
    for &position in erased {
        indicator[position] = 1;
    }
    let mut logarithms: Vec<u32> = (0..known.len())
        .map(|value| tables.log[value] as u32)
        .collect();
    walsh_hadamard(&mut indicator);
    walsh_hadamard(&mut logarithms);
    for (left, right) in indicator.iter_mut().zip(logarithms) {
        *left = ((*left as u64 * right as u64) % MULTIPLICATIVE_ORDER as u64) as u32;
    }
    walsh_hadamard(&mut indicator);
    // N is a power of two and 2^-1 = 32768 mod 65535.
    let inverse_size = mod_pow(32_768, known.len().trailing_zeros());
    for value in &mut indicator {
        *value = ((*value as u64 * inverse_size as u64) % MULTIPLICATIVE_ORDER as u64) as u32;
    }

    let mut values = vec![GfElem::ZERO; known.len()];
    let mut derivatives = vec![GfElem::ZERO; known.len()];
    for (position, &is_known) in known.iter().enumerate() {
        let product = tables.exp[indicator[position] as usize];
        if is_known {
            values[position] = product;
        } else {
            derivatives[position] = product;
        }
    }
    (values, derivatives)
}

fn walsh_hadamard(values: &mut [u32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(half * 2) {
            for position in 0..half {
                let left = block[position];
                let right = block[half + position];
                let sum = left + right;
                block[position] = if sum >= MULTIPLICATIVE_ORDER {
                    sum - MULTIPLICATIVE_ORDER
                } else {
                    sum
                };
                block[half + position] = if left >= right {
                    left - right
                } else {
                    left + MULTIPLICATIVE_ORDER - right
                };
            }
        }
        half *= 2;
    }
}

fn mod_pow(mut base: u32, mut exponent: u32) -> u32 {
    let mut result = 1u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = ((result as u64 * base as u64) % MULTIPLICATIVE_ORDER as u64) as u32;
        }
        base = ((base as u64 * base as u64) % MULTIPLICATIVE_ORDER as u64) as u32;
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::afft::SystematicEncoder;

    fn data_block(k: usize, symbol_len: usize) -> Vec<u8> {
        (0..k * symbol_len)
            .map(|index| (index.wrapping_mul(37) ^ index.rotate_left(3) ^ 0x5a) as u8)
            .collect()
    }

    fn codeword(k: usize, m: usize, symbol_len: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
        let data = data_block(k, symbol_len);
        let encoder = SystematicEncoder::new(k, m, symbol_len).unwrap();
        let repairs = encoder.encode(&data).unwrap();
        let mut word: Vec<Vec<u8>> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
        word.extend(repairs);
        (word, data)
    }

    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut state: Vec<_> = (0..k).collect();
        loop {
            result.push(state.clone());
            let Some(position) = (0..k)
                .rev()
                .find(|&position| state[position] < n - k + position)
            else {
                return result;
            };
            state[position] += 1;
            for next in position + 1..k {
                state[next] = state[next - 1] + 1;
            }
        }
    }

    fn direct_locator_evaluations(known: &[bool], erased: &[usize]) -> (Vec<GfElem>, Vec<GfElem>) {
        let mut values = vec![GfElem::ZERO; known.len()];
        for (position, &is_known) in known.iter().enumerate() {
            if is_known {
                values[position] = erased.iter().fold(GfElem::ONE, |product, &erasure| {
                    product.mul(GfElem((position ^ erasure) as u16))
                });
            }
        }
        let mut derivatives = vec![GfElem::ZERO; known.len()];
        for (erased_position, &position) in erased.iter().enumerate() {
            derivatives[position] = erased
                .iter()
                .enumerate()
                .filter(|&(other_position, _)| other_position != erased_position)
                .fold(GfElem::ONE, |product, (_, &other)| {
                    product.mul(GfElem((position ^ other) as u16))
                });
        }
        (values, derivatives)
    }

    #[test]
    fn walsh_locator_matches_direct_products() {
        for size in [2, 4, 8, 16, 64, 256] {
            let known: Vec<_> = (0..size)
                .map(|position| position % 3 == 0 || position % 7 == 1)
                .collect();
            let erased: Vec<_> = known
                .iter()
                .enumerate()
                .filter_map(|(position, &is_known)| (!is_known).then_some(position))
                .collect();
            assert_eq!(
                locator_evaluations(&known, &erased),
                direct_locator_evaluations(&known, &erased)
            );
        }
    }

    #[test]
    fn exhaustive_power_of_two_recovery() {
        let k = 4;
        let m = 4;
        let symbol_len = 10;
        let (word, expected) = codeword(k, m, symbol_len);
        for selected in combinations(k + m, k) {
            let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
            for index in selected {
                decoder.push_symbol(index, &word[index]).unwrap();
            }
            assert_eq!(decoder.finalize_ref().unwrap(), expected);
        }
    }

    #[test]
    fn exhaustive_truncated_recovery() {
        let k = 5;
        let m = 3;
        let symbol_len = 6;
        let (word, expected) = codeword(k, m, symbol_len);
        for selected in combinations(k + m, k) {
            let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
            for index in selected {
                decoder.push_symbol(index, &word[index]).unwrap();
            }
            assert_eq!(decoder.finalize_ref().unwrap(), expected);
        }
    }

    #[test]
    fn randomized_larger_patterns() {
        let k = 32;
        let m = 16;
        let symbol_len = 34;
        let (word, expected) = codeword(k, m, symbol_len);
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..20 {
            let mut indices: Vec<_> = (0..k + m).collect();
            for position in (1..indices.len()).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                indices.swap(position, state as usize % (position + 1));
            }
            let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
            for &index in &indices[..k] {
                decoder.push_symbol(index, &word[index]).unwrap();
            }
            assert_eq!(decoder.finalize_ref().unwrap(), expected);
        }
    }

    #[test]
    fn push_remains_payload_lazy() {
        let mut decoder = LazyDecoderState::new(300, 2, 2).unwrap();
        assert_eq!(
            decoder.push_symbol(299, &[1, 2]).unwrap(),
            PushOutcome::Advanced {
                rank: 1,
                received: 1
            }
        );
        assert!(decoder.has_symbol(299));
        assert_eq!(
            decoder.push_symbol(299, &[3, 4]).unwrap(),
            PushOutcome::Dependent
        );
    }

    #[test]
    fn finalize_into_with_matches_finalize_ref_and_reuses_scratch() {
        // High r (> TARGETED_MAX_MISSING) exercises the complete transform path;
        // low r exercises the targeted path. Both must match finalize_ref and
        // reuse the payload workspaces without reallocating.
        for (k, m, symbol_len, drop_count) in [(32usize, 16usize, 34usize, 12usize), (16, 8, 64, 3)]
        {
            let (word, expected) = codeword(k, m, symbol_len);
            let decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
            let mut scratch = decoder.decode_scratch();
            let mut out = vec![0u8; k * symbol_len];

            let mut ptrs: Option<(*const u8, *const u8, usize, usize)> = None;
            for round in 0..6 {
                // Rotate which symbols are received so the erasure pattern varies.
                let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
                let mut received = 0;
                for offset in 0..k + m {
                    let index = (offset + round) % (k + m);
                    if index < drop_count {
                        continue; // simulate a fixed-size erasure set of size drop_count
                    }
                    if received == k {
                        break;
                    }
                    decoder.push_symbol(index, &word[index]).unwrap();
                    received += 1;
                }
                if received < k {
                    // Top up from the front skipping already-used indices.
                    for index in 0..k + m {
                        if received == k {
                            break;
                        }
                        if !decoder.has_symbol(index) {
                            decoder.push_symbol(index, &word[index]).unwrap();
                            received += 1;
                        }
                    }
                }
                decoder.finalize_into_with(&mut out, &mut scratch).unwrap();
                assert_eq!(out, expected);
                assert_eq!(decoder.finalize_ref().unwrap(), expected);

                let snapshot = (
                    scratch.work0.as_ptr(),
                    scratch.work1.as_ptr(),
                    scratch.work0.capacity(),
                    scratch.work1.capacity(),
                );
                if let Some(previous) = ptrs {
                    assert_eq!(snapshot, previous, "decode scratch reallocated between calls");
                }
                ptrs = Some(snapshot);
            }
        }
    }
}
