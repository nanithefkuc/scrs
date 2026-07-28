//! Payload-lazy reduced Tower Cauchy decoder.

use crate::error::{ConfigError, DecodeError};
use crate::gf65536::GfElem;
use crate::stream::{PushOutcome, SymbolSink};

use super::{MAX_SYMBOLS, TowerCauchyView, payload};

/// Lazy streaming decoder for the GF(65536) Tower Cauchy code.
///
/// `push_symbol` validates, copies, and marks receipt in a dynamic bitmap. Once
/// any `k` distinct symbols have arrived, finalization reconstructs only the `r`
/// missing systematic symbols through an `r × r` Cauchy inverse. Payload bytes
/// are never eliminated on the receive path.
pub struct LazyDecoderState {
    k: usize,
    m: usize,
    n: usize,
    symbol_len: usize,
    cauchy: TowerCauchyView,
    payloads: Vec<u8>,
    received_bits: Vec<u64>,
    distinct: usize,
    received: usize,
}

/// Reusable metadata workspace for allocation-free tower reconstruction.
#[derive(Debug)]
pub struct DecodeScratch {
    k: usize,
    m: usize,
    symbol_len: usize,
    missing_data: Vec<usize>,
    present_data: Vec<usize>,
    repair_columns: Vec<usize>,
    row_variables: Vec<GfElem>,
    column_variables: Vec<GfElem>,
    present_variables: Vec<GfElem>,
    row_cross: Vec<GfElem>,
    column_cross: Vec<GfElem>,
    reciprocals: Vec<GfElem>,
    inversion_prefixes: Vec<GfElem>,
    row_factors: Vec<GfElem>,
    column_factors: Vec<GfElem>,
    inverse: Vec<GfElem>,
    present_coefficients: Vec<GfElem>,
    prefix: Vec<GfElem>,
    suffix: Vec<GfElem>,
    source_indices: Vec<usize>,
    source_coefficients: Vec<GfElem>,
}

impl DecodeScratch {
    fn new(k: usize, m: usize, symbol_len: usize) -> Self {
        let max_r = k.min(m);
        let factor_capacity = 2 * max_r + max_r * max_r + k;
        Self {
            k,
            m,
            symbol_len,
            missing_data: Vec::with_capacity(max_r),
            present_data: Vec::with_capacity(k),
            repair_columns: Vec::with_capacity(max_r),
            row_variables: Vec::with_capacity(max_r),
            column_variables: Vec::with_capacity(max_r),
            present_variables: Vec::with_capacity(k),
            row_cross: Vec::with_capacity(max_r),
            column_cross: Vec::with_capacity(max_r),
            reciprocals: Vec::with_capacity(factor_capacity),
            inversion_prefixes: Vec::with_capacity(factor_capacity),
            row_factors: Vec::with_capacity(max_r),
            column_factors: Vec::with_capacity(max_r),
            inverse: Vec::with_capacity(max_r * max_r),
            present_coefficients: Vec::with_capacity(k * max_r),
            prefix: vec![GfElem::ONE; max_r + 1],
            suffix: vec![GfElem::ONE; max_r + 1],
            source_indices: Vec::with_capacity(k),
            source_coefficients: Vec::with_capacity(k * max_r),
        }
    }
}

impl LazyDecoderState {
    /// Construct a decoder.
    ///
    /// Returns [`ConfigError`] for zero dimensions, zero or odd symbol
    /// lengths, or a codeword length exceeding the tower capacity.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        if symbol_len % 2 != 0 {
            return Err(ConfigError::OddSymbolLen);
        }
        let cap = ConfigError::TooManySymbols { cap: MAX_SYMBOLS };
        let cauchy = TowerCauchyView::new(k, m).ok_or(cap.clone())?;
        let n = k.checked_add(m).ok_or(cap.clone())?;
        let payload_len = n.checked_mul(symbol_len).ok_or(cap.clone())?;
        let payloads = zeroed_bytes(payload_len).ok_or(cap.clone())?;
        let bit_words = n.checked_add(63).ok_or(cap.clone())? / 64;
        let mut received_bits = Vec::new();
        received_bits
            .try_reserve_exact(bit_words)
            .map_err(|_| cap)?;
        received_bits.resize(bit_words, 0);
        Ok(Self {
            k,
            m,
            n,
            symbol_len,
            cauchy,
            payloads,
            received_bits,
            distinct: 0,
            received: 0,
        })
    }

    /// Number of systematic data symbols.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Total codeword length.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of distinct accepted symbols, capped at `k`.
    pub const fn rank(&self) -> usize {
        self.distinct
    }

    /// Number of pushes, including duplicates accepted before the cap.
    pub const fn received(&self) -> usize {
        self.received
    }

    /// Whether codeword position `index` has been accepted.
    pub fn has_symbol(&self, index: usize) -> bool {
        index < self.n && self.bit(index)
    }
    /// Clear receipt state for another block while retaining allocations.
    pub fn reset(&mut self) {
        self.received_bits.fill(0);
        self.distinct = 0;
        self.received = 0;
    }

    /// Validate and record one codeword symbol.
    pub fn push_symbol(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, DecodeError> {
        self.push(index, symbol)
    }

    /// Allocate reusable decode metadata for this geometry.
    pub fn decode_scratch(&self) -> DecodeScratch {
        DecodeScratch::new(self.k, self.m, self.symbol_len)
    }

    /// Reconstruct all systematic data without consuming the decoder.
    pub fn finalize_ref(&self) -> Result<Vec<u8>, DecodeError> {
        let mut output = zeroed_bytes(self.k * self.symbol_len)
            .expect("output size was validated by the constructor");
        let mut scratch = self.decode_scratch();
        self.finalize_into_with_scratch(&mut output, &mut scratch)?;
        Ok(output)
    }

    fn ensure_complete(&self) -> Result<(), DecodeError> {
        if self.distinct < self.k {
            return Err(DecodeError::InsufficientRank {
                rank: self.distinct,
                k: self.k,
            });
        }
        Ok(())
    }

    fn bit(&self, index: usize) -> bool {
        self.received_bits[index / 64] & (1u64 << (index % 64)) != 0
    }

    fn set_bit(&mut self, index: usize) {
        self.received_bits[index / 64] |= 1u64 << (index % 64);
    }

    fn finalize_into_with_scratch(
        &self,
        output: &mut [u8],
        scratch: &mut DecodeScratch,
    ) -> Result<(), DecodeError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if output.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }
        self.build_recipe_into(scratch)?;
        self.apply_recipe_into(scratch, output);
        Ok(())
    }

    fn build_recipe_into(&self, scratch: &mut DecodeScratch) -> Result<(), DecodeError> {
        if (scratch.k, scratch.m, scratch.symbol_len) != (self.k, self.m, self.symbol_len) {
            return Err(DecodeError::ScratchMismatch);
        }
        scratch.missing_data.clear();
        scratch.present_data.clear();
        for index in 0..self.k {
            if self.bit(index) {
                scratch.present_data.push(index);
            } else {
                scratch.missing_data.push(index);
            }
        }

        let r = scratch.missing_data.len();
        scratch.repair_columns.clear();
        for repair in 0..self.m {
            if self.bit(self.k + repair) {
                scratch.repair_columns.push(repair);
                if scratch.repair_columns.len() == r {
                    break;
                }
            }
        }
        if scratch.repair_columns.len() != r {
            return Err(DecodeError::InsufficientRank {
                rank: self.distinct,
                k: self.k,
            });
        }

        scratch.source_indices.clear();
        scratch.source_coefficients.clear();
        if r == 0 {
            return Ok(());
        }

        scratch.row_variables.clear();
        scratch.row_variables.extend(
            scratch
                .repair_columns
                .iter()
                .map(|&repair| self.cauchy.y_var(repair)),
        );
        scratch.column_variables.clear();
        scratch.column_variables.extend(
            scratch
                .missing_data
                .iter()
                .map(|&data| self.cauchy.x_var(data)),
        );
        scratch.present_variables.clear();
        scratch.present_variables.extend(
            scratch
                .present_data
                .iter()
                .map(|&data| self.cauchy.x_var(data)),
        );
        rational_lagrange_coefficients_into(scratch, r);

        for (repair_position, &repair) in scratch.repair_columns.iter().enumerate() {
            scratch.source_indices.push(self.k + repair);
            for missing_position in 0..r {
                scratch
                    .source_coefficients
                    .push(scratch.inverse[missing_position * r + repair_position]);
            }
        }
        for (present_position, &data) in scratch.present_data.iter().enumerate() {
            scratch.source_indices.push(data);
            let start = present_position * r;
            scratch
                .source_coefficients
                .extend_from_slice(&scratch.present_coefficients[start..start + r]);
        }
        Ok(())
    }

    fn apply_recipe_into(&self, scratch: &DecodeScratch, output: &mut [u8]) {
        let symbol_len = self.symbol_len;
        for &data in &scratch.missing_data {
            let start = data * symbol_len;
            output[start..start + symbol_len].fill(0);
        }
        for &data in &scratch.present_data {
            let start = data * symbol_len;
            output[start..start + symbol_len]
                .copy_from_slice(&self.payloads[start..start + symbol_len]);
        }
        let r = scratch.missing_data.len();
        for (missing_position, &data) in scratch.missing_data.iter().enumerate() {
            let output_start = data * symbol_len;
            let output_row = &mut output[output_start..output_start + symbol_len];
            for (term_position, &source_index) in scratch.source_indices.iter().enumerate() {
                let source_start = source_index * symbol_len;
                payload::xor_scaled_bytes(
                    output_row,
                    scratch.source_coefficients[term_position * r + missing_position],
                    &self.payloads[source_start..source_start + symbol_len],
                );
            }
        }
    }
}

impl crate::codec::Coded for LazyDecoderState {
    fn k(&self) -> usize {
        self.k
    }
    fn m(&self) -> usize {
        self.m
    }
    fn symbol_len(&self) -> usize {
        self.symbol_len
    }
    fn n(&self) -> usize {
        self.n
    }
}

impl crate::codec::Decoder for LazyDecoderState {
    type Scratch = DecodeScratch;

    fn scratch(&self) -> Self::Scratch {
        self.decode_scratch()
    }

    fn rank(&self) -> usize {
        self.distinct
    }

    fn received(&self) -> usize {
        self.received
    }

    fn reset(&mut self) {
        LazyDecoderState::reset(self);
    }

    fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        let mut scratch = self.decode_scratch();
        self.finalize_into_with_scratch(out, &mut scratch)
    }

    fn finalize_into_with(
        &mut self,
        out: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError> {
        self.finalize_into_with_scratch(out, scratch)
    }
}

impl SymbolSink for LazyDecoderState {
    fn push(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, DecodeError> {
        if index >= self.n {
            return Err(DecodeError::IndexOutOfRange { index, n: self.n });
        }
        if symbol.len() != self.symbol_len {
            return Err(DecodeError::WrongPayloadLen {
                expected: self.symbol_len,
                got: symbol.len(),
            });
        }
        if self.received >= self.n {
            return Err(DecodeError::TooManySymbols {
                cap: self.n,
                received: self.received,
            });
        }
        self.received += 1;

        if self.distinct >= self.k || self.bit(index) {
            return Ok(PushOutcome::Dependent);
        }
        let start = index * self.symbol_len;
        self.payloads[start..start + self.symbol_len].copy_from_slice(symbol);
        self.set_bit(index);
        self.distinct += 1;
        if self.distinct == self.k {
            Ok(PushOutcome::Complete)
        } else {
            Ok(PushOutcome::Advanced {
                rank: self.distinct,
                received: self.received,
            })
        }
    }

    fn is_complete(&self) -> bool {
        self.distinct >= self.k
    }

    fn finalize(self) -> Result<Vec<u8>, DecodeError> {
        self.finalize_ref()
    }
}

fn batch_invert_into(values: &mut [GfElem], prefixes: &mut Vec<GfElem>) {
    prefixes.clear();
    let mut product = GfElem::ONE;
    for &value in values.iter() {
        debug_assert_ne!(value, GfElem::ZERO, "batch inversion contains zero");
        prefixes.push(product);
        product = product.mul(value);
    }
    let mut reciprocal = product.inv();
    for index in (0..values.len()).rev() {
        let value = values[index];
        values[index] = reciprocal.mul(prefixes[index]);
        reciprocal = reciprocal.mul(value);
    }
}

/// Construct `A^-1` and fused present-column coefficients in reusable storage.
fn rational_lagrange_coefficients_into(scratch: &mut DecodeScratch, r: usize) {
    debug_assert_eq!(scratch.row_variables.len(), r);
    debug_assert_eq!(scratch.column_variables.len(), r);

    scratch.row_cross.clear();
    for &row in &scratch.row_variables {
        scratch.row_cross.push(
            scratch
                .column_variables
                .iter()
                .fold(GfElem::ONE, |product, &column| product.mul(row.add(column))),
        );
    }
    scratch.column_cross.clear();
    for &column in &scratch.column_variables {
        scratch.column_cross.push(
            scratch
                .row_variables
                .iter()
                .fold(GfElem::ONE, |product, &row| product.mul(column.add(row))),
        );
    }

    let row_within_start = 0;
    let column_within_start = r;
    let cross_start = 2 * r;
    let present_row_start = cross_start + r * r;
    scratch.reciprocals.clear();
    for (position, &row) in scratch.row_variables.iter().enumerate() {
        scratch.reciprocals.push(
            scratch
                .row_variables
                .iter()
                .enumerate()
                .filter(|&(other_position, _)| other_position != position)
                .fold(GfElem::ONE, |product, (_, &other)| {
                    product.mul(row.add(other))
                }),
        );
    }
    for (position, &column) in scratch.column_variables.iter().enumerate() {
        scratch.reciprocals.push(
            scratch
                .column_variables
                .iter()
                .enumerate()
                .filter(|&(other_position, _)| other_position != position)
                .fold(GfElem::ONE, |product, (_, &other)| {
                    product.mul(column.add(other))
                }),
        );
    }
    for &column in &scratch.column_variables {
        for &row in &scratch.row_variables {
            scratch.reciprocals.push(column.add(row));
        }
    }
    for &present in &scratch.present_variables {
        scratch.reciprocals.push(
            scratch
                .row_variables
                .iter()
                .fold(GfElem::ONE, |product, &row| product.mul(present.add(row))),
        );
    }
    batch_invert_into(&mut scratch.reciprocals, &mut scratch.inversion_prefixes);

    scratch.row_factors.clear();
    for (position, &cross) in scratch.row_cross.iter().enumerate() {
        scratch
            .row_factors
            .push(cross.mul(scratch.reciprocals[row_within_start + position]));
    }
    scratch.column_factors.clear();
    for (position, &cross) in scratch.column_cross.iter().enumerate() {
        scratch
            .column_factors
            .push(cross.mul(scratch.reciprocals[column_within_start + position]));
    }

    scratch.inverse.clear();
    for (column_position, &column_factor) in scratch.column_factors.iter().enumerate() {
        for (row_position, &row_factor) in scratch.row_factors.iter().enumerate() {
            scratch.inverse.push(
                row_factor
                    .mul(column_factor)
                    .mul(scratch.reciprocals[cross_start + column_position * r + row_position]),
            );
        }
    }

    scratch.present_coefficients.clear();
    for (present_position, &variable) in scratch.present_variables.iter().enumerate() {
        scratch.prefix[0] = GfElem::ONE;
        for position in 0..r {
            scratch.prefix[position + 1] =
                scratch.prefix[position].mul(variable.add(scratch.column_variables[position]));
        }
        scratch.suffix[r] = GfElem::ONE;
        for position in (0..r).rev() {
            scratch.suffix[position] =
                scratch.suffix[position + 1].mul(variable.add(scratch.column_variables[position]));
        }
        let row_product_inverse = scratch.reciprocals[present_row_start + present_position];
        for position in 0..r {
            let all_but_position = scratch.prefix[position].mul(scratch.suffix[position + 1]);
            scratch.present_coefficients.push(
                scratch.column_factors[position]
                    .mul(all_but_position)
                    .mul(row_product_inverse),
            );
        }
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
    use crate::tower::StreamingEncoder;

    fn data_symbols(k: usize, symbol_len: usize) -> Vec<Vec<u8>> {
        (0..k)
            .map(|row| {
                (0..symbol_len)
                    .map(|byte| (row.wrapping_mul(37) ^ byte.wrapping_mul(19) ^ 0x5a) as u8)
                    .collect()
            })
            .collect()
    }

    fn codeword(k: usize, m: usize, symbol_len: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
        let data = data_symbols(k, symbol_len);
        let expected = data.concat();
        let mut encoder = StreamingEncoder::new(k, m, symbol_len).unwrap();
        for (index, symbol) in data.iter().enumerate() {
            encoder.feed_data_symbol(index, symbol).unwrap();
        }
        let mut word = data;
        word.extend(encoder.into_repairs());
        (word, expected)
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

    #[test]
    fn exhaustive_small_k_of_n_recovery() {
        let k = 4;
        let m = 3;
        let symbol_len = 18;
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
    fn recovers_configuration_beyond_gf256_capacity() {
        let k = 256;
        let m = 3;
        let symbol_len = 4;
        let (word, expected) = codeword(k, m, symbol_len);
        let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
        for (index, symbol) in word.iter().enumerate() {
            if index != 17 && index != 255 && index != k + 2 {
                decoder.push_symbol(index, symbol).unwrap();
            }
            if decoder.is_complete() {
                break;
            }
        }
        assert_eq!(decoder.finalize_ref().unwrap(), expected);
        assert!(decoder.has_symbol(k + 1));
    }

    #[test]
    fn randomized_larger_erasure_patterns() {
        let k = 64;
        let m = 32;
        let symbol_len = 66;
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
    fn push_is_dynamic_bitmap_bookkeeping() {
        let k = 300;
        let m = 2;
        let mut decoder = LazyDecoderState::new(k, m, 2).unwrap();
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
        assert_eq!(decoder.rank(), 1);
    }
}
