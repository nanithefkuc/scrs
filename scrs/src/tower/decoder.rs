//! Payload-lazy reduced Tower Cauchy decoder.

use crate::gf65536::GfElem;
use crate::stream::{PushOutcome, StreamError, SymbolSink};

use super::{TowerCauchyView, cauchy::batch_invert, payload};

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

impl LazyDecoderState {
    /// Construct a decoder.
    ///
    /// Returns `None` for invalid dimensions, zero or odd symbol lengths, size
    /// overflow, or allocation failure.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let cauchy = TowerCauchyView::new(k, m)?;
        if symbol_len == 0 || symbol_len % 2 != 0 {
            return None;
        }
        let n = k.checked_add(m)?;
        let payload_len = n.checked_mul(symbol_len)?;
        let payloads = zeroed_bytes(payload_len)?;
        let bit_words = n.checked_add(63)? / 64;
        let mut received_bits = Vec::new();
        received_bits.try_reserve_exact(bit_words).ok()?;
        received_bits.resize(bit_words, 0);
        Some(Self {
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

    /// Validate and record one codeword symbol.
    pub fn push_symbol(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, StreamError> {
        self.push(index, symbol)
    }

    /// Reconstruct all systematic data without consuming the decoder.
    pub fn finalize_ref(&self) -> Result<Vec<u8>, StreamError> {
        self.ensure_complete()?;
        let recipe = self.build_recipe()?;
        let mut output = zeroed_bytes(self.k * self.symbol_len)
            .expect("output size was validated by the constructor");
        self.apply_recipe_into(&recipe, &mut output);
        Ok(output)
    }

    /// Reconstruct into a caller-provided `k * symbol_len` byte buffer.
    ///
    /// On success every output byte is overwritten. On error the output may be
    /// partially modified.
    pub fn finalize_into(&self, output: &mut [u8]) -> Result<(), StreamError> {
        self.ensure_complete()?;
        let expected = self.k * self.symbol_len;
        if output.len() != expected {
            return Err(StreamError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }
        let recipe = self.build_recipe()?;
        self.apply_recipe_into(&recipe, output);
        Ok(())
    }

    fn ensure_complete(&self) -> Result<(), StreamError> {
        if self.distinct < self.k {
            return Err(StreamError::InsufficientRank {
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

    fn build_recipe(&self) -> Result<ReconstructionRecipe, StreamError> {
        let mut missing_data = Vec::new();
        let mut present_data = Vec::new();
        for index in 0..self.k {
            if self.bit(index) {
                present_data.push(index);
            } else {
                missing_data.push(index);
            }
        }

        let r = missing_data.len();
        let mut repair_columns = Vec::with_capacity(r);
        for repair in 0..self.m {
            if self.bit(self.k + repair) {
                repair_columns.push(repair);
                if repair_columns.len() == r {
                    break;
                }
            }
        }
        if repair_columns.len() != r {
            return Err(StreamError::InsufficientRank {
                rank: self.distinct,
                k: self.k,
            });
        }
        if r == 0 {
            return Ok(ReconstructionRecipe {
                missing_data,
                present_data,
                source_terms: Vec::new(),
            });
        }

        let row_variables: Vec<_> = repair_columns
            .iter()
            .map(|&repair| self.cauchy.y_var(repair))
            .collect();
        let column_variables: Vec<_> = missing_data
            .iter()
            .map(|&data| self.cauchy.x_var(data))
            .collect();
        let present_variables: Vec<_> = present_data
            .iter()
            .map(|&data| self.cauchy.x_var(data))
            .collect();
        let factors =
            rational_lagrange_coefficients(&row_variables, &column_variables, &present_variables);

        let mut source_terms = Vec::with_capacity(self.k);
        for (repair_position, &repair) in repair_columns.iter().enumerate() {
            let coefficients = (0..r)
                .map(|missing_position| factors.inverse[missing_position * r + repair_position])
                .collect();
            source_terms.push(SourceTerm {
                source_index: self.k + repair,
                coefficients,
            });
        }
        for (present_position, &data) in present_data.iter().enumerate() {
            let coefficients = (0..r)
                .map(|missing_position| factors.present[present_position * r + missing_position])
                .collect();
            source_terms.push(SourceTerm {
                source_index: data,
                coefficients,
            });
        }

        Ok(ReconstructionRecipe {
            missing_data,
            present_data,
            source_terms,
        })
    }

    fn apply_recipe_into(&self, recipe: &ReconstructionRecipe, output: &mut [u8]) {
        let symbol_len = self.symbol_len;
        for &data in &recipe.missing_data {
            let start = data * symbol_len;
            output[start..start + symbol_len].fill(0);
        }
        for &data in &recipe.present_data {
            let start = data * symbol_len;
            output[start..start + symbol_len]
                .copy_from_slice(&self.payloads[start..start + symbol_len]);
        }
        for (missing_position, &data) in recipe.missing_data.iter().enumerate() {
            let output_start = data * symbol_len;
            let output_row = &mut output[output_start..output_start + symbol_len];
            for term in &recipe.source_terms {
                let source_start = term.source_index * symbol_len;
                payload::xor_scaled_bytes(
                    output_row,
                    term.coefficients[missing_position],
                    &self.payloads[source_start..source_start + symbol_len],
                );
            }
        }
    }
}

impl SymbolSink for LazyDecoderState {
    fn push(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, StreamError> {
        if index >= self.n {
            return Err(StreamError::IndexOutOfRange { index, n: self.n });
        }
        if symbol.len() != self.symbol_len {
            return Err(StreamError::WrongPayloadLen {
                expected: self.symbol_len,
                got: symbol.len(),
            });
        }
        if self.received >= self.n {
            return Err(StreamError::TooManySymbols {
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

    fn finalize(self) -> Result<Vec<u8>, StreamError> {
        self.finalize_ref()
    }
}

struct ReconstructionRecipe {
    missing_data: Vec<usize>,
    present_data: Vec<usize>,
    source_terms: Vec<SourceTerm>,
}

struct SourceTerm {
    source_index: usize,
    coefficients: Vec<GfElem>,
}

struct RationalLagrangeCoefficients {
    inverse: Vec<GfElem>,
    present: Vec<GfElem>,
}

/// Construct `A^-1` and fused present-column coefficients in
/// `O(r^2 + r*(k-r))` field operations.
fn rational_lagrange_coefficients(
    row_variables: &[GfElem],
    column_variables: &[GfElem],
    present_variables: &[GfElem],
) -> RationalLagrangeCoefficients {
    debug_assert_eq!(row_variables.len(), column_variables.len());
    let r = row_variables.len();
    if r == 0 {
        return RationalLagrangeCoefficients {
            inverse: Vec::new(),
            present: Vec::new(),
        };
    }

    let row_cross: Vec<_> = row_variables
        .iter()
        .map(|&row| {
            column_variables
                .iter()
                .fold(GfElem::ONE, |product, &column| product.mul(row.add(column)))
        })
        .collect();
    let column_cross: Vec<_> = column_variables
        .iter()
        .map(|&column| {
            row_variables
                .iter()
                .fold(GfElem::ONE, |product, &row| product.mul(column.add(row)))
        })
        .collect();

    let row_within_start = 0;
    let column_within_start = r;
    let cross_start = 2 * r;
    let present_row_start = cross_start + r * r;
    let mut reciprocals = Vec::with_capacity(present_row_start + present_variables.len());
    for (position, &row) in row_variables.iter().enumerate() {
        reciprocals.push(
            row_variables
                .iter()
                .enumerate()
                .filter(|&(other_position, _)| other_position != position)
                .fold(GfElem::ONE, |product, (_, &other)| {
                    product.mul(row.add(other))
                }),
        );
    }
    for (position, &column) in column_variables.iter().enumerate() {
        reciprocals.push(
            column_variables
                .iter()
                .enumerate()
                .filter(|&(other_position, _)| other_position != position)
                .fold(GfElem::ONE, |product, (_, &other)| {
                    product.mul(column.add(other))
                }),
        );
    }
    for &column in column_variables {
        for &row in row_variables {
            reciprocals.push(column.add(row));
        }
    }
    for &present in present_variables {
        reciprocals.push(
            row_variables
                .iter()
                .fold(GfElem::ONE, |product, &row| product.mul(present.add(row))),
        );
    }
    batch_invert(&mut reciprocals);

    let row_factors: Vec<_> = row_cross
        .iter()
        .enumerate()
        .map(|(position, &cross)| cross.mul(reciprocals[row_within_start + position]))
        .collect();
    let column_factors: Vec<_> = column_cross
        .iter()
        .enumerate()
        .map(|(position, &cross)| cross.mul(reciprocals[column_within_start + position]))
        .collect();

    let mut inverse = Vec::with_capacity(r * r);
    for (column_position, &column_factor) in column_factors.iter().enumerate() {
        for (row_position, &row_factor) in row_factors.iter().enumerate() {
            inverse.push(
                row_factor
                    .mul(column_factor)
                    .mul(reciprocals[cross_start + column_position * r + row_position]),
            );
        }
    }

    let mut present = Vec::with_capacity(present_variables.len() * r);
    let mut prefix = vec![GfElem::ONE; r + 1];
    let mut suffix = vec![GfElem::ONE; r + 1];
    for (present_position, &variable) in present_variables.iter().enumerate() {
        for position in 0..r {
            prefix[position + 1] = prefix[position].mul(variable.add(column_variables[position]));
        }
        for position in (0..r).rev() {
            suffix[position] = suffix[position + 1].mul(variable.add(column_variables[position]));
        }
        let row_product_inverse = reciprocals[present_row_start + present_position];
        for position in 0..r {
            let all_but_position = prefix[position].mul(suffix[position + 1]);
            present.push(
                column_factors[position]
                    .mul(all_but_position)
                    .mul(row_product_inverse),
            );
        }
    }

    RationalLagrangeCoefficients { inverse, present }
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
