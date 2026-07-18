//! Additive FFT over nested binary subspaces of GF(65536).
//!
//! Coefficients use the novel basis `X_i(x)` from Lin, Chung, and Han, formed
//! from normalized subspace-vanishing polynomials. This gives the recursive
//! butterfly `f = f0 + W*f1`, where normalized `W` is constant on each affine
//! half-subspace and differs by one between sibling halves.

use crate::gf65536::GfElem;

/// Largest transform supported by GF(2^16).
pub const MAX_TRANSFORM_SIZE: usize = 65_536;

/// Error returned when a transform input has the wrong length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformLengthError {
    /// Length required by the plan.
    pub expected: usize,
    /// Length supplied by the caller.
    pub got: usize,
}

/// Reusable additive-FFT plan for a power-of-two number of field elements.
///
/// Plan construction precomputes one normalized subspace-polynomial value per
/// recursive node. Forward and inverse execution each take `N log2(N)` field
/// butterflies and allocate no memory.
#[derive(Clone, Debug)]
pub struct TransformPlan {
    size: usize,
    log_size: usize,
    /// Binary-heap node layout; root is index one.
    factors: Vec<GfElem>,
    /// Derivative of each normalized subspace polynomial.
    derivative_factors: Vec<GfElem>,
}

impl TransformPlan {
    /// Construct a plan for a power-of-two size in `1..=65536`.
    pub fn new(size: usize) -> Option<Self> {
        if size == 0 || !size.is_power_of_two() || size > MAX_TRANSFORM_SIZE {
            return None;
        }
        let log_size = size.trailing_zeros() as usize;
        let polynomials = subspace_polynomials(log_size);
        let derivative_factors = polynomials
            .iter()
            .map(|polynomial| polynomial.coefficients[0].mul(polynomial.normalizer_inverse))
            .collect();
        let mut factors = vec![GfElem::ZERO; size];
        if log_size != 0 {
            fill_factors(&mut factors, &polynomials, 1, log_size, GfElem::ZERO);
        }
        Some(Self {
            size,
            log_size,
            factors,
            derivative_factors,
        })
    }

    /// Number of transform points.
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Base-two logarithm of [`TransformPlan::size`].
    pub const fn log_size(&self) -> usize {
        self.log_size
    }

    /// Evaluate novel-basis coefficients at the additive points
    /// `0, 1, ..., size-1` in binary-basis order.
    pub fn forward(&self, values: &mut [GfElem]) -> Result<(), TransformLengthError> {
        self.check_len(values.len())?;
        if self.log_size != 0 {
            forward_node(values, &self.factors, 1, self.log_size);
        }
        Ok(())
    }

    /// Convert evaluations at the plan's additive points back to novel-basis
    /// coefficients.
    pub fn inverse(&self, values: &mut [GfElem]) -> Result<(), TransformLengthError> {
        self.check_len(values.len())?;
        if self.log_size != 0 {
            inverse_node(values, &self.factors, 1, self.log_size);
        }
        Ok(())
    }

    pub(crate) fn forward_bytes(&self, rows: &mut [u8], symbol_len: usize) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        if self.log_size != 0 {
            forward_bytes_node(rows, symbol_len, &self.factors, 1, self.log_size);
        }
    }

    pub(crate) fn inverse_bytes(&self, rows: &mut [u8], symbol_len: usize) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        if self.log_size != 0 {
            inverse_bytes_node(rows, symbol_len, &self.factors, 1, self.log_size);
        }
    }

    pub(crate) fn derivative_bytes(
        &self,
        coefficients: &[u8],
        symbol_len: usize,
        derivative: &mut [u8],
    ) {
        debug_assert_eq!(coefficients.len(), self.size * symbol_len);
        debug_assert_eq!(derivative.len(), coefficients.len());
        derivative.fill(0);
        for index in 1..self.size {
            let source_start = index * symbol_len;
            let source = &coefficients[source_start..source_start + symbol_len];
            let mut remaining = index;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let destination_index = index ^ (1 << bit);
                let destination_start = destination_index * symbol_len;
                crate::tower::payload::xor_scaled_bytes(
                    &mut derivative[destination_start..destination_start + symbol_len],
                    self.derivative_factors[bit],
                    source,
                );
                remaining &= remaining - 1;
            }
        }
    }

    fn check_len(&self, got: usize) -> Result<(), TransformLengthError> {
        if got == self.size {
            Ok(())
        } else {
            Err(TransformLengthError {
                expected: self.size,
                got,
            })
        }
    }
}

#[derive(Clone, Debug)]
struct NormalizedSubspacePolynomial {
    /// Coefficients of x^(2^j), low exponent first.
    coefficients: Vec<GfElem>,
    normalizer_inverse: GfElem,
}

impl NormalizedSubspacePolynomial {
    fn evaluate(&self, value: GfElem) -> GfElem {
        let mut power = value;
        let mut result = GfElem::ZERO;
        for &coefficient in &self.coefficients {
            result = result.add(coefficient.mul(power));
            power = power.square();
        }
        result.mul(self.normalizer_inverse)
    }
}

fn subspace_polynomials(count: usize) -> Vec<NormalizedSubspacePolynomial> {
    let mut result = Vec::with_capacity(count);
    let mut coefficients = vec![GfElem::ONE]; // W_0(x) = x
    for dimension in 0..count {
        let basis = GfElem(1u16 << dimension);
        let normalizer = evaluate_linearized(&coefficients, basis);
        debug_assert_ne!(normalizer, GfElem::ZERO);
        result.push(NormalizedSubspacePolynomial {
            coefficients: coefficients.clone(),
            normalizer_inverse: normalizer.inv(),
        });

        if dimension + 1 != count {
            // W_{i+1}(x) = W_i(x)^2 + W_i(beta_i) W_i(x).
            let mut next = vec![GfElem::ZERO; coefficients.len() + 1];
            for (index, &coefficient) in coefficients.iter().enumerate() {
                next[index] = next[index].add(normalizer.mul(coefficient));
                next[index + 1] = next[index + 1].add(coefficient.square());
            }
            coefficients = next;
        }
    }
    result
}

fn evaluate_linearized(coefficients: &[GfElem], value: GfElem) -> GfElem {
    let mut power = value;
    let mut result = GfElem::ZERO;
    for &coefficient in coefficients {
        result = result.add(coefficient.mul(power));
        power = power.square();
    }
    result
}

fn fill_factors(
    factors: &mut [GfElem],
    polynomials: &[NormalizedSubspacePolynomial],
    node: usize,
    dimension: usize,
    shift: GfElem,
) {
    factors[node] = polynomials[dimension - 1].evaluate(shift);
    if dimension > 1 {
        fill_factors(factors, polynomials, node * 2, dimension - 1, shift);
        fill_factors(
            factors,
            polynomials,
            node * 2 + 1,
            dimension - 1,
            shift.add(GfElem(1u16 << (dimension - 1))),
        );
    }
}

fn forward_node(values: &mut [GfElem], factors: &[GfElem], node: usize, dimension: usize) {
    let half = values.len() / 2;
    let factor = factors[node];
    for position in 0..half {
        let low = values[position];
        let high = values[half + position];
        let left = low.add(factor.mul(high));
        values[position] = left;
        values[half + position] = left.add(high);
    }
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        forward_node(left, factors, node * 2, dimension - 1);
        forward_node(right, factors, node * 2 + 1, dimension - 1);
    }
}

fn inverse_node(values: &mut [GfElem], factors: &[GfElem], node: usize, dimension: usize) {
    let half = values.len() / 2;
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        inverse_node(left, factors, node * 2, dimension - 1);
        inverse_node(right, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    for position in 0..half {
        let left = values[position];
        let right = values[half + position];
        let high = left.add(right);
        values[position] = left.add(factor.mul(high));
        values[half + position] = high;
    }
}

fn forward_bytes_node(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    for position in 0..low_half.len() / symbol_len {
        let start = position * symbol_len;
        let low = &mut low_half[start..start + symbol_len];
        let high = &mut high_half[start..start + symbol_len];
        crate::tower::payload::xor_scaled_bytes(low, factor, high);
        crate::tower::payload::xor_scaled_bytes(high, GfElem::ONE, low);
    }
    if dimension > 1 {
        forward_bytes_node(low_half, symbol_len, factors, node * 2, dimension - 1);
        forward_bytes_node(high_half, symbol_len, factors, node * 2 + 1, dimension - 1);
    }
}

fn inverse_bytes_node(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if dimension > 1 {
        inverse_bytes_node(low_half, symbol_len, factors, node * 2, dimension - 1);
        inverse_bytes_node(high_half, symbol_len, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    for position in 0..low_half.len() / symbol_len {
        let start = position * symbol_len;
        let low = &mut low_half[start..start + symbol_len];
        let high = &mut high_half[start..start + symbol_len];
        crate::tower::payload::xor_scaled_bytes(high, GfElem::ONE, low);
        crate::tower::payload::xor_scaled_bytes(low, factor, high);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_evaluate(coefficients: &[GfElem], point: GfElem) -> GfElem {
        let polynomials = subspace_polynomials(coefficients.len().trailing_zeros() as usize);
        let normalized: Vec<_> = polynomials
            .iter()
            .map(|poly| poly.evaluate(point))
            .collect();
        let mut result = GfElem::ZERO;
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let mut basis_value = GfElem::ONE;
            for (bit, &value) in normalized.iter().enumerate() {
                if index & (1 << bit) != 0 {
                    basis_value = basis_value.mul(value);
                }
            }
            result = result.add(coefficient.mul(basis_value));
        }
        result
    }

    #[test]
    fn validates_transform_sizes_and_lengths() {
        assert!(TransformPlan::new(0).is_none());
        assert!(TransformPlan::new(3).is_none());
        assert!(TransformPlan::new(MAX_TRANSFORM_SIZE).is_some());
        let plan = TransformPlan::new(8).unwrap();
        assert_eq!(
            plan.forward(&mut [GfElem::ZERO; 4]),
            Err(TransformLengthError {
                expected: 8,
                got: 4
            })
        );
    }

    #[test]
    fn transform_matches_direct_novel_basis_evaluation() {
        for size in [1, 2, 4, 8, 16, 32] {
            let coefficients: Vec<_> = (0..size)
                .map(|index| GfElem((index as u16).wrapping_mul(0x2917).wrapping_add(0x1357)))
                .collect();
            let expected: Vec<_> = (0..size)
                .map(|point| direct_evaluate(&coefficients, GfElem(point as u16)))
                .collect();
            let mut actual = coefficients;
            TransformPlan::new(size)
                .unwrap()
                .forward(&mut actual)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn forward_inverse_roundtrips() {
        let mut state = 0x1234_5678u32;
        for size in [1, 2, 4, 8, 16, 64, 256, 1024] {
            let plan = TransformPlan::new(size).unwrap();
            let mut values: Vec<_> = (0..size)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    GfElem(state as u16)
                })
                .collect();
            let expected = values.clone();
            plan.forward(&mut values).unwrap();
            plan.inverse(&mut values).unwrap();
            assert_eq!(values, expected);
        }
    }
}
