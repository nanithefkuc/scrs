//! Additive FFT over nested binary subspaces of GF(65536).
//!
//! Coefficients use the novel basis `X_i(x)` from Lin, Chung, and Han, formed
//! from normalized subspace-vanishing polynomials. This gives the recursive
//! butterfly `f = f0 + W*f1`, where normalized `W` is constant on each affine
//! half-subspace and differs by one between sibling halves.

use crate::{
    gf65536::GfElem,
    tower::payload::{self, ButterflyBackend},
};
use core::ops::Range;
use std::sync::{Arc, OnceLock};

macro_rules! dispatch_butterfly {
    ($function:ident($($argument:expr),* $(,)?)) => {
        match payload::butterfly_backend() {
            payload::ButterflyBackendKind::Scalar => {
                $function::<payload::ScalarBackend>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            payload::ButterflyBackendKind::Gfni => {
                $function::<payload::GfniBackend>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            payload::ButterflyBackendKind::Avx2 => {
                $function::<payload::Avx2Backend>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            payload::ButterflyBackendKind::Ssse3 => {
                $function::<payload::Ssse3Backend>($($argument),*)
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            payload::ButterflyBackendKind::Neon => {
                $function::<payload::NeonBackend>($($argument),*)
            }
        }
    };
}

/// Largest transform supported by GF(2^16).
pub const MAX_TRANSFORM_SIZE: usize = 65_536;

const TRANSFORM_PLAN_COUNT: usize = MAX_TRANSFORM_SIZE.trailing_zeros() as usize + 1;
static SHARED_TRANSFORM_PLANS: [OnceLock<Arc<TransformPlan>>; TRANSFORM_PLAN_COUNT] =
    [const { OnceLock::new() }; TRANSFORM_PLAN_COUNT];

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

    pub(crate) fn shared(size: usize) -> Option<Arc<Self>> {
        if size == 0 || !size.is_power_of_two() || size > MAX_TRANSFORM_SIZE {
            return None;
        }
        let index = size.trailing_zeros() as usize;
        Some(Arc::clone(SHARED_TRANSFORM_PLANS[index].get_or_init(
            || Arc::new(Self::new(size).expect("shared transform size was validated")),
        )))
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

    #[cfg(test)]
    pub(crate) fn forward_bytes(&self, rows: &mut [u8], symbol_len: usize) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        if self.log_size != 0 {
            dispatch_butterfly!(forward_bytes_node(rows, &self.factors, 1, self.log_size,));
        }
    }

    /// Evaluate only the sorted output rows in `selected`.
    pub(crate) fn forward_bytes_selected(
        &self,
        rows: &mut [u8],
        symbol_len: usize,
        selected: &[usize],
    ) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        debug_assert!(selected.iter().all(|&index| index < self.size));
        debug_assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
        if self.log_size != 0 && !selected.is_empty() {
            dispatch_butterfly!(forward_bytes_selected_node(
                rows,
                symbol_len,
                &self.factors,
                1,
                self.log_size,
                0,
                selected,
            ));
        }
    }

    /// Forward transform that exploits a leading-`active` nonzero coefficient
    /// prefix (rows `active..size` must be zero on entry) *and* restricts output
    /// to `range`.
    ///
    /// Butterflies whose entire high coefficient half lies in the zero region
    /// reduce to a copy with no field multiply, so the padding between the
    /// systematic dimension and the power-of-two domain costs no arithmetic.
    /// On the requested rows the result is identical to a full forward transform.
    pub(crate) fn forward_bytes_trunc_range(
        &self,
        rows: &mut [u8],
        symbol_len: usize,
        active: usize,
        range: Range<usize>,
    ) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        debug_assert!(range.start <= range.end);
        debug_assert!(range.end <= self.size);
        debug_assert!(active <= self.size);
        if self.log_size != 0 && !range.is_empty() && active != 0 {
            dispatch_butterfly!(forward_bytes_trunc_range_node(
                rows,
                symbol_len,
                &self.factors,
                1,
                self.log_size,
                active,
                range,
            ));
        }
    }

    /// Evaluate the `size / 2` novel-basis coefficients in `rows` at the high
    /// coset (transform points `size/2 .. size`), returning the first `range`
    /// evaluations in place.
    ///
    /// This is the forward subtree rooted at node 3 (the high child of the
    /// root). The fused encoder calls it directly on the IFFT output so the
    /// repair evaluations need neither a copy into the padded high half nor a
    /// full-domain workspace. Requires `size >= 4` (`log_size >= 2`).
    pub(crate) fn forward_bytes_high_coset_range(
        &self,
        rows: &mut [u8],
        symbol_len: usize,
        range: Range<usize>,
    ) {
        let half = self.size / 2;
        debug_assert_eq!(rows.len(), half * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        debug_assert!(range.start <= range.end);
        debug_assert!(range.end <= half);
        debug_assert!(self.log_size >= 2);
        if !range.is_empty() {
            dispatch_butterfly!(forward_bytes_range_node(
                rows,
                symbol_len,
                &self.factors,
                3,
                self.log_size - 1,
                range.start,
                range.end,
            ));
        }
    }

    pub(crate) fn inverse_bytes(&self, rows: &mut [u8], symbol_len: usize) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        if self.log_size != 0 {
            dispatch_butterfly!(inverse_bytes_node(rows, &self.factors, 1, self.log_size,));
        }
    }
    /// Temporary rows required by [`inverse_truncated_bytes`](Self::inverse_truncated_bytes).
    pub(crate) fn inverse_truncated_scratch_rows(&self, active: usize) -> usize {
        debug_assert!(active > 0 && active <= self.size);
        let mut row_count = self.size;
        while active < row_count {
            let half_rows = row_count / 2;
            if active > half_rows {
                return half_rows;
            }
            row_count = half_rows;
        }
        0
    }

    pub(crate) fn inverse_truncated_bytes(
        &self,
        rows: &mut [u8],
        symbol_len: usize,
        active: usize,
        scratch: &mut [u8],
    ) {
        debug_assert_eq!(rows.len(), self.size * symbol_len);
        debug_assert_eq!(symbol_len % 2, 0);
        debug_assert!(active > 0 && active <= self.size);
        debug_assert!(scratch.len() >= self.inverse_truncated_scratch_rows(active) * symbol_len);
        if self.log_size != 0 {
            dispatch_butterfly!(inverse_truncated_bytes_node(
                rows,
                symbol_len,
                &self.factors,
                1,
                self.log_size,
                active,
                scratch,
            ));
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

#[cfg(test)]
fn forward_bytes_node<B: ButterflyBackend>(
    rows: &mut [u8],
    factors: &[GfElem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    payload::fused_forward::<B>(low_half, high_half, factor);
    if dimension > 1 {
        forward_bytes_node::<B>(low_half, factors, node * 2, dimension - 1);
        forward_bytes_node::<B>(high_half, factors, node * 2 + 1, dimension - 1);
    }
}

fn forward_bytes_selected_node<B: ButterflyBackend>(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
    row_offset: usize,
    selected: &[usize],
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / symbol_len;
    let middle = row_offset + half_rows;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    payload::fused_forward::<B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    let split = selected.partition_point(|&index| index < middle);
    if split != 0 {
        forward_bytes_selected_node::<B>(
            low_half,
            symbol_len,
            factors,
            node * 2,
            dimension - 1,
            row_offset,
            &selected[..split],
        );
    }
    if split != selected.len() {
        forward_bytes_selected_node::<B>(
            high_half,
            symbol_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            middle,
            &selected[split..],
        );
    }
}

fn forward_bytes_range_node<B: ButterflyBackend>(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
    range_start: usize,
    range_end: usize,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / symbol_len;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    payload::fused_forward::<B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range_start < half_rows {
        forward_bytes_range_node::<B>(
            low_half,
            symbol_len,
            factors,
            node * 2,
            dimension - 1,
            range_start,
            range_end.min(half_rows),
        );
    }
    if range_end > half_rows {
        forward_bytes_range_node::<B>(
            high_half,
            symbol_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            range_start.saturating_sub(half_rows),
            range_end - half_rows,
        );
    }
}

fn forward_bytes_trunc_range_node<B: ButterflyBackend>(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
    active: usize,
    range: Range<usize>,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / symbol_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);

    if active <= half_rows {
        if dimension <= 1 {
            if range.end > half_rows {
                high_half[..active * symbol_len].copy_from_slice(&low_half[..active * symbol_len]);
            }
            return;
        }
        let need_low = range.start < half_rows;
        let need_high = range.end > half_rows;
        if need_high {
            high_half[..active * symbol_len].copy_from_slice(&low_half[..active * symbol_len]);
        }
        if need_low {
            forward_bytes_trunc_range_node::<B>(
                low_half,
                symbol_len,
                factors,
                node * 2,
                dimension - 1,
                active,
                range.start..range.end.min(half_rows),
            );
        }
        if need_high {
            forward_bytes_trunc_range_node::<B>(
                high_half,
                symbol_len,
                factors,
                node * 2 + 1,
                dimension - 1,
                active,
                range.start.saturating_sub(half_rows)..range.end - half_rows,
            );
        }
        return;
    }

    let factor = factors[node];
    payload::fused_forward::<B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range.start < half_rows {
        forward_bytes_trunc_range_node::<B>(
            low_half,
            symbol_len,
            factors,
            node * 2,
            dimension - 1,
            half_rows,
            range.start..range.end.min(half_rows),
        );
    }
    if range.end > half_rows {
        forward_bytes_trunc_range_node::<B>(
            high_half,
            symbol_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            half_rows,
            range.start.saturating_sub(half_rows)..range.end - half_rows,
        );
    }
}

fn inverse_truncated_bytes_node<B: ButterflyBackend>(
    rows: &mut [u8],
    symbol_len: usize,
    factors: &[GfElem],
    node: usize,
    dimension: usize,
    active: usize,
    scratch: &mut [u8],
) {
    let row_count = rows.len() / symbol_len;
    if row_count == 1 {
        return;
    }
    if active == row_count {
        inverse_bytes_node::<B>(rows, factors, node, dimension);
        return;
    }
    let half_rows = row_count / 2;
    let half_bytes = half_rows * symbol_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if active <= half_rows {
        inverse_truncated_bytes_node::<B>(
            low_half,
            symbol_len,
            factors,
            node * 2,
            dimension - 1,
            active,
            scratch,
        );
        return;
    }

    let right_active = active - half_rows;
    inverse_bytes_node::<B>(low_half, factors, node * 2, dimension - 1);

    let tail_start = right_active * symbol_len;
    {
        let known_tail = &mut scratch[..half_bytes];
        known_tail.fill(0);
        known_tail[tail_start..].copy_from_slice(&low_half[tail_start..]);
        forward_bytes_range_node::<B>(
            known_tail,
            symbol_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            0,
            right_active,
        );
        for (evaluation, contribution) in high_half[..tail_start]
            .iter_mut()
            .zip(&known_tail[..tail_start])
        {
            *evaluation ^= contribution;
        }
    }
    inverse_truncated_bytes_node::<B>(
        high_half,
        symbol_len,
        factors,
        node * 2 + 1,
        dimension - 1,
        right_active,
        scratch,
    );

    let factor = factors[node];
    let active_bytes = right_active * symbol_len;
    payload::fused_inverse::<B>(
        &mut low_half[..active_bytes],
        &mut high_half[..active_bytes],
        factor,
    );
}

fn inverse_bytes_node<B: ButterflyBackend>(
    rows: &mut [u8],
    factors: &[GfElem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if dimension > 1 {
        inverse_bytes_node::<B>(low_half, factors, node * 2, dimension - 1);
        inverse_bytes_node::<B>(high_half, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    payload::fused_inverse::<B>(low_half, high_half, factor);
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
    #[test]
    fn truncated_inverse_recovers_active_coefficients() {
        for size in [2, 4, 8, 16, 32] {
            let plan = TransformPlan::new(size).unwrap();
            for active in 1..=size {
                let coefficients: Vec<_> = (0..size)
                    .map(|index| {
                        if index < active {
                            GfElem((index as u16).wrapping_mul(0x1f31).wrapping_add(0x2345))
                        } else {
                            GfElem::ZERO
                        }
                    })
                    .collect();
                let mut evaluations = coefficients.clone();
                plan.forward(&mut evaluations).unwrap();
                let mut bytes = vec![0; size * 2];
                for (row, &evaluation) in evaluations.iter().enumerate().take(active) {
                    bytes[row * 2..row * 2 + 2].copy_from_slice(&evaluation.to_bytes());
                }
                let mut scratch = vec![0; plan.inverse_truncated_scratch_rows(active) * 2];
                plan.inverse_truncated_bytes(&mut bytes, 2, active, &mut scratch);
                for (row, &coefficient) in coefficients.iter().enumerate() {
                    assert_eq!(&bytes[row * 2..row * 2 + 2], &coefficient.to_bytes());
                }
            }
        }
    }

    #[test]
    fn trunc_range_forward_matches_full_forward_with_zero_padding() {
        for size in [2, 4, 8, 16, 32, 64] {
            let plan = TransformPlan::new(size).unwrap();
            for active in 1..=size {
                // Coefficients nonzero only in the leading `active` prefix.
                let mut source = vec![0u8; size * 2];
                for row in 0..active {
                    let value = GfElem((row as u16).wrapping_mul(0x2917).wrapping_add(0x1357));
                    source[row * 2..row * 2 + 2].copy_from_slice(&value.to_bytes());
                }
                let mut full = source.clone();
                plan.forward_bytes(&mut full, 2);
                for start in 0..size {
                    for end in start + 1..=size {
                        let mut ranged = source.clone();
                        plan.forward_bytes_trunc_range(&mut ranged, 2, active, start..end);
                        assert_eq!(
                            &ranged[start * 2..end * 2],
                            &full[start * 2..end * 2],
                            "size={size} active={active} range={start}..{end}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn selected_forward_matches_full_forward() {
        for size in [2, 4, 8, 16, 64] {
            let plan = TransformPlan::new(size).unwrap();
            let source: Vec<u8> = (0..size * 6)
                .map(|index| (index.wrapping_mul(137) ^ 0xa5) as u8)
                .collect();
            let mut full = source.clone();
            plan.forward_bytes(&mut full, 6);
            for stride in 1..=size {
                let selected: Vec<_> = (0..size).step_by(stride).collect();
                let mut partial = source.clone();
                plan.forward_bytes_selected(&mut partial, 6, &selected);
                for &row in &selected {
                    assert_eq!(
                        &partial[row * 6..row * 6 + 6],
                        &full[row * 6..row * 6 + 6],
                        "size={size} stride={stride} row={row}"
                    );
                }
            }
        }
    }
}
