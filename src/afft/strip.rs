//! Strip-blocked systematic encoding.
//!
//! A systematic RS block of `k` data rows and `m` repair rows is one
//! truncated inverse transform (data evaluations → novel-basis
//! coefficients) followed by one restricted forward transform (coefficients
//! → repair evaluations). Done naively over full-width rows, both walk a
//! `transform_size × row_len` working set and thrash L2 on every level of
//! the recursion.
//!
//! So the transform runs over *column strips*: a fixed number of bytes from
//! every row at a time, sized so one strip's working set stays inside L2.
//! The row count is the transform length either way; narrowing the rows is
//! the only free parameter. Strips also make the whole encode
//! allocation-free — one strip-sized scratch buffer serves every column.
//!
//! When `k` is a power of two and `m <= k`, the repair points are exactly
//! the domain's high coset, so the interpolation output is evaluated in
//! place through [`TransformPlan::forward_bytes_high_coset_range`]: no copy
//! into a padded high half, no full-domain strip.

use std::sync::Arc;
use std::vec::Vec;

use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::{PlanError, TransformLengthError};

use super::Field;

/// Target working sets for one column strip. Transforms below 1024 rows
/// benefit from using most of a performance core's L2; larger recursion
/// trees run faster with a narrower strip that leaves L2 capacity for
/// sibling subtrees.
const SMALL_TRANSFORM_STRIP_BYTES: usize = 768 * 1024;
const LARGE_TRANSFORM_STRIP_BYTES: usize = 512 * 1024;

/// Widest strip whose working set fits the target for its row count,
/// rounded down to whole elements and clamped to `1..=row_len`.
///
/// # Panics
/// Panics if `rows_per_strip` is zero, or if `row_len` is zero or holds a
/// partial trailing element.
#[must_use]
pub fn strip_width<F: Field>(rows_per_strip: usize, row_len: usize) -> usize {
    assert_ne!(rows_per_strip, 0, "a strip holds at least one row");
    assert_ne!(row_len, 0, "row length must be nonzero");
    assert_eq!(row_len % F::BYTES, 0, "partial trailing element");
    let target = if rows_per_strip >= 1024 {
        LARGE_TRANSFORM_STRIP_BYTES
    } else {
        SMALL_TRANSFORM_STRIP_BYTES
    };
    let elements = (target / rows_per_strip / F::BYTES).max(1);
    (elements * F::BYTES).min(row_len)
}

/// Reusable workspace for [`StripEncoder::encode`].
#[derive(Clone, Debug, Default)]
pub struct EncodeScratch {
    workspace: Vec<u8>,
}

#[cfg(test)]
impl EncodeScratch {
    /// Empty scratch, sized on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Systematic encoder over a power-of-two additive domain.
///
/// Data occupies evaluation points `0..data_points`, repairs the points
/// `data_points..data_points + repair_points`. Rows are flat byte buffers of
/// packed field elements, `row_len` bytes each; the encoder never
/// reserializes its input.
#[derive(Clone, Debug)]
pub struct StripEncoder<F: Field> {
    interpolation: Arc<TransformPlan<F>>,
    evaluation: Arc<TransformPlan<F>>,
    data_points: usize,
    repair_points: usize,
    row_len: usize,
    fused: bool,
}

impl<F: Field> StripEncoder<F> {
    /// Build an encoder for `data_points` data rows, `repair_points` repair
    /// rows, and `row_len`-byte rows.
    ///
    /// Both transform plans come from the shared plan cache.
    ///
    /// # Errors
    /// Returns [`PlanError::InvalidSize`] for zero `data_points` or
    /// `repair_points`, and [`PlanError::DomainTooLarge`] when
    /// `(data_points + repair_points).next_power_of_two()` exceeds the
    /// field's domain cap.
    ///
    /// # Panics
    /// Panics if `row_len` is zero or holds a partial trailing element, or if
    /// the total byte geometry is not representable by [`usize`].
    pub fn new(data_points: usize, repair_points: usize, row_len: usize) -> Result<Self, PlanError>
    where
        F::Elem: Send + Sync,
    {
        if data_points == 0 || repair_points == 0 {
            return Err(PlanError::InvalidSize { size: 0 });
        }
        assert_ne!(row_len, 0, "row length must be nonzero");
        assert_eq!(row_len % F::BYTES, 0, "partial trailing element");
        let total = data_points
            .checked_add(repair_points)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(PlanError::InvalidSize { size: 0 })?;
        let padded = data_points
            .checked_next_power_of_two()
            .ok_or(PlanError::InvalidSize { size: 0 })?;
        total
            .checked_mul(row_len)
            .expect("strip byte length overflow");
        let interpolation = TransformPlan::<F>::shared(padded)?;
        let evaluation = TransformPlan::<F>::shared(total)?;
        Ok(Self {
            interpolation,
            evaluation,
            data_points,
            repair_points,
            row_len,
            // The repair points are exactly the high coset only when the
            // interpolation domain is the low half and the repairs fit it.
            fused: padded == data_points && total == 2 * data_points && data_points >= 2,
        })
    }

    /// Rows one strip holds, and the temporary rows its truncated inverse
    /// needs on top.
    fn strip_geometry(&self) -> (usize, usize) {
        let rows = if self.fused {
            self.data_points
        } else {
            self.evaluation.size()
        };
        (
            rows,
            self.interpolation
                .inverse_truncated_scratch_rows(self.data_points),
        )
    }

    /// Allocate scratch for one column strip; reuse allocates nothing.
    #[must_use]
    pub fn scratch(&self) -> EncodeScratch {
        let (rows, temporaries) = self.strip_geometry();
        let width = strip_width::<F>(rows, self.row_len);
        EncodeScratch {
            workspace: vec![0u8; (rows + temporaries) * width],
        }
    }

    /// Encode `repair_points` repair rows from `data_points` data rows.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `data.len() == data_points * row_len` and
    /// `repairs.len() == repair_points * row_len`.
    pub fn encode(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut EncodeScratch,
    ) -> Result<(), TransformLengthError> {
        let (rows, _) = self.strip_geometry();
        self.encode_with_width(data, repairs, scratch, strip_width::<F>(rows, self.row_len))
    }

    /// [`StripEncoder::encode`] with an explicit strip width, for tuning and
    /// for tests that need to force many narrow strips.
    ///
    /// # Errors
    /// As [`StripEncoder::encode`].
    ///
    /// # Panics
    /// Panics if `width` is zero, holds a partial trailing element, or
    /// exceeds `row_len`.
    pub fn encode_with_width(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut EncodeScratch,
        width: usize,
    ) -> Result<(), TransformLengthError> {
        let row_len = self.row_len;
        let expected_data = self.data_points * row_len;
        if data.len() != expected_data {
            return Err(TransformLengthError {
                expected: expected_data,
                got: data.len(),
            });
        }
        let expected_repairs = self.repair_points * row_len;
        if repairs.len() != expected_repairs {
            return Err(TransformLengthError {
                expected: expected_repairs,
                got: repairs.len(),
            });
        }
        assert_ne!(width, 0, "strip width must be nonzero");
        assert_eq!(width % F::BYTES, 0, "partial trailing element");
        assert!(width <= row_len, "strip wider than a row");

        let data_points = self.data_points;
        let repair_points = self.repair_points;
        let transmitted = data_points + repair_points;
        let padded = self.interpolation.size();
        let (rows, temporaries) = self.strip_geometry();

        let workspace = &mut scratch.workspace;
        let capacity = (rows + temporaries) * width;
        if workspace.len() != capacity {
            workspace.clear();
            workspace.resize(capacity, 0);
        }

        let mut column = 0;
        while column < row_len {
            let w = width.min(row_len - column);
            let used = &mut workspace[..(rows + temporaries) * w];
            let (strip, temporary) = used.split_at_mut(rows * w);
            for row in 0..data_points {
                let source = row * row_len + column;
                strip[row * w..row * w + w].copy_from_slice(&data[source..source + w]);
            }

            if self.fused {
                // padded == data_points: the truncated inverse fills exactly
                // the strip, and the repair coset is the root's high child,
                // evaluated in place. Repairs land in the first rows.
                self.interpolation
                    .inverse_truncated_bytes(strip, w, data_points, temporary)
                    .expect("strip geometry validated");
                self.evaluation
                    .forward_bytes_high_coset_range(strip, w, 0..repair_points)
                    .expect("strip geometry validated");
                for row in 0..repair_points {
                    let destination = row * row_len + column;
                    repairs[destination..destination + w]
                        .copy_from_slice(&strip[row * w..row * w + w]);
                }
            } else {
                // Coefficient padding and the repair rows are read as zero by
                // the truncated transforms; zeroing them per strip is cheap
                // and in cache.
                strip[data_points * w..].fill(0);
                self.interpolation
                    .inverse_truncated_bytes(&mut strip[..padded * w], w, data_points, temporary)
                    .expect("strip geometry validated");
                self.evaluation
                    .forward_bytes_trunc_range(strip, w, data_points, data_points..transmitted)
                    .expect("strip geometry validated");
                for row in data_points..transmitted {
                    let destination = (row - data_points) * row_len + column;
                    repairs[destination..destination + w]
                        .copy_from_slice(&strip[row * w..row * w + w]);
                }
            }
            column += w;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::field::Elem;
    use fgf::{Gf8, Gf16};

    use super::super::generator::generator_row;
    use super::super::locator::SystematicLocators;

    fn bytes(len: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[5]
            })
            .collect()
    }

    /// Ground truth: the systematic generator matrix, built from discrete
    /// logarithms and the systematic erasure locator — additive-domain
    /// Lagrange interpolation, sharing no code with the truncated
    /// interpolation and restricted-evaluation walkers under test.
    fn reference<F: Field>(
        data_points: usize,
        repair_points: usize,
        row_len: usize,
        data: &[u8],
    ) -> Vec<u8>
    where
        F::Elem: Send + Sync,
    {
        let total = (data_points + repair_points).next_power_of_two();
        let lanes = row_len / F::BYTES;
        let plan = TransformPlan::<F>::new(total).unwrap();
        let locators = SystematicLocators::<F>::new();
        let locator = locators.get(&plan, data_points).unwrap();
        let mut coefficients = vec![F::Elem::ZERO; data_points];
        let mut repairs = vec![0u8; repair_points * row_len];
        for row in 0..repair_points {
            generator_row(&plan, &locator, data_points + row, &mut coefficients);
            for lane in 0..lanes {
                let mut value = F::Elem::ZERO;
                for (source, &coefficient) in coefficients.iter().enumerate() {
                    let start = source * row_len + lane * F::BYTES;
                    value = value.add(coefficient.mul(F::read(&data[start..start + F::BYTES])));
                }
                let start = row * row_len + lane * F::BYTES;
                F::write(&mut repairs[start..start + F::BYTES], value);
            }
        }
        repairs
    }

    #[test]
    fn repairs_match_the_element_domain_reference() {
        for (data_points, repair_points, row_len) in [
            (1usize, 1usize, 2usize),
            (2, 2, 2),
            (4, 4, 4),
            (5, 3, 2),
            (9, 7, 6),
            (17, 15, 2),
        ] {
            let data = bytes(data_points * row_len);
            let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
            let mut repairs = vec![0u8; repair_points * row_len];
            let mut scratch = encoder.scratch();
            encoder.encode(&data, &mut repairs, &mut scratch).unwrap();
            assert_eq!(
                repairs,
                reference::<Gf16>(data_points, repair_points, row_len, &data),
                "k={data_points} m={repair_points} row_len={row_len}"
            );
        }
    }

    #[test]
    fn repairs_match_over_gf8() {
        for (data_points, repair_points, row_len) in [(4usize, 4usize, 1usize), (5, 3, 3)] {
            let data = bytes(data_points * row_len);
            let encoder = StripEncoder::<Gf8>::new(data_points, repair_points, row_len).unwrap();
            let mut repairs = vec![0u8; repair_points * row_len];
            let mut scratch = encoder.scratch();
            encoder.encode(&data, &mut repairs, &mut scratch).unwrap();
            assert_eq!(
                repairs,
                reference::<Gf8>(data_points, repair_points, row_len, &data)
            );
        }
    }

    #[test]
    fn strip_blocking_is_width_invariant() {
        // Narrow strips exercise the gather/scatter and the last-strip
        // remainder that a single full-width strip never reaches.
        for (data_points, repair_points, row_len) in [
            (5usize, 3usize, 64usize),
            (100, 20, 64),
            (17, 7, 130),
            (256, 128, 40),
        ] {
            let data = bytes(data_points * row_len);
            let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
            let mut single = vec![0u8; repair_points * row_len];
            let mut scratch = EncodeScratch::new();
            encoder
                .encode_with_width(&data, &mut single, &mut scratch, row_len)
                .unwrap();
            for width in [2, 4, 6, 32] {
                let mut narrow = vec![0u8; repair_points * row_len];
                let mut scratch = EncodeScratch::new();
                encoder
                    .encode_with_width(&data, &mut narrow, &mut scratch, width)
                    .unwrap();
                assert_eq!(single, narrow, "width {width}");
            }
            let mut tuned = vec![0u8; repair_points * row_len];
            let mut scratch = encoder.scratch();
            encoder.encode(&data, &mut tuned, &mut scratch).unwrap();
            assert_eq!(single, tuned, "tuned width");
        }
    }

    #[test]
    fn fused_and_padded_paths_agree() {
        // k = 8, m = 8: fused (transform_size == 2k). Forcing the padded
        // path must produce identical repairs.
        let (data_points, repair_points, row_len) = (8usize, 8usize, 8usize);
        let data = bytes(data_points * row_len);
        let fused = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
        assert!(fused.fused);
        let mut unfused = fused.clone();
        unfused.fused = false;
        let mut expected = vec![0u8; repair_points * row_len];
        let mut scratch = fused.scratch();
        fused.encode(&data, &mut expected, &mut scratch).unwrap();
        let mut actual = vec![0u8; repair_points * row_len];
        let mut scratch = unfused.scratch();
        unfused.encode(&data, &mut actual, &mut scratch).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn rejects_invalid_geometry() {
        assert_eq!(
            StripEncoder::<Gf16>::new(0, 4, 2).unwrap_err(),
            PlanError::InvalidSize { size: 0 }
        );
        assert_eq!(
            StripEncoder::<Gf16>::new(4, 0, 2).unwrap_err(),
            PlanError::InvalidSize { size: 0 }
        );
        assert!(matches!(
            StripEncoder::<Gf8>::new(200, 200, 1).unwrap_err(),
            PlanError::DomainTooLarge { .. }
        ));
        let encoder = StripEncoder::<Gf16>::new(4, 2, 2).unwrap();
        let mut scratch = encoder.scratch();
        assert_eq!(
            encoder
                .encode(&[0; 6], &mut [0; 4], &mut scratch)
                .unwrap_err(),
            TransformLengthError {
                expected: 8,
                got: 6
            }
        );
        assert_eq!(
            encoder
                .encode(&[0; 8], &mut [0; 6], &mut scratch)
                .unwrap_err(),
            TransformLengthError {
                expected: 4,
                got: 6
            }
        );
    }

    #[test]
    fn strip_width_is_element_aligned() {
        assert_eq!(strip_width::<Gf16>(16, 2), 2);
        assert_eq!(strip_width::<Gf16>(16, 1 << 20) % 2, 0);
        assert_eq!(strip_width::<Gf8>(1 << 20, 4096), 1);
        assert_eq!(strip_width::<Gf16>(1 << 20, 4096), 2);
    }
}
