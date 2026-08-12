//! Forney-style row recovery through the novel-basis formal derivative.
//!
//! Let `F` be the codeword's evaluation function on the domain and `Λ` the
//! erasure locator. `FΛ` is known everywhere — it vanishes at exactly the
//! erased points, where `F` is unknown — so its coefficients follow from one
//! inverse transform. Differentiating the product and using `Λ(e) = 0`,
//!
//! ```text
//! (FΛ)'(e) = F'(e)·Λ(e) + F(e)·Λ'(e) = F(e)·Λ'(e)
//! ```
//!
//! so each missing evaluation is `(FΛ)'(e) / Λ'(e)`: one inverse transform,
//! one formal derivative, one forward transform restricted to the missing
//! points, and one scaling per recovered row. `Λ'(e)` is never zero, so the
//! division is total.

use std::vec::Vec;

use butterfly_fft::core::kernel::xor_scaled_bytes;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::TransformLengthError;
use fgf::field::Elem;

use super::locator::ErasureLocator;
use super::tables::RsField;

/// Byte-row workspace for [`recover_rows`]: two full-domain buffers.
#[derive(Clone, Debug, Default)]
pub struct RecoveryScratch {
    product: Vec<u8>,
    derivative: Vec<u8>,
}

impl RecoveryScratch {
    /// Empty scratch, sized on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes each of the two internal buffers needs for this geometry.
    ///
    /// # Panics
    /// Panics if `size * row_len` is not representable by [`usize`].
    #[must_use]
    pub fn required_bytes(size: usize, row_len: usize) -> usize {
        size.checked_mul(row_len)
            .expect("recovery byte length overflow")
    }

    fn ensure(&mut self, bytes: usize) {
        if self.product.len() != bytes {
            self.product.clear();
            self.product.resize(bytes, 0);
            self.derivative.clear();
            self.derivative.resize(bytes, 0);
        }
    }
}

/// Recover the erased evaluation rows named by `missing`.
///
/// `received` holds one `row_len`-byte row per transform point; rows at
/// erased points are ignored and may hold anything. `recovered` receives the
/// rows for `missing`, in that order — this module deals in domains and
/// point sets, so scattering them back to wire positions is the caller's
/// job.
///
/// `locator` must describe the same erasure pattern that `received` reflects
/// (`locator.values()[i]` nonzero exactly where a row is present); every
/// index in `missing` must be one of its erased points. The pattern may
/// erase more points than `missing` names; those are recovered internally
/// and discarded.
///
/// Allocation-free after the first call sizes `scratch` for the geometry.
///
/// # Errors
/// Returns [`TransformLengthError`] (lengths in bytes) unless
/// `received.len() == plan.size() * row_len` and
/// `recovered.len() == missing.len() * row_len`, or if the locator does not
/// cover the plan's domain (reported in points).
///
/// # Panics
/// Panics if `row_len` is zero or holds a partial trailing element, if
/// `missing` is not strictly increasing or names a point outside the domain,
/// or if a point in `missing` is not erased by `locator`.
pub fn recover_rows<F: RsField>(
    plan: &TransformPlan<F>,
    locator: &ErasureLocator<F>,
    received: &[u8],
    row_len: usize,
    missing: &[usize],
    scratch: &mut RecoveryScratch,
    recovered: &mut [u8],
) -> Result<(), TransformLengthError> {
    let size = plan.size();
    if locator.size() != size {
        return Err(TransformLengthError {
            expected: size,
            got: locator.size(),
        });
    }
    let expected = RecoveryScratch::required_bytes(size, row_len);
    if received.len() != expected {
        return Err(TransformLengthError {
            expected,
            got: received.len(),
        });
    }
    let expected_out = missing
        .len()
        .checked_mul(row_len)
        .expect("recovery byte length overflow");
    if recovered.len() != expected_out {
        return Err(TransformLengthError {
            expected: expected_out,
            got: recovered.len(),
        });
    }
    assert!(
        missing.iter().all(|&point| point < size),
        "missing point out of range"
    );
    assert!(
        missing.windows(2).all(|pair| pair[0] < pair[1]),
        "missing points must be sorted and unique"
    );
    assert!(
        missing
            .iter()
            .all(|&point| locator.values()[point].is_zero()),
        "missing point is not erased by the locator"
    );
    if missing.is_empty() {
        return Ok(());
    }
    scratch.ensure(expected);

    // FΛ over the domain: zero at erased points by construction.
    let product = &mut scratch.product[..expected];
    product.fill(0);
    for (point, &value) in locator.values().iter().enumerate() {
        if value.is_zero() {
            continue;
        }
        let start = point * row_len;
        xor_scaled_bytes::<F>(
            &mut product[start..start + row_len],
            value,
            &received[start..start + row_len],
        );
    }

    // Coefficients of FΛ, then of (FΛ)', then (FΛ)' at the missing points.
    plan.inverse_bytes(product, row_len)?;
    let derivative = &mut scratch.derivative[..expected];
    plan.derivative_bytes(product, row_len, derivative)?;
    plan.forward_bytes_selected(derivative, row_len, missing)?;

    recovered.fill(0);
    for (row, &point) in missing.iter().enumerate() {
        let source = point * row_len;
        let destination = row * row_len;
        xor_scaled_bytes::<F>(
            &mut recovered[destination..destination + row_len],
            locator.derivatives()[point].inv(),
            &derivative[source..source + row_len],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::field::Field as FgfField;
    use fgf::{Gf8, Gf16};

    use super::super::locator::LocatorScratch;

    /// A random-ish codeword: `size` evaluation rows of a novel-basis
    /// polynomial with `active` nonzero coefficient rows.
    fn codeword<F: RsField>(plan: &TransformPlan<F>, row_len: usize, active: usize) -> Vec<u8> {
        let size = plan.size();
        let mut rows = vec![0u8; size * row_len];
        let mut state = 0x1234_5678_9abc_def0u64;
        for byte in &mut rows[..active * row_len] {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = state.to_le_bytes()[5];
        }
        plan.forward_bytes(&mut rows, row_len).unwrap();
        rows
    }

    fn check_recovery<F: RsField>(log_size: usize, row_len: usize, erased: &[usize]) {
        let plan = TransformPlan::<F>::new(1 << log_size).unwrap();
        let size = plan.size();
        let active = size - erased.len();
        let original = codeword::<F>(&plan, row_len, active);

        let known: Vec<bool> = (0..size).map(|point| !erased.contains(&point)).collect();
        let mut locator = ErasureLocator::<F>::for_domain(size);
        let mut locator_scratch = LocatorScratch::new();
        locator
            .recompute(&plan, &known, &mut locator_scratch)
            .unwrap();

        // Poison the erased rows: recovery must not read them.
        let mut received = original.clone();
        for &point in erased {
            received[point * row_len..(point + 1) * row_len].fill(0xA5);
        }

        let mut scratch = RecoveryScratch::new();
        let mut recovered = vec![0u8; erased.len() * row_len];
        recover_rows(
            &plan,
            &locator,
            &received,
            row_len,
            erased,
            &mut scratch,
            &mut recovered,
        )
        .unwrap();
        for (row, &point) in erased.iter().enumerate() {
            assert_eq!(
                &recovered[row * row_len..(row + 1) * row_len],
                &original[point * row_len..(point + 1) * row_len],
                "point {point} of {size}"
            );
        }
    }

    #[test]
    fn recovers_erased_evaluations() {
        for log_size in 1..=6 {
            let size = 1usize << log_size;
            for erased_count in 1..=size / 2 {
                let prefix: Vec<usize> = (0..erased_count).collect();
                check_recovery::<Gf16>(log_size, 2, &prefix);
                let suffix: Vec<usize> = (size - erased_count..size).collect();
                check_recovery::<Gf16>(log_size, 6, &suffix);
                let strided: Vec<usize> = (0..erased_count)
                    .map(|i| i * (size / erased_count))
                    .collect();
                check_recovery::<Gf16>(log_size, 2, &strided);
            }
        }
    }

    #[test]
    fn recovers_over_gf8_and_wide_rows() {
        check_recovery::<Gf8>(3, 1, &[0, 3]);
        check_recovery::<Gf8>(4, 5, &[1, 2, 7, 15]);
        check_recovery::<Gf16>(5, 64, &[4, 9, 30]);
    }

    #[test]
    fn recovers_a_subset_of_the_erased_points() {
        // The locator erases four points; only two are requested.
        let plan = TransformPlan::<Gf16>::new(16).unwrap();
        let row_len = 4;
        let original = codeword::<Gf16>(&plan, row_len, 12);
        let erased = [1usize, 4, 9, 12];
        let known: Vec<bool> = (0..16).map(|point| !erased.contains(&point)).collect();
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let mut scratch = RecoveryScratch::new();
        let mut recovered = vec![0u8; 2 * row_len];
        recover_rows(
            &plan,
            &locator,
            &original,
            row_len,
            &[4, 12],
            &mut scratch,
            &mut recovered,
        )
        .unwrap();
        assert_eq!(&recovered[..row_len], &original[4 * row_len..5 * row_len]);
        assert_eq!(&recovered[row_len..], &original[12 * row_len..13 * row_len]);
    }

    #[test]
    fn empty_request_is_a_no_op() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let known = [true; 8];
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let mut scratch = RecoveryScratch::new();
        recover_rows(&plan, &locator, &[0u8; 16], 2, &[], &mut scratch, &mut []).unwrap();
    }

    #[test]
    fn reports_length_mismatches() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let known = [false, true, true, true, true, true, true, true];
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let mut scratch = RecoveryScratch::new();
        assert_eq!(
            recover_rows(
                &plan,
                &locator,
                &[0u8; 14],
                2,
                &[0],
                &mut scratch,
                &mut [0, 0]
            )
            .unwrap_err(),
            TransformLengthError {
                expected: 16,
                got: 14
            }
        );
        assert_eq!(
            recover_rows(
                &plan,
                &locator,
                &[0u8; 16],
                2,
                &[0],
                &mut scratch,
                &mut [0; 4]
            )
            .unwrap_err(),
            TransformLengthError {
                expected: 2,
                got: 4
            }
        );
        let wrong = ErasureLocator::<Gf16>::for_domain(4);
        assert_eq!(
            recover_rows(
                &plan,
                &wrong,
                &[0u8; 16],
                2,
                &[0],
                &mut scratch,
                &mut [0, 0]
            )
            .unwrap_err(),
            TransformLengthError {
                expected: 8,
                got: 4
            }
        );
    }

    #[test]
    #[should_panic(expected = "missing point is not erased")]
    fn rejects_a_known_point() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let known = [true; 8];
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let mut scratch = RecoveryScratch::new();
        let _ = recover_rows(
            &plan,
            &locator,
            &[0u8; 16],
            2,
            &[3],
            &mut scratch,
            &mut [0, 0],
        );
    }

    #[test]
    #[should_panic(expected = "must be sorted and unique")]
    fn rejects_unsorted_points() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let known = [true, false, true, false, true, true, true, true];
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let mut scratch = RecoveryScratch::new();
        let _ = recover_rows(
            &plan,
            &locator,
            &[0u8; 16],
            2,
            &[3, 1],
            &mut scratch,
            &mut [0; 4],
        );
    }

    #[test]
    fn locator_derivatives_are_invertible_at_erased_points() {
        // Guards the totality of the Forney division directly.
        let plan = TransformPlan::<Gf8>::new(8).unwrap();
        for pattern in 0u32..256 {
            let known: Vec<bool> = (0..8).map(|index| pattern & (1 << index) != 0).collect();
            let locator = ErasureLocator::new(&plan, &known).unwrap();
            for (index, &is_known) in known.iter().enumerate() {
                if !is_known {
                    let derivative = locator.derivatives()[index];
                    assert_ne!(derivative, <Gf8 as FgfField>::Elem::ZERO);
                    assert_eq!(
                        derivative.mul(derivative.inv()),
                        <Gf8 as FgfField>::Elem::ONE
                    );
                }
            }
        }
    }
}
