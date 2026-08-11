//! Generator rows for targeted additive-FFT recovery.

use butterfly_fft::core::transform::TransformPlan;

use super::Field;
use super::locator::ErasureLocator;

/// Coefficients expressing the evaluation at `point` as a combination of the
/// evaluations at the systematic points `0..row.len()`.
///
/// This is additive-domain Lagrange interpolation with the systematic locator
/// `Λ(x) = ∏_{d < k} (x ⊕ d_point)`:
///
/// ```text
/// row[d] = Λ(point) / ((point ⊕ d_point) · Λ'(d_point))
/// ```
///
/// Computed in exponents, so one row costs `k` table lookups and no field
/// multiplications or inversions.
///
/// # Panics
///
/// Panics unless `row.len() <= point < plan.size()` and `locator` covers the
/// plan's domain.
pub(super) fn generator_row<F: Field>(
    plan: &TransformPlan<F>,
    locator: &ErasureLocator<F>,
    point: usize,
    row: &mut [F::Elem],
) {
    assert_eq!(locator.size(), plan.size(), "locator covers another domain");
    assert!(point < plan.size(), "evaluation point out of range");
    assert!(
        row.len() <= point,
        "the evaluation point must lie outside the systematic prefix"
    );
    let tables = F::log_exp();
    let modulus = tables.order();
    let numerator = tables.log(locator.values()[point]);
    for (data, coefficient) in row.iter_mut().enumerate() {
        let difference = tables.log(plan.point_element(point ^ data));
        let derivative = tables.log(locator.derivatives()[data]);
        let exponent = (numerator + 2 * modulus - difference - derivative) % modulus;
        *coefficient = tables.exp(exponent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::field::{Elem, Field as FgfField};
    use fgf::{Gf8, Gf16};

    use super::super::locator::SystematicLocators;

    type Gf16Elem = <Gf16 as FgfField>::Elem;

    fn elem(raw: u16) -> Gf16Elem {
        let bytes = raw.to_le_bytes();
        Gf16::read(&bytes)
    }

    #[test]
    fn generator_rows_reproduce_the_systematic_encoding() {
        for (log_size, systematic) in [(3usize, 5usize), (4, 9), (5, 20), (6, 33)] {
            let size = 1usize << log_size;
            let plan = TransformPlan::<Gf16>::new(size).unwrap();
            let locators = SystematicLocators::<Gf16>::new();
            let locator = locators.get(&plan, systematic).unwrap();

            let mut evaluations = vec![Gf16Elem::ZERO; size];
            for (index, slot) in evaluations[..systematic].iter_mut().enumerate() {
                let seed = u16::try_from(index).expect("small index");
                *slot = elem(seed.wrapping_mul(7_919).wrapping_add(3));
            }
            plan.forward(&mut evaluations).unwrap();
            let data = evaluations[..systematic].to_vec();

            let mut row = vec![Gf16Elem::ZERO; systematic];
            for (point, &expected) in evaluations.iter().enumerate().skip(systematic) {
                generator_row(&plan, &locator, point, &mut row);
                let combined = row
                    .iter()
                    .zip(data.iter())
                    .fold(Gf16Elem::ZERO, |accumulator, (&coefficient, &value)| {
                        accumulator.add(coefficient.mul(value))
                    });
                assert_eq!(combined, expected, "point {point} of {size}");
            }
        }
    }

    #[test]
    fn generator_rows_work_over_gf8() {
        let plan = TransformPlan::<Gf8>::new(8).unwrap();
        let locators = SystematicLocators::<Gf8>::new();
        let locator = locators.get(&plan, 5).unwrap();
        let mut row = vec![<Gf8 as FgfField>::Elem::ZERO; 5];
        for point in 5..8 {
            generator_row(&plan, &locator, point, &mut row);
            assert!(row.iter().all(|coefficient| !coefficient.is_zero()));
        }
    }

    #[test]
    #[should_panic(expected = "outside the systematic prefix")]
    fn rejects_a_systematic_point() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let locators = SystematicLocators::<Gf16>::new();
        let locator = locators.get(&plan, 5).unwrap();
        let mut row = vec![Gf16Elem::ZERO; 5];
        generator_row(&plan, &locator, 4, &mut row);
    }
}
