//! Erasure-locator evaluation over a transform domain.
//!
//! For an erasure set `E` the locator is `Λ(x) = ∏_{e ∈ E} (x ⊕ e_point)`.
//! A decoder needs `Λ` at every *known* point and `Λ'` at every *erased*
//! point. Evaluated naively that is `|E| · size` multiplies; evaluated in
//! exponents it is an XOR-convolution, and an XOR-convolution over a
//! GF(2)-subspace domain is a Walsh–Hadamard transform.
//!
//! Because the domain is the span of an ordered basis, `point(i) ⊕ point(j)
//! = point(i ⊕ j)`: differences of domain points stay in the domain and
//! depend only on the index XOR. So with
//! `L[i] = log(point(i))` and `1_E` the erased indicator,
//!
//! ```text
//! (1_E ⋆ L)[x] = Σ_e L[x ⊕ e] = Σ_e log(point(x) ⊕ point(e)) = log Λ(point(x))
//! ```
//!
//! and three length-`size` Walsh–Hadamard transforms modulo `|F*|` deliver
//! every evaluation. Since only differences of points appear, the result is
//! identical for a shifted plan: pass
//! [`butterfly_fft::shifted::ShiftedPlan::plan`].
//!
//! At an erased index the self-term `log(point(0))` contributes the table's
//! sentinel zero, so the same convolution yields
//! `Λ'(e) = ∏_{e' ≠ e} (e ⊕ e')` there — the formal derivative of a
//! characteristic-two product is its self-excluding product.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use fgf::field::Elem;

use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::TransformLengthError;

use super::tables::{RsField, element_bits, mod_pow, mul_mod};

/// Locator evaluations over one transform domain and one erasure pattern.
///
/// Exactly one of the two tables is populated at each index: `values` at
/// known points, `derivatives` at erased points. The other holds
/// [`Elem::ZERO`], which is also the true value of `Λ` at an erased point.
#[derive(Clone, Debug)]
pub struct ErasureLocator<F: RsField> {
    values: Vec<F::Elem>,
    derivatives: Vec<F::Elem>,
}

/// Exponent-domain workspace for [`ErasureLocator::recompute`].
#[derive(Clone, Debug, Default)]
pub struct LocatorScratch {
    indicator: Vec<u32>,
    logarithms: Vec<u32>,
}

impl LocatorScratch {
    /// Empty scratch, sized on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scratch sized for a `size`-point domain; reuse allocates nothing.
    #[must_use]
    pub fn for_domain(size: usize) -> Self {
        Self {
            indicator: vec![0; size],
            logarithms: vec![0; size],
        }
    }

    fn ensure(&mut self, size: usize) {
        if self.indicator.len() != size {
            self.indicator.clear();
            self.indicator.resize(size, 0);
            self.logarithms.clear();
            self.logarithms.resize(size, 0);
        }
    }
}

impl<F: RsField> ErasureLocator<F> {
    /// Zeroed locator tables for a `size`-point domain, to be filled by
    /// [`ErasureLocator::recompute`].
    #[must_use]
    pub fn for_domain(size: usize) -> Self {
        Self {
            values: vec![F::Elem::ZERO; size],
            derivatives: vec![F::Elem::ZERO; size],
        }
    }

    /// Locator evaluations for the erasure pattern `known`, allocating.
    ///
    /// `known[i]` is whether the evaluation at transform point `i` survived.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in points) unless
    /// `known.len() == plan.size()`.
    pub fn new(plan: &TransformPlan<F>, known: &[bool]) -> Result<Self, TransformLengthError> {
        let mut locator = Self::for_domain(plan.size());
        let mut scratch = LocatorScratch::for_domain(plan.size());
        locator.recompute(plan, known, &mut scratch)?;
        Ok(locator)
    }

    /// Recompute the tables in place for a new erasure pattern.
    ///
    /// Allocation-free once `self` and `scratch` are sized for the domain
    /// (see [`ErasureLocator::for_domain`] and
    /// [`LocatorScratch::for_domain`]).
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in points) unless
    /// `known.len() == plan.size()`.
    ///
    /// # Panics
    /// Panics if the plan's dimension does not fit a [`u32`], which no
    /// constructible plan reaches.
    pub fn recompute(
        &mut self,
        plan: &TransformPlan<F>,
        known: &[bool],
        scratch: &mut LocatorScratch,
    ) -> Result<(), TransformLengthError> {
        let size = plan.size();
        if known.len() != size {
            return Err(TransformLengthError {
                expected: size,
                got: known.len(),
            });
        }
        if self.values.len() != size {
            *self = Self::for_domain(size);
        }
        scratch.ensure(size);

        let tables = F::log_exp();
        let modulus = tables.order();
        let indicator = &mut scratch.indicator[..size];
        let logarithms = &mut scratch.logarithms[..size];
        for (index, (slot, logarithm)) in
            indicator.iter_mut().zip(logarithms.iter_mut()).enumerate()
        {
            *slot = u32::from(!known[index]);
            *logarithm = tables.log(plan.point_element(index));
        }

        walsh_hadamard(indicator, modulus);
        walsh_hadamard(logarithms, modulus);
        for (slot, &logarithm) in indicator.iter_mut().zip(logarithms.iter()) {
            *slot = mul_mod(*slot, logarithm, modulus);
        }
        walsh_hadamard(indicator, modulus);
        // Undo the `size` gain of the three transforms. Two is invertible
        // modulo the odd `2^BITS - 1`, with inverse `2^(BITS-1)`.
        let inverse_two = modulus / 2 + 1;
        let log_size = u32::try_from(plan.log_size()).expect("a plan dimension fits u32");
        let inverse_size = mod_pow(inverse_two, log_size, modulus);
        for slot in indicator.iter_mut() {
            *slot = mul_mod(*slot, inverse_size, modulus);
        }

        self.values.fill(F::Elem::ZERO);
        self.derivatives.fill(F::Elem::ZERO);
        for (index, &is_known) in known.iter().enumerate() {
            let product = tables.exp(indicator[index]);
            if is_known {
                self.values[index] = product;
            } else {
                self.derivatives[index] = product;
            }
        }
        Ok(())
    }

    /// `Λ(point(i))` at every known point; [`Elem::ZERO`] at erased points,
    /// which is also the mathematical value there.
    #[must_use]
    pub fn values(&self) -> &[F::Elem] {
        &self.values
    }

    /// `Λ'(point(i))` at every erased point; [`Elem::ZERO`] elsewhere.
    ///
    /// Never zero at an erased point: the self-excluding product of distinct
    /// domain points has no vanishing factor. That is what makes the Forney
    /// division in [`super::recovery::recover_rows`] total.
    #[must_use]
    pub fn derivatives(&self) -> &[F::Elem] {
        &self.derivatives
    }

    /// Number of transform points covered.
    #[must_use]
    pub fn size(&self) -> usize {
        self.values.len()
    }
}

/// Walsh–Hadamard transform over `Z_modulus`, in place.
///
/// The additive characters of a GF(2)-index domain: this is the XOR
/// convolution's diagonalizing transform. Applied three times (twice
/// forward, pointwise product, once more) it convolves, up to a factor of
/// the length.
///
/// # Panics
/// Panics unless `values.len()` is a power of two.
pub fn walsh_hadamard(values: &mut [u32], modulus: u32) {
    assert!(
        values.len().is_power_of_two(),
        "Walsh-Hadamard length must be a power of two"
    );
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(half * 2) {
            let (low, high) = block.split_at_mut(half);
            for (left, right) in low.iter_mut().zip(high.iter_mut()) {
                let sum = *left + *right;
                let difference = if *left >= *right {
                    *left - *right
                } else {
                    *left + modulus - *right
                };
                *left = if sum >= modulus { sum - modulus } else { sum };
                *right = difference;
            }
        }
        half *= 2;
    }
}

const SYSTEMATIC_LOCATOR_CAPACITY: usize = 32;

type LocatorKey = (usize, Vec<u64>);

/// Cache of locators for the *systematic* erasure pattern: points
/// `0..systematic` erased, the rest known.
///
/// That pattern depends only on the domain and the systematic count, not on
/// which symbols were actually lost, so every decoder sharing a code
/// geometry shares one locator. Keyed by systematic count and the plan's
/// ordered-basis prefix, so plans over different bases never collide.
///
/// Insertion-ordered eviction at 32 entries. Owned rather than
/// process-global: a locator is two `size`-element tables, and a decoder
/// that goes away should take its tables with it.
#[derive(Debug)]
pub struct SystematicLocators<F: RsField> {
    entries: Mutex<Cache<F>>,
}

#[derive(Debug)]
struct Cache<F: RsField> {
    map: HashMap<LocatorKey, Arc<ErasureLocator<F>>>,
    order: VecDeque<LocatorKey>,
}

impl<F: RsField> Default for SystematicLocators<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: RsField> SystematicLocators<F> {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Cache {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// The locator for the pattern "points `0..systematic` erased", built on
    /// first request for this `(systematic, basis)` pair.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in points) unless
    /// `systematic <= plan.size()`.
    ///
    /// # Panics
    /// Panics if the cache mutex is poisoned by an unwind inside a
    /// concurrent build — the guard is recovered, so this is unreachable in
    /// practice.
    pub fn get(
        &self,
        plan: &TransformPlan<F>,
        systematic: usize,
    ) -> Result<Arc<ErasureLocator<F>>, TransformLengthError> {
        if systematic > plan.size() {
            return Err(TransformLengthError {
                expected: plan.size(),
                got: systematic,
            });
        }
        let key: LocatorKey = (
            systematic,
            plan.basis()
                .iter()
                .map(|&element| element_bits::<F>(element))
                .collect(),
        );
        {
            let cache = self.lock();
            if let Some(locator) = cache.map.get(&key) {
                return Ok(Arc::clone(locator));
            }
        }
        let known: Vec<bool> = (0..plan.size()).map(|index| index >= systematic).collect();
        let computed = Arc::new(ErasureLocator::new(plan, &known)?);

        let mut cache = self.lock();
        if let Some(locator) = cache.map.get(&key) {
            return Ok(Arc::clone(locator));
        }
        if cache.map.len() == SYSTEMATIC_LOCATOR_CAPACITY {
            let evicted = cache
                .order
                .pop_front()
                .expect("a full locator cache has an insertion");
            cache.map.remove(&evicted);
        }
        cache.order.push_back(key.clone());
        cache.map.insert(key, Arc::clone(&computed));
        Ok(computed)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Cache<F>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::{Gf8, Gf16};

    /// Direct `∏_{e ∈ E, e ≠ x} (point(x) ⊕ point(e))` — the definition,
    /// with no logarithms and no convolution.
    fn naive<F: RsField>(plan: &TransformPlan<F>, known: &[bool]) -> (Vec<F::Elem>, Vec<F::Elem>) {
        let size = plan.size();
        let mut values = vec![F::Elem::ZERO; size];
        let mut derivatives = vec![F::Elem::ZERO; size];
        for point in 0..size {
            let mut product = F::Elem::ONE;
            for (erased, &is_known) in known.iter().enumerate() {
                if is_known || erased == point {
                    continue;
                }
                product = product.mul(plan.point_element(point ^ erased));
            }
            if known[point] {
                values[point] = product;
            } else {
                derivatives[point] = product;
            }
        }
        (values, derivatives)
    }

    fn check_pattern<F: RsField>(plan: &TransformPlan<F>, known: &[bool]) {
        let locator = ErasureLocator::new(plan, known).unwrap();
        let (values, derivatives) = naive(plan, known);
        assert_eq!(locator.values(), &values[..], "locator values");
        assert_eq!(
            locator.derivatives(),
            &derivatives[..],
            "locator derivatives"
        );
        for (index, &is_known) in known.iter().enumerate() {
            if is_known {
                assert_ne!(locator.values()[index], F::Elem::ZERO);
                assert_eq!(locator.derivatives()[index], F::Elem::ZERO);
            } else {
                assert_eq!(locator.values()[index], F::Elem::ZERO);
                assert_ne!(locator.derivatives()[index], F::Elem::ZERO);
            }
        }
    }

    #[test]
    fn matches_the_product_definition() {
        for log_size in 0..=6 {
            let plan = TransformPlan::<Gf16>::new(1 << log_size).unwrap();
            let size = plan.size();
            // Prefix erasures, suffix erasures, strided, all-known.
            for erased_count in 0..=size {
                let known: Vec<bool> = (0..size).map(|index| index >= erased_count).collect();
                check_pattern(&plan, &known);
                let known: Vec<bool> = (0..size).map(|index| index < size - erased_count).collect();
                check_pattern(&plan, &known);
            }
            let known: Vec<bool> = (0..size).map(|index| index % 3 != 0).collect();
            check_pattern(&plan, &known);
        }
    }

    #[test]
    fn matches_the_product_definition_over_gf8_and_custom_bases() {
        let plan = TransformPlan::<Gf8>::new(8).unwrap();
        let known: Vec<bool> = (0..8).map(|index| index % 2 == 0).collect();
        check_pattern(&plan, &known);

        // A reversed bit basis: point order changes, the identity does not.
        let mut basis = plan.basis().to_vec();
        basis.reverse();
        let reversed = TransformPlan::<Gf8>::with_basis(8, &basis).unwrap();
        check_pattern(&reversed, &known);

        // A non-bit basis over Gf16.
        let basis = [
            fgf::gf16::Elem(0x1234),
            fgf::gf16::Elem(0x0108),
            fgf::gf16::Elem(0xabcd),
        ];
        let plan = TransformPlan::<Gf16>::with_basis(8, &basis).unwrap();
        let known: Vec<bool> = (0..8).map(|index| index != 2 && index != 5).collect();
        check_pattern(&plan, &known);
    }

    #[test]
    fn shift_invariance() {
        // The locator only sees differences, so a coset plan agrees with the
        // unshifted plan it was derived from.
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let shifted =
            butterfly_fft::shifted::ShiftedPlan::<Gf16>::new(8, fgf::gf16::Elem(0x9e3f)).unwrap();
        let known: Vec<bool> = (0..8).map(|index| index % 4 != 1).collect();
        let direct = ErasureLocator::new(&plan, &known).unwrap();
        let coset = ErasureLocator::new(shifted.plan(), &known).unwrap();
        assert_eq!(direct.values(), coset.values());
        assert_eq!(direct.derivatives(), coset.derivatives());
    }

    #[test]
    fn recompute_reuses_scratch() {
        let plan = TransformPlan::<Gf16>::new(16).unwrap();
        let mut locator = ErasureLocator::<Gf16>::for_domain(16);
        let mut scratch = LocatorScratch::for_domain(16);
        for erased_count in 1..16 {
            let known: Vec<bool> = (0..16).map(|index| index >= erased_count).collect();
            locator.recompute(&plan, &known, &mut scratch).unwrap();
            let fresh = ErasureLocator::new(&plan, &known).unwrap();
            assert_eq!(locator.values(), fresh.values());
            assert_eq!(locator.derivatives(), fresh.derivatives());
        }
    }

    #[test]
    fn rejects_a_mismatched_pattern() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let error = ErasureLocator::new(&plan, &[true; 7]).unwrap_err();
        assert_eq!(
            error,
            TransformLengthError {
                expected: 8,
                got: 7
            }
        );
    }

    #[test]
    fn systematic_cache_returns_the_same_locator() {
        let plan = TransformPlan::<Gf16>::new(16).unwrap();
        let cache = SystematicLocators::<Gf16>::new();
        let first = cache.get(&plan, 5).unwrap();
        let second = cache.get(&plan, 5).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let known: Vec<bool> = (0..16).map(|index| index >= 5).collect();
        let expected = ErasureLocator::new(&plan, &known).unwrap();
        assert_eq!(first.values(), expected.values());
        assert_eq!(first.derivatives(), expected.derivatives());

        // A different basis over the same size is a different entry.
        let mut basis = plan.basis().to_vec();
        basis.reverse();
        let reversed = TransformPlan::<Gf16>::with_basis(16, &basis).unwrap();
        let other = cache.get(&reversed, 5).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        assert_ne!(first.derivatives(), other.derivatives());

        assert_eq!(
            cache.get(&plan, 17).unwrap_err(),
            TransformLengthError {
                expected: 16,
                got: 17
            }
        );
    }

    #[test]
    fn systematic_cache_evicts_in_insertion_order() {
        let plan = TransformPlan::<Gf16>::new(64).unwrap();
        let cache = SystematicLocators::<Gf16>::new();
        let first = cache.get(&plan, 1).unwrap();
        for systematic in 2..=SYSTEMATIC_LOCATOR_CAPACITY {
            cache.get(&plan, systematic).unwrap();
        }
        assert_eq!(cache.lock().map.len(), SYSTEMATIC_LOCATOR_CAPACITY);
        cache.get(&plan, SYSTEMATIC_LOCATOR_CAPACITY + 1).unwrap();
        assert_eq!(cache.lock().map.len(), SYSTEMATIC_LOCATOR_CAPACITY);
        // The oldest entry was dropped, so it is rebuilt as a fresh Arc.
        assert!(!Arc::ptr_eq(&first, &cache.get(&plan, 1).unwrap()));
    }

    #[test]
    fn walsh_hadamard_is_its_own_inverse_up_to_length() {
        let modulus = 65_535u32;
        let mut values: Vec<u32> = (0..16).map(|index| index * 4_097 % modulus).collect();
        let original = values.clone();
        walsh_hadamard(&mut values, modulus);
        walsh_hadamard(&mut values, modulus);
        let inverse_size = mod_pow(modulus / 2 + 1, 4, modulus);
        for (value, expected) in values.iter().zip(original.iter()) {
            assert_eq!(mul_mod(*value, inverse_size, modulus), *expected);
        }
    }
}
