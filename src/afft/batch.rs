//! Native block-final additive-FFT erasure decoder.

use std::sync::{Arc, OnceLock};

use cafft::core::kernel::{xor_scaled_bytes, xor_scaled_bytes_rows};
use cafft::rs::{
    ErasureLocator, LocatorScratch, generator_row, inverse_scratch_elements, invert_square_into,
};
use fff::field::Elem as _;

use super::crossover::{RecoveryPath, recovery_path, targeted_max_missing};
use super::decoder::systematic_locators;
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
///
/// Erasure patterns that recur are better served by
/// [`prepare_decode`](Self::prepare_decode), which hoists the whole
/// pattern-dependent solve out of the timed path;
/// [`decode_into_with`](crate::codec::BatchDecoder::decode_into_with)
/// memoizes only the most recent pattern.
#[derive(Clone, Debug)]
pub struct BatchDecoder<F: Field> {
    profile: Profile<F>,
    plan: Arc<TransformPlan<F>>,
    systematic_locator: OnceLock<Arc<ErasureLocator<F>>>,
}

/// Reusable workspace for allocation-free native AFFT batch decode.
///
/// The workspace also memoizes the pattern-dependent solve of the most recent
/// decode — the targeted generator rows and reduced inverse, or the locator
/// path's erasure locator — so repeated decodes of one erasure pattern pay
/// only validation and payload arithmetic.
#[derive(Debug)]
pub struct BatchDecodeScratch<F: Field> {
    k: usize,
    m: usize,
    symbol_len: usize,
    transform_size: usize,
    /// Largest erasure count this scratch's targeted buffers are sized for;
    /// wider patterns take the locator path by construction.
    targeted_max: usize,
    missing_data: Vec<usize>,
    repair_indices: Vec<usize>,
    /// Receipt map over the evaluation domain, doubling as the locator path's
    /// erasure map.
    known: Vec<bool>,
    /// Missing/repair lists the memoized solve below was built for; empty
    /// while nothing is memoized.
    cached_missing: Vec<usize>,
    cached_repairs: Vec<usize>,
    cached: bool,
    locator: ErasureLocator<F>,
    locator_scratch: LocatorScratch,
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
            targeted_max: 0,
            missing_data: Vec::new(),
            repair_indices: Vec::new(),
            known: Vec::new(),
            cached_missing: Vec::new(),
            cached_repairs: Vec::new(),
            cached: false,
            locator: ErasureLocator::for_domain(0),
            locator_scratch: LocatorScratch::new(),
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

    /// Whether the memoized solve matches the pattern just validated.
    fn memo_hits(&self) -> bool {
        self.cached
            && self.cached_missing == self.missing_data
            && self.cached_repairs == self.repair_indices
    }

    /// Record the pattern the freshly built solve belongs to.
    fn remember(&mut self) {
        self.cached_missing.clear();
        self.cached_missing.extend_from_slice(&self.missing_data);
        self.cached_repairs.clear();
        self.cached_repairs.extend_from_slice(&self.repair_indices);
        self.cached = true;
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

    /// Widest erasure pattern this scratch can solve with the targeted path.
    #[must_use]
    pub fn targeted_max(&self) -> usize {
        self.targeted_max
    }

    /// Missing systematic indices from the latest decode, ascending.
    #[must_use]
    pub fn missing_data(&self) -> &[usize] {
        &self.missing_data
    }

    /// Received repair codeword indices from the latest decode, ascending.
    #[must_use]
    pub fn repair_indices(&self) -> &[usize] {
        &self.repair_indices
    }

    /// Receipt map over the evaluation domain from the latest decode.
    #[must_use]
    pub fn known(&self) -> &[bool] {
        &self.known
    }

    /// Whether a pattern solve is memoized.
    #[must_use]
    pub fn has_memoized_solve(&self) -> bool {
        self.cached
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

/// A prepared additive-FFT decode plan for one erasure pattern.
///
/// Built by [`BatchDecoder::prepare_decode`]. The plan owns the entire
/// pattern-dependent solve — the targeted generator rows and reduced inverse,
/// or the locator path's erasure locator — plus every staging buffer its path
/// needs, so applying it costs symbol validation and payload arithmetic only.
/// Build one per scheduled loss pattern and keep them all resident.
#[derive(Clone, Debug)]
pub struct DecodePlan<F: Field> {
    profile: Profile<F>,
    plan: Arc<TransformPlan<F>>,
    path: RecoveryPath,
    /// Receipt map over the evaluation domain, doubling as the locator path's
    /// erasure map.
    pattern: Vec<bool>,
    /// Per-call duplicate detection, cleared on entry.
    seen: Vec<bool>,
    missing_data: Vec<usize>,
    /// Received repair codeword indices, ascending; row `i` of the reduced
    /// system belongs to `repair_indices[i]`.
    repair_indices: Vec<usize>,
    generator: Vec<F::Elem>,
    inverse: Vec<F::Elem>,
    coefficients: Vec<F::Elem>,
    /// Locator path only; the targeted path leaves this empty.
    locator: Option<ErasureLocator<F>>,
    product: Vec<u8>,
    derivative: Vec<u8>,
    residuals: Vec<u8>,
    recovered: Vec<u8>,
}

impl<F: Field> DecodePlan<F> {
    /// Systematic dimension of the geometry this plan was built for.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.profile.k
    }

    /// Repair count of the geometry this plan was built for.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.profile.m
    }

    /// Per-symbol byte length.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Number of erased systematic symbols this plan reconstructs.
    #[must_use]
    pub fn erasure_count(&self) -> usize {
        self.missing_data.len()
    }

    /// Ascending systematic indices this plan reconstructs.
    #[must_use]
    pub fn missing_indices(&self) -> &[usize] {
        &self.missing_data
    }

    /// Recovery path this plan applies.
    #[must_use]
    pub const fn path(&self) -> RecoveryPath {
        self.path
    }

    /// Decode all `k` systematic symbols into `out` (`k * symbol_len` bytes):
    /// surviving rows are copied, erased rows reconstructed.
    ///
    /// `symbols` must be exactly the `k` received symbols the plan was
    /// prepared for, in any order. Allocates nothing.
    pub fn decode_into(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let k = self.profile.k;
        let symbol_len = self.profile.symbol_len;
        if symbols.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: symbols.len(),
            });
        }
        let expected = k * symbol_len;
        if out.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: out.len(),
            });
        }
        self.seen[..self.profile.n].fill(false);
        for &(index, payload) in symbols {
            if index >= self.profile.n {
                return Err(DecodeError::IndexOutOfRange {
                    index,
                    n: self.profile.n,
                });
            }
            if !self.pattern[index] {
                return Err(DecodeError::UnexpectedIndex { index });
            }
            if self.seen[index] {
                return Err(DecodeError::DuplicateIndex { index });
            }
            self.seen[index] = true;
            if payload.len() != symbol_len {
                return Err(DecodeError::WrongPayloadLen {
                    expected: symbol_len,
                    got: payload.len(),
                });
            }
        }

        for &(index, payload) in symbols {
            if index < k {
                let start = index * symbol_len;
                out[start..start + symbol_len].copy_from_slice(payload);
            }
        }
        if self.missing_data.is_empty() {
            return Ok(());
        }
        match self.path {
            RecoveryPath::Targeted => apply_targeted::<F>(
                k,
                symbol_len,
                symbols,
                &self.missing_data,
                &self.repair_indices,
                &self.generator,
                &self.inverse,
                &mut self.coefficients,
                &mut self.residuals,
                &mut self.recovered,
                out,
            ),
            RecoveryPath::Locator => apply_locator::<F>(
                &self.plan,
                self.locator
                    .as_ref()
                    .expect("the locator path prepares a locator"),
                &self.profile,
                symbols,
                &self.missing_data,
                &mut self.product,
                &mut self.derivative,
                &mut self.recovered,
                out,
            ),
        }
        Ok(())
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<F: Field> DecodePlan<F> {
    /// Received repair codeword indices, ascending.
    #[must_use]
    pub fn repair_indices(&self) -> &[usize] {
        &self.repair_indices
    }

    /// Targeted-path generator rows, `(r, k)` row-major; empty on the locator
    /// path.
    #[must_use]
    pub fn generator(&self) -> &[F::Elem] {
        &self.generator
    }

    /// Targeted-path reduced inverse, `(r, r)` row-major; empty on the locator
    /// path.
    #[must_use]
    pub fn inverse(&self) -> &[F::Elem] {
        &self.inverse
    }

    /// The prepared erasure locator; `None` on the targeted path.
    #[must_use]
    pub fn locator(&self) -> Option<&ErasureLocator<F>> {
        self.locator.as_ref()
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

    /// Widest erasure pattern this geometry solves with the targeted dense
    /// path; wider patterns take the locator path.
    ///
    /// See [`super::crossover`] for how the threshold is derived.
    #[must_use]
    pub fn targeted_max_missing(&self) -> usize {
        targeted_max_missing::<F>(
            self.profile.k,
            self.profile.transform_size,
            self.profile.symbol_len,
        )
        .min(self.profile.m)
    }

    /// Allocate fully sized reusable batch decode scratch.
    #[must_use]
    pub fn decode_scratch(&self) -> BatchDecodeScratch<F> {
        let k = self.profile.k;
        let targeted = self.targeted_max_missing();
        let _ = self.systematic_locator();
        BatchDecodeScratch {
            k,
            m: self.profile.m,
            symbol_len: self.profile.symbol_len,
            transform_size: self.profile.transform_size,
            targeted_max: targeted,
            missing_data: Vec::with_capacity(k),
            repair_indices: Vec::with_capacity(self.profile.m),
            known: vec![false; self.profile.transform_size],
            cached_missing: Vec::with_capacity(k),
            cached_repairs: Vec::with_capacity(self.profile.m),
            cached: false,
            locator: ErasureLocator::for_domain(self.profile.transform_size),
            locator_scratch: LocatorScratch::for_domain(self.profile.transform_size),
            generator: vec![F::Elem::ZERO; targeted * k],
            system: vec![F::Elem::ZERO; targeted * targeted],
            inverse: vec![F::Elem::ZERO; targeted * targeted],
            augmented: vec![F::Elem::ZERO; inverse_scratch_elements(targeted)],
            coefficients: vec![F::Elem::ZERO; targeted],
            // The targeted path stages `r <= targeted_max` rows; the locator
            // path can need one per systematic symbol.
            residuals: vec![0u8; targeted * self.profile.symbol_len],
            recovered: vec![0u8; k * self.profile.symbol_len],
            product: vec![0u8; self.profile.transform_size * self.profile.symbol_len],
            derivative: vec![0u8; self.profile.transform_size * self.profile.symbol_len],
        }
    }

    /// Prepare a decode plan for one erasure pattern.
    ///
    /// `indices` are the codeword indices of the `k` symbols decode will be
    /// given, in any order. Validation, path selection, and the whole
    /// pattern-dependent solve happen here, once; applying the returned
    /// [`DecodePlan`] is payload arithmetic only and allocates nothing.
    pub fn prepare_decode(&self, indices: &[usize]) -> Result<DecodePlan<F>, DecodeError> {
        self.prepare_decode_inner(indices, None)
    }

    /// Prepare a decode plan that applies `path` regardless of the crossover
    /// heuristic.
    ///
    /// Both paths are correct for every erasure pattern, so this only trades
    /// performance: it exists for measuring the crossover itself and for
    /// deployments whose profile disagrees with
    /// [`targeted_max_missing`](Self::targeted_max_missing).
    pub fn prepare_decode_with_path(
        &self,
        indices: &[usize],
        path: RecoveryPath,
    ) -> Result<DecodePlan<F>, DecodeError> {
        self.prepare_decode_inner(indices, Some(path))
    }

    fn prepare_decode_inner(
        &self,
        indices: &[usize],
        forced: Option<RecoveryPath>,
    ) -> Result<DecodePlan<F>, DecodeError> {
        let (k, n) = (self.profile.k, self.profile.n);
        let symbol_len = self.profile.symbol_len;
        if indices.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: indices.len(),
            });
        }
        let mut pattern = vec![false; self.profile.transform_size];
        for &index in indices {
            if index >= n {
                return Err(DecodeError::IndexOutOfRange { index, n });
            }
            if pattern[index] {
                return Err(DecodeError::DuplicateIndex { index });
            }
            pattern[index] = true;
        }
        let missing_data: Vec<usize> = (0..k).filter(|&index| !pattern[index]).collect();
        let repair_indices: Vec<usize> = (k..n).filter(|&index| pattern[index]).collect();
        debug_assert_eq!(missing_data.len(), repair_indices.len());
        let r = missing_data.len();

        // `r <= min(k, m)` always holds, so the `m` clamp on the threshold
        // cannot change the decision here.
        let path = forced
            .unwrap_or_else(|| recovery_path::<F>(k, self.profile.transform_size, symbol_len, r));
        let mut plan = DecodePlan {
            profile: self.profile,
            plan: Arc::clone(&self.plan),
            path,
            pattern,
            seen: vec![false; self.profile.transform_size],
            missing_data,
            repair_indices,
            generator: Vec::new(),
            inverse: Vec::new(),
            coefficients: Vec::new(),
            locator: None,
            product: Vec::new(),
            derivative: Vec::new(),
            residuals: Vec::new(),
            recovered: vec![0u8; r * symbol_len],
        };
        if r == 0 {
            return Ok(plan);
        }
        match path {
            RecoveryPath::Targeted => {
                plan.generator = vec![F::Elem::ZERO; r * k];
                plan.inverse = vec![F::Elem::ZERO; r * r];
                plan.coefficients = vec![F::Elem::ZERO; r];
                plan.residuals = vec![0u8; r * symbol_len];
                let mut system = vec![F::Elem::ZERO; r * r];
                let mut augmented = vec![F::Elem::ZERO; inverse_scratch_elements(r)];
                build_targeted_system::<F>(
                    &self.plan,
                    self.systematic_locator(),
                    k,
                    &plan.missing_data,
                    &plan.repair_indices,
                    &mut plan.generator,
                    &mut system,
                    &mut augmented,
                    &mut plan.inverse,
                );
            }
            RecoveryPath::Locator => {
                let transform_bytes = self.profile.transform_size * symbol_len;
                let mut locator = ErasureLocator::for_domain(self.profile.transform_size);
                let mut locator_scratch = LocatorScratch::for_domain(self.profile.transform_size);
                locator
                    .recompute(&self.plan, &plan.pattern, &mut locator_scratch)
                    .expect("locator domain matches the profile transform");
                plan.locator = Some(locator);
                plan.product = vec![0u8; transform_bytes];
                plan.derivative = vec![0u8; transform_bytes];
            }
        }
        Ok(plan)
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

    /// Validate the offered symbols and derive the erasure pattern.
    fn validate_symbols(
        &self,
        symbols: &[(usize, &[u8])],
        scratch: &mut BatchDecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        let (k, n) = (self.profile.k, self.profile.n);
        if symbols.len() != k {
            return Err(DecodeError::WrongCount {
                expected: k,
                got: symbols.len(),
            });
        }
        scratch.known[..n].fill(false);
        for &(index, payload) in symbols {
            if index >= n {
                return Err(DecodeError::IndexOutOfRange { index, n });
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
        }
        // Both lists ascend, so the reduced system's row order is the repair
        // symbols' codeword order regardless of arrival order.
        scratch.missing_data.clear();
        scratch
            .missing_data
            .extend((0..k).filter(|&index| !scratch.known[index]));
        scratch.repair_indices.clear();
        scratch
            .repair_indices
            .extend((k..n).filter(|&index| scratch.known[index]));
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
        let symbol_len = self.profile.symbol_len;
        let expected = self.profile.k * symbol_len;
        if output.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }

        for &(index, payload) in symbols {
            if index < self.profile.k {
                let start = index * symbol_len;
                output[start..start + symbol_len].copy_from_slice(payload);
            }
        }
        let r = scratch.missing_data.len();
        if r == 0 {
            return Ok(());
        }
        // The threshold the targeted buffers were sized for is the decision:
        // a pattern wider than that cannot take the dense path.
        if r <= scratch.targeted_max {
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
        let r = scratch.missing_data.len();
        if !scratch.memo_hits() {
            build_targeted_system::<F>(
                &self.plan,
                self.systematic_locator(),
                k,
                &scratch.missing_data,
                &scratch.repair_indices,
                &mut scratch.generator[..r * k],
                &mut scratch.system[..r * r],
                &mut scratch.augmented,
                &mut scratch.inverse[..r * r],
            );
            scratch.remember();
        }
        apply_targeted::<F>(
            k,
            symbol_len,
            symbols,
            &scratch.missing_data,
            &scratch.repair_indices,
            &scratch.generator[..r * k],
            &scratch.inverse[..r * r],
            &mut scratch.coefficients[..r],
            &mut scratch.residuals[..r * symbol_len],
            &mut scratch.recovered[..r * symbol_len],
            output,
        );
    }

    fn decode_locator_into(
        &self,
        symbols: &[(usize, &[u8])],
        output: &mut [u8],
        scratch: &mut BatchDecodeScratch<F>,
    ) {
        let symbol_len = self.profile.symbol_len;
        let r = scratch.missing_data.len();
        if !scratch.memo_hits() {
            scratch
                .locator
                .recompute(&self.plan, &scratch.known, &mut scratch.locator_scratch)
                .expect("locator domain matches the profile transform");
            scratch.remember();
        }
        apply_locator::<F>(
            &self.plan,
            &scratch.locator,
            &self.profile,
            symbols,
            &scratch.missing_data,
            &mut scratch.product,
            &mut scratch.derivative,
            &mut scratch.recovered[..r * symbol_len],
            output,
        );
    }
}

/// Build the targeted path's generator rows and reduced inverse.
///
/// `generator` receives one `k`-wide row per received repair symbol
/// (`repair_indices` order), and `inverse` the inverse of that generator
/// restricted to the missing columns. Both are pure functions of the erasure
/// pattern, which is what makes them cacheable.
fn build_targeted_system<F: Field>(
    plan: &TransformPlan<F>,
    systematic_locator: &ErasureLocator<F>,
    k: usize,
    missing: &[usize],
    repair_indices: &[usize],
    generator: &mut [F::Elem],
    system: &mut [F::Elem],
    augmented: &mut [F::Elem],
    inverse: &mut [F::Elem],
) {
    let r = missing.len();
    debug_assert_eq!(repair_indices.len(), r);
    for (row, &wire_index) in repair_indices.iter().enumerate() {
        generator_row(
            plan,
            systematic_locator,
            wire_index,
            &mut generator[row * k..(row + 1) * k],
        );
    }
    for row in 0..r {
        for (column, &data_index) in missing.iter().enumerate() {
            system[row * r + column] = generator[row * k + data_index];
        }
    }
    assert!(
        invert_square_into(system, r, augmented, inverse),
        "every supported AFFT erasure pattern is invertible"
    );
}

/// Apply a prepared targeted solve to the borrowed payloads.
///
/// Folds the surviving systematic rows out of the received repair rows and
/// applies the reduced inverse, writing the recovered rows into `output`.
#[allow(clippy::too_many_arguments)]
fn apply_targeted<F: Field>(
    k: usize,
    symbol_len: usize,
    symbols: &[(usize, &[u8])],
    missing: &[usize],
    repair_indices: &[usize],
    generator: &[F::Elem],
    inverse: &[F::Elem],
    coefficients: &mut [F::Elem],
    residuals: &mut [u8],
    recovered: &mut [u8],
    output: &mut [u8],
) {
    let r = missing.len();
    for &(index, payload) in symbols {
        if index < k {
            continue;
        }
        let row = repair_indices
            .binary_search(&index)
            .expect("validated repair index is one of the prepared rows");
        residuals[row * symbol_len..(row + 1) * symbol_len].copy_from_slice(payload);
    }
    for &(index, payload) in symbols {
        if index >= k {
            continue;
        }
        for row in 0..r {
            coefficients[row] = generator[row * k + index];
        }
        xor_scaled_bytes_rows::<F>(residuals, symbol_len, coefficients, payload);
    }

    recovered.fill(0);
    for residual_row in 0..r {
        for output_row in 0..r {
            coefficients[output_row] = inverse[output_row * r + residual_row];
        }
        let source = residual_row * symbol_len;
        xor_scaled_bytes_rows::<F>(
            recovered,
            symbol_len,
            coefficients,
            &residuals[source..source + symbol_len],
        );
    }
    scatter_recovered(missing, symbol_len, recovered, output);
}

/// Apply Forney-style recovery for a prepared erasure locator.
#[allow(clippy::too_many_arguments)]
fn apply_locator<F: Field>(
    plan: &TransformPlan<F>,
    locator: &ErasureLocator<F>,
    profile: &Profile<F>,
    symbols: &[(usize, &[u8])],
    missing: &[usize],
    product: &mut [u8],
    derivative: &mut [u8],
    recovered: &mut [u8],
    output: &mut [u8],
) {
    let symbol_len = profile.symbol_len;
    // `F * Λ` over the domain: zero at the erased points by construction, so
    // only the received rows contribute and the erased ones are never read.
    product.fill(0);
    for &(index, payload) in symbols {
        let point = profile.evaluation_index(index);
        let value = locator.values()[point];
        debug_assert!(!value.is_zero());
        let start = point * symbol_len;
        xor_scaled_bytes::<F>(&mut product[start..start + symbol_len], value, payload);
    }
    plan.inverse_bytes(product, symbol_len)
        .expect("product geometry matches the transform");
    plan.derivative_bytes(product, symbol_len, derivative)
        .expect("derivative geometry matches the transform");
    plan.forward_bytes_selected(derivative, symbol_len, missing)
        .expect("missing points match the transform");

    recovered.fill(0);
    for (row, &point) in missing.iter().enumerate() {
        let source = point * symbol_len;
        let destination = row * symbol_len;
        xor_scaled_bytes::<F>(
            &mut recovered[destination..destination + symbol_len],
            locator.derivatives()[point].inv(),
            &derivative[source..source + symbol_len],
        );
    }
    scatter_recovered(missing, symbol_len, recovered, output);
}

/// Scatter contiguous recovered rows to their systematic output positions.
fn scatter_recovered(missing: &[usize], symbol_len: usize, recovered: &[u8], output: &mut [u8]) {
    for (row, &data_index) in missing.iter().enumerate() {
        let destination = data_index * symbol_len;
        output[destination..destination + symbol_len]
            .copy_from_slice(&recovered[row * symbol_len..(row + 1) * symbol_len]);
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

    struct Case {
        data: Vec<u8>,
        word: Vec<u8>,
        k: usize,
        symbol_len: usize,
    }

    impl Case {
        fn new<F: Field>(k: usize, m: usize, symbol_len: usize) -> Self {
            let data: Vec<u8> = (0..k * symbol_len)
                .map(|index| (index.wrapping_mul(131) + 7) as u8)
                .collect();
            let encoder = super::super::SystematicEncoder::<F>::new(k, m, symbol_len).unwrap();
            let mut word = data.clone();
            word.resize((k + m) * symbol_len, 0);
            let (data_rows, repairs) = word.split_at_mut(k * symbol_len);
            encoder.encode_into(data_rows, repairs).unwrap();
            Self {
                data,
                word,
                k,
                symbol_len,
            }
        }

        fn symbol(&self, index: usize) -> &[u8] {
            &self.word[index * self.symbol_len..(index + 1) * self.symbol_len]
        }

        /// Received symbols for an erasure of the first `missing` data rows,
        /// offered in reverse arrival order.
        fn received(&self, missing: usize) -> Vec<(usize, &[u8])> {
            let mut indices: Vec<usize> =
                (missing..self.k).chain(self.k..self.k + missing).collect();
            indices.reverse();
            indices
                .into_iter()
                .map(|index| (index, self.symbol(index)))
                .collect()
        }
    }

    /// Every geometry must decode identically through the scratch path and
    /// through a prepared plan, on whichever recovery path the crossover picks
    /// **and** on the one it rejects: both are mathematically total, so a
    /// mismatch is a bug in one of them rather than a tuning question.
    #[test]
    fn every_path_reconstructs_the_message() {
        fn check<F: Field>(k: usize, m: usize, symbol_len: usize) {
            let case = Case::new::<F>(k, m, symbol_len);
            let mut decoder = BatchDecoder::<F>::new(k, m, symbol_len).unwrap();
            let mut scratch = decoder.decode_scratch();
            for missing in 0..=m.min(k) {
                let received = case.received(missing);
                let indices: Vec<usize> = received.iter().map(|&(index, _)| index).collect();
                let mut output = vec![0xA5; k * symbol_len];
                decoder
                    .decode_into_with(&received, &mut output, &mut scratch)
                    .unwrap();
                assert_eq!(output, case.data, "scratch k={k} m={m} r={missing}");

                for path in [RecoveryPath::Targeted, RecoveryPath::Locator] {
                    let mut plan = decoder.prepare_decode_with_path(&indices, path).unwrap();
                    assert_eq!(plan.erasure_count(), missing);
                    let mut planned = vec![0x5A; k * symbol_len];
                    plan.decode_into(&received, &mut planned).unwrap();
                    assert_eq!(planned, case.data, "{path:?} k={k} m={m} r={missing}");
                }

                let mut plan = decoder.prepare_decode(&indices).unwrap();
                let mut planned = vec![0x5A; k * symbol_len];
                plan.decode_into(&received, &mut planned).unwrap();
                assert_eq!(planned, case.data, "planned k={k} m={m} r={missing}");
            }
        }

        check::<fff::Gf8>(16, 8, 63);
        check::<fff::Gf8>(64, 32, 257);
        check::<fff::Gf16>(16, 8, 64);
        check::<fff::Gf16>(64, 32, 258);
    }

    /// Repeated decodes of one pattern must reuse the memoized solve, and a
    /// pattern change must invalidate it rather than reconstruct from stale
    /// coefficients.
    #[test]
    fn memoized_solve_follows_the_pattern() {
        let (k, m, symbol_len) = (32usize, 16usize, 128usize);
        let case = Case::new::<fff::Gf8>(k, m, symbol_len);
        let mut decoder = Gf8BatchDecoder::new(k, m, symbol_len).unwrap();
        let mut scratch = decoder.decode_scratch();
        let mut output = vec![0u8; k * symbol_len];
        // Alternate two patterns on both sides of the crossover, twice each,
        // so a stale memo would surface as a wrong reconstruction.
        for missing in [1usize, 3, 1, 3, 12, 12, 1] {
            let received = case.received(missing);
            for _ in 0..2 {
                decoder
                    .decode_into_with(&received, &mut output, &mut scratch)
                    .unwrap();
                assert_eq!(output, case.data, "r={missing}");
            }
        }
    }

    /// Path selection must follow the geometry rather than a constant. The
    /// regime the old fixed threshold of five got wrong is the anchor: at
    /// `k64 m32 s1400` the dense solve measured ~4x faster than the locator
    /// path at eight erasures, so eight erasures must not take the transforms.
    #[test]
    fn crossover_follows_geometry() {
        let long_symbols = Gf8BatchDecoder::new(64, 32, 1400).unwrap();
        let short_symbols = Gf8BatchDecoder::new(64, 32, 64).unwrap();
        assert!(
            long_symbols.targeted_max_missing() > short_symbols.targeted_max_missing(),
            "long={} short={}",
            long_symbols.targeted_max_missing(),
            short_symbols.targeted_max_missing()
        );

        let indices: Vec<usize> = (8..64).chain(64..72).collect();
        assert_eq!(
            long_symbols.prepare_decode(&indices).unwrap().path(),
            RecoveryPath::Targeted,
            "the fixed threshold of five conceded this pattern to the transforms"
        );
        // Every data row erased is beyond any geometry's dense threshold.
        let wide = Gf8BatchDecoder::new(64, 64, 1400).unwrap();
        let indices: Vec<usize> = (64..128).collect();
        assert_eq!(
            wide.prepare_decode(&indices).unwrap().path(),
            RecoveryPath::Locator
        );
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

    #[test]
    fn prepared_plan_rejects_invalid_inputs() {
        let decoder = Gf8BatchDecoder::new(4, 2, 8).unwrap();
        let payload = [0u8; 8];
        let short = [0u8; 7];
        let mut output = [0u8; 32];

        assert_eq!(
            decoder.prepare_decode(&[0, 1]).unwrap_err(),
            DecodeError::WrongCount {
                expected: 4,
                got: 2
            }
        );
        assert_eq!(
            decoder.prepare_decode(&[0, 1, 2, 6]).unwrap_err(),
            DecodeError::IndexOutOfRange { index: 6, n: 6 }
        );
        assert_eq!(
            decoder.prepare_decode(&[0, 1, 1, 4]).unwrap_err(),
            DecodeError::DuplicateIndex { index: 1 }
        );

        // Prepared for data symbols 0 and 1 missing.
        let mut plan = decoder.prepare_decode(&[2, 3, 4, 5]).unwrap();
        let received: [(usize, &[u8]); 4] =
            [(2, &payload), (3, &payload), (4, &payload), (5, &payload)];
        assert_eq!(
            plan.decode_into(&received[..3], &mut output),
            Err(DecodeError::WrongCount {
                expected: 4,
                got: 3
            })
        );
        assert_eq!(
            plan.decode_into(&received, &mut output[..31]),
            Err(DecodeError::WrongOutputLen {
                expected: 32,
                got: 31
            })
        );
        assert_eq!(
            plan.decode_into(
                &[(0, &payload), (3, &payload), (4, &payload), (5, &payload)],
                &mut output,
            ),
            Err(DecodeError::UnexpectedIndex { index: 0 })
        );
        assert_eq!(
            plan.decode_into(
                &[(2, &payload), (2, &payload), (4, &payload), (5, &payload)],
                &mut output,
            ),
            Err(DecodeError::DuplicateIndex { index: 2 })
        );
        assert_eq!(
            plan.decode_into(
                &[(2, &short), (3, &payload), (4, &payload), (5, &payload)],
                &mut output,
            ),
            Err(DecodeError::WrongPayloadLen {
                expected: 8,
                got: 7
            })
        );
        assert_eq!(
            plan.decode_into(
                &[(2, &payload), (3, &payload), (4, &payload), (7, &payload)],
                &mut output,
            ),
            Err(DecodeError::IndexOutOfRange { index: 7, n: 6 })
        );
        // The rejected calls must not poison the plan.
        plan.decode_into(&received, &mut output).unwrap();
    }
}
