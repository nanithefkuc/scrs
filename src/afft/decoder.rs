//! Payload-lazy additive-FFT erasure decoder.

use std::sync::{Arc, LazyLock, OnceLock};

use cafft::core::kernel::xor_scaled_bytes_rows;
use cafft::rs::{
    ErasureLocator, LocatorScratch, RecoveryScratch, SystematicLocators, generator_row,
    inverse_scratch_elements, invert_square_into, recover_rows,
};

use fff::field::Elem as _;

use crate::codec::{Coded, Decoder};
use crate::error::{ConfigError, DecodeError};
use crate::stream::{PushOutcome, SymbolSink};

use super::profile::{Profile, zeroed_bytes};
use super::{Field, TransformPlan};

/// Erasure counts at or below this use the targeted dense solve instead of the
/// full locator path.
///
/// The locator path costs three domain-sized transforms regardless of how many
/// symbols are missing, so for a handful of erasures an `r x r` dense solve
/// against the repair rows is far cheaper. cafft deliberately leaves this
/// crossover to the consumer.
const TARGETED_MAX_MISSING: usize = 5;

/// Locators for the fixed systematic point set, shared process-wide.
///
/// These depend only on `(k, plan)` and not on the erasure pattern, so every
/// decoder with the same geometry reuses one. cafft's cache is internally
/// locked and evicts at 32 entries.
static GF8_LOCATORS: LazyLock<SystematicLocators<fff::Gf8>> =
    LazyLock::new(SystematicLocators::new);
static GF16_LOCATORS: LazyLock<SystematicLocators<fff::Gf16>> =
    LazyLock::new(SystematicLocators::new);

/// The shared systematic-locator cache for `F`.
///
/// `SystematicLocators` is generic but a `static` cannot be, so each supported
/// field gets one and this resolves between them by type.
fn systematic_locators<F: Field>() -> &'static SystematicLocators<F> {
    let any: &dyn core::any::Any = if core::any::TypeId::of::<F>() == core::any::TypeId::of::<fff::Gf8>()
    {
        &*GF8_LOCATORS
    } else {
        &*GF16_LOCATORS
    };
    any.downcast_ref()
        .expect("afft::Field is implemented only for Gf8 and Gf16")
}

/// Lazy erasure decoder for [`super::SystematicEncoder`].
///
/// Receipt processing only validates, copies, and marks a dynamic bitmap. All
/// transform work is deferred to finalization, which takes one of two paths:
/// a dense solve against the received repair rows when at most
/// [`TARGETED_MAX_MISSING`] symbols are missing, otherwise cafft's Forney-style
/// locator recovery.
#[derive(Clone, Debug)]
pub struct LazyDecoderState<F: Field> {
    profile: Profile<F>,
    /// The evaluation-domain plan, resolved once from the shared cache.
    plan: Arc<TransformPlan<F>>,
    payloads: Vec<u8>,
    received_bits: Vec<u64>,
    distinct: usize,
    received: usize,
    /// Locator for this geometry's systematic point set, resolved once.
    ///
    /// `SystematicLocators::get` builds its cache key by collecting the plan
    /// basis into a `Vec`, so it allocates on every call — including hits.
    /// Holding the `Arc` keeps the targeted finalize path allocation-free.
    systematic_locator: OnceLock<Arc<ErasureLocator<F>>>,
}

/// Reusable workspace for allocation-free additive-FFT decoding.
#[derive(Debug)]
pub struct DecodeScratch<F: Field> {
    k: usize,
    m: usize,
    symbol_len: usize,
    transform_size: usize,
    missing_data: Vec<usize>,
    /// Full path: erasure map over the evaluation domain, plus cafft's locator
    /// and recovery workspaces and the domain-sized received/recovered buffers.
    known: Vec<bool>,
    locator: ErasureLocator<F>,
    locator_scratch: LocatorScratch,
    recovery: RecoveryScratch,
    recovered: Vec<u8>,
    /// Targeted path: the `r x k` generator rows, the `r x r` system and its
    /// inverse, and the residual rows.
    repair_indices: Vec<usize>,
    generator: Vec<F::Elem>,
    system: Vec<F::Elem>,
    inverse: Vec<F::Elem>,
    augmented: Vec<F::Elem>,
    coefficients: Vec<F::Elem>,
    residuals: Vec<u8>,
}

impl<F: Field> DecodeScratch<F> {
    /// Create empty scratch that is sized on first use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: 0,
            m: 0,
            symbol_len: 0,
            transform_size: 0,
            missing_data: Vec::new(),
            known: Vec::new(),
            locator: ErasureLocator::for_domain(0),
            locator_scratch: LocatorScratch::new(),
            recovery: RecoveryScratch::new(),
            recovered: Vec::new(),
            repair_indices: Vec::new(),
            generator: Vec::new(),
            system: Vec::new(),
            inverse: Vec::new(),
            augmented: Vec::new(),
            coefficients: Vec::new(),
            residuals: Vec::new(),
        }
    }
}

impl<F: Field> Default for DecodeScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> LazyDecoderState<F> {
    /// Construct a decoder matching an additive-FFT encoder configuration.
    ///
    /// Fails with the same [`ConfigError`] variants as
    /// [`SystematicEncoder::new`](super::SystematicEncoder::new).
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
        // Sized to the evaluation domain, not to `n`: the locator path hands
        // this buffer straight to cafft as the received-rows domain, so the
        // padding rows past `n` cost memory but save a full-domain copy and
        // zeroing on every finalize. They are always erased, hence never read.
        let payloads = zeroed_bytes(profile.transform_size * symbol_len)
            .ok_or(ConfigError::TooManySymbols { cap })?;
        let bit_words = profile.n.div_ceil(64);
        let mut received_bits = Vec::new();
        received_bits
            .try_reserve_exact(bit_words)
            .map_err(|_| ConfigError::TooManySymbols { cap })?;
        received_bits.resize(bit_words, 0);
        let plan = TransformPlan::shared(profile.transform_size)
            .map_err(|_| ConfigError::TooManySymbols { cap })?;
        Ok(Self {
            profile,
            plan,
            payloads,
            received_bits,
            distinct: 0,
            received: 0,
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

    /// Power-of-two plan size used for truncated systematic interpolation.
    #[must_use]
    pub const fn padded_k(&self) -> usize {
        self.profile.padded_k
    }

    /// Full power-of-two transform domain size.
    #[must_use]
    pub const fn transform_size(&self) -> usize {
        self.profile.transform_size
    }

    /// Number of distinct accepted symbols, capped at `k`.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.distinct
    }

    /// Total pushes, including duplicates.
    #[must_use]
    pub const fn received(&self) -> usize {
        self.received
    }

    /// Whether a transmitted codeword position has been accepted.
    #[must_use]
    pub fn has_symbol(&self, index: usize) -> bool {
        index < self.profile.n && self.bit(index)
    }

    /// Clear receipt state for another block while retaining allocations.
    pub fn reset(&mut self) {
        self.received_bits.fill(0);
        self.distinct = 0;
        self.received = 0;
    }

    /// Validate and record one systematic or repair symbol.
    pub fn push_symbol(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, DecodeError> {
        self.push(index, symbol)
    }

    /// Reconstruct every systematic symbol without consuming the decoder.
    pub fn finalize_ref(&self) -> Result<Vec<u8>, DecodeError> {
        self.ensure_complete()?;
        let mut output = zeroed_bytes(self.profile.k * self.profile.symbol_len)
            .expect("profile validated output allocation size");
        let mut scratch = self.decode_scratch();
        self.finalize_complete_into(&mut output, &mut scratch)?;
        Ok(output)
    }

    /// Allocate fully sized reusable decode scratch.
    ///
    /// Every buffer either path can need is sized here, so steady-state
    /// finalization allocates nothing regardless of which path an erasure
    /// pattern selects.
    #[must_use]
    pub fn decode_scratch(&self) -> DecodeScratch<F> {
        let k = self.profile.k;
        let symbol_len = self.profile.symbol_len;
        let transform_size = self.profile.transform_size;
        let targeted = TARGETED_MAX_MISSING;

        // Resolve the systematic locator now so the first targeted finalize does
        // not build it under the allocation-free contract.
        let _ = self.systematic_locator();

        DecodeScratch {
            k,
            m: self.profile.m,
            symbol_len,
            transform_size,
            missing_data: Vec::with_capacity(k),
            known: vec![false; transform_size],
            locator: ErasureLocator::for_domain(transform_size),
            locator_scratch: LocatorScratch::for_domain(transform_size),
            // Grown by the path that runs, not pre-sized for the widest case:
            // a single-erasure finalize must not pay for domain-sized buffers
            // it never reads.
            recovery: RecoveryScratch::new(),
            recovered: Vec::new(),
            repair_indices: Vec::with_capacity(targeted),
            generator: vec![F::Elem::ZERO; targeted * k],
            system: vec![F::Elem::ZERO; targeted * targeted],
            inverse: vec![F::Elem::ZERO; targeted * targeted],
            augmented: vec![F::Elem::ZERO; inverse_scratch_elements(targeted)],
            coefficients: vec![F::Elem::ZERO; targeted],
            residuals: Vec::new(),
        }
    }

    fn ensure_decode_scratch(&self, scratch: &mut DecodeScratch<F>) -> Result<(), DecodeError> {
        if scratch.k == 0 {
            *scratch = self.decode_scratch();
            return Ok(());
        }
        let geometry = (
            self.profile.k,
            self.profile.m,
            self.profile.symbol_len,
            self.profile.transform_size,
        );
        if (
            scratch.k,
            scratch.m,
            scratch.symbol_len,
            scratch.transform_size,
        ) != geometry
        {
            return Err(DecodeError::ScratchMismatch);
        }
        Ok(())
    }

    /// The locator for this geometry's systematic point set.
    ///
    /// Resolved once per decoder from the process-wide cache, then held, so the
    /// targeted finalize path never pays the cache lookup's key allocation.
    fn systematic_locator(&self) -> &Arc<ErasureLocator<F>> {
        self.systematic_locator.get_or_init(|| {
            systematic_locators::<F>()
                .get(&self.plan, self.profile.k)
                .expect("systematic locator domain matches the profile transform")
        })
    }

    fn finalize_complete_into(
        &self,
        output: &mut [u8],
        scratch: &mut DecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        self.ensure_decode_scratch(scratch)?;
        let symbol_len = self.profile.symbol_len;
        scratch.missing_data.clear();
        for data in 0..self.profile.k {
            let start = data * symbol_len;
            if self.bit(data) {
                output[start..start + symbol_len]
                    .copy_from_slice(&self.payloads[start..start + symbol_len]);
            } else {
                output[start..start + symbol_len].fill(0);
                scratch.missing_data.push(data);
            }
        }
        if scratch.missing_data.is_empty() {
            return Ok(());
        }
        if scratch.missing_data.len() <= TARGETED_MAX_MISSING {
            return self.finalize_targeted_into(output, scratch);
        }
        self.finalize_locator_into(output, scratch)
    }

    /// Forney-style recovery over the whole evaluation domain.
    ///
    /// Fixed cost of three domain-sized transforms, so this is the path for
    /// erasure counts where the dense solve would be worse.
    fn finalize_locator_into(
        &self,
        output: &mut [u8],
        scratch: &mut DecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        let symbol_len = self.profile.symbol_len;
        let plan = &self.plan;
        let DecodeScratch {
            missing_data,
            known,
            locator,
            locator_scratch,
            recovery,
            recovered,
            ..
        } = scratch;

        // Domain points beyond `n` were never transmitted and count as erased.
        known.fill(false);
        for wire_index in 0..self.profile.n {
            if self.bit(wire_index) {
                known[self.profile.evaluation_index(wire_index)] = true;
            }
        }

        locator
            .recompute(plan, known, locator_scratch)
            .expect("locator domain matches the profile transform");

        let recovered = fit(recovered, missing_data.len() * symbol_len);
        recover_rows(
            plan,
            locator,
            &self.payloads,
            symbol_len,
            missing_data,
            recovery,
            recovered,
        )
        .expect("recovery geometry matches the profile");

        for (row, &data) in missing_data.iter().enumerate() {
            let destination = data * symbol_len;
            output[destination..destination + symbol_len]
                .copy_from_slice(&recovered[row * symbol_len..(row + 1) * symbol_len]);
        }
        Ok(())
    }

    /// Dense `r x r` solve against the received repair rows.
    ///
    /// Builds the generator rows for the repair points actually received,
    /// inverts the submatrix at the missing columns, folds the surviving data
    /// out of the repairs, and applies the inverse. Cost scales with `r`, not
    /// with the domain, which is why it wins for small `r`.
    fn finalize_targeted_into(
        &self,
        output: &mut [u8],
        scratch: &mut DecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        let k = self.profile.k;
        let symbol_len = self.profile.symbol_len;
        let missing_count = scratch.missing_data.len();
        let plan = &self.plan;

        scratch.repair_indices.clear();
        scratch
            .repair_indices
            .extend((k..self.profile.n).filter(|&wire_index| self.bit(wire_index)));
        debug_assert_eq!(scratch.repair_indices.len(), missing_count);

        let locator = self.systematic_locator();

        let generator = &mut scratch.generator[..missing_count * k];
        for (row, &wire_index) in scratch.repair_indices.iter().enumerate() {
            let point = self.profile.evaluation_index(wire_index);
            generator_row(
                plan,
                &locator,
                point,
                &mut generator[row * k..(row + 1) * k],
            );
        }

        let system = &mut scratch.system[..missing_count * missing_count];
        for row in 0..missing_count {
            for (column, &data_index) in scratch.missing_data.iter().enumerate() {
                system[row * missing_count + column] = generator[row * k + data_index];
            }
        }
        let inverse = &mut scratch.inverse[..missing_count * missing_count];
        assert!(
            invert_square_into(system, missing_count, &mut scratch.augmented, inverse),
            "every supported AFFT erasure pattern is invertible"
        );

        let residuals = fit(&mut scratch.residuals, missing_count * symbol_len);
        for (row, &wire_index) in scratch.repair_indices.iter().enumerate() {
            let source = wire_index * symbol_len;
            residuals[row * symbol_len..(row + 1) * symbol_len]
                .copy_from_slice(&self.payloads[source..source + symbol_len]);
        }
        let coefficients = &mut scratch.coefficients[..missing_count];
        for data_index in 0..k {
            if !self.bit(data_index) {
                continue;
            }
            for row in 0..missing_count {
                coefficients[row] = generator[row * k + data_index];
            }
            let source = data_index * symbol_len;
            xor_scaled_bytes_rows::<F>(
                residuals,
                symbol_len,
                coefficients,
                &self.payloads[source..source + symbol_len],
            );
        }

        let recovered = fit(&mut scratch.recovered, missing_count * symbol_len);
        for residual_row in 0..missing_count {
            for output_row in 0..missing_count {
                coefficients[output_row] = inverse[output_row * missing_count + residual_row];
            }
            let source = residual_row * symbol_len;
            xor_scaled_bytes_rows::<F>(
                recovered,
                symbol_len,
                coefficients,
                &residuals[source..source + symbol_len],
            );
        }
        for (row, &data_index) in scratch.missing_data.iter().enumerate() {
            let destination = data_index * symbol_len;
            output[destination..destination + symbol_len]
                .copy_from_slice(&recovered[row * symbol_len..(row + 1) * symbol_len]);
        }
        Ok(())
    }

    fn ensure_complete(&self) -> Result<(), DecodeError> {
        if self.distinct < self.profile.k {
            Err(DecodeError::InsufficientRank {
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

impl<F: Field> SymbolSink for LazyDecoderState<F> {
    fn push(&mut self, index: usize, symbol: &[u8]) -> Result<PushOutcome, DecodeError> {
        if index >= self.profile.n {
            return Err(DecodeError::IndexOutOfRange {
                index,
                n: self.profile.n,
            });
        }
        if symbol.len() != self.profile.symbol_len {
            return Err(DecodeError::WrongPayloadLen {
                expected: self.profile.symbol_len,
                got: symbol.len(),
            });
        }
        if self.received >= self.profile.n {
            return Err(DecodeError::TooManySymbols {
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

    fn finalize(self) -> Result<Vec<u8>, DecodeError> {
        self.finalize_ref()
    }
}

impl<F: Field> Coded for LazyDecoderState<F> {
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

impl<F: Field> Decoder for LazyDecoderState<F> {
    type Scratch = DecodeScratch<F>;

    fn scratch(&self) -> DecodeScratch<F> {
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

    /// Reconstruct into a caller-provided `k * symbol_len` buffer, allocating a
    /// throwaway transform scratch per call.
    fn finalize_into(&mut self, output: &mut [u8]) -> Result<(), DecodeError> {
        let mut scratch = DecodeScratch::new();
        self.finalize_into_with(output, &mut scratch)
    }

    /// Reconstruct into a caller-provided buffer using reusable scratch.
    ///
    /// After the first sizing call the workspaces are reused without
    /// reallocation.
    fn finalize_into_with(
        &mut self,
        output: &mut [u8],
        scratch: &mut DecodeScratch<F>,
    ) -> Result<(), DecodeError> {
        self.ensure_complete()?;
        let expected = self.profile.k * self.profile.symbol_len;
        if output.len() != expected {
            return Err(DecodeError::WrongOutputLen {
                expected,
                got: output.len(),
            });
        }
        self.finalize_complete_into(output, scratch)
    }
}

/// Resize `buffer` to exactly `len` zeroed bytes, reusing its capacity.
///
/// Mirrors what the decode paths need: a buffer sized to the work at hand
/// rather than to the widest case, so a single-erasure finalize never pays for
/// a domain-sized allocation it will not read.
fn fit(buffer: &mut Vec<u8>, len: usize) -> &mut [u8] {
    buffer.clear();
    buffer.resize(len, 0);
    &mut buffer[..]
}
