//! Unstable implementation APIs for experimentation and downstream tuning.
//!
//! This module is available only with the `internals` feature. Its contents are
//! not covered by SCRS's public compatibility guarantees and may change between
//! patch releases.

/// Additive-FFT planning and byte-transform internals.
pub mod afft {
    use std::ops::Range;
    use std::sync::Arc;

    use crate::afft::TransformPlan;
    pub use crate::afft::profile::{Profile, zeroed_bytes};

    /// Return the process-wide transform plan for `size`.
    #[must_use]
    pub fn shared_transform_plan(size: usize) -> Option<Arc<TransformPlan>> {
        TransformPlan::shared(size)
    }

    /// Access byte-oriented additive-FFT operations used by the AFFT codec.
    pub trait TransformPlanExt {
        /// Evaluate interleaved byte rows at every transform point.
        fn forward_bytes(&self, rows: &mut [u8], symbol_len: usize);
        /// Evaluate only the sorted transform rows in `selected`.
        fn forward_bytes_selected(&self, rows: &mut [u8], symbol_len: usize, selected: &[usize]);
        /// Evaluate a zero-padded coefficient prefix over an output range.
        fn forward_bytes_trunc_range(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            active: usize,
            range: Range<usize>,
        );
        /// Evaluate a half-size coefficient block over the high affine coset.
        fn forward_bytes_high_coset_range(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            range: Range<usize>,
        );
        /// Convert interleaved evaluations to novel-basis coefficients.
        fn inverse_bytes(&self, rows: &mut [u8], symbol_len: usize);
        /// Return temporary rows required by a truncated inverse transform.
        fn inverse_truncated_scratch_rows(&self, active: usize) -> usize;
        /// Convert an active evaluation prefix using caller-provided scratch.
        fn inverse_truncated_bytes(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            active: usize,
            scratch: &mut [u8],
        );
        /// Differentiate interleaved novel-basis coefficients.
        fn derivative_bytes(&self, coefficients: &[u8], symbol_len: usize, derivative: &mut [u8]);
    }

    impl TransformPlanExt for TransformPlan {
        fn forward_bytes(&self, rows: &mut [u8], symbol_len: usize) {
            TransformPlan::forward_bytes(self, rows, symbol_len);
        }

        fn forward_bytes_selected(&self, rows: &mut [u8], symbol_len: usize, selected: &[usize]) {
            TransformPlan::forward_bytes_selected(self, rows, symbol_len, selected);
        }

        fn forward_bytes_trunc_range(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            active: usize,
            range: Range<usize>,
        ) {
            TransformPlan::forward_bytes_trunc_range(self, rows, symbol_len, active, range);
        }

        fn forward_bytes_high_coset_range(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            range: Range<usize>,
        ) {
            TransformPlan::forward_bytes_high_coset_range(self, rows, symbol_len, range);
        }

        fn inverse_bytes(&self, rows: &mut [u8], symbol_len: usize) {
            TransformPlan::inverse_bytes(self, rows, symbol_len);
        }

        fn inverse_truncated_scratch_rows(&self, active: usize) -> usize {
            TransformPlan::inverse_truncated_scratch_rows(self, active)
        }

        fn inverse_truncated_bytes(
            &self,
            rows: &mut [u8],
            symbol_len: usize,
            active: usize,
            scratch: &mut [u8],
        ) {
            TransformPlan::inverse_truncated_bytes(self, rows, symbol_len, active, scratch);
        }

        fn derivative_bytes(&self, coefficients: &[u8], symbol_len: usize, derivative: &mut [u8]) {
            TransformPlan::derivative_bytes(self, coefficients, symbol_len, derivative);
        }
    }
}

/// Construction helpers hidden from the stable codec API.
pub mod codec {
    use crate::{Engine, Profile};

    /// Assemble a profile from parts already validated by the caller.
    #[must_use]
    pub const fn profile_from_parts(
        engine: Engine,
        k: usize,
        m: usize,
        symbol_len: usize,
    ) -> Profile {
        Profile::from_parts(engine, k, m, symbol_len)
    }
}

/// Decoder recipe and coefficient-generation internals.
pub mod decoder {
    use std::sync::Arc;

    use crate::decoder::RecipeCache;
    pub use crate::decoder::cauchy_inverse::{
        RationalLagrangeCoefficients, rational_lagrange_coefficients,
    };
    pub use crate::decoder::recipe::{RecipeKey, ReconstructionRecipe, SourceTerm};

    /// Access mutable recipe-cache implementation details.
    pub trait RecipeCacheExt {
        /// Maximum number of retained recipes.
        fn capacity(&self) -> usize;
        /// Set the maximum number of retained recipes without evicting entries.
        fn set_capacity(&mut self, capacity: usize);
        /// Entries in most-recently-used order.
        fn entries(&self) -> &[(RecipeKey, Arc<ReconstructionRecipe>)];
        /// Mutable entries in most-recently-used order.
        fn entries_mut(&mut self) -> &mut Vec<(RecipeKey, Arc<ReconstructionRecipe>)>;
        /// Cached fast-path entry.
        fn last(&self) -> Option<&(RecipeKey, Arc<ReconstructionRecipe>)>;
        /// Mutable cached fast-path entry.
        fn last_mut(&mut self) -> &mut Option<(RecipeKey, Arc<ReconstructionRecipe>)>;
        /// Look up and promote a reconstruction recipe.
        fn get(&mut self, key: RecipeKey) -> Option<Arc<ReconstructionRecipe>>;
        /// Insert a reconstruction recipe.
        fn insert(&mut self, key: RecipeKey, recipe: Arc<ReconstructionRecipe>);
    }

    impl RecipeCacheExt for RecipeCache {
        fn capacity(&self) -> usize {
            self.capacity
        }

        fn set_capacity(&mut self, capacity: usize) {
            self.capacity = capacity;
        }

        fn entries(&self) -> &[(RecipeKey, Arc<ReconstructionRecipe>)] {
            &self.entries
        }

        fn entries_mut(&mut self) -> &mut Vec<(RecipeKey, Arc<ReconstructionRecipe>)> {
            &mut self.entries
        }

        fn last(&self) -> Option<&(RecipeKey, Arc<ReconstructionRecipe>)> {
            self.last.as_ref()
        }

        fn last_mut(&mut self) -> &mut Option<(RecipeKey, Arc<ReconstructionRecipe>)> {
            &mut self.last
        }

        fn get(&mut self, key: RecipeKey) -> Option<Arc<ReconstructionRecipe>> {
            RecipeCache::get(self, key)
        }

        fn insert(&mut self, key: RecipeKey, recipe: Arc<ReconstructionRecipe>) {
            RecipeCache::insert(self, key, recipe);
        }
    }
}

/// GF(256) payload kernels shared by batch and streaming codecs.
pub mod payload {
    pub use crate::payload::{
        xor_scaled_bytes, xor_scaled_bytes_rows, xor_scaled_bytes_rows_terms,
    };
}

/// Runtime-dispatched GF(256) SIMD kernels.
#[cfg(feature = "simd")]
pub mod simd {
    pub use crate::simd::*;
}

/// GF(65536) Tower Cauchy implementation details.
pub mod tower {
    pub use crate::tower::cauchy::batch_invert;

    /// Fixed-coefficient GF(65536) payload and butterfly kernels.
    pub mod payload {
        pub use crate::tower::payload::*;
    }
}
