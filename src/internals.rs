//! Unstable implementation APIs for experimentation and downstream tuning.
//!
//! This module is available only with the `internals` feature. Its contents are
//! not covered by SCRS's public compatibility guarantees and may change between
//! patch releases.

/// Additive-FFT planning and byte-transform internals.
///
/// The transform itself is [`cafft`], whose plan already exposes every
/// byte-oriented entry point publicly, so this module only re-exports SCRS's own
/// planning types and the shared-plan accessor. Call transforms directly on
/// [`TransformPlan`].
pub mod afft {
    use std::sync::Arc;

    pub use crate::afft::profile::{Profile, zeroed_bytes};
    pub use crate::afft::{Field, MAX_TRANSFORM_SIZE, TransformPlan};

    /// Return the process-wide transform plan for `size`.
    ///
    /// `None` when `size` is not a power of two or exceeds the field's domain.
    #[must_use]
    pub fn shared_transform_plan(size: usize) -> Option<Arc<TransformPlan>> {
        TransformPlan::shared(size).ok()
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

/// Active SIMD backends for SCRS's two kernel layers.
///
/// Both resolve to the same [`fff::kernel::Backend`] enum but can differ: cafft
/// caps what its butterflies support (`Avx512` falls back to `Gfni`) and applies
/// its own downgrade-only `CAFFT_BACKEND` override *after* fff's `FFF_BACKEND`.
/// Payload arithmetic follows [`payload_backend`], additive-FFT butterflies
/// follow [`transform_backend`].
pub mod backend {
    pub use fff::kernel::{Backend, backend_for, has_vector_elementwise};

    /// Backend used by the GF(2^8) and GF(2^16) payload kernels.
    #[must_use]
    pub fn payload_backend() -> Backend {
        fff::kernel::backend()
    }

    /// Backend used by the additive-FFT butterflies.
    #[must_use]
    pub fn transform_backend() -> Backend {
        cafft::core::kernel::backend()
    }
}

/// GF(65536) Tower Cauchy implementation details.
///
/// The GF(65536) payload kernels and fused butterflies that used to live here
/// are now [`fff::ops`] and [`cafft::core::kernel`] respectively, both public.
pub mod tower {
    pub use crate::tower::cauchy::{batch_invert, batch_invert_into};
}
