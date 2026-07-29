//! Unstable implementation APIs for experimentation and downstream tuning.
//!
//! This module is available only with the `internals` feature. Its contents are
//! not covered by SRS's public compatibility guarantees and may change between
//! patch releases.
//!
//! # How the surface is exposed
//!
//! Most of SRS's implementation is not re-exported here, because the feature
//! makes it reachable in place. Three mechanisms are in play, and knowing which
//! one applies tells you where to look:
//!
//! 1. **Gated modules.** A module that is private in the default build becomes
//!    `pub` under the feature, and its free functions, consts and statics come
//!    with it. Nothing is re-exported; use the real path.
//! 2. **Gated methods.** A private inherent method on a type that the *stable*
//!    API already exports becomes `pub` under the feature. These cannot be
//!    re-exported — call them on the type.
//! 3. **Gated accessors.** A private field on such a type gets a borrowing
//!    accessor under the feature, so cross-field invariants survive.
//!
//! This module holds only what none of the three can express: constructors that
//! bypass validation, an extension trait for `pub(crate)` fields, and backend
//! reporting that spans two dependencies.
//!
//! # Map of the gated modules
//!
//! | path | what lives there |
//! |---|---|
//! | [`crate::afft::decoder`] | `TARGETED_MAX_MISSING`, the systematic-locator caches, `fit` |
//! | [`crate::afft::encoder`] | strip-encoder and scratch accessors |
//! | [`crate::afft::profile`] | [`afft::Profile`], [`afft::zeroed_bytes`] |
//! | [`crate::batch::codec`] | `invert_square_into`, codec and scratch accessors |
//! | [`crate::decoder::cache`] | [`RecipeCache`](crate::decoder::RecipeCache) internals, via [`decoder::RecipeCacheExt`] |
//! | [`crate::decoder::cauchy_inverse`] | `rational_lagrange_small`, GF(256) `batch_invert` |
//! | [`crate::decoder::recipe`] | recipe types with public fields |
//! | [`crate::decoder::streaming`] | `MAX_SOURCES`, decoder-state accessors |
//! | [`crate::encoder::streaming`] | incremental encoder accessors |
//! | [`crate::payload`] | the GF(256) payload kernel seam |
//! | [`crate::tower::cauchy`] | `power`, `batch_invert`, `batch_invert_into` |
//! | [`crate::tower::decoder`] | `rational_lagrange_coefficients_into`, scratch accessors |
//! | [`crate::tower::encoder`] | tower encoder accessors |
//!
//! Matrix and selector internals are reached as methods on already-public types
//! ([`CauchyView::x_at`](crate::cauchy::CauchyView::x_at),
//! [`MatrixView::buf`](crate::matrices::MatrixView::buf)) or as free items in
//! already-public modules ([`crate::selector::engine_capacity`],
//! [`crate::good_cauchy::exp`], [`crate::cauchy::combinations`]).
//!
//! # Upstream internals
//!
//! This feature also turns on `fff/internals` and `cafft/internals`, so the two
//! dependencies' own unstable surfaces come with it — `fff::kernel`'s per-field
//! kernel modules and table banks, and `cafft::core::factors`. The former
//! `internals::simd` and `internals::tower::payload` modules are gone because
//! their contents are now public in [`fff::ops`], [`fff::kernel`] and
//! [`cafft::core::kernel`], or were deleted with the hand-written kernels they
//! dispatched. The one exception is [`tables`], re-exported here because the
//! pre-migration `internals::simd` published it.

/// Additive-FFT planning internals.
///
/// The transform itself is [`cafft`], whose plan already exposes every
/// byte-oriented entry point publicly, so this module only re-exports SRS's own
/// planning types and the shared-plan accessor. Call transforms directly on
/// [`afft::TransformPlan`].
pub mod afft {
    use std::sync::Arc;

    pub use crate::afft::profile::{Profile, zeroed_bytes};
    pub use crate::afft::{Field, TransformPlan};

    /// Return the process-wide transform plan for `size` over field `F`.
    ///
    /// `None` when `size` is not a power of two or exceeds the field's domain.
    #[must_use]
    pub fn shared_transform_plan<F: Field>(size: usize) -> Option<Arc<TransformPlan<F>>> {
        TransformPlan::<F>::shared(size).ok()
    }
}

/// Construction helpers hidden from the stable codec API.
pub mod codec {
    use crate::{Engine, Profile};

    /// Assemble a profile from parts already validated by the caller.
    ///
    /// Skips the geometry checks [`Profile::resolve`] performs, so an
    /// out-of-range `(k, m)` produces a profile no engine can honour.
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
///
/// The recipe types themselves live in [`crate::decoder::recipe`] and the
/// coefficient helpers in [`crate::decoder::cauchy_inverse`], both public under
/// this feature. Only [`decoder::RecipeCacheExt`] must live here:
/// [`crate::decoder::RecipeCache`]'s fields are `pub(crate)`, which no module
/// gate can widen.
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
        /// Reset the hit and miss counters without dropping cached recipes.
        fn reset_statistics(&mut self);
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

        fn reset_statistics(&mut self) {
            self.hits = 0;
            self.misses = 0;
        }
    }
}

/// GF(256) payload kernels shared by batch and streaming codecs.
///
/// Re-exported for continuity; [`crate::payload`] is public under this feature
/// and carries the module documentation.
pub mod payload {
    pub use crate::payload::{
        xor_scaled_bytes, xor_scaled_bytes_rows, xor_scaled_bytes_rows_terms,
    };
}

/// Split-nibble and tower multiplication tables from [`fff::kernel::tables`].
///
/// These are the shared table banks the vector kernels index when the host has
/// no `Gfni` path: [`tables::ScaleTable`] holds `lo[i] = coeff * i` and
/// `hi[i] = coeff * (i << 4)` for one GF(2^8) coefficient, and
/// [`tables::TowerCoeff`]
/// decomposes a GF(2^16) tower multiply into two byte-wide GF(2^8) multiplies.
///
/// Re-exported because the pre-migration `internals::simd` module published
/// `ScaleTable` and `scale_table`, and downstream tuners indexed them directly.
/// The tables are `fff`'s, not SRS's — nothing here adapts them, and
/// [`fff::kernel::tables`] is equally reachable under this feature.
pub mod tables {
    pub use fff::kernel::tables::{ScaleTable, TowerCoeff, TowerTables, scale_table};
}

/// Active SIMD backends for SRS's two kernel layers.
///
/// Both resolve to the same [`fff::kernel::Backend`] enum but can differ: cafft
/// caps what its butterflies support (`Avx512` falls back to `Gfni`) and applies
/// its own downgrade-only `CAFFT_BACKEND` override *after* fff's `FFF_BACKEND`.
/// Payload arithmetic follows [`backend::payload_backend`], additive-FFT
/// butterflies follow [`backend::transform_backend`].
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
