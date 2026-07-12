//! Compact recipe types for decoder memoization and reconstruction.

use std::sync::Arc;

use crate::gf256::GfElem;
use crate::pattern_key::PatternKey;

/// Key used by [`RecipeCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RecipeKey {
    pub k: usize,
    pub m: usize,
    /// Concrete `CodingMatrix` implementation that produced the coefficients.
    pub matrix_type: &'static str,
    pub pattern: PatternKey,
}

/// A compact reconstruction plan for one erasure pattern.
///
/// Coefficients are output-major. The row for `missing_data[missing_pos]` is
/// `coefficients[missing_pos * source_indices.len()..][..source_indices.len()]`.
pub(crate) struct ReconstructionRecipe {
    pub missing_data: Vec<usize>,
    pub present_data: Vec<usize>,
    /// Codeword indices for the received repair and data symbols used by every
    /// missing output.
    pub source_indices: Vec<usize>,
    /// Raw GF(256) coefficient bytes, stored output-major.
    pub coefficients: Vec<GfElem>,
}

/// Fixed-capacity LRU cache for reconstruction recipes.
///
/// The cache key includes `(matrix type, k, m, pattern)` so one cache can safely
/// be shared across decoder configurations and Cauchy constructions. Recipes
/// are reference-counted, making cache hits independent of recipe size.
pub struct RecipeCache {
    pub(crate) capacity: usize,
    pub(crate) entries: Vec<(RecipeKey, Arc<ReconstructionRecipe>)>,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
}

impl RecipeCache {
    /// Create an LRU cache that stores up to `capacity` recipes.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Number of recipes currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache contains no recipes.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of cache hits since construction.
    pub const fn hits(&self) -> usize {
        self.hits
    }

    /// Number of cache misses since construction.
    pub const fn misses(&self) -> usize {
        self.misses
    }

    /// Estimated bytes allocated for cached recipe payloads.
    ///
    /// This includes recipe/vector storage and vector capacity, but excludes
    /// allocator bookkeeping and the small `Arc` control block for each entry.
    pub fn recipe_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, recipe)| recipe.allocated_bytes())
            .sum()
    }

    pub(crate) fn get(&mut self, key: RecipeKey) -> Option<Arc<ReconstructionRecipe>> {
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        self.hits += 1;
        let entry = self.entries.remove(pos);
        let recipe = Arc::clone(&entry.1);
        self.entries.insert(0, entry);
        Some(recipe)
    }

    pub(crate) fn insert(&mut self, key: RecipeKey, recipe: Arc<ReconstructionRecipe>) {
        self.misses += 1;
        if self.capacity == 0 {
            return;
        }
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries.remove(pos);
        }
        self.entries.insert(0, (key, recipe));
        if self.entries.len() > self.capacity {
            self.entries.pop();
        }
    }
}

impl ReconstructionRecipe {
    fn allocated_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.missing_data.capacity() * core::mem::size_of::<usize>()
            + self.present_data.capacity() * core::mem::size_of::<usize>()
            + self.source_indices.capacity() * core::mem::size_of::<usize>()
            + self.coefficients.capacity() * core::mem::size_of::<GfElem>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_recipe(k: usize, r: usize) -> Arc<ReconstructionRecipe> {
        Arc::new(ReconstructionRecipe {
            missing_data: (0..r).collect(),
            present_data: (r..k).collect(),
            source_indices: (0..k).collect(),
            coefficients: vec![GfElem(0x53); k * r],
        })
    }

    #[test]
    fn cache_hits_share_recipe_allocation() {
        let key = RecipeKey {
            k: 4,
            m: 3,
            matrix_type: "test",
            pattern: PatternKey::empty(),
        };
        let recipe = synthetic_recipe(4, 2);
        let mut cache = RecipeCache::new(1);
        cache.insert(key, Arc::clone(&recipe));

        let first = cache.get(key).unwrap();
        let second = cache.get(key).unwrap();
        assert!(Arc::ptr_eq(&recipe, &first));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn compact_coefficients_are_materially_smaller_than_embedded_tables() {
        let (k, r) = (128, 64);
        let mut cache = RecipeCache::new(1);
        cache.insert(
            RecipeKey {
                k,
                m: 64,
                matrix_type: "test",
                pattern: PatternKey::empty(),
            },
            synthetic_recipe(k, r),
        );

        let embedded_table_bytes = k * r * core::mem::size_of::<crate::simd::ScaleTable>();
        assert_eq!(cache.recipe_bytes(), 10_336);
        assert!(cache.recipe_bytes() * 25 < embedded_table_bytes);
    }

    #[test]
    fn recipe_memory_stays_bounded_across_cache_working_sets() {
        let (k, r) = (128, 127);
        for capacity in [16, 64, 256] {
            let mut cache = RecipeCache::new(capacity);
            for pattern_number in 0..=capacity {
                let mut pattern = PatternKey::empty();
                pattern.set(pattern_number % 256);
                cache.insert(
                    RecipeKey {
                        k,
                        m: 127,
                        matrix_type: "test",
                        pattern,
                    },
                    synthetic_recipe(k, r),
                );
            }
            assert_eq!(cache.len(), capacity);
            assert!(cache.recipe_bytes() <= capacity * 18_400);
        }
    }
}
