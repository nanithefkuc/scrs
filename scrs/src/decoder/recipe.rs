//! Recipe types for decoder memoization and reconstruction.

use crate::codec::Engine;
use crate::gf256::GfElem;
use crate::pattern_key::PatternKey;

/// Key used by [`RecipeCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RecipeKey {
    pub k: usize,
    pub m: usize,
    /// Coding engine that produced the coefficients.
    pub engine: Engine,
    pub pattern: PatternKey,
}

/// A decoded reconstruction plan for a single erasure pattern.
///
/// Terms are stored in source-major order. On AVX2+GFNI, complete groups of four
/// outputs use a fully unrolled kernel that shares each source load; small and
/// remainder groups, plus other architectures, execute the same coefficients
/// output-major. `coefficients[missing_pos]` is the source's coefficient for
/// `missing_data[missing_pos]`.
#[derive(Clone)]
pub(crate) struct ReconstructionRecipe {
    pub missing_data: Vec<usize>,
    pub present_data: Vec<usize>,
    pub source_terms: Vec<SourceTerm>,
}

/// One received symbol and its coefficients for every missing output.
#[derive(Clone)]
pub(crate) struct SourceTerm {
    /// Codeword index in the decoder payload buffer.
    pub source_idx: usize,
    /// Coefficient bytes in the same order as `ReconstructionRecipe::missing_data`.
    pub coefficients: Vec<GfElem>,
}

impl ReconstructionRecipe {
    pub(super) fn allocated_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.missing_data.capacity() * core::mem::size_of::<usize>()
            + self.present_data.capacity() * core::mem::size_of::<usize>()
            + self.source_terms.capacity() * core::mem::size_of::<SourceTerm>()
            + self
                .source_terms
                .iter()
                .map(|term| term.coefficients.capacity() * core::mem::size_of::<GfElem>())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::cache::RecipeCache;
    use std::sync::Arc;

    #[test]
    fn cache_hits_share_recipe_allocation() {
        let key = RecipeKey {
            k: 4,
            m: 3,
            engine: Engine::GoodCauchy,
            pattern: PatternKey::empty(),
        };
        let recipe = Arc::new(ReconstructionRecipe {
            missing_data: vec![0, 1],
            present_data: vec![2, 3],
            source_terms: vec![SourceTerm {
                source_idx: 4,
                coefficients: vec![GfElem(2), GfElem(3)],
            }],
        });
        let mut cache = RecipeCache::new(1);
        cache.insert(key, Arc::clone(&recipe));

        let first = cache.get(key).unwrap();
        let second = cache.get(key).unwrap();
        assert!(Arc::ptr_eq(&recipe, &first));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn single_pattern_fast_path_does_not_corrupt_distinct_lookups() {
        let mut key_a = RecipeKey {
            k: 4,
            m: 3,
            engine: Engine::GoodCauchy,
            pattern: PatternKey::empty(),
        };
        key_a.pattern.set(0);
        let mut key_b = key_a;
        key_b.pattern = PatternKey::empty();
        key_b.pattern.set(1);

        let recipe_a = synthetic_recipe(4, 2);
        let recipe_b = synthetic_recipe(4, 1);
        let mut cache = RecipeCache::new(2);
        cache.insert(key_a, Arc::clone(&recipe_a));
        cache.insert(key_b, Arc::clone(&recipe_b));

        // Repeat B (MRU) several times: fast path must keep returning B.
        for _ in 0..3 {
            assert!(Arc::ptr_eq(&cache.get(key_b).unwrap(), &recipe_b));
        }
        // A distinct key must still resolve to its own recipe and refresh `last`.
        assert!(Arc::ptr_eq(&cache.get(key_a).unwrap(), &recipe_a));
        assert!(Arc::ptr_eq(&cache.get(key_a).unwrap(), &recipe_a));
        // Missing key still misses.
        let mut key_c = key_a;
        key_c.pattern.set(2);
        assert!(cache.get(key_c).is_none());
        assert_eq!(cache.hits(), 5);
    }

    fn synthetic_recipe(k: usize, r: usize) -> Arc<ReconstructionRecipe> {
        Arc::new(ReconstructionRecipe {
            missing_data: (0..r).collect(),
            present_data: (r..k).collect(),
            source_terms: (0..k)
                .map(|source_idx| SourceTerm {
                    source_idx,
                    coefficients: vec![GfElem(0x53); r],
                })
                .collect(),
        })
    }

    #[cfg(feature = "simd")]
    #[test]
    fn compact_coefficients_are_materially_smaller_than_embedded_tables() {
        let (k, r) = (128, 64);
        let mut cache = RecipeCache::new(1);
        cache.insert(
            RecipeKey {
                k,
                m: 64,
                engine: Engine::GoodCauchy,
                pattern: PatternKey::empty(),
            },
            synthetic_recipe(k, r),
        );

        let embedded_table_bytes = k * r * core::mem::size_of::<crate::simd::ScaleTable>();
        assert_eq!(cache.recipe_bytes(), 13_384);
        assert!(cache.recipe_bytes() * 20 < embedded_table_bytes);
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
                        engine: Engine::GoodCauchy,
                        pattern,
                    },
                    synthetic_recipe(k, r),
                );
            }
            assert_eq!(cache.len(), capacity);
            assert!(cache.recipe_bytes() <= capacity * 21_500);
        }
    }
}
