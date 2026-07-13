//! Reconstruction-recipe cache.

use std::sync::Arc;

use super::recipe::{RecipeKey, ReconstructionRecipe};

/// Fixed-capacity LRU cache for reconstruction recipes.
///
/// The cache key includes `(matrix type, k, m, pattern)` so one cache can safely
/// be shared across decoder configurations and Cauchy constructions. Entries are stored in most-recently-used order
/// with a tiny `Vec`; expected capacities are small (tens to hundreds of link
/// patterns), so linear lookup avoids an extra dependency and keeps the crate
/// simple.
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
    /// This includes recipe/vector storage and coefficient capacity, but excludes
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
