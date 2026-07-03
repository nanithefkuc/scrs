//! Recipe types for decoder memoization and reconstruction.

use crate::pattern_key::PatternKey;

pub(crate) use crate::simd::ScaleTable;

/// Key used by [`RecipeCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RecipeKey {
    pub k: usize,
    pub m: usize,
    pub pattern: PatternKey,
}

/// A decoded reconstruction plan for a single erasure pattern.
///
/// The recipe stores *fused* coefficients so that [`crate::decoder::LazyDecoderState::apply_recipe`]
/// can reconstruct all missing symbols in a single source-symbol-major pass,
/// reading each received payload byte exactly once instead of once per repair
/// row.
#[derive(Clone)]
pub(crate) struct ReconstructionRecipe {
    pub missing_data: Vec<usize>,
    pub present_data: Vec<usize>,
    pub repair_cols: Vec<usize>,
    /// For each missing output `m_j`, the scaled repair-symbol contributions:
    /// `out[m_j] = sum P[m_j, r_i] * repair[r_i]`.
    pub repair_terms: Vec<Vec<RhsTerm>>,
    /// For each missing output `m_j`, the scaled present-data contributions:
    /// `out[m_j] ^= sum Q[m_j, d] * data[d]`, where
    /// `Q[m_j, d] = sum_{r_i} Ainv[m_j, r_i] * C[d, r_i]`.
    pub data_terms: Vec<Vec<DataTerm>>,
}

/// A scaled contribution from a received data symbol.
#[derive(Clone)]
pub(crate) struct DataTerm {
    pub data_idx: usize,
    pub scale: ScaleTable,
}

/// A scaled contribution from a reduced-system RHS row.
#[derive(Clone)]
pub(crate) struct RhsTerm {
    pub rhs_pos: usize,
    pub scale: ScaleTable,
}

/// Fixed-capacity LRU cache for reconstruction recipes.
///
/// The cache key includes `(k, m, pattern)` so one cache can safely be shared
/// across decoder configurations. Entries are stored in most-recently-used order
/// with a tiny `Vec`; expected capacities are small (tens to hundreds of link
/// patterns), so linear lookup avoids an extra dependency and keeps the crate
/// simple.
pub struct RecipeCache {
    pub(crate) capacity: usize,
    pub(crate) entries: Vec<(RecipeKey, ReconstructionRecipe)>,
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

    pub(crate) fn get(&mut self, key: RecipeKey) -> Option<ReconstructionRecipe> {
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        self.hits += 1;
        let entry = self.entries.remove(pos);
        let recipe = entry.1.clone();
        self.entries.insert(0, entry);
        Some(recipe)
    }

    pub(crate) fn insert(&mut self, key: RecipeKey, recipe: ReconstructionRecipe) {
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
