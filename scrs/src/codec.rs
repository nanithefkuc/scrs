//! Unified v2 codec vocabulary: [`Field`], [`Engine`], [`Profile`], and the
//! field-agnostic trait family every engine implements.
//!
//! The concrete engines stay separate types (their internals differ
//! irreducibly — reduced Cauchy recipe vs additive-FFT transform), but they
//! share one surface:
//!
//! - [`Coded`] — dimension accessors (`k`, `m`, `n`, `symbol_len`);
//! - [`IncrementalEncoder`] — data sendable immediately, per-source repair
//!   updates (GF(256) Good Cauchy, GF(65536) tower);
//! - [`BatchEncoder`] — block-final encode with a reusable scratch (GF(256)
//!   batch, GF(65536) additive-FFT);
//! - [`Decoder`] — lazy payload-deferred streaming decode with a reusable
//!   scratch (all engines).

use crate::error::{DecodeError, EncodeError};
use crate::stream::SymbolSink;

/// Finite field underlying a codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    /// GF(2^8), the AES field. Capacity `k + m <= 256`.
    Gf256,
    /// GF(2^16) as a quadratic extension of GF(256). Capacity up to 65535/65536.
    Gf65536,
}

/// Concrete coding engine. Each fixes a field and a construction; a sender and
/// receiver MUST agree on the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// GF(256) Standard Cauchy (`k + m <= 256`).
    StandardCauchy,
    /// GF(256) Good Cauchy (`k + m <= 255`), incremental streaming encode.
    GoodCauchy,
    /// GF(65536) incremental Tower Cauchy.
    Tower,
    /// GF(65536) block-final additive FFT.
    Afft,
}

impl Engine {
    /// The field this engine codes over.
    #[must_use]
    pub const fn field(self) -> Field {
        match self {
            Engine::StandardCauchy | Engine::GoodCauchy => Field::Gf256,
            Engine::Tower | Engine::Afft => Field::Gf65536,
        }
    }
}

/// A fully resolved codec geometry: which engine, and the block shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    engine: Engine,
    k: usize,
    m: usize,
    symbol_len: usize,
}

impl Profile {
    /// The engine.
    #[must_use]
    pub const fn engine(&self) -> Engine {
        self.engine
    }
    /// The field.
    #[must_use]
    pub const fn field(&self) -> Field {
        self.engine.field()
    }
    /// Number of data symbols.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }
    /// Number of repair symbols.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }
    /// Codeword length `n = k + m`.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.k + self.m
    }
    /// Per-symbol byte length.
    #[must_use]
    pub const fn symbol_len(&self) -> usize {
        self.symbol_len
    }
}

/// Dimension accessors shared by every codec.
pub trait Coded {
    /// Number of data symbols `k`.
    fn k(&self) -> usize;
    /// Number of repair symbols `m`.
    fn m(&self) -> usize;
    /// Per-symbol byte length.
    fn symbol_len(&self) -> usize;
    /// Codeword length `n = k + m`.
    fn n(&self) -> usize {
        self.k() + self.m()
    }
}

/// Incremental (streaming) encoder: data symbols are sendable immediately and
/// each source arrival updates every repair. Implemented by GF(256) Good
/// Cauchy and GF(65536) tower.
pub trait IncrementalEncoder: Coded {
    /// Add one data symbol's contribution to every repair.
    fn feed(&mut self, index: usize, data: &[u8]) -> Result<(), EncodeError>;
    /// Borrow a finished repair symbol (`index` in `0..m`).
    fn repair(&self, index: usize) -> Result<&[u8], EncodeError>;
    /// Number of distinct data symbols fed so far.
    fn fed_count(&self) -> usize;
    /// Clear fed state for reuse, retaining allocations.
    fn reset(&mut self);
}

/// Block-final encoder: all data present, compute every repair. A reusable
/// [`Scratch`](BatchEncoder::Scratch) enables zero-alloc steady state.
/// Implemented by GF(256) batch and GF(65536) additive-FFT.
pub trait BatchEncoder: Coded {
    /// Reusable, caller-owned encode workspace (`()` where none is needed).
    type Scratch;
    /// Allocate a scratch sized for this codec.
    fn scratch(&self) -> Self::Scratch;
    /// Encode `data` (`k * symbol_len`) into `repairs` (`m * symbol_len`),
    /// using a throwaway scratch.
    fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError>;
    /// Zero-alloc steady state: the caller owns and reuses `scratch`.
    fn encode_into_with(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), EncodeError>;
}

/// Lazy, payload-deferred streaming decoder. `push`/`is_complete`/`finalize`
/// come from [`SymbolSink`]; this trait adds progress accessors and the
/// canonical reconstruct-into-buffer methods with a reusable scratch.
pub trait Decoder: Coded + SymbolSink {
    /// Reusable, caller-owned decode workspace (recipe cache / transform
    /// scratch; `()` where none is needed).
    type Scratch;
    /// Allocate a scratch sized for this codec.
    fn scratch(&self) -> Self::Scratch;
    /// Distinct independent symbols received (capped by `k` at completion).
    fn rank(&self) -> usize;
    /// Total symbols received, including duplicates/dependents.
    fn received(&self) -> usize;
    /// Reconstruct `k * symbol_len` bytes into `out`, using a throwaway
    /// scratch. Non-consuming; the decoder remains usable.
    fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError>;
    /// Zero-alloc steady state: the caller owns and reuses `scratch`.
    fn finalize_into_with(
        &mut self,
        out: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError>;
}
