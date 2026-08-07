//! Geometry-driven codec selection and type-erased dispatch.
//!
//! [`Profile::resolve`] / [`Profile::recommended`] turn a `(field, k, m,
//! symbol_len)` request into a validated [`Profile`]. The [`decoder`],
//! [`batch_decoder`], [`batch_encoder`], and [`incremental_encoder`] free
//! functions build type-erased codecs from a profile, so callers that do not
//! want to name a concrete engine still get the unified trait API. Concrete
//! engine types remain public for zero-cost monomorphized use.

use crate::afft;
use crate::batch::{BatchCodec, DecodeScratch as CauchyDecodeScratch};
use crate::cauchy::CauchyView;
use crate::codec::{
    BatchDecoder, BatchEncoder, Coded, Decoder, Engine, Field, IncrementalEncoder, Profile,
};
use crate::decoder::{LazyDecoderState, RecipeCache};
use crate::encoder::StreamingEncoder;
use crate::error::{ConfigError, DecodeError, EncodeError};
use crate::good_cauchy::GoodCauchyView;
use crate::stream::{PushOutcome, SymbolSink};
use crate::tower;
use crate::{Gf8Engine, Gf16Engine, recommended_gf8_engine, recommended_gf16_engine};

internals_pub! {
/// Maximum `k + m` for an engine.
///
/// Fixed by the codeword geometry each engine can address: Good Cauchy loses a
/// position to its `x`/`y` disjointness requirement, and the additive-FFT
/// engines are bounded by their field's evaluation domain.
const fn engine_capacity(engine: Engine) -> usize {
    match engine {
        Engine::StandardCauchy => 256,
        Engine::GoodCauchy => 255,
        Engine::Tower => 65_535,
        Engine::Gf8Afft => 256,
        Engine::Gf16Afft => 65_536,
    }
}
}

impl Profile {
    /// Validate a `(engine, k, m, symbol_len)` request into a [`Profile`].
    ///
    /// Applies the same dimension / capacity / symbol-length rules the concrete
    /// constructors enforce, so a successful `resolve` guarantees the matching
    /// [`decoder`] / `*_encoder` constructor accepts it.
    pub fn resolve(
        engine: Engine,
        k: usize,
        m: usize,
        symbol_len: usize,
    ) -> Result<Self, ConfigError> {
        if k == 0 || m == 0 {
            return Err(ConfigError::ZeroDimension);
        }
        if symbol_len == 0 {
            return Err(ConfigError::ZeroSymbolLen);
        }
        if engine.field() == Field::Gf65536 && symbol_len % 2 != 0 {
            return Err(ConfigError::OddSymbolLen);
        }
        let cap = engine_capacity(engine);
        if k + m > cap {
            return Err(ConfigError::TooManySymbols { cap });
        }
        // Safe to construct via the private-field path.
        Ok(Self::from_parts(engine, k, m, symbol_len))
    }

    /// Pick a geometry-appropriate engine for `field` and resolve it.
    ///
    /// GF(256): Good Cauchy when `k + m <= 255` (enables streaming), else
    /// Standard Cauchy. GF(65536): [`recommended_gf16_engine`].
    pub fn recommended(
        field: Field,
        k: usize,
        m: usize,
        symbol_len: usize,
    ) -> Result<Self, ConfigError> {
        let engine = match field {
            Field::Gf256 => match recommended_gf8_engine(k, m) {
                Gf8Engine::GoodCauchy => Engine::GoodCauchy,
                Gf8Engine::StandardCauchy => Engine::StandardCauchy,
                Gf8Engine::Afft => Engine::Gf8Afft,
            },
            Field::Gf65536 => match recommended_gf16_engine(k, m) {
                Gf16Engine::Tower => Engine::Tower,
                Gf16Engine::Afft => Engine::Gf16Afft,
            },
        };
        Self::resolve(engine, k, m, symbol_len)
    }
}

// ---------------------------------------------------------------------------
// Type-erased decoder
// ---------------------------------------------------------------------------

/// A decoder for any engine, selected at runtime from a [`Profile`].
pub enum AnyDecoder {
    /// GF(256) Standard Cauchy.
    StandardCauchy(LazyDecoderState<CauchyView>),
    /// GF(256) Good Cauchy.
    GoodCauchy(LazyDecoderState<GoodCauchyView>),
    /// GF(65536) tower.
    Tower(tower::LazyDecoderState),
    /// GF(256) additive FFT.
    Gf8Afft(afft::Gf8Decoder),
    /// GF(65536) additive FFT.
    Gf16Afft(afft::Gf16Decoder),
}

/// Reusable decode scratch for [`AnyDecoder`], matching its active engine.
pub enum AnyDecodeScratch {
    /// GF(256) recipe cache (shared by both Cauchy variants).
    Cauchy(RecipeCache),
    /// GF(65536) tower reconstruction workspace.
    Tower(tower::DecodeScratch),
    /// GF(256) additive-FFT transform scratch.
    Gf8Afft(afft::DecodeScratch<fff::Gf8>),
    /// GF(65536) additive-FFT transform scratch.
    Gf16Afft(afft::DecodeScratch<fff::Gf16>),
}

/// Build a decoder for `profile`.
pub fn decoder(profile: &Profile) -> Result<AnyDecoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    Ok(match profile.engine() {
        Engine::StandardCauchy => AnyDecoder::StandardCauchy(LazyDecoderState::new(k, m, s)?),
        Engine::GoodCauchy => AnyDecoder::GoodCauchy(LazyDecoderState::new(k, m, s)?),
        Engine::Tower => AnyDecoder::Tower(tower::LazyDecoderState::new(k, m, s)?),
        Engine::Gf8Afft => AnyDecoder::Gf8Afft(afft::Gf8Decoder::new(k, m, s)?),
        Engine::Gf16Afft => AnyDecoder::Gf16Afft(afft::Gf16Decoder::new(k, m, s)?),
    })
}

impl Coded for AnyDecoder {
    fn k(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.k(),
            AnyDecoder::GoodCauchy(d) => d.k(),
            AnyDecoder::Tower(d) => d.k(),
            AnyDecoder::Gf8Afft(d) => d.k(),
            AnyDecoder::Gf16Afft(d) => d.k(),
        }
    }
    fn m(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.m(),
            AnyDecoder::GoodCauchy(d) => d.m(),
            AnyDecoder::Tower(d) => d.m(),
            AnyDecoder::Gf8Afft(d) => d.m(),
            AnyDecoder::Gf16Afft(d) => d.m(),
        }
    }
    fn symbol_len(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.symbol_len(),
            AnyDecoder::GoodCauchy(d) => d.symbol_len(),
            AnyDecoder::Tower(d) => d.symbol_len(),
            AnyDecoder::Gf8Afft(d) => d.symbol_len(),
            AnyDecoder::Gf16Afft(d) => d.symbol_len(),
        }
    }
}

impl SymbolSink for AnyDecoder {
    fn push(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.push(idx, payload),
            AnyDecoder::GoodCauchy(d) => d.push(idx, payload),
            AnyDecoder::Tower(d) => d.push(idx, payload),
            AnyDecoder::Gf8Afft(d) => d.push(idx, payload),
            AnyDecoder::Gf16Afft(d) => d.push(idx, payload),
        }
    }
    fn is_complete(&self) -> bool {
        match self {
            AnyDecoder::StandardCauchy(d) => d.is_complete(),
            AnyDecoder::GoodCauchy(d) => d.is_complete(),
            AnyDecoder::Tower(d) => d.is_complete(),
            AnyDecoder::Gf8Afft(d) => d.is_complete(),
            AnyDecoder::Gf16Afft(d) => d.is_complete(),
        }
    }
    fn finalize(self) -> Result<Vec<u8>, DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.finalize(),
            AnyDecoder::GoodCauchy(d) => d.finalize(),
            AnyDecoder::Tower(d) => d.finalize(),
            AnyDecoder::Gf8Afft(d) => d.finalize(),
            AnyDecoder::Gf16Afft(d) => d.finalize(),
        }
    }
}

impl Decoder for AnyDecoder {
    type Scratch = AnyDecodeScratch;

    fn scratch(&self) -> Self::Scratch {
        match self {
            AnyDecoder::StandardCauchy(d) => AnyDecodeScratch::Cauchy(Decoder::scratch(d)),
            AnyDecoder::GoodCauchy(d) => AnyDecodeScratch::Cauchy(Decoder::scratch(d)),
            AnyDecoder::Tower(d) => AnyDecodeScratch::Tower(d.decode_scratch()),
            AnyDecoder::Gf8Afft(d) => AnyDecodeScratch::Gf8Afft(d.decode_scratch()),
            AnyDecoder::Gf16Afft(d) => AnyDecodeScratch::Gf16Afft(d.decode_scratch()),
        }
    }
    fn rank(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.rank(),
            AnyDecoder::GoodCauchy(d) => d.rank(),
            AnyDecoder::Tower(d) => d.rank(),
            AnyDecoder::Gf8Afft(d) => d.rank(),
            AnyDecoder::Gf16Afft(d) => d.rank(),
        }
    }
    fn received(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.received(),
            AnyDecoder::GoodCauchy(d) => d.received(),
            AnyDecoder::Tower(d) => d.received(),
            AnyDecoder::Gf8Afft(d) => d.received(),
            AnyDecoder::Gf16Afft(d) => d.received(),
        }
    }
    fn reset(&mut self) {
        match self {
            AnyDecoder::StandardCauchy(d) => d.reset(),
            AnyDecoder::GoodCauchy(d) => d.reset(),
            AnyDecoder::Tower(d) => d.reset(),
            AnyDecoder::Gf8Afft(d) => d.reset(),
            AnyDecoder::Gf16Afft(d) => d.reset(),
        }
    }

    fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.finalize_into(out),
            AnyDecoder::GoodCauchy(d) => d.finalize_into(out),
            AnyDecoder::Tower(d) => d.finalize_into(out),
            AnyDecoder::Gf8Afft(d) => d.finalize_into(out),
            AnyDecoder::Gf16Afft(d) => d.finalize_into(out),
        }
    }
    fn finalize_into_with(
        &mut self,
        out: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError> {
        match (self, scratch) {
            (AnyDecoder::StandardCauchy(d), AnyDecodeScratch::Cauchy(c)) => {
                d.finalize_into_with(out, c)
            }
            (AnyDecoder::GoodCauchy(d), AnyDecodeScratch::Cauchy(c)) => {
                d.finalize_into_with(out, c)
            }
            (AnyDecoder::Tower(d), AnyDecodeScratch::Tower(c)) => d.finalize_into_with(out, c),
            (AnyDecoder::Gf8Afft(d), AnyDecodeScratch::Gf8Afft(c)) => d.finalize_into_with(out, c),
            (AnyDecoder::Gf16Afft(d), AnyDecodeScratch::Gf16Afft(c)) => {
                d.finalize_into_with(out, c)
            }
            _ => Err(DecodeError::ScratchMismatch),
        }
    }
}
// ---------------------------------------------------------------------------
// Type-erased batch decoder
// ---------------------------------------------------------------------------

/// A block-final decoder for any engine.
pub enum AnyBatchDecoder {
    /// GF(256) Standard Cauchy reduced-erasure decoder.
    StandardCauchy(BatchCodec<CauchyView>),
    /// GF(256) Good Cauchy reduced-erasure decoder.
    GoodCauchy(BatchCodec<GoodCauchyView>),
    /// GF(65536) tower decoder.
    Tower(tower::LazyDecoderState),
    /// GF(256) native additive-FFT batch decoder.
    Gf8Afft(afft::Gf8BatchDecoder),
    /// GF(65536) native additive-FFT batch decoder.
    Gf16Afft(afft::Gf16BatchDecoder),
}

/// Reusable scratch matching an [`AnyBatchDecoder`].
pub enum AnyBatchDecodeScratch {
    /// GF(256) reduced-erasure workspace.
    Cauchy(CauchyDecodeScratch),
    /// GF(65536) tower reconstruction workspace.
    Tower(tower::DecodeScratch),
    /// GF(256) native additive-FFT batch workspace.
    Gf8Afft(afft::BatchDecodeScratch<fff::Gf8>),
    /// GF(65536) native additive-FFT batch workspace.
    Gf16Afft(afft::BatchDecodeScratch<fff::Gf16>),
}

/// Build a first-class batch decoder for `profile`.
pub fn batch_decoder(profile: &Profile) -> Result<AnyBatchDecoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    Ok(match profile.engine() {
        Engine::StandardCauchy => AnyBatchDecoder::StandardCauchy(BatchCodec::new(k, m, s)?),
        Engine::GoodCauchy => AnyBatchDecoder::GoodCauchy(BatchCodec::new(k, m, s)?),
        Engine::Tower => AnyBatchDecoder::Tower(tower::LazyDecoderState::new(k, m, s)?),
        Engine::Gf8Afft => AnyBatchDecoder::Gf8Afft(afft::Gf8BatchDecoder::new(k, m, s)?),
        Engine::Gf16Afft => AnyBatchDecoder::Gf16Afft(afft::Gf16BatchDecoder::new(k, m, s)?),
    })
}

impl Coded for AnyBatchDecoder {
    fn k(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.k(),
            AnyBatchDecoder::GoodCauchy(d) => d.k(),
            AnyBatchDecoder::Tower(d) => d.k(),
            AnyBatchDecoder::Gf8Afft(d) => d.k(),
            AnyBatchDecoder::Gf16Afft(d) => d.k(),
        }
    }

    fn m(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.m(),
            AnyBatchDecoder::GoodCauchy(d) => d.m(),
            AnyBatchDecoder::Tower(d) => d.m(),
            AnyBatchDecoder::Gf8Afft(d) => d.m(),
            AnyBatchDecoder::Gf16Afft(d) => d.m(),
        }
    }

    fn symbol_len(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.symbol_len(),
            AnyBatchDecoder::GoodCauchy(d) => d.symbol_len(),
            AnyBatchDecoder::Tower(d) => d.symbol_len(),
            AnyBatchDecoder::Gf8Afft(d) => d.symbol_len(),
            AnyBatchDecoder::Gf16Afft(d) => d.symbol_len(),
        }
    }
}

impl BatchDecoder for AnyBatchDecoder {
    type Scratch = AnyBatchDecodeScratch;

    fn scratch(&self) -> Self::Scratch {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => AnyBatchDecodeScratch::Cauchy(d.decode_scratch()),
            AnyBatchDecoder::GoodCauchy(d) => AnyBatchDecodeScratch::Cauchy(d.decode_scratch()),
            AnyBatchDecoder::Tower(d) => AnyBatchDecodeScratch::Tower(d.decode_scratch()),
            AnyBatchDecoder::Gf8Afft(d) => AnyBatchDecodeScratch::Gf8Afft(d.decode_scratch()),
            AnyBatchDecoder::Gf16Afft(d) => AnyBatchDecodeScratch::Gf16Afft(d.decode_scratch()),
        }
    }

    fn decode_into(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
    ) -> Result<(), DecodeError> {
        let mut scratch = <Self as BatchDecoder>::scratch(self);
        self.decode_into_with(symbols, out, &mut scratch)
    }

    fn decode_into_with(
        &mut self,
        symbols: &[(usize, &[u8])],
        out: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), DecodeError> {
        match (self, scratch) {
            (AnyBatchDecoder::StandardCauchy(d), AnyBatchDecodeScratch::Cauchy(s)) => {
                BatchCodec::decode_into_with(d, symbols, out, s)
            }
            (AnyBatchDecoder::GoodCauchy(d), AnyBatchDecodeScratch::Cauchy(s)) => {
                BatchCodec::decode_into_with(d, symbols, out, s)
            }
            (AnyBatchDecoder::Tower(d), AnyBatchDecodeScratch::Tower(s)) => {
                BatchDecoder::decode_into_with(d, symbols, out, s)
            }
            (AnyBatchDecoder::Gf8Afft(d), AnyBatchDecodeScratch::Gf8Afft(s)) => {
                BatchDecoder::decode_into_with(d, symbols, out, s)
            }
            (AnyBatchDecoder::Gf16Afft(d), AnyBatchDecodeScratch::Gf16Afft(s)) => {
                BatchDecoder::decode_into_with(d, symbols, out, s)
            }
            _ => Err(DecodeError::ScratchMismatch),
        }
    }
}

// ---------------------------------------------------------------------------
// Type-erased incremental encoder
// ---------------------------------------------------------------------------

/// An incremental (streaming) encoder for any engine that supports the mode.
pub enum AnyIncrementalEncoder {
    /// GF(256) Good Cauchy.
    GoodCauchy(StreamingEncoder),
    /// GF(65536) tower.
    Tower(tower::StreamingEncoder),
}

/// Build an incremental encoder for `profile`.
///
/// Errors with [`ConfigError::UnsupportedMode`] when the engine is block-final
/// (Standard Cauchy, additive FFT).
pub fn incremental_encoder(profile: &Profile) -> Result<AnyIncrementalEncoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    match profile.engine() {
        Engine::GoodCauchy => Ok(AnyIncrementalEncoder::GoodCauchy(StreamingEncoder::new(
            k, m, s,
        )?)),
        Engine::Tower => Ok(AnyIncrementalEncoder::Tower(tower::StreamingEncoder::new(
            k, m, s,
        )?)),
        engine => Err(ConfigError::UnsupportedMode { engine }),
    }
}

impl Coded for AnyIncrementalEncoder {
    fn k(&self) -> usize {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.k(),
            AnyIncrementalEncoder::Tower(e) => e.k(),
        }
    }
    fn m(&self) -> usize {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.m(),
            AnyIncrementalEncoder::Tower(e) => e.m(),
        }
    }
    fn symbol_len(&self) -> usize {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.symbol_len(),
            AnyIncrementalEncoder::Tower(e) => e.symbol_len(),
        }
    }
}

impl IncrementalEncoder for AnyIncrementalEncoder {
    fn feed(&mut self, index: usize, data: &[u8]) -> Result<(), EncodeError> {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.feed(index, data),
            AnyIncrementalEncoder::Tower(e) => e.feed(index, data),
        }
    }
    fn repair(&self, index: usize) -> Result<&[u8], EncodeError> {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.repair(index),
            AnyIncrementalEncoder::Tower(e) => e.repair(index),
        }
    }
    fn fed_count(&self) -> usize {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.fed_count(),
            AnyIncrementalEncoder::Tower(e) => e.fed_count(),
        }
    }
    fn reset(&mut self) {
        match self {
            AnyIncrementalEncoder::GoodCauchy(e) => e.reset(),
            AnyIncrementalEncoder::Tower(e) => e.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// Type-erased batch encoder
// ---------------------------------------------------------------------------

/// A block-final encoder for any engine that supports the mode.
pub enum AnyBatchEncoder {
    /// GF(256) Standard Cauchy.
    StandardCauchy(BatchCodec<CauchyView>),
    /// GF(256) Good Cauchy.
    GoodCauchy(BatchCodec<GoodCauchyView>),
    /// GF(256) additive FFT.
    Gf8Afft(afft::Gf8Encoder),
    /// GF(65536) additive FFT.
    Gf16Afft(afft::Gf16Encoder),
}

/// Reusable encode scratch for [`AnyBatchEncoder`].
pub enum AnyEncodeScratch {
    /// GF(256) batch needs no workspace.
    Unit,
    /// GF(256) additive-FFT transform workspace.
    Gf8Afft(afft::EncodeScratch),
    /// GF(65536) additive-FFT transform workspace.
    Gf16Afft(afft::EncodeScratch),
}

/// Build a batch encoder for `profile`.
///
/// Errors with [`ConfigError::UnsupportedMode`] when the engine is
/// incremental-only (tower).
pub fn batch_encoder(profile: &Profile) -> Result<AnyBatchEncoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    match profile.engine() {
        Engine::StandardCauchy => Ok(AnyBatchEncoder::StandardCauchy(BatchCodec::new(k, m, s)?)),
        Engine::GoodCauchy => Ok(AnyBatchEncoder::GoodCauchy(BatchCodec::new(k, m, s)?)),
        Engine::Gf8Afft => Ok(AnyBatchEncoder::Gf8Afft(afft::Gf8Encoder::new(k, m, s)?)),
        Engine::Gf16Afft => Ok(AnyBatchEncoder::Gf16Afft(afft::Gf16Encoder::new(k, m, s)?)),
        engine => Err(ConfigError::UnsupportedMode { engine }),
    }
}

impl Coded for AnyBatchEncoder {
    fn k(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.k(),
            AnyBatchEncoder::GoodCauchy(e) => e.k(),
            AnyBatchEncoder::Gf8Afft(e) => e.k(),
            AnyBatchEncoder::Gf16Afft(e) => e.k(),
        }
    }
    fn m(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.m(),
            AnyBatchEncoder::GoodCauchy(e) => e.m(),
            AnyBatchEncoder::Gf8Afft(e) => e.m(),
            AnyBatchEncoder::Gf16Afft(e) => e.m(),
        }
    }
    fn symbol_len(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.symbol_len(),
            AnyBatchEncoder::GoodCauchy(e) => e.symbol_len(),
            AnyBatchEncoder::Gf8Afft(e) => e.symbol_len(),
            AnyBatchEncoder::Gf16Afft(e) => e.symbol_len(),
        }
    }
}

impl BatchEncoder for AnyBatchEncoder {
    type Scratch = AnyEncodeScratch;

    fn scratch(&self) -> Self::Scratch {
        match self {
            AnyBatchEncoder::StandardCauchy(_) => AnyEncodeScratch::Unit,
            AnyBatchEncoder::GoodCauchy(_) => AnyEncodeScratch::Unit,
            AnyBatchEncoder::Gf8Afft(e) => AnyEncodeScratch::Gf8Afft(e.scratch()),
            AnyBatchEncoder::Gf16Afft(e) => AnyEncodeScratch::Gf16Afft(e.scratch()),
        }
    }
    fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.encode_into(data, repairs),
            AnyBatchEncoder::GoodCauchy(e) => e.encode_into(data, repairs),
            AnyBatchEncoder::Gf8Afft(e) => e.encode_into(data, repairs),
            AnyBatchEncoder::Gf16Afft(e) => e.encode_into(data, repairs),
        }
    }
    fn encode_into_with(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut Self::Scratch,
    ) -> Result<(), EncodeError> {
        match (self, scratch) {
            (AnyBatchEncoder::StandardCauchy(e), AnyEncodeScratch::Unit) => {
                e.encode_into_with(data, repairs, &mut ())
            }
            (AnyBatchEncoder::GoodCauchy(e), AnyEncodeScratch::Unit) => {
                e.encode_into_with(data, repairs, &mut ())
            }
            (AnyBatchEncoder::Gf8Afft(e), AnyEncodeScratch::Gf8Afft(s)) => {
                e.encode_into_with(data, repairs, s)
            }
            (AnyBatchEncoder::Gf16Afft(e), AnyEncodeScratch::Gf16Afft(s)) => {
                e.encode_into_with(data, repairs, s)
            }
            _ => Err(EncodeError::ScratchMismatch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(k: usize, symbol_len: usize) -> Vec<u8> {
        (0..k * symbol_len).map(|i| (i * 7 + 1) as u8).collect()
    }

    #[test]
    fn recommended_picks_engine_by_geometry() {
        // Small, low-redundancy GF(256): Good Cauchy, which also keeps the
        // incremental encoder available.
        assert_eq!(
            Profile::recommended(Field::Gf256, 10, 4, 64)
                .unwrap()
                .engine(),
            Engine::GoodCauchy
        );
        // Needs the 256th codeword position that Good Cauchy cannot address.
        assert_eq!(
            Profile::recommended(Field::Gf256, 250, 6, 64)
                .unwrap()
                .engine(),
            Engine::StandardCauchy
        );
        // The AFFT is never auto-recommended for GF(256): it is the better
        // encoder but a worse decoder at realistic erasure counts, and the
        // recommendation cannot see the erasure count. See
        // `recommended_gf8_engine`.
        assert_eq!(
            Profile::recommended(Field::Gf256, 64, 32, 64)
                .unwrap()
                .engine(),
            Engine::GoodCauchy
        );
        assert_eq!(
            Profile::recommended(Field::Gf256, 8, 4, 64)
                .unwrap()
                .engine(),
            Engine::GoodCauchy
        );
        assert_eq!(
            Profile::recommended(Field::Gf65536, 32, 4, 64)
                .unwrap()
                .engine(),
            Engine::Tower
        );
        assert_eq!(
            Profile::recommended(Field::Gf65536, 8, 4, 64)
                .unwrap()
                .engine(),
            Engine::Gf16Afft
        );
    }

    #[test]
    fn resolve_rejects_bad_geometry() {
        assert_eq!(
            Profile::resolve(Engine::GoodCauchy, 0, 4, 64),
            Err(ConfigError::ZeroDimension)
        );
        assert_eq!(
            Profile::resolve(Engine::Tower, 4, 2, 0),
            Err(ConfigError::ZeroSymbolLen)
        );
        assert_eq!(
            Profile::resolve(Engine::Gf16Afft, 4, 2, 3),
            Err(ConfigError::OddSymbolLen)
        );
        // GF(2^8) elements are one byte, so no symbol length is odd for it.
        assert!(Profile::resolve(Engine::Gf8Afft, 4, 2, 3).is_ok());
        assert_eq!(
            Profile::resolve(Engine::Gf8Afft, 200, 57, 64),
            Err(ConfigError::TooManySymbols { cap: 256 })
        );
        assert_eq!(
            Profile::resolve(Engine::GoodCauchy, 200, 100, 64),
            Err(ConfigError::TooManySymbols { cap: 255 })
        );
    }

    #[test]
    fn unsupported_modes_are_rejected() {
        let afft = Profile::resolve(Engine::Gf16Afft, 8, 4, 64).unwrap();
        assert!(matches!(
            incremental_encoder(&afft),
            Err(ConfigError::UnsupportedMode {
                engine: Engine::Gf16Afft
            })
        ));
        let gf8_afft = Profile::resolve(Engine::Gf8Afft, 8, 4, 64).unwrap();
        assert!(matches!(
            incremental_encoder(&gf8_afft),
            Err(ConfigError::UnsupportedMode {
                engine: Engine::Gf8Afft
            })
        ));
        let tower = Profile::resolve(Engine::Tower, 32, 4, 64).unwrap();
        assert!(matches!(
            batch_encoder(&tower),
            Err(ConfigError::UnsupportedMode {
                engine: Engine::Tower
            })
        ));
    }

    // Reconstruct via AnyDecoder from `k` symbols: all repairs then leading data.
    fn recover(profile: &Profile, repairs: &[Vec<u8>], original: &[u8]) {
        let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
        let mut dec = decoder(profile).unwrap();
        let mut scratch = Decoder::scratch(&dec);
        // Drop data symbol 0: feed repairs first, then data 1..k.
        for (j, repair) in repairs.iter().enumerate().take(m) {
            dec.push(k + j, repair).unwrap();
            if dec.is_complete() {
                break;
            }
        }
        for idx in 1..k {
            if dec.is_complete() {
                break;
            }
            dec.push(idx, &original[idx * s..(idx + 1) * s]).unwrap();
        }
        let mut out = vec![0u8; k * s];
        dec.finalize_into_with(&mut out, &mut scratch).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn any_batch_round_trip_across_engines() {
        for profile in [
            Profile::resolve(Engine::StandardCauchy, 10, 4, 64).unwrap(),
            Profile::resolve(Engine::GoodCauchy, 10, 4, 64).unwrap(),
            Profile::resolve(Engine::Gf8Afft, 8, 4, 64).unwrap(),
            Profile::resolve(Engine::Gf8Afft, 8, 4, 63).unwrap(),
            Profile::resolve(Engine::Gf16Afft, 8, 4, 64).unwrap(),
        ] {
            let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
            let original = data(k, s);
            let enc = batch_encoder(&profile).unwrap();
            let mut scratch = enc.scratch();
            let mut flat = vec![0u8; m * s];
            enc.encode_into_with(&original, &mut flat, &mut scratch)
                .unwrap();
            let repairs: Vec<Vec<u8>> = (0..m).map(|j| flat[j * s..(j + 1) * s].to_vec()).collect();
            recover(&profile, &repairs, &original);
        }
    }

    #[test]
    fn any_incremental_round_trip_good_cauchy_and_tower() {
        for profile in [
            Profile::resolve(Engine::GoodCauchy, 10, 4, 64).unwrap(),
            Profile::resolve(Engine::Tower, 32, 4, 64).unwrap(),
        ] {
            let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
            let original = data(k, s);
            let mut enc = incremental_encoder(&profile).unwrap();
            for idx in 0..k {
                enc.feed(idx, &original[idx * s..(idx + 1) * s]).unwrap();
            }
            let repairs: Vec<Vec<u8>> = (0..m).map(|j| enc.repair(j).unwrap().to_vec()).collect();
            recover(&profile, &repairs, &original);
        }
    }
}
