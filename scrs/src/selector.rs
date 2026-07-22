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
use crate::{Gf16Engine, recommended_gf16_engine};

/// Maximum `k + m` for an engine.
const fn engine_capacity(engine: Engine) -> usize {
    match engine {
        Engine::StandardCauchy => 256,
        Engine::GoodCauchy => 255,
        Engine::Tower => 65_535,
        Engine::Afft => 65_536,
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
            Field::Gf256 => {
                if k + m <= 255 {
                    Engine::GoodCauchy
                } else {
                    Engine::StandardCauchy
                }
            }
            Field::Gf65536 => match recommended_gf16_engine(k, m) {
                Gf16Engine::Tower => Engine::Tower,
                Gf16Engine::Afft => Engine::Afft,
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
    /// GF(65536) additive FFT.
    Afft(afft::LazyDecoderState),
}

/// Reusable decode scratch for [`AnyDecoder`], matching its active engine.
pub enum AnyDecodeScratch {
    /// GF(256) recipe cache (shared by both Cauchy variants).
    Cauchy(RecipeCache),
    /// GF(65536) tower reconstruction workspace.
    Tower(tower::DecodeScratch),
    /// GF(65536) additive-FFT transform scratch.
    Afft(afft::DecodeScratch),
}

/// Build a decoder for `profile`.
pub fn decoder(profile: &Profile) -> Result<AnyDecoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    Ok(match profile.engine() {
        Engine::StandardCauchy => AnyDecoder::StandardCauchy(LazyDecoderState::new(k, m, s)?),
        Engine::GoodCauchy => AnyDecoder::GoodCauchy(LazyDecoderState::new(k, m, s)?),
        Engine::Tower => AnyDecoder::Tower(tower::LazyDecoderState::new(k, m, s)?),
        Engine::Afft => AnyDecoder::Afft(afft::LazyDecoderState::new(k, m, s)?),
    })
}

impl Coded for AnyDecoder {
    fn k(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.k(),
            AnyDecoder::GoodCauchy(d) => d.k(),
            AnyDecoder::Tower(d) => d.k(),
            AnyDecoder::Afft(d) => d.k(),
        }
    }
    fn m(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.m(),
            AnyDecoder::GoodCauchy(d) => d.m(),
            AnyDecoder::Tower(d) => d.m(),
            AnyDecoder::Afft(d) => d.m(),
        }
    }
    fn symbol_len(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.symbol_len(),
            AnyDecoder::GoodCauchy(d) => d.symbol_len(),
            AnyDecoder::Tower(d) => d.symbol_len(),
            AnyDecoder::Afft(d) => d.symbol_len(),
        }
    }
}

impl SymbolSink for AnyDecoder {
    fn push(&mut self, idx: usize, payload: &[u8]) -> Result<PushOutcome, DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.push(idx, payload),
            AnyDecoder::GoodCauchy(d) => d.push(idx, payload),
            AnyDecoder::Tower(d) => d.push(idx, payload),
            AnyDecoder::Afft(d) => d.push(idx, payload),
        }
    }
    fn is_complete(&self) -> bool {
        match self {
            AnyDecoder::StandardCauchy(d) => d.is_complete(),
            AnyDecoder::GoodCauchy(d) => d.is_complete(),
            AnyDecoder::Tower(d) => d.is_complete(),
            AnyDecoder::Afft(d) => d.is_complete(),
        }
    }
    fn finalize(self) -> Result<Vec<u8>, DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.finalize(),
            AnyDecoder::GoodCauchy(d) => d.finalize(),
            AnyDecoder::Tower(d) => d.finalize(),
            AnyDecoder::Afft(d) => d.finalize(),
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
            AnyDecoder::Afft(d) => AnyDecodeScratch::Afft(d.decode_scratch()),
        }
    }
    fn rank(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.rank(),
            AnyDecoder::GoodCauchy(d) => d.rank(),
            AnyDecoder::Tower(d) => d.rank(),
            AnyDecoder::Afft(d) => d.rank(),
        }
    }
    fn received(&self) -> usize {
        match self {
            AnyDecoder::StandardCauchy(d) => d.received(),
            AnyDecoder::GoodCauchy(d) => d.received(),
            AnyDecoder::Tower(d) => d.received(),
            AnyDecoder::Afft(d) => d.received(),
        }
    }
    fn reset(&mut self) {
        match self {
            AnyDecoder::StandardCauchy(d) => d.reset(),
            AnyDecoder::GoodCauchy(d) => d.reset(),
            AnyDecoder::Tower(d) => d.reset(),
            AnyDecoder::Afft(d) => d.reset(),
        }
    }

    fn finalize_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        match self {
            AnyDecoder::StandardCauchy(d) => d.finalize_into(out),
            AnyDecoder::GoodCauchy(d) => d.finalize_into(out),
            AnyDecoder::Tower(d) => d.finalize_into(out),
            AnyDecoder::Afft(d) => d.finalize_into(out),
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
            (AnyDecoder::Afft(d), AnyDecodeScratch::Afft(c)) => d.finalize_into_with(out, c),
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
    /// GF(65536) additive-FFT decoder.
    Afft(afft::LazyDecoderState),
}

/// Reusable scratch matching an [`AnyBatchDecoder`].
pub enum AnyBatchDecodeScratch {
    /// GF(256) reduced-erasure workspace.
    Cauchy(CauchyDecodeScratch),
    /// GF(65536) tower reconstruction workspace.
    Tower(tower::DecodeScratch),
    /// GF(65536) additive-FFT workspace.
    Afft(afft::DecodeScratch),
}

/// Build a first-class batch decoder for `profile`.
pub fn batch_decoder(profile: &Profile) -> Result<AnyBatchDecoder, ConfigError> {
    let (k, m, s) = (profile.k(), profile.m(), profile.symbol_len());
    Ok(match profile.engine() {
        Engine::StandardCauchy => AnyBatchDecoder::StandardCauchy(BatchCodec::new(k, m, s)?),
        Engine::GoodCauchy => AnyBatchDecoder::GoodCauchy(BatchCodec::new(k, m, s)?),
        Engine::Tower => AnyBatchDecoder::Tower(tower::LazyDecoderState::new(k, m, s)?),
        Engine::Afft => AnyBatchDecoder::Afft(afft::LazyDecoderState::new(k, m, s)?),
    })
}

impl Coded for AnyBatchDecoder {
    fn k(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.k(),
            AnyBatchDecoder::GoodCauchy(d) => d.k(),
            AnyBatchDecoder::Tower(d) => d.k(),
            AnyBatchDecoder::Afft(d) => d.k(),
        }
    }

    fn m(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.m(),
            AnyBatchDecoder::GoodCauchy(d) => d.m(),
            AnyBatchDecoder::Tower(d) => d.m(),
            AnyBatchDecoder::Afft(d) => d.m(),
        }
    }

    fn symbol_len(&self) -> usize {
        match self {
            AnyBatchDecoder::StandardCauchy(d) => d.symbol_len(),
            AnyBatchDecoder::GoodCauchy(d) => d.symbol_len(),
            AnyBatchDecoder::Tower(d) => d.symbol_len(),
            AnyBatchDecoder::Afft(d) => d.symbol_len(),
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
            AnyBatchDecoder::Afft(d) => AnyBatchDecodeScratch::Afft(d.decode_scratch()),
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
            (AnyBatchDecoder::Afft(d), AnyBatchDecodeScratch::Afft(s)) => {
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
    /// GF(65536) additive FFT.
    Afft(afft::SystematicEncoder),
}

/// Reusable encode scratch for [`AnyBatchEncoder`].
pub enum AnyEncodeScratch {
    /// GF(256) batch needs no workspace.
    Unit,
    /// GF(65536) additive-FFT transform workspace.
    Afft(afft::EncodeScratch),
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
        Engine::Afft => Ok(AnyBatchEncoder::Afft(afft::SystematicEncoder::new(
            k, m, s,
        )?)),
        engine => Err(ConfigError::UnsupportedMode { engine }),
    }
}

impl Coded for AnyBatchEncoder {
    fn k(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.k(),
            AnyBatchEncoder::GoodCauchy(e) => e.k(),
            AnyBatchEncoder::Afft(e) => e.k(),
        }
    }
    fn m(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.m(),
            AnyBatchEncoder::GoodCauchy(e) => e.m(),
            AnyBatchEncoder::Afft(e) => e.m(),
        }
    }
    fn symbol_len(&self) -> usize {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.symbol_len(),
            AnyBatchEncoder::GoodCauchy(e) => e.symbol_len(),
            AnyBatchEncoder::Afft(e) => e.symbol_len(),
        }
    }
}

impl BatchEncoder for AnyBatchEncoder {
    type Scratch = AnyEncodeScratch;

    fn scratch(&self) -> Self::Scratch {
        match self {
            AnyBatchEncoder::StandardCauchy(_) => AnyEncodeScratch::Unit,
            AnyBatchEncoder::GoodCauchy(_) => AnyEncodeScratch::Unit,
            AnyBatchEncoder::Afft(e) => AnyEncodeScratch::Afft(e.scratch()),
        }
    }
    fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        match self {
            AnyBatchEncoder::StandardCauchy(e) => e.encode_into(data, repairs),
            AnyBatchEncoder::GoodCauchy(e) => e.encode_into(data, repairs),
            AnyBatchEncoder::Afft(e) => e.encode_into(data, repairs),
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
            (AnyBatchEncoder::Afft(e), AnyEncodeScratch::Afft(s)) => {
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
        assert_eq!(
            Profile::recommended(Field::Gf256, 10, 4, 64)
                .unwrap()
                .engine(),
            Engine::GoodCauchy
        );
        assert_eq!(
            Profile::recommended(Field::Gf256, 250, 6, 64)
                .unwrap()
                .engine(),
            Engine::StandardCauchy
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
            Engine::Afft
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
            Profile::resolve(Engine::Afft, 4, 2, 3),
            Err(ConfigError::OddSymbolLen)
        );
        assert_eq!(
            Profile::resolve(Engine::GoodCauchy, 200, 100, 64),
            Err(ConfigError::TooManySymbols { cap: 255 })
        );
    }

    #[test]
    fn unsupported_modes_are_rejected() {
        let afft = Profile::resolve(Engine::Afft, 8, 4, 64).unwrap();
        assert!(matches!(
            incremental_encoder(&afft),
            Err(ConfigError::UnsupportedMode {
                engine: Engine::Afft
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
    fn any_batch_round_trip_gf256_and_afft() {
        for profile in [
            Profile::resolve(Engine::StandardCauchy, 10, 4, 64).unwrap(),
            Profile::resolve(Engine::GoodCauchy, 10, 4, 64).unwrap(),
            Profile::resolve(Engine::Afft, 8, 4, 64).unwrap(),
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
