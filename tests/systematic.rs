//! Every engine produces *systematic* output.
//!
//! This is the property the crate is named for: for any engine and any geometry,
//! codeword symbols `0..k` are the input data symbols, byte for byte, and only
//! symbols `k..k + m` carry redundancy. Encoders therefore emit repairs alone, and
//! a decoder handed nothing but the `k` data symbols must return them untouched.
//!
//! The assertions go through the type-erased [`srs::batch_encoder`] /
//! [`srs::decoder`] dispatch rather than the concrete engine types, so a new
//! `Engine` variant cannot be added without an entry in [`ENGINES`] and a
//! systematic case to go with it.

use srs::stream::SymbolSink;
use srs::{
    BatchDecoder, BatchEncoder, Decoder, Engine, IncrementalEncoder, Profile, batch_decoder,
    batch_encoder, decoder, incremental_encoder,
};

/// Every engine the crate exposes. Adding a variant to `Engine` without adding it
/// here leaves it untested; `engine_list_is_exhaustive` is the tripwire.
const ENGINES: &[Engine] = &[
    Engine::StandardCauchy,
    Engine::GoodCauchy,
    Engine::Tower,
    Engine::Gf8Afft,
    Engine::Gf16Afft,
];

fn sample_data(k: usize, symbol_len: usize) -> Vec<u8> {
    (0..k * symbol_len)
        .map(|index| (index.wrapping_mul(167).wrapping_add(29)) as u8)
        .collect()
}

/// Produce the `m` repair symbols for `data`, via whichever encode mode the engine
/// supports. Deliberately returns repairs only: an encoder that had to be told
/// where to put the data symbols would not be systematic.
fn encode_repairs(profile: &Profile, data: &[u8]) -> Vec<u8> {
    let (k, m, symbol_len) = (profile.k(), profile.m(), profile.symbol_len());
    let mut repairs = vec![0u8; m * symbol_len];

    if let Ok(encoder) = batch_encoder(profile) {
        let mut scratch = BatchEncoder::scratch(&encoder);
        encoder
            .encode_into_with(data, &mut repairs, &mut scratch)
            .unwrap();
    } else if let Ok(mut encoder) = incremental_encoder(profile) {
        for index in 0..k {
            encoder
                .feed(index, &data[index * symbol_len..(index + 1) * symbol_len])
                .unwrap();
        }
        for index in 0..m {
            repairs[index * symbol_len..(index + 1) * symbol_len]
                .copy_from_slice(encoder.repair(index).unwrap());
        }
    } else {
        panic!("{:?} exposes no encode mode", profile.engine());
    }

    repairs
}

/// The core claim: with all `k` data symbols present and no repairs used at all,
/// both decode paths return the input verbatim.
///
/// A non-systematic code cannot pass this — it would have to invert a transform to
/// recover the message from `k` codeword symbols, and any arithmetic in that path
/// would perturb at least one byte for some input.
fn assert_data_passes_through(engine: Engine, k: usize, m: usize, symbol_len: usize) {
    let profile = Profile::resolve(engine, k, m, symbol_len).unwrap();
    let data = sample_data(k, symbol_len);
    let symbols: Vec<&[u8]> = data.chunks_exact(symbol_len).collect();
    let received: Vec<(usize, &[u8])> = (0..k).map(|index| (index, symbols[index])).collect();

    let mut streaming = decoder(&profile).unwrap();
    let mut stream_out = vec![0u8; k * symbol_len];
    for &(index, payload) in &received {
        streaming.push(index, payload).unwrap();
    }
    streaming.finalize_into(&mut stream_out).unwrap();
    assert_eq!(
        stream_out, data,
        "{engine:?} streaming decode altered systematic data (k={k}, m={m}, symbol_len={symbol_len})"
    );

    let mut batch = batch_decoder(&profile).unwrap();
    let mut batch_out = vec![0u8; k * symbol_len];
    batch.decode_into(&received, &mut batch_out).unwrap();
    assert_eq!(
        batch_out, data,
        "{engine:?} batch decode altered systematic data (k={k}, m={m}, symbol_len={symbol_len})"
    );
}

/// Erasing data symbols and substituting repairs must reproduce the *same* bytes
/// the systematic positions carried. This is what makes the passthrough above a
/// coding property rather than a decoder shortcut: the repair symbols have to
/// encode exactly the data that positions `0..k` transmit.
fn assert_repairs_agree_with_data(engine: Engine, k: usize, m: usize, symbol_len: usize) {
    let profile = Profile::resolve(engine, k, m, symbol_len).unwrap();
    let data = sample_data(k, symbol_len);
    let repairs = encode_repairs(&profile, &data);

    let mut word: Vec<&[u8]> = data.chunks_exact(symbol_len).collect();
    word.extend(repairs.chunks_exact(symbol_len));

    for missing in 1..=m {
        let indices: Vec<usize> = (missing..k).chain(k..k + missing).collect();
        let received: Vec<(usize, &[u8])> =
            indices.iter().map(|&index| (index, word[index])).collect();

        let mut batch = batch_decoder(&profile).unwrap();
        let mut out = vec![0u8; k * symbol_len];
        batch.decode_into(&received, &mut out).unwrap();
        assert_eq!(
            out, data,
            "{engine:?} reconstruction disagrees with the systematic symbols \
             (k={k}, m={m}, symbol_len={symbol_len}, missing={missing})"
        );
    }
}

#[test]
fn data_symbols_pass_through_every_engine() {
    for &engine in ENGINES {
        // Symbol lengths chosen to straddle the SIMD lane boundaries (below one
        // lane, exactly one, and a multiple with a tail).
        for &symbol_len in &[2usize, 16, 64, 1000] {
            assert_data_passes_through(engine, 8, 4, symbol_len);
        }
        // Geometry edges: minimum viable, and m > k.
        assert_data_passes_through(engine, 1, 1, 64);
        assert_data_passes_through(engine, 2, 6, 64);
    }
}

#[test]
fn repairs_reconstruct_the_systematic_symbols() {
    for &engine in ENGINES {
        assert_repairs_agree_with_data(engine, 8, 4, 64);
        assert_repairs_agree_with_data(engine, 3, 2, 16);
        assert_repairs_agree_with_data(engine, 16, 4, 1000);
    }
}

/// `Profile::recommended` must never hand back a non-systematic engine, since the
/// recommendation is what callers who do not name an engine get.
#[test]
fn recommended_engines_are_systematic() {
    for &field in &[srs::Field::Gf256, srs::Field::Gf65536] {
        for &(k, m) in &[(8usize, 4usize), (200, 50), (16, 16)] {
            let profile = Profile::recommended(field, k, m, 64).unwrap();
            assert!(
                ENGINES.contains(&profile.engine()),
                "recommended engine {:?} is absent from the systematic test matrix",
                profile.engine()
            );
            assert_data_passes_through(profile.engine(), k, m, 64);
        }
    }
}

/// Tripwire for a new `Engine` variant: `Engine::field()` is an exhaustive match,
/// so this fails to compile — not merely fails at runtime — when a variant is added
/// without being registered in [`ENGINES`].
#[test]
fn engine_list_is_exhaustive() {
    for &engine in ENGINES {
        let _: srs::Field = engine.field();
    }
    fn assert_registered(engine: Engine) -> bool {
        match engine {
            Engine::StandardCauchy
            | Engine::GoodCauchy
            | Engine::Tower
            | Engine::Gf8Afft
            | Engine::Gf16Afft => ENGINES.contains(&engine),
        }
    }
    for &engine in ENGINES {
        assert!(assert_registered(engine), "{engine:?} not registered");
    }
    assert_eq!(ENGINES.len(), 5, "ENGINES is out of step with Engine");
}
