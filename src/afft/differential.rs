//! Differential fuzzing of each additive-FFT engine against an independent
//! Cauchy engine over the same field.
//!
//! The AFFT engines share no code with the Cauchy ones — different construction,
//! different reconstruction algorithm, different kernels — so agreement on the
//! decoded message across random geometries, symbol lengths, and arrival orders
//! is strong evidence that both are right.

use proptest::prelude::*;

use super::{Gf8Decoder, Gf8Encoder, Gf16Decoder, Gf16Encoder};
use crate::batch::BatchCodec;
use crate::good_cauchy::GoodCauchyView;
use crate::stream::SymbolSink;

fn tower_codeword(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let mut encoder = crate::tower::StreamingEncoder::new(k, m, symbol_len).unwrap();
    for (index, symbol) in data.chunks_exact(symbol_len).enumerate() {
        encoder.feed_data_symbol(index, symbol).unwrap();
    }
    let mut word: Vec<_> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
    word.extend(encoder.into_repairs());
    word
}

fn good_cauchy_codeword(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let codec = BatchCodec::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
    codec.encode(data).unwrap()
}

fn gf16_afft_codeword(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let encoder = Gf16Encoder::new(k, m, symbol_len).unwrap();
    let mut word: Vec<_> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
    word.extend(encoder.encode(data).unwrap());
    word
}

fn gf8_afft_codeword(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let encoder = Gf8Encoder::new(k, m, symbol_len).unwrap();
    let mut word: Vec<_> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
    word.extend(encoder.encode(data).unwrap());
    word
}

/// The `k` codeword positions that arrive, in a pseudo-random order.
fn arrival(k: usize, m: usize, ordering: &[u16]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..k + m).collect();
    indices.sort_by_key(|&index| (ordering[index], index));
    indices.truncate(k);
    indices
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn gf16_differential_fuzz_against_tower_cauchy(
        k in 1usize..=12,
        m in 1usize..=12,
        field_elements in 1usize..=8,
        data_source in prop::collection::vec(any::<u8>(), 12 * 16),
        ordering in prop::collection::vec(any::<u16>(), 24),
    ) {
        let symbol_len = field_elements * 2;
        let data = &data_source[..k * symbol_len];
        let tower_word = tower_codeword(k, m, symbol_len, data);
        let afft_word = gf16_afft_codeword(k, m, symbol_len, data);

        let mut tower_decoder = crate::tower::LazyDecoderState::new(k, m, symbol_len).unwrap();
        let mut afft_decoder = Gf16Decoder::new(k, m, symbol_len).unwrap();
        for index in arrival(k, m, &ordering) {
            tower_decoder.push_symbol(index, &tower_word[index]).unwrap();
            afft_decoder.push_symbol(index, &afft_word[index]).unwrap();
        }

        let tower_output = tower_decoder.finalize().unwrap();
        let afft_output = afft_decoder.finalize().unwrap();
        prop_assert_eq!(&tower_output, data);
        prop_assert_eq!(&afft_output, data);
        prop_assert_eq!(afft_output, tower_output);
    }

    /// GF(2^8), including the odd symbol lengths GF(2^16) cannot express.
    #[test]
    fn gf8_differential_fuzz_against_good_cauchy(
        k in 1usize..=12,
        m in 1usize..=12,
        symbol_len in 1usize..=17,
        data_source in prop::collection::vec(any::<u8>(), 12 * 17),
        ordering in prop::collection::vec(any::<u16>(), 24),
    ) {
        let data = &data_source[..k * symbol_len];
        let cauchy_word = good_cauchy_codeword(k, m, symbol_len, data);
        let afft_word = gf8_afft_codeword(k, m, symbol_len, data);

        let mut cauchy_decoder =
            crate::decoder::LazyDecoderState::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
        let mut afft_decoder = Gf8Decoder::new(k, m, symbol_len).unwrap();
        for index in arrival(k, m, &ordering) {
            cauchy_decoder.push_symbol(index, &cauchy_word[index]).unwrap();
            afft_decoder.push_symbol(index, &afft_word[index]).unwrap();
        }

        let cauchy_output = cauchy_decoder.finalize().unwrap();
        let afft_output = afft_decoder.finalize().unwrap();
        prop_assert_eq!(&cauchy_output, data);
        prop_assert_eq!(&afft_output, data);
        prop_assert_eq!(afft_output, cauchy_output);
    }
}

/// The GF(2^8) domain boundary: `k + m == 256` is the largest legal geometry, and
/// `257` must be rejected rather than silently wrapping into a smaller domain.
#[test]
fn gf8_domain_boundary() {
    assert!(Gf8Encoder::new(128, 128, 4).is_ok());
    assert!(Gf8Decoder::new(128, 128, 4).is_ok());
    assert!(Gf8Encoder::new(200, 56, 4).is_ok());
    assert!(Gf8Encoder::new(200, 57, 4).is_err());
    assert!(Gf8Encoder::new(256, 1, 4).is_err());

    // Round-trip at the boundary with every repair symbol in use.
    let (k, m, symbol_len) = (128usize, 128usize, 3usize);
    let data: Vec<u8> = (0..k * symbol_len).map(|i| (i * 31 + 7) as u8).collect();
    let word = gf8_afft_codeword(k, m, symbol_len, &data);
    let mut decoder = Gf8Decoder::new(k, m, symbol_len).unwrap();
    for index in k..k + m {
        decoder.push_symbol(index, &word[index]).unwrap();
    }
    assert_eq!(decoder.finalize_ref().unwrap(), data);
}

/// GF(2^8) accepts any symbol length; GF(2^16) requires whole two-byte elements.
#[test]
fn symbol_length_rules_follow_element_width() {
    for symbol_len in 1..=8 {
        assert!(
            Gf8Encoder::new(4, 2, symbol_len).is_ok(),
            "GF(2^8) rejected symbol_len={symbol_len}"
        );
        assert_eq!(
            Gf16Encoder::new(4, 2, symbol_len).is_ok(),
            symbol_len % 2 == 0,
            "GF(2^16) parity rule wrong at symbol_len={symbol_len}"
        );
    }
    assert!(Gf8Encoder::new(4, 2, 0).is_err());
}
