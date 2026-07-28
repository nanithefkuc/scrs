use proptest::prelude::*;

use super::{LazyDecoderState, SystematicEncoder};
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

fn afft_codeword(k: usize, m: usize, symbol_len: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let encoder = SystematicEncoder::new(k, m, symbol_len).unwrap();
    let mut word: Vec<_> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
    word.extend(encoder.encode(data).unwrap());
    word
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn differential_fuzz_against_tower_cauchy(
        k in 1usize..=12,
        m in 1usize..=12,
        field_elements in 1usize..=8,
        data_source in prop::collection::vec(any::<u8>(), 12 * 16),
        ordering in prop::collection::vec(any::<u16>(), 24),
    ) {
        let symbol_len = field_elements * 2;
        let data = &data_source[..k * symbol_len];
        let tower_word = tower_codeword(k, m, symbol_len, data);
        let afft_word = afft_codeword(k, m, symbol_len, data);

        let mut indices: Vec<_> = (0..k + m).collect();
        indices.sort_by_key(|&index| (ordering[index], index));
        indices.truncate(k);

        let mut tower_decoder = crate::tower::LazyDecoderState::new(k, m, symbol_len).unwrap();
        let mut afft_decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
        for index in indices {
            tower_decoder.push_symbol(index, &tower_word[index]).unwrap();
            afft_decoder.push_symbol(index, &afft_word[index]).unwrap();
        }

        let tower_output = tower_decoder.finalize().unwrap();
        let afft_output = afft_decoder.finalize().unwrap();
        prop_assert_eq!(&tower_output, data);
        prop_assert_eq!(&afft_output, data);
        prop_assert_eq!(afft_output, tower_output);
    }
}
