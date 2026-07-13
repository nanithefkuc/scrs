//! Reuse reconstruction recipes across decodes.

use scrs::batch::GoodCauchyBatchCodec;
use scrs::decoder::{LazyDecoderState, RecipeCache};
use scrs::good_cauchy::GoodCauchyView;

fn main() {
    let (k, m, symbol_len) = (4, 2, 8);
    let data = vec![3u8; k * symbol_len];
    let codeword = GoodCauchyBatchCodec::new(k, m, symbol_len)
        .unwrap()
        .encode(&data)
        .unwrap();
    let mut decoder = LazyDecoderState::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
    for index in [0, 3, 4, 5] {
        decoder.push_symbol(index, &codeword[index]).unwrap();
    }
    let mut cache = RecipeCache::new(8);
    assert_eq!(decoder.finalize_ref_cached(&mut cache).unwrap(), data);
    assert_eq!(decoder.finalize_ref_cached(&mut cache).unwrap(), data);
    assert_eq!(cache.hits(), 1);
}
