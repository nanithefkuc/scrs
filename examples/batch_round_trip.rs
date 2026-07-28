//! Batch encode and decode round trip.

use scrs::batch::GoodCauchyBatchCodec;
use scrs::decoder::LazyDecoderState;
use scrs::good_cauchy::GoodCauchyView;

fn main() {
    let (k, m, symbol_len) = (4, 2, 8);
    let data = (0..k * symbol_len).map(|i| i as u8).collect::<Vec<_>>();
    let codec = GoodCauchyBatchCodec::new(k, m, symbol_len).unwrap();
    let codeword = codec.encode(&data).unwrap();

    let mut decoder = LazyDecoderState::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
    for index in [0, 3, 4, 5] {
        decoder.push_symbol(index, &codeword[index]).unwrap();
    }
    assert_eq!(decoder.finalize_ref().unwrap(), data);
}
