//! Decode symbols as they arrive and finalize into a buffer.

use scrs::batch::GoodCauchyBatchCodec;
use scrs::decoder::LazyDecoderState;
use scrs::good_cauchy::GoodCauchyView;
use scrs::stream::SymbolSink;

fn main() {
    let (k, m, symbol_len) = (4, 2, 8);
    let data = vec![7u8; k * symbol_len];
    let codeword = GoodCauchyBatchCodec::new(k, m, symbol_len)
        .unwrap()
        .encode(&data)
        .unwrap();
    let mut decoder = LazyDecoderState::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
    for index in [0, 3, 4, 5] {
        decoder.push_symbol(index, &codeword[index]).unwrap();
    }
    assert!(decoder.is_complete());
    let mut recovered = vec![0u8; k * symbol_len];
    decoder.finalize_into(&mut recovered).unwrap();
    assert_eq!(recovered, data);
}
