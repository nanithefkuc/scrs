//! Block-final additive-FFT encoding and payload-lazy recovery.

use scrs::afft::{LazyDecoderState, SystematicEncoder};
use scrs::{BatchEncoder, Decoder};

fn main() {
    let k = 5;
    let m = 3;
    let symbol_len = 8;
    let data: Vec<u8> = (0..k * symbol_len)
        .map(|index| ((index * 37) ^ 0x5a) as u8)
        .collect();

    let encoder = SystematicEncoder::new(k, m, symbol_len).expect("valid profile");
    let mut encode_scratch = encoder.scratch();
    let mut repairs = vec![0u8; m * symbol_len];
    encoder
        .encode_into_with(&data, &mut repairs, &mut encode_scratch)
        .expect("correct data length");

    let mut decoder = LazyDecoderState::new(k, m, symbol_len).expect("matching profile");
    for data_index in 2..k {
        let start = data_index * symbol_len;
        decoder
            .push_symbol(data_index, &data[start..start + symbol_len])
            .expect("valid systematic symbol");
    }
    decoder
        .push_symbol(k, &repairs[0..symbol_len])
        .expect("valid repair symbol");
    decoder
        .push_symbol(k + 1, &repairs[symbol_len..2 * symbol_len])
        .expect("valid repair symbol");

    let mut recovered = vec![0u8; k * symbol_len];
    let mut decode_scratch = decoder.scratch();
    decoder
        .finalize_into_with(&mut recovered, &mut decode_scratch)
        .expect("five symbols recover data");
    assert_eq!(recovered, data);
    println!("recovered {k} systematic symbols from three data and two aFFT repairs");
}
