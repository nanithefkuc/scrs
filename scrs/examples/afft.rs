//! Block-final additive-FFT encoding and payload-lazy recovery.

use scrs::afft::{LazyDecoderState, SystematicEncoder};

fn main() {
    let k = 5;
    let m = 3;
    let symbol_len = 8;
    let data: Vec<u8> = (0..k * symbol_len)
        .map(|index| ((index * 37) ^ 0x5a) as u8)
        .collect();

    let encoder = SystematicEncoder::new(k, m, symbol_len).expect("valid profile");
    let repairs = encoder.encode(&data).expect("correct data length");

    let mut decoder = LazyDecoderState::new(k, m, symbol_len).expect("matching profile");
    for data_index in 2..k {
        let start = data_index * symbol_len;
        decoder
            .push_symbol(data_index, &data[start..start + symbol_len])
            .expect("valid systematic symbol");
    }
    decoder
        .push_symbol(k, &repairs[0])
        .expect("valid repair symbol");
    decoder
        .push_symbol(k + 1, &repairs[1])
        .expect("valid repair symbol");

    let recovered = decoder.finalize_ref().expect("five symbols recover data");
    assert_eq!(recovered, data);
    println!("recovered {k} systematic symbols from three data and two aFFT repairs");
}
