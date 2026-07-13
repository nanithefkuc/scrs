//! Incrementally encode repair symbols as data arrives.

use scrs::encoder::StreamingEncoder;

fn main() {
    let (k, m, symbol_len) = (4, 2, 8);
    let mut encoder = StreamingEncoder::new(k, m, symbol_len).unwrap();
    for index in 0..k {
        let symbol = vec![index as u8; symbol_len];
        encoder.feed_data_symbol(index, &symbol).unwrap();
    }
    for repair in 0..m {
        assert_eq!(encoder.repair_symbol(repair).unwrap().len(), symbol_len);
    }
}
