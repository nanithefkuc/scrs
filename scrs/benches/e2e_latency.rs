//! Full end-to-end latency benchmark: data-ready → first-symbol-sent → data-ready.
//!
//! Measures the complete pipeline:
//! 1. **Encode**: source data is available → all symbols ready to send.
//!    For systematic codes, data symbols are sent immediately. Repair symbols
//!    are the bottleneck.
//! 2. **Network**: k symbols transmitted (worst case: only repairs arrive).
//!    We simulate this by selecting k repair symbols.
//! 3. **Decode**: k symbols received → original data recovered.
//!
//! The total metric is the wall-clock time from when the encoder first has
//! data to when the decoder finishes reconstruction.

#![allow(clippy::needless_range_loop)]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scrs::batch::BatchCodec;
use scrs::cauchy::CauchyView;
use scrs::decoder::{LazyDecoderState, RecipeCache};
use scrs::encoder::StreamingEncoder;
use scrs::good_cauchy::GoodCauchyView;

/// Configurations spanning small to medium block sizes.
const CONFIGS: &[(usize, usize)] = &[(4, 4), (16, 8), (16, 16), (64, 32), (128, 64)];

const SYMBOL_LEN: usize = 1400;

fn make_data(k: usize, slen: usize) -> Vec<u8> {
    (0..k * slen)
        .map(|i| (i as u8).wrapping_mul(0x9E))
        .collect()
}

// ---------------------------------------------------------------------------
// Standard Cauchy: batch encode + batch decode (worst case: all repairs)
// ---------------------------------------------------------------------------

fn bench_e2e_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_standard");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let codec = BatchCodec::<CauchyView>::new(k, m, SYMBOL_LEN).unwrap();
        let _symbols = codec.encode(&data).unwrap();
        // Worst case for decode: receive as many repair symbols as possible,
        // then fill the rest with data symbols. Total = k symbols.
        let mut arrival: Vec<usize> = (k..k + m).collect();
        arrival.extend(0..k - m);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut cache = RecipeCache::new(256);
            b.iter(|| {
                // Encode
                let enc_symbols = black_box(codec.encode(black_box(&data)).unwrap());
                // Decode (simulate network by pushing k repair symbols)
                let mut dec = LazyDecoderState::<CauchyView>::new(k, m, SYMBOL_LEN).unwrap();
                for &idx in &arrival {
                    let _ = dec.push_symbol(idx, black_box(&enc_symbols[idx]));
                }
                let _ = black_box(dec.finalize_ref_cached(black_box(&mut cache)).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Good Cauchy + batch encode + batch decode (worst case: all repairs)
// ---------------------------------------------------------------------------

fn bench_e2e_good_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_good_batch");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let codec = BatchCodec::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
        let _symbols = codec.encode(&data).unwrap();
        let mut arrival: Vec<usize> = (k..k + m).collect();
        arrival.extend(0..k - m);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut cache = RecipeCache::new(256);
            b.iter(|| {
                let enc_symbols = black_box(codec.encode(black_box(&data)).unwrap());
                let mut dec = LazyDecoderState::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
                for &idx in &arrival {
                    let _ = dec.push_symbol(idx, black_box(&enc_symbols[idx]));
                }
                let _ = black_box(dec.finalize_ref_cached(black_box(&mut cache)).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Good Cauchy + streaming encode + batch decode (worst case: all repairs)
// ---------------------------------------------------------------------------

fn bench_e2e_good_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_good_streaming");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let mut arrival: Vec<usize> = (k..k + m).collect();
        arrival.extend(0..k - m);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut cache = RecipeCache::new(256);
            b.iter(|| {
                // Streaming encode: feed data symbols incrementally
                let mut enc = StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
                for i in 0..k {
                    let start = i * SYMBOL_LEN;
                    enc.feed_data_symbol(i, &data[start..start + SYMBOL_LEN])
                        .unwrap();
                }
                // Collect all n symbols
                let mut all_symbols: Vec<Vec<u8>> = Vec::with_capacity(k + m);
                for i in 0..k {
                    let start = i * SYMBOL_LEN;
                    all_symbols.push(data[start..start + SYMBOL_LEN].to_vec());
                }
                for j in 0..m {
                    all_symbols.push(enc.repair_symbol(j).unwrap().to_vec());
                }

                // Decode (worst case: only repairs arrive)
                let mut dec = LazyDecoderState::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
                for &idx in &arrival {
                    let _ = dec.push_symbol(idx, black_box(&all_symbols[idx]));
                }
                let _ = black_box(dec.finalize_ref_cached(black_box(&mut cache)).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_e2e_standard,
    bench_e2e_good_batch,
    bench_e2e_good_streaming,
);
criterion_main!(benches);
