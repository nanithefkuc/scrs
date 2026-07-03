//! Encoder latency benchmarks for the data-avail-to-first-send metric.
//!
//! The key metric for media streaming is:
//!   "time from when source data is first available to when the first symbol
//!    is sent on the wire."
//!
//! For a systematic code, data symbols have zero latency (they are sent
//! immediately as they arrive). Repair symbols are the bottleneck. We
//! measure:
//!
//! 1. **Batch encode latency**: time for `BatchCodec::encode` to produce all
//!    `n` symbols after the full `k * symbol_len` data block is available.
//!    The first repair can only be sent after this completes.
//!
//! 2. **Streaming incremental latency**: time for `StreamingEncoder` to
//!    process the final data symbol and finish all repair symbols. With
//!    incremental encoding, repairs are ready as soon as the k-th data
//!    symbol is fed.
//!
//! 3. **Per-symbol incremental cost**: the cost of `feed_data_symbol` for a
//!    single symbol, which is the background work done while streaming.

#![allow(clippy::needless_range_loop)]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scrs::batch::BatchCodec;
use scrs::encoder::StreamingEncoder;
use scrs::good_cauchy::GoodCauchyView;

/// Configurations spanning small to medium block sizes.
const CONFIGS: &[(usize, usize)] = &[(4, 4), (16, 8), (16, 16), (64, 32), (128, 64)];

const SYMBOL_LEN: usize = 1400;

/// Generate deterministic source data.
fn make_data(k: usize, slen: usize) -> Vec<u8> {
    (0..k * slen)
        .map(|i| (i as u8).wrapping_mul(0x9E))
        .collect()
}

// ---------------------------------------------------------------------------
// Batch encode (standard Cauchy)
// ---------------------------------------------------------------------------

fn bench_batch_encode_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode_standard");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let codec = BatchCodec::<scrs::cauchy::CauchyView>::new(k, m, SYMBOL_LEN)
            .expect("codec construction failed");

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            b.iter(|| {
                let _ = black_box(codec.encode(black_box(&data)).unwrap());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Batch encode (Good Cauchy)
// ---------------------------------------------------------------------------

fn bench_batch_encode_good(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode_good");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let cauchy = GoodCauchyView::new(k, m).expect("good cauchy construction failed");

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            b.iter(|| {
                // Replicate the batch encode logic with Good Cauchy
                let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(k + m);
                for i in 0..k {
                    let start = i * SYMBOL_LEN;
                    symbols.push(data[start..start + SYMBOL_LEN].to_vec());
                }
                for j in 0..m {
                    let mut repair = vec![0u8; SYMBOL_LEN];
                    for i in 0..k {
                        let coeff = cauchy.get(i, j);
                        if coeff == scrs::gf256::GfElem::ZERO {
                            continue;
                        }
                        let data_start = i * SYMBOL_LEN;
                        let data_slice = &data[data_start..data_start + SYMBOL_LEN];
                        if coeff == scrs::gf256::GfElem::ONE {
                            for (out, &b) in repair.iter_mut().zip(data_slice.iter()) {
                                *out ^= b;
                            }
                        } else {
                            for (out, &b) in repair.iter_mut().zip(data_slice.iter()) {
                                *out ^= scrs::gf256::GfElem(b).mul(coeff).0;
                            }
                        }
                    }
                    symbols.push(repair);
                }
                let _ = black_box(symbols);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Streaming incremental: feed all k symbols (Good Cauchy)
// ---------------------------------------------------------------------------

fn bench_streaming_feed_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_feed_all_good");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            b.iter_with_setup(
                || {
                    let mut enc = StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
                    // Pre-feed k-1 symbols so the timed region is just the last one
                    for i in 0..k - 1 {
                        let start = i * SYMBOL_LEN;
                        enc.feed_data_symbol(i, &data[start..start + SYMBOL_LEN])
                            .unwrap();
                    }
                    enc
                },
                |mut enc| {
                    let start = (k - 1) * SYMBOL_LEN;
                    let _ = black_box(
                        enc.feed_data_symbol(k - 1, &data[start..start + SYMBOL_LEN])
                            .unwrap(),
                    );
                },
            );
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Streaming: full encode (feed k symbols + collect m repairs)
// ---------------------------------------------------------------------------

fn bench_streaming_full_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_full_encode_good");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            b.iter(|| {
                let mut enc = StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
                for i in 0..k {
                    let start = i * SYMBOL_LEN;
                    enc.feed_data_symbol(i, &data[start..start + SYMBOL_LEN])
                        .unwrap();
                }
                for j in 0..m {
                    let _ = black_box(enc.repair_symbol(j).unwrap());
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Per-symbol incremental cost (Good Cauchy)
// ---------------------------------------------------------------------------

fn bench_streaming_per_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_per_symbol_good");
    group.sample_size(100);

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                let mut enc = StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
                let start = i * SYMBOL_LEN;
                let _ = black_box(
                    enc.feed_data_symbol(i, &data[start..start + SYMBOL_LEN])
                        .unwrap(),
                );
                i = (i + 1) % k;
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
    bench_batch_encode_standard,
    bench_batch_encode_good,
    bench_streaming_feed_all,
    bench_streaming_full_encode,
    bench_streaming_per_symbol,
);
criterion_main!(benches);
