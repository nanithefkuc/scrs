//! Comparative benchmarks across SRS's coding engines.
//!
//! GF(65536): Tower Cauchy versus additive FFT.
//! GF(256): Good Cauchy versus additive FFT — the data behind
//! `recommended_gf8_engine`'s crossover.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use srs::batch::GoodCauchyBatchCodec;
use srs::codec::BatchDecoder;
use srs::{BatchEncoder, afft, tower};

const SYMBOL_LEN: usize = 1400;
const CONFIGS: &[(usize, usize)] = &[(100, 20), (128, 64), (256, 128), (512, 256), (1024, 512)];
const ERASURE_COUNTS: &[usize] = &[1, 4, 16, 32, 64];

fn make_data(k: usize, symbol_len: usize) -> Vec<u8> {
    (0..k * symbol_len)
        .map(|index| ((index as u8).wrapping_mul(0x9d)) ^ 0x5a)
        .collect()
}

fn tower_codeword(k: usize, m: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let mut encoder = tower::StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
    for (index, symbol) in data.chunks_exact(SYMBOL_LEN).enumerate() {
        encoder.feed_data_symbol(index, symbol).unwrap();
    }
    let mut word: Vec<_> = data.chunks_exact(SYMBOL_LEN).map(<[u8]>::to_vec).collect();
    word.extend(encoder.into_repairs());
    word
}

fn afft_codeword(k: usize, m: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let encoder = afft::Gf16Encoder::new(k, m, SYMBOL_LEN).unwrap();
    let mut word: Vec<_> = data.chunks_exact(SYMBOL_LEN).map(<[u8]>::to_vec).collect();
    word.extend(encoder.encode(data).unwrap());
    word
}

fn arrival_pattern(k: usize, erasures: usize) -> Vec<usize> {
    let mut arrival: Vec<_> = (erasures..k).collect();
    arrival.extend(k..k + erasures);
    arrival
}

fn benchmark_encoder_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("gf65536_encoder_setup");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for &(k, m) in CONFIGS {
        let configuration = format!("k{k}_m{m}_s{SYMBOL_LEN}");
        group.bench_with_input(BenchmarkId::new("tower", &configuration), &(), |b, _| {
            b.iter(|| {
                black_box(tower::StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap());
            });
        });
        group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
            b.iter(|| {
                black_box(afft::Gf16Encoder::new(k, m, SYMBOL_LEN).unwrap());
            });
        });
    }
    group.finish();
}

fn benchmark_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("gf65536_encode_hot");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for &(k, m) in CONFIGS {
        let configuration = format!("k{k}_m{m}_s{SYMBOL_LEN}");
        let data = make_data(k, SYMBOL_LEN);
        group.throughput(Throughput::Bytes((k * SYMBOL_LEN) as u64));

        let mut tower_encoder = tower::StreamingEncoder::new(k, m, SYMBOL_LEN).unwrap();
        group.bench_with_input(BenchmarkId::new("tower", &configuration), &(), |b, _| {
            b.iter(|| {
                tower_encoder.reset();
                for (index, symbol) in data.chunks_exact(SYMBOL_LEN).enumerate() {
                    tower_encoder
                        .feed_data_symbol(index, black_box(symbol))
                        .unwrap();
                }
                black_box(tower_encoder.repair_symbol(m - 1).unwrap());
            });
        });

        let afft_encoder = afft::Gf16Encoder::new(k, m, SYMBOL_LEN).unwrap();
        let mut repairs = vec![0; m * SYMBOL_LEN];
        let mut afft_scratch = afft_encoder.encode_scratch();
        group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
            b.iter(|| {
                afft_encoder
                    .encode_into_with(black_box(&data), black_box(&mut repairs), &mut afft_scratch)
                    .unwrap();
                black_box(&repairs);
            });
        });
    }
    group.finish();
}

fn benchmark_decode_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("gf65536_decode_finalize");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for &(k, m) in CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let tower_word = tower_codeword(k, m, &data);
        let afft_word = afft_codeword(k, m, &data);

        for &erasures in ERASURE_COUNTS.iter().filter(|&&count| count <= m) {
            let configuration = format!("k{k}_m{m}_r{erasures}_s{SYMBOL_LEN}");
            let arrival = arrival_pattern(k, erasures);
            group.throughput(Throughput::Bytes((k * SYMBOL_LEN) as u64));

            group.bench_with_input(BenchmarkId::new("tower", &configuration), &(), |b, _| {
                b.iter_batched(
                    || {
                        let mut decoder = tower::LazyDecoderState::new(k, m, SYMBOL_LEN).unwrap();
                        for &index in &arrival {
                            decoder.push_symbol(index, &tower_word[index]).unwrap();
                        }
                        decoder
                    },
                    |decoder| black_box(decoder.finalize_ref().unwrap()),
                    BatchSize::LargeInput,
                );
            });

            group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
                b.iter_batched(
                    || {
                        let mut decoder = afft::Gf16Decoder::new(k, m, SYMBOL_LEN).unwrap();
                        for &index in &arrival {
                            decoder.push_symbol(index, &afft_word[index]).unwrap();
                        }
                        decoder
                    },
                    |decoder| black_box(decoder.finalize_ref().unwrap()),
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}


/// GF(256) geometries spanning the Good-Cauchy/AFFT crossover. All are
/// high-redundancy (`m == k / 2`), which is where the additive FFT should win
/// once `k` is large enough to amortise the transform's constant factor.
const GF8_CONFIGS: &[(usize, usize)] = &[(8, 4), (16, 8), (32, 16), (64, 32), (100, 50), (160, 80)];

fn gf8_afft_codeword(k: usize, m: usize, data: &[u8]) -> Vec<Vec<u8>> {
    let encoder = afft::Gf8Encoder::new(k, m, SYMBOL_LEN).unwrap();
    let mut word: Vec<_> = data.chunks_exact(SYMBOL_LEN).map(<[u8]>::to_vec).collect();
    word.extend(encoder.encode(data).unwrap());
    word
}

fn benchmark_gf8_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("gf256_encode_hot");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for &(k, m) in GF8_CONFIGS {
        let configuration = format!("k{k}_m{m}_s{SYMBOL_LEN}");
        let data = make_data(k, SYMBOL_LEN);
        group.throughput(Throughput::Bytes((k * SYMBOL_LEN) as u64));

        let cauchy = GoodCauchyBatchCodec::new(k, m, SYMBOL_LEN).unwrap();
        let mut repairs = vec![0u8; m * SYMBOL_LEN];
        group.bench_with_input(BenchmarkId::new("good_cauchy", &configuration), &(), |b, _| {
            b.iter(|| {
                cauchy
                    .encode_into(black_box(&data), black_box(&mut repairs))
                    .unwrap();
                black_box(&repairs);
            });
        });

        let encoder = afft::Gf8Encoder::new(k, m, SYMBOL_LEN).unwrap();
        let mut afft_repairs = vec![0u8; m * SYMBOL_LEN];
        let mut scratch = encoder.encode_scratch();
        group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
            b.iter(|| {
                encoder
                    .encode_into_with(
                        black_box(&data),
                        black_box(&mut afft_repairs),
                        &mut scratch,
                    )
                    .unwrap();
                black_box(&afft_repairs);
            });
        });
    }
    group.finish();
}

fn benchmark_gf8_decode_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("gf256_decode_finalize");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for &(k, m) in GF8_CONFIGS {
        let data = make_data(k, SYMBOL_LEN);
        let cauchy_word = GoodCauchyBatchCodec::new(k, m, SYMBOL_LEN)
            .unwrap()
            .encode(&data)
            .unwrap();
        let afft_word = gf8_afft_codeword(k, m, &data);

        for &erasures in ERASURE_COUNTS.iter().filter(|&&count| count <= m) {
            let configuration = format!("k{k}_m{m}_r{erasures}_s{SYMBOL_LEN}");
            let arrival = arrival_pattern(k, erasures);
            group.throughput(Throughput::Bytes((k * SYMBOL_LEN) as u64));

            let received: Vec<(usize, &[u8])> = arrival
                .iter()
                .map(|&index| (index, cauchy_word[index].as_slice()))
                .collect();
            let cauchy = GoodCauchyBatchCodec::new(k, m, SYMBOL_LEN).unwrap();
            let mut cauchy_scratch = BatchDecoder::scratch(&cauchy);
            let mut out = vec![0u8; k * SYMBOL_LEN];
            group.bench_with_input(
                BenchmarkId::new("good_cauchy", &configuration),
                &(),
                |b, _| {
                    b.iter(|| {
                        cauchy
                            .decode_into_with(
                                black_box(&received),
                                black_box(&mut out),
                                &mut cauchy_scratch,
                            )
                            .unwrap();
                        black_box(&out);
                    });
                },
            );

            let mut missing_out = vec![0u8; erasures * SYMBOL_LEN];
            group.bench_with_input(
                BenchmarkId::new("good_cauchy_reconstruct", &configuration),
                &(),
                |b, _| {
                    b.iter(|| {
                        cauchy
                            .reconstruct_missing_into_with(
                                black_box(&received),
                                black_box(&mut missing_out),
                                &mut cauchy_scratch,
                            )
                            .unwrap();
                        black_box(&missing_out);
                    });
                },
            );

            let plan = cauchy.prepare_decode(&arrival).unwrap();
            group.bench_with_input(
                BenchmarkId::new("good_cauchy_prepared", &configuration),
                &(),
                |b, _| {
                    b.iter(|| {
                        plan.reconstruct_missing_into(
                            black_box(&received),
                            black_box(&mut missing_out),
                        )
                        .unwrap();
                        black_box(&missing_out);
                    });
                },
            );

            let afft_received: Vec<(usize, &[u8])> = arrival
                .iter()
                .map(|&index| (index, afft_word[index].as_slice()))
                .collect();
            let mut afft = afft::Gf8BatchDecoder::new(k, m, SYMBOL_LEN).unwrap();
            let mut afft_scratch = afft.decode_scratch();
            group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
                b.iter(|| {
                    afft.decode_into_with(
                        black_box(&afft_received),
                        black_box(&mut out),
                        &mut afft_scratch,
                    )
                    .unwrap();
                    black_box(&out);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_encoder_setup,
    benchmark_encode,
    benchmark_decode_finalize,
    benchmark_gf8_encode,
    benchmark_gf8_decode_finalize
);
criterion_main!(benches);
