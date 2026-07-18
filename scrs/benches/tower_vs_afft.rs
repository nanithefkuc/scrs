//! Comparative GF(65536) Tower Cauchy versus additive-FFT benchmarks.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use scrs::{afft, tower};

const SYMBOL_LEN: usize = 1400;
const CONFIGS: &[(usize, usize)] = &[(128, 64), (256, 128), (512, 256), (1024, 512)];
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
    let encoder = afft::SystematicEncoder::new(k, m, SYMBOL_LEN).unwrap();
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
                black_box(afft::SystematicEncoder::new(k, m, SYMBOL_LEN).unwrap());
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

        let afft_encoder = afft::SystematicEncoder::new(k, m, SYMBOL_LEN).unwrap();
        let mut repairs = vec![0; m * SYMBOL_LEN];
        group.bench_with_input(BenchmarkId::new("afft", &configuration), &(), |b, _| {
            b.iter(|| {
                afft_encoder
                    .encode_into(black_box(&data), black_box(&mut repairs))
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
                        let mut decoder = afft::LazyDecoderState::new(k, m, SYMBOL_LEN).unwrap();
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

criterion_group!(
    benches,
    benchmark_encoder_setup,
    benchmark_encode,
    benchmark_decode_finalize
);
criterion_main!(benches);
