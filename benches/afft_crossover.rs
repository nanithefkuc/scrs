//! Targeted-versus-locator crossover measurement for the additive-FFT decoder.
//!
//! Both recovery paths reconstruct every erasure pattern, so the choice is
//! purely a cost question and `afft::crossover` encodes the answer. This bench
//! is the evidence behind that model: it times prepared plans forced onto each
//! path across `k`, symbol length, and erasure count, so the crossover can be
//! read off directly rather than guessed.
//!
//! Steady state only — the plan hoists the pattern-dependent solve out, which
//! is what a decoder with a resident plan or a warm scratch memo actually pays.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use srs::afft::{self, RecoveryPath};
use srs::codec::BatchEncoder;

/// GF(2^8) geometries spanning the domain's `k` range at fixed `m = k / 2`.
const GF8_CONFIGS: &[(usize, usize)] = &[(16, 8), (64, 32), (160, 80)];
/// GF(2^16) geometry wide enough that the padded domain dwarfs `k`.
const GF16_CONFIGS: &[(usize, usize)] = &[(512, 256)];
const SYMBOL_LENS: &[usize] = &[64, 1400];
const ERASURES: &[usize] = &[1, 2, 4, 8, 16, 24, 32, 48, 64];

fn codeword<F: afft::Field>(k: usize, m: usize, symbol_len: usize) -> Vec<u8> {
    let mut word = vec![0u8; (k + m) * symbol_len];
    let (data, repairs) = word.split_at_mut(k * symbol_len);
    for (index, byte) in data.iter_mut().enumerate() {
        *byte = (index.wrapping_mul(131) + 7) as u8;
    }
    let encoder = afft::SystematicEncoder::<F>::new(k, m, symbol_len).unwrap();
    let mut scratch = encoder.encode_scratch();
    encoder
        .encode_into_with(data, repairs, &mut scratch)
        .unwrap();
    word
}

fn sweep<F: afft::Field>(c: &mut Criterion, label: &str, configs: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("{label}_afft_crossover"));
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));
    group.warm_up_time(Duration::from_millis(200));

    for &(k, m) in configs {
        for &symbol_len in SYMBOL_LENS {
            let word = codeword::<F>(k, m, symbol_len);
            let decoder = afft::BatchDecoder::<F>::new(k, m, symbol_len).unwrap();
            for &erasures in ERASURES.iter().filter(|&&count| count <= m.min(k)) {
                let indices: Vec<usize> = (erasures..k).chain(k..k + erasures).collect();
                let received: Vec<(usize, &[u8])> = indices
                    .iter()
                    .map(|&index| (index, &word[index * symbol_len..(index + 1) * symbol_len]))
                    .collect();
                let mut out = vec![0u8; k * symbol_len];
                let configuration = format!("k{k}_m{m}_r{erasures}_s{symbol_len}");
                for (path, name) in [
                    (RecoveryPath::Targeted, "targeted"),
                    (RecoveryPath::Locator, "locator"),
                ] {
                    let mut plan = decoder.prepare_decode_with_path(&indices, path).unwrap();
                    group.bench_with_input(BenchmarkId::new(name, &configuration), &(), |b, _| {
                        b.iter(|| {
                            plan.decode_into(black_box(&received), black_box(&mut out))
                                .unwrap();
                            black_box(&out);
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

fn benchmark_crossover(c: &mut Criterion) {
    sweep::<fff::Gf8>(c, "gf256", GF8_CONFIGS);
    sweep::<fff::Gf16>(c, "gf65536", GF16_CONFIGS);
}

criterion_group!(benches, benchmark_crossover);
criterion_main!(benches);
