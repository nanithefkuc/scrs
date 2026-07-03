//! SCRS decoder latency benchmarks.
//!
//! Measures the three stages of the decode latency budget:
//!
//! - **Per-symbol `push_symbol`**: the incremental coefficient-matrix work.
//! - **`finalize_ref`**: the fused reconstruction pass (matrix solve + payload).
//! - **First-receive-to-ready total**: k pushes + finalize.
//!
//! Benchmarks are run at several `(k, m)` configurations to expose how the
//! decode cost scales with block size. The `criterion` harness prints results
//! to the console; no external visualization tooling is required.

#![allow(clippy::needless_range_loop)]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scrs::batch::BatchCodec;
use scrs::decoder::{LazyDecoderState, RecipeCache};

/// Configurations spanning the GF(256) scope.
///
/// Each entry is `(k, m)`, satisfying `k + m <= 256`. The spread covers
/// small (k=4), medium (k=16, k=64), and large (k=128, k=192) block sizes
/// at 50% and 25% overhead ratios.
const CONFIGS: &[(usize, usize)] = &[
    (4, 4),
    (16, 8),
    (16, 16),
    (64, 32),
    (128, 64),
    (128, 128),
    (192, 64),
];

const SYMBOL_LEN: usize = 1400;

/// Generate deterministic source data and encode it into `n` symbols.
fn make_symbols(k: usize, m: usize, slen: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let codec = BatchCodec::new(k, m, slen).expect("codec construction failed");
    let data: Vec<u8> = (0..k * slen)
        .map(|i| (i as u8).wrapping_mul(0x9E))
        .collect();
    let symbols = codec.encode(&data).expect("encode failed");
    (data, symbols)
}

/// Benchmark the full first-receive-to-ready path: push k repair symbols
/// (worst case: no data symbols arrive), then finalize.
///
/// Uses a global recipe cache to measure the steady-state cost (cache hits
/// after the first rep). The cache is shared across reps of the same config
/// so the closed-form inverse is only computed once.
fn bench_first_receive_to_ready(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_receive_to_ready");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);
        // Worst case: only repair symbols arrive (all data erased).
        let arrival: Vec<usize> = (k..k + m).collect();

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut cache = RecipeCache::new(256);
            b.iter(|| {
                let mut dec = LazyDecoderState::new(k, m, SYMBOL_LEN).unwrap();
                for &idx in &arrival {
                    let _ = dec.push_symbol(idx, black_box(&symbols[idx]));
                }
                let _ = dec.finalize_ref_cached(black_box(&mut cache));
            });
        });
    }

    group.finish();
}

/// Benchmark only `finalize_ref` — the fused reconstruction pass.
///
/// The decoder is pre-filled with k symbols before the timed region, so this
/// isolates the matrix-solve + payload-reconstruction cost from the push cost.
fn bench_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("finalize_ref");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);
        let arrival: Vec<usize> = (k..k + m).collect();

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            let mut cache = RecipeCache::new(256);
            b.iter_with_setup(
                || {
                    let mut dec = LazyDecoderState::new(k, m, SYMBOL_LEN).unwrap();
                    for &idx in &arrival {
                        let _ = dec.push_symbol(idx, &symbols[idx]);
                    }
                    dec
                },
                |mut dec| {
                    let _ = dec.finalize_ref_cached(&mut cache);
                },
            );
        });
    }

    group.finish();
}

/// Benchmark per-symbol `push_symbol` cost at various ranks.
///
/// The cost is expected to be flat (O(k²) on the coefficient matrix only,
/// no payload work) regardless of rank, since v2 defers all payload work to
/// finalize.
fn bench_push_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_symbol");
    group.sample_size(100);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);
        let arrival: Vec<usize> = (k..k + m).collect();

        group.bench_with_input(BenchmarkId::new(format!("k{k}_m{m}"), ""), &(), |b, _| {
            b.iter_with_setup(
                || {
                    let dec = LazyDecoderState::new(k, m, SYMBOL_LEN).unwrap();
                    (dec, 0usize)
                },
                |(mut dec, mut i)| {
                    let idx = arrival[i % arrival.len()];
                    let _ = dec.push_symbol(idx, black_box(&symbols[idx]));
                    i += 1;
                    (dec, i)
                },
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_first_receive_to_ready,
    bench_finalize,
    bench_push_symbol,
);
criterion_main!(benches);
