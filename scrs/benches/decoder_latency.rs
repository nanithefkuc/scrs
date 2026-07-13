//! SCRS decoder latency benchmarks.
//!
//! Measures the three stages of the decode latency budget:
//!
//! - **Per-symbol `push_symbol`**: receipt tracking and the payload copy.
//! - **Finalize**: cold recipe construction or hot cached reconstruction.
//! - **First-receive-to-ready total**: k pushes plus cold or hot finalize.
//!
//! Benchmarks are run at several `(k, m)` configurations to expose how the
//! decode cost scales with block size. The `criterion` harness prints results
//! to the console; no external visualization tooling is required.

#![allow(clippy::needless_range_loop)]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use scrs::batch::BatchCodec;
use scrs::decoder::{LazyDecoderState, RecipeCache};
use scrs::good_cauchy::GoodCauchyView;

/// Configurations spanning the GF(256) scope.
///
/// Each entry is `(k, m)`, satisfying the GoodCauchy limit `k + m <= 255`.
/// The spread covers small (k=4), medium (k=16, k=64), and large
/// (k=128, k=192) block sizes, including near-capacity configurations.
const CONFIGS: &[(usize, usize)] = &[
    (4, 4),
    (16, 8),
    (16, 16),
    (64, 32),
    (128, 64),
    (128, 127),
    (192, 63),
];

const SYMBOL_LEN: usize = 1400;

/// Generate deterministic source data and encode it into `n` symbols.
fn make_symbols(k: usize, m: usize, slen: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let codec = BatchCodec::<GoodCauchyView>::new(k, m, slen).expect("codec construction failed");
    let data: Vec<u8> = (0..k * slen)
        .map(|i| (i as u8).wrapping_mul(0x9E))
        .collect();
    let symbols = codec.encode(&data).expect("encode failed");
    (data, symbols)
}

/// Select exactly `k` symbols, with `r` repairs and `k - r` data symbols.
fn arrival_with_repairs(k: usize, m: usize, r: usize) -> Vec<usize> {
    assert!(r <= k.min(m));
    (k..k + r).chain(0..k - r).collect()
}

/// Sample the full missing-data range without making the benchmark matrix huge.
fn repair_counts(k: usize, m: usize) -> Vec<usize> {
    let max_r = k.min(m);
    let mut counts = vec![0, 1.min(max_r), max_r / 4, max_r / 2, max_r];
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// Sample ranks from an empty decoder through the final incomplete rank.
fn push_ranks(k: usize) -> Vec<usize> {
    let mut ranks = vec![0, k / 4, k / 2, 3 * k / 4, k - 1];
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

fn prefilled_decoder(
    k: usize,
    m: usize,
    symbols: &[Vec<u8>],
    arrival: &[usize],
) -> LazyDecoderState<GoodCauchyView> {
    let mut dec = LazyDecoderState::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
    for &idx in arrival {
        dec.push_symbol(idx, &symbols[idx]).unwrap();
    }
    dec
}

/// Benchmark the full first-receive-to-ready path for independent values of
/// missing-data count `r`: push exactly `r` repairs and `k - r` data symbols,
/// then finalize. Cold cases build a recipe each time; hot cases use a
/// persistent cache that is populated before measurement starts.
fn bench_first_receive_to_ready(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_receive_to_ready");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);

        for r in repair_counts(k, m) {
            let arrival = arrival_with_repairs(k, m, r);
            let case = format!("k{k}_m{m}_r{r}");

            group.bench_function(BenchmarkId::new("cold", &case), |b| {
                b.iter(|| {
                    let mut dec =
                        LazyDecoderState::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
                    for &idx in &arrival {
                        dec.push_symbol(idx, black_box(&symbols[idx])).unwrap();
                    }
                    black_box(dec.finalize_ref().unwrap());
                });
            });

            let mut cache = RecipeCache::new(1);
            let mut warm = prefilled_decoder(k, m, &symbols, &arrival);
            black_box(warm.finalize_ref_cached(&mut cache).unwrap());
            group.bench_function(BenchmarkId::new("hot", &case), |b| {
                b.iter(|| {
                    let mut dec =
                        LazyDecoderState::<GoodCauchyView>::new(k, m, SYMBOL_LEN).unwrap();
                    for &idx in &arrival {
                        dec.push_symbol(idx, black_box(&symbols[idx])).unwrap();
                    }
                    black_box(dec.finalize_ref_cached(&mut cache).unwrap());
                });
            });
        }
    }

    group.finish();
}

/// Benchmark only finalization for independent missing-data counts `r`.
///
/// Decoder construction and pushes happen outside the timed region. Cold cases
/// call uncached `finalize_ref`; hot cases use a persistent, prewarmed cache.
fn bench_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("finalize");
    group.sample_size(50);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);

        for r in repair_counts(k, m) {
            let arrival = arrival_with_repairs(k, m, r);
            let case = format!("k{k}_m{m}_r{r}");

            group.bench_function(BenchmarkId::new("cold", &case), |b| {
                b.iter_with_setup(
                    || prefilled_decoder(k, m, &symbols, &arrival),
                    |mut dec| black_box(dec.finalize_ref().unwrap()),
                );
            });

            let mut cache = RecipeCache::new(1);
            let mut warm = prefilled_decoder(k, m, &symbols, &arrival);
            black_box(warm.finalize_ref_cached(&mut cache).unwrap());
            group.bench_function(BenchmarkId::new("hot", &case), |b| {
                b.iter_with_setup(
                    || prefilled_decoder(k, m, &symbols, &arrival),
                    |mut dec| black_box(dec.finalize_ref_cached(&mut cache).unwrap()),
                );
            });
        }
    }

    group.finish();
}

/// Benchmark one `push_symbol` at selected decoder ranks.
///
/// Push validates the symbol, updates receipt bookkeeping, and copies one
/// payload. It does no coefficient-matrix or reconstruction work, so its cost
/// should be approximately rank-independent and linear in the symbol length.
fn bench_push_symbol(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_symbol");
    group.sample_size(100);

    for &(k, m) in CONFIGS {
        let (_data, symbols) = make_symbols(k, m, SYMBOL_LEN);
        let arrival = arrival_with_repairs(k, m, k.min(m));

        for rank in push_ranks(k) {
            let next_idx = arrival[rank];
            let case = format!("k{k}_m{m}_rank{rank}");
            group.bench_function(BenchmarkId::new("push", &case), |b| {
                b.iter_with_setup(
                    || prefilled_decoder(k, m, &symbols, &arrival[..rank]),
                    |mut dec| {
                        black_box(
                            dec.push_symbol(next_idx, black_box(&symbols[next_idx]))
                                .unwrap(),
                        )
                    },
                );
            });
        }
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
