# Changelog

All notable changes to SCRS are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

The arithmetic and transform engines moved upstream. SCRS now depends on
[`fff`](https://github.com/nanithefkuc/fff) for finite-field arithmetic and
vector kernels, and [`cafft`](https://github.com/nanithefkuc/cafft) for the
additive FFT, keeping only the wire format, codec shells, and erasure recipes.
Net deletion is roughly 4600 lines of hand-written SIMD and transform code.

### Added

- **GF(256) additive-FFT engine** (`Engine::Gf8Afft`), a third GF(256) engine
  alongside Standard and Good Cauchy. Unlike the GF(65536) engines it accepts
  **any** `symbol_len`, including odd. It is **not** auto-recommended: measured
  at `symbol_len = 1400` it is the better encoder from `k >= 16` (up to 2.8x at
  `k = 160`) but a worse decoder at every erasure count except near-total
  redundancy consumption, and `Profile::recommended` cannot see the erasure
  count. Select it explicitly for encode-bound or high-loss workloads; see
  `recommended_gf8_engine` for the measurement table.
- `afft` is generic over a sealed `afft::Field` trait, implemented for
  `fff::Gf8` and `fff::Gf16`.
- **`internals` feature** exposing implementation APIs for benchmarking and
  research — scratch inspection, both additive-FFT finalize paths, coding-matrix
  evaluation points, recipe internals, and the kernel table banks. Enables
  `fff/internals` and `cafft/internals`. Exempt from compatibility guarantees.
  Supersedes the former `public-api` branch.
- `tests/systematic.rs` asserts the systematic guarantee — output symbols `0..k`
  are the input verbatim — for every engine across a spread of geometries and
  symbol lengths.
- `benches/engines.rs` compares Cauchy against the additive FFT within each
  field (renamed from `benches/tower_vs_afft.rs`, now covering GF(256) too).
- `internals::backend::{payload_backend, transform_backend}` report the resolved
  SIMD backend for each kernel layer, which can legitimately differ.

### Changed

- **MSRV 1.85 -> 1.89**, forced by both dependencies.
- `Engine::Afft` split into `Engine::Gf8Afft` and `Engine::Gf16Afft`.
- `afft::{Encoder, Decoder}` -> `afft::{Gf16Encoder, Gf16Decoder}`, with
  `Gf8Encoder`/`Gf8Decoder` added. These are aliases of the generic
  `SystematicEncoder<F>`/`LazyDecoderState<F>`.
- `afft::TransformPlan` is now `TransformPlan<F>`, a re-export of cafft's plan.
  Constructors return `Result` rather than `Option`, and byte transforms return
  `Result<(), TransformLengthError>` rather than `()`.
- Repository is a plain library crate; the Cargo workspace is gone, so
  `cargo ... -p scrs` becomes plain `cargo ...`.
- Mainline branch renamed `v2` -> `main`. `v1` and `v2` are retained as
  deprecated, unsupported historical references; `gf8`, `gf16`, and `public-api`
  were removed.
- The single-erasure decode path writes straight into the output row instead of
  staging and copying, worth ~8.6% on the `r = 1` scenarios that dominate real
  loss patterns.

### Removed

- **`scrs::gf256` and `scrs::gf65536`** -> use `scrs::gf8` and `scrs::gf16`
  (re-exports of `fff::gf8`/`fff::gf16`). `GfElem` is now `Elem`, constructed
  with `Elem::from_raw` and unwrapped with `Elem::to_raw`.
- **`scrs::simd`** — the hand-written kernels are now `fff::ops` and
  `fff::kernel`. `IndexedDestinationRows` and its indexed-row scatter family
  have no successor; fff's flat-row kernels replace them.
- **`scrs::tower::payload`** — the butterfly backends are now
  `cafft::core::kernel`.
- **`scrs::matrix`** — the transitional facade from the v2 rename. Use
  `scrs::matrices`.
- **`gf256-tables` feature** — table construction belongs to `fff` and is no
  longer a SCRS build option.
- The `std` feature, which was documented but never existed. SCRS requires
  `std`; there is nothing to toggle.
- A `#[cfg(test)]` timing harness in `batch/codec.rs` that asserted nothing and
  accounted for ~269s of the ~271s test run. The unit suite now finishes in
  ~0.13s, making it usable as a fast gate.

### Performance

Measured against the pre-migration baseline (`61466fa`) on an Intel Core Ultra 7
258V, `taskset -c 0`, `--warm-up-time 2 --measurement-time 8`, judged on
per-target medians. See `.plans/baseline-main-pinned/README.md` for why
individual ns-scale benchmarks are not a valid gate.

## [0.2.0]

Unified engine API: type-erased dispatch from a `Profile`, a shared trait family
across all engines, and the geometry-driven `Profile::recommended` selector.
