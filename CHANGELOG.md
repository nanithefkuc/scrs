# Changelog

All notable changes to SRS are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `BatchCodec::reconstruct_missing_into` /
  `reconstruct_missing_into_with`: reconstruct only the missing data symbols
  into a contiguous `r * symbol_len` destination, leaving surviving shards
  borrowed in place. Receivers that keep survivors referenced (ring buffers,
  mmap regions, in-place recovery adapters) no longer pay a copy for data
  that never moved. The `_with` variant allocates nothing after scratch
  warm-up.
- `BatchCodec::prepare_decode` and `batch::DecodePlan`: a prepared decode
  plan for one erasure pattern, built from the `k` received-symbol indices.
  `DecodePlan::reconstruct_missing_into` and `DecodePlan::decode_into`
  validate the offered symbols against the prepared pattern and then run a
  single fused matrix kernel call — no partitioning, no coefficient
  construction, no heap allocation. This is the honest `decode_prepared`
  primitive: the recurring-pattern cost is paid once at preparation, not
  amortized implicitly.
- `DecodeError::UnexpectedIndex` for symbols outside a prepared plan's
  receipt pattern.
- `afft::Gf8BatchDecoder` / `Gf16BatchDecoder`: native block-final AFFT
  decode from borrowed rows, with reusable `BatchDecodeScratch`.
- `afft::BatchDecoder::prepare_decode` and `afft::DecodePlan`: a prepared AFFT
  decode plan for one erasure pattern. Preparation validates the pattern,
  picks the recovery path, and builds the whole pattern-dependent solve — the
  targeted generator rows and reduced inverse, or the locator path's erasure
  locator — so applying the plan is symbol validation plus payload arithmetic
  with no heap allocation. `prepare_decode_with_path` overrides the crossover
  heuristic for tuning.
- `afft::crossover`: the geometry-driven targeted-versus-locator model, its
  calibration data, and `afft::RecoveryPath`.

### Changed

- Replaced private dense elimination, matrix views, and Cauchy inversion with
  `gfm` matrices, `Ple`, and `Cauchy`; field arithmetic now uses `fgf`.
  Decoder outputs, wire-visible coefficients, engine selection, and
  zero-allocation steady-state behavior are unchanged.
- RS-specific AFFT strip encoding, erasure locators, Forney recovery,
  systematic locator caches, and generator rows now live in SRS. The
  `butterfly-fft` dependency supplies only codec-neutral transforms and
  butterfly kernels; targeted inversion uses a reusable `gfm::Ple`
  decomposition and remains allocation-free after scratch construction.
- `BatchCodec` batch decode now reconstructs through one fused source-major
  matrix kernel pass: the reduced inverse comes from the rational-Lagrange
  closed form (no Gauss-Jordan on the hot path), present-data coefficients
  are composed against the precomputed coding matrix, and the separate
  cancellation pass and `r x r` apply tail are gone. `DecodeScratch` memoizes
  the fused coefficients per receipt pattern, so repeated decodes of one
  erasure pattern pay only validation and payload arithmetic. Output is
  bit-identical; the zero-allocation steady state is unchanged.
- AFFT batch selection no longer uses the streaming decoder's
  `reset + push*k + finalize` blanket path. Native batch decode copies each
  surviving data row directly to output and constructs targeted residuals or
  locator-transform input from borrowed payloads, eliminating the
  domain-sized receipt buffer and both payload staging passes.
- The AFFT targeted/locator crossover is now geometry-driven
  (`afft::crossover::targeted_max_missing`) instead of the fixed
  `TARGETED_MAX_MISSING = 5`. The threshold follows `k`, the padded transform
  size, `symbol_len`, and the field's element width, calibrated against
  `benches/afft_crossover.rs`. Both AFFT decoders dispatch on it, so
  mid-erasure long-symbol patterns stop paying for domain transforms: GF(2^8)
  `k64 m32 s1400 r16` decodes in 21.5 us against 41.3 us before (−48%), and
  `k160 m80 s1400 r16` in 54.4 us against 107 us (−49%).
- `afft::BatchDecodeScratch` memoizes the pattern-dependent solve of the most
  recent decode, so repeated decodes of one erasure pattern skip the generator
  rows, the reduced inversion, and the locator recomputation.
- `internals` feature: `afft::decoder::TARGETED_MAX_MISSING` is replaced by the
  generic `afft::decoder::targeted_max_missing::<F>(k, transform_size,
  symbol_len)`; `afft::DecodeScratch` gains `targeted_max()`.
- `internals` feature: `batch::DecodeScratch` exposes `present()` and
  `inverse()` instead of `b()`/`b_inv()`, matching the fused layout.

## [0.3.0]

**The crate is renamed `scrs` -> `srs`, and "Streaming Cauchy Reed-Solomon"
becomes "Systematic Reed-Solomon".** The old name described a Cauchy-matrix
library; SRS now ships five engines across two fields, two of them additive-FFT
constructions that use no Cauchy matrix at all. Systematic output is the
property every engine shares, so the name states that instead.

To migrate, rename the dependency and the import root:

```toml
# was: scrs = { git = "https://github.com/nanithefkuc/scrs.git" }
srs = { git = "https://github.com/nanithefkuc/srs.git" }
```

```rust
// was: use scrs::{BatchEncoder, Profile};
use srs::{BatchEncoder, Profile};
```

The GitHub repository moved to
[`nanithefkuc/srs`](https://github.com/nanithefkuc/srs); the old URL redirects.
SRS is **not** published to crates.io — `srs` is taken there, and the `fff` on
crates.io is an unrelated abandoned `ff` fork for prime-field zero-knowledge
work rather than this crate's binary-field dependency. The manifest sets
`publish = false` so an accidental `cargo publish` fails immediately rather than
confusingly.

The arithmetic and transform engines also moved upstream. SRS now depends on
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
- Branch `v2` is renamed `main`.
  `v1` and `v2` are retained as deprecated, unsupported historical references;
  `gf8`, `gf16`, and `public-api` were removed.
- The single-erasure decode path writes straight into the output row instead of
  staging and copying, worth ~8.6% on the `r = 1` scenarios that dominate real
  loss patterns.

### Removed

Paths below are shown as `scrs::…` (0.2.0) -> `srs::…` (0.3.0); the crate rename
applies on top of every other change here.

- **`scrs::gf256` and `scrs::gf65536`** -> use `srs::gf8` and `srs::gf16`
  (re-exports of `fff::gf8`/`fff::gf16`). `GfElem` is now `Elem`, constructed
  with `Elem::from_raw` and unwrapped with `Elem::to_raw`.
- **`scrs::simd`** — the hand-written kernels are now `fff::ops` and
  `fff::kernel`. `IndexedDestinationRows` and its indexed-row scatter family
  have no successor; fff's flat-row kernels replace them.
- **`scrs::tower::payload`** — the butterfly backends are now
  `cafft::core::kernel`.
- **`scrs::matrix`** — the transitional facade from the v2 rename. Use
  `srs::matrices`.
- **`gf256-tables` feature** — table construction belongs to `fff` and is no
  longer a build option here.
- The `std` feature, which was documented but never existed. SRS requires
  `std`; there is nothing to toggle.
- A `#[cfg(test)]` timing harness in `batch/codec.rs` that asserted nothing and
  accounted for ~269s of the ~271s test run. The unit suite now finishes in
  ~0.13s, making it usable as a fast gate.

### Performance

Measured against the pre-migration baseline (`61466fa`) on an Intel Core Ultra 7
258V, `taskset -c 0`, `--warm-up-time 2 --measurement-time 8`, judged on
per-target medians. Individual ns-scale benchmarks are not a valid gate on this
host.

| target | benchmarks | baseline | 0.3.0 | shift |
|---|--:|--:|--:|--:|
| `decoder_latency` | 170 | 4898.0 ns | 4383.4 ns | **-5.99%** |
| `engines` (was `tower_vs_afft`) | 66 | 233480.0 ns | 160930.0 ns | **-16.67%** |
| `e2e_latency` | 15 | 11047.0 ns | 10469.0 ns | -3.93% |
| `encoder_latency` | 25 | 2509.6 ns | 2747.5 ns | +4.46% |

The GF(65536) gain splits into a uniform **-16.6%** across both decode families
and **-49%** on Tower encode, where fff's kernels replaced the hand-written
ones. Two known non-regressions ride along: `gf65536_encoder_setup/afft` is
**+91%** because cafft resolves shared plans through a keyed cache rather than
SCRS's old `OnceLock` array — construction-time only, 33 ns to 64 ns — and
`encoder_latency`'s +4.46% is within this host's noise band, unchanged across
three separate measurement sessions.

118 new GF(256) engine benchmarks have no baseline counterpart and are unscored.

## [0.2.0]

Unified engine API: type-erased dispatch from a `Profile`, a shared trait family
across all engines, and the geometry-driven `Profile::recommended` selector.
