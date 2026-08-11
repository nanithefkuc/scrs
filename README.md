> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# SRS — Systematic Reed–Solomon erasure coding

SRS is a systematic Reed–Solomon erasure-coding library for Rust, covering
**GF(256)** and **GF(65536)** behind one unified API. It is built for
streaming transports: data symbols can be put on the wire before any repair is
computed, received symbols are recorded cheaply as they arrive, and payload
reconstruction is deferred until enough symbols are present.

Five engines across two fields — Cauchy-matrix and additive-FFT constructions —
are selected per block geometry behind one trait family. The name is no longer
Cauchy-specific because the library no longer is.

Field arithmetic comes from [`fgf`](https://github.com/nanithefkuc/fgf), GF
linear algebra from [`gfm`](https://github.com/nanithefkuc/gfm), and additive
transforms from
[`butterfly-fft`](https://github.com/nanithefkuc/butterfly-fft). SRS owns the
wire format, codec shells, RS-specific AFFT algorithms, and erasure recipes.

> **Not on crates.io.** `srs` is taken there, and `fgf` on crates.io is an
> unrelated abandoned fork of `ff` for prime-field zero-knowledge work — not the
> binary-field library this crate depends on. Consume SRS as a git dependency:
>
> ```toml
> srs = { git = "https://github.com/nanithefkuc/srs.git" }
> ```

## What it does

A codeword is `n = k + m` symbols: `k` original **data** symbols at indices
`0..k` and `m` **repair** symbols at indices `k..n`. Every symbol is
`symbol_len` bytes. The code is *MDS* (maximum distance separable): the original
data is recoverable from **any `k` distinct symbols** of the codeword, as long as
the sender and receiver use the same profile.

### The systematic guarantee

**For every engine and every geometry, transmitted symbols `0..k` are the input
data verbatim.** Encoding never rewrites them; it only appends `m` repairs.

This is a contract, not an implementation detail. It means a receiver that loses
nothing does no arithmetic at all, and a receiver that loses symbol `i`
reconstructs only symbol `i` — which is what makes `finalize` cost scale with the
erasure count rather than with `k`. Build a transport on it.

[`tests/systematic.rs`](tests/systematic.rs) enforces it: for every engine, at a
spread of geometries and symbol lengths, it asserts that encode leaves output
symbols `0..k` bit-identical to the input and that a decode with zero erasures
returns the input untouched.

## Fields and engines

An **engine** is a concrete construction over a field. A sender and receiver
must use the same engine: the coding matrices are unrelated, so a codeword is
only meaningful to the engine that produced it.

| Field | Engine | Capacity (`k + m ≤`) | Encode model | `symbol_len` | Notes |
| --- | --- | ---: | --- | --- | --- |
| GF(256) | Standard Cauchy | 256 | block-final | any | Cauchy matrix over the AES field |
| GF(256) | Good Cauchy | 255 | incremental or block-final | any | geometric-progression Cauchy; supports streaming encode |
| GF(256) | Additive FFT | 256 | block-final | any | opt-in; faster encode, slower decode — see below |
| GF(65536) | Tower | 65535 | incremental only | even | quadratic tower field `GF((2⁸)²)`; reduced `r × r` reconstruction |
| GF(65536) | Additive FFT | 65536 | block-final | even | `O(n log n)` transform; scales to large blocks and high redundancy |

Every engine decodes both ways (streaming or block-final); the table's column is
the *encode* model. `batch_encoder` rejects Tower and `incremental_encoder`
rejects the block-final engines, both with `ConfigError::UnsupportedMode`.

The **even** `symbol_len` requirement belongs to the *field*, not the transform:
GF(65536) wire elements are two interleaved bytes. The GF(256) additive FFT
accepts any symbol length, including odd.

### Choosing an engine

```rust
use srs::{Field, Profile};

// Explicit engine:
let p = Profile::resolve(srs::Engine::Tower, 32, 4, 1024)?;

// Or a default derived from (field, k, m): both peers compute the same choice.
let p = Profile::recommended(Field::Gf65536, 32, 4, 1024)?;
```

`Profile::recommended` sees only the block geometry — never the actual erasure
count, which is unknown at encode time — so both peers derive the same engine
from `(field, k, m)`.

For **GF(65536)** it returns Tower for small, low-redundancy blocks and the
additive FFT for large or high-redundancy ones, where the transform's fixed
`O(n log n)` cost beats Tower's `O(r · k)` reconstruction.

For **GF(256)** it returns Good Cauchy, or Standard Cauchy when the geometry
needs the 256th codeword position. It **never returns the additive FFT**,
deliberately: measured at `symbol_len = 1400`, the GF(256) AFFT is the better
*encoder* from `k ≥ 16` (up to 2.8× at `k = 160`) but a worse *decoder* at every
erasure count except near-total redundancy consumption — 1.5–2× slower at the
common small-`r` case, winning only as `r` approaches `m`. Since the crossover
lives in `r` and a geometry-only rule cannot reach it, SRS optimises the receive
path by default. Select `Engine::Gf8Afft` explicitly when your workload is
encode-bound or your loss profile genuinely consumes most of the redundancy;
`recommended_gf8_engine`'s docs carry the full measurement table.

## How it works

**Incremental encode** (Good Cauchy, Tower). Each data symbol is fed as it
becomes available and its contribution is folded into every repair immediately,
so a data symbol can be sent the moment it exists and repairs are ready as soon
as the last data symbol arrives.

**Block-final encode** (Good/Standard Cauchy, Additive FFT). All `k` data
symbols are present up front and the `m` repairs are produced in one call.

**Payload-lazy streaming decode** (all engines). `push` records each arriving
symbol and updates a small receipt/rank; no payload arithmetic happens on the
receive path. When `k` independent symbols are present, `finalize_into`
reconstructs the missing data — the reconstruction touches only what is needed
for the symbols that were actually lost.

**Batch decode** (all engines). Submit exactly `k` indexed symbols together and
reconstruct directly into caller-owned output. The batch and streaming decoders
share the same code compatibility but expose separate operation contracts.

**Reusable scratch.** The `*_with` methods take caller-owned scratch and output
buffers. After scratch warm-up, encode, batch decode, and reset/push/finalize
streaming loops perform no heap allocation.

## The trait family

All engines implement a small, field-agnostic surface. Concrete engine types
stay separate (their internals differ), but you program against the traits:

| Trait | Purpose | Key methods |
| --- | --- | --- |
| `Coded` | dimensions | `k` · `m` · `n` · `symbol_len` |
| `IncrementalEncoder` | streaming encode | `feed` · `repair` · `fed_count` · `reset` |
| `BatchEncoder` | block-final encode | `scratch` · `encode_into` · `encode_into_with` |
| `BatchDecoder` | block-final decode | `scratch` · `decode_into` · `decode_into_with` |
| `Decoder` | streaming decode | `push` · `reset` · `rank` · `finalize_into_with` |

## Two ways to use it

**Type-erased** — don't name the engine; dispatch at runtime from a `Profile`:

```rust
use srs::{
    BatchDecoder, BatchEncoder, Decoder, Field, Profile, batch_decoder,
    batch_encoder, decoder,
};

let profile = Profile::recommended(Field::Gf65536, 8, 4, 1024)?;

let enc = batch_encoder(&profile)?;
let mut escratch = enc.scratch();
let mut repairs = vec![0u8; profile.m() * profile.symbol_len()];
enc.encode_into_with(&data, &mut repairs, &mut escratch)?;

let mut batch_dec = batch_decoder(&profile)?;
let mut batch_scratch = batch_dec.scratch();
let mut batch_out = vec![0u8; profile.k() * profile.symbol_len()];
batch_dec.decode_into_with(&received, &mut batch_out, &mut batch_scratch)?;

let mut dec = decoder(&profile)?;
let mut dscratch = Decoder::scratch(&dec);
for &(idx, symbol) in &received {        // any k of the n symbols
    dec.push(idx, symbol)?;
}
let mut out = vec![0u8; profile.k() * profile.symbol_len()];
dec.finalize_into_with(&mut out, &mut dscratch)?;
```

The selector functions return boxing-free `Any*` enums implementing the traits
above. `batch_decoder` and `decoder` let callers explicitly choose block-final
or incremental receive processing for the same profile.

**Concrete** — name the engine type directly for monomorphized calls, using the
same trait methods:

- `batch::BatchCodec<C>` (`GoodCauchyBatchCodec`, `StandardCauchyBatchCodec`)
- `encoder::StreamingEncoder` (GF(256) Good Cauchy)
- `decoder::LazyDecoderState<C>` (GF(256) streaming decode)
- `tower::{StreamingEncoder, LazyDecoderState}` (GF(65536) tower)
- `afft::{Gf8Encoder, Gf8Decoder}` (GF(256) additive FFT)
- `afft::{Gf16Encoder, Gf16Decoder}` (GF(65536) additive FFT)

The `afft` types are aliases of `afft::SystematicEncoder<F>` and
`afft::LazyDecoderState<F>`, generic over a sealed `afft::Field` trait
implemented for `fgf::Gf8` and `fgf::Gf16`.

## Errors

Three crate-level enums, all returned as `Result`:

- `ConfigError` — construction (zero/oversized dimensions, bad `symbol_len`,
  unsupported engine mode). Every constructor returns `Result<Self, ConfigError>`.
- `EncodeError` — encode-time input faults (wrong lengths, duplicate/out-of-range
  index).
- `DecodeError` — decode faults (wrong lengths, too many symbols, insufficient
  rank).

## Features

Default: `simd`.

- `simd` — runtime-dispatched vector kernels in both dependencies (AVX-512/GFNI/
  AVX2/SSSE3 on x86, NEON on AArch64). Disabling leaves their portable scalar
  backends; correctness and output are unchanged either way.
- `internals` — exposes implementation APIs for benchmarking and research:
  scratch inspection, both additive-FFT finalize paths, coding-matrix evaluation
  points, and the kernel table banks. Enables `fgf/internals` and
  `butterfly-fft/internals` too. **Exempt from compatibility guarantees.**

SRS requires `std` (for runtime CPU detection and the shared plan caches), so
there is no `std` feature to toggle.

### Backend overrides

`SIMD_BACKEND` selects the payload-arithmetic backend, downgrade-only — it
cannot select a backend the host does not support — and is read once at first
use:

```sh
SIMD_BACKEND=scalar cargo test        # force the portable path
SIMD_BACKEND=scalar cargo bench      # cap the transform kernels
```

The two layers can legitimately differ: butterfly-fft caps its butterflies at
`Gfni` even when fgf resolves to `Avx512`. With `internals`,
`internals::backend::{payload_backend, transform_backend}` reports what each
layer actually chose.

## Layout

A plain library crate. Runnable programs live in `examples/`, criterion
benchmarks in `benches/`.

```sh
cargo test --all-features
cargo run  --example afft
cargo bench --bench engines      # Cauchy vs additive FFT, per field
```

`benches/engines.rs` is the engine comparison the `Gf8Afft` recommendation above
is derived from; `decoder_latency`, `encoder_latency`, and `e2e_latency` cover
the receive-path figures.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).

## Minimum supported Rust version

1.89, edition 2024. An MSRV bump is a minor-version change.