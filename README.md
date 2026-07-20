# SCRS — Streaming Cauchy Reed-Solomon erasure coding (v2)

SCRS provides systematic Reed-Solomon coding over **GF(256)** and **GF(65536)**,
optimized for first-packet-to-data-ready latency: data symbols are sendable
immediately, repair updates are incremental, the receive path is payload-lazy,
and reconstruction cost is proportional to the number of missing symbols.

v2 unifies both fields behind one trait family. The concrete engines stay
separate (their internals differ — reduced Cauchy recipe vs additive-FFT
transform), but they share one surface.

## Concepts

| Item | Meaning |
| --- | --- |
| [`Field`] | `Gf256` or `Gf65536`. |
| [`Engine`] | Concrete coding engine: `StandardCauchy`, `GoodCauchy` (GF256); `Tower`, `Afft` (GF65536). A sender and receiver MUST agree on the engine. |
| [`Profile`] | A validated `(engine, k, m, symbol_len)`. Build with `Profile::resolve` (explicit engine) or `Profile::recommended` (geometry default). |

## Trait family

| Trait | Role | Engines |
| --- | --- | --- |
| `Coded` | `k` / `m` / `n` / `symbol_len` accessors | all |
| `IncrementalEncoder` | data sendable immediately, per-source repair updates (`feed` / `repair` / `fed_count` / `reset`) | GoodCauchy, Tower |
| `BatchEncoder` | block-final encode with a reusable `Scratch` (`encode_into` / `encode_into_with`) | StandardCauchy, GoodCauchy, Afft |
| `Decoder` | lazy payload-deferred streaming decode with a reusable `Scratch` (`push` / `finalize_into` / `finalize_into_with`) | all |

Every finalize path collapses to two methods: `finalize_into(out)` (uses a
throwaway scratch) and `finalize_into_with(out, &mut scratch)` (zero-alloc steady
state). The owned-`Vec` convenience is the inherent `finalize_ref`.

## Two ways to use it

**Type-erased (don't name the engine):** `Profile::recommended` + the
`decoder` / `batch_encoder` / `incremental_encoder` free functions return
`AnyDecoder` / `AnyBatchEncoder` / `AnyIncrementalEncoder`, which implement the
traits by runtime dispatch.

```rust
use scrs::{Field, Profile, BatchEncoder, Decoder, batch_encoder, decoder};

let profile = Profile::recommended(Field::Gf65536, 8, 4, 1024)?; // picks the engine
let enc = batch_encoder(&profile)?;
let mut escr = enc.scratch();
let mut repairs = vec![0u8; profile.m() * profile.symbol_len()];
enc.encode_into_with(&data, &mut repairs, &mut escr)?;

let mut dec = decoder(&profile)?;
let mut dscr = dec.scratch();
// push any k of the n symbols ...
let mut out = vec![0u8; profile.k() * profile.symbol_len()];
dec.finalize_into_with(&mut out, &mut dscr)?;
```

**Concrete (zero-cost, monomorphized):** name the engine type directly —
`batch::BatchCodec<C>`, `encoder::StreamingEncoder`, `decoder::LazyDecoderState<C>`,
`tower::{StreamingEncoder, LazyDecoderState}`, `afft::{SystematicEncoder,
LazyDecoderState}` — and call the same trait methods.

## Errors

Three crate-level enums: [`ConfigError`] (construction), [`EncodeError`],
[`DecodeError`]. Constructors return `Result<Self, ConfigError>`.

## Capacities

| Engine | Capacity |
| --- | ---: |
| Standard Cauchy | `k + m <= 256` |
| Good Cauchy | `k + m <= 255` |
| Tower | `k + m <= 65535` |
| additive FFT | `k + m <= 65536` |

GF(65536) engines require an even `symbol_len` (interleaved two-byte elements).

## Features

Default: `std`, `simd`, `gf256-tables`.

- `std` — standard library (default).
- `simd` — runtime-dispatched SIMD kernels (GFNI on x86, NEON on AArch64);
  implies `std`.
- `gf256-tables` — compile-time GF(256) log/exp tables (also the GF(65536)
  base field).

Build and test:

```sh
cargo test -p scrs --all-features
```

## License

MIT. See [`LICENSE`](LICENSE).
