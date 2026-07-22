# SCRS — Systematic Cauchy Reed–Solomon erasure coding

SCRS is a systematic Reed–Solomon erasure-coding library for Rust, covering
**GF(256)** and **GF(65536)** behind one unified API. It is built for
streaming transports: data symbols can be put on the wire before any repair is
computed, received symbols are recorded cheaply as they arrive, and payload
reconstruction is deferred until enough symbols are present.

## What it does

A codeword is `n = k + m` symbols: `k` original **data** symbols at indices
`0..k` and `m` **repair** symbols at indices `k..n`. Every symbol is
`symbol_len` bytes. The code is *systematic* (data symbols travel unchanged) and
*MDS* (maximum distance separable): the original data is recoverable from **any
`k` distinct symbols** of the codeword, as long as the sender and receiver use
the same profile.

## Fields and engines

An **engine** is a concrete construction over a field. A sender and receiver
must use the same engine.

| Field | Engine | Capacity (`k + m ≤`) | Encode model | Notes |
| --- | --- | ---: | --- | --- |
| GF(256) | Standard Cauchy | 256 | block-final | Cauchy matrix over the AES field |
| GF(256) | Good Cauchy | 255 | incremental or block-final | geometric-progression Cauchy; supports streaming encode |
| GF(65536) | Tower | 65535 | incremental | quadratic tower field `GF((2⁸)²)`; reduced `r × r` reconstruction |
| GF(65536) | Additive FFT | 65536 | block-final | additive-FFT transform; scales to large blocks and high redundancy |

GF(65536) engines use interleaved two-byte field elements and therefore require
an **even** `symbol_len`. The two GF(65536) engines have incompatible parity and
are not interchangeable.

Pick an engine explicitly, or let SCRS choose a geometry-based default:

```rust
use scrs::{Field, Profile};

// Explicit engine:
let p = Profile::resolve(scrs::Engine::Tower, 32, 4, 1024)?;

// Or a default derived from (field, k, m): both peers compute the same choice.
let p = Profile::recommended(Field::Gf65536, 32, 4, 1024)?;
```

## How it works

**Systematic, MDS.** Encoding leaves the `k` data symbols untouched and produces
`m` repairs. Decoding recovers the data from any `k` received symbols.

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
use scrs::{
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
- `afft::{SystematicEncoder, LazyDecoderState}` (GF(65536) additive FFT)

## Errors

Three crate-level enums, all returned as `Result`:

- `ConfigError` — construction (zero/oversized dimensions, bad `symbol_len`,
  unsupported engine mode). Every constructor returns `Result<Self, ConfigError>`.
- `EncodeError` — encode-time input faults (wrong lengths, duplicate/out-of-range
  index).
- `DecodeError` — decode faults (wrong lengths, too many symbols, insufficient
  rank).

## Features

Default: `std`, `simd`, `gf256-tables`.

- `std` — standard library (default; SCRS currently requires it).
- `simd` — runtime-dispatched SIMD kernels (GFNI on x86, NEON on AArch64);
  implies `std`. Disable for portable scalar processing.
- `gf256-tables` — compile-time GF(256) log/exp tables (also the base field for
  the GF(65536) tower construction).

## Layout

This repository is a Cargo workspace; the publishable library is in `scrs/`,
with runnable programs in `scrs/examples/`.

```sh
cargo test -p scrs --all-features
cargo run  -p scrs --example afft
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
