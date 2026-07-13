# SCRS — Streaming Cauchy Reed-Solomon Erasure Coding

SCRS is a systematic Cauchy Reed-Solomon erasure code with:

- a **streaming encoder** (`scrs::encoder::StreamingEncoder`) that incrementally computes repair symbols, and
- a **lazy, payload-deferred streaming decoder** (`scrs::decoder::LazyDecoderState`) optimized for predictable receive-path latency.

## Design

The decoder tracks only the **coefficient system** while symbols arrive. Payload bytes are not combined until decode completion (`rank == k`), then a single fused reconstruction pass recovers missing data.

Key properties:

- **Lazy payload combination**: `push_symbol` only records symbol presence/payload; heavy payload math runs once in `finalize_ref`.
- **Closed-form Cauchy inverse**: reduced `r × r` erasure system is solved analytically from Cauchy structure.
- **Reduced solve**: only missing-data rows/columns are inverted (`r = erased data symbols`), not full `k × k`.
- **Recipe cache**: repeated erasure patterns can skip rebuild of reconstruction coefficients.
- **Good Cauchy default**: default codec/decoder usage targets `GoodCauchyView`.

## Scope

Current scope is `k + m <= 255` for Good Cauchy in GF(256).

## Usage

### Batch encode + streaming decode

```rust
use scrs::batch::BatchCodec;
use scrs::decoder::LazyDecoderState;
use scrs::good_cauchy::GoodCauchyView;
use scrs::stream::SymbolSink;

let (k, m, symbol_len) = (16, 8, 1400);

// Encode all symbols (data + repair).
let codec = BatchCodec::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
let data = vec![0xAB; k * symbol_len];
let codeword = codec.encode(&data).unwrap(); // n = k + m symbols

// Decode from any k distinct symbols (example: 8 repairs + 8 data).
let mut decoder = LazyDecoderState::<GoodCauchyView>::new(k, m, symbol_len).unwrap();
for idx in (k..k + m).chain(0..(k - m)) {
    decoder.push_symbol(idx, &codeword[idx]).unwrap();
}

assert_eq!(decoder.finalize_ref().unwrap(), data);
```

### Streaming encoder

```rust
use scrs::encoder::StreamingEncoder;

let (k, m, symbol_len) = (8, 4, 1024);
let mut enc = StreamingEncoder::new(k, m, symbol_len).unwrap();

// Feed data symbols as they become available.
for i in 0..k {
    let payload = vec![i as u8; symbol_len];
    enc.feed_data_symbol(i, &payload).unwrap();
    // Data symbol i can be sent immediately (systematic code).
}

// Repairs are incrementally accumulated and now ready.
for j in 0..m {
    let repair = enc.repair_symbol(j).unwrap();
    let _ = repair;
}
```

## Features

SCRS currently requires the Rust standard library; `no_std` is not a supported
configuration. The default feature set is `std`, `simd`, and `gf256-tables`.

- `simd` enables runtime-dispatched SIMD payload kernels and requires `std`.
  Disable it for portable scalar payload processing.
- `gf256-tables` enables compile-time GF(256) log/exp tables. Disable it to
  use the portable shift-and-XOR field backend.

Supported library configurations include:

```sh
cargo test
cargo test --no-default-features --features std
cargo test --no-default-features --features std,gf256-tables
cargo test --no-default-features --features std,simd
```

## Benchmarks

```sh
cargo bench --bench decoder_latency
cargo bench --bench encoder_latency
cargo bench --bench e2e_latency
```

## Branches

- `master`: `unsafe` denied.
- `simd`: same base logic as `master`, with `unsafe` allowed only for SIMD kernel implementations.

## License

MIT OR Apache-2.0
