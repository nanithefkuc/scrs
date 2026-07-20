# SCRS — Streaming Cauchy Reed-Solomon erasure coding (GF(256) testbed)

> **Branch `gf8`.** Single-field testbed for the GF(256) engines. The GF(65536)
> engines live on the `gf16` branch; both are re-unified in `v2`.

SCRS provides systematic Reed-Solomon encoding and decoding over GF(256). A
codeword contains `k` data symbols and `m` repair symbols; any `k` distinct
symbols recover the original data when both sides use the same coding profile.

## Choose an API

- **Batch:** use `BatchCodec` (or one of its matrix-specific aliases) when a
  complete block is available. `encode` returns all `k + m` symbols, and
  `decode` reconstructs from exactly `k` indexed symbols.
- **Streaming encode:** use `StreamingEncoder` when data symbols arrive over
  time. Each data symbol can be sent immediately while repair symbols are
  accumulated in reusable buffers.
- **Streaming decode:** use `LazyDecoderState` when symbols arrive one at a
  time. `push_symbol` validates and stores symbols; `finalize_ref` reconstructs
  only after `k` independent symbols have arrived. Payload arithmetic is
  deferred until finalization.

This repo is a Cargo workspace. The publishable library lives in `scrs/`.
`scrs/examples/` contains compilable programs for each workflow, including
recipe-cache reuse. Build and run them with:

```sh
cargo test -p scrs --examples --all-features
```

## Coding parameters and recovery

`k` and `m` must be non-zero, and every symbol must have the configured
positive `symbol_len`. The code is systematic: indices `0..k` are the original
data symbols and indices `k..k + m` are repairs. The MDS property guarantees
recovery from any `k` distinct, valid symbols. Streaming decoder finalization
can return an owned `Vec<u8>` or write into a caller-provided buffer with
`finalize_into`; the decoder remains usable after non-consuming finalization.

Available coding profiles:

| Profile | Capacity | API |
| --- | ---: | --- |
| GF(256) Standard Cauchy | `k + m <= 256` | `StandardCauchyBatchCodec` |
| GF(256) Good Cauchy | `k + m <= 255` | `GoodCauchyBatchCodec`, `StreamingEncoder` |

GF(256) batch callers select the matrix explicitly and use the matching
`LazyDecoderState<C>` decoder type.

## Features

SCRS requires the Rust standard library. The default feature set is `std`,
`simd`, and `gf256-tables`.

- `std` enables the standard-library implementation and is included by
  default.
- `simd` enables runtime-dispatched SIMD payload kernels; it implies `std`.
  Disable it for portable scalar payload processing.
- `gf256-tables` enables compile-time GF(256) log/exp tables. Without it,
  field operations use the portable shift-and-XOR backend.

## License

SCRS is distributed under the MIT License. See [`LICENSE`](LICENSE).
