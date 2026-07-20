# SCRS — GF(65536) testbed branch (`gf16`)

> [!WARNING]
> **This is a testbed branch — not canonical.** `gf16` exists only to iterate on
> the GF(65536) engines in isolation (it supersedes the former `gf65536`
> branch). Do not depend on it: its API and layout may change or be rebased at
> any time. The GF(256) engines live on the `gf8` testbed branch, and the
> canonical, unified library is the **`v2`** branch. Use `v2`.

SCRS provides systematic Reed-Solomon coding over GF(65536), extending the
codeword capacity far beyond the 255-symbol GF(256) limit. A codeword contains
`k` data symbols and `m` repair symbols; any `k` distinct symbols recover the
original data when both sides use the same coding profile. Symbols use the
stable interleaved two-byte `[a, b]` field representation and must have an even
byte length.

## Two engines

GF(65536) offers two explicit, **non-interchangeable** profiles (a sender and
receiver must use the same one):

| Profile | Capacity | Best for | API |
| --- | ---: | --- | --- |
| Tower Cauchy | `k + m <= 65535` | small blocks, low erasure counts, strict incremental repair | `tower::StreamingEncoder`, `tower::LazyDecoderState` |
| additive FFT | `k.next_power_of_two() + m <= 65536` | large blocks, high redundancy | `afft::SystematicEncoder`, `afft::LazyDecoderState` |

- **Tower** is incremental: `feed_data_symbol` updates every repair as each
  source arrives, and `repair_symbol` borrows a finished repair — data is
  sendable immediately.
- **afft** is block-final: `encode_into_with` computes all repairs from a
  complete data block using a reusable `EncodeScratch` (zero-alloc steady
  state); the lazy decoder reconstructs with a reusable `DecodeScratch`.

Both decoders are payload-lazy: `push_symbol` validates and stores symbols, and
finalization defers all payload arithmetic until `k` independent symbols arrive.

[`recommended_gf16_engine(k, m)`](src/lib.rs) gives a geometry-based default
(afft for `k + m > 256` or `m >= k / 3`, else tower) that both peers can derive
independently.

## Layout

This repo is a Cargo workspace; the publishable library lives in `scrs/`.
`scrs/examples/afft.rs` shows an end-to-end round trip. Build and test with:

```sh
cargo test -p scrs --all-features
```

## Features

The default feature set is `std`, `simd`, and `gf256-tables`.

- `std` enables the standard-library implementation (default).
- `simd` enables runtime-dispatched SIMD payload/butterfly kernels (GFNI on
  x86, NEON on AArch64); it implies `std`. Disable for portable scalar
  processing.
- `gf256-tables` enables compile-time log/exp tables for the GF(256) base field
  underlying the GF(65536) tower construction.

## License

SCRS is distributed under the MIT License. See [`LICENSE`](LICENSE).
