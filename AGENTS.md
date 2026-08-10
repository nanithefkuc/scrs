# AGENTS.md — SRS

Contributor guide. For what the library *is* and how to use it, read the
rustdoc; this file covers the things a change can silently break.

## What this crate is

Systematic Reed–Solomon erasure coding for Rust, covering GF(256) and
GF(65536) behind one unified API. Built for streaming transports: data symbols
go on the wire before any repair is computed, received symbols are recorded
cheaply as they arrive, and payload reconstruction is deferred until enough
symbols are present.

Field arithmetic and vector kernels come from
[`fgf`](https://github.com/nanithefkuc/fgf); the additive-FFT engine comes from
[`cafft`](https://github.com/nanithefkuc/cafft); GF linear algebra comes from
[`gfm`](https://github.com/nanithefkuc/gfm). SRS owns the wire format, the
codec shells, and the erasure recipes.

## Invariants (do not break)

- **Systematic guarantee.** For every engine and every geometry, transmitted
  symbols `0..k` are the input data verbatim. A receiver that loses nothing does
  no arithmetic, and a receiver that loses symbol `i` reconstructs only symbol
  `i`. This is a contract — `tests/systematic.rs` asserts it across all
  engines.
- **Engine agreement.** A sender and receiver MUST use the same engine: their
  coding matrices are unrelated, so a codeword is only meaningful to the engine
  that produced it. `Profile::recommended` derives a default both peers reach
  independently from `(field, k, m)`.
- **Zero-alloc steady state.** After scratch warm-up, encode, batch decode, and
  reset/push/finalize streaming loops perform no heap allocation. This is
  tested, not assumed.
- **Wire and field conventions are frozen.** The field reduction polynomial,
  generator, tower constant, and wire encoding are pinned by
  `tests/field_basis.rs`. Changing them breaks interop with old peers.

## Dependencies

- `fgf` — GF(256)/GF(65536) field arithmetic and runtime-dispatched SIMD
  kernels. Pinned by git revision. The `SIMD_BACKEND` env var (owned by
  `simdispatch`) selects the runtime backend, downgrade-only.
- `cafft` — additive FFT for the AFFT engines.
- `gfm` — GF linear algebra: matrices, PLE, Cauchy inversion.

**Do not write `unsafe` SIMD in this crate.** All intrinsics live upstream in
`fgf` and `cafft`. The crate root carries `#![forbid(unsafe_code)]`.

## Public surface

The compatibility promise covers the crate root re-exports and the public
modules (`batch`, `decoder`, `encoder`, `afft`, `tower`, `transport`,
`matrices`, `codec`, `error`, `selector`). The `internals` feature exposes
implementation APIs for benchmarking and research — scratch inspection,
finalize paths, coding-matrix evaluation points, and kernel table banks.
Nothing behind it is a compatibility promise.

## Code conventions

- Rust 1.89 or newer, edition 2024. Requires `std`.
- No `unsafe` code (`#![forbid(unsafe_code)]` at the crate root).
- Field arithmetic goes through `fgf`; never hand-roll a field loop.
- GF linear algebra goes through `gfm`; do not reintroduce a private
  elimination.
- Do not put development history in doc comments: no milestone tags, no
  references to superseded designs, no phase numbering.

## Build & test

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Any performance change MUST be measured through the criterion harness with
`--save-baseline` / `--baseline`. Do not land a performance change on the
strength of reasoning alone.