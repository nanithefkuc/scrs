# SCRS — Streaming Cauchy Reed-Solomon Erasure Coding

SCRS is a systematic Cauchy Reed-Solomon erasure code with a lazy, payload-deferred streaming decoder optimized for predictable receive-path latency.

## Design

The decoder performs incremental tracking on the **coefficient matrix only** payload bytes are not touched until the block reaches full rank. At that point, a single fused reconstruction pass recovers all missing data symbols using a closed-form Cauchy matrix inverse.

Key properties:

- **Lazy payload combination**: `push_symbol` does O(k²) work on the coefficient matrix (a few KB). The heavy O(r × k × S) payload pass happens once in `finalize_ref`, where `r` is the number of erasures and `S` is the symbol length.
- **Closed-form Cauchy inverse**: no pivot search or data-dependent branching in the matrix solve — the inverse is computed analytically from the Cauchy structure, giving deterministic latency.
- **Reduced system**: only the `r × r` erasure submatrix is inverted, not the full `k × k` matrix.
- **Recipe cache**: erasure patterns are memoized. Repeated loss patterns (common on real links) skip the matrix solve entirely.

## Scope

The current implementation supports `k + m ≤ 256` (GF(2^8) index assignment). A future GF(2^16) backend will lift this ceiling to `k + m ≤ 65536`.

## Usage

```rust
use scrs::batch::BatchCodec;
use scrs::stream::SymbolSink;
use scrs::decoder::LazyDecoderState;

let (k, m, symbol_len) = (16, 8, 1400);

// Encode.
let codec = BatchCodec::new(k, m, symbol_len).unwrap();
let data = vec![0xAB; k * symbol_len];
let codeword = codec.encode(&data).unwrap(); // n = k + m symbols

// Decode (worst case: only repair symbols arrive).
let mut decoder = LazyDecoderState::new(k, m, symbol_len).unwrap();
for (i, symbol) in codeword.iter().skip(k).enumerate() {
    decoder.push_symbol(k + i, symbol).unwrap();
}
assert_eq!(decoder.finalize_ref().unwrap(), data);
```

## Benchmarks

```sh
cargo bench -p scrs --bench decoder_latency
```

Measures per-symbol push cost, finalize (reconstruction) cost, and
end-to-end first-receive-to-ready latency across configurations from
`k=4` to `k=192`.

## License

MIT OR Apache-2.0
