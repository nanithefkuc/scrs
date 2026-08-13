//! Allocation contracts for reusable v2 facade operations.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use srs::stream::SymbolSink;
use srs::{
    BatchDecoder, BatchEncoder, Decoder, Engine, IncrementalEncoder, Profile, batch_decoder,
    batch_encoder, decoder, incremental_encoder,
};

struct CountingAllocator;

thread_local! {
    // Counting is thread-local: libtest runs each `#[test]` on its own worker
    // thread, so a process-global counter would also tally the harness thread's
    // allocations and flake. Const initializers keep these off the lazy-init
    // path, so touching them inside the allocator hook never re-enters `alloc`.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_alloc() {
    if COUNTING.with(Cell::get) {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
    }
}

/// Begins counting allocations on the current thread from a clean slate.
fn start_counting() {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
}

/// Stops counting on the current thread and returns the tally.
fn stop_counting() -> usize {
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(Cell::get)
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_alloc();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_alloc();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_alloc();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn assert_zero_alloc_case(engine: Engine, k: usize, m: usize, symbol_len: usize, missing: usize) {
    let profile = Profile::resolve(engine, k, m, symbol_len).unwrap();
    let data: Vec<u8> = (0..k * symbol_len)
        .map(|index| (index.wrapping_mul(131) + 7) as u8)
        .collect();
    let mut repairs = vec![0u8; m * symbol_len];

    let mut block_encoder = batch_encoder(&profile).ok();
    let mut encode_scratch = block_encoder.as_ref().map(BatchEncoder::scratch);
    let mut stream_encoder = incremental_encoder(&profile).ok();
    if let (Some(encoder), Some(scratch)) = (&mut block_encoder, &mut encode_scratch) {
        encoder
            .encode_into_with(&data, &mut repairs, scratch)
            .unwrap();
    } else if let Some(encoder) = &mut stream_encoder {
        for index in 0..k {
            encoder
                .feed(index, &data[index * symbol_len..(index + 1) * symbol_len])
                .unwrap();
        }
        for index in 0..m {
            repairs[index * symbol_len..(index + 1) * symbol_len]
                .copy_from_slice(encoder.repair(index).unwrap());
        }
        encoder.reset();
    } else {
        panic!("engine exposes no encode mode");
    }

    let mut word: Vec<Vec<u8>> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
    word.extend(repairs.chunks_exact(symbol_len).map(<[u8]>::to_vec));
    let indices: Vec<usize> = (missing..k).chain(k..k + missing).collect();
    let received: Vec<(usize, &[u8])> = indices
        .iter()
        .map(|&index| (index, word[index].as_slice()))
        .collect();

    let mut streaming = decoder(&profile).unwrap();
    let mut stream_scratch = Decoder::scratch(&streaming);
    let mut stream_out = vec![0u8; k * symbol_len];
    for &(index, payload) in &received {
        streaming.push(index, payload).unwrap();
    }
    streaming
        .finalize_into_with(&mut stream_out, &mut stream_scratch)
        .unwrap();
    streaming.reset();

    let mut batch = batch_decoder(&profile).unwrap();
    let mut batch_scratch = BatchDecoder::scratch(&batch);
    let mut batch_out = vec![0u8; k * symbol_len];
    batch
        .decode_into_with(&received, &mut batch_out, &mut batch_scratch)
        .unwrap();

    start_counting();
    if let (Some(encoder), Some(scratch)) = (&mut block_encoder, &mut encode_scratch) {
        encoder
            .encode_into_with(
                std::hint::black_box(&data),
                std::hint::black_box(&mut repairs),
                std::hint::black_box(scratch),
            )
            .unwrap();
    }
    if let Some(encoder) = &mut stream_encoder {
        encoder.reset();
        for index in 0..k {
            encoder
                .feed(
                    index,
                    std::hint::black_box(&data[index * symbol_len..(index + 1) * symbol_len]),
                )
                .unwrap();
        }
    }
    for &(index, payload) in std::hint::black_box(&received) {
        streaming.push(index, payload).unwrap();
    }
    streaming
        .finalize_into_with(
            std::hint::black_box(&mut stream_out),
            std::hint::black_box(&mut stream_scratch),
        )
        .unwrap();
    batch
        .decode_into_with(
            std::hint::black_box(&received),
            std::hint::black_box(&mut batch_out),
            std::hint::black_box(&mut batch_scratch),
        )
        .unwrap();
    let allocations = stop_counting();

    assert_eq!(allocations, 0, "{engine:?}");
    assert_eq!(stream_out, data);
    assert_eq!(batch_out, data);
}

/// The Cauchy-only reconstruct-only batch path keeps the same zero-allocation
/// steady state as full decode. Not a separate `#[test]`: the counting
/// allocator is process-global, so every case must run on this one test's
/// thread to avoid counting a concurrent test's setup.
fn assert_reconstruct_missing_zero_alloc() {
    use srs::batch::{GoodCauchyBatchCodec, StandardCauchyBatchCodec};

    fn check<C: srs::coding_matrix::CodingMatrix>(
        codec: &srs::batch::BatchCodec<C>,
        k: usize,
        m: usize,
        symbol_len: usize,
        missing: usize,
    ) {
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|index| (index.wrapping_mul(131) + 7) as u8)
            .collect();
        let mut repairs = vec![0u8; m * symbol_len];
        codec.encode_into(&data, &mut repairs).unwrap();
        let mut word: Vec<Vec<u8>> = data.chunks_exact(symbol_len).map(<[u8]>::to_vec).collect();
        word.extend(repairs.chunks_exact(symbol_len).map(<[u8]>::to_vec));
        let indices: Vec<usize> = (missing..k).chain(k..k + missing).collect();
        let received: Vec<(usize, &[u8])> = indices
            .iter()
            .map(|&index| (index, word[index].as_slice()))
            .collect();
        let mut scratch = codec.decode_scratch();
        let mut missing_out = vec![0u8; missing * symbol_len];
        codec
            .reconstruct_missing_into_with(&received, &mut missing_out, &mut scratch)
            .unwrap();
        let mut plan = codec.prepare_decode(&indices).unwrap();
        let mut plan_out = vec![0u8; k * symbol_len];
        plan.decode_into(&received, &mut plan_out).unwrap();

        start_counting();
        codec
            .reconstruct_missing_into_with(
                std::hint::black_box(&received),
                std::hint::black_box(&mut missing_out),
                std::hint::black_box(&mut scratch),
            )
            .unwrap();
        plan.reconstruct_missing_into(
            std::hint::black_box(&received),
            std::hint::black_box(&mut missing_out),
        )
        .unwrap();
        plan.decode_into(
            std::hint::black_box(&received),
            std::hint::black_box(&mut plan_out),
        )
        .unwrap();
        let allocations = stop_counting();

        assert_eq!(allocations, 0);
        // The absent data symbols are exactly `0..missing`; reconstructed rows
        // arrive in ascending data-index order.
        for (row, expected) in data.chunks_exact(symbol_len).take(missing).enumerate() {
            assert_eq!(
                &missing_out[row * symbol_len..(row + 1) * symbol_len],
                expected
            );
        }
        assert_eq!(plan_out, data);
    }

    check(
        &StandardCauchyBatchCodec::new(8, 4, 64).unwrap(),
        8,
        4,
        64,
        2,
    );
    check(
        &StandardCauchyBatchCodec::new(8, 4, 64).unwrap(),
        8,
        4,
        64,
        1,
    );
    check(&GoodCauchyBatchCodec::new(8, 4, 64).unwrap(), 8, 4, 64, 2);
    check(&GoodCauchyBatchCodec::new(16, 8, 63).unwrap(), 16, 8, 63, 5);
}

/// Prepared AFFT plans must apply without allocating, on both recovery paths.
/// Shares this file's process-global counting allocator, so it runs inside the
/// single `#[test]` below rather than as its own.
fn assert_afft_plan_zero_alloc() {
    use srs::afft::{self, RecoveryPath};

    fn check<F: afft::Field>(k: usize, m: usize, symbol_len: usize, missing: usize) {
        let mut word = vec![0u8; (k + m) * symbol_len];
        let (data, repairs) = word.split_at_mut(k * symbol_len);
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(131) + 7) as u8;
        }
        let encoder = afft::SystematicEncoder::<F>::new(k, m, symbol_len).unwrap();
        encoder.encode_into(data, repairs).unwrap();
        let expected = word[..k * symbol_len].to_vec();

        let decoder = afft::BatchDecoder::<F>::new(k, m, symbol_len).unwrap();
        let indices: Vec<usize> = (missing..k).chain(k..k + missing).collect();
        let received: Vec<(usize, &[u8])> = indices
            .iter()
            .map(|&index| (index, &word[index * symbol_len..(index + 1) * symbol_len]))
            .collect();
        let mut out = vec![0u8; k * symbol_len];

        for path in [RecoveryPath::Targeted, RecoveryPath::Locator] {
            let mut plan = decoder.prepare_decode_with_path(&indices, path).unwrap();
            plan.decode_into(&received, &mut out).unwrap();

            start_counting();
            plan.decode_into(
                std::hint::black_box(&received),
                std::hint::black_box(&mut out),
            )
            .unwrap();
            let allocations = stop_counting();

            assert_eq!(allocations, 0, "{path:?}");
            assert_eq!(out, expected, "{path:?}");
        }
    }

    check::<fgf::Gf8>(16, 8, 63, 2);
    check::<fgf::Gf8>(16, 8, 63, 6);
    check::<fgf::Gf16>(16, 8, 64, 6);
}

#[test]
fn reusable_v2_facades_allocate_nothing() {
    assert_zero_alloc_case(Engine::StandardCauchy, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::GoodCauchy, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Tower, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Gf16Afft, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Gf16Afft, 16, 8, 64, 6);
    // Both AFFT finalize paths: the targeted dense solve and the locator path.
    assert_zero_alloc_case(Engine::Gf8Afft, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Gf8Afft, 16, 8, 64, 6);
    // GF(2^8) has no symbol-length parity rule; exercise an odd one.
    assert_zero_alloc_case(Engine::Gf8Afft, 8, 4, 63, 2);
    assert_reconstruct_missing_zero_alloc();
    assert_afft_plan_zero_alloc();
}
