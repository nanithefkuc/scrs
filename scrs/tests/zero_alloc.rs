//! Allocation contracts for reusable v2 facade operations.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scrs::stream::SymbolSink;
use scrs::{
    BatchDecoder, BatchEncoder, Decoder, Engine, IncrementalEncoder, Profile, batch_decoder,
    batch_encoder, decoder, incremental_encoder,
};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
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

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
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
    COUNTING.store(false, Ordering::SeqCst);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0, "{engine:?}");
    assert_eq!(stream_out, data);
    assert_eq!(batch_out, data);
}

#[test]
fn reusable_v2_facades_allocate_nothing() {
    assert_zero_alloc_case(Engine::StandardCauchy, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::GoodCauchy, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Tower, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Afft, 8, 4, 64, 2);
    assert_zero_alloc_case(Engine::Afft, 16, 8, 64, 6);
}
