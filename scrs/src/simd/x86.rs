//! x86/x86_64 SIMD kernels.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unsafe_code)]
#![allow(clippy::incompatible_msrv)] // GFNI intrinsics require a newer compiler than the crate MSRV.

use super::{
    scalar::{xor_bytes_scalar, xor_scaled_bytes_nibble_tail},
    scale_table::{ScaleTable, scale_table},
};
use crate::gf256::GfElem;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_bytes_avx2(dst: &mut [u8], src: &[u8]) {
    let mut offset = 0;
    let len = dst.len();
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, and both pointers are derived from
        // valid slices. Unaligned loads/stores are used intentionally.
        let d = unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()) };
        let s = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
        let mixed = _mm256_xor_si256(d, s);
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(offset).cast::<__m256i>(), mixed) };
        offset += 32;
    }

    xor_bytes_sse2_tail(&mut dst[offset..], &src[offset..]);
}

#[target_feature(enable = "sse2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_bytes_sse2(dst: &mut [u8], src: &[u8]) {
    xor_bytes_sse2_tail(dst, src);
}

#[target_feature(enable = "sse2")]
fn xor_bytes_sse2_tail(dst: &mut [u8], src: &[u8]) {
    let mut offset = 0;
    let len = dst.len();
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len`, and both pointers are derived from
        // valid slices. Unaligned loads/stores are used intentionally.
        let d = unsafe { _mm_loadu_si128(dst.as_ptr().add(offset).cast::<__m128i>()) };
        let s = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast::<__m128i>()) };
        let mixed = _mm_xor_si128(d, s);
        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast::<__m128i>(), mixed) };
        offset += 16;
    }

    xor_bytes_scalar(&mut dst[offset..], &src[offset..]);
}

#[target_feature(enable = "avx2,gfni")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_gfni(dst: &mut [u8], coeff: GfElem, src: &[u8]) {
    let coefficient = _mm256_set1_epi8(coeff.0 as i8);
    let len = dst.len();
    let mut offset = 0;
    // 4x-unrolled main loop: four independent GF2P8MULB chains per iteration
    // keep the multiplier pipeline full on latency-bound single-destination AXPY.
    while offset + 128 <= len {
        let (x0, x1, x2, x3, d0, d1, d2, d3);
        // SAFETY: `offset + 128 <= len` bounds all eight unaligned vector loads.
        unsafe {
            let sp = src.as_ptr().add(offset);
            let dp = dst.as_ptr().add(offset);
            x0 = _mm256_loadu_si256(sp.cast::<__m256i>());
            x1 = _mm256_loadu_si256(sp.add(32).cast::<__m256i>());
            x2 = _mm256_loadu_si256(sp.add(64).cast::<__m256i>());
            x3 = _mm256_loadu_si256(sp.add(96).cast::<__m256i>());
            d0 = _mm256_loadu_si256(dp.cast::<__m256i>());
            d1 = _mm256_loadu_si256(dp.add(32).cast::<__m256i>());
            d2 = _mm256_loadu_si256(dp.add(64).cast::<__m256i>());
            d3 = _mm256_loadu_si256(dp.add(96).cast::<__m256i>());
        }
        let r0 = _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x0, coefficient));
        let r1 = _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x1, coefficient));
        let r2 = _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x2, coefficient));
        let r3 = _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x3, coefficient));
        // SAFETY: `offset + 128 <= len` bounds all four unaligned vector stores.
        unsafe {
            let dp = dst.as_mut_ptr().add(offset);
            _mm256_storeu_si256(dp.cast::<__m256i>(), r0);
            _mm256_storeu_si256(dp.add(32).cast::<__m256i>(), r1);
            _mm256_storeu_si256(dp.add(64).cast::<__m256i>(), r2);
            _mm256_storeu_si256(dp.add(96).cast::<__m256i>(), r3);
        }
        offset += 128;
    }
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len` bounds each unaligned vector access.
        let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
        let d = unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()) };
        let scaled = _mm256_gf2p8mul_epi8(x, coefficient);
        unsafe {
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(offset).cast::<__m256i>(),
                _mm256_xor_si256(d, scaled),
            );
        }
        offset += 32;
    }

    let remaining = dst.len() - offset;
    if remaining != 0 && remaining % 4 == 0 {
        let mut mask_words = [0i32; 8];
        mask_words[..remaining / 4].fill(-1);
        let mask = unsafe { _mm256_loadu_si256(mask_words.as_ptr().cast::<__m256i>()) };
        let x = unsafe { _mm256_maskload_epi32(src.as_ptr().add(offset).cast::<i32>(), mask) };
        let d = unsafe { _mm256_maskload_epi32(dst.as_ptr().add(offset).cast::<i32>(), mask) };
        let scaled = _mm256_gf2p8mul_epi8(x, coefficient);
        unsafe {
            _mm256_maskstore_epi32(
                dst.as_mut_ptr().add(offset).cast::<i32>(),
                mask,
                _mm256_xor_si256(d, scaled),
            );
        }
    } else if remaining != 0 {
        let mut source_tail = [0u8; 32];
        source_tail[..remaining].copy_from_slice(&src[offset..]);
        let x = unsafe { _mm256_loadu_si256(source_tail.as_ptr().cast::<__m256i>()) };
        let scaled = _mm256_gf2p8mul_epi8(x, coefficient);
        let mut scaled_tail = [0u8; 32];
        unsafe {
            _mm256_storeu_si256(scaled_tail.as_mut_ptr().cast::<__m256i>(), scaled);
        }
        for (destination, &product) in dst[offset..].iter_mut().zip(&scaled_tail) {
            *destination ^= product;
        }
    }
}

/// Flat multi-destination streaming encode kernel.
///
/// Complete groups of four repairs use the fully unrolled source-major path
/// (hoisted row pointers + one source load per tile). Remainders use the
/// single-destination GFNI AXPY.
#[target_feature(enable = "avx2,gfni")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_rows_gfni(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    let m = coeffs.len();
    let mut j = 0;
    while j + 4 <= m {
        let indices = [j, j + 1, j + 2, j + 3];
        let coefficients = [coeffs[j], coeffs[j + 1], coeffs[j + 2], coeffs[j + 3]];
        // SAFETY: Caller validated `repairs` geometry; indices are in-range
        // and address disjoint symbol rows.
        unsafe {
            xor_scaled_bytes_4_indexed_gfni(repairs, symbol_len, &indices, &coefficients, src);
        }
        j += 4;
    }
    while j < m {
        let start = j * symbol_len;
        // SAFETY: Single-row slice is in-bounds by geometry check above.
        unsafe {
            xor_scaled_bytes_gfni(&mut repairs[start..start + symbol_len], coeffs[j], src);
        }
        j += 1;
    }
}

/// Flat multi-destination AVX2 nibble kernel with hoisted row pointers.
#[target_feature(enable = "avx2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_rows_avx2(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    let m = coeffs.len();
    let mask = _mm256_set1_epi8(0x0f);
    let dst_ptr = repairs.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= m {
        let rows = [
            unsafe { dst_ptr.add(j * symbol_len) },
            unsafe { dst_ptr.add((j + 1) * symbol_len) },
            unsafe { dst_ptr.add((j + 2) * symbol_len) },
            unsafe { dst_ptr.add((j + 3) * symbol_len) },
        ];
        let mut low_tables = [_mm256_setzero_si256(); 4];
        let mut high_tables = [_mm256_setzero_si256(); 4];
        for slot in 0..4 {
            let scale = scale_table(coeffs[j + slot]);
            let low = unsafe { _mm_loadu_si128(scale.lo.as_ptr().cast::<__m128i>()) };
            let high = unsafe { _mm_loadu_si128(scale.hi.as_ptr().cast::<__m128i>()) };
            low_tables[slot] = _mm256_broadcastsi128_si256(low);
            high_tables[slot] = _mm256_broadcastsi128_si256(high);
        }

        let mut offset = 0;
        while offset + 32 <= symbol_len {
            let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            let low_nibbles = _mm256_and_si256(x, mask);
            let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);

            // Fully unrolled four-destination update: one source load.
            let d0 = unsafe { _mm256_loadu_si256(rows[0].add(offset).cast::<__m256i>()) };
            let d1 = unsafe { _mm256_loadu_si256(rows[1].add(offset).cast::<__m256i>()) };
            let d2 = unsafe { _mm256_loadu_si256(rows[2].add(offset).cast::<__m256i>()) };
            let d3 = unsafe { _mm256_loadu_si256(rows[3].add(offset).cast::<__m256i>()) };
            let p0 = _mm256_xor_si256(
                _mm256_shuffle_epi8(low_tables[0], low_nibbles),
                _mm256_shuffle_epi8(high_tables[0], high_nibbles),
            );
            let p1 = _mm256_xor_si256(
                _mm256_shuffle_epi8(low_tables[1], low_nibbles),
                _mm256_shuffle_epi8(high_tables[1], high_nibbles),
            );
            let p2 = _mm256_xor_si256(
                _mm256_shuffle_epi8(low_tables[2], low_nibbles),
                _mm256_shuffle_epi8(high_tables[2], high_nibbles),
            );
            let p3 = _mm256_xor_si256(
                _mm256_shuffle_epi8(low_tables[3], low_nibbles),
                _mm256_shuffle_epi8(high_tables[3], high_nibbles),
            );
            unsafe {
                _mm256_storeu_si256(
                    rows[0].add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(d0, p0),
                );
                _mm256_storeu_si256(
                    rows[1].add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(d1, p1),
                );
                _mm256_storeu_si256(
                    rows[2].add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(d2, p2),
                );
                _mm256_storeu_si256(
                    rows[3].add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(d3, p3),
                );
            }
            offset += 32;
        }

        for slot in 0..4 {
            let scale = scale_table(coeffs[j + slot]);
            let row = unsafe {
                core::slice::from_raw_parts_mut(rows[slot].add(offset), symbol_len - offset)
            };
            xor_scaled_bytes_ssse3_tail(row, &scale.lo, &scale.hi, &src[offset..]);
        }
        j += 4;
    }

    while j < m {
        let start = j * symbol_len;
        let scale = scale_table(coeffs[j]);
        // SAFETY: Plan is Avx2; single-row slice is in-bounds.
        unsafe {
            xor_scaled_bytes_avx2(
                &mut repairs[start..start + symbol_len],
                &scale.lo,
                &scale.hi,
                src,
            );
        }
        j += 1;
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "avx2,gfni")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_gfni(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    // Hoist destination pointers once per group of four, matching the flat
    // rows kernel. Avoids re-borrowing Vecs on every tile.
    for destination_start in (0..destinations.len()).step_by(4) {
        let output_count = (destinations.len() - destination_start).min(4);
        if output_count == 4 {
            let rows = [
                destinations[destination_start].as_mut_ptr(),
                destinations[destination_start + 1].as_mut_ptr(),
                destinations[destination_start + 2].as_mut_ptr(),
                destinations[destination_start + 3].as_mut_ptr(),
            ];
            // SAFETY: four disjoint Vec buffers; same length as `src`.
            // Temporarily treat them as a synthetic flat layout via raw
            // pointers by inlining the unrolled body.
            let factors = [
                _mm256_set1_epi8(coeffs[destination_start].0 as i8),
                _mm256_set1_epi8(coeffs[destination_start + 1].0 as i8),
                _mm256_set1_epi8(coeffs[destination_start + 2].0 as i8),
                _mm256_set1_epi8(coeffs[destination_start + 3].0 as i8),
            ];
            let mut offset = 0;
            while offset + 32 <= src.len() {
                let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
                let d0 = unsafe { _mm256_loadu_si256(rows[0].add(offset).cast::<__m256i>()) };
                let d1 = unsafe { _mm256_loadu_si256(rows[1].add(offset).cast::<__m256i>()) };
                let d2 = unsafe { _mm256_loadu_si256(rows[2].add(offset).cast::<__m256i>()) };
                let d3 = unsafe { _mm256_loadu_si256(rows[3].add(offset).cast::<__m256i>()) };
                unsafe {
                    _mm256_storeu_si256(
                        rows[0].add(offset).cast::<__m256i>(),
                        _mm256_xor_si256(d0, _mm256_gf2p8mul_epi8(x, factors[0])),
                    );
                    _mm256_storeu_si256(
                        rows[1].add(offset).cast::<__m256i>(),
                        _mm256_xor_si256(d1, _mm256_gf2p8mul_epi8(x, factors[1])),
                    );
                    _mm256_storeu_si256(
                        rows[2].add(offset).cast::<__m256i>(),
                        _mm256_xor_si256(d2, _mm256_gf2p8mul_epi8(x, factors[2])),
                    );
                    _mm256_storeu_si256(
                        rows[3].add(offset).cast::<__m256i>(),
                        _mm256_xor_si256(d3, _mm256_gf2p8mul_epi8(x, factors[3])),
                    );
                }
                offset += 32;
            }
            for slot in 0..4 {
                let coeff = coeffs[destination_start + slot];
                let destination = &mut destinations[destination_start + slot][offset..];
                if coeff == GfElem::ONE {
                    xor_bytes_sse2_tail(destination, &src[offset..]);
                } else if coeff != GfElem::ZERO {
                    // SAFETY: remainder of a GFNI plan buffer.
                    unsafe {
                        xor_scaled_bytes_gfni(destination, coeff, &src[offset..]);
                    }
                }
            }
            continue;
        }

        for slot in 0..output_count {
            let coeff = coeffs[destination_start + slot];
            if coeff == GfElem::ZERO {
                continue;
            }
            // SAFETY: single destination, lengths validated by caller.
            unsafe {
                xor_scaled_bytes_gfni(&mut destinations[destination_start + slot], coeff, src);
            }
        }
    }
}

#[target_feature(enable = "avx2,gfni")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_4_indexed_gfni(
    dst: &mut [u8],
    row_stride: usize,
    destination_indices: &[usize; 4],
    coefficients: &[GfElem; 4],
    src: &[u8],
) {
    let dst_ptr = dst.as_mut_ptr();
    let rows = [
        unsafe { dst_ptr.add(destination_indices[0] * row_stride) },
        unsafe { dst_ptr.add(destination_indices[1] * row_stride) },
        unsafe { dst_ptr.add(destination_indices[2] * row_stride) },
        unsafe { dst_ptr.add(destination_indices[3] * row_stride) },
    ];
    let factors = [
        _mm256_set1_epi8(coefficients[0].0 as i8),
        _mm256_set1_epi8(coefficients[1].0 as i8),
        _mm256_set1_epi8(coefficients[2].0 as i8),
        _mm256_set1_epi8(coefficients[3].0 as i8),
    ];

    let mut offset = 0;
    while offset + 32 <= src.len() {
        let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };

        let d0 = unsafe { _mm256_loadu_si256(rows[0].add(offset).cast::<__m256i>()) };
        let d1 = unsafe { _mm256_loadu_si256(rows[1].add(offset).cast::<__m256i>()) };
        let d2 = unsafe { _mm256_loadu_si256(rows[2].add(offset).cast::<__m256i>()) };
        let d3 = unsafe { _mm256_loadu_si256(rows[3].add(offset).cast::<__m256i>()) };
        let p0 = _mm256_gf2p8mul_epi8(x, factors[0]);
        let p1 = _mm256_gf2p8mul_epi8(x, factors[1]);
        let p2 = _mm256_gf2p8mul_epi8(x, factors[2]);
        let p3 = _mm256_gf2p8mul_epi8(x, factors[3]);
        unsafe {
            _mm256_storeu_si256(
                rows[0].add(offset).cast::<__m256i>(),
                _mm256_xor_si256(d0, p0),
            );
            _mm256_storeu_si256(
                rows[1].add(offset).cast::<__m256i>(),
                _mm256_xor_si256(d1, p1),
            );
            _mm256_storeu_si256(
                rows[2].add(offset).cast::<__m256i>(),
                _mm256_xor_si256(d2, p2),
            );
            _mm256_storeu_si256(
                rows[3].add(offset).cast::<__m256i>(),
                _mm256_xor_si256(d3, p3),
            );
        }
        offset += 32;
    }

    let remaining = src.len() - offset;
    if remaining != 0 && remaining % 4 == 0 {
        let mut mask_words = [0i32; 8];
        mask_words[..remaining / 4].fill(-1);
        let mask = unsafe { _mm256_loadu_si256(mask_words.as_ptr().cast::<__m256i>()) };
        let x = unsafe { _mm256_maskload_epi32(src.as_ptr().add(offset).cast::<i32>(), mask) };
        for slot in 0..4 {
            let destination = unsafe { rows[slot].add(offset) };
            let d = unsafe { _mm256_maskload_epi32(destination.cast::<i32>(), mask) };
            let product = _mm256_gf2p8mul_epi8(x, factors[slot]);
            unsafe {
                _mm256_maskstore_epi32(
                    destination.cast::<i32>(),
                    mask,
                    _mm256_xor_si256(d, product),
                );
            }
        }
    } else {
        for (byte, &source) in src.iter().enumerate().skip(offset) {
            let value = GfElem(source);
            for slot in 0..4 {
                unsafe {
                    *rows[slot].add(byte) ^= value.mul(coefficients[slot]).0;
                }
            }
        }
    }
}

#[allow(dead_code)]
#[target_feature(enable = "avx2,gfni")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_indexed_gfni(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    let dst_ptr = dst.as_mut_ptr();
    for destination_start in (0..destination_indices.len()).step_by(4) {
        let output_count = (destination_indices.len() - destination_start).min(4);
        let zero = _mm256_setzero_si256();
        let mut coefficients = [zero; 4];
        for slot in 0..output_count {
            coefficients[slot] = _mm256_set1_epi8(scales[destination_start + slot].coeff.0 as i8);
        }

        let mut offset = 0;
        while offset + 32 <= range_len {
            let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                if scale.coeff == GfElem::ZERO {
                    continue;
                }
                let row_offset =
                    destination_indices[destination_start + slot] * row_stride + byte_offset;
                let destination = unsafe { dst_ptr.add(row_offset + offset) };
                let d = unsafe { _mm256_loadu_si256(destination.cast::<__m256i>()) };
                let scaled = if scale.coeff == GfElem::ONE {
                    x
                } else {
                    _mm256_gf2p8mul_epi8(x, coefficients[slot])
                };
                unsafe {
                    _mm256_storeu_si256(destination.cast::<__m256i>(), _mm256_xor_si256(d, scaled));
                }
            }
            offset += 32;
        }

        for slot in 0..output_count {
            let scale = &scales[destination_start + slot];
            if scale.coeff == GfElem::ZERO {
                continue;
            }
            let row_offset =
                destination_indices[destination_start + slot] * row_stride + byte_offset;
            let tail_len = range_len - offset;
            let destination = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr.add(row_offset + offset), tail_len)
            };
            if scale.coeff == GfElem::ONE {
                xor_bytes_sse2_tail(destination, &src[offset..]);
            } else {
                xor_scaled_bytes_ssse3_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_avx2(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    // SAFETY: Loading fixed 16-byte tables from valid array references.
    let lo128 = unsafe { _mm_loadu_si128(lo.as_ptr().cast::<__m128i>()) };
    let hi128 = unsafe { _mm_loadu_si128(hi.as_ptr().cast::<__m128i>()) };
    let lo_tbl = _mm256_broadcastsi128_si256(lo128);
    let hi_tbl = _mm256_broadcastsi128_si256(hi128);
    let mask = _mm256_set1_epi8(0x0f);

    let mut offset = 0;
    let len = dst.len();
    while offset + 32 <= len {
        // SAFETY: `offset + 32 <= len`, and both pointers are derived from
        // valid slices. Unaligned loads/stores are used intentionally.
        let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
        let d = unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()) };

        let low_nibbles = _mm256_and_si256(x, mask);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);
        let low_prod = _mm256_shuffle_epi8(lo_tbl, low_nibbles);
        let high_prod = _mm256_shuffle_epi8(hi_tbl, high_nibbles);
        let scaled = _mm256_xor_si256(low_prod, high_prod);
        let mixed = _mm256_xor_si256(d, scaled);

        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(offset).cast::<__m256i>(), mixed) };
        offset += 32;
    }

    xor_scaled_bytes_ssse3_tail(&mut dst[offset..], lo, hi, &src[offset..]);
}

#[cfg_attr(not(test), allow(dead_code))]
#[target_feature(enable = "avx2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_avx2(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    let mask = _mm256_set1_epi8(0x0f);

    for destination_start in (0..destinations.len()).step_by(4) {
        let destination_end = (destination_start + 4).min(destinations.len());
        let output_count = destination_end - destination_start;
        let zero = _mm256_setzero_si256();
        let mut low_tables = [zero; 4];
        let mut high_tables = [zero; 4];
        for slot in 0..output_count {
            let scale = scale_table(coeffs[destination_start + slot]);
            let low = unsafe { _mm_loadu_si128(scale.lo.as_ptr().cast::<__m128i>()) };
            let high = unsafe { _mm_loadu_si128(scale.hi.as_ptr().cast::<__m128i>()) };
            low_tables[slot] = _mm256_broadcastsi128_si256(low);
            high_tables[slot] = _mm256_broadcastsi128_si256(high);
        }

        let mut offset = 0;
        while offset + 32 <= src.len() {
            let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            let low_nibbles = _mm256_and_si256(x, mask);
            let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask);

            for slot in 0..output_count {
                let destination = &mut destinations[destination_start + slot];
                let d = unsafe {
                    _mm256_loadu_si256(destination.as_ptr().add(offset).cast::<__m256i>())
                };
                let low_product = _mm256_shuffle_epi8(low_tables[slot], low_nibbles);
                let high_product = _mm256_shuffle_epi8(high_tables[slot], high_nibbles);
                let mixed = _mm256_xor_si256(d, _mm256_xor_si256(low_product, high_product));
                unsafe {
                    _mm256_storeu_si256(
                        destination.as_mut_ptr().add(offset).cast::<__m256i>(),
                        mixed,
                    );
                }
            }
            offset += 32;
        }

        for slot in 0..output_count {
            let scale = scale_table(coeffs[destination_start + slot]);
            xor_scaled_bytes_ssse3_tail(
                &mut destinations[destination_start + slot][offset..],
                &scale.lo,
                &scale.hi,
                &src[offset..],
            );
        }
    }
}

#[allow(dead_code)]
#[target_feature(enable = "avx2")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_indexed_avx2(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    let mask = _mm256_set1_epi8(0x0f);
    let dst_ptr = dst.as_mut_ptr();

    for destination_start in (0..destination_indices.len()).step_by(4) {
        let output_count = (destination_indices.len() - destination_start).min(4);
        let zero = _mm256_setzero_si256();
        let mut low_tables = [zero; 4];
        let mut high_tables = [zero; 4];
        let mut has_general_scale = false;

        for slot in 0..output_count {
            let scale = &scales[destination_start + slot];
            if scale.coeff != GfElem::ZERO && scale.coeff != GfElem::ONE {
                let low = unsafe { _mm_loadu_si128(scale.lo.as_ptr().cast::<__m128i>()) };
                let high = unsafe { _mm_loadu_si128(scale.hi.as_ptr().cast::<__m128i>()) };
                low_tables[slot] = _mm256_broadcastsi128_si256(low);
                high_tables[slot] = _mm256_broadcastsi128_si256(high);
                has_general_scale = true;
            }
        }

        let mut offset = 0;
        while offset + 32 <= range_len {
            let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            let (low_nibbles, high_nibbles) = if has_general_scale {
                (
                    _mm256_and_si256(x, mask),
                    _mm256_and_si256(_mm256_srli_epi16(x, 4), mask),
                )
            } else {
                (zero, zero)
            };

            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                if scale.coeff == GfElem::ZERO {
                    continue;
                }
                let row_offset =
                    destination_indices[destination_start + slot] * row_stride + byte_offset;
                let destination = unsafe { dst_ptr.add(row_offset + offset) };
                let d = unsafe { _mm256_loadu_si256(destination.cast::<__m256i>()) };
                let scaled = if scale.coeff == GfElem::ONE {
                    x
                } else {
                    let low_product = _mm256_shuffle_epi8(low_tables[slot], low_nibbles);
                    let high_product = _mm256_shuffle_epi8(high_tables[slot], high_nibbles);
                    _mm256_xor_si256(low_product, high_product)
                };
                unsafe {
                    _mm256_storeu_si256(destination.cast::<__m256i>(), _mm256_xor_si256(d, scaled));
                }
            }
            offset += 32;
        }

        for slot in 0..output_count {
            let scale = &scales[destination_start + slot];
            if scale.coeff == GfElem::ZERO {
                continue;
            }
            let row_offset =
                destination_indices[destination_start + slot] * row_stride + byte_offset;
            let tail_len = range_len - offset;
            let destination = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr.add(row_offset + offset), tail_len)
            };
            if scale.coeff == GfElem::ONE {
                xor_bytes_sse2_tail(destination, &src[offset..]);
            } else {
                xor_scaled_bytes_ssse3_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_ssse3(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    xor_scaled_bytes_ssse3_tail(dst, lo, hi, src);
}

#[target_feature(enable = "ssse3")]
fn xor_scaled_bytes_ssse3_tail(dst: &mut [u8], lo: &[u8; 16], hi: &[u8; 16], src: &[u8]) {
    // SAFETY: Loading fixed 16-byte tables from valid array references.
    let lo_tbl = unsafe { _mm_loadu_si128(lo.as_ptr().cast::<__m128i>()) };
    let hi_tbl = unsafe { _mm_loadu_si128(hi.as_ptr().cast::<__m128i>()) };
    let mask = _mm_set1_epi8(0x0f);

    let mut offset = 0;
    let len = dst.len();
    while offset + 16 <= len {
        // SAFETY: `offset + 16 <= len`, and both pointers are derived from
        // valid slices. Unaligned loads/stores are used intentionally.
        let x = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast::<__m128i>()) };
        let d = unsafe { _mm_loadu_si128(dst.as_ptr().add(offset).cast::<__m128i>()) };

        let low_nibbles = _mm_and_si128(x, mask);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
        let low_prod = _mm_shuffle_epi8(lo_tbl, low_nibbles);
        let high_prod = _mm_shuffle_epi8(hi_tbl, high_nibbles);
        let scaled = _mm_xor_si128(low_prod, high_prod);
        let mixed = _mm_xor_si128(d, scaled);

        unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast::<__m128i>(), mixed) };
        offset += 16;
    }

    xor_scaled_bytes_nibble_tail(&mut dst[offset..], lo, hi, &src[offset..]);
}

// Safe boundaries used by generic dispatch after it has selected the matching plan.
// Each wrapper keeps the CPU-feature proof and geometry proof adjacent to the
// architecture-specific unsafe call.
pub(super) fn dispatch_xor_bytes_avx2(dst: &mut [u8], src: &[u8]) {
    // SAFETY: Runtime dispatch selected AVX2 and the caller supplied equal-length slices.
    unsafe { xor_bytes_avx2(dst, src) }
}

pub(super) fn dispatch_xor_bytes_sse2(dst: &mut [u8], src: &[u8]) {
    // SAFETY: Runtime dispatch selected SSSE3, which implies SSE2; slices have equal lengths.
    unsafe { xor_bytes_sse2(dst, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_gfni(dst: &mut [u8], coeff: GfElem, src: &[u8]) {
    // SAFETY: Runtime dispatch selected AVX2+GFNI and the caller supplied equal-length slices.
    unsafe { xor_scaled_bytes_gfni(dst, coeff, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_avx2(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2; tables and equal-length slices are valid.
    unsafe { xor_scaled_bytes_avx2(dst, lo, hi, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_ssse3(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected SSSE3; tables and equal-length slices are valid.
    unsafe { xor_scaled_bytes_ssse3(dst, lo, hi, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_many_indexed_gfni(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2+GFNI; the rows view validated in-bounds,
    // non-overlapping ranges and `src.len() == range_len` before this call.
    unsafe {
        xor_scaled_bytes_many_indexed_gfni(
            dst,
            row_stride,
            byte_offset,
            range_len,
            destination_indices,
            scales,
            src,
        )
    }
}

pub(super) fn dispatch_xor_scaled_bytes_many_indexed_avx2(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2; the rows view validated in-bounds,
    // non-overlapping ranges and `src.len() == range_len` before this call.
    unsafe {
        xor_scaled_bytes_many_indexed_avx2(
            dst,
            row_stride,
            byte_offset,
            range_len,
            destination_indices,
            scales,
            src,
        )
    }
}

pub(super) fn dispatch_xor_scaled_bytes_many_gfni(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2+GFNI; every destination was validated to match `src`.
    unsafe { xor_scaled_bytes_many_gfni(destinations, coeffs, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_many_avx2(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2; every destination was validated to match `src`.
    unsafe { xor_scaled_bytes_many_avx2(destinations, coeffs, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_4_indexed_gfni(
    dst: &mut [u8],
    row_stride: usize,
    destination_indices: &[usize; 4],
    coefficients: &[GfElem; 4],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2+GFNI; `IndexedDestinationRows` validated
    // all four non-empty row ranges as in-bounds and disjoint before this call.
    unsafe {
        xor_scaled_bytes_4_indexed_gfni(dst, row_stride, destination_indices, coefficients, src)
    }
}

pub(super) fn dispatch_xor_scaled_bytes_rows_gfni(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2+GFNI; flat row geometry and source length were validated.
    unsafe { xor_scaled_bytes_rows_gfni(repairs, symbol_len, coeffs, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_rows_avx2(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: Runtime dispatch selected AVX2; flat row geometry and source length were validated.
    unsafe { xor_scaled_bytes_rows_avx2(repairs, symbol_len, coeffs, src) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::dispatch::gfni_available;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn gfni_multiplication_matches_xtime_exhaustively() {
        if !gfni_available() {
            return;
        }

        let src: Vec<u8> = (0..=u8::MAX).collect();
        for coefficient in 0..=u8::MAX {
            let scale = ScaleTable::new(GfElem(coefficient));
            let mut actual = vec![0u8; src.len()];
            // SAFETY: The feature check above guarantees AVX2 and GFNI. Calling
            // the low-level kernel ensures every pair executes GF2P8MULB,
            // including the zero and one coefficient cases.
            unsafe {
                xor_scaled_bytes_gfni(&mut actual, scale.coeff, &src);
            }
            for (value, &product) in actual.iter().enumerate() {
                assert_eq!(
                    product,
                    GfElem(value as u8).mul_xtime(GfElem(coefficient)).0,
                    "value={value:#04x}, coefficient={coefficient:#04x}"
                );
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn gfni_axpy_matches_scalar_across_lengths_and_offsets() {
        if !gfni_available() {
            return;
        }
        // Lengths span the 128-byte unrolled body, the 32-byte loop, the
        // mask-tail (multiple of 4), and the scalar tail; dst starts nonzero to
        // exercise the XOR accumulation.
        for &len in &[
            0usize, 1, 3, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 160, 191, 192, 255, 256, 257,
            384, 1400,
        ] {
            for &coeff in &[0u8, 1, 2, 0x53, 0xff] {
                let src: Vec<u8> = (0..len)
                    .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
                    .collect();
                let mut actual: Vec<u8> = (0..len)
                    .map(|i| (i as u8).wrapping_mul(17).wrapping_add(0x5a))
                    .collect();
                let mut expected = actual.clone();
                for (slot, &value) in expected.iter_mut().zip(&src) {
                    *slot ^= GfElem(value).mul_xtime(GfElem(coeff)).0;
                }
                // SAFETY: gfni_available() confirmed AVX2+GFNI; slices are equal length.
                unsafe {
                    xor_scaled_bytes_gfni(&mut actual, GfElem(coeff), &src);
                }
                assert_eq!(actual, expected, "len={len}, coeff={coeff:#04x}");
            }
        }
    }
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn forced_gfni_grouped_indexed_matches_reference() {
        if !gfni_available() {
            return;
        }

        for output_count in [0, 1, 4, 5, 9] {
            for range_len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let row_stride = range_len + 11;
                let byte_offset = 5;
                let row_count = output_count * 2 + 3;
                let indices: Vec<_> = (0..output_count).map(|i| i * 2 + 1).collect();
                let scales: Vec<_> = (0..output_count)
                    .map(|i| match i % 4 {
                        0 => ScaleTable::new(GfElem::ZERO),
                        1 => ScaleTable::new(GfElem::ONE),
                        _ => ScaleTable::new(GfElem(0x53u8.wrapping_add(i as u8))),
                    })
                    .collect();
                let src: Vec<_> = (0..range_len)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                    .collect();
                let mut expected: Vec<_> = (0..row_count * row_stride)
                    .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                    .collect();
                let mut actual = expected.clone();

                for (&index, scale) in indices.iter().zip(&scales) {
                    let start = index * row_stride + byte_offset;
                    for (destination, &source) in
                        expected[start..start + range_len].iter_mut().zip(&src)
                    {
                        *destination ^= GfElem(source).mul_xtime(scale.coeff).0;
                    }
                }

                // SAFETY: The feature check above guarantees AVX2 and GFNI;
                // all generated rows are in bounds, sorted, and disjoint.
                unsafe {
                    xor_scaled_bytes_many_indexed_gfni(
                        &mut actual,
                        row_stride,
                        byte_offset,
                        range_len,
                        &indices,
                        &scales,
                        &src,
                    );
                }
                assert_eq!(actual, expected, "outputs={output_count}, len={range_len}");
            }
        }
    }
}
