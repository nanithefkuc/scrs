//! AArch64 NEON kernels.
#![cfg(target_arch = "aarch64")]
#![allow(unsafe_code)]

use super::{
    scalar::{xor_bytes_scalar, xor_scaled_bytes_nibble_tail},
    scale_table::{ScaleTable, scale_table},
};
use crate::gf256::GfElem;
use std::arch::aarch64::*;

use std::arch::aarch64::*;

/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_bytes_neon(dst: &mut [u8], src: &[u8]) {
    let mut offset = 0;
    let len = dst.len();
    // Dual 16-byte unroll for better ILP on typical Cortex cores.
    while offset + 32 <= len {
        // SAFETY: bounds checked; AArch64 permits unaligned vector accesses.
        let d0 = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
        let s0 = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
        let d1 = unsafe { vld1q_u8(dst.as_ptr().add(offset + 16)) };
        let s1 = unsafe { vld1q_u8(src.as_ptr().add(offset + 16)) };
        unsafe {
            vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(d0, s0));
            vst1q_u8(dst.as_mut_ptr().add(offset + 16), veorq_u8(d1, s1));
        }
        offset += 32;
    }
    while offset + 16 <= len {
        let d = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
        let s = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
        unsafe { vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(d, s)) };
        offset += 16;
    }

    xor_bytes_scalar(&mut dst[offset..], &src[offset..]);
}

/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_neon(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    // SAFETY: Loading fixed 16-byte tables from valid array references.
    let lo_tbl = unsafe { vld1q_u8(lo.as_ptr()) };
    let hi_tbl = unsafe { vld1q_u8(hi.as_ptr()) };
    let mask = unsafe { vdupq_n_u8(0x0f) };

    let mut offset = 0;
    let len = dst.len();
    while offset + 32 <= len {
        // SAFETY: bounds checked; unaligned NEON loads/stores are allowed.
        let x0 = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
        let d0 = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
        let x1 = unsafe { vld1q_u8(src.as_ptr().add(offset + 16)) };
        let d1 = unsafe { vld1q_u8(dst.as_ptr().add(offset + 16)) };

        let low0 = unsafe { vandq_u8(x0, mask) };
        let high0 = unsafe { vandq_u8(vshrq_n_u8(x0, 4), mask) };
        let low1 = unsafe { vandq_u8(x1, mask) };
        let high1 = unsafe { vandq_u8(vshrq_n_u8(x1, 4), mask) };

        let scaled0 = unsafe { veorq_u8(vqtbl1q_u8(lo_tbl, low0), vqtbl1q_u8(hi_tbl, high0)) };
        let scaled1 = unsafe { veorq_u8(vqtbl1q_u8(lo_tbl, low1), vqtbl1q_u8(hi_tbl, high1)) };
        unsafe {
            vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(d0, scaled0));
            vst1q_u8(dst.as_mut_ptr().add(offset + 16), veorq_u8(d1, scaled1));
        }
        offset += 32;
    }
    while offset + 16 <= len {
        let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
        let d = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
        let low_nibbles = unsafe { vandq_u8(x, mask) };
        let high_nibbles = unsafe { vandq_u8(vshrq_n_u8(x, 4), mask) };
        let scaled = unsafe {
            veorq_u8(
                vqtbl1q_u8(lo_tbl, low_nibbles),
                vqtbl1q_u8(hi_tbl, high_nibbles),
            )
        };
        unsafe { vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(d, scaled)) };
        offset += 16;
    }

    xor_scaled_bytes_nibble_tail(&mut dst[offset..], lo, hi, &src[offset..]);
}

/// Four-output source-major nibble AXPY: one source load updates four rows.
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_4_indexed_neon(
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

    let mask = unsafe { vdupq_n_u8(0x0f) };
    let zero = unsafe { vdupq_n_u8(0) };
    let mut low_tables = [zero; 4];
    let mut high_tables = [zero; 4];
    let mut kinds = [0u8; 4]; // 0=zero, 1=one, 2=general
    for slot in 0..4 {
        let coeff = coefficients[slot];
        if coeff == GfElem::ZERO {
            kinds[slot] = 0;
        } else if coeff == GfElem::ONE {
            kinds[slot] = 1;
        } else {
            kinds[slot] = 2;
            let scale = scale_table(coeff);
            low_tables[slot] = unsafe { vld1q_u8(scale.lo.as_ptr()) };
            high_tables[slot] = unsafe { vld1q_u8(scale.hi.as_ptr()) };
        }
    }

    let mut offset = 0;
    while offset + 16 <= src.len() {
        let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
        let low_nibbles = unsafe { vandq_u8(x, mask) };
        let high_nibbles = unsafe { vandq_u8(vshrq_n_u8(x, 4), mask) };

        for slot in 0..4 {
            if kinds[slot] == 0 {
                continue;
            }
            let destination = unsafe { rows[slot].add(offset) };
            let d = unsafe { vld1q_u8(destination) };
            let scaled = if kinds[slot] == 1 {
                x
            } else {
                unsafe {
                    veorq_u8(
                        vqtbl1q_u8(low_tables[slot], low_nibbles),
                        vqtbl1q_u8(high_tables[slot], high_nibbles),
                    )
                }
            };
            unsafe { vst1q_u8(destination, veorq_u8(d, scaled)) };
        }
        offset += 16;
    }

    // Scalar/nibble tail for the remainder.
    for slot in 0..4 {
        if kinds[slot] == 0 {
            continue;
        }
        let tail_len = src.len() - offset;
        let destination =
            unsafe { std::slice::from_raw_parts_mut(rows[slot].add(offset), tail_len) };
        if kinds[slot] == 1 {
            xor_bytes_scalar(destination, &src[offset..]);
        } else {
            let scale = scale_table(coefficients[slot]);
            xor_scaled_bytes_nibble_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
        }
    }
}

#[allow(dead_code)]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_indexed_neon(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    let mask = unsafe { vdupq_n_u8(0x0f) };
    let zero = unsafe { vdupq_n_u8(0) };
    let dst_ptr = dst.as_mut_ptr();

    for destination_start in (0..destination_indices.len()).step_by(4) {
        let output_count = (destination_indices.len() - destination_start).min(4);
        let mut low_tables = [zero; 4];
        let mut high_tables = [zero; 4];
        let mut has_general_scale = false;

        for slot in 0..output_count {
            let scale = &scales[destination_start + slot];
            if scale.coeff != GfElem::ZERO && scale.coeff != GfElem::ONE {
                low_tables[slot] = unsafe { vld1q_u8(scale.lo.as_ptr()) };
                high_tables[slot] = unsafe { vld1q_u8(scale.hi.as_ptr()) };
                has_general_scale = true;
            }
        }

        let mut offset = 0;
        while offset + 16 <= range_len {
            let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
            let (low_nibbles, high_nibbles) = if has_general_scale {
                (unsafe { vandq_u8(x, mask) }, unsafe {
                    vandq_u8(vshrq_n_u8(x, 4), mask)
                })
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
                let d = unsafe { vld1q_u8(destination) };
                let scaled = if scale.coeff == GfElem::ONE {
                    x
                } else {
                    unsafe {
                        veorq_u8(
                            vqtbl1q_u8(low_tables[slot], low_nibbles),
                            vqtbl1q_u8(high_tables[slot], high_nibbles),
                        )
                    }
                };
                unsafe { vst1q_u8(destination, veorq_u8(d, scaled)) };
            }
            offset += 16;
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
                xor_bytes_scalar(destination, &src[offset..]);
            } else {
                xor_scaled_bytes_nibble_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
            }
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_many_neon(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    let mask = unsafe { vdupq_n_u8(0x0f) };
    let zero = unsafe { vdupq_n_u8(0) };

    for destination_start in (0..destinations.len()).step_by(4) {
        let destination_end = (destination_start + 4).min(destinations.len());
        let output_count = destination_end - destination_start;
        let mut low_tables = [zero; 4];
        let mut high_tables = [zero; 4];
        let mut kinds = [0u8; 4]; // 0=zero, 1=one, 2=general
        for slot in 0..output_count {
            let coeff = coeffs[destination_start + slot];
            if coeff == GfElem::ZERO {
                kinds[slot] = 0;
            } else if coeff == GfElem::ONE {
                kinds[slot] = 1;
            } else {
                kinds[slot] = 2;
                let scale = scale_table(coeff);
                low_tables[slot] = unsafe { vld1q_u8(scale.lo.as_ptr()) };
                high_tables[slot] = unsafe { vld1q_u8(scale.hi.as_ptr()) };
            }
        }

        let mut offset = 0;
        while offset + 16 <= src.len() {
            let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
            let low_nibbles = unsafe { vandq_u8(x, mask) };
            let high_nibbles = unsafe { vandq_u8(vshrq_n_u8(x, 4), mask) };

            for slot in 0..output_count {
                if kinds[slot] == 0 {
                    continue;
                }
                let destination = &mut destinations[destination_start + slot];
                let d = unsafe { vld1q_u8(destination.as_ptr().add(offset)) };
                let scaled = if kinds[slot] == 1 {
                    x
                } else {
                    unsafe {
                        veorq_u8(
                            vqtbl1q_u8(low_tables[slot], low_nibbles),
                            vqtbl1q_u8(high_tables[slot], high_nibbles),
                        )
                    }
                };
                unsafe { vst1q_u8(destination.as_mut_ptr().add(offset), veorq_u8(d, scaled)) };
            }
            offset += 16;
        }

        for slot in 0..output_count {
            if kinds[slot] == 0 {
                continue;
            }
            let destination = &mut destinations[destination_start + slot][offset..];
            if kinds[slot] == 1 {
                xor_bytes_scalar(destination, &src[offset..]);
            } else {
                let scale = scale_table(coeffs[destination_start + slot]);
                xor_scaled_bytes_nibble_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
            }
        }
    }
}

/// Flat multi-destination NEON kernel for the streaming encoder.
/// # Safety
/// Callers must enable the function's target features; supply valid, equal-length source and destination ranges for byte kernels; and, for row kernels, ensure every computed row range is in-bounds and non-overlapping.
pub(super) unsafe fn xor_scaled_bytes_rows_neon(
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
        // SAFETY: Caller validated geometry; indices address disjoint rows.
        unsafe {
            xor_scaled_bytes_4_indexed_neon(repairs, symbol_len, &indices, &coefficients, src);
        }
        j += 4;
    }
    while j < m {
        let start = j * symbol_len;
        let scale = scale_table(coeffs[j]);
        // SAFETY: Single-row slice is in-bounds.
        unsafe {
            xor_scaled_bytes_neon(
                &mut repairs[start..start + symbol_len],
                &scale.lo,
                &scale.hi,
                src,
            );
        }
        j += 1;
    }
}

// Safe boundaries used by generic dispatch after it has selected the AArch64 plan.
pub(super) fn dispatch_xor_bytes_neon(dst: &mut [u8], src: &[u8]) {
    // SAFETY: NEON is an AArch64 baseline feature and the caller supplied equal-length slices.
    unsafe { xor_bytes_neon(dst, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_neon(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    // SAFETY: NEON is an AArch64 baseline feature; tables and equal-length slices are valid.
    unsafe { xor_scaled_bytes_neon(dst, lo, hi, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_many_indexed_neon(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    // SAFETY: NEON is baseline; the rows view validated in-bounds, non-overlapping ranges
    // and `src.len() == range_len` before this call.
    unsafe {
        xor_scaled_bytes_many_indexed_neon(
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

pub(super) fn dispatch_xor_scaled_bytes_many_neon(
    destinations: &mut [Vec<u8>],
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: NEON is baseline and every destination was validated to match `src`.
    unsafe { xor_scaled_bytes_many_neon(destinations, coeffs, src) }
}

pub(super) fn dispatch_xor_scaled_bytes_4_indexed_neon(
    dst: &mut [u8],
    row_stride: usize,
    destination_indices: &[usize; 4],
    coefficients: &[GfElem; 4],
    src: &[u8],
) {
    // SAFETY: NEON is baseline; `IndexedDestinationRows` validated all four non-empty
    // row ranges as in-bounds and disjoint before this call.
    unsafe {
        xor_scaled_bytes_4_indexed_neon(dst, row_stride, destination_indices, coefficients, src)
    }
}

pub(super) fn dispatch_xor_scaled_bytes_rows_neon(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    // SAFETY: NEON is baseline; flat row geometry and source length were validated.
    unsafe { xor_scaled_bytes_rows_neon(repairs, symbol_len, coeffs, src) }
}
