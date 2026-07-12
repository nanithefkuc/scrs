//! SIMD payload kernels.
//!
#![allow(unsafe_code)]
//!
//! The public surface of this module is safe. Architecture-specific internals
//! use small `unsafe` blocks for Rust SIMD intrinsics and are guarded by runtime
//! feature detection where needed.

use crate::gf256::GfElem;

/// Whether the AVX2 GFNI kernels can run on this CPU.
#[inline]
pub(crate) fn gfni_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Compact precomputed multiplication tables for one GF(256) coefficient.
#[derive(Clone, Debug)]
pub(crate) struct ScaleTable {
    pub(crate) coeff: GfElem,
    lo: [u8; 16],
    hi: [u8; 16],
}

impl ScaleTable {
    /// Build compact nibble tables for `coeff`.
    pub(crate) fn new(coeff: GfElem) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        if coeff == GfElem::ONE {
            for i in 0..16 {
                lo[i] = i as u8;
                hi[i] = (i << 4) as u8;
            }
        } else if coeff != GfElem::ZERO {
            for i in 0..16 {
                lo[i] = GfElem(i as u8).mul(coeff).0;
                hi[i] = GfElem((i << 4) as u8).mul(coeff).0;
            }
        }

        Self { coeff, lo, hi }
    }
}

/// `dst[:] <- dst[:] ^ src[:]`.
pub(crate) fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "xor length mismatch");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: The runtime check above guarantees AVX2 is available. The
            // kernel uses unaligned loads/stores and is bounded by slice lengths.
            unsafe {
                x86::xor_bytes_avx2(dst, src);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: The runtime check above guarantees SSE2 is available. The
            // kernel uses unaligned loads/stores and is bounded by slice lengths.
            unsafe {
                x86::xor_bytes_sse2(dst, src);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON/AdvSIMD is part of the aarch64 baseline used by Android ARM64.
        // SAFETY: The kernel uses unaligned loads/stores and is bounded by slice
        // lengths. NEON is available on aarch64 targets.
        unsafe {
            neon::xor_bytes_neon(dst, src);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    xor_bytes_scalar(dst, src);
}

/// Force the direct GFNI AXPY path when it is available.
///
/// Returns `false` without modifying `dst` when this build or CPU cannot run
/// the AVX2 GFNI kernel.
#[allow(dead_code)]
pub(crate) fn xor_scaled_bytes_gfni(dst: &mut [u8], scale: &ScaleTable, src: &[u8]) -> bool {
    assert_eq!(dst.len(), src.len(), "scaled byte axpy length mismatch");
    if !gfni_available() {
        return false;
    }
    if scale.coeff == GfElem::ZERO {
        return true;
    }
    if scale.coeff == GfElem::ONE {
        xor_bytes(dst, src);
        return true;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // SAFETY: `gfni_available` guarantees AVX2 and GFNI. The kernel uses
        // unaligned, slice-bounded accesses.
        unsafe {
            x86::xor_scaled_bytes_gfni(dst, scale, src);
        }
        true
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// `dst[:] <- dst[:] ^ scale.coeff * src[:]` over GF(256).
pub(crate) fn xor_scaled_bytes(dst: &mut [u8], scale: &ScaleTable, src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "scaled byte axpy length mismatch");
    if scale.coeff == GfElem::ZERO {
        return;
    }
    if scale.coeff == GfElem::ONE {
        xor_bytes(dst, src);
        return;
    }

    let lo = &scale.lo;
    let hi = &scale.hi;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if gfni_available() {
            // SAFETY: Runtime detection guarantees AVX2 and GFNI. The kernel
            // uses unaligned loads/stores and is bounded by slice lengths.
            unsafe {
                x86::xor_scaled_bytes_gfni(dst, scale, src);
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: The runtime check above guarantees AVX2 is available. The
            // kernel uses unaligned loads/stores and is bounded by slice lengths.
            unsafe {
                x86::xor_scaled_bytes_avx2(dst, lo, hi, src);
            }
            return;
        }
        if std::is_x86_feature_detected!("ssse3") {
            // SAFETY: The runtime check above guarantees SSSE3 is available. The
            // kernel uses unaligned loads/stores and is bounded by slice lengths.
            unsafe {
                x86::xor_scaled_bytes_ssse3(dst, lo, hi, src);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON/AdvSIMD is part of the aarch64 baseline used by Android ARM64.
        // SAFETY: The kernel uses unaligned loads/stores and is bounded by slice
        // lengths. NEON is available on aarch64 targets.
        unsafe {
            neon::xor_scaled_bytes_neon(dst, lo, hi, src);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    xor_scaled_bytes_nibble_tail(dst, lo, hi, src);
}

/// Validated indexed rows in a flat reconstruction buffer.
pub(crate) struct IndexedDestinationRows<'dst, 'indices> {
    dst: &'dst mut [u8],
    symbol_len: usize,
    indices: &'indices [usize],
}

impl<'dst, 'indices> IndexedDestinationRows<'dst, 'indices> {
    /// Validate row bounds and disjointness once, before the source-major loop.
    pub(crate) fn new(dst: &'dst mut [u8], symbol_len: usize, indices: &'indices [usize]) -> Self {
        let mut previous = None;
        for &index in indices {
            let start = index
                .checked_mul(symbol_len)
                .expect("destination row offset overflow");
            let end = start
                .checked_add(symbol_len)
                .expect("destination row end overflow");
            assert!(end <= dst.len(), "destination row out of bounds");
            if let Some(previous) = previous {
                assert!(
                    index > previous,
                    "destination indices must be unique and strictly increasing"
                );
            }
            previous = Some(index);
        }
        Self {
            dst,
            symbol_len,
            indices,
        }
    }

    /// Add one source symbol, with distinct scales, to all indexed rows.
    pub(crate) fn xor_scaled(&mut self, scales: &[ScaleTable], src: &[u8]) {
        assert_eq!(src.len(), self.symbol_len, "source symbol length mismatch");
        self.xor_scaled_range(scales, src, 0);
    }

    /// Add a source range, with distinct scales, to all indexed rows.
    pub(crate) fn xor_scaled_range(
        &mut self,
        scales: &[ScaleTable],
        src_chunk: &[u8],
        byte_offset: usize,
    ) {
        assert_eq!(
            self.indices.len(),
            scales.len(),
            "destination/scale count mismatch"
        );
        let range_end = byte_offset
            .checked_add(src_chunk.len())
            .expect("source range end overflow");
        assert!(range_end <= self.symbol_len, "source range out of bounds");

        xor_scaled_bytes_many_indexed_trusted(
            self.dst,
            self.symbol_len,
            byte_offset,
            src_chunk.len(),
            self.indices,
            scales,
            src_chunk,
        );
    }
}

/// Add one source symbol, with distinct scales, to indexed rows in a flat buffer.
#[allow(dead_code)]
pub(crate) fn xor_scaled_bytes_many_indexed(
    dst: &mut [u8],
    symbol_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    IndexedDestinationRows::new(dst, symbol_len, destination_indices).xor_scaled(scales, src);
}

fn xor_scaled_bytes_many_indexed_trusted(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if gfni_available() {
            // SAFETY: AVX2 and GFNI are runtime-detected, and all row ranges
            // were checked above and are disjoint.
            unsafe {
                x86::xor_scaled_bytes_many_indexed_gfni(
                    dst,
                    row_stride,
                    byte_offset,
                    range_len,
                    destination_indices,
                    scales,
                    src,
                );
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is runtime-detected, and all row ranges were checked
            // above and are disjoint because destination indices are unique.
            unsafe {
                x86::xor_scaled_bytes_many_indexed_avx2(
                    dst,
                    row_stride,
                    byte_offset,
                    range_len,
                    destination_indices,
                    scales,
                    src,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 baseline. All row ranges were
        // checked above and are disjoint because destination indices are unique.
        unsafe {
            neon::xor_scaled_bytes_many_indexed_neon(
                dst,
                row_stride,
                byte_offset,
                range_len,
                destination_indices,
                scales,
                src,
            );
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        debug_assert_eq!(range_len, src.len());
        for (&index, scale) in destination_indices.iter().zip(scales) {
            let start = index * row_stride + byte_offset;
            xor_scaled_bytes(&mut dst[start..start + range_len], scale, src);
        }
    }
}

/// Add one source symbol, with distinct scales, to several destinations.
pub(crate) fn xor_scaled_bytes_many(
    destinations: &mut [Vec<u8>],
    scales: &[crate::encoder::EncoderScaleTable],
    src: &[u8],
) {
    assert_eq!(destinations.len(), scales.len());
    assert!(destinations.iter().all(|dst| dst.len() == src.len()));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if gfni_available() {
            // SAFETY: Runtime detection guarantees AVX2 and GFNI. All
            // destination and source lengths are validated above.
            unsafe {
                x86::xor_scaled_bytes_many_gfni(destinations, scales, src);
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime detection guarantees AVX2. All destination and
            // source lengths are validated above.
            unsafe {
                x86::xor_scaled_bytes_many_avx2(destinations, scales, src);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 baseline and accesses are bounded.
        unsafe {
            neon::xor_scaled_bytes_many_neon(destinations, scales, src);
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    for (dst, scale) in destinations.iter_mut().zip(scales) {
        scale.xor_scaled(dst, src);
    }
}

fn xor_scaled_bytes_nibble_tail(dst: &mut [u8], lo: &[u8; 16], hi: &[u8; 16], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= lo[(s & 0x0f) as usize] ^ hi[(s >> 4) as usize];
    }
}

fn xor_bytes_scalar(dst: &mut [u8], src: &[u8]) {
    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mut d_arr = [0u8; 8];
        let mut s_arr = [0u8; 8];
        d_arr.copy_from_slice(d);
        s_arr.copy_from_slice(s);
        let mixed = u64::from_ne_bytes(d_arr) ^ u64::from_ne_bytes(s_arr);
        d.copy_from_slice(&mixed.to_ne_bytes());
    }

    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
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

        super::xor_bytes_scalar(&mut dst[offset..], &src[offset..]);
    }

    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn xor_scaled_bytes_gfni(
        dst: &mut [u8],
        scale: &super::ScaleTable,
        src: &[u8],
    ) {
        let coefficient = _mm256_set1_epi8(scale.coeff.0 as i8);
        let mut offset = 0;
        while offset + 32 <= dst.len() {
            // SAFETY: The loop condition bounds each unaligned vector access.
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

        xor_scaled_bytes_ssse3_tail(&mut dst[offset..], &scale.lo, &scale.hi, &src[offset..]);
    }

    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn xor_scaled_bytes_many_gfni(
        destinations: &mut [Vec<u8>],
        scales: &[crate::encoder::EncoderScaleTable],
        src: &[u8],
    ) {
        for destination_start in (0..destinations.len()).step_by(4) {
            let output_count = (destinations.len() - destination_start).min(4);
            let zero = _mm256_setzero_si256();
            let mut coefficients = [zero; 4];
            for slot in 0..output_count {
                coefficients[slot] =
                    _mm256_set1_epi8(scales[destination_start + slot].coeff.0 as i8);
            }

            let mut offset = 0;
            while offset + 32 <= src.len() {
                let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
                for slot in 0..output_count {
                    let scale = &scales[destination_start + slot];
                    if scale.coeff == super::GfElem::ZERO {
                        continue;
                    }
                    let destination = &mut destinations[destination_start + slot];
                    let d = unsafe {
                        _mm256_loadu_si256(destination.as_ptr().add(offset).cast::<__m256i>())
                    };
                    let scaled = if scale.coeff == super::GfElem::ONE {
                        x
                    } else {
                        _mm256_gf2p8mul_epi8(x, coefficients[slot])
                    };
                    unsafe {
                        _mm256_storeu_si256(
                            destination.as_mut_ptr().add(offset).cast::<__m256i>(),
                            _mm256_xor_si256(d, scaled),
                        );
                    }
                }
                offset += 32;
            }

            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                if scale.coeff == super::GfElem::ZERO {
                    continue;
                }
                let destination = &mut destinations[destination_start + slot][offset..];
                if scale.coeff == super::GfElem::ONE {
                    xor_bytes_sse2_tail(destination, &src[offset..]);
                } else {
                    xor_scaled_bytes_ssse3_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
                }
            }
        }
    }

    #[allow(dead_code)]
    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn xor_scaled_bytes_many_indexed_gfni(
        dst: &mut [u8],
        row_stride: usize,
        byte_offset: usize,
        range_len: usize,
        destination_indices: &[usize],
        scales: &[super::ScaleTable],
        src: &[u8],
    ) {
        let dst_ptr = dst.as_mut_ptr();
        for destination_start in (0..destination_indices.len()).step_by(4) {
            let output_count = (destination_indices.len() - destination_start).min(4);
            let zero = _mm256_setzero_si256();
            let mut coefficients = [zero; 4];
            for slot in 0..output_count {
                coefficients[slot] =
                    _mm256_set1_epi8(scales[destination_start + slot].coeff.0 as i8);
            }

            let mut offset = 0;
            while offset + 32 <= range_len {
                let x = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
                for slot in 0..output_count {
                    let scale = &scales[destination_start + slot];
                    if scale.coeff == super::GfElem::ZERO {
                        continue;
                    }
                    let row_offset =
                        destination_indices[destination_start + slot] * row_stride + byte_offset;
                    let destination = unsafe { dst_ptr.add(row_offset + offset) };
                    let d = unsafe { _mm256_loadu_si256(destination.cast::<__m256i>()) };
                    let scaled = if scale.coeff == super::GfElem::ONE {
                        x
                    } else {
                        _mm256_gf2p8mul_epi8(x, coefficients[slot])
                    };
                    unsafe {
                        _mm256_storeu_si256(
                            destination.cast::<__m256i>(),
                            _mm256_xor_si256(d, scaled),
                        );
                    }
                }
                offset += 32;
            }

            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                if scale.coeff == super::GfElem::ZERO {
                    continue;
                }
                let row_offset =
                    destination_indices[destination_start + slot] * row_stride + byte_offset;
                let tail_len = range_len - offset;
                let destination = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.add(row_offset + offset), tail_len)
                };
                if scale.coeff == super::GfElem::ONE {
                    xor_bytes_sse2_tail(destination, &src[offset..]);
                } else {
                    xor_scaled_bytes_ssse3_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
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

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn xor_scaled_bytes_many_avx2(
        destinations: &mut [Vec<u8>],
        scales: &[crate::encoder::EncoderScaleTable],
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
                let scale = &scales[destination_start + slot];
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
                let scale = &scales[destination_start + slot];
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
    pub(super) unsafe fn xor_scaled_bytes_many_indexed_avx2(
        dst: &mut [u8],
        row_stride: usize,
        byte_offset: usize,
        range_len: usize,
        destination_indices: &[usize],
        scales: &[super::ScaleTable],
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
                if scale.coeff != super::GfElem::ZERO && scale.coeff != super::GfElem::ONE {
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
                    if scale.coeff == super::GfElem::ZERO {
                        continue;
                    }
                    let row_offset =
                        destination_indices[destination_start + slot] * row_stride + byte_offset;
                    let destination = unsafe { dst_ptr.add(row_offset + offset) };
                    let d = unsafe { _mm256_loadu_si256(destination.cast::<__m256i>()) };
                    let scaled = if scale.coeff == super::GfElem::ONE {
                        x
                    } else {
                        let low_product = _mm256_shuffle_epi8(low_tables[slot], low_nibbles);
                        let high_product = _mm256_shuffle_epi8(high_tables[slot], high_nibbles);
                        _mm256_xor_si256(low_product, high_product)
                    };
                    unsafe {
                        _mm256_storeu_si256(
                            destination.cast::<__m256i>(),
                            _mm256_xor_si256(d, scaled),
                        );
                    }
                }
                offset += 32;
            }

            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                if scale.coeff == super::GfElem::ZERO {
                    continue;
                }
                let row_offset =
                    destination_indices[destination_start + slot] * row_stride + byte_offset;
                let tail_len = range_len - offset;
                let destination = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.add(row_offset + offset), tail_len)
                };
                if scale.coeff == super::GfElem::ONE {
                    xor_bytes_sse2_tail(destination, &src[offset..]);
                } else {
                    xor_scaled_bytes_ssse3_tail(destination, &scale.lo, &scale.hi, &src[offset..]);
                }
            }
        }
    }

    #[target_feature(enable = "ssse3")]
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

        super::xor_scaled_bytes_nibble_tail(&mut dst[offset..], lo, hi, &src[offset..]);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub(super) unsafe fn xor_bytes_neon(dst: &mut [u8], src: &[u8]) {
        let mut offset = 0;
        let len = dst.len();
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len`, and both pointers are derived from
            // valid slices. AArch64 permits unaligned vector loads/stores.
            let d = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
            let s = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
            let mixed = unsafe { veorq_u8(d, s) };
            unsafe { vst1q_u8(dst.as_mut_ptr().add(offset), mixed) };
            offset += 16;
        }

        super::xor_bytes_scalar(&mut dst[offset..], &src[offset..]);
    }

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
        while offset + 16 <= len {
            // SAFETY: `offset + 16 <= len`, and both pointers are derived from
            // valid slices. AArch64 permits unaligned vector loads/stores.
            let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
            let d = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };

            let low_nibbles = unsafe { vandq_u8(x, mask) };
            let shifted = unsafe { vshrq_n_u8(x, 4) };
            let high_nibbles = unsafe { vandq_u8(shifted, mask) };
            let low_prod = unsafe { vqtbl1q_u8(lo_tbl, low_nibbles) };
            let high_prod = unsafe { vqtbl1q_u8(hi_tbl, high_nibbles) };
            let scaled = unsafe { veorq_u8(low_prod, high_prod) };
            let mixed = unsafe { veorq_u8(d, scaled) };

            unsafe { vst1q_u8(dst.as_mut_ptr().add(offset), mixed) };
            offset += 16;
        }

        super::xor_scaled_bytes_nibble_tail(&mut dst[offset..], lo, hi, &src[offset..]);
    }

    #[allow(dead_code)]
    pub(super) unsafe fn xor_scaled_bytes_many_indexed_neon(
        dst: &mut [u8],
        row_stride: usize,
        byte_offset: usize,
        range_len: usize,
        destination_indices: &[usize],
        scales: &[super::ScaleTable],
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
                if scale.coeff != super::GfElem::ZERO && scale.coeff != super::GfElem::ONE {
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
                    if scale.coeff == super::GfElem::ZERO {
                        continue;
                    }
                    let row_offset =
                        destination_indices[destination_start + slot] * row_stride + byte_offset;
                    let destination = unsafe { dst_ptr.add(row_offset + offset) };
                    let d = unsafe { vld1q_u8(destination) };
                    let scaled = if scale.coeff == super::GfElem::ONE {
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
                if scale.coeff == super::GfElem::ZERO {
                    continue;
                }
                let row_offset =
                    destination_indices[destination_start + slot] * row_stride + byte_offset;
                let tail_len = range_len - offset;
                let destination = unsafe {
                    std::slice::from_raw_parts_mut(dst_ptr.add(row_offset + offset), tail_len)
                };
                if scale.coeff == super::GfElem::ONE {
                    super::xor_bytes_scalar(destination, &src[offset..]);
                } else {
                    super::xor_scaled_bytes_nibble_tail(
                        destination,
                        &scale.lo,
                        &scale.hi,
                        &src[offset..],
                    );
                }
            }
        }
    }

    pub(super) unsafe fn xor_scaled_bytes_many_neon(
        destinations: &mut [Vec<u8>],
        scales: &[crate::encoder::EncoderScaleTable],
        src: &[u8],
    ) {
        let mask = unsafe { vdupq_n_u8(0x0f) };
        let zero = unsafe { vdupq_n_u8(0) };

        for destination_start in (0..destinations.len()).step_by(4) {
            let destination_end = (destination_start + 4).min(destinations.len());
            let output_count = destination_end - destination_start;
            let mut low_tables = [zero; 4];
            let mut high_tables = [zero; 4];
            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                low_tables[slot] = unsafe { vld1q_u8(scale.lo.as_ptr()) };
                high_tables[slot] = unsafe { vld1q_u8(scale.hi.as_ptr()) };
            }

            let mut offset = 0;
            while offset + 16 <= src.len() {
                let x = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
                let low_nibbles = unsafe { vandq_u8(x, mask) };
                let high_nibbles = unsafe { vandq_u8(vshrq_n_u8(x, 4), mask) };

                for slot in 0..output_count {
                    let destination = &mut destinations[destination_start + slot];
                    let d = unsafe { vld1q_u8(destination.as_ptr().add(offset)) };
                    let low_product = unsafe { vqtbl1q_u8(low_tables[slot], low_nibbles) };
                    let high_product = unsafe { vqtbl1q_u8(high_tables[slot], high_nibbles) };
                    let mixed = unsafe { veorq_u8(d, veorq_u8(low_product, high_product)) };
                    unsafe { vst1q_u8(destination.as_mut_ptr().add(offset), mixed) };
                }
                offset += 16;
            }

            for slot in 0..output_count {
                let scale = &scales[destination_start + slot];
                super::xor_scaled_bytes_nibble_tail(
                    &mut destinations[destination_start + slot][offset..],
                    &scale.lo,
                    &scale.hi,
                    &src[offset..],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table(coeff: GfElem) -> ([u8; 256], [u8; 16], [u8; 16]) {
        let mut table = [0u8; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = GfElem(i as u8).mul(coeff).0;
        }
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for i in 0..16 {
            lo[i] = table[i];
            hi[i] = table[i << 4];
        }
        (table, lo, hi)
    }

    #[test]
    fn xor_scaled_many_matches_individual_updates() {
        for output_count in [1, 3, 4, 5, 16] {
            for symbol_len in [1, 31, 32, 33, 1400] {
                let src: Vec<u8> = (0..symbol_len)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                    .collect();
                let scales: Vec<_> = (0..output_count)
                    .map(|i| {
                        crate::encoder::EncoderScaleTable::new(GfElem(
                            (i as u8).wrapping_mul(29).wrapping_add(1),
                        ))
                    })
                    .collect();
                let mut expected = vec![vec![0xa5; symbol_len]; output_count];
                for (dst, scale) in expected.iter_mut().zip(&scales) {
                    scale.xor_scaled(dst, &src);
                }
                let mut actual = vec![vec![0xa5; symbol_len]; output_count];
                xor_scaled_bytes_many(&mut actual, &scales, &src);
                assert_eq!(actual, expected, "outputs={output_count}, len={symbol_len}");
            }
        }
    }

    #[test]
    fn xor_scaled_many_indexed_matches_individual_updates() {
        for output_count in [0, 1, 2, 4, 5, 16] {
            for symbol_len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                for coefficient_phase in 0..3 {
                    let row_count = output_count * 2 + 3;
                    let destination_indices: Vec<_> =
                        (0..output_count).map(|i| i * 2 + 1).collect();
                    let scales: Vec<_> = (0..output_count)
                        .map(|i| match (i + coefficient_phase) % 3 {
                            0 => ScaleTable::new(GfElem::ZERO),
                            1 => ScaleTable::new(GfElem::ONE),
                            _ => ScaleTable::new(GfElem(0x53u8.wrapping_add(i as u8))),
                        })
                        .collect();
                    let src: Vec<_> = (0..symbol_len)
                        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                        .collect();
                    let initial: Vec<_> = (0..row_count * symbol_len)
                        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                        .collect();
                    let mut expected = initial.clone();
                    for (&index, scale) in destination_indices.iter().zip(&scales) {
                        let start = index * symbol_len;
                        xor_scaled_bytes(&mut expected[start..start + symbol_len], scale, &src);
                    }

                    let mut actual = initial;
                    xor_scaled_bytes_many_indexed(
                        &mut actual,
                        symbol_len,
                        &destination_indices,
                        &scales,
                        &src,
                    );
                    assert_eq!(
                        actual, expected,
                        "outputs={output_count}, len={symbol_len}, phase={coefficient_phase}"
                    );
                }
            }
        }
    }

    #[test]
    fn xor_scaled_many_indexed_ranges_match_full_updates() {
        for output_count in [0, 1, 2, 4, 5, 16] {
            for symbol_len in [0, 1, 7, 15, 16, 17, 31, 32, 33, 65, 140] {
                for coefficient_phase in 0..3 {
                    let row_count = output_count * 2 + 3;
                    let destination_indices: Vec<_> =
                        (0..output_count).map(|i| i * 2 + 1).collect();
                    let scales: Vec<_> = (0..output_count)
                        .map(|i| match (i + coefficient_phase) % 3 {
                            0 => ScaleTable::new(GfElem::ZERO),
                            1 => ScaleTable::new(GfElem::ONE),
                            _ => ScaleTable::new(GfElem(0x53u8.wrapping_add(i as u8))),
                        })
                        .collect();
                    let src: Vec<_> = (0..symbol_len)
                        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                        .collect();
                    let initial: Vec<_> = (0..row_count * symbol_len)
                        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                        .collect();
                    let mut expected = initial.clone();
                    for (&index, scale) in destination_indices.iter().zip(&scales) {
                        let start = index * symbol_len;
                        xor_scaled_bytes(&mut expected[start..start + symbol_len], scale, &src);
                    }

                    for chunk_ends in [
                        vec![0, symbol_len],
                        vec![
                            0,
                            1.min(symbol_len),
                            6.min(symbol_len),
                            23.min(symbol_len),
                            55.min(symbol_len),
                            symbol_len,
                        ],
                    ] {
                        let mut actual = initial.clone();
                        let mut rows = IndexedDestinationRows::new(
                            &mut actual,
                            symbol_len,
                            &destination_indices,
                        );
                        let mut byte_offset = 0;
                        for chunk_end in chunk_ends {
                            rows.xor_scaled_range(
                                &scales,
                                &src[byte_offset..chunk_end],
                                byte_offset,
                            );
                            byte_offset = chunk_end;
                        }
                        assert_eq!(
                            actual, expected,
                            "outputs={output_count}, len={symbol_len}, phase={coefficient_phase}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "source range out of bounds")]
    fn xor_scaled_many_indexed_rejects_out_of_bounds_range() {
        let mut dst = [0u8; 8];
        let indices = [0];
        let scales = [ScaleTable::new(GfElem::ONE)];
        IndexedDestinationRows::new(&mut dst, 4, &indices).xor_scaled_range(&scales, &[0; 2], 3);
    }

    #[test]
    #[should_panic(expected = "destination/scale count mismatch")]
    fn xor_scaled_many_indexed_range_rejects_scale_count_mismatch() {
        let mut dst = [0u8; 8];
        let indices = [0];
        IndexedDestinationRows::new(&mut dst, 4, &indices).xor_scaled_range(&[], &[0; 1], 0);
    }

    #[test]
    #[should_panic(expected = "destination indices must be unique and strictly increasing")]
    fn xor_scaled_many_indexed_rejects_duplicate_indices() {
        let mut dst = [0u8; 8];
        let scales = [ScaleTable::new(GfElem::ONE), ScaleTable::new(GfElem::ONE)];
        xor_scaled_bytes_many_indexed(&mut dst, 4, &[1, 1], &scales, &[0; 4]);
    }

    #[test]
    #[should_panic(expected = "destination row out of bounds")]
    fn xor_scaled_many_indexed_rejects_out_of_bounds_indices() {
        let mut dst = [0u8; 8];
        let scales = [ScaleTable::new(GfElem::ONE)];
        xor_scaled_bytes_many_indexed(&mut dst, 4, &[2], &scales, &[0; 4]);
    }

    #[test]
    fn xor_scaled_matches_reference() {
        for coeff in [GfElem(1), GfElem(2), GfElem(0x53), GfElem(0xff)] {
            let (table, lo, hi) = make_table(coeff);
            for len in [0, 1, 7, 16, 31, 1400, 4099] {
                let src: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37)).collect();
                let mut expected: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(13)).collect();
                let mut actual = expected.clone();

                for (d, &s) in expected.iter_mut().zip(&src) {
                    *d ^= table[s as usize];
                }
                let scale = ScaleTable { coeff, lo, hi };
                xor_scaled_bytes(&mut actual, &scale, &src);
                assert_eq!(actual, expected, "coeff={coeff:?}, len={len}");
            }
        }
    }

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
                x86::xor_scaled_bytes_gfni(&mut actual, &scale, &src);
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
    fn forced_gfni_axpy_handles_vector_boundaries_and_existing_data() {
        if !gfni_available() {
            return;
        }

        for coefficient in [0, 1, 2, 0x53, 0xff] {
            let scale = ScaleTable::new(GfElem(coefficient));
            for len in [0, 1, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127] {
                let src: Vec<_> = (0..len)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                    .collect();
                let mut expected: Vec<_> = (0..len)
                    .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                    .collect();
                let mut actual = expected.clone();
                for (destination, &source) in expected.iter_mut().zip(&src) {
                    *destination ^= GfElem(source).mul_xtime(GfElem(coefficient)).0;
                }

                assert!(xor_scaled_bytes_gfni(&mut actual, &scale, &src));
                assert_eq!(
                    actual, expected,
                    "coefficient={coefficient:#04x}, len={len}"
                );
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
                    x86::xor_scaled_bytes_many_indexed_gfni(
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
