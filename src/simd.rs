//! SIMD payload kernels.
//!
#![allow(unsafe_code)]
//!
//! The public surface of this module is safe. Architecture-specific internals
//! use small `unsafe` blocks for Rust SIMD intrinsics and are guarded by runtime
//! feature detection where needed.

use crate::gf256::GfElem;

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
    debug_assert_eq!(dst.len(), src.len(), "xor length mismatch");

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

/// `dst[:] <- dst[:] ^ scale.coeff * src[:]` over GF(256).
pub(crate) fn xor_scaled_bytes(dst: &mut [u8], scale: &ScaleTable, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len(), "scaled byte axpy length mismatch");
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

/// Add one source symbol, with distinct scales, to several destinations.
pub(crate) fn xor_scaled_bytes_many(
    destinations: &mut [Vec<u8>],
    scales: &[crate::encoder::EncoderScaleTable],
    src: &[u8],
) {
    debug_assert_eq!(destinations.len(), scales.len());
    debug_assert!(destinations.iter().all(|dst| dst.len() == src.len()));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
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
}
