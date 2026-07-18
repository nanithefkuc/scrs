//! Fixed-coefficient payload operations over interleaved GF(65536) elements.
#![allow(unsafe_code)]

use crate::gf65536::GfElem;

/// XOR `coefficient * src` into `dst` element by element.
pub(crate) fn xor_scaled_bytes(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(src.len() % 2, 0);

    if coefficient == GfElem::ZERO {
        return;
    }
    if coefficient == GfElem::ONE {
        for (out, &input) in dst.iter_mut().zip(src) {
            *out ^= input;
        }
        return;
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("gfni") {
        // SAFETY: runtime detection established both required target features;
        // the safe wrapper established equal, even-length slices.
        unsafe { x86::xor_scaled_bytes_gfni(dst, coefficient, src) };
        return;
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above and slice invariants were checked.
        unsafe { x86::xor_scaled_bytes_avx2(dst, coefficient, src) };
        return;
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: SSSE3 was detected above and slice invariants were checked.
        unsafe { x86::xor_scaled_bytes_ssse3(dst, coefficient, src) };
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64 and slice invariants were checked.
        unsafe { aarch64::xor_scaled_bytes_neon(dst, coefficient, src) };
    }

    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    xor_scaled_bytes_scalar(dst, coefficient, src);
}

/// XOR one scaled source into each flat destination row.
pub(crate) fn xor_scaled_bytes_rows(
    destinations: &mut [u8],
    symbol_len: usize,
    coefficients: &[GfElem],
    src: &[u8],
) {
    debug_assert_eq!(src.len(), symbol_len);
    debug_assert_eq!(symbol_len % 2, 0);
    debug_assert_eq!(destinations.len(), coefficients.len() * symbol_len);

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("gfni")
        && symbol_len >= 32
    {
        // SAFETY: runtime detection established both required target features;
        // row ranges are disjoint by construction and all slice sizes were
        // checked above.
        unsafe {
            x86::xor_scaled_bytes_rows_gfni(destinations, symbol_len, coefficients, src);
        }
        return;
    }

    for (destination, &coefficient) in destinations.chunks_exact_mut(symbol_len).zip(coefficients) {
        xor_scaled_bytes(destination, coefficient, src);
    }
}

fn xor_scaled_bytes_scalar(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(src.len() % 2, 0);
    if coefficient == GfElem::ZERO {
        return;
    }
    if coefficient == GfElem::ONE {
        for (out, &input) in dst.iter_mut().zip(src) {
            *out ^= input;
        }
        return;
    }
    for (out, input) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let product = GfElem::from_bytes([input[0], input[1]]).mul(coefficient);
        let bytes = product.to_bytes();
        out[0] ^= bytes[0];
        out[1] ^= bytes[1];
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    #![allow(clippy::incompatible_msrv)]
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use crate::{gf256::GfElem as BaseElem, gf65536::GfElem};

    const SWAP_ADJACENT: [u8; 32] = [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    ];

    #[inline]
    fn factor_words(coefficient: GfElem) -> (i16, i16) {
        let (c0, c1) = coefficient.components();
        let delta_c1 = crate::gf65536::DELTA.mul(c1);
        let same = u16::from_le_bytes([c0.0, c0.add(c1).0]) as i16;
        let cross = u16::from_le_bytes([delta_c1.0, c1.0]) as i16;
        (same, cross)
    }

    struct ScaleTable {
        low: [u8; 32],
        high: [u8; 32],
    }

    fn scale_table(coefficient: BaseElem) -> ScaleTable {
        let mut low = [0; 32];
        let mut high = [0; 32];
        for nibble in 0..16 {
            low[nibble] = BaseElem(nibble as u8).mul(coefficient).0;
            high[nibble] = BaseElem((nibble as u8) << 4).mul(coefficient).0;
            low[16 + nibble] = low[nibble];
            high[16 + nibble] = high[nibble];
        }
        ScaleTable { low, high }
    }

    fn factor_tables(coefficient: GfElem) -> [ScaleTable; 4] {
        let (c0, c1) = coefficient.components();
        [
            scale_table(c0),
            scale_table(c0.add(c1)),
            scale_table(crate::gf65536::DELTA.mul(c1)),
            scale_table(c1),
        ]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn multiply_avx2(value: __m256i, table: &ScaleTable) -> __m256i {
        let low_nibbles = _mm256_and_si256(value, _mm256_set1_epi8(0x0f));
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(value, 4), _mm256_set1_epi8(0x0f));
        let low_table = unsafe { _mm256_loadu_si256(table.low.as_ptr().cast::<__m256i>()) };
        let high_table = unsafe { _mm256_loadu_si256(table.high.as_ptr().cast::<__m256i>()) };
        _mm256_xor_si256(
            _mm256_shuffle_epi8(low_table, low_nibbles),
            _mm256_shuffle_epi8(high_table, high_nibbles),
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scaled_vector_avx2(source: __m256i, tables: &[ScaleTable; 4]) -> __m256i {
        let swap_mask = unsafe { _mm256_loadu_si256(SWAP_ADJACENT.as_ptr().cast::<__m256i>()) };
        let swapped = _mm256_shuffle_epi8(source, swap_mask);
        let even_mask = _mm256_set1_epi16(0x00ff);
        let direct_even = unsafe { multiply_avx2(source, &tables[0]) };
        let direct_odd = unsafe { multiply_avx2(source, &tables[1]) };
        let cross_even = unsafe { multiply_avx2(swapped, &tables[2]) };
        let cross_odd = unsafe { multiply_avx2(swapped, &tables[3]) };
        let direct = _mm256_xor_si256(
            _mm256_and_si256(direct_even, even_mask),
            _mm256_andnot_si256(even_mask, direct_odd),
        );
        let crossed = _mm256_xor_si256(
            _mm256_and_si256(cross_even, even_mask),
            _mm256_andnot_si256(even_mask, cross_odd),
        );
        _mm256_xor_si256(direct, crossed)
    }

    #[target_feature(enable = "ssse3")]
    unsafe fn multiply_ssse3(value: __m128i, table: &ScaleTable) -> __m128i {
        let low_nibbles = _mm_and_si128(value, _mm_set1_epi8(0x0f));
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(value, 4), _mm_set1_epi8(0x0f));
        let low_table = unsafe { _mm_loadu_si128(table.low.as_ptr().cast::<__m128i>()) };
        let high_table = unsafe { _mm_loadu_si128(table.high.as_ptr().cast::<__m128i>()) };
        _mm_xor_si128(
            _mm_shuffle_epi8(low_table, low_nibbles),
            _mm_shuffle_epi8(high_table, high_nibbles),
        )
    }

    #[target_feature(enable = "ssse3")]
    unsafe fn scaled_vector_ssse3(source: __m128i, tables: &[ScaleTable; 4]) -> __m128i {
        let swap_mask = unsafe { _mm_loadu_si128(SWAP_ADJACENT.as_ptr().cast::<__m128i>()) };
        let swapped = _mm_shuffle_epi8(source, swap_mask);
        let even_mask = _mm_set1_epi16(0x00ff);
        let direct_even = unsafe { multiply_ssse3(source, &tables[0]) };
        let direct_odd = unsafe { multiply_ssse3(source, &tables[1]) };
        let cross_even = unsafe { multiply_ssse3(swapped, &tables[2]) };
        let cross_odd = unsafe { multiply_ssse3(swapped, &tables[3]) };
        let direct = _mm_xor_si128(
            _mm_and_si128(direct_even, even_mask),
            _mm_andnot_si128(even_mask, direct_odd),
        );
        let crossed = _mm_xor_si128(
            _mm_and_si128(cross_even, even_mask),
            _mm_andnot_si128(even_mask, cross_odd),
        );
        _mm_xor_si128(direct, crossed)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn xor_scaled_bytes_avx2(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
        let tables = factor_tables(coefficient);
        let vector_len = src.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let source = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            let destination =
                unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()) };
            let scaled = unsafe { scaled_vector_avx2(source, &tables) };
            unsafe {
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(destination, scaled),
                );
            }
            offset += 32;
        }
        super::xor_scaled_bytes_scalar(&mut dst[vector_len..], coefficient, &src[vector_len..]);
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn xor_scaled_bytes_ssse3(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
        let tables = factor_tables(coefficient);
        let vector_len = src.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let source = unsafe { _mm_loadu_si128(src.as_ptr().add(offset).cast::<__m128i>()) };
            let destination =
                unsafe { _mm_loadu_si128(dst.as_ptr().add(offset).cast::<__m128i>()) };
            let scaled = unsafe { scaled_vector_ssse3(source, &tables) };
            unsafe {
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(offset).cast::<__m128i>(),
                    _mm_xor_si128(destination, scaled),
                );
            }
            offset += 16;
        }
        super::xor_scaled_bytes_scalar(&mut dst[vector_len..], coefficient, &src[vector_len..]);
    }

    #[target_feature(enable = "avx2,gfni")]
    unsafe fn scaled_vector(source: __m256i, same: i16, cross: i16) -> __m256i {
        // Interleaved source bytes are [a,b]. Multiplication by c+d*u is:
        // [c*a + DELTA*d*b, d*a + (c+d)*b]. Multiplying the original and
        // adjacent-byte-swapped vectors by alternating GF(256) coefficients
        // computes both components without planar conversion.
        let swap_mask = unsafe { _mm256_loadu_si256(SWAP_ADJACENT.as_ptr().cast::<__m256i>()) };
        let swapped = _mm256_shuffle_epi8(source, swap_mask);
        let direct = _mm256_gf2p8mul_epi8(source, _mm256_set1_epi16(same));
        let crossed = _mm256_gf2p8mul_epi8(swapped, _mm256_set1_epi16(cross));
        _mm256_xor_si256(direct, crossed)
    }

    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn xor_scaled_bytes_gfni(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
        let (same, cross) = factor_words(coefficient);
        let vector_len = src.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let source = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            let destination =
                unsafe { _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()) };
            let scaled = unsafe { scaled_vector(source, same, cross) };
            unsafe {
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(offset).cast::<__m256i>(),
                    _mm256_xor_si256(destination, scaled),
                );
            }
            offset += 32;
        }
        super::xor_scaled_bytes_scalar(&mut dst[vector_len..], coefficient, &src[vector_len..]);
    }

    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn xor_scaled_bytes_rows_gfni(
        destinations: &mut [u8],
        symbol_len: usize,
        coefficients: &[GfElem],
        src: &[u8],
    ) {
        let vector_len = symbol_len / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let source = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()) };
            for (row, &coefficient) in coefficients.iter().enumerate() {
                if coefficient == GfElem::ZERO {
                    continue;
                }
                let destination_ptr =
                    unsafe { destinations.as_mut_ptr().add(row * symbol_len + offset) };
                let destination = unsafe { _mm256_loadu_si256(destination_ptr.cast::<__m256i>()) };
                let scaled = if coefficient == GfElem::ONE {
                    source
                } else {
                    let (same, cross) = factor_words(coefficient);
                    unsafe { scaled_vector(source, same, cross) }
                };
                unsafe {
                    _mm256_storeu_si256(
                        destination_ptr.cast::<__m256i>(),
                        _mm256_xor_si256(destination, scaled),
                    );
                }
            }
            offset += 32;
        }

        if vector_len != symbol_len {
            for (destination, &coefficient) in
                destinations.chunks_exact_mut(symbol_len).zip(coefficients)
            {
                super::xor_scaled_bytes_scalar(
                    &mut destination[vector_len..],
                    coefficient,
                    &src[vector_len..],
                );
            }
        }
    }

    // Keep the base-field type import tied to the polynomial used by GFNI.
    const _: BaseElem = crate::gf65536::DELTA;
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64 {
    use core::arch::aarch64::*;

    use crate::gf65536::GfElem;

    #[target_feature(enable = "neon")]
    unsafe fn multiply_base_vector(mut value: uint8x16_t, mut factor: uint8x16_t) -> uint8x16_t {
        let mut product = vdupq_n_u8(0);
        let one = vdupq_n_u8(1);
        let high_threshold = vdupq_n_u8(0x7f);
        let reduction = vdupq_n_u8(0x1b);
        for _ in 0..8 {
            let active = vceqq_u8(vandq_u8(factor, one), one);
            product = veorq_u8(product, vandq_u8(value, active));
            let high = vcgtq_u8(value, high_threshold);
            value = veorq_u8(vshlq_n_u8(value, 1), vandq_u8(high, reduction));
            factor = vshrq_n_u8(factor, 1);
        }
        product
    }

    #[target_feature(enable = "neon")]
    unsafe fn scaled_vector(source: uint8x16_t, coefficient: GfElem) -> uint8x16_t {
        let (c0, c1) = coefficient.components();
        let same_word = u16::from_le_bytes([c0.0, c0.add(c1).0]);
        let cross_word = u16::from_le_bytes([crate::gf65536::DELTA.mul(c1).0, c1.0]);
        let same = vreinterpretq_u8_u16(vdupq_n_u16(same_word));
        let cross = vreinterpretq_u8_u16(vdupq_n_u16(cross_word));
        let direct = unsafe { multiply_base_vector(source, same) };
        let crossed = unsafe { multiply_base_vector(vrev16q_u8(source), cross) };
        veorq_u8(direct, crossed)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn xor_scaled_bytes_neon(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
        let vector_len = src.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let source = unsafe { vld1q_u8(src.as_ptr().add(offset)) };
            let destination = unsafe { vld1q_u8(dst.as_ptr().add(offset)) };
            let scaled = unsafe { scaled_vector(source, coefficient) };
            unsafe {
                vst1q_u8(dst.as_mut_ptr().add(offset), veorq_u8(destination, scaled));
            }
            offset += 16;
        }
        super::xor_scaled_bytes_scalar(&mut dst[vector_len..], coefficient, &src[vector_len..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(29) ^ 0xa5) as u8)
            .collect()
    }

    #[test]
    fn scalar_axpy_matches_element_arithmetic() {
        let src = source(130);
        let coefficient = GfElem(0x9b37);
        let mut dst = source(130).into_iter().rev().collect::<Vec<_>>();
        let mut expected = dst.clone();
        for (out, input) in expected.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let product = GfElem::from_bytes([input[0], input[1]])
                .mul(coefficient)
                .to_bytes();
            out[0] ^= product[0];
            out[1] ^= product[1];
        }
        xor_scaled_bytes_scalar(&mut dst, coefficient, &src);
        assert_eq!(dst, expected);
    }

    #[test]
    fn row_kernel_matches_independent_scalar_rows() {
        let src = source(130);
        let coefficients = [GfElem::ZERO, GfElem::ONE, GfElem(0x0108), GfElem(0xbeef)];
        let mut actual = source(130 * coefficients.len());
        let mut expected = actual.clone();
        for (row, &coefficient) in expected.chunks_exact_mut(130).zip(&coefficients) {
            xor_scaled_bytes_scalar(row, coefficient, &src);
        }
        xor_scaled_bytes_rows(&mut actual, 130, &coefficients, &src);
        assert_eq!(actual, expected);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn forced_gfni_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("gfni")
        {
            return;
        }
        let src = source(194);
        for coefficient in [GfElem::ONE, GfElem(0x0108), GfElem(0x1234), GfElem(0xffff)] {
            let mut expected = source(194);
            let mut actual = expected.clone();
            xor_scaled_bytes_scalar(&mut expected, coefficient, &src);
            // SAFETY: target features were detected immediately above.
            unsafe { x86::xor_scaled_bytes_gfni(&mut actual, coefficient, &src) };
            assert_eq!(actual, expected);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn forced_nibble_fallbacks_match_scalar_when_available() {
        let src = source(194);
        for coefficient in [GfElem::ONE, GfElem(0x0108), GfElem(0x1234), GfElem(0xffff)] {
            let mut expected = source(194);
            xor_scaled_bytes_scalar(&mut expected, coefficient, &src);
            if std::arch::is_x86_feature_detected!("avx2") {
                let mut actual = source(194);
                // SAFETY: AVX2 was detected immediately above.
                unsafe { x86::xor_scaled_bytes_avx2(&mut actual, coefficient, &src) };
                assert_eq!(actual, expected);
            }
            if std::arch::is_x86_feature_detected!("ssse3") {
                let mut actual = source(194);
                // SAFETY: SSSE3 was detected immediately above.
                unsafe { x86::xor_scaled_bytes_ssse3(&mut actual, coefficient, &src) };
                assert_eq!(actual, expected);
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn forced_neon_matches_scalar() {
        let src = source(194);
        for coefficient in [GfElem::ONE, GfElem(0x0108), GfElem(0x1234), GfElem(0xffff)] {
            let mut expected = source(194);
            let mut actual = source(194);
            xor_scaled_bytes_scalar(&mut expected, coefficient, &src);
            // SAFETY: NEON is mandatory on AArch64.
            unsafe { aarch64::xor_scaled_bytes_neon(&mut actual, coefficient, &src) };
            assert_eq!(actual, expected);
        }
    }
}
