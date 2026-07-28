//! Fixed-coefficient payload operations over interleaved GF(65536) elements.
#![allow(unsafe_code)]
#![cfg_attr(feature = "internals", allow(missing_docs))]

use fff::gf16::Elem as GfElem;

#[derive(Clone, Copy, Debug)]
pub enum ButterflyBackendKind {
    Scalar,
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Gfni,
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Avx2,
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Ssse3,
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    Neon,
}

pub trait ButterflyBackend {
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem);
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem);
}

pub struct ScalarBackend;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub struct GfniBackend;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub struct Avx2Backend;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub struct Ssse3Backend;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub struct NeonBackend;

static BUTTERFLY_BACKEND: std::sync::LazyLock<ButterflyBackendKind> =
    std::sync::LazyLock::new(select_butterfly_backend);

#[inline]
pub fn butterfly_backend() -> ButterflyBackendKind {
    *BUTTERFLY_BACKEND
}

/// XOR `coefficient * src` into `dst` element by element.
pub fn xor_scaled_bytes(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(src.len() % 2, 0);
    fff::ops::mul_add::<fff::Gf16>(dst, coefficient, src);
}

/// Fused forward butterfly over two equal-length interleaved halves.
#[inline]
pub fn fused_forward<B: ButterflyBackend>(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % 2, 0);
    if coefficient == GfElem::ZERO {
        xor_coupling(high, low);
    } else {
        B::forward_nonzero(low, high, coefficient);
    }
}

/// Fused inverse butterfly over two equal-length interleaved halves.
#[inline]
pub fn fused_inverse<B: ButterflyBackend>(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % 2, 0);
    if coefficient == GfElem::ZERO {
        xor_coupling(high, low);
    } else {
        B::inverse_nonzero(low, high, coefficient);
    }
}

fn select_butterfly_backend() -> ButterflyBackendKind {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("gfni")
        {
            return ButterflyBackendKind::Gfni;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return ButterflyBackendKind::Avx2;
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return ButterflyBackendKind::Ssse3;
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return ButterflyBackendKind::Neon;
    }
    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    ButterflyBackendKind::Scalar
}

impl ButterflyBackend for ScalarBackend {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        fused_forward_scalar(low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        fused_inverse_scalar(low, high, coefficient);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl ButterflyBackend for GfniBackend {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting AVX2 and GFNI.
        unsafe { x86::fused_forward_gfni(low, high, coefficient) };
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting AVX2 and GFNI.
        unsafe { x86::fused_inverse_gfni(low, high, coefficient) };
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl ButterflyBackend for Avx2Backend {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting AVX2.
        unsafe { x86::fused_forward_avx2(low, high, coefficient) };
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting AVX2.
        unsafe { x86::fused_inverse_avx2(low, high, coefficient) };
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl ButterflyBackend for Ssse3Backend {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting SSSE3.
        unsafe { x86::fused_forward_ssse3(low, high, coefficient) };
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: this backend is selected only after detecting SSSE3.
        unsafe { x86::fused_inverse_ssse3(low, high, coefficient) };
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
impl ButterflyBackend for NeonBackend {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { aarch64::fused_forward_neon(low, high, coefficient) };
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { aarch64::fused_inverse_neon(low, high, coefficient) };
    }
}

#[cfg(test)]
fn fused_forward_bytes(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    match butterfly_backend() {
        ButterflyBackendKind::Scalar => fused_forward::<ScalarBackend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Gfni => fused_forward::<GfniBackend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Avx2 => fused_forward::<Avx2Backend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Ssse3 => fused_forward::<Ssse3Backend>(low, high, coefficient),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        ButterflyBackendKind::Neon => fused_forward::<NeonBackend>(low, high, coefficient),
    }
}

#[cfg(test)]
fn fused_inverse_bytes(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    match butterfly_backend() {
        ButterflyBackendKind::Scalar => fused_inverse::<ScalarBackend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Gfni => fused_inverse::<GfniBackend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Avx2 => fused_inverse::<Avx2Backend>(low, high, coefficient),
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        ButterflyBackendKind::Ssse3 => fused_inverse::<Ssse3Backend>(low, high, coefficient),
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        ButterflyBackendKind::Neon => fused_inverse::<NeonBackend>(low, high, coefficient),
    }
}

/// `high[:] <- high[:] ^ low[:]`: the butterfly coupling XOR, used directly when
/// the multiplier is zero so `low` is neither re-read nor rewritten.
fn xor_coupling(high: &mut [u8], low: &[u8]) {
    debug_assert_eq!(high.len(), low.len());
    fff::ops::add_assign::<fff::Gf16>(high, low);
}

/// XOR one scaled source into each flat destination row.
pub fn xor_scaled_bytes_rows(
    destinations: &mut [u8],
    symbol_len: usize,
    coefficients: &[GfElem],
    src: &[u8],
) {
    debug_assert_eq!(src.len(), symbol_len);
    debug_assert_eq!(symbol_len % 2, 0);
    debug_assert_eq!(destinations.len(), coefficients.len() * symbol_len);
    fff::ops::mul_add_scatter::<fff::Gf16>(destinations, symbol_len, coefficients, src);
}


fn fused_forward_scalar(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % 2, 0);
    for (l, h) in low.chunks_exact_mut(2).zip(high.chunks_exact_mut(2)) {
        let lo = GfElem::from_bytes([l[0], l[1]]);
        let hi = GfElem::from_bytes([h[0], h[1]]);
        let new_low = lo.add(coefficient.mul(hi));
        let new_high = hi.add(new_low);
        let low_bytes = new_low.to_bytes();
        let high_bytes = new_high.to_bytes();
        l[0] = low_bytes[0];
        l[1] = low_bytes[1];
        h[0] = high_bytes[0];
        h[1] = high_bytes[1];
    }
}

fn fused_inverse_scalar(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % 2, 0);
    for (l, h) in low.chunks_exact_mut(2).zip(high.chunks_exact_mut(2)) {
        let lo = GfElem::from_bytes([l[0], l[1]]);
        let hi = GfElem::from_bytes([h[0], h[1]]);
        let new_high = hi.add(lo);
        let new_low = lo.add(coefficient.mul(new_high));
        let low_bytes = new_low.to_bytes();
        let high_bytes = new_high.to_bytes();
        l[0] = low_bytes[0];
        l[1] = low_bytes[1];
        h[0] = high_bytes[0];
        h[1] = high_bytes[1];
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    #![allow(clippy::incompatible_msrv)]
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use fff::gf8::Elem as BaseElem;
    use fff::gf16::Elem as GfElem;

    const SWAP_ADJACENT: [u8; 32] = [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    ];

    #[inline]
    fn factor_words(coefficient: GfElem) -> (i16, i16) {
        let (c0, c1) = coefficient.components();
        let delta_c1 = fff::gf16::DELTA.mul(c1);
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
            scale_table(fff::gf16::DELTA.mul(c1)),
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
    pub(super) unsafe fn fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let (same, cross) = factor_words(coefficient);
        let vector_len = low.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()) };
            let h = unsafe { _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()) };
            let scaled = unsafe { scaled_vector(h, same, cross) };
            let new_low = _mm256_xor_si256(l, scaled);
            let new_high = _mm256_xor_si256(h, new_low);
            unsafe {
                _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
                _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
            }
            offset += 32;
        }
        super::fused_forward_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let (same, cross) = factor_words(coefficient);
        let vector_len = low.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()) };
            let h = unsafe { _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()) };
            let new_high = _mm256_xor_si256(h, l);
            let scaled = unsafe { scaled_vector(new_high, same, cross) };
            let new_low = _mm256_xor_si256(l, scaled);
            unsafe {
                _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
                _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
            }
            offset += 32;
        }
        super::fused_inverse_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fused_forward_avx2(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let tables = factor_tables(coefficient);
        let vector_len = low.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()) };
            let h = unsafe { _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()) };
            let scaled = unsafe { scaled_vector_avx2(h, &tables) };
            let new_low = _mm256_xor_si256(l, scaled);
            let new_high = _mm256_xor_si256(h, new_low);
            unsafe {
                _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
                _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
            }
            offset += 32;
        }
        super::fused_forward_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fused_inverse_avx2(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let tables = factor_tables(coefficient);
        let vector_len = low.len() / 32 * 32;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()) };
            let h = unsafe { _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()) };
            let new_high = _mm256_xor_si256(h, l);
            let scaled = unsafe { scaled_vector_avx2(new_high, &tables) };
            let new_low = _mm256_xor_si256(l, scaled);
            unsafe {
                _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
                _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
            }
            offset += 32;
        }
        super::fused_inverse_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn fused_forward_ssse3(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let tables = factor_tables(coefficient);
        let vector_len = low.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()) };
            let h = unsafe { _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()) };
            let scaled = unsafe { scaled_vector_ssse3(h, &tables) };
            let new_low = _mm_xor_si128(l, scaled);
            let new_high = _mm_xor_si128(h, new_low);
            unsafe {
                _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
                _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
            }
            offset += 16;
        }
        super::fused_forward_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn fused_inverse_ssse3(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let tables = factor_tables(coefficient);
        let vector_len = low.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()) };
            let h = unsafe { _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()) };
            let new_high = _mm_xor_si128(h, l);
            let scaled = unsafe { scaled_vector_ssse3(new_high, &tables) };
            let new_low = _mm_xor_si128(l, scaled);
            unsafe {
                _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
                _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
            }
            offset += 16;
        }
        super::fused_inverse_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    // Keep the base-field type import tied to the polynomial used by GFNI.
    const _: BaseElem = fff::gf16::DELTA;
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64 {
    use core::arch::aarch64::*;

    use fff::gf16::Elem as GfElem;

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
        let cross_word = u16::from_le_bytes([fff::gf16::DELTA.mul(c1).0, c1.0]);
        let same = vreinterpretq_u8_u16(vdupq_n_u16(same_word));
        let cross = vreinterpretq_u8_u16(vdupq_n_u16(cross_word));
        let direct = unsafe { multiply_base_vector(source, same) };
        let crossed = unsafe { multiply_base_vector(vrev16q_u8(source), cross) };
        veorq_u8(direct, crossed)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn fused_forward_neon(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let vector_len = low.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { vld1q_u8(low.as_ptr().add(offset)) };
            let h = unsafe { vld1q_u8(high.as_ptr().add(offset)) };
            let scaled = unsafe { scaled_vector(h, coefficient) };
            let new_low = veorq_u8(l, scaled);
            let new_high = veorq_u8(h, new_low);
            unsafe {
                vst1q_u8(low.as_mut_ptr().add(offset), new_low);
                vst1q_u8(high.as_mut_ptr().add(offset), new_high);
            }
            offset += 16;
        }
        super::fused_forward_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn fused_inverse_neon(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let vector_len = low.len() / 16 * 16;
        let mut offset = 0;
        while offset < vector_len {
            let l = unsafe { vld1q_u8(low.as_ptr().add(offset)) };
            let h = unsafe { vld1q_u8(high.as_ptr().add(offset)) };
            let new_high = veorq_u8(h, l);
            let scaled = unsafe { scaled_vector(new_high, coefficient) };
            let new_low = veorq_u8(l, scaled);
            unsafe {
                vst1q_u8(low.as_mut_ptr().add(offset), new_low);
                vst1q_u8(high.as_mut_ptr().add(offset), new_high);
            }
            offset += 16;
        }
        super::fused_inverse_scalar(&mut low[vector_len..], &mut high[vector_len..], coefficient);
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

    /// Element-by-element `dst ^= c * src`, independent of fff and of every kernel
    /// under test.
    fn axpy_reference(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
        for (out, input) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
            let product = GfElem::from_bytes([input[0], input[1]])
                .mul(coefficient)
                .to_bytes();
            out[0] ^= product[0];
            out[1] ^= product[1];
        }
    }

    /// The AXPY seam delegates to fff; assert the shape adaptation, not the kernel.
    /// fff differentially tests its own backends against its portable path.
    #[test]
    fn axpy_seam_matches_element_arithmetic() {
        let src = source(130);
        for coefficient in [GfElem::ZERO, GfElem::ONE, GfElem(0x9b37), GfElem(0xffff)] {
            let mut dst = source(130).into_iter().rev().collect::<Vec<_>>();
            let mut expected = dst.clone();
            axpy_reference(&mut expected, coefficient, &src);
            xor_scaled_bytes(&mut dst, coefficient, &src);
            assert_eq!(dst, expected, "coefficient {coefficient:?}");
        }
    }

    #[test]
    fn row_seam_matches_independent_rows() {
        let src = source(130);
        let coefficients = [GfElem::ZERO, GfElem::ONE, GfElem(0x0108), GfElem(0xbeef)];
        let mut actual = source(130 * coefficients.len());
        let mut expected = actual.clone();
        for (row, &coefficient) in expected.chunks_exact_mut(130).zip(&coefficients) {
            axpy_reference(row, coefficient, &src);
        }
        xor_scaled_bytes_rows(&mut actual, 130, &coefficients, &src);
        assert_eq!(actual, expected);
    }

    /// Two-pass reference for the fused butterfly, written in element arithmetic so
    /// it shares no code with the kernels under test.
    fn reference_forward(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        axpy_reference(low, coefficient, high);
        let coupling: Vec<u8> = low.to_vec();
        axpy_reference(high, GfElem::ONE, &coupling);
    }

    fn reference_inverse(low: &mut [u8], high: &mut [u8], coefficient: GfElem) {
        let coupling: Vec<u8> = low.to_vec();
        axpy_reference(high, GfElem::ONE, &coupling);
        axpy_reference(low, coefficient, high);
    }

    #[test]
    fn scalar_fused_matches_two_pass_reference() {
        for coefficient in [GfElem::ZERO, GfElem::ONE, GfElem(0x0108), GfElem(0xbeef)] {
            let mut low = source(130);
            let mut high = source(130).into_iter().rev().collect::<Vec<_>>();
            let (mut ref_low, mut ref_high) = (low.clone(), high.clone());
            fused_forward_scalar(&mut low, &mut high, coefficient);
            reference_forward(&mut ref_low, &mut ref_high, coefficient);
            assert_eq!((&low, &high), (&ref_low, &ref_high));

            fused_inverse_scalar(&mut low, &mut high, coefficient);
            reference_inverse(&mut ref_low, &mut ref_high, coefficient);
            assert_eq!((&low, &high), (&ref_low, &ref_high));
        }
    }

    #[test]
    fn fused_inverse_undoes_fused_forward() {
        for coefficient in [GfElem::ZERO, GfElem::ONE, GfElem(0x1234), GfElem(0xffff)] {
            let low = source(258);
            let high = source(258).into_iter().rev().collect::<Vec<_>>();
            let (mut work_low, mut work_high) = (low.clone(), high.clone());
            fused_forward_bytes(&mut work_low, &mut work_high, coefficient);
            fused_inverse_bytes(&mut work_low, &mut work_high, coefficient);
            assert_eq!((work_low, work_high), (low, high));
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn forced_fused_kernels_match_scalar_when_available() {
        for coefficient in [GfElem::ZERO, GfElem::ONE, GfElem(0x0108), GfElem(0xffff)] {
            let low = source(194);
            let high = source(194).into_iter().rev().collect::<Vec<_>>();
            let (mut fwd_low, mut fwd_high) = (low.clone(), high.clone());
            let (mut inv_low, mut inv_high) = (low.clone(), high.clone());
            fused_forward_scalar(&mut fwd_low, &mut fwd_high, coefficient);
            fused_inverse_scalar(&mut inv_low, &mut inv_high, coefficient);

            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("gfni")
            {
                let (mut fl, mut fh) = (low.clone(), high.clone());
                let (mut il, mut ih) = (low.clone(), high.clone());
                // SAFETY: both features detected immediately above.
                unsafe {
                    x86::fused_forward_gfni(&mut fl, &mut fh, coefficient);
                    x86::fused_inverse_gfni(&mut il, &mut ih, coefficient);
                }
                assert_eq!((&fl, &fh), (&fwd_low, &fwd_high));
                assert_eq!((&il, &ih), (&inv_low, &inv_high));
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                let (mut fl, mut fh) = (low.clone(), high.clone());
                let (mut il, mut ih) = (low.clone(), high.clone());
                // SAFETY: AVX2 detected immediately above.
                unsafe {
                    x86::fused_forward_avx2(&mut fl, &mut fh, coefficient);
                    x86::fused_inverse_avx2(&mut il, &mut ih, coefficient);
                }
                assert_eq!((&fl, &fh), (&fwd_low, &fwd_high));
                assert_eq!((&il, &ih), (&inv_low, &inv_high));
            }
            if std::arch::is_x86_feature_detected!("ssse3") {
                let (mut fl, mut fh) = (low.clone(), high.clone());
                let (mut il, mut ih) = (low.clone(), high.clone());
                // SAFETY: SSSE3 detected immediately above.
                unsafe {
                    x86::fused_forward_ssse3(&mut fl, &mut fh, coefficient);
                    x86::fused_inverse_ssse3(&mut il, &mut ih, coefficient);
                }
                assert_eq!((&fl, &fh), (&fwd_low, &fwd_high));
                assert_eq!((&il, &ih), (&inv_low, &inv_high));
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn forced_fused_neon_matches_scalar() {
        for coefficient in [GfElem::ZERO, GfElem::ONE, GfElem(0x0108), GfElem(0xffff)] {
            let low = source(194);
            let high = source(194).into_iter().rev().collect::<Vec<_>>();
            let (mut fwd_low, mut fwd_high) = (low.clone(), high.clone());
            let (mut inv_low, mut inv_high) = (low.clone(), high.clone());
            fused_forward_scalar(&mut fwd_low, &mut fwd_high, coefficient);
            fused_inverse_scalar(&mut inv_low, &mut inv_high, coefficient);
            let (mut fl, mut fh) = (low.clone(), high.clone());
            let (mut il, mut ih) = (low.clone(), high.clone());
            // SAFETY: NEON is mandatory on AArch64.
            unsafe {
                aarch64::fused_forward_neon(&mut fl, &mut fh, coefficient);
                aarch64::fused_inverse_neon(&mut il, &mut ih, coefficient);
            }
            assert_eq!((&fl, &fh), (&fwd_low, &fwd_high));
            assert_eq!((&il, &ih), (&inv_low, &inv_high));
        }
    }
}
