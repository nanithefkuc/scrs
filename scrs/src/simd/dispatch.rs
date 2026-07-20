use crate::gf256::GfElem;

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::x86;
use super::{
    scalar,
    scale_table::{ScaleTable, scale_table},
};

/// Selected SIMD backend for payload kernels.
///
/// Detected once per process via [`kernel_plan`] so hot AXPY loops do not repeat
/// runtime feature checks on every coefficient term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Variants are architecture-selected at runtime.
pub(crate) enum KernelPlan {
    /// AVX2 + GFNI direct field multiply.
    Gfni,
    /// AVX2 nibble shuffle.
    Avx2Nibble,
    /// SSSE3 nibble shuffle.
    Ssse3Nibble,
    /// AArch64 NEON nibble shuffle.
    Neon,
    /// Portable scalar path.
    Scalar,
}

impl KernelPlan {
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni") {
                return Self::Gfni;
            }
            if std::is_x86_feature_detected!("avx2") {
                return Self::Avx2Nibble;
            }
            if std::is_x86_feature_detected!("ssse3") {
                return Self::Ssse3Nibble;
            }
            Self::Scalar
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Scalar
        }
    }

    /// Whether this plan is the AVX2+GFNI backend.
    #[inline]
    pub(crate) const fn is_gfni(self) -> bool {
        matches!(self, Self::Gfni)
    }

    /// Whether this plan supports four-output source-major reconstruction.
    #[inline]
    pub(crate) const fn supports_grouped_source_major(self) -> bool {
        matches!(self, Self::Gfni | Self::Neon)
    }
}

static KERNEL_PLAN: std::sync::LazyLock<KernelPlan> = std::sync::LazyLock::new(KernelPlan::detect);

/// Process-wide SIMD backend selected once at first use.
#[inline]
pub(crate) fn kernel_plan() -> KernelPlan {
    *KERNEL_PLAN
}

/// Whether the AVX2 GFNI kernels can run on this CPU.
#[inline]
pub(crate) fn gfni_available() -> bool {
    kernel_plan().is_gfni()
}
/// `dst[:] <- dst[:] ^ src[:]`.
pub(crate) fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    xor_bytes_with_plan(kernel_plan(), dst, src);
}

/// `dst[:] <- dst[:] ^ src[:]` using an already-resolved backend plan.
pub(crate) fn xor_bytes_with_plan(plan: KernelPlan, dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "xor length mismatch");

    match plan {
        KernelPlan::Gfni | KernelPlan::Avx2Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_bytes_avx2(dst, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            scalar::xor_bytes_scalar(dst, src);
        }
        KernelPlan::Ssse3Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_bytes_sse2(dst, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            scalar::xor_bytes_scalar(dst, src);
        }
        KernelPlan::Neon => {
            #[cfg(target_arch = "aarch64")]
            aarch64::dispatch_xor_bytes_neon(dst, src);

            #[cfg(not(target_arch = "aarch64"))]
            scalar::xor_bytes_scalar(dst, src);
        }
        KernelPlan::Scalar => scalar::xor_bytes_scalar(dst, src),
    }
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
        x86::dispatch_xor_scaled_bytes_gfni(dst, scale.coeff, src);

        true
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// `dst[:] <- dst[:] ^ scale.coeff * src[:]` over GF(256).
pub(crate) fn xor_scaled_bytes(dst: &mut [u8], scale: &ScaleTable, src: &[u8]) {
    xor_scaled_bytes_with_plan(kernel_plan(), dst, scale, src);
}

/// `xor_scaled_bytes` using an already-resolved backend plan.
pub(crate) fn xor_scaled_bytes_with_plan(
    plan: KernelPlan,
    dst: &mut [u8],
    scale: &ScaleTable,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len(), "scaled byte axpy length mismatch");
    if scale.coeff == GfElem::ZERO {
        return;
    }
    if scale.coeff == GfElem::ONE {
        xor_bytes_with_plan(plan, dst, src);
        return;
    }

    let lo = &scale.lo;
    let hi = &scale.hi;

    match plan {
        KernelPlan::Gfni => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_gfni(dst, scale.coeff, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            scalar::xor_scaled_bytes_nibble_tail(dst, lo, hi, src);
        }
        KernelPlan::Avx2Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_avx2(dst, lo, hi, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            scalar::xor_scaled_bytes_nibble_tail(dst, lo, hi, src);
        }
        KernelPlan::Ssse3Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_ssse3(dst, lo, hi, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            scalar::xor_scaled_bytes_nibble_tail(dst, lo, hi, src);
        }
        KernelPlan::Neon => {
            #[cfg(target_arch = "aarch64")]
            aarch64::dispatch_xor_scaled_bytes_neon(dst, lo, hi, src);

            #[cfg(not(target_arch = "aarch64"))]
            scalar::xor_scaled_bytes_nibble_tail(dst, lo, hi, src);
        }
        KernelPlan::Scalar => scalar::xor_scaled_bytes_nibble_tail(dst, lo, hi, src),
    }
}

/// `dst[:] <- dst[:] ^ coeff * src[:]` using compact coefficient storage.
#[inline]
pub(crate) fn xor_scaled_bytes_coeff(dst: &mut [u8], coeff: GfElem, src: &[u8]) {
    xor_scaled_bytes_coeff_with_plan(kernel_plan(), dst, coeff, src);
}

/// `xor_scaled_bytes_coeff` using an already-resolved backend plan.
///
/// Callers on a hot reconstruction loop resolve [`kernel_plan`] once and pass it
/// here so each coefficient term skips the process-wide plan load.
#[inline]
pub(crate) fn xor_scaled_bytes_coeff_with_plan(
    plan: KernelPlan,
    dst: &mut [u8],
    coeff: GfElem,
    src: &[u8],
) {
    assert_eq!(dst.len(), src.len(), "scaled byte axpy length mismatch");
    if coeff == GfElem::ZERO {
        return;
    }
    if coeff == GfElem::ONE {
        xor_bytes_with_plan(plan, dst, src);
        return;
    }

    if plan.is_gfni() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            x86::dispatch_xor_scaled_bytes_gfni(dst, coeff, src);
            return;
        }
    }

    xor_scaled_bytes_with_plan(plan, dst, scale_table(coeff), src);
}
pub(super) fn xor_scaled_bytes_many_indexed_trusted(
    dst: &mut [u8],
    row_stride: usize,
    byte_offset: usize,
    range_len: usize,
    destination_indices: &[usize],
    scales: &[ScaleTable],
    src: &[u8],
) {
    match kernel_plan() {
        KernelPlan::Gfni => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_many_indexed_gfni(
                dst,
                row_stride,
                byte_offset,
                range_len,
                destination_indices,
                scales,
                src,
            );

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                debug_assert_eq!(range_len, src.len());
                for (&index, scale) in destination_indices.iter().zip(scales) {
                    let start = index * row_stride + byte_offset;
                    xor_scaled_bytes(&mut dst[start..start + range_len], scale, src);
                }
            }
        }
        KernelPlan::Avx2Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_many_indexed_avx2(
                dst,
                row_stride,
                byte_offset,
                range_len,
                destination_indices,
                scales,
                src,
            );

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                debug_assert_eq!(range_len, src.len());
                for (&index, scale) in destination_indices.iter().zip(scales) {
                    let start = index * row_stride + byte_offset;
                    xor_scaled_bytes(&mut dst[start..start + range_len], scale, src);
                }
            }
        }
        KernelPlan::Neon => {
            #[cfg(target_arch = "aarch64")]
            aarch64::dispatch_xor_scaled_bytes_many_indexed_neon(
                dst,
                row_stride,
                byte_offset,
                range_len,
                destination_indices,
                scales,
                src,
            );

            #[cfg(not(target_arch = "aarch64"))]
            {
                debug_assert_eq!(range_len, src.len());
                for (&index, scale) in destination_indices.iter().zip(scales) {
                    let start = index * row_stride + byte_offset;
                    xor_scaled_bytes(&mut dst[start..start + range_len], scale, src);
                }
            }
        }
        KernelPlan::Ssse3Nibble | KernelPlan::Scalar => {
            debug_assert_eq!(range_len, src.len());
            for (&index, scale) in destination_indices.iter().zip(scales) {
                let start = index * row_stride + byte_offset;
                xor_scaled_bytes(&mut dst[start..start + range_len], scale, src);
            }
        }
    }
}
/// Add one source symbol, with distinct coefficients, to several destinations.
///
/// Prefer [`xor_scaled_bytes_rows`] for the streaming encoder (flat repair
/// storage). This `Vec`-of-`Vec` entry point remains for callers/tests that
/// already own separate buffers.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn xor_scaled_bytes_many(destinations: &mut [Vec<u8>], coeffs: &[GfElem], src: &[u8]) {
    assert_eq!(destinations.len(), coeffs.len());
    assert!(destinations.iter().all(|dst| dst.len() == src.len()));

    // Fast path when every destination has the same length: still per-Vec, but
    // route GFNI through the hoisted 4-wide kernel via temporary pointer groups.
    match kernel_plan() {
        KernelPlan::Gfni => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_many_gfni(destinations, coeffs, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            for (dst, &coeff) in destinations.iter_mut().zip(coeffs) {
                xor_scaled_bytes(dst, scale_table(coeff), src);
            }
        }
        KernelPlan::Avx2Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_many_avx2(destinations, coeffs, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            for (dst, &coeff) in destinations.iter_mut().zip(coeffs) {
                xor_scaled_bytes(dst, scale_table(coeff), src);
            }
        }
        KernelPlan::Neon => {
            #[cfg(target_arch = "aarch64")]
            aarch64::dispatch_xor_scaled_bytes_many_neon(destinations, coeffs, src);

            #[cfg(not(target_arch = "aarch64"))]
            for (dst, &coeff) in destinations.iter_mut().zip(coeffs) {
                xor_scaled_bytes(dst, scale_table(coeff), src);
            }
        }
        KernelPlan::Ssse3Nibble | KernelPlan::Scalar => {
            for (dst, &coeff) in destinations.iter_mut().zip(coeffs) {
                xor_scaled_bytes(dst, scale_table(coeff), src);
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
                let coeffs: Vec<_> = (0..output_count)
                    .map(|i| GfElem((i as u8).wrapping_mul(29).wrapping_add(1)))
                    .collect();
                let mut expected = vec![vec![0xa5; symbol_len]; output_count];
                for (dst, &coeff) in expected.iter_mut().zip(&coeffs) {
                    xor_scaled_bytes_coeff(dst, coeff, &src);
                }
                let mut actual = vec![vec![0xa5; symbol_len]; output_count];
                xor_scaled_bytes_many(&mut actual, &coeffs, &src);
                assert_eq!(actual, expected, "outputs={output_count}, len={symbol_len}");
            }
        }
    }
    #[test]
    fn compact_coefficient_axpy_matches_reference() {
        for coefficient in [0, 1, 2, 0x53, 0xff] {
            for len in [0, 1, 15, 16, 17, 31, 32, 33, 65, 1400] {
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
                xor_scaled_bytes_coeff(&mut actual, GfElem(coefficient), &src);
                assert_eq!(actual, expected, "coefficient={coefficient}, len={len}");
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
    fn selected_simd_matches_scalar_byte_paths_at_boundaries() {
        for len in [0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 1400] {
            let src: Vec<_> = (0..len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let initial: Vec<_> = (0..len)
                .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                .collect();
            let mut scalar_xor = initial.clone();
            let mut selected_xor = initial.clone();
            scalar::xor_bytes_scalar(&mut scalar_xor, &src);
            xor_bytes(&mut selected_xor, &src);
            assert_eq!(selected_xor, scalar_xor, "xor len={len}");
            for coeff in [
                GfElem::ZERO,
                GfElem::ONE,
                GfElem(2),
                GfElem(0x53),
                GfElem(0xff),
            ] {
                let scale = ScaleTable::new(coeff);
                let mut scalar_scaled = initial.clone();
                let mut selected_scaled = initial.clone();
                scalar::xor_scaled_bytes_nibble_tail(
                    &mut scalar_scaled,
                    &scale.lo,
                    &scale.hi,
                    &src,
                );
                xor_scaled_bytes(&mut selected_scaled, &scale, &src);
                assert_eq!(selected_scaled, scalar_scaled, "coeff={coeff:?}, len={len}");
            }
        }
    }
}
