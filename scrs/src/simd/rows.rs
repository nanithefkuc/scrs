use crate::gf256::GfElem;

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::x86;
use super::{dispatch, scale_table::ScaleTable};

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

        dispatch::xor_scaled_bytes_many_indexed_trusted(
            self.dst,
            self.symbol_len,
            byte_offset,
            src_chunk.len(),
            self.indices,
            scales,
            src_chunk,
        );
    }

    /// Add one source symbol to exactly four rows with a grouped source-major kernel.
    ///
    /// Uses the unrolled GFNI path on x86 when available, otherwise the NEON
    /// four-output nibble kernel on AArch64. Returns `false` without modifying
    /// the destination when this view does not contain exactly four rows or the
    /// active plan has no grouped kernel.
    pub(crate) fn xor_scaled_4_grouped(&mut self, coefficients: &[GfElem], src: &[u8]) -> bool {
        assert_eq!(src.len(), self.symbol_len, "source symbol length mismatch");
        assert_eq!(
            self.indices.len(),
            coefficients.len(),
            "destination/coefficient count mismatch"
        );
        if self.indices.len() != 4 || !dispatch::kernel_plan().supports_grouped_source_major() {
            return false;
        }

        let indices: [usize; 4] = self.indices.try_into().expect("four destination indices");
        let coefficients: [GfElem; 4] = coefficients.try_into().expect("four coefficients");

        match dispatch::kernel_plan() {
            dispatch::KernelPlan::Gfni => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    x86::dispatch_xor_scaled_bytes_4_indexed_gfni(
                        self.dst,
                        self.symbol_len,
                        &indices,
                        &coefficients,
                        src,
                    );

                    true
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            }
            dispatch::KernelPlan::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    aarch64::dispatch_xor_scaled_bytes_4_indexed_neon(
                        self.dst,
                        self.symbol_len,
                        &indices,
                        &coefficients,
                        src,
                    );

                    true
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    false
                }
            }
            dispatch::KernelPlan::Avx2Nibble
            | dispatch::KernelPlan::Ssse3Nibble
            | dispatch::KernelPlan::Scalar => false,
        }
    }

    /// Backward-compatible alias for the GFNI-only call sites/tests.
    #[cfg(test)]
    pub(crate) fn xor_scaled_4_gfni(&mut self, coefficients: &[GfElem], src: &[u8]) -> bool {
        if !dispatch::gfni_available() {
            return false;
        }
        self.xor_scaled_4_grouped(coefficients, src)
    }

    /// Add one source symbol using compact coefficient bytes.
    #[cfg(test)]
    pub(crate) fn xor_scaled_coefficients(&mut self, coefficients: &[GfElem], src: &[u8]) {
        assert_eq!(src.len(), self.symbol_len, "source symbol length mismatch");
        assert_eq!(
            self.indices.len(),
            coefficients.len(),
            "destination/coefficient count mismatch"
        );
        for (&index, &coefficient) in self.indices.iter().zip(coefficients) {
            let start = index * self.symbol_len;
            dispatch::xor_scaled_bytes_coeff(
                &mut self.dst[start..start + self.symbol_len],
                coefficient,
                src,
            );
        }
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

/// Add one source symbol into `m` contiguous repair rows in a flat buffer.
///
/// Layout: `repairs[j * symbol_len .. (j+1) * symbol_len]` is repair `j`.
/// `coeffs` has length `m` and holds the per-repair GF(256) scales for this
/// source. This is the streaming-encoder hot path: one data arrival updates
/// every repair buffer.
///
/// On GFNI hosts, complete groups of four repairs use the fully unrolled
/// source-major kernel (load source once, update four destinations) that also
/// powers decoder reconstruction. Remainders and non-GFNI backends fall back
/// to per-destination AXPY.
pub(crate) fn xor_scaled_bytes_rows(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    let m = coeffs.len();
    assert_eq!(src.len(), symbol_len, "source length must equal symbol_len");
    assert_eq!(
        repairs.len(),
        m * symbol_len,
        "flat repair buffer must be m * symbol_len"
    );
    if m == 0 || symbol_len == 0 {
        return;
    }

    match dispatch::kernel_plan() {
        dispatch::KernelPlan::Gfni => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_rows_gfni(repairs, symbol_len, coeffs, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            xor_scaled_bytes_rows_scalar(repairs, symbol_len, coeffs, src);
        }
        dispatch::KernelPlan::Avx2Nibble => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x86::dispatch_xor_scaled_bytes_rows_avx2(repairs, symbol_len, coeffs, src);

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            xor_scaled_bytes_rows_scalar(repairs, symbol_len, coeffs, src);
        }
        dispatch::KernelPlan::Neon => {
            #[cfg(target_arch = "aarch64")]
            aarch64::dispatch_xor_scaled_bytes_rows_neon(repairs, symbol_len, coeffs, src);

            #[cfg(not(target_arch = "aarch64"))]
            xor_scaled_bytes_rows_scalar(repairs, symbol_len, coeffs, src);
        }
        dispatch::KernelPlan::Ssse3Nibble | dispatch::KernelPlan::Scalar => {
            xor_scaled_bytes_rows_scalar(repairs, symbol_len, coeffs, src);
        }
    }
}

fn xor_scaled_bytes_rows_scalar(
    repairs: &mut [u8],
    symbol_len: usize,
    coeffs: &[GfElem],
    src: &[u8],
) {
    for (j, &coeff) in coeffs.iter().enumerate() {
        let start = j * symbol_len;
        dispatch::xor_scaled_bytes_coeff(&mut repairs[start..start + symbol_len], coeff, src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::dispatch::{gfni_available, kernel_plan, xor_scaled_bytes};

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
    fn unrolled_gfni_4_output_matches_reference() {
        if !gfni_available() {
            return;
        }

        let indices = [1, 3, 5, 7];
        let coefficients = [GfElem::ZERO, GfElem::ONE, GfElem(0x53), GfElem(0xff)];
        for symbol_len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1400, 4099] {
            let src: Vec<_> = (0..symbol_len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let mut expected: Vec<_> = (0..9 * symbol_len)
                .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                .collect();
            let mut actual = expected.clone();
            for (&index, &coefficient) in indices.iter().zip(&coefficients) {
                let start = index * symbol_len;
                for (destination, &source) in
                    expected[start..start + symbol_len].iter_mut().zip(&src)
                {
                    *destination ^= GfElem(source).mul_xtime(coefficient).0;
                }
            }

            let mut rows = IndexedDestinationRows::new(&mut actual, symbol_len, &indices);
            assert!(rows.xor_scaled_4_gfni(&coefficients, &src));
            assert_eq!(actual, expected, "len={symbol_len}");
        }
    }

    #[test]
    fn unrolled_grouped_4_output_matches_reference() {
        if !kernel_plan().supports_grouped_source_major() {
            return;
        }

        let indices = [1, 3, 5, 7];
        let coefficients = [GfElem::ZERO, GfElem::ONE, GfElem(0x53), GfElem(0xff)];
        for symbol_len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1400, 4099] {
            let src: Vec<_> = (0..symbol_len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let mut expected: Vec<_> = (0..9 * symbol_len)
                .map(|i| (i as u8).wrapping_mul(13).wrapping_add(0xa5))
                .collect();
            let mut actual = expected.clone();
            for (&index, &coefficient) in indices.iter().zip(&coefficients) {
                let start = index * symbol_len;
                for (destination, &source) in
                    expected[start..start + symbol_len].iter_mut().zip(&src)
                {
                    *destination ^= GfElem(source).mul_xtime(coefficient).0;
                }
            }

            let mut rows = IndexedDestinationRows::new(&mut actual, symbol_len, &indices);
            assert!(rows.xor_scaled_4_grouped(&coefficients, &src));
            assert_eq!(actual, expected, "len={symbol_len}");
        }
    }
}
