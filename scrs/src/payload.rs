//! Portable payload operations shared by accelerated and scalar builds.

use crate::gf256::GfElem;

/// XOR a scaled source symbol into a destination symbol.
pub(crate) fn xor_scaled_bytes(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(feature = "simd")]
    {
        crate::simd::xor_scaled_bytes_coeff(dst, coefficient, src);
    }

    #[cfg(not(feature = "simd"))]
    {
        if coefficient == GfElem::ZERO {
            return;
        }
        if coefficient == GfElem::ONE {
            for (out, &input) in dst.iter_mut().zip(src) {
                *out ^= input;
            }
            return;
        }
        for (out, &input) in dst.iter_mut().zip(src) {
            *out ^= GfElem(input).mul(coefficient).0;
        }
    }
}

/// Apply many source terms to one contiguous group of `e` flat rows.
///
/// For each `(coeffs, src)` term (`coeffs.len() == e`, `src.len() ==
/// symbol_len`): `row[j] ^= coeffs[j] * src` for every `j` in `0..e`.
/// Equivalent to calling [`xor_scaled_bytes_rows`] per term, but resolves
/// the backend once for the whole batch and, on GFNI hosts, keeps each
/// destination tile in registers across all sources.
pub(crate) fn xor_scaled_bytes_rows_terms(
    dst: &mut [u8],
    symbol_len: usize,
    e: usize,
    terms: &[(&[GfElem], &[u8])],
) {
    debug_assert_eq!(dst.len(), e * symbol_len);

    #[cfg(feature = "simd")]
    {
        crate::simd::xor_scaled_bytes_rows_terms(dst, symbol_len, e, terms);
    }

    #[cfg(not(feature = "simd"))]
    {
        for &(coeffs, src) in terms {
            xor_scaled_bytes_rows(dst, symbol_len, coeffs, src);
        }
    }
}

/// XOR a source symbol into each row of a flat destination buffer.
pub(crate) fn xor_scaled_bytes_rows(
    destinations: &mut [u8],
    symbol_len: usize,
    coefficients: &[GfElem],
    src: &[u8],
) {
    debug_assert_eq!(src.len(), symbol_len);
    debug_assert_eq!(destinations.len(), coefficients.len() * symbol_len);

    #[cfg(feature = "simd")]
    {
        crate::simd::xor_scaled_bytes_rows(destinations, symbol_len, coefficients, src);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (destination, &coefficient) in
            destinations.chunks_exact_mut(symbol_len).zip(coefficients)
        {
            xor_scaled_bytes(destination, coefficient, src);
        }
    }
}
