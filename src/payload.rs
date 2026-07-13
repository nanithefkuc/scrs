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
