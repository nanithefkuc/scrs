//! GF(2^8) payload kernels.
//!
//! A thin seam over [`fgf::ops`], which owns the runtime SIMD dispatch. Every function
//! here is a shape adaptation, never arithmetic: SRS decides *which* rows and
//! coefficients participate, fgf decides how to move the bytes.
//!
//! fgf always provides a portable backend, so unlike the hand-written kernels these
//! replaced there is no scalar fallback to maintain — SRS's `simd` feature only chooses
//! whether fgf compiles its vector paths.

use fgf::Gf8;
use fgf::gf8::Elem as GfElem;

/// XOR a scaled source symbol into a destination symbol: `dst ^= coefficient * src`.
///
/// `coefficient == 0` is a no-op and `coefficient == 1` degrades to a plain XOR; both
/// short-circuits live in fgf.
pub fn xor_scaled_bytes(dst: &mut [u8], coefficient: GfElem, src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    fgf::ops::mul_add::<Gf8>(dst, coefficient, src);
}

/// Apply many source terms to one contiguous group of flat destination rows.
///
/// `dst` is `row_count` rows of `symbol_len` bytes; each term supplies one coefficient
/// per row. fgf fuses the whole term list so a destination row is loaded once for all
/// sources rather than once per source.
pub fn xor_scaled_bytes_rows_terms(
    dst: &mut [u8],
    symbol_len: usize,
    row_count: usize,
    terms: &[(&[GfElem], &[u8])],
) {
    debug_assert_eq!(dst.len(), row_count * symbol_len);
    fgf::ops::mul_add_matrix::<Gf8>(dst, symbol_len, row_count, terms);
}

/// XOR one source symbol into every row of a flat destination buffer, each row scaled
/// by its own coefficient.
///
/// This is the systematic-encode shape: one data symbol fanned out across the repair
/// rows, with the source held in registers across all destinations.
pub fn xor_scaled_bytes_rows(
    destinations: &mut [u8],
    symbol_len: usize,
    coefficients: &[GfElem],
    src: &[u8],
) {
    debug_assert_eq!(src.len(), symbol_len);
    debug_assert_eq!(destinations.len(), coefficients.len() * symbol_len);
    fgf::ops::mul_add_scatter::<Gf8>(destinations, symbol_len, coefficients, src);
}
