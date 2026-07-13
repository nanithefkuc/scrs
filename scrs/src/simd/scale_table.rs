use crate::gf256::GfElem;

/// Compact precomputed multiplication tables for one GF(256) coefficient.
#[derive(Clone, Debug)]
pub(crate) struct ScaleTable {
    pub(crate) coeff: GfElem,
    pub(super) lo: [u8; 16],
    pub(super) hi: [u8; 16],
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

/// Shared immutable multiplication tables indexed by raw coefficient byte.
static SCALE_TABLE_BANK: std::sync::LazyLock<[ScaleTable; 256]> =
    std::sync::LazyLock::new(|| core::array::from_fn(|i| ScaleTable::new(GfElem(i as u8))));

/// Return the shared shuffle table for a coefficient.
#[inline]
pub(crate) fn scale_table(coeff: GfElem) -> &'static ScaleTable {
    &SCALE_TABLE_BANK[coeff.0 as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_scale_table_bank_matches_individual_tables() {
        for coefficient in 0..=u8::MAX {
            let coefficient = GfElem(coefficient);
            let shared = scale_table(coefficient);
            let individual = ScaleTable::new(coefficient);
            assert_eq!(shared.coeff, individual.coeff);
            assert_eq!(shared.lo, individual.lo);
            assert_eq!(shared.hi, individual.hi);
        }
    }
}
