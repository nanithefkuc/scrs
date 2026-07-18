//! Good-Cauchy coordinates over GF(65536).

use crate::gf65536::{GENERATOR, GfElem};

/// Maximum number of symbols in one nonzero-coordinate Tower Cauchy codeword.
pub const MAX_SYMBOLS: usize = 65_535;

/// A geometric-progression Good-Cauchy matrix over GF(65536).
///
/// Data coordinates are `X = {g^0, ..., g^(k-1)}` and repair coordinates are
/// `Y = {g^k, ..., g^(k+m-1)}`, where `g = 0x08 + u` is primitive. Therefore
/// every coordinate is nonzero and distinct while `k + m <= 65535`.
#[derive(Clone, Copy, Debug)]
pub struct TowerCauchyView {
    k: usize,
    m: usize,
}

impl TowerCauchyView {
    /// Construct a view for nonzero `(k, m)` with `k + m <= 65535`.
    pub fn new(k: usize, m: usize) -> Option<Self> {
        let n = k.checked_add(m)?;
        if k == 0 || m == 0 || n > MAX_SYMBOLS {
            return None;
        }
        Some(Self { k, m })
    }

    /// Number of systematic data symbols.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Number of repair symbols.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Total codeword length.
    pub const fn n(&self) -> usize {
        self.k + self.m
    }

    /// Data coordinate `x_i = g^i`.
    #[inline]
    pub fn x_var(&self, i: usize) -> GfElem {
        debug_assert!(i < self.k);
        power(i)
    }

    /// Repair coordinate `y_j = g^(k+j)`.
    #[inline]
    pub fn y_var(&self, j: usize) -> GfElem {
        debug_assert!(j < self.m);
        power(self.k + j)
    }

    /// Cauchy coefficient `1 / (x_i + y_j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> GfElem {
        debug_assert!(i < self.k && j < self.m);
        self.x_var(i).add(self.y_var(j)).inv()
    }

    /// Materialize the `k × m` coefficients in data-major order.
    ///
    /// The diagonal factorization needs only one extension-field inversion via
    /// batch inversion. `None` reports allocation failure or size overflow.
    pub fn coefficient_matrix(&self) -> Option<Vec<GfElem>> {
        let coefficient_count = self.k.checked_mul(self.m)?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(coefficient_count).ok()?;

        let mut powers = Vec::new();
        powers.try_reserve_exact(self.n()).ok()?;
        let mut current = GfElem::ONE;
        for _ in 0..self.n() {
            powers.push(current);
            current = current.mul(GENERATOR);
        }

        // base[d] = 1 / (1 + g^d), d=1..n-1. d=0 is never used because
        // X and Y are disjoint.
        let mut base = Vec::new();
        base.try_reserve_exact(self.n()).ok()?;
        base.push(GfElem::ZERO);
        base.extend(
            powers
                .iter()
                .take(self.n())
                .skip(1)
                .map(|&value| GfElem::ONE.add(value)),
        );
        batch_invert(&mut base[1..]);

        let generator_inverse = GENERATOR.inv();
        let mut row_scale = GfElem::ONE;
        for i in 0..self.k {
            for j in 0..self.m {
                let diagonal = self.k + j - i;
                coefficients.push(row_scale.mul(base[diagonal]));
            }
            row_scale = row_scale.mul(generator_inverse);
        }
        Some(coefficients)
    }
}

/// Invert nonzero elements with one field inversion and three multiplies per
/// element. The input is replaced in place by its reciprocals.
pub(crate) fn batch_invert(values: &mut [GfElem]) {
    if values.is_empty() {
        return;
    }
    let mut prefixes = Vec::with_capacity(values.len());
    let mut product = GfElem::ONE;
    for &value in values.iter() {
        debug_assert_ne!(value, GfElem::ZERO, "batch inversion contains zero");
        prefixes.push(product);
        product = product.mul(value);
    }
    let mut reciprocal = product.inv();
    for i in (0..values.len()).rev() {
        let value = values[i];
        values[i] = reciprocal.mul(prefixes[i]);
        reciprocal = reciprocal.mul(value);
    }
}

#[inline]
fn power(exponent: usize) -> GfElem {
    debug_assert!(exponent < MAX_SYMBOLS);
    GENERATOR.pow(exponent as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_extend_beyond_gf256_limit() {
        assert!(TowerCauchyView::new(256, 256).is_some());
        assert!(TowerCauchyView::new(32_768, 32_767).is_some());
        assert!(TowerCauchyView::new(32_768, 32_768).is_none());
        assert!(TowerCauchyView::new(0, 1).is_none());
        assert!(TowerCauchyView::new(1, 0).is_none());
    }

    #[test]
    fn coordinate_sets_are_disjoint() {
        let view = TowerCauchyView::new(300, 200).unwrap();
        let mut coordinates = std::collections::HashSet::new();
        for i in 0..view.k() {
            assert!(coordinates.insert(view.x_var(i)));
        }
        for j in 0..view.m() {
            assert!(coordinates.insert(view.y_var(j)));
        }
    }

    #[test]
    fn factorized_coefficients_match_direct_formula() {
        for (k, m) in [(1, 1), (4, 3), (17, 19), (256, 2)] {
            let view = TowerCauchyView::new(k, m).unwrap();
            let matrix = view.coefficient_matrix().unwrap();
            for i in 0..k {
                for j in 0..m {
                    assert_eq!(matrix[i * m + j], view.get(i, j));
                }
            }
        }
    }

    #[test]
    fn batch_inversion_matches_individual_inverses() {
        let mut values: Vec<_> = (1..1000).map(GfElem).collect();
        let expected: Vec<_> = values.iter().map(|value| value.inv()).collect();
        batch_invert(&mut values);
        assert_eq!(values, expected);
    }
}
