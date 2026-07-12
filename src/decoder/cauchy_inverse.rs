//! Closed-form Cauchy matrix inverse over GF(256).

use crate::gf256::GfElem;

/// Closed-form inverse of a square Cauchy matrix over GF(256).
///
/// `row_vars` and `col_vars` define the matrix
/// `C[i, j] = 1 / (row_vars[i] + col_vars[j])`. They must have equal length,
/// contain distinct elements within each set, and the two sets must be disjoint.
/// These conditions hold for SCRS submatrices because data indices are drawn
/// from `X = {0..k}` and repair indices from `Y = {k..k+m}`.
///
/// The returned inverse is row-major with rows corresponding to `col_vars` and
/// columns corresponding to `row_vars`, i.e. `inverse * C = I` in the natural
/// solve layout used by [`crate::decoder::LazyDecoderState`].
pub fn cauchy_inverse_closed_form(row_vars: &[GfElem], col_vars: &[GfElem]) -> Vec<GfElem> {
    rational_lagrange_coefficients(row_vars, col_vars, &[]).inverse
}

/// Factorized coefficients used by the reduced Cauchy decoder.
pub(crate) struct RationalLagrangeCoefficients {
    /// `A^-1`, row-major as `[missing_col][repair_row]`.
    pub inverse: Vec<GfElem>,
    /// Direct coefficients for additional Cauchy columns, row-major as
    /// `[present_col][missing_col]`.
    pub present: Vec<GfElem>,
}

/// Build inverse and fused coefficients using rational-Lagrange factors.
///
/// For `A[i,j] = 1 / (row[i] + col[j])`, define
///
/// ```text
/// R_i = ∏_h(row_i + col_h) / ∏_{l≠i}(row_i + row_l)
/// C_j = ∏_l(col_j + row_l) / ∏_{h≠j}(col_j + col_h).
/// ```
///
/// Then `A^-1[j,i] = R_i C_j / (col_j + row_i)`. For an additional
/// Cauchy column `1 / (row_i + z)`, its already-fused coefficient is
///
/// ```text
/// F[z,j] = C_j ∏_{h≠j}(z + col_h) / ∏_l(z + row_l).
/// ```
///
/// The four set products cost `O(r²)`, each additional column costs `O(r)`,
/// and emitting the inverse costs `O(r²)`. Thus coefficient construction is
/// `O(r² + r*p)`, rather than recomputing length-`r` products and dot products
/// for every output coefficient. In characteristic two, addition and
/// subtraction are identical, so no sign factors are required.
pub(crate) fn rational_lagrange_coefficients(
    row_vars: &[GfElem],
    col_vars: &[GfElem],
    present_vars: &[GfElem],
) -> RationalLagrangeCoefficients {
    debug_assert_eq!(
        row_vars.len(),
        col_vars.len(),
        "Cauchy inverse shape mismatch"
    );
    let n = row_vars.len();
    if n == 0 {
        return RationalLagrangeCoefficients {
            inverse: Vec::new(),
            present: Vec::new(),
        };
    }
    if n <= 2 {
        return rational_lagrange_small(row_vars, col_vars, present_vars);
    }

    let row_cross: Vec<_> = row_vars
        .iter()
        .map(|&row| {
            col_vars
                .iter()
                .fold(GfElem::ONE, |acc, &col| acc.mul(row.add(col)))
        })
        .collect();
    let col_cross: Vec<_> = col_vars
        .iter()
        .map(|&col| {
            row_vars
                .iter()
                .fold(GfElem::ONE, |acc, &row| acc.mul(col.add(row)))
        })
        .collect();

    // Batch all denominator inversions together: two within-set products, every
    // cross-set term used by A^-1, and one row product per present column.
    let row_within_start = 0;
    let col_within_start = n;
    let cross_start = 2 * n;
    let present_row_start = cross_start + n * n;
    let mut reciprocals = Vec::with_capacity(present_row_start + present_vars.len());
    for (i, &row) in row_vars.iter().enumerate() {
        reciprocals.push(
            row_vars
                .iter()
                .enumerate()
                .filter(|&(l, _)| l != i)
                .fold(GfElem::ONE, |acc, (_, &other)| acc.mul(row.add(other))),
        );
    }
    for (j, &col) in col_vars.iter().enumerate() {
        reciprocals.push(
            col_vars
                .iter()
                .enumerate()
                .filter(|&(h, _)| h != j)
                .fold(GfElem::ONE, |acc, (_, &other)| acc.mul(col.add(other))),
        );
    }
    for &col in col_vars {
        for &row in row_vars {
            reciprocals.push(col.add(row));
        }
    }
    for &z in present_vars {
        reciprocals.push(
            row_vars
                .iter()
                .fold(GfElem::ONE, |acc, &row| acc.mul(z.add(row))),
        );
    }
    batch_invert(&mut reciprocals);

    let row_factors: Vec<_> = row_cross
        .iter()
        .enumerate()
        .map(|(i, &cross)| cross.mul(reciprocals[row_within_start + i]))
        .collect();
    let col_factors: Vec<_> = col_cross
        .iter()
        .enumerate()
        .map(|(j, &cross)| cross.mul(reciprocals[col_within_start + j]))
        .collect();

    let mut inverse = Vec::with_capacity(n * n);
    for (j, &col_factor) in col_factors.iter().enumerate() {
        for (i, &row_factor) in row_factors.iter().enumerate() {
            inverse.push(
                row_factor
                    .mul(col_factor)
                    .mul(reciprocals[cross_start + j * n + i]),
            );
        }
    }

    let mut present = Vec::with_capacity(present_vars.len() * n);
    let mut prefix = vec![GfElem::ONE; n + 1];
    let mut suffix = vec![GfElem::ONE; n + 1];
    for (present_pos, &z) in present_vars.iter().enumerate() {
        for j in 0..n {
            prefix[j + 1] = prefix[j].mul(z.add(col_vars[j]));
        }
        for j in (0..n).rev() {
            suffix[j] = suffix[j + 1].mul(z.add(col_vars[j]));
        }
        let row_product_inv = reciprocals[present_row_start + present_pos];
        for j in 0..n {
            let all_but_j = prefix[j].mul(suffix[j + 1]);
            present.push(col_factors[j].mul(all_but_j).mul(row_product_inv));
        }
    }

    RationalLagrangeCoefficients { inverse, present }
}

/// Allocation-light closed forms avoid general factor setup at tiny `r`.
fn rational_lagrange_small(
    row_vars: &[GfElem],
    col_vars: &[GfElem],
    present_vars: &[GfElem],
) -> RationalLagrangeCoefficients {
    if row_vars.len() == 1 {
        let cross = row_vars[0].add(col_vars[0]);
        let inverse = vec![cross];
        let present = present_vars
            .iter()
            .map(|&z| cross.div(z.add(row_vars[0])))
            .collect();
        return RationalLagrangeCoefficients { inverse, present };
    }

    let r0 = row_vars[0];
    let r1 = row_vars[1];
    let c0 = col_vars[0];
    let c1 = col_vars[1];
    let row_delta = r0.add(r1);
    let col_delta = c0.add(c1);
    let row_factors = [
        r0.add(c0).mul(r0.add(c1)).div(row_delta),
        r1.add(c0).mul(r1.add(c1)).div(row_delta),
    ];
    let col_factors = [
        c0.add(r0).mul(c0.add(r1)).div(col_delta),
        c1.add(r0).mul(c1.add(r1)).div(col_delta),
    ];
    let inverse = vec![
        row_factors[0].mul(col_factors[0]).div(c0.add(r0)),
        row_factors[1].mul(col_factors[0]).div(c0.add(r1)),
        row_factors[0].mul(col_factors[1]).div(c1.add(r0)),
        row_factors[1].mul(col_factors[1]).div(c1.add(r1)),
    ];
    let mut present = Vec::with_capacity(2 * present_vars.len());
    for &z in present_vars {
        let row_product = z.add(r0).mul(z.add(r1));
        present.push(col_factors[0].mul(z.add(c1)).div(row_product));
        present.push(col_factors[1].mul(z.add(c0)).div(row_product));
    }
    RationalLagrangeCoefficients { inverse, present }
}

/// Invert nonzero field elements with one inversion and linear multiplications.
fn batch_invert(values: &mut [GfElem]) {
    let mut prefixes = Vec::with_capacity(values.len());
    let mut product = GfElem::ONE;
    for &value in values.iter() {
        debug_assert_ne!(value, GfElem::ZERO, "rational-Lagrange denominator is zero");
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

/// Original unfactored formula retained as a coefficient-level test oracle.
#[cfg(test)]
fn cauchy_inverse_unfactored(row_vars: &[GfElem], col_vars: &[GfElem]) -> Vec<GfElem> {
    let n = row_vars.len();
    let mut inv = vec![GfElem::ZERO; n * n];
    for (col_pos, &col_var) in col_vars.iter().enumerate() {
        for (row_pos, &row_var) in row_vars.iter().enumerate() {
            let mut num = GfElem::ONE;
            for &cv in col_vars {
                num = num.mul(row_var.add(cv));
            }
            for (other_pos, &rv) in row_vars.iter().enumerate() {
                if other_pos != row_pos {
                    num = num.mul(col_var.add(rv));
                }
            }

            let mut den = GfElem::ONE;
            for (other_pos, &rv) in row_vars.iter().enumerate() {
                if other_pos != row_pos {
                    den = den.mul(row_var.add(rv));
                }
            }
            for (other_pos, &cv) in col_vars.iter().enumerate() {
                if other_pos != col_pos {
                    den = den.mul(col_var.add(cv));
                }
            }
            inv[col_pos * n + row_pos] = num.div(den);
        }
    }
    inv
}

/// Invert an `n × n` matrix over GF(256), returning a row-major inverse.
///
/// This reference routine is intentionally coefficient-only: payload bytes are
/// never included in the elimination. `None` indicates a singular matrix.
#[cfg(test)]
fn invert_square(matrix: &[GfElem], n: usize) -> Option<Vec<GfElem>> {
    debug_assert_eq!(matrix.len(), n * n, "matrix shape mismatch");
    if n == 0 {
        return Some(Vec::new());
    }

    let stride = 2 * n;
    let mut aug = vec![GfElem::ZERO; n * stride];
    for r in 0..n {
        for c in 0..n {
            aug[r * stride + c] = matrix[r * n + c];
        }
        aug[r * stride + n + r] = GfElem::ONE;
    }

    for col in 0..n {
        let pivot = (col..n).find(|&r| aug[r * stride + col] != GfElem::ZERO)?;
        if pivot != col {
            for c in 0..stride {
                aug.swap(col * stride + c, pivot * stride + c);
            }
        }

        let pv = aug[col * stride + col];
        if pv != GfElem::ONE {
            let inv = pv.inv();
            for c in 0..stride {
                aug[col * stride + c] = aug[col * stride + c].mul(inv);
            }
        }

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * stride + col];
            if factor == GfElem::ZERO {
                continue;
            }
            for c in 0..stride {
                let prod = factor.mul(aug[col * stride + c]);
                aug[r * stride + c] = aug[r * stride + c].add(prod);
            }
        }
    }

    let mut inv = vec![GfElem::ZERO; n * n];
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = aug[r * stride + n + c];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_inverse_matches_reference() {
        let row_vars = [GfElem(4), GfElem(5), GfElem(6)];
        let col_vars = [GfElem(0), GfElem(2), GfElem(3)];
        let n = row_vars.len();
        let mut matrix = vec![GfElem::ZERO; n * n];
        for (r, &rv) in row_vars.iter().enumerate() {
            for (c, &cv) in col_vars.iter().enumerate() {
                matrix[r * n + c] = rv.add(cv).inv();
            }
        }

        let closed = cauchy_inverse_closed_form(&row_vars, &col_vars);
        let reference = invert_square(&matrix, n).unwrap();
        assert_eq!(closed, reference);

        // Verify closed * matrix = identity.
        for r in 0..n {
            for c in 0..n {
                let mut acc = GfElem::ZERO;
                for p in 0..n {
                    acc = acc.add(closed[r * n + p].mul(matrix[p * n + c]));
                }
                assert_eq!(acc, if r == c { GfElem::ONE } else { GfElem::ZERO });
            }
        }
    }

    fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut state: Vec<_> = (0..k).collect();
        loop {
            result.push(state.clone());
            let Some(i) = (0..k).rev().find(|&i| state[i] < n - k + i) else {
                return result;
            };
            state[i] += 1;
            for j in i + 1..k {
                state[j] = state[j - 1] + 1;
            }
        }
    }

    fn assert_factorized_coefficients<C: crate::coding_matrix::CodingMatrix>() {
        let (k, m) = (4, 3);
        let matrix = C::new(k, m).unwrap();
        for subset in k_subsets(k + m, k) {
            let missing: Vec<_> = (0..k).filter(|i| !subset.contains(i)).collect();
            let present: Vec<_> = (0..k).filter(|i| subset.contains(i)).collect();
            let repairs: Vec<_> = subset.iter().filter_map(|&i| i.checked_sub(k)).collect();
            let row_vars: Vec<_> = repairs.iter().map(|&i| matrix.y_var(i)).collect();
            let col_vars: Vec<_> = missing.iter().map(|&i| matrix.x_var(i)).collect();
            let present_vars: Vec<_> = present.iter().map(|&i| matrix.x_var(i)).collect();

            let factored = rational_lagrange_coefficients(&row_vars, &col_vars, &present_vars);
            let oracle = cauchy_inverse_unfactored(&row_vars, &col_vars);
            assert_eq!(factored.inverse, oracle, "subset={subset:?}");

            for (present_pos, &data_idx) in present.iter().enumerate() {
                for missing_pos in 0..missing.len() {
                    let expected = repairs.iter().enumerate().fold(
                        GfElem::ZERO,
                        |acc, (repair_pos, &repair)| {
                            acc.add(
                                oracle[missing_pos * repairs.len() + repair_pos]
                                    .mul(matrix.get(data_idx, repair)),
                            )
                        },
                    );
                    assert_eq!(
                        factored.present[present_pos * missing.len() + missing_pos],
                        expected,
                        "subset={subset:?}, present={data_idx}, missing_pos={missing_pos}"
                    );
                }
            }
        }
    }

    #[test]
    fn factorized_coefficients_match_oracle_for_all_small_subsets() {
        assert_factorized_coefficients::<crate::cauchy::CauchyView>();
        assert_factorized_coefficients::<crate::good_cauchy::GoodCauchyView>();
    }
}
