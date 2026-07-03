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
    debug_assert_eq!(
        row_vars.len(),
        col_vars.len(),
        "Cauchy inverse shape mismatch"
    );
    let n = row_vars.len();
    if n == 0 {
        return Vec::new();
    }

    let mut inv = vec![GfElem::ZERO; n * n];
    for (col_pos, &col_var) in col_vars.iter().enumerate() {
        for (row_pos, &row_var) in row_vars.iter().enumerate() {
            // Inverse entry B[col_pos, row_pos] is the residue coefficient of
            // the rational Lagrange basis function that is 1 at `row_var` and
            // 0 at every other row variable:
            //
            //   B[j,i] = Q(row_i) * ∏_{l≠i}(col_j + row_l)
            //            ---------------------------------------
            //            ∏_{l≠i}(row_i + row_l) * ∏_{h≠j}(col_j + col_h)
            //
            // where Q(z) = ∏_h(z + col_h). In characteristic 2, `+` is also
            // subtraction, so no sign correction is needed.
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
}
