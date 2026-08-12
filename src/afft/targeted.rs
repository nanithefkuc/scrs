//! Reusable `gfm` adapter for targeted AFFT recovery.

use fgf::field::Elem;
use gfm::{Matrix, Ple, PleScratch};

use super::Field;

/// Fixed-capacity square inversion workspace.
///
/// Smaller systems are embedded as the leading block of `A ⊕ I`, allowing one
/// `gfm::Ple` allocation to serve every targeted erasure count up to `order`.
#[derive(Debug)]
pub(super) struct TargetedInverse<F: Field> {
    order: usize,
    ple: Option<Ple<F>>,
    inverse: Option<Matrix<F>>,
    scratch: PleScratch<F>,
}

impl<F: Field> TargetedInverse<F> {
    pub(super) fn new(order: usize) -> Self {
        let mut scratch = PleScratch::new();
        let (ple, inverse) = if order == 0 {
            (None, None)
        } else {
            let identity = Matrix::<F>::identity(order).expect("targeted geometry is validated");
            (
                Some(Ple::decompose(identity, &mut scratch)),
                Some(Matrix::zeros(order, order).expect("targeted geometry is validated")),
            )
        };
        Self {
            order,
            ple,
            inverse,
            scratch,
        }
    }

    /// Invert the leading `size × size` row-major system into `output`.
    ///
    /// Returns `false` and leaves `output` untouched when the system is
    /// singular. Repeated calls allocate nothing.
    pub(super) fn invert_into(
        &mut self,
        system: &[F::Elem],
        size: usize,
        output: &mut [F::Elem],
    ) -> bool {
        assert!(
            size > 0 && size <= self.order,
            "targeted system exceeds capacity"
        );
        assert!(system.len() >= size * size, "targeted system is too small");
        assert_eq!(
            output.len(),
            size * size,
            "targeted inverse has the wrong size"
        );

        let ple = self
            .ple
            .as_mut()
            .expect("positive capacity has a decomposition");
        ple.redecompose_with(&mut self.scratch, |matrix| {
            for row in 0..size {
                for column in 0..size {
                    matrix.set(row, column, system[row * size + column]);
                }
            }
            for diagonal in size..self.order {
                matrix.set(diagonal, diagonal, F::Elem::ONE);
            }
        });
        if ple.rank() != self.order {
            return false;
        }

        let inverse = self
            .inverse
            .as_mut()
            .expect("positive capacity has inverse storage");
        ple.inverse_into(inverse)
            .expect("the rank check established invertibility");
        for row in 0..size {
            for column in 0..size {
                output[row * size + column] = inverse.get(row, column);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::Gf16;
    use fgf::field::Field as FgfField;

    type E = <Gf16 as FgfField>::Elem;

    fn elem(raw: u16) -> E {
        Gf16::read(&raw.to_le_bytes())
    }

    #[test]
    fn inverse_times_matrix_is_the_identity_across_reused_sizes() {
        let mut solver = TargetedInverse::<Gf16>::new(6);
        let mut state = 0x9e37_79b9u32;
        for size in 1..=6 {
            let mut matrix = vec![E::ZERO; size * size];
            loop {
                for entry in &mut matrix {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *entry = elem(u16::from_le_bytes(
                        state.to_le_bytes()[1..3].try_into().unwrap(),
                    ));
                }
                let mut inverse = vec![E::ZERO; size * size];
                if !solver.invert_into(&matrix, size, &mut inverse) {
                    continue;
                }
                for row in 0..size {
                    for column in 0..size {
                        let actual = (0..size).fold(E::ZERO, |accumulator, index| {
                            accumulator
                                .add(matrix[row * size + index].mul(inverse[index * size + column]))
                        });
                        let expected = if row == column { E::ONE } else { E::ZERO };
                        assert_eq!(actual, expected, "size {size} at ({row},{column})");
                    }
                }
                break;
            }
        }
    }

    #[test]
    fn singular_system_leaves_output_untouched() {
        let mut solver = TargetedInverse::<Gf16>::new(3);
        let matrix = [
            elem(1),
            elem(2),
            elem(3),
            elem(4),
            elem(8),
            elem(12),
            elem(5),
            elem(10),
            elem(15),
        ];
        let mut inverse = vec![E::ONE; 9];
        assert!(!solver.invert_into(&matrix, 3, &mut inverse));
        assert_eq!(inverse, vec![E::ONE; 9]);
    }
}
