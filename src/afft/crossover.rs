//! Which recovery path an additive-FFT erasure pattern should take.
//!
//! The two reconstruction paths scale differently, so the choice is geometry
//! arithmetic rather than a constant:
//!
//! - The **targeted** dense solve folds every received row into `r` residuals
//!   and applies an `r x r` inverse, so it touches `r * (k + r)` symbol rows.
//!   Cost grows with the erasure count and ignores the transform domain.
//! - The **locator** path runs one inverse transform, one formal derivative,
//!   and one selected forward transform over the padded domain, so it costs
//!   about `transform_size * log2(transform_size)` butterflies whatever the
//!   erasure count is.
//!
//! Each path also pays a per-row-call overhead that dominates at short symbols,
//! which is why the crossover moves with `symbol_len` and not only with the
//! `r * (k + r)` versus `N * log2(N)` ratio: at 64-byte symbols the dense
//! path's per-source kernel calls are the whole cost, while at 1400 bytes the
//! payload arithmetic is.
//!
//! # Calibration
//!
//! Measured by `benches/afft_crossover.rs` on Lunar Lake (258V, AVX2+GFNI,
//! pinned physical core 0), timing prepared plans forced onto each path — the
//! steady state a resident plan or a warm scratch memo actually pays:
//!
//! | geometry | `symbol_len` | measured crossover | this model |
//! |---|--:|--:|--:|
//! | GF(2^8) `k16 m8` (`N=32`) | 64 | ~5.5 | 4 |
//! | GF(2^8) `k16 m8` (`N=32`) | 1400 | >8 (`m`-capped) | 13 |
//! | GF(2^8) `k64 m32` (`N=128`) | 64 | ~6.5 | 7 |
//! | GF(2^8) `k64 m32` (`N=128`) | 1400 | ~27 | 25 |
//! | GF(2^8) `k160 m80` (`N=256`) | 64 | ~5.5 | 6 |
//! | GF(2^8) `k160 m80` (`N=256`) | 1400 | ~27 | 27 |
//! | GF(2^16) `k512 m256` (`N=1024`) | 64 | ~3.5 | 3 |
//! | GF(2^16) `k512 m256` (`N=1024`) | 1400 | ~16 | 15 |
//!
//! The old fixed threshold of five was geometry- and symbol-blind: at
//! `k64 s1400` it conceded erasure counts 6 through 27 to the locator path,
//! which measured up to 5.5x slower there (`r=6`: ~9 us targeted versus ~39 us
//! locator).
//!
//! Being a few erasures off the true crossover costs almost nothing — the two
//! curves meet shallowly, so near the threshold either path is within a few
//! percent by construction. Being an order of magnitude off, as a constant is
//! for wide `k` and long symbols, does not.

use super::Field;

/// Butterfly weight of the locator path's three domain passes, over
/// [`SCALE`].
///
/// Fitted, not derived: the three transforms plus the derivative and the
/// per-point scaling do not decompose into a clean pass count.
const LOCATOR_BUTTERFLY_WEIGHT: u128 = 27;
/// Denominator shared by [`LOCATOR_BUTTERFLY_WEIGHT`].
const SCALE: u128 = 8;

/// Per-butterfly overhead of the locator path, in field elements.
const LOCATOR_ROW_OVERHEAD: u128 = 32;

/// Per-row-call overhead of the dense path's SIMD kernel invocations, in field
/// elements. Large enough to dominate short-symbol geometries, which is
/// exactly what the measurements show.
const DENSE_ROW_OVERHEAD: u128 = 512;

/// Which reconstruction path a decode should take.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RecoveryPath {
    /// Dense `r x r` solve against the received repair rows.
    Targeted,
    /// Forney-style recovery over the whole evaluation domain.
    Locator,
}

/// Relative cost of one dense-path element operation against one transform
/// element operation, as a fraction.
///
/// GF(2^16)'s dense multiply is materially more expensive per element than its
/// transform butterflies, so the same `r * (k + r)` versus `N * log2(N)` ratio
/// favours the locator path sooner there. Measured at about `5/2`.
const fn dense_element_weight<F: Field>() -> (u128, u128) {
    match F::BYTES {
        1 => (1, 1),
        _ => (5, 2),
    }
}

/// Largest erasure count that should take the targeted dense solve for this
/// geometry.
///
/// Always at least one — a single erasure never justifies three domain-sized
/// transforms — and never more than `k`, which bounds the erasure count any
/// systematic pattern can present. Callers additionally clamp to `m`.
#[must_use]
pub fn targeted_max_missing<F: Field>(k: usize, transform_size: usize, symbol_len: usize) -> usize {
    debug_assert!(k > 0 && transform_size.is_power_of_two());
    let elements = (symbol_len / F::BYTES) as u128;
    let butterflies = transform_size as u128 * transform_size.trailing_zeros() as u128;
    let (weight_numerator, weight_denominator) = dense_element_weight::<F>();

    // Equate the two costs and solve `r * (k + r) = quotient` for `r`.
    let locator = LOCATOR_BUTTERFLY_WEIGHT * butterflies * (elements + LOCATOR_ROW_OVERHEAD);
    let dense_row = SCALE * (elements + DENSE_ROW_OVERHEAD) * weight_numerator / weight_denominator;
    let quotient = locator / dense_row.max(1);

    let k128 = k as u128;
    let root = (k128 * k128 + 4 * quotient).isqrt();
    let estimate = (root.saturating_sub(k128) / 2) as usize;
    estimate.clamp(1, k)
}

/// Pick the recovery path for `missing` erasures at this geometry.
#[must_use]
pub fn recovery_path<F: Field>(
    k: usize,
    transform_size: usize,
    symbol_len: usize,
    missing: usize,
) -> RecoveryPath {
    if missing <= targeted_max_missing::<F>(k, transform_size, symbol_len) {
        RecoveryPath::Targeted
    } else {
        RecoveryPath::Locator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold must stay inside the range an erasure pattern can
    /// actually present, for every geometry the engines accept — including the
    /// extreme symbol lengths that stress the integer arithmetic.
    #[test]
    fn threshold_stays_within_the_erasure_range() {
        for k in [1usize, 2, 5, 8, 100, 255, 1024, 65_534] {
            for m in [1usize, 2, 32, 512] {
                let n = k + m;
                if n > 1 << 16 {
                    continue;
                }
                let transform_size = n.next_power_of_two();
                for symbol_len in [2usize, 64, 1400, 1 << 20] {
                    let threshold =
                        targeted_max_missing::<fff::Gf16>(k, transform_size, symbol_len);
                    assert!(
                        (1..=k).contains(&threshold),
                        "gf16 k={k} m={m} s={symbol_len} threshold={threshold}"
                    );
                    if n <= 256 {
                        let threshold =
                            targeted_max_missing::<fff::Gf8>(k, transform_size, symbol_len);
                        assert!(
                            (1..=k).contains(&threshold),
                            "gf8 k={k} m={m} s={symbol_len} threshold={threshold}"
                        );
                    }
                }
            }
        }
    }

    /// The measured anchors from the module's calibration table. These are the
    /// evidence the model is fitted to, so a constant edited without a fresh
    /// sweep must fail here rather than silently mis-route decodes.
    #[test]
    fn model_reproduces_the_measured_crossovers() {
        // (k, transform_size, symbol_len, measured crossover)
        let gf8 = [
            (16usize, 32usize, 64usize, 5.5f64),
            (64, 128, 64, 6.5),
            (64, 128, 1400, 27.0),
            (160, 256, 64, 5.5),
            (160, 256, 1400, 27.0),
        ];
        for (k, transform_size, symbol_len, measured) in gf8 {
            let predicted = targeted_max_missing::<fff::Gf8>(k, transform_size, symbol_len) as f64;
            let ratio = predicted / measured;
            assert!(
                (0.6..=1.6).contains(&ratio),
                "gf8 k={k} s={symbol_len}: predicted {predicted} vs measured {measured}"
            );
        }
        for (k, transform_size, symbol_len, measured) in [
            (512usize, 1024usize, 64usize, 3.5f64),
            (512, 1024, 1400, 16.0),
        ] {
            let predicted = targeted_max_missing::<fff::Gf16>(k, transform_size, symbol_len) as f64;
            let ratio = predicted / measured;
            assert!(
                (0.6..=1.6).contains(&ratio),
                "gf16 k={k} s={symbol_len}: predicted {predicted} vs measured {measured}"
            );
        }
    }

    /// A single erasure always takes the targeted path, and an all-data-erased
    /// pattern at large `k` always takes the locator path: the two regimes the
    /// evidence is unambiguous about.
    #[test]
    fn extremes_pick_the_expected_paths() {
        assert_eq!(
            recovery_path::<fff::Gf8>(64, 128, 1400, 1),
            RecoveryPath::Targeted,
            "one erasure never justifies domain transforms"
        );
        assert_eq!(
            recovery_path::<fff::Gf16>(1024, 2048, 1400, 1024),
            RecoveryPath::Locator,
            "erasing every data row must not run a 1024-wide dense solve"
        );
    }

    /// Longer symbols amortize the dense path's per-call overhead and push the
    /// crossover out; wider `k` makes each dense row more expensive and pulls
    /// it back in.
    #[test]
    fn threshold_follows_symbol_length_and_k() {
        assert!(
            targeted_max_missing::<fff::Gf8>(64, 128, 1400)
                > targeted_max_missing::<fff::Gf8>(64, 128, 64)
        );
        assert!(
            targeted_max_missing::<fff::Gf8>(160, 256, 1400)
                > targeted_max_missing::<fff::Gf16>(160, 256, 1400),
            "GF(2^16) dense multiplies cede to the transforms sooner"
        );
        // Both `k` here sit below the clamp, so this compares the model rather
        // than the `1..=k` bound.
        assert!(
            targeted_max_missing::<fff::Gf8>(200, 256, 1400)
                < targeted_max_missing::<fff::Gf8>(128, 256, 1400)
        );
    }
}
