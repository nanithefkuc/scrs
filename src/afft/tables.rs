//! Discrete logarithm tables over the multiplicative group.
//!
//! The erasure-locator transform works in exponents rather than field
//! elements: an `n`-fold product becomes a sum modulo `|F*|`, which is what
//! lets one XOR-convolution evaluate the locator at every domain point at
//! once (see [`crate::afft::locator`]).

use std::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::{Elem, Field as FgfField};
use fgf::{Gf8, Gf16};

/// A field with tabulated discrete logarithms, usable by the `rs` helpers.
///
/// Implemented for every field whose order admits full tables — extension
/// degree at most 16, i.e. at most 64Ki entries per direction. Wider fields
/// are excluded at compile time rather than failing at run time: a GF(2^32)
/// log table is 16 GiB.
///
/// Sealed in practice: [`ButterflyKernels`] is sealed, so no downstream type
/// can satisfy the supertrait.
pub trait RsField: ButterflyKernels {
    /// This field's log/exp tables, built once per process on first use.
    fn log_exp() -> &'static LogExpTables<Self>;
}

/// Discrete logarithms and their inverse, base [`FgfField::GENERATOR`].
///
/// `exp` holds `g^0 … g^(|F*|-1)`; `log` inverts it. `log(0)` is not defined
/// mathematically and is reported as `0`; callers rely on that sentinel
/// deliberately — see [`crate::afft::locator::ErasureLocator`], where the
/// vanishing self-term of a formal derivative must contribute the exponent
/// zero (the factor one).
pub struct LogExpTables<F: FgfField> {
    log: Vec<u32>,
    exp: Vec<F::Elem>,
    order: u32,
}

impl<F: FgfField> LogExpTables<F> {
    /// Tabulate the multiplicative group by walking the generator.
    fn build() -> Self {
        let order = Self::group_order();
        let entries = order as usize + 1;
        let mut log = vec![0u32; entries];
        let mut exp = Vec::with_capacity(order as usize);
        let mut value = F::Elem::ONE;
        for exponent in 0..order {
            exp.push(value);
            log[Self::index(value)] = exponent;
            value = value.mul(F::GENERATOR);
        }
        assert!(
            value == F::Elem::ONE,
            "the field generator does not have full multiplicative order"
        );
        Self { log, exp, order }
    }

    /// `|F*| = 2^BITS - 1`.
    const fn group_order() -> u32 {
        assert!(
            F::BITS <= 16,
            "log tables need an extension degree of 16 or less"
        );
        (1u32 << F::BITS) - 1
    }

    /// An element's table slot: its bit pattern, which `BITS <= 16` keeps
    /// inside `usize` on every target.
    fn index(value: F::Elem) -> usize {
        let bits = element_bits::<F>(value);
        debug_assert!(bits < 1 << F::BITS);
        usize::try_from(bits).expect("a 16-bit pattern fits usize")
    }

    /// The order of the multiplicative group, the modulus of every exponent.
    #[must_use]
    pub const fn order(&self) -> u32 {
        self.order
    }

    /// `log_g(value)`, with `log(0) == 0` by the sentinel convention above.
    #[must_use]
    pub fn log(&self, value: F::Elem) -> u32 {
        self.log[Self::index(value)]
    }

    /// `g^exponent`, for `exponent < order()`.
    #[must_use]
    pub fn exp(&self, exponent: u32) -> F::Elem {
        self.exp[exponent as usize]
    }
}

macro_rules! impl_rs_field {
    ($($field:ty),+ $(,)?) => {$(
        impl RsField for $field {
            fn log_exp() -> &'static LogExpTables<Self> {
                static TABLES: ::std::sync::LazyLock<LogExpTables<$field>> =
                    ::std::sync::LazyLock::new(LogExpTables::build);
                &TABLES
            }
        }
    )+};
}

impl_rs_field!(Gf8, Gf16);

pub(super) fn element_bits<F: FgfField>(value: F::Elem) -> u64 {
    debug_assert!(F::BYTES <= 8);
    let mut bytes = [0u8; 8];
    F::write(&mut bytes[..F::BYTES], value);
    u64::from_le_bytes(bytes)
}

/// `a · b mod modulus`, for a modulus below `2^32`.
///
/// The product needs 64 bits; the residue is below `modulus`, so the
/// narrowing cast cannot truncate.
#[expect(clippy::cast_possible_truncation)]
pub(crate) fn mul_mod(a: u32, b: u32, modulus: u32) -> u32 {
    (u64::from(a) * u64::from(b) % u64::from(modulus)) as u32
}

/// `base^exponent mod modulus`, for a modulus below `2^32`.
pub(crate) fn mod_pow(base: u32, exponent: u32, modulus: u32) -> u32 {
    let mut base = base % modulus;
    let mut exponent = exponent;
    let mut result = 1u32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_tables<F: RsField>() {
        let tables = F::log_exp();
        assert_eq!(tables.order(), (1u32 << F::BITS) - 1);
        assert_eq!(tables.exp(0), F::Elem::ONE);
        for exponent in 0..tables.order() {
            let value = tables.exp(exponent);
            assert_ne!(value, F::Elem::ZERO);
            assert_eq!(tables.log(value), exponent);
        }
        // Logarithms are additive: log(a·b) = log a + log b mod |F*|.
        let order = tables.order();
        for a in 1..order.min(97) {
            for b in 1..order.min(53) {
                let product = tables.exp(a).mul(tables.exp(b));
                assert_eq!(tables.log(product), (a + b) % order);
            }
        }
    }

    #[test]
    fn tables_invert_each_other() {
        check_tables::<Gf8>();
        check_tables::<Gf16>();
    }

    #[test]
    fn zero_logarithm_is_the_neutral_exponent() {
        let tables = Gf16::log_exp();
        assert_eq!(tables.log(<Gf16 as FgfField>::Elem::ZERO), 0);
        assert_eq!(
            tables.exp(tables.log(<Gf16 as FgfField>::Elem::ZERO)),
            <Gf16 as FgfField>::Elem::ONE
        );
    }

    #[test]
    fn mod_pow_matches_repeated_multiplication() {
        let modulus = 65_535;
        for exponent in 0..17 {
            let mut expected = 1u64;
            for _ in 0..exponent {
                expected = expected * 32_768 % u64::from(modulus);
            }
            assert_eq!(
                u64::from(mod_pow(32_768, exponent, modulus)),
                expected,
                "exponent {exponent}"
            );
        }
        // 2^(BITS-1) inverts two modulo 2^BITS - 1.
        assert_eq!(u64::from(mod_pow(32_768, 1, modulus)) * 2 % 65_535, 1);
        assert_eq!(u64::from(mod_pow(128, 1, 255)) * 2 % 255, 1);
    }
}
