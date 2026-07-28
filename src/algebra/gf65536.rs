//! GF(2^16) arithmetic as a quadratic extension of SCRS's AES field.
//!
//! Let `F = GF(2^8)` use the Rijndael polynomial `0x11B`. This module
//! represents an extension element as `a + b*u`, with `a,b in F`, and uses
//!
//! ```text
//! u^2 + u + 0x20 = 0.
//! ```
//!
//! The absolute trace of `0x20` in `F` is one, so the quadratic is
//! irreducible. The stable two-byte representation is `[a, b]`: constant
//! component first, extension component second. It is also the little-endian
//! representation of the wrapped `u16`.

use core::fmt;

use super::gf256::GfElem as BaseElem;

/// Constant term of the irreducible tower polynomial `u^2 + u + DELTA`.
pub const DELTA: BaseElem = BaseElem(0x20);

/// A primitive element of the extension field, represented as `0x08 + u`.
///
/// Its multiplicative order is 65535. Tower Cauchy coordinate sets use
/// successive powers of this element.
pub const GENERATOR: GfElem = GfElem(0x0108);

/// An element of GF((2^8)^2), stored as `a + b*u`.
///
/// The low byte is `a` and the high byte is `b`. [`GfElem::to_bytes`] defines
/// the corresponding stable wire representation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GfElem(pub u16);

impl GfElem {
    /// The additive identity.
    pub const ZERO: Self = Self(0);

    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Construct `a + b*u` from its two GF(256) components.
    pub const fn from_components(a: BaseElem, b: BaseElem) -> Self {
        Self((a.0 as u16) | ((b.0 as u16) << 8))
    }

    /// Return the `(a, b)` components of `a + b*u`.
    pub const fn components(self) -> (BaseElem, BaseElem) {
        (BaseElem(self.0 as u8), BaseElem((self.0 >> 8) as u8))
    }

    /// Construct an element from the stable `[a, b]` byte representation.
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// Return the stable `[a, b]` byte representation.
    pub const fn to_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Return the wrapped component bits.
    pub const fn to_u16(self) -> u16 {
        self.0
    }

    /// Field addition. Addition and subtraction are component-wise XOR.
    pub const fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    /// Field subtraction. Identical to [`GfElem::add`] in characteristic two.
    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }

    /// Karatsuba multiplication followed by `u^2 = u + DELTA` reduction.
    #[must_use]
    pub const fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.components();
        let (c, d) = rhs.components();
        let ac = a.mul(c);
        let bd = b.mul(d);
        let constant = ac.add(DELTA.mul(bd));
        // (a+b)(c+d) + ac = ad + bc + bd, the coefficient of u after
        // replacing bd*u^2 by bd*u + DELTA*bd.
        let extension = a.add(b).mul(c.add(d)).add(ac);
        Self::from_components(constant, extension)
    }

    /// Square using the tower reduction relation.
    #[must_use]
    pub const fn square(self) -> Self {
        let (a, b) = self.components();
        let a2 = a.mul(a);
        let b2 = b.mul(b);
        Self::from_components(a2.add(DELTA.mul(b2)), b2)
    }

    /// Multiplicative inverse through the quadratic conjugate and norm.
    ///
    /// Zero maps to zero, matching SCRS's GF(256) convention.
    #[must_use]
    pub const fn inv(self) -> Self {
        let (a, b) = self.components();
        if a.0 == 0 && b.0 == 0 {
            return Self::ZERO;
        }
        // conjugate(a + b*u) = (a+b) + b*u
        // norm = a^2 + a*b + DELTA*b^2
        let norm = a.mul(a).add(a.mul(b)).add(DELTA.mul(b.mul(b)));
        let norm_inv = norm.inv();
        Self::from_components(a.add(b).mul(norm_inv), b.mul(norm_inv))
    }

    /// Field division. Division by zero returns zero in release builds and
    /// triggers a debug assertion, matching SCRS's GF(256) convention.
    #[must_use]
    pub const fn div(self, rhs: Self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        debug_assert!(rhs.0 != 0, "division by zero in GF(65536)");
        if rhs.0 == 0 {
            return Self::ZERO;
        }
        self.mul(rhs.inv())
    }

    /// Raise this element to an unsigned integer power.
    #[must_use]
    pub const fn pow(self, mut exponent: u32) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }
}

impl fmt::Debug for GfElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GfElem(0x{:04x})", self.0)
    }
}

impl core::ops::Add for GfElem {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        GfElem::add(self, rhs)
    }
}

impl core::ops::Sub for GfElem {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        GfElem::sub(self, rhs)
    }
}

impl core::ops::Mul for GfElem {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        GfElem::mul(self, rhs)
    }
}

impl core::ops::Div for GfElem {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        GfElem::div(self, rhs)
    }
}

impl core::ops::AddAssign for GfElem {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl core::ops::MulAssign for GfElem {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn schoolbook_mul(lhs: GfElem, rhs: GfElem) -> GfElem {
        let (a, b) = lhs.components();
        let (c, d) = rhs.components();
        GfElem::from_components(
            a.mul(c).add(DELTA.mul(b.mul(d))),
            a.mul(d).add(b.mul(c)).add(b.mul(d)),
        )
    }

    #[test]
    fn selected_delta_has_absolute_trace_one() {
        let mut trace = BaseElem::ZERO;
        let mut power = DELTA;
        for _ in 0..8 {
            trace = trace.add(power);
            power = power.mul(power);
        }
        assert_eq!(trace, BaseElem::ONE);
    }

    #[test]
    fn known_answers_pin_basis_and_wire_order() {
        assert_eq!(GfElem(0x1234).mul(GfElem(0xabcd)), GfElem(0x8ee3));
        assert_eq!(GfElem(0x0100).square(), GfElem(0x0120));
        assert_eq!(GfElem(0x1234).inv(), GfElem(0xbea3));
        assert_eq!(GfElem(0x1234).to_bytes(), [0x34, 0x12]);
        assert_eq!(GfElem::from_bytes([0x34, 0x12]), GfElem(0x1234));
    }

    #[test]
    fn selected_generator_is_primitive() {
        for prime_factor in [3, 5, 17, 257] {
            assert_ne!(GENERATOR.pow(65535 / prime_factor), GfElem::ONE);
        }
        assert_eq!(GENERATOR.pow(65535), GfElem::ONE);
    }

    proptest! {
        #[test]
        fn optimized_multiplication_matches_schoolbook(a in any::<u16>(), b in any::<u16>()) {
            prop_assert_eq!(GfElem(a).mul(GfElem(b)), schoolbook_mul(GfElem(a), GfElem(b)));
        }

        #[test]
        fn field_axioms(a in any::<u16>(), b in any::<u16>(), c in any::<u16>()) {
            let a = GfElem(a);
            let b = GfElem(b);
            let c = GfElem(c);
            prop_assert_eq!(a.mul(b), b.mul(a));
            prop_assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
            prop_assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
            prop_assert_eq!(a.add(GfElem::ZERO), a);
            prop_assert_eq!(a.mul(GfElem::ONE), a);
        }

        #[test]
        fn nonzero_inverse(a in 1u16..=u16::MAX) {
            let a = GfElem(a);
            prop_assert_eq!(a.mul(a.inv()), GfElem::ONE);
        }
    }
}
