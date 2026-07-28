//! GF(256) arithmetic with the AES/Rijndael polynomial `0x11B`.
//!
//! The field is GF(2)\[x\] / (x^8 + x^4 + x^3 + x + 1). Addition and subtraction
//! are both bitwise XOR. Multiplication uses generator `0x03` for the
//! discrete-log tables when the `gf256-tables` feature is enabled.
//!
//! Elements are wrapped in the [`GfElem`] newtype so raw `u8`s cannot be
//! accidentally mixed with field elements.
//!
//! # Backends
//!
//! - [`GfElem::mul_xtime`] is always available: a shift-and-XOR (Russian
//!   peasant) implementation that needs no static storage. It is the reference
//!   all other backends are fuzzed against.
//! - When the `gf256-tables` feature is enabled (default), [`GfElem::mul`]
//!   and [`GfElem::inv`] dispatch through compile-time-built `LOG`/`EXP`
//!   tables (~1 KiB total) for much faster field multiplication. The lookup
//!   path is the one place SCRS deliberately stores data: the tables fit in L1
//!   and the per-op latency win on the hot decode path is large.
//! - Disabling `gf256-tables` selects the `xtime` backend, which remains useful
//!   for table-free builds and as the portable reference implementation.

use core::fmt;

/// Irreducible reduction polynomial: x^8 + x^4 + x^3 + x + 1, i.e. `0x11B`.
///
/// Only the low 8 bits (`0x1B`) are ever XORed into a byte-sized accumulator;
/// the implicit bit 8 is set when we reduce a value that overflowed the field.
pub const REDUCTION_POLY: u16 = 0x11B;

/// Low byte of [`REDUCTION_POLY`], used for in-place reduction of `u8`.
pub const REDUCTION_LOW: u8 = 0x1B;

/// Generator of the multiplicative group of GF(256) under this polynomial.
///
/// `0x03` is a generator for the Rijndael polynomial; using the conventional
/// AES generator keeps our log/exp tables byte-for-byte compatible with
/// standard references and lets us cross-check against published tables.
pub const GENERATOR: u8 = 0x03;

/// A field element of GF(256) under the Rijndael polynomial.
///
/// The inner `u8` is the coefficient vector of the polynomial representation.
/// The additive identity is `GfElem(0)`; the multiplicative identity is
/// `GfElem(1)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GfElem(pub u8);

impl GfElem {
    /// The additive identity (also the absorbing element for multiplication).
    pub const ZERO: GfElem = GfElem(0);

    /// The multiplicative identity.
    pub const ONE: GfElem = GfElem(1);

    /// Unwrap to the raw byte. Intentionally explicit: callers should not
    /// silently treat field elements as integers.
    pub const fn to_byte(self) -> u8 {
        self.0
    }

    /// Field addition. In characteristic-2 fields this is identical to
    /// subtraction and to bitwise XOR.
    pub const fn add(self, rhs: GfElem) -> GfElem {
        GfElem(self.0 ^ rhs.0)
    }

    /// Field subtraction. Identical to [`add`][GfElem::add] in GF(2^n).
    pub const fn sub(self, rhs: GfElem) -> GfElem {
        self.add(rhs)
    }

    /// Multiplication via the shift-and-XOR "Russian peasant" algorithm.
    ///
    /// This is the reference backend: it is `const`, needs no static storage,
    /// and is used to validate the lookup-table backend. When the
    /// `gf256-tables` feature is disabled, this *is* the public [`mul`][GfElem::mul].
    #[must_use]
    pub const fn mul_xtime(self, rhs: GfElem) -> GfElem {
        let mut a = self.0;
        let b = rhs.0;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            // xtime: multiply a by x, reducing mod the Rijndael polynomial
            // if the high bit overflowed.
            let hi = a & 0x80 != 0;
            a = a << 1;
            if hi {
                a ^= REDUCTION_LOW;
            }
            i += 1;
        }
        GfElem(acc)
    }

    /// Multiplicative inverse via Fermat's little theorem: a^254 = a^-1.
    ///
    /// This is the reference inverse, independent of any lookup table. It is
    /// always available so that the lookup backend can be validated against it.
    #[must_use]
    pub const fn inv_xtime(self) -> GfElem {
        if self.0 == 0 {
            return GfElem(0);
        }
        // a^254 = a^-1, via left-to-right binary exponentiation.
        // 254 = 0b11111110: the LSB is the only clear bit, so we square on
        // every step and multiply by `self` on every step except the last.
        let mut result = GfElem::ONE;
        let mut i = 0;
        while i < 8 {
            result = result.mul_xtime(result); // square
            let bit = (254u32 >> (7 - i)) & 1;
            if bit == 1 {
                result = result.mul_xtime(self); // multiply
            }
            i += 1;
        }
        result
    }
}

#[cfg(feature = "gf256-tables")]
impl GfElem {
    /// Field multiplication.
    ///
    /// Dispatches through compile-time `LOG`/`EXP` tables when the
    /// `gf256-tables` feature is enabled (the default). The lookup costs two
    /// array reads and one add/XOR, versus the eight iterations of
    /// [`mul_xtime`][GfElem::mul_xtime].
    #[must_use]
    pub const fn mul(self, rhs: GfElem) -> GfElem {
        // 0 * x = x * 0 = 0; the log table has no entry for 0 (logs are 0..254
        // for the 255 nonzero elements), so we must short-circuit.
        if self.0 == 0 || rhs.0 == 0 {
            return GfElem::ZERO;
        }
        // a*b = EXP[(LOG[a] + LOG[b]) mod 255]
        let la = LOG[self.0 as usize] as usize;
        let lb = LOG[rhs.0 as usize] as usize;
        let sum = (la + lb) % 255;
        GfElem(EXP[sum])
    }

    /// Multiplicative inverse. `GfElem(0)` returns `GfElem(0)` by convention.
    #[must_use]
    pub const fn inv(self) -> GfElem {
        if self.0 == 0 {
            return GfElem::ZERO;
        }
        // a^-1 = EXP[(255 - LOG[a]) mod 255]
        //   since a * EXP[(255 - LOG[a]) mod 255]
        //      = EXP[(LOG[a] + 255 - LOG[a]) mod 255]
        //      = EXP[255 mod 255] = EXP[0] = 1.
        // The `mod 255` matters when LOG[a] == 0 (a == 1): 255 - 0 = 255
        // is out of bounds for the 255-entry EXP table.
        let l = LOG[self.0 as usize] as usize;
        GfElem(EXP[(255 - l) % 255])
    }

    /// Field division: `self / rhs`. Returns `GfElem::ZERO` when `self` is zero
    /// and panics (debug) / returns zero (release) when `rhs` is zero.
    #[must_use]
    pub const fn div(self, rhs: GfElem) -> GfElem {
        if self.0 == 0 {
            return GfElem::ZERO;
        }
        debug_assert!(rhs.0 != 0, "division by zero in GF(256)");
        if rhs.0 == 0 {
            return GfElem::ZERO;
        }
        self.mul(rhs.inv())
    }
}

#[cfg(not(feature = "gf256-tables"))]
impl GfElem {
    /// Field multiplication. Falls back to the `xtime` backend when the
    /// lookup-table feature is disabled.
    #[must_use]
    pub const fn mul(self, rhs: GfElem) -> GfElem {
        self.mul_xtime(rhs)
    }

    /// Multiplicative inverse. Falls back to the `xtime` backend.
    #[must_use]
    pub const fn inv(self) -> GfElem {
        self.inv_xtime()
    }

    /// Field division: `self / rhs`.
    #[must_use]
    pub const fn div(self, rhs: GfElem) -> GfElem {
        if self.0 == 0 {
            return GfElem::ZERO;
        }
        debug_assert!(rhs.0 != 0, "division by zero in GF(256)");
        if rhs.0 == 0 {
            return GfElem::ZERO;
        }
        self.mul(rhs.inv())
    }
}

impl fmt::Debug for GfElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GfElem(0x{:02x})", self.0)
    }
}

impl core::ops::Add for GfElem {
    type Output = GfElem;
    fn add(self, rhs: GfElem) -> GfElem {
        GfElem::add(self, rhs)
    }
}

impl core::ops::Sub for GfElem {
    type Output = GfElem;
    fn sub(self, rhs: GfElem) -> GfElem {
        GfElem::sub(self, rhs)
    }
}

impl core::ops::Mul for GfElem {
    type Output = GfElem;
    fn mul(self, rhs: GfElem) -> GfElem {
        GfElem::mul(self, rhs)
    }
}

impl core::ops::AddAssign for GfElem {
    fn add_assign(&mut self, rhs: GfElem) {
        *self = self.add(rhs);
    }
}

impl core::ops::MulAssign for GfElem {
    fn mul_assign(&mut self, rhs: GfElem) {
        *self = self.mul(rhs);
    }
}

// ---------------------------------------------------------------------------
// Compile-time log/exp tables for the lookup backend.
// ---------------------------------------------------------------------------

#[cfg(feature = "gf256-tables")]
mod tables {
    //! Discrete-log / exponential tables over GF(256) with the Rijndael
    //! polynomial and generator `0x03`.
    //!
    //! `EXP[i] = GENERATOR^i` for `i in 0..=254`.
    //! `LOG[EXP[i]] = i` for nonzero elements; `LOG[0]` is undefined and left
    //! as `0` (callers must short-circuit on zero before indexing).

    /// `EXP[i] = 0x03 ^ i mod REDUCTION_POLY`, for `i in 0..=254`.
    pub const EXP: [u8; 255] = build_exp();

    /// `LOG[x] = i` such that `0x03^i == x`, for nonzero `x`. `LOG[0]` is set
    /// to `0` and must never be read by callers (zero is handled separately).
    pub const LOG: [u8; 256] = build_log();

    /// Build the exponential table at compile time.
    const fn build_exp() -> [u8; 255] {
        let mut table = [0u8; 255];
        let mut v: u8 = 1; // GENERATOR^0 = 1
        let mut i = 0;
        while i < 255 {
            table[i] = v;
            v = xtime_mul(v, super::GENERATOR);
            i += 1;
        }
        table
    }

    /// Build the log table at compile time by inverting `EXP`.
    const fn build_log() -> [u8; 256] {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 255 {
            table[EXP[i] as usize] = i as u8;
            i += 1;
        }
        // table[0] intentionally left as 0: undefined, never read by callers.
        table
    }

    /// `const`-friendly single multiplication via `xtime`, used only to build
    /// the table at compile time. Mirrors [`GfElem::mul_xtime`].
    const fn xtime_mul(a: u8, b: u8) -> u8 {
        let mut a = a;
        let mut acc: u8 = 0;
        let mut i = 0;
        while i < 8 {
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
            let hi = a & 0x80 != 0;
            a = a << 1;
            if hi {
                a ^= super::REDUCTION_LOW;
            }
            i += 1;
        }
        acc
    }
}

#[cfg(feature = "gf256-tables")]
pub use tables::{EXP, LOG};

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn any_gf() -> impl Strategy<Value = GfElem> {
        (0u8..=255).prop_map(GfElem)
    }

    // ---- Field axioms (run against whichever backend is active) ----

    proptest! {
        #[test]
        fn add_is_xor(a in any_gf(), b in any_gf()) {
            prop_assert_eq!(a.add(b).0, a.0 ^ b.0);
        }

        #[test]
        fn add_is_sub(a in any_gf(), b in any_gf()) {
            prop_assert_eq!(a.add(b), a.sub(b));
        }

        #[test]
        fn add_associative(a in any_gf(), b in any_gf(), c in any_gf()) {
            prop_assert_eq!(a.add(b).add(c), a.add(b.add(c)));
        }

        #[test]
        fn add_identity(a in any_gf()) {
            prop_assert_eq!(a.add(GfElem::ZERO), a);
        }

        #[test]
        fn add_self_is_zero(a in any_gf()) {
            prop_assert_eq!(a.add(a), GfElem::ZERO);
        }

        #[test]
        fn mul_zero(a in any_gf()) {
            prop_assert_eq!(a.mul(GfElem::ZERO), GfElem::ZERO);
            prop_assert_eq!(GfElem::ZERO.mul(a), GfElem::ZERO);
        }

        #[test]
        fn mul_one(a in any_gf()) {
            prop_assert_eq!(a.mul(GfElem::ONE), a);
        }

        #[test]
        fn mul_commutative(a in any_gf(), b in any_gf()) {
            prop_assert_eq!(a.mul(b), b.mul(a));
        }

        #[test]
        fn mul_associative(a in any_gf(), b in any_gf(), c in any_gf()) {
            prop_assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
        }

        #[test]
        fn distributive(a in any_gf(), b in any_gf(), c in any_gf()) {
            prop_assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
        }

        #[test]
        fn inv_roundtrip_nonzero(a in any_gf()) {
            prop_assume!(a.0 != 0);
            prop_assert_eq!(a.mul(a.inv()), GfElem::ONE);
            prop_assert_eq!(a.inv().inv(), a);
        }

        #[test]
        fn div_roundtrip_nonzero(a in any_gf(), b in any_gf()) {
            prop_assume!(a.0 != 0 && b.0 != 0);
            prop_assert_eq!(a.mul(b).div(b), a);
        }
    }

    // ---- Cross-backend equivalence ----

    proptest! {
        #[test]
        fn mul_matches_xtime(a in any_gf(), b in any_gf()) {
            // The active `mul` (lookup when the feature is on, xtime otherwise)
            // must agree with the reference `mul_xtime`.
            prop_assert_eq!(a.mul(b), a.mul_xtime(b));
        }

        #[test]
        fn inv_matches_xtime(a in any_gf()) {
            prop_assume!(a.0 != 0);
            prop_assert_eq!(a.inv(), a.inv_xtime());
        }
    }

    // ---- Known-answer tests pinned to the Rijndael polynomial ----

    #[test]
    fn known_powers_of_generator() {
        // 0x03 is a generator: successive powers should cycle through all
        // 255 nonzero field elements and return to 1 at exponent 255.
        let mut g = GfElem::ONE;
        let mut seen = [false; 256];
        for _ in 0..255 {
            seen[g.0 as usize] = true;
            g = g.mul(GfElem(GENERATOR));
        }
        assert_eq!(g, GfElem::ONE, "generator should cycle back to 1");
        assert!(!seen[0], "zero should never appear as a power");
        for (i, &present) in seen.iter().enumerate().skip(1) {
            assert!(present, "element 0x{:02x} missing from generator cycle", i);
        }
    }

    #[test]
    fn known_small_products() {
        // x * x = x^2, i.e. 0x02 * 0x02 = 0x04 (no reduction needed).
        assert_eq!(GfElem(0x02).mul(GfElem(0x02)), GfElem(0x04));
        // x * (x^4 + x^3 + x + 1) = x^5 + x^4 + x^2 + x, no reduction.
        assert_eq!(GfElem(0x02).mul(GfElem(0x1B)), GfElem(0x36));
        // 0x80 * 0x02 overflows: 0x100 reduces to 0x1B.
        assert_eq!(GfElem(0x80).mul(GfElem(0x02)), GfElem(0x1B));
        // Generator times itself: 0x03 * 0x03 = 0x05 (x^2 + 1).
        assert_eq!(GfElem(0x03).mul(GfElem(0x03)), GfElem(0x05));
    }

    #[test]
    fn inv_known_values() {
        // 1^-1 = 1.
        assert_eq!(GfElem::ONE.inv(), GfElem::ONE);
        // x = 0x02: its inverse under Rijndael is 0x8D (a well-known
        // Rijndael S-box entry; S-box[0x01] = 0x7c is the *affine* output,
        // but the pure multiplicative inverse of 0x02 is 0x8D).
        assert_eq!(GfElem(0x02).inv(), GfElem(0x8D));
        // 0x03 = generator: its inverse is 0xF6 (= 0x03^254).
        assert_eq!(GfElem(0x03).inv(), GfElem(0xF6));
    }

    #[test]
    fn zero_inv_is_zero() {
        assert_eq!(GfElem::ZERO.inv(), GfElem::ZERO);
    }

    #[cfg(feature = "gf256-tables")]
    #[test]
    fn tables_are_mutual_inverses() {
        for i in 0..255u32 {
            let e = EXP[i as usize];
            assert_eq!(LOG[e as usize] as u32, i, "LOG[EXP[{}]] != {}", i, i);
        }
    }
}
