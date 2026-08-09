//! The finite-field bases SRS's wire format depends on.
//!
//! SRS delegates all field arithmetic to [`fff`], but the *choice* of field
//! representation is not an implementation detail: it fixes the on-wire encoding of
//! every symbol and the coordinate sets of every coding matrix. An upstream change to
//! any constant below silently makes new parity undecodable by old peers, with no
//! compile error and no test failure anywhere else in the suite.
//!
//! These assertions replace the ad-hoc `const _: BaseElem = crate::gf65536::DELTA;`
//! tripwire that lived in the hand-written GF(65536) SIMD backend before the fff
//! retarget.

use fgf::field::Field as _;
use fgf::{Gf8, Gf16, gf8, gf16};

#[test]
fn gf8_basis_is_the_aes_field() {
    assert_eq!(gf8::REDUCTION_POLY, 0x11B, "GF(2^8) reduction polynomial");
    assert_eq!(gf8::REDUCTION_LOW, 0x1B, "GF(2^8) reduction low byte");
    assert_eq!(gf8::GENERATOR, gf8::Elem(0x03), "GF(2^8) generator");
    assert_eq!(Gf8::BITS, 8);
    assert_eq!(Gf8::BYTES, 1);
    assert_eq!(Gf8::ORDER, 256);
}

#[test]
fn gf8_generator_has_full_multiplicative_order() {
    // 0x03 must generate all 255 nonzero elements, or Good Cauchy's coordinate sets
    // collide and the matrix stops being MDS.
    let mut seen = [false; 256];
    let mut value = gf8::Elem::ONE;
    for _ in 0..255 {
        assert!(
            !seen[value.0 as usize],
            "generator repeats before order 255"
        );
        seen[value.0 as usize] = true;
        value = value.mul(gf8::GENERATOR);
    }
    assert_eq!(value, gf8::Elem::ONE, "generator order is not 255");
}

#[test]
fn gf16_basis_is_the_quadratic_tower_over_gf8() {
    // u^2 + u + DELTA = 0 over GF(2^8). A flat primitive-polynomial GF(65536) would
    // encode every element differently.
    assert_eq!(gf16::DELTA, gf8::Elem(0x20), "GF(2^16) tower constant");
    assert_eq!(gf16::GENERATOR, gf16::Elem(0x0108), "GF(2^16) generator");
    assert_eq!(Gf16::BITS, 16);
    assert_eq!(Gf16::BYTES, 2);
    assert_eq!(Gf16::ORDER, 65536);
}

#[test]
fn gf16_wire_encoding_is_little_endian_component_order() {
    // The two-byte wire element is [a, b] where the element is a + b*u: low byte first.
    // Reversing it would swap the two interleaved base-field planes in every payload.
    let value = gf16::Elem::from_components(gf8::Elem(0xAA), gf8::Elem(0xBB));
    assert_eq!(value.to_bytes(), [0xAA, 0xBB]);
    assert_eq!(value.to_raw(), 0xBBAA);
    assert_eq!(gf16::Elem::from_bytes([0xAA, 0xBB]), value);

    let (a, b) = value.components();
    assert_eq!((a, b), (gf8::Elem(0xAA), gf8::Elem(0xBB)));
}

#[test]
fn gf16_generator_has_full_multiplicative_order() {
    // Order exactly 65535: Tower Cauchy uses g^i for the x-coordinates, so a shorter
    // order would produce a duplicate coordinate and a singular submatrix.
    assert_eq!(gf16::GENERATOR.pow(65_535), gf16::Elem::ONE);
    for factor in [3u64, 5, 17, 257] {
        // 65535 = 3 * 5 * 17 * 257; g^(65535/p) != 1 for every prime factor p.
        assert_ne!(
            gf16::GENERATOR.pow(65_535 / factor),
            gf16::Elem::ONE,
            "generator order divides 65535/{factor}"
        );
    }
}

#[test]
fn inversion_is_total() {
    // SRS's Cauchy inverse and batch-inversion paths rely on inv(0) == 0 rather than
    // branching on zero; a panicking or UB-on-zero upstream would be a silent hazard.
    assert_eq!(gf8::Elem::ZERO.inv(), gf8::Elem::ZERO);
    assert_eq!(gf16::Elem::ZERO.inv(), gf16::Elem::ZERO);
    assert_eq!(gf8::Elem(0x57).div(gf8::Elem::ZERO), gf8::Elem::ZERO);
    assert_eq!(gf16::Elem(0x1234).div(gf16::Elem::ZERO), gf16::Elem::ZERO);
}
