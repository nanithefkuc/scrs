//! Systematic additive-FFT encoder.

use super::profile::{Profile, zeroed_bytes};

/// Error returned when additive-FFT encoding receives the wrong data length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeError {
    /// Required length, `k * symbol_len`.
    pub expected: usize,
    /// Supplied length.
    pub got: usize,
}

/// Reusable scratch for allocation-free additive-FFT encoding.
///
/// Construct one with [`SystematicEncoder::encode_scratch`] and pass it to
/// [`SystematicEncoder::encode_into_with`] to run steady-state encoding without
/// heap allocation, as required by Aeron-style ring-buffer producers.
#[derive(Clone, Debug, Default)]
pub struct EncodeScratch {
    workspace: Vec<u8>,
}

impl EncodeScratch {
    /// Create empty scratch that grows on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Target working-set for one symbol-column strip. Chosen so a strip's IFFT/FFT
/// stays resident in a P-core L2 (~1.25 MiB), avoiding the cache cliff that a
/// full-width transform hits once `transform_size * symbol_len` spills L2.
const STRIP_TARGET_BYTES: usize = 768 * 1024;

/// Widest even column strip whose transform working set fits [`STRIP_TARGET_BYTES`],
/// where `rows_per_strip` is the number of transform rows a strip holds.
/// Always in `2..=symbol_len` (both bounds even), so strips tile an even symbol.
fn strip_width(rows_per_strip: usize, symbol_len: usize) -> usize {
    let cap = ((STRIP_TARGET_BYTES / rows_per_strip) & !1).max(2);
    cap.min(symbol_len)
}

/// Block-systematic Reed-Solomon encoder using the additive FFT.
///
/// Non-power-of-two `k` uses a truncated inverse transform over the first
/// `k.next_power_of_two()` points. Repair symbols occupy evaluation points
/// `k..k + m`, and construction therefore requires `k + m <= 65536`.
#[derive(Clone, Debug)]
pub struct SystematicEncoder {
    profile: Profile,
}

impl SystematicEncoder {
    /// Construct an encoder.
    ///
    /// Returns `None` for zero dimensions, zero or odd symbol lengths, size
    /// overflow, or when `k + m > 65536`.
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        Some(Self {
            profile: Profile::new(k, m, symbol_len)?,
        })
    }

    /// Number of systematic symbols.
    pub const fn k(&self) -> usize {
        self.profile.k
    }

    /// Number of repair symbols.
    pub const fn m(&self) -> usize {
        self.profile.m
    }

    /// Number of transmitted symbols, `k + m`.
    pub const fn n(&self) -> usize {
        self.profile.n
    }

    /// Per-symbol byte length.
    pub const fn symbol_len(&self) -> usize {
        self.profile.symbol_len
    }

    /// Power-of-two plan size used for truncated systematic interpolation.
    pub const fn padded_k(&self) -> usize {
        self.profile.padded_k
    }

    /// Power-of-two full evaluation transform size.
    pub const fn transform_size(&self) -> usize {
        self.profile.transform_size
    }

    /// Encode a flat `k * symbol_len` systematic block and return `m` repairs.
    ///
    /// The input bytes are the systematic symbols on the wire and are never
    /// modified or reserialized.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<Vec<u8>>, EncodeError> {
        let repair_len = self.profile.m * self.profile.symbol_len;
        let mut repairs =
            zeroed_bytes(repair_len).expect("profile validated repair allocation size");
        self.encode_into(data, &mut repairs)?;
        Ok(repairs
            .chunks_exact(self.profile.symbol_len)
            .map(<[u8]>::to_vec)
            .collect())
    }

    /// Whether the fused encode path applies, plus the transform rows one strip
    /// holds. Fusion applies when `k` is a power of two and `m <= k`
    /// (`padded_k == k`, `transform_size == 2k`): the repair coset is exactly
    /// the root's high child, so the IFFT output is evaluated in place without a
    /// copy or a full-domain workspace. `k >= 2` guarantees `log_size >= 2`.
    fn strip_plan(&self) -> (bool, usize) {
        let p = &self.profile;
        let fused = p.padded_k == p.k && p.transform_size == 2 * p.k && p.k >= 2;
        let rows_per_strip = if fused { p.padded_k } else { p.transform_size };
        (fused, rows_per_strip)
    }

    /// Allocate scratch sized for one symbol-column strip of this encoder.
    ///
    /// The returned [`EncodeScratch`] can be reused across any number of
    /// [`encode_into_with`](Self::encode_into_with) calls with no further
    /// allocation.
    pub fn encode_scratch(&self) -> EncodeScratch {
        let (_, rows_per_strip) = self.strip_plan();
        let width = strip_width(rows_per_strip, self.profile.symbol_len);
        let mut workspace = Vec::new();
        workspace.reserve_exact(rows_per_strip * width);
        EncodeScratch { workspace }
    }

    /// Encode repairs into a caller-provided flat `m * symbol_len` buffer.
    ///
    /// Allocates a transform workspace per call; use
    /// [`encode_into_with`](Self::encode_into_with) with reusable scratch for an
    /// allocation-free steady state.
    pub fn encode_into(&self, data: &[u8], repairs: &mut [u8]) -> Result<(), EncodeError> {
        let mut scratch = self.encode_scratch();
        self.encode_into_with(data, repairs, &mut scratch)
    }

    /// Encode repairs into a caller-provided buffer using reusable scratch.
    ///
    /// After the first sizing call, steady-state use performs no heap
    /// allocation. The transform workspace is zeroed and reused each call.
    pub fn encode_into_with(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut EncodeScratch,
    ) -> Result<(), EncodeError> {
        let expected_data = self.profile.k * self.profile.symbol_len;
        if data.len() != expected_data {
            return Err(EncodeError {
                expected: expected_data,
                got: data.len(),
            });
        }
        let expected_repairs = self.profile.m * self.profile.symbol_len;
        if repairs.len() != expected_repairs {
            return Err(EncodeError {
                expected: expected_repairs,
                got: repairs.len(),
            });
        }

        let (fused, rows_per_strip) = self.strip_plan();
        let width = strip_width(rows_per_strip, self.profile.symbol_len);
        self.encode_blocked(data, repairs, scratch, width, fused);
        Ok(())
    }

    /// Strip-blocked encode core. `width` is the symbol-column strip width in
    /// bytes (even, in `2..=symbol_len`); production callers pass
    /// [`strip_width`]. Lengths must already be validated. Each strip is
    /// interpolated and evaluated entirely within the reusable scratch buffer.
    ///
    /// When `fused`, the strip holds only the `padded_k` systematic rows and the
    /// repair coset is evaluated in place (no copy, no padded high half);
    /// otherwise a full `transform_size`-row strip is interpolated then forward-
    /// transformed with the truncated evaluation.
    fn encode_blocked(
        &self,
        data: &[u8],
        repairs: &mut [u8],
        scratch: &mut EncodeScratch,
        width: usize,
        fused: bool,
    ) {
        let l = self.profile.symbol_len;
        let k = self.profile.k;
        let n = self.profile.n;
        let m = self.profile.m;
        let pk = self.profile.padded_k;
        let ts = self.profile.transform_size;
        let rows_per_strip = if fused { pk } else { ts };

        let strip = &mut scratch.workspace;
        let strip_capacity = rows_per_strip * width;
        if strip.len() != strip_capacity {
            strip.clear();
            strip.resize(strip_capacity, 0);
        }

        let mut col = 0;
        while col < l {
            let w = width.min(l - col);
            let strip = &mut strip[..rows_per_strip * w];
            // Gather this column strip of the data into the systematic rows.
            for r in 0..k {
                let src = r * l + col;
                strip[r * w..r * w + w].copy_from_slice(&data[src..src + w]);
            }

            if fused {
                // padded_k == k: the inverse fills exactly the systematic rows,
                // and the repair coset (transform points k..2k) is evaluated in
                // place. Repairs land in the first `m` rows.
                self.profile
                    .interpolation_plan
                    .inverse_truncated_bytes(strip, w, k);
                self.profile
                    .transform_plan
                    .forward_bytes_high_coset_range(strip, w, 0..m);
                for r in 0..m {
                    let dst = r * l + col;
                    repairs[dst..dst + w].copy_from_slice(&strip[r * w..r * w + w]);
                }
            } else {
                // Coefficient padding and repair rows are read as zero by the
                // truncated transforms; zero them (cheap, in-cache) per strip.
                strip[k * w..].fill(0);
                self.profile
                    .interpolation_plan
                    .inverse_truncated_bytes(&mut strip[..pk * w], w, k);
                self.profile
                    .transform_plan
                    .forward_bytes_trunc_range(strip, w, k, k..n);
                for r in k..n {
                    let dst = (r - k) * l + col;
                    repairs[dst..dst + w].copy_from_slice(&strip[r * w..r * w + w]);
                }
            }
            col += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf65536::GfElem;

    #[test]
    fn validates_transform_capacity() {
        assert!(SystematicEncoder::new(257, 128, 8).is_some());
        assert!(SystematicEncoder::new(32_769, 1, 2).is_some());
        assert!(SystematicEncoder::new(65_535, 2, 2).is_none());
        assert!(SystematicEncoder::new(1, 1, 3).is_none());
        let encoder = SystematicEncoder::new(5, 3, 2).unwrap();
        assert_eq!(encoder.padded_k(), 8);
        assert_eq!(encoder.transform_size(), 8);
    }

    #[test]
    fn strip_blocking_matches_single_strip() {
        // Force many narrow strips (width 2) and compare against a single strip
        // (width = symbol_len) and the public tuned path. Exercises the
        // gather/scatter and last-strip remainder logic the small-symbol tests
        // (single strip) never reach.
        for (k, m, l) in [(5, 3, 64), (100, 20, 64), (17, 7, 130), (512, 128, 40)] {
            let enc = SystematicEncoder::new(k, m, l).unwrap();
            let data: Vec<u8> = (0..k * l).map(|i| (i * 137 + 11) as u8).collect();

            let mut single = vec![0u8; m * l];
            let mut s1 = EncodeScratch::new();
            enc.encode_blocked(&data, &mut single, &mut s1, l, false);

            let mut multi = vec![0u8; m * l];
            let mut s2 = EncodeScratch::new();
            enc.encode_blocked(&data, &mut multi, &mut s2, 2, false);
            assert_eq!(single, multi, "multi-strip mismatch k={k} m={m} l={l}");

            let mut tuned = vec![0u8; m * l];
            let mut s3 = enc.encode_scratch();
            enc.encode_into_with(&data, &mut tuned, &mut s3).unwrap();
            assert_eq!(single, tuned, "tuned-path mismatch k={k} m={m} l={l}");
        }
    }

    #[test]
    fn fused_matches_unfused() {
        // The fused (in-place coset) path must be byte-identical to the general
        // interpolate-then-truncated-forward path, for both single and multi
        // strip widths. Configs are power-of-two k with m <= k so fusion applies.
        for (k, m, l) in [(4, 2, 64), (16, 4, 40), (256, 64, 48), (512, 128, 40)] {
            let enc = SystematicEncoder::new(k, m, l).unwrap();
            assert!(enc.strip_plan().0, "expected fused path for k={k} m={m}");
            let data: Vec<u8> = (0..k * l).map(|i| (i * 149 + 3) as u8).collect();

            let mut unfused = vec![0u8; m * l];
            let mut su = EncodeScratch::new();
            enc.encode_blocked(&data, &mut unfused, &mut su, l, false);

            let mut fused = vec![0u8; m * l];
            let mut sf = EncodeScratch::new();
            enc.encode_blocked(&data, &mut fused, &mut sf, l, true);
            assert_eq!(unfused, fused, "fused mismatch k={k} m={m} l={l}");

            let mut fused_multi = vec![0u8; m * l];
            let mut sm = EncodeScratch::new();
            enc.encode_blocked(&data, &mut fused_multi, &mut sm, 2, true);
            assert_eq!(unfused, fused_multi, "fused multi-strip mismatch k={k} m={m} l={l}");
        }
    }

    #[test]
    fn encoding_is_systematic_and_repairs_match_scalar_transform() {
        let k = 5;
        let m = 3;
        let symbol_len = 6;
        let data: Vec<_> = (0..k * symbol_len)
            .map(|index| ((index * 29) ^ 0xa5) as u8)
            .collect();
        let encoder = SystematicEncoder::new(k, m, symbol_len).unwrap();
        let repairs = encoder.encode(&data).unwrap();

        for element_offset in (0..symbol_len).step_by(2) {
            for (repair, repair_bytes) in repairs.iter().enumerate() {
                let evaluation = GfElem((k + repair) as u16);
                let mut expected = GfElem::ZERO;
                for data_index in 0..k {
                    let mut numerator = GfElem::ONE;
                    let mut denominator = GfElem::ONE;
                    for other in 0..k {
                        if other == data_index {
                            continue;
                        }
                        numerator = numerator.mul(evaluation.add(GfElem(other as u16)));
                        denominator =
                            denominator.mul(GfElem(data_index as u16).add(GfElem(other as u16)));
                    }
                    let start = data_index * symbol_len + element_offset;
                    let value = GfElem::from_bytes([data[start], data[start + 1]]);
                    expected = expected.add(value.mul(numerator).mul(denominator.inv()));
                }
                assert_eq!(
                    &repair_bytes[element_offset..element_offset + 2],
                    &expected.to_bytes()
                );
            }
        }
    }

    #[test]
    fn encode_into_with_matches_encode_and_reuses_scratch() {
        let (k, m, symbol_len) = (100usize, 20usize, 1024usize);
        let encoder = SystematicEncoder::new(k, m, symbol_len).unwrap();
        let data: Vec<u8> = (0..k * symbol_len)
            .map(|index| (index.wrapping_mul(31) ^ 0xa5) as u8)
            .collect();
        let reference: Vec<u8> = encoder
            .encode(&data)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        let mut scratch = encoder.encode_scratch();
        let mut repairs = vec![0u8; m * symbol_len];
        encoder
            .encode_into_with(&data, &mut repairs, &mut scratch)
            .unwrap();
        assert_eq!(repairs, reference);

        let ptr = scratch.workspace.as_ptr();
        let cap = scratch.workspace.capacity();
        for _ in 0..8 {
            encoder
                .encode_into_with(&data, &mut repairs, &mut scratch)
                .unwrap();
            assert_eq!(repairs, reference);
        }
        assert_eq!(scratch.workspace.as_ptr(), ptr, "encode workspace reallocated");
        assert_eq!(scratch.workspace.capacity(), cap);
    }
}
