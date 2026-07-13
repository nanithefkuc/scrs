//! Portable byte-kernel tails.

pub(super) fn xor_scaled_bytes_nibble_tail(
    dst: &mut [u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    src: &[u8],
) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= lo[(s & 0x0f) as usize] ^ hi[(s >> 4) as usize];
    }
}

pub(super) fn xor_bytes_scalar(dst: &mut [u8], src: &[u8]) {
    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mut d_arr = [0u8; 8];
        let mut s_arr = [0u8; 8];
        d_arr.copy_from_slice(d);
        s_arr.copy_from_slice(s);
        let mixed = u64::from_ne_bytes(d_arr) ^ u64::from_ne_bytes(s_arr);
        d.copy_from_slice(&mixed.to_ne_bytes());
    }

    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d ^= s;
    }
}
