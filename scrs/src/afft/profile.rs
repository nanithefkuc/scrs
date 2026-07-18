use super::TransformPlan;

#[derive(Clone, Debug)]
pub(crate) struct Profile {
    pub k: usize,
    pub m: usize,
    pub n: usize,
    pub symbol_len: usize,
    pub padded_k: usize,
    pub transform_size: usize,
    pub interpolation_plan: TransformPlan,
    pub transform_plan: TransformPlan,
}

impl Profile {
    pub fn new(k: usize, m: usize, symbol_len: usize) -> Option<Self> {
        let n = k.checked_add(m)?;
        if k == 0 || m == 0 || symbol_len == 0 || symbol_len % 2 != 0 {
            return None;
        }
        let padded_k = k.checked_next_power_of_two()?;
        let used_points = padded_k.checked_add(m)?;
        if used_points > super::MAX_TRANSFORM_SIZE {
            return None;
        }
        let transform_size = used_points.checked_next_power_of_two()?;
        transform_size.checked_mul(symbol_len)?;
        n.checked_mul(symbol_len)?;
        Some(Self {
            k,
            m,
            n,
            symbol_len,
            padded_k,
            transform_size,
            interpolation_plan: TransformPlan::new(padded_k)?,
            transform_plan: TransformPlan::new(transform_size)?,
        })
    }

    #[inline]
    pub fn evaluation_index(&self, wire_index: usize) -> usize {
        debug_assert!(wire_index < self.n);
        if wire_index < self.k {
            wire_index
        } else {
            self.padded_k + wire_index - self.k
        }
    }
}

pub(crate) fn zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).ok()?;
    bytes.resize(len, 0);
    Some(bytes)
}
