//! SIMD payload kernels.
//!
//! Safe crate-private façade functions select an architecture backend once.
//! Architecture-local implementations contain the narrowly scoped unsafe SIMD
//! operations and document their pointer and CPU-feature invariants.

mod aarch64;
mod dispatch;
mod rows;
mod scalar;
mod scale_table;
mod x86;

#[allow(unused_imports)]
pub(crate) use dispatch::{
    KernelPlan, gfni_available, kernel_plan, xor_bytes, xor_scaled_bytes, xor_scaled_bytes_coeff,
    xor_scaled_bytes_coeff_with_plan, xor_scaled_bytes_gfni, xor_scaled_bytes_many,
};
#[allow(unused_imports)]
pub(crate) use rows::{
    IndexedDestinationRows, xor_scaled_bytes_many_indexed, xor_scaled_bytes_rows,
};
#[allow(unused_imports)]
pub(crate) use scale_table::{ScaleTable, scale_table};
