//! Old-kernel throughput reference.
use std::time::Instant;
use fff::gf8::Elem as GfElem;
fn bench(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters/10 { f(); }
    let s = Instant::now();
    for _ in 0..iters { f(); }
    s.elapsed().as_secs_f64()/iters as f64*1e9
}
fn main() {
    println!("backend: {:?}", fff::kernel::backend());
    for &slen in &[64usize, 256, 1000, 1400, 4096] {
        let src = vec![0xa5u8; slen];
        let mut dst = vec![0u8; slen];
        let c = GfElem(0x53);
        let iters = ((1<<24)/slen).max(2000);
        let axpy = bench(iters, || fff::ops::mul_add::<fff::Gf8>(std::hint::black_box(&mut dst), c, std::hint::black_box(&src)));
        let coeffs: Vec<GfElem> = (0..16).map(|i| GfElem((i as u8)|1)).collect();
        let mut rows = vec![0u8; 16*slen];
        let rowsb = bench(iters/4+1, || fff::ops::mul_add_scatter::<fff::Gf8>(std::hint::black_box(&mut rows), slen, &coeffs, std::hint::black_box(&src)));
        println!("slen={slen:<6} axpy={axpy:>9.1}ns  scatter16={rowsb:>10.1}ns");
    }
}
